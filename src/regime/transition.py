"""
Regime Transition Engine for RAIN-REPAIR X.
Detects meteorologically meaningful regime state transitions, probability vector shifts,
and maintains chronological probability history per grid cell without temporal leakage.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.regime.schemas import (
    REGIME_CLASSES,
    RegimeProbabilityVector,
    RegimeTransitionOutput,
    TransitionStatus,
    WeatherRegime,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.transition")

# Documented Transition Engine Thresholds
# 1. Minimum dominant probability required to be considered confident
CONFIDENCE_UNCERTAIN_THRESHOLD: float = 0.38
# 2. Maximum Shannon entropy fraction (entropy / log(6)) beyond which state is uncertain
ENTROPY_UNCERTAIN_THRESHOLD: float = 0.82
# 3. Minimum Total Variation distance between consecutive probability vectors to signal transition
TV_DISTANCE_TRANSITION_THRESHOLD: float = 0.22
# 4. Minimum probability delta on the rising regime to confirm an active transition
RISING_PROB_DELTA_THRESHOLD: float = 0.15


def compute_total_variation_distance(p: np.ndarray, q: np.ndarray) -> float:
    """
    Total Variation Distance between two discrete probability distributions:
    TV(p, q) = 0.5 * sum_k |p_k - q_k|
    Bounded strictly in [0.0, 1.0].
    """
    return float(0.5 * np.sum(np.abs(p - q)))


def compute_normalized_entropy(probs: np.ndarray) -> float:
    """
    Normalized Shannon entropy: H(p) / log(K)
    0.0 represents complete certainty (degenerate distribution).
    1.0 represents uniform maximum uncertainty across all 6 regimes.
    """
    k = len(probs)
    if k <= 1:
        return 0.0
    clipped = np.clip(probs, 1e-9, 1.0)
    entropy = -float(np.sum(clipped * np.log(clipped)))
    max_entropy = float(np.log(k))
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


class RegimeHistoryBuffer:
    """
    Leakage-safe rolling chronological history buffer of regime probability vectors.
    Stores past probability vectors keyed by grid_id (or coordinate tuple).
    """

    def __init__(self, max_history_len: int = 14):
        self.max_history_len = max_history_len
        # grid_id -> list of (timestamp, RegimeProbabilityVector)
        self._history: Dict[str, List[Tuple[str, RegimeProbabilityVector]]] = defaultdict(list)
        # grid_id -> time_steps_since_last_dominant_change
        self._steps_in_current_regime: Dict[str, int] = defaultdict(int)

    def get_latest(self, grid_id: str) -> Optional[Tuple[str, RegimeProbabilityVector]]:
        """Return the most recent past prediction for this grid point."""
        records = self._history.get(grid_id, [])
        return records[-1] if records else None

    def get_history(self, grid_id: str) -> List[Tuple[str, RegimeProbabilityVector]]:
        return list(self._history.get(grid_id, []))

    def get_steps_in_current_regime(self, grid_id: str) -> int:
        return self._steps_in_current_regime.get(grid_id, 0)

    def append(self, grid_id: str, timestamp: str, prob_vector: RegimeProbabilityVector) -> None:
        """Append a newly predicted probability vector to history."""
        curr_dominant, _ = prob_vector.dominant_regime()
        prev = self.get_latest(grid_id)

        if prev is not None:
            prev_dominant, _ = prev[1].dominant_regime()
            if curr_dominant == prev_dominant:
                self._steps_in_current_regime[grid_id] += 1
            else:
                self._steps_in_current_regime[grid_id] = 1
        else:
            self._steps_in_current_regime[grid_id] = 1

        records = self._history[grid_id]
        records.append((timestamp, prob_vector))
        if len(records) > self.max_history_len:
            self._history[grid_id] = records[-self.max_history_len:]

    def clear(self) -> None:
        self._history.clear()
        self._steps_in_current_regime.clear()


class RegimeTransitionEngine:
    """
    Meteorological Regime Transition Engine.
    Analyzes current probability vectors against chronological history
    to evaluate stability, regime transition dynamics, and uncertainty.
    """

    def __init__(
        self,
        history_buffer: Optional[RegimeHistoryBuffer] = None,
        confidence_threshold: float = CONFIDENCE_UNCERTAIN_THRESHOLD,
        tv_distance_threshold: float = TV_DISTANCE_TRANSITION_THRESHOLD,
    ):
        self.history = history_buffer or RegimeHistoryBuffer()
        self.conf_threshold = confidence_threshold
        self.tv_threshold = tv_distance_threshold

    def evaluate_transition(
        self,
        grid_id: str,
        timestamp: str,
        current_prob: RegimeProbabilityVector,
        update_history: bool = True,
    ) -> RegimeTransitionOutput:
        """
        Evaluate transition state for a single grid point at a given timestamp.
        """
        curr_arr = current_prob.to_array()
        curr_dominant, curr_conf = current_prob.dominant_regime()
        entropy_ratio = compute_normalized_entropy(curr_arr)

        prev_record = self.history.get_latest(grid_id)
        
        if prev_record is None:
            # First observation in sequence: No previous state available
            if curr_conf < self.conf_threshold or entropy_ratio > ENTROPY_UNCERTAIN_THRESHOLD:
                status = TransitionStatus.UNCERTAIN
            else:
                status = TransitionStatus.STABLE

            output = RegimeTransitionOutput(
                timestamp=timestamp,
                grid_id=grid_id,
                current_regime=curr_dominant if status != TransitionStatus.UNCERTAIN else WeatherRegime.REGIME_UNCERTAIN.value,
                current_regime_probabilities=current_prob.to_dict(),
                previous_regime=None,
                transition_status=status,
                transition_from=None,
                transition_to=None,
                transition_strength=0.0,
                transition_confidence=round(curr_conf, 4),
            )
            if update_history:
                self.history.append(grid_id, timestamp, current_prob)
            return output

        # Previous state is available
        _, prev_prob = prev_record
        prev_arr = prev_prob.to_array()
        prev_dominant, prev_conf = prev_prob.dominant_regime()

        tv_dist = compute_total_variation_distance(curr_arr, prev_arr)
        delta_arr = curr_arr - prev_arr

        # Determine Transition State:
        # Case A: UNCERTAIN
        # If confidence is low or entropy is excessively flat
        if curr_conf < self.conf_threshold or entropy_ratio > ENTROPY_UNCERTAIN_THRESHOLD:
            status = TransitionStatus.UNCERTAIN
            trans_from = prev_dominant
            trans_to = None
            trans_strength = round(tv_dist, 4)
            trans_confidence = round(curr_conf, 4)
            assigned_regime = WeatherRegime.REGIME_UNCERTAIN.value

        # Case B: TRANSITION
        # True transition requires either:
        # 1. Change in dominant regime with sufficient confidence
        # 2. Substantial probability redistribution (TV distance >= threshold)
        elif (curr_dominant != prev_dominant and curr_conf >= self.conf_threshold) or (
            tv_dist >= self.tv_threshold and np.max(delta_arr) >= RISING_PROB_DELTA_THRESHOLD
        ):
            status = TransitionStatus.TRANSITION
            trans_from = prev_dominant
            trans_to = curr_dominant
            trans_strength = round(tv_dist, 4)
            trans_confidence = round(float(np.mean([curr_conf, 1.0 - entropy_ratio])), 4)
            assigned_regime = curr_dominant

        # Case C: STABLE
        # Dominant regime maintained with low drift
        else:
            status = TransitionStatus.STABLE
            trans_from = None
            trans_to = None
            trans_strength = round(tv_dist, 4)
            trans_confidence = round(curr_conf, 4)
            assigned_regime = curr_dominant

        output = RegimeTransitionOutput(
            timestamp=timestamp,
            grid_id=grid_id,
            current_regime=assigned_regime,
            current_regime_probabilities=current_prob.to_dict(),
            previous_regime=prev_dominant,
            transition_status=status,
            transition_from=trans_from,
            transition_to=trans_to,
            transition_strength=trans_strength,
            transition_confidence=trans_confidence,
        )

        if update_history:
            self.history.append(grid_id, timestamp, current_prob)

        return output

    def evaluate_batch(
        self,
        df: pd.DataFrame,
        prob_vectors: List[RegimeProbabilityVector],
        grid_id_col: str = "grid_id",
        timestamp_col: str = "forecast_time",
    ) -> List[RegimeTransitionOutput]:
        """
        Evaluate transitions for a chronological batch of samples.
        Maintains order to avoid temporal leakage.
        """
        results: List[RegimeTransitionOutput] = []
        n = len(df)

        grid_ids = df[grid_id_col].astype(str).tolist() if grid_id_col in df.columns else [f"grid_{i}" for i in range(n)]
        timestamps = df[timestamp_col].astype(str).tolist() if timestamp_col in df.columns else [f"T{i}" for i in range(n)]

        for i in range(n):
            out = self.evaluate_transition(
                grid_id=grid_ids[i],
                timestamp=timestamps[i],
                current_prob=prob_vectors[i],
                update_history=True,
            )
            results.append(out)

        return results
