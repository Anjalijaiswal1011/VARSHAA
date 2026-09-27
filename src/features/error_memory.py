"""
Multi-Scale Recent Error Memory and NWP Error DNA Module for RAIN-REPAIR X.
Computes rolling 1/3/7/14-day forecast error buffers with strict causal anti-leakage guards.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.exceptions import TemporalAlignmentError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.features.error_memory")


class ErrorMemoryBuffer:
    """
    In-memory or persistent store maintaining recent NWP forecast errors:
        Error = NWP_forecast - Ground_truth_observation

    Enforces strict temporal causality: observations from t >= T_0 can NEVER
    be queried to compute error features for forecast cycle T_0.
    """

    def __init__(self):
        # Maps ISO date string 'YYYY-MM-DD' to 2D numpy array of errors (NWP - Obs)
        self._history: Dict[str, np.ndarray] = {}

    def record_historical_error(
        self,
        date_str: str,
        nwp_grid: np.ndarray,
        obs_grid: np.ndarray,
    ) -> None:
        """
        Record a verified historical forecast error.
        Error = NWP - Obs. Positive = model over-forecast; Negative = model under-forecast.
        """
        nwp = np.asarray(nwp_grid, dtype=np.float32)
        obs = np.asarray(obs_grid, dtype=np.float32)

        if nwp.shape != obs.shape:
            raise ValueError(f"Shape mismatch between NWP {nwp.shape} and Obs {obs.shape}")

        error = nwp - obs
        self._history[date_str] = error

    def get_multi_timescale_error_memory(
        self,
        current_cycle_date: str,
        grid_shape: Tuple[int, int],
    ) -> Dict[str, np.ndarray]:
        """
        Compute multi-scale trailing error features:
        - error_memory_lag1: yesterday's error (t - 1)
        - error_memory_lag3_mean: trailing 3-day average error (t-1 to t-3)
        - error_memory_lag7_mean: trailing 7-day average error (t-1 to t-7)
        - error_memory_lag14_mean: trailing 14-day average error (t-1 to t-14)

        Strictly enforces that dates >= current_cycle_date are ignored.
        """
        curr_dt = pd.to_datetime(current_cycle_date)

        # Retrieve available dates strictly prior to current cycle date
        valid_past_dates = sorted(
            [d for d in self._history.keys() if pd.to_datetime(d) < curr_dt],
            reverse=True,
        )

        # Anti-leakage sanity check
        for d in valid_past_dates:
            if pd.to_datetime(d) >= curr_dt:
                raise TemporalAlignmentError(
                    f"DATA LEAKAGE DETECTED: Observation date {d} is >= current forecast cycle {current_cycle_date}!"
                )

        # Default neutral zeros for cold-start initialization
        zeros_field = np.zeros(grid_shape, dtype=np.float32)

        # Lag 1
        lag1_field = self._history.get(valid_past_dates[0], zeros_field) if len(valid_past_dates) >= 1 else zeros_field

        # Lag 3 mean
        if len(valid_past_dates) >= 3:
            lag3_stack = np.stack([self._history[d] for d in valid_past_dates[:3]], axis=0)
            lag3_field = np.mean(lag3_stack, axis=0).astype(np.float32)
        elif len(valid_past_dates) > 0:
            lag3_stack = np.stack([self._history[d] for d in valid_past_dates], axis=0)
            lag3_field = np.mean(lag3_stack, axis=0).astype(np.float32)
        else:
            lag3_field = zeros_field

        # Lag 7 mean
        if len(valid_past_dates) >= 7:
            lag7_stack = np.stack([self._history[d] for d in valid_past_dates[:7]], axis=0)
            lag7_field = np.mean(lag7_stack, axis=0).astype(np.float32)
        else:
            lag7_field = lag3_field

        # Lag 14 mean
        if len(valid_past_dates) >= 14:
            lag14_stack = np.stack([self._history[d] for d in valid_past_dates[:14]], axis=0)
            lag14_field = np.mean(lag14_stack, axis=0).astype(np.float32)
        else:
            lag14_field = lag7_field

        return {
            "error_memory_lag1": lag1_field,
            "error_memory_lag3_mean": lag3_field,
            "error_memory_lag7_mean": lag7_field,
            "error_memory_lag14_mean": lag14_field,
        }

    def compute_nwp_error_dna(
        self,
        lookback_days: int = 14,
    ) -> Dict[str, np.ndarray]:
        """
        Compute NWP Error DNA summary metrics across the lookback window:
        - persistent_bias_magnitude: Mean absolute bias across history
        - bias_stability: Standard deviation of error (high = unpredictable convective bursts)
        - false_alarm_tendency: Frequency of positive over-prediction bias
        """
        all_dates = sorted(list(self._history.keys()))[-lookback_days:]
        if not all_dates:
            return {}

        stack = np.stack([self._history[d] for d in all_dates], axis=0)
        mean_bias = np.mean(stack, axis=0)
        bias_std = np.std(stack, axis=0)
        fa_tendency = np.mean(stack > 2.5, axis=0).astype(np.float32)

        return {
            "dna_mean_bias": mean_bias.astype(np.float32),
            "dna_bias_stability": bias_std.astype(np.float32),
            "dna_false_alarm_rate": fa_tendency,
        }
