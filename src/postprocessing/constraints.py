"""
Physical Realizability & Monotonicity Constraints Engine for RAIN-REPAIR X.
Enforces CONTRACT-ML-002 Invariants:
1. Non-Negativity: Precipitation >= 0.0 mm/24h.
2. Quantile Monotonicity (Non-Crossing via Rearrangement Operator):
   0.0 <= Q_10 <= Q_50 <= Q_75 <= Q_90 <= Q_95
3. Probability Bounds & Monotonicity:
   0.0 <= P(>= 204.5mm) <= P(>= 115.6mm) <= P(>= 64.5mm) <= 1.0
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.exceptions import PhysicalConstraintViolationError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.constraints")

CANONICAL_QUANTILES: List[float] = [0.10, 0.50, 0.75, 0.90, 0.95]


def enforce_quantile_monotonicity(
    quantiles_dict: Dict[float, np.ndarray],
    enforce_non_negativity: bool = True,
    strict_check: bool = False,
) -> Dict[float, np.ndarray]:
    """
    Applies the mathematical Rearrangement Operator (Chernozhukov et al.)
    to eliminate quantile crossing while strictly preserving calibration.

    Given raw predictions {q_alpha}, sorts along the quantile dimension:
        Q_sorted = sort(Q_raw)
    Ensures:
        0.0 <= Q_10 <= Q_50 <= Q_75 <= Q_90 <= Q_95
    """
    sorted_alphas = sorted(quantiles_dict.keys())
    if not sorted_alphas:
        return {}

    n_samples = len(quantiles_dict[sorted_alphas[0]])
    stack = np.column_stack([quantiles_dict[a] for a in sorted_alphas])

    if enforce_non_negativity:
        stack = np.maximum(stack, 0.0)

    # Sort each row across the quantile axis to guarantee monotonicity
    rearranged = np.sort(stack, axis=1)

    if strict_check:
        # Check if crossing occurred before sorting
        crossings = np.sum(np.diff(stack, axis=1) < -1e-4)
        if crossings > 0:
            logger.info("Rearrangement Operator resolved %d quantile crossing instances.", int(crossings))

    return {
        sorted_alphas[idx]: rearranged[:, idx].astype(np.float32)
        for idx, alpha in enumerate(sorted_alphas)
    }


def enforce_probability_monotonicity(
    prob_heavy: np.ndarray,
    prob_very_heavy: np.ndarray,
    prob_extreme: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Enforces cumulative threshold monotonicity:
    Since Heavy (>=64.5mm) encompasses Very Heavy (>=115.6mm) which encompasses Extreme (>=204.5mm):
        0.0 <= P(>=204.5) <= P(>=115.6) <= P(>=64.5) <= 1.0

    Applies cumulative minimum adjustment.
    """
    p_heavy = np.clip(np.asarray(prob_heavy, dtype=np.float32), 0.0, 1.0)
    p_vheavy = np.clip(np.asarray(prob_very_heavy, dtype=np.float32), 0.0, 1.0)
    p_extreme = np.clip(np.asarray(prob_extreme, dtype=np.float32), 0.0, 1.0)

    # Monotonic upper-bound projection:
    # Very heavy cannot exceed heavy; extreme cannot exceed very heavy
    p_vheavy = np.minimum(p_vheavy, p_heavy)
    p_extreme = np.minimum(p_extreme, p_vheavy)

    return p_heavy, p_vheavy, p_extreme


def validate_physical_realizability(
    df: pd.DataFrame,
    strict: bool = True,
) -> bool:
    """
    Validates CONTRACT-ML-002 physical realizability invariants on prediction outputs.
    Raises PhysicalConstraintViolationError if any rule is violated.
    """
    # 1. Non-negativity check
    precip_cols = [c for c in df.columns if "precip" in c or c.startswith("corrected_")]
    for col in precip_cols:
        if np.any(df[col] < -1e-4):
            violating_min = float(df[col].min())
            msg = f"Non-negativity violation in column {col}: minimum value is {violating_min:.4f} mm."
            logger.error(msg)
            if strict:
                raise PhysicalConstraintViolationError(msg)
            return False

    # 2. Quantile monotonicity check
    q_cols = ["corrected_p10", "corrected_p50", "corrected_p75", "corrected_p90", "corrected_p95"]
    present_q = [c for c in q_cols if c in df.columns]
    if len(present_q) >= 2:
        for i in range(len(present_q) - 1):
            diff = df[present_q[i + 1]] - df[present_q[i]]
            if np.any(diff < -1e-4):
                violating_diff = float(diff.min())
                msg = f"Quantile crossing between {present_q[i]} and {present_q[i+1]}: max deficit {violating_diff:.4f} mm."
                logger.error(msg)
                if strict:
                    raise PhysicalConstraintViolationError(msg)
                return False

    # 3. Probability bounds check
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    for col in prob_cols:
        vals = df[col].to_numpy()
        if np.any((vals < 0.0) | (vals > 1.0001)):
            msg = f"Probability bounds violation in {col}: range [{vals.min():.4f}, {vals.max():.4f}] outside [0, 1]."
            logger.error(msg)
            if strict:
                raise PhysicalConstraintViolationError(msg)
            return False

    return True
