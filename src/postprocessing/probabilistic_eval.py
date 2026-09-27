"""
Probabilistic & Extreme Verification Metrics for RAIN-REPAIR X (PART 7).
Evaluates:
1. Multi-Quantile Pinball Loss (P10, P50, P75, P90, P95).
2. Continuous Ranked Probability Score (CRPS) approximation via quantile integration.
3. Brier Score (BS) & Brier Skill Score (BSS) for IMD heavy rainfall alert thresholds.
4. Quantile Reliability & Coverage Verification (Nominal vs Empirical Exceedance Rates).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.quantiles import compute_pinball_loss
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.probabilistic_eval")


def compute_crps_from_quantiles(
    y_true: Union[np.ndarray, pd.Series],
    quantiles_dict: Dict[float, np.ndarray],
) -> float:
    """
    Approximates Continuous Ranked Probability Score (CRPS) using the quantile integration identity:
        CRPS(F, y) = 2 * int_0^1 L_alpha(y, q_alpha) d(alpha)
    where L_alpha is the pinball loss at level alpha.
    """
    y = np.asarray(y_true, dtype=np.float32)
    sorted_alphas = sorted(quantiles_dict.keys())

    if len(sorted_alphas) < 2:
        return 0.0

    pinball_losses = [compute_pinball_loss(y, quantiles_dict[a], a) for a in sorted_alphas]

    # Trapezoidal integration across alpha in [0, 1]
    try:
        from scipy.integrate import trapezoid
        crps = 2.0 * float(trapezoid(pinball_losses, sorted_alphas))
    except (ImportError, AttributeError):
        crps = 2.0 * float(
            sum(
                0.5 * (pinball_losses[i] + pinball_losses[i + 1]) * (sorted_alphas[i + 1] - sorted_alphas[i])
                for i in range(len(sorted_alphas) - 1)
            )
        )
    return round(crps, 4)


def compute_brier_score(
    observed_precip: Union[np.ndarray, pd.Series],
    predicted_probs: Union[np.ndarray, pd.Series],
    threshold_mm: float = 64.5,
) -> Dict[str, float]:
    """
    Computes Brier Score (BS) and Brier Skill Score (BSS) against climatological reference:
        BS = (1 / N) * sum (p_i - o_i)^2
        BSS = 1 - (BS / BS_ref)
    """
    obs = np.asarray(observed_precip, dtype=np.float32)
    p = np.asarray(predicted_probs, dtype=np.float32)

    o = (obs >= threshold_mm).astype(np.float32)
    bs = float(np.mean((p - o) ** 2))

    # Reference climatological base rate
    clim_rate = float(np.mean(o))
    bs_ref = float(np.mean((clim_rate - o) ** 2))

    bss = (1.0 - (bs / bs_ref)) if bs_ref > 1e-6 else 0.0

    return {
        "brier_score": round(bs, 4),
        "brier_skill_score": round(bss, 4),
        "climatological_rate": round(clim_rate, 4),
    }


def evaluate_quantile_calibration(
    observed_precip: Union[np.ndarray, pd.Series],
    quantiles_dict: Dict[float, np.ndarray],
) -> pd.DataFrame:
    """
    Verifies quantile reliability / nominal coverage:
    For nominal quantile alpha, the empirical frequency of observed precipitation
    falling below q_alpha should ideally equal alpha.
        Empirical Coverage = Mean(y_true <= q_alpha)
    """
    y = np.asarray(observed_precip, dtype=np.float32)
    records = []

    for alpha in sorted(quantiles_dict.keys()):
        q_vals = quantiles_dict[alpha]
        emp_coverage = float(np.mean(y <= q_vals))
        coverage_gap = emp_coverage - alpha

        records.append(
            {
                "nominal_quantile": alpha,
                "nominal_percentage": f"{int(alpha * 100)}%",
                "empirical_coverage": round(emp_coverage, 4),
                "coverage_gap": round(coverage_gap, 4),
                "status": "WELL_CALIBRATED" if abs(coverage_gap) <= 0.08 else "MISCALIBRATED",
            }
        )

    return pd.DataFrame(records)
