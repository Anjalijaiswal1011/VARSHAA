"""
NWP Error Target Definition and Verification Module for RAIN-REPAIR X.
Standardizes:
    NWP Error = Observed Rainfall - Raw NWP Rainfall

Guarantees target isolation: ground-truth observations must NEVER
be accessible in predictor matrices at inference time.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.exceptions import TemporalLeakageError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.target")


def compute_nwp_error_target(
    obs_precip: Union[np.ndarray, pd.Series],
    raw_nwp_precip: Union[np.ndarray, pd.Series],
) -> pd.DataFrame:
    """
    Compute canonical NWP error targets:
        signed_error = Observed - Raw NWP
        absolute_error = |Observed - Raw NWP|

    Positive signed error indicates NWP under-forecast (dry bias).
    Negative signed error indicates NWP over-forecast (wet bias).
    """
    obs = np.asarray(obs_precip, dtype=np.float32)
    nwp = np.asarray(raw_nwp_precip, dtype=np.float32)

    if obs.shape != nwp.shape:
        raise ValueError(f"Shape mismatch: obs {obs.shape} vs nwp {nwp.shape}")

    signed_err = obs - nwp
    abs_err = np.abs(signed_err)

    return pd.DataFrame(
        {
            "signed_error": signed_err,
            "absolute_error": abs_err,
        }
    )


def apply_error_correction(
    raw_nwp_precip: Union[np.ndarray, pd.Series],
    predicted_error: Union[np.ndarray, pd.Series],
    enforce_non_negativity: bool = True,
) -> np.ndarray:
    """
    Reconstruct corrected rainfall from NWP baseline and predicted error:
        Corrected Rainfall = Raw NWP + Predicted Error

    Enforces physical realizability: rainfall >= 0.0 mm/24h.
    """
    nwp = np.asarray(raw_nwp_precip, dtype=np.float32)
    err = np.asarray(predicted_error, dtype=np.float32)

    corrected = nwp + err

    if enforce_non_negativity:
        # Precipitation cannot physically be negative
        corrected = np.maximum(corrected, 0.0)

    return corrected.astype(np.float32)


def validate_target_isolation(
    feature_df: pd.DataFrame,
    prohibited_cols: Optional[List[str]] = None,
) -> None:
    """
    Strict validation ensuring target labels and observations do not leak
    into the feature predictor matrix.
    Raises TemporalLeakageError if any prohibited column or alias is present.
    """
    if prohibited_cols is None:
        prohibited_cols = [
            "obs_precip",
            "observation",
            "ground_truth",
            "target",
            "signed_error",
            "nwp_error",
            "absolute_error",
            "residual_error",
        ]

    leaked = [col for col in feature_df.columns if col.lower() in [p.lower() for p in prohibited_cols]]
    if leaked:
        raise TemporalLeakageError(
            f"TARGET LEAKAGE DETECTED in feature matrix: prohibited columns found {leaked}. "
            f"Observed rainfall and target errors must be strictly isolated from predictors!"
        )
