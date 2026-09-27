"""
Verification & Residual Analysis Engine for RAIN-REPAIR X Post-Processing.
Evaluates:
1. Overall Residual Analysis (Raw NWP Error vs Corrected Error: MAE, RMSE, Bias).
2. Categorical Extreme Event Metrics at thresholds (POD, FAR, CSI, ETS).
3. Regime-Wise Performance Breakdown across the 6 canonical weather regimes.
   Strictly marks sparse regimes as 'INSUFFICIENT TEST SAMPLE' (< 10 samples).
4. Lead-Time Error Breakdown across actual horizons.
5. Spatial Error Analysis across grid coordinates.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.regime.schemas import REGIME_CLASSES
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.evaluation")

MIN_REGIME_SAMPLE_THRESHOLD: int = 10


def compute_deterministic_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> Dict[str, float]:
    """
    Standard meteorological continuous verification metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - Mean Bias: Mean(Predicted - Observed)
    - Pearson Correlation: r
    """
    obs = np.asarray(observed, dtype=np.float32)
    pred = np.asarray(predicted, dtype=np.float32)

    diff = pred - obs
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    bias = float(np.mean(diff))

    std_obs = float(np.std(obs))
    std_pred = float(np.std(pred))
    if std_obs > 1e-6 and std_pred > 1e-6:
        corr = float(np.corrcoef(obs, pred)[0, 1])
    else:
        corr = 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "correlation": round(corr, 4),
        "sample_count": len(obs),
    }


def compute_contingency_table_metrics(
    observed: np.ndarray,
    predicted: np.ndarray,
    threshold_mm: float = 64.5,
) -> Dict[str, Union[float, str]]:
    """
    Dichotomous contingency verification for heavy rainfall:
    - POD: Probability of Detection (Hits / (Hits + Misses))
    - FAR: False Alarm Ratio (False Alarms / (Hits + False Alarms))
    - CSI: Critical Success Index / Threat Score (Hits / (Hits + Misses + False Alarms))
    - ETS: Equitable Threat Score
    """
    obs_event = observed >= threshold_mm
    pred_event = predicted >= threshold_mm

    hits = int(np.sum(obs_event & pred_event))
    misses = int(np.sum(obs_event & ~pred_event))
    false_alarms = int(np.sum(~obs_event & pred_event))
    correct_negatives = int(np.sum(~obs_event & ~pred_event))
    total = len(observed)

    if (hits + misses) == 0:
        return {
            "pod": "NO_OBSERVED_EVENTS",
            "far": "NO_OBSERVED_EVENTS",
            "csi": "NO_OBSERVED_EVENTS",
            "ets": "NO_OBSERVED_EVENTS",
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
        }

    pod = float(hits / (hits + misses)) if (hits + misses) > 0 else 0.0
    far = float(false_alarms / (hits + false_alarms)) if (hits + false_alarms) > 0 else 0.0
    csi = float(hits / (hits + misses + false_alarms)) if (hits + misses + false_alarms) > 0 else 0.0

    # Equitable Threat Score (ETS)
    hits_ref = ((hits + misses) * (hits + false_alarms)) / total if total > 0 else 0.0
    denom = hits + misses + false_alarms - hits_ref
    ets = float((hits - hits_ref) / denom) if denom > 1e-6 else 0.0

    return {
        "pod": round(pod, 4),
        "far": round(far, 4),
        "csi": round(csi, 4),
        "ets": round(ets, 4),
        "hits": hits,
        "misses": misses,
        "false_alarms": false_alarms,
    }


def evaluate_residual_correction(
    observed: np.ndarray,
    raw_nwp: np.ndarray,
    corrected: np.ndarray,
    threshold_mm: float = 64.5,
) -> Dict[str, Any]:
    """
    Full residual performance comparison: Raw NWP vs Corrected.
    Computes absolute and percentage improvements.
    """
    raw_metrics = compute_deterministic_metrics(observed, raw_nwp)
    corrected_metrics = compute_deterministic_metrics(observed, corrected)

    raw_event = compute_contingency_table_metrics(observed, raw_nwp, threshold_mm)
    corr_event = compute_contingency_table_metrics(observed, corrected, threshold_mm)

    mae_reduction = raw_metrics["mae"] - corrected_metrics["mae"]
    mae_pct_improvement = (mae_reduction / raw_metrics["mae"] * 100.0) if raw_metrics["mae"] > 1e-4 else 0.0

    rmse_reduction = raw_metrics["rmse"] - corrected_metrics["rmse"]
    rmse_pct_improvement = (rmse_reduction / raw_metrics["rmse"] * 100.0) if raw_metrics["rmse"] > 1e-4 else 0.0

    return {
        "raw_nwp": {**raw_metrics, "event_metrics": raw_event},
        "corrected": {**corrected_metrics, "event_metrics": corr_event},
        "improvement": {
            "mae_reduction_mm": round(mae_reduction, 4),
            "mae_pct_improvement": round(mae_pct_improvement, 2),
            "rmse_reduction_mm": round(rmse_reduction, 4),
            "rmse_pct_improvement": round(rmse_pct_improvement, 2),
        },
    }


def evaluate_regime_wise_performance(
    df_eval: pd.DataFrame,
    obs_col: str = "obs_precip",
    raw_nwp_col: str = "nwp_precip",
    corrected_col: str = "corrected_precip",
    regime_col: str = "regime",
    min_samples: int = MIN_REGIME_SAMPLE_THRESHOLD,
) -> pd.DataFrame:
    """
    Evaluates correction performance separately for each of the six canonical weather regimes.
    If test sample count < min_samples, marks as 'INSUFFICIENT TEST SAMPLE' to prevent misleading claims.
    """
    records = []

    for r_name in REGIME_CLASSES:
        sub = df_eval[df_eval[regime_col] == r_name] if regime_col in df_eval.columns else pd.DataFrame()
        n = len(sub)

        if n < min_samples:
            records.append(
                {
                    "regime": r_name,
                    "sample_count": n,
                    "status": "INSUFFICIENT TEST SAMPLE",
                    "raw_mae": np.nan,
                    "corr_mae": np.nan,
                    "mae_reduction": np.nan,
                    "raw_rmse": np.nan,
                    "corr_rmse": np.nan,
                    "rmse_reduction": np.nan,
                }
            )
            continue

        obs = sub[obs_col].to_numpy(dtype=np.float32)
        raw = sub[raw_nwp_col].to_numpy(dtype=np.float32)
        corr = sub[corrected_col].to_numpy(dtype=np.float32)

        raw_m = compute_deterministic_metrics(obs, raw)
        corr_m = compute_deterministic_metrics(obs, corr)

        records.append(
            {
                "regime": r_name,
                "sample_count": n,
                "status": "VALIDATED",
                "raw_mae": raw_m["mae"],
                "corr_mae": corr_m["mae"],
                "mae_reduction": round(raw_m["mae"] - corr_m["mae"], 4),
                "raw_rmse": raw_m["rmse"],
                "corr_rmse": corr_m["rmse"],
                "rmse_reduction": round(raw_m["rmse"] - corr_m["rmse"], 4),
            }
        )

    return pd.DataFrame(records)


def evaluate_lead_time_performance(
    df_eval: pd.DataFrame,
    obs_col: str = "obs_precip",
    raw_nwp_col: str = "nwp_precip",
    corrected_col: str = "corrected_precip",
    lead_time_col: str = "lead_time",
) -> pd.DataFrame:
    """
    Evaluates post-processing performance across each actual available forecast horizon.
    """
    if lead_time_col not in df_eval.columns:
        return pd.DataFrame()

    unique_leads = sorted(df_eval[lead_time_col].unique())
    records = []

    for lt in unique_leads:
        sub = df_eval[df_eval[lead_time_col] == lt]
        obs = sub[obs_col].to_numpy(dtype=np.float32)
        raw = sub[raw_nwp_col].to_numpy(dtype=np.float32)
        corr = sub[corrected_col].to_numpy(dtype=np.float32)

        raw_m = compute_deterministic_metrics(obs, raw)
        corr_m = compute_deterministic_metrics(obs, corr)

        records.append(
            {
                "lead_time_hours": int(lt),
                "sample_count": len(sub),
                "raw_mae": raw_m["mae"],
                "corr_mae": corr_m["mae"],
                "mae_reduction": round(raw_m["mae"] - corr_m["mae"], 4),
                "raw_rmse": raw_m["rmse"],
                "corr_rmse": corr_m["rmse"],
                "rmse_reduction": round(raw_m["rmse"] - corr_m["rmse"], 4),
            }
        )

    return pd.DataFrame(records)


def evaluate_spatial_error_summary(
    df_eval: pd.DataFrame,
    obs_col: str = "obs_precip",
    raw_nwp_col: str = "nwp_precip",
    corrected_col: str = "corrected_precip",
    lat_col: str = "lat",
    lon_col: str = "lon",
) -> pd.DataFrame:
    """
    Aggregates spatial error distribution across geographic coordinates.
    Computes Raw vs Corrected MAE per grid coordinate.
    """
    if lat_col not in df_eval.columns or lon_col not in df_eval.columns:
        return pd.DataFrame()

    df_eval = df_eval.copy()
    df_eval["raw_abs_err"] = np.abs(df_eval[obs_col] - df_eval[raw_nwp_col])
    df_eval["corr_abs_err"] = np.abs(df_eval[obs_col] - df_eval[corrected_col])

    grouped = df_eval.groupby([lat_col, lon_col]).agg(
        raw_mae=("raw_abs_err", "mean"),
        corr_mae=("corr_abs_err", "mean"),
        sample_count=("raw_abs_err", "count"),
    ).reset_index()

    grouped["mae_reduction"] = (grouped["raw_mae"] - grouped["corr_mae"]).round(4)
    return grouped
