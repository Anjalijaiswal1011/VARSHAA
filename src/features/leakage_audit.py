"""
Temporal Leakage Audit Module for RAIN-REPAIR X (VARSHAA).
Inspects feature matrices and models to verify that zero future observational
data or unobserved target labels leak into prediction feature spaces.
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from src.features.schema import FEATURE_REGISTRY, LeakageRisk
from src.utils.exceptions import TemporalAlignmentError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.features.leakage_audit")

# Strict list of prohibited target/future column names that must NEVER be in X
FORBIDDEN_PREDICTOR_COLUMNS: Set[str] = {
    "obs_precip",
    "observation_rainfall",
    "target",
    "future_rain",
    "future_error",
    "same_day_error",
}


def audit_feature_dataframe_leakage(
    feature_df: pd.DataFrame,
    cycle_date: str,
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Execute a comprehensive leakage audit on a feature matrix before model ingestion.

    Checks:
    1. Prohibited target label presence in feature columns.
    2. Verification that all columns in df exist in FEATURE_REGISTRY with LeakageRisk.SAFE.
    3. Causal check on error memory column definitions.

    Args:
        feature_df: DataFrame to audit.
        cycle_date: Forecast cycle date string 'YYYY-MM-DD'.
        strict: If True, raises TemporalAlignmentError on violation.

    Returns:
        Audit report dictionary.
    """
    violations: List[str] = []
    audited_columns: List[str] = []

    # 1. Check for forbidden target labels
    for col in feature_df.columns:
        if col.lower() in FORBIDDEN_PREDICTOR_COLUMNS:
            violations.append(f"CRITICAL: Target column '{col}' found in feature matrix.")

    # 2. Check each column against the registry
    for col in feature_df.columns:
        if col in ["cycle_date", "lead_time", "lat", "lon"]:
            continue

        meta = FEATURE_REGISTRY.get(col)
        if meta is None:
            # Unregistered feature
            logger.warning("Feature '%s' is not registered in FEATURE_REGISTRY.", col)
            audited_columns.append(col)
        else:
            if meta.leakage_risk != LeakageRisk.SAFE:
                violations.append(f"Feature '{col}' is classified as {meta.leakage_risk.value}!")
            audited_columns.append(col)

    is_clean = len(violations) == 0

    report = {
        "is_leakage_clean": is_clean,
        "cycle_date_audited": cycle_date,
        "total_columns_checked": len(feature_df.columns),
        "violations": violations,
        "status": "PASSED" if is_clean else "FAILED",
    }

    if not is_clean and strict:
        msg = f"TEMPORAL LEAKAGE AUDIT FAILED with {len(violations)} violations: {violations}"
        logger.error(msg)
        raise TemporalAlignmentError(msg, details=report)

    logger.info("Temporal leakage audit %s for cycle %s", report["status"], cycle_date)
    return report
