"""
Feature Diagnostics & Redundancy Analysis Module for RAIN-REPAIR X.
Computes missingness, near-zero variance, and pairwise correlation diagnostics
without prematurely discarding domain-relevant features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


def generate_feature_diagnostics_report(
    df: pd.DataFrame,
    correlation_threshold: float = 0.95,
    variance_threshold: float = 1e-5,
) -> Dict[str, Any]:
    """
    Generate comprehensive health, variance, and redundancy diagnostics across features.

    Args:
        df: Input DataFrame containing features.
        correlation_threshold: Absolute Pearson correlation threshold for flagging high collinearity.
        variance_threshold: Minimum variance threshold to flag near-constant features.

    Returns:
        Structured diagnostic dictionary with:
        - 'total_samples': int
        - 'total_features': int
        - 'missingness_report': Dict[str, float]
        - 'near_zero_variance_features': List[str]
        - 'high_correlation_pairs': List[Dict[str, Any]]
        - 'summary_stats': Dict[str, Dict[str, float]]
    """
    num_df = df.select_dtypes(include=[np.number])
    feature_cols = [c for c in num_df.columns if not (c.endswith("_id") or c.startswith("cycle_"))]

    total_samples = len(df)
    missingness: Dict[str, float] = {}
    nzv_features: List[str] = []
    summary_stats: Dict[str, Dict[str, float]] = {}

    for c in feature_cols:
        series = num_df[c]
        missing_pct = float(series.isna().mean() * 100.0)
        missingness[c] = round(missing_pct, 2)

        var = float(series.var())
        if var < variance_threshold or np.isnan(var):
            nzv_features.append(c)

        summary_stats[c] = {
            "mean": round(float(series.mean()), 4) if not np.isnan(series.mean()) else 0.0,
            "std": round(float(series.std()), 4) if not np.isnan(series.std()) else 0.0,
            "min": round(float(series.min()), 4) if not np.isnan(series.min()) else 0.0,
            "max": round(float(series.max()), 4) if not np.isnan(series.max()) else 0.0,
        }

    # Pairwise correlation analysis
    corr_matrix = num_df[feature_cols].corr().abs()
    high_corr_pairs: List[Dict[str, Any]] = []

    for i in range(len(feature_cols)):
        for j in range(i + 1, len(feature_cols)):
            f1, f2 = feature_cols[i], feature_cols[j]
            corr_val = corr_matrix.loc[f1, f2]
            if not np.isnan(corr_val) and corr_val >= correlation_threshold:
                high_corr_pairs.append(
                    {
                        "feature_1": f1,
                        "feature_2": f2,
                        "abs_correlation": round(float(corr_val), 4),
                        "recommendation": "Review for collinearity; tree models can handle correlated features natively.",
                    }
                )

    return {
        "total_samples": total_samples,
        "total_features": len(feature_cols),
        "missingness_report": missingness,
        "near_zero_variance_features": nzv_features,
        "high_correlation_pairs": high_corr_pairs,
        "summary_stats": summary_stats,
    }
