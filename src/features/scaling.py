"""
Feature Preprocessing, Imputation, and Scaling Module for RAIN-REPAIR X.
Handles missing values, domain-aware outlier sanitization without discarding extreme weather,
and leakage-safe feature standardization for non-tree models.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.features.schema import FEATURE_REGISTRY, LeakageRisk
from src.utils.logging import get_logger

logger = get_logger("rain_repair.features.scaling")


def sanitize_domain_outliers(
    df: pd.DataFrame,
    preserve_extreme_weather: bool = True,
) -> pd.DataFrame:
    """
    Sanitize corrupted or impossible sensor/model artifacts without truncating legitimate extreme weather.
    For example: 300 mm/24h rainfall is a valid catastrophic cloudburst, whereas -5.0 mm is an invalid artifact.

    Args:
        df: Input DataFrame with raw/derived features.
        preserve_extreme_weather: If True, high precipitation is preserved up to 1500 mm.

    Returns:
        Cleaned copy of DataFrame.
    """
    cleaned = df.copy()

    # Physical clipping bounds
    if "nwp_precip" in cleaned.columns:
        cleaned["nwp_precip"] = np.clip(cleaned["nwp_precip"], 0.0, 1500.0)

    if "rh850" in cleaned.columns:
        cleaned["rh850"] = np.clip(cleaned["rh850"], 0.0, 100.0)

    if "t2m" in cleaned.columns:
        cleaned["t2m"] = np.clip(cleaned["t2m"], 220.0, 335.0)

    if "mslp" in cleaned.columns:
        cleaned["mslp"] = np.clip(cleaned["mslp"], 900.0, 1050.0)

    return cleaned


def handle_missing_features(
    df: pd.DataFrame,
    fill_defaults: Optional[Dict[str, float]] = None,
    add_missing_indicators: bool = False,
) -> pd.DataFrame:
    """
    Handle missing values with domain-specific defaults.

    Args:
        df: Input DataFrame.
        fill_defaults: Optional mapping of column to fill value.
        add_missing_indicators: If True, adds a binary column '{col}_is_missing'.

    Returns:
        Imputed DataFrame.
    """
    imputed = df.copy()

    standard_defaults: Dict[str, float] = {
        "nwp_precip": 0.0,
        "nwp_precip_log": 0.0,
        "t2m": 298.15,
        "rh850": 70.0,
        "mslp": 1008.0,
        "u850": 0.0,
        "v850": 0.0,
        "orographic_uplift": 0.0,
        "moisture_flux_conv": 0.0,
        "vorticity_850": 0.0,
        "bulk_wind_shear": 15.0,
        "error_memory_lag1": 0.0,
        "error_memory_lag3_mean": 0.0,
        "error_memory_lag7_mean": 0.0,
        "error_memory_lag14_mean": 0.0,
    }

    if fill_defaults is not None:
        standard_defaults.update(fill_defaults)

    for col, default_val in standard_defaults.items():
        if col in imputed.columns:
            n_missing = int(imputed[col].isna().sum())
            if n_missing > 0:
                if add_missing_indicators:
                    imputed[f"{col}_is_missing"] = imputed[col].isna().astype(np.int8)
                imputed[col] = imputed[col].fillna(default_val)

    return imputed


class RobustFeatureScaler:
    """
    Leakage-safe median/IQR feature scaler for non-tree models (e.g. neural nets, analog search).
    Fits statistics strictly on training data; transforms validation and test data.
    Tree-based models (LightGBM/XGBoost) can consume raw unscaled data directly.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None):
        self.feature_cols = feature_cols
        self.medians: Dict[str, float] = {}
        self.iqrs: Dict[str, float] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> RobustFeatureScaler:
        """Fit median and IQR statistics strictly on training DataFrame."""
        cols = self.feature_cols or [c for c in df.select_dtypes(include=[np.number]).columns if not c.endswith("_id")]
        self.feature_cols = cols

        for c in cols:
            series = df[c].dropna()
            med = float(series.median())
            q75, q25 = float(series.quantile(0.75)), float(series.quantile(0.25))
            iqr = max(q75 - q25, 1e-4)

            self.medians[c] = med
            self.iqrs[c] = iqr

        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply scaling: z = (x - median) / IQR."""
        if not self.is_fitted:
            raise RuntimeError("RobustFeatureScaler must be fitted on training data before transform.")

        scaled = df.copy()
        for c in self.feature_cols or []:
            if c in scaled.columns:
                scaled[c] = (scaled[c] - self.medians[c]) / self.iqrs[c]

        return scaled

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)
