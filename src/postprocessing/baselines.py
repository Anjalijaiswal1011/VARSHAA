"""
Baseline Post-Processing Correction Models for RAIN-REPAIR X.
Provides benchmark post-processing strategies:
    BASELINE 1: Raw NWP (Identity / Zero-correction baseline)
    BASELINE 2: Statistical Bias Correction (Lead-time conditioned mean error)
    BASELINE 3: Generic ML Error Correction (LightGBM without regime or analog memory)

Establishes empirical baselines to verify value-add of regime-aware architectures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from src.postprocessing.target import apply_error_correction
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.baselines")


class RawNWPBaseline:
    """
    BASELINE 1: Raw NWP Forecast (Zero Correction).
    Predicts zero error; corrected rainfall equals raw NWP precipitation.
    """

    def fit(self, X: pd.DataFrame, y: Union[np.ndarray, pd.Series]) -> RawNWPBaseline:
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X), dtype=np.float32)

    def predict(self, X: pd.DataFrame, nwp_col: str = "nwp_precip") -> np.ndarray:
        raw_nwp = X[nwp_col].to_numpy(dtype=np.float32)
        return apply_error_correction(raw_nwp, self.predict_error(X), enforce_non_negativity=True)


class StatisticalBiasCorrectionBaseline:
    """
    BASELINE 2: Simple Statistical Bias Correction.
    Computes historical mean signed error (Obs - NWP) conditioned on lead time:
        bias(h) = Mean(Observed - NWP | lead_time = h)
        Corrected = max(0, NWP + bias(h))
    """

    def __init__(self, lead_time_col: str = "lead_time", nwp_col: str = "nwp_precip"):
        self.lead_time_col = lead_time_col
        self.nwp_col = nwp_col
        self.lead_time_biases: Dict[float, float] = {}
        self.global_mean_bias: float = 0.0
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
    ) -> StatisticalBiasCorrectionBaseline:
        """
        Fit historical mean bias.
        y is signed error = Obs - NWP.
        """
        signed_errors = np.asarray(y, dtype=np.float32)
        self.global_mean_bias = float(np.mean(signed_errors))

        if self.lead_time_col in X.columns:
            df_bias = pd.DataFrame(
                {
                    "lead_time": X[self.lead_time_col].values,
                    "error": signed_errors,
                }
            )
            grouped = df_bias.groupby("lead_time")["error"].mean()
            self.lead_time_biases = {float(k): float(v) for k, v in grouped.items()}
        else:
            self.lead_time_biases = {}

        self.is_fitted = True
        logger.info(
            "StatisticalBiasCorrection fitted. Global bias: %.3f mm. Lead-time biases: %s",
            self.global_mean_bias,
            self.lead_time_biases,
        )
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("StatisticalBiasCorrectionBaseline must be fitted before predict.")

        if self.lead_time_col in X.columns and self.lead_time_biases:
            return np.array(
                [self.lead_time_biases.get(float(lt), self.global_mean_bias) for lt in X[self.lead_time_col]],
                dtype=np.float32,
            )
        return np.full(len(X), self.global_mean_bias, dtype=np.float32)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        raw_nwp = X[self.nwp_col].to_numpy(dtype=np.float32)
        predicted_err = self.predict_error(X)
        return apply_error_correction(raw_nwp, predicted_err, enforce_non_negativity=True)


class GenericMLErrorCorrectionBaseline:
    """
    BASELINE 3: Generic Machine Learning Error Correction without Regime Awareness.
    Trains LightGBM regressor on standard atmospheric and terrain features,
    strictly omitting regime probabilities, regime transitions, and analog memory.
    """

    GENERIC_FEATURE_COLS: List[str] = [
        "nwp_precip",
        "nwp_precip_log",
        "lead_time",
        "lead_time_days",
        "lat",
        "lon",
        "elevation",
        "dem_slope",
        "dist_to_coast",
        "t2m",
        "rh850",
        "mslp",
        "u850",
        "v850",
        "wind_speed_850",
        "orographic_uplift",
        "vorticity_850",
        "moisture_flux_conv",
        "doy_sin",
        "doy_cos",
    ]

    def __init__(
        self,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.model: Any = None
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X: pd.DataFrame,
        y: Union[np.ndarray, pd.Series],
        feature_cols: Optional[List[str]] = None,
    ) -> GenericMLErrorCorrectionBaseline:
        cols = feature_cols or [c for c in self.GENERIC_FEATURE_COLS if c in X.columns]
        self.feature_cols = cols

        X_train = X[cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_train = np.asarray(y, dtype=np.float32)

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
                importance_type="gain",
                verbose=-1,
            )
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
            )

        self.model.fit(X_train, y_train)
        self.is_fitted = True
        logger.info(
            "GenericMLErrorCorrection fitted on %d samples with %d features.",
            len(X_train),
            len(cols),
        )
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("GenericMLErrorCorrectionBaseline must be fitted before predict.")
        X_test = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        return self.model.predict(X_test).astype(np.float32)

    def predict(self, X: pd.DataFrame, nwp_col: str = "nwp_precip") -> np.ndarray:
        raw_nwp = X[nwp_col].to_numpy(dtype=np.float32)
        predicted_err = self.predict_error(X)
        return apply_error_correction(raw_nwp, predicted_err, enforce_non_negativity=True)
