"""
Soft Regime-Aware NWP Error Repair Engine for RAIN-REPAIR X.
Implements:
    OPTION A: Single Unified LightGBM with Soft Regime Probability Features.
    OPTION B: Soft Mixture-of-Experts (MoE) with 6 Regime Regressors and Probability-Weighted Blending.
    Enforces Physical Non-Negativity: Corrected Rainfall = max(0.0, Raw NWP + Predicted Error).

Includes automated sample sufficiency gating to prevent overfitting in sparse regimes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingRegressor

from src.postprocessing.target import apply_error_correction, validate_target_isolation
from src.regime.schemas import REGIME_CLASSES
from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.repair_model")


class SharedRegimeLightGBMRepair:
    """
    OPTION A: Single Unified Gradient Boosted Regressor.
    Learns continuous error corrections conditioned on the full Error DNA,
    including continuous soft regime probability vectors P(Regime).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.04,
        max_depth: int = 7,
        num_leaves: int = 31,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_seed: int = 42,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_seed = random_seed

        self.model: Any = None
        self.feature_cols: List[str] = []
        self.feature_importances_: Dict[str, float] = {}
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[np.ndarray, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> SharedRegimeLightGBMRepair:
        validate_target_isolation(X_train)
        if X_val is not None:
            validate_target_isolation(X_val)

        cols = feature_cols or [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "signed_error", "absolute_error"]
        ]
        self.feature_cols = cols

        X_tr = X_train[cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_tr = np.asarray(y_train, dtype=np.float32)

        logger.info(
            "Fitting SharedRegimeLightGBMRepair (Option A) on %d samples across %d features.",
            len(X_tr),
            len(cols),
        )

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMRegressor(
                objective="regression",
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                random_state=self.random_seed,
                importance_type="gain",
                verbose=-1,
            )
            eval_set = None
            if X_val is not None and y_val is not None:
                X_v = X_val[cols].fillna(0.0).to_numpy(dtype=np.float32)
                y_v = np.asarray(y_val, dtype=np.float32)
                eval_set = [(X_v, y_v)]

            self.model.fit(X_tr, y_tr, eval_set=eval_set)
            importances = self.model.feature_importances_
            self.feature_importances_ = {
                cols[i]: round(float(importances[i]), 4)
                for i in range(len(cols))
            }
        else:
            self.model = HistGradientBoostingRegressor(
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
            )
            self.model.fit(X_tr, y_tr)

        self.is_fitted = True
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.model is None:
            raise RuntimeError("SharedRegimeLightGBMRepair must be fitted before predict.")
        X_mat = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        return self.model.predict(X_mat).astype(np.float32)

    def predict_corrected_rainfall(
        self,
        X: pd.DataFrame,
        nwp_col: str = "nwp_precip",
        enforce_non_negativity: bool = True,
    ) -> np.ndarray:
        raw_nwp = X[nwp_col].to_numpy(dtype=np.float32)
        pred_err = self.predict_error(X)
        return apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=enforce_non_negativity)


class SoftMixtureOfExpertsRepair:
    """
    OPTION B: Soft Mixture-of-Experts (MoE) Architecture.
    Trains dedicated regime-specific LightGBM experts for each of the 6 canonical regimes.
    At inference time, dynamically blends predictions weighted by soft regime probabilities:
        Predicted Error = Sum_k [ P(Regime_k) * Expert_k(X) ]

    Includes sample sufficiency check: If any regime has fewer than min_samples_per_regime,
    falls back cleanly to the shared model (Option A) to prevent overfitting on sparse classes.
    """

    def __init__(
        self,
        min_samples_per_regime: int = 25,
        n_estimators: int = 150,
        learning_rate: float = 0.04,
        max_depth: int = 6,
        random_seed: int = 42,
    ):
        self.min_samples_per_regime = min_samples_per_regime
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed

        # Dict mapping regime_name -> fitted expert model
        self.experts: Dict[str, Any] = {}
        self.shared_fallback_model: Optional[SharedRegimeLightGBMRepair] = None
        self.regimes_with_experts: List[str] = []
        self.regimes_using_fallback: List[str] = []
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[np.ndarray, pd.Series],
        dominant_regimes: Optional[pd.Series] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> SoftMixtureOfExpertsRepair:
        validate_target_isolation(X_train)

        cols = feature_cols or [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "signed_error", "absolute_error"]
        ]
        self.feature_cols = cols

        # 1. Fit Shared Fallback Model first
        self.shared_fallback_model = SharedRegimeLightGBMRepair(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            random_seed=self.random_seed,
        )
        self.shared_fallback_model.fit(X_train, y_train, feature_cols=cols)

        # 2. Derive dominant regimes if not explicitly provided
        if dominant_regimes is None:
            prob_cols = [f"prob_{r.lower()}" for r in REGIME_CLASSES if f"prob_{r.lower()}" in X_train.columns]
            if len(prob_cols) == len(REGIME_CLASSES):
                probs = X_train[prob_cols].to_numpy()
                dom_indices = np.argmax(probs, axis=1)
                dom_series = pd.Series([REGIME_CLASSES[idx] for idx in dom_indices], index=X_train.index)
            else:
                dom_series = pd.Series(["UNKNOWN"] * len(X_train), index=X_train.index)
        else:
            dom_series = dominant_regimes

        # 3. Train Regime Experts where data allows
        y_arr = np.asarray(y_train, dtype=np.float32)
        for r_name in REGIME_CLASSES:
            mask = (dom_series == r_name).to_numpy()
            n_samples = int(np.sum(mask))

            if n_samples >= self.min_samples_per_regime:
                X_sub = X_train[cols].iloc[mask].fillna(0.0).to_numpy(dtype=np.float32)
                y_sub = y_arr[mask]

                if HAS_LIGHTGBM:
                    expert = lgb.LGBMRegressor(
                        n_estimators=self.n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        random_state=self.random_seed,
                        verbose=-1,
                    )
                else:
                    expert = HistGradientBoostingRegressor(
                        max_iter=self.n_estimators,
                        learning_rate=self.learning_rate,
                        max_depth=self.max_depth,
                        random_state=self.random_seed,
                    )

                expert.fit(X_sub, y_sub)
                self.experts[r_name] = expert
                self.regimes_with_experts.append(r_name)
                logger.info("Fitted dedicated expert for regime %s with %d samples.", r_name, n_samples)
            else:
                self.regimes_using_fallback.append(r_name)
                logger.warning(
                    "Regime %s has insufficient samples (%d < %d threshold). Blending fallback to shared model.",
                    r_name,
                    n_samples,
                    self.min_samples_per_regime,
                )

        self.is_fitted = True
        return self

    def predict_error(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.shared_fallback_model is None:
            raise RuntimeError("SoftMixtureOfExpertsRepair must be fitted before predict.")

        n_samples = len(X)
        X_mat = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        shared_pred = self.shared_fallback_model.predict_error(X)

        # Check if soft regime probability columns are present
        prob_cols = [f"prob_{r.lower()}" for r in REGIME_CLASSES]
        has_all_probs = all(col in X.columns for col in prob_cols)

        if not has_all_probs or len(self.experts) == 0:
            # If probabilities absent or no experts survived sample check, use shared model
            return shared_pred

        prob_mat = X[prob_cols].to_numpy(dtype=np.float32)
        # Ensure rows sum to 1.0
        prob_mat = prob_mat / np.clip(np.sum(prob_mat, axis=1, keepdims=True), 1e-6, None)

        blended_error = np.zeros(n_samples, dtype=np.float32)

        for idx, r_name in enumerate(REGIME_CLASSES):
            weight = prob_mat[:, idx]
            if r_name in self.experts:
                expert_pred = self.experts[r_name].predict(X_mat).astype(np.float32)
            else:
                expert_pred = shared_pred

            blended_error += weight * expert_pred

        return blended_error.astype(np.float32)

    def predict_corrected_rainfall(
        self,
        X: pd.DataFrame,
        nwp_col: str = "nwp_precip",
        enforce_non_negativity: bool = True,
    ) -> np.ndarray:
        raw_nwp = X[nwp_col].to_numpy(dtype=np.float32)
        pred_err = self.predict_error(X)
        return apply_error_correction(raw_nwp, pred_err, enforce_non_negativity=enforce_non_negativity)
