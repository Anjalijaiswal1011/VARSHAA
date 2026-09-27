"""
Multi-Quantile Regression Engine for RAIN-REPAIR X.
Fits multi-quantile Gradient Boosted Decision Trees (LightGBM) under Pinball Loss:
    alpha in [0.10, 0.50, 0.75, 0.90, 0.95]

Features:
- Full Error DNA context (NWP, physics, terrain, soft regimes, transitions, memory, analogs).
- Post-hoc Rearrangement Operator enforcing strict monotonicity and non-negativity.
- Automated Pinball Loss verification.
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

from src.postprocessing.constraints import CANONICAL_QUANTILES, enforce_quantile_monotonicity
from src.postprocessing.target import validate_target_isolation
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.quantiles")


def compute_pinball_loss(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    alpha: float,
) -> float:
    """
    Computes asymmetric quantile pinball loss:
        L_alpha(y, q) = max(alpha * (y - q), (1 - alpha) * (q - y))
    """
    y = np.asarray(y_true, dtype=np.float32)
    q = np.asarray(y_pred, dtype=np.float32)
    diff = y - q
    loss = np.maximum(alpha * diff, (alpha - 1.0) * diff)
    return float(np.mean(loss))


class MultiQuantileRegressor:
    """
    Trains and predicts multi-quantile precipitation plumes across canonical confidence bounds:
        P10 (lower optimistic/conservative bound)
        P50 (median / deterministic expectation)
        P75 (moderate heavy rain upper boundary)
        P90 (convective extreme risk threshold)
        P95 (catastrophic tail bound)
    """

    def __init__(
        self,
        quantiles: Optional[List[float]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        num_leaves: int = 31,
        random_seed: int = 42,
    ):
        self.quantiles = quantiles or CANONICAL_QUANTILES
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.num_leaves = num_leaves
        self.random_seed = random_seed

        # Dict mapping alpha -> fitted model
        self.models: Dict[float, Any] = {}
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: Union[np.ndarray, pd.Series],
        feature_cols: Optional[List[str]] = None,
    ) -> MultiQuantileRegressor:
        """
        Fits independent quantile regressors minimizing pinball loss per alpha level.
        y_train is target precipitation (mm/24h).
        """
        validate_target_isolation(X_train)

        cols = feature_cols or [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "signed_error", "absolute_error"]
        ]
        self.feature_cols = cols

        X_mat = X_train[cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_arr = np.asarray(y_train, dtype=np.float32)

        logger.info(
            "Fitting MultiQuantileRegressor across quantiles %s with %d samples x %d features.",
            self.quantiles,
            len(X_mat),
            len(cols),
        )

        for alpha in self.quantiles:
            if HAS_LIGHTGBM:
                model = lgb.LGBMRegressor(
                    objective="quantile",
                    alpha=alpha,
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    num_leaves=self.num_leaves,
                    random_state=self.random_seed,
                    verbose=-1,
                )
            else:
                model = HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=alpha,
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_seed,
                )

            model.fit(X_mat, y_arr)
            self.models[alpha] = model

        self.is_fitted = True
        return self

    def predict_quantiles(
        self,
        X: pd.DataFrame,
        enforce_monotonicity: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts quantiles and applies post-hoc Rearrangement Operator to eliminate crossing.
        Returns:
            Dict mapping 'corrected_p10', 'corrected_p50', etc. -> np.ndarray
        """
        if not self.is_fitted:
            raise RuntimeError("MultiQuantileRegressor must be fitted before predict.")

        X_mat = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        raw_preds: Dict[float, np.ndarray] = {}

        for alpha, model in self.models.items():
            raw_preds[alpha] = model.predict(X_mat).astype(np.float32)

        if enforce_monotonicity:
            sorted_preds = enforce_quantile_monotonicity(raw_preds, enforce_non_negativity=True)
        else:
            sorted_preds = {a: np.maximum(raw_preds[a], 0.0) for a in raw_preds}

        # Map to standardized CONTRACT-ML-002 field names
        field_map = {
            0.10: "corrected_p10",
            0.50: "corrected_p50",
            0.75: "corrected_p75",
            0.90: "corrected_p90",
            0.95: "corrected_p95",
        }

        return {
            field_map.get(alpha, f"corrected_p{int(alpha*100)}"): sorted_preds[alpha]
            for alpha in sorted(sorted_preds.keys())
        }

    def evaluate_pinball_losses(
        self,
        X_test: pd.DataFrame,
        y_test: Union[np.ndarray, pd.Series],
    ) -> Dict[str, float]:
        """Calculates empirical pinball loss for each fitted quantile level."""
        preds = self.predict_quantiles(X_test, enforce_monotonicity=True)
        losses: Dict[str, float] = {}

        y_true = np.asarray(y_test, dtype=np.float32)
        field_map = {
            0.10: "corrected_p10",
            0.50: "corrected_p50",
            0.75: "corrected_p75",
            0.90: "corrected_p90",
            0.95: "corrected_p95",
        }

        for alpha in self.quantiles:
            col_name = field_map.get(alpha, f"corrected_p{int(alpha*100)}")
            if col_name in preds:
                loss = compute_pinball_loss(y_true, preds[col_name], alpha)
                losses[f"pinball_loss_p{int(alpha*100)}"] = round(loss, 4)

        return losses
