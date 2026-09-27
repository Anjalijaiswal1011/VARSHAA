"""
Baseline Regime Classifiers for RAIN-REPAIR X.
Provides:
1. RuleBasedRegimeClassifier: Deterministic meteorological heuristic classifier.
2. MultinomialLogisticRegimeBaseline: Standard linear probabilistic baseline.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.regime.labels import derive_provisional_regime_labels
from src.regime.schemas import (
    IDX_TO_REGIME,
    REGIME_CLASSES,
    REGIME_TO_IDX,
    RegimeProbabilityVector,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.baseline")


class RuleBasedRegimeClassifier:
    """
    Deterministic rule-based baseline that applies heuristic meteorological logic
    and outputs smoothed probability distributions.
    """

    def __init__(self, smoothing_factor: float = 0.05):
        self.smoothing = smoothing_factor

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return (n_samples, 6) probability matrix."""
        provisional = derive_provisional_regime_labels(df)
        n = len(df)
        k = len(REGIME_CLASSES)
        probs = np.full((n, k), self.smoothing / (k - 1), dtype=np.float32)

        for i, label in enumerate(provisional):
            idx = REGIME_TO_IDX[label]
            probs[i, idx] = 1.0 - self.smoothing

        # Ensure exact row normalization
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, df: pd.DataFrame) -> List[str]:
        probs = self.predict_proba(df)
        best_indices = np.argmax(probs, axis=1)
        return [IDX_TO_REGIME[idx] for idx in best_indices]


class MultinomialLogisticRegimeBaseline:
    """
    Standard linear probabilistic baseline using Multinomial Logistic Regression.
    """

    def __init__(self, feature_cols: Optional[List[str]] = None, max_iter: int = 300):
        self.feature_cols = feature_cols
        self.max_iter = max_iter
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            multi_class="multinomial",
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=42,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> MultinomialLogisticRegimeBaseline:
        cols = self.feature_cols or [
            c for c in X.select_dtypes(include=[np.number]).columns
            if c not in ["lead_time", "lat", "lon"]
        ]
        self.feature_cols = cols

        X_num = X[cols].fillna(0.0)
        X_scaled = self.scaler.fit_transform(X_num)

        # Convert string labels to integer classes
        y_int = y.map(REGIME_TO_IDX).to_numpy()

        logger.info("Fitting Multinomial Logistic Regression baseline on %d samples.", len(X))
        self.model.fit(X_scaled, y_int)
        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("MultinomialLogisticRegimeBaseline must be fitted before predict_proba.")

        X_num = X[self.feature_cols].fillna(0.0)
        X_scaled = self.scaler.transform(X_num)
        probs_present = self.model.predict_proba(X_scaled)

        # Map to full 6-class space if subset present
        n_samples = len(X)
        full_probs = np.zeros((n_samples, len(REGIME_CLASSES)), dtype=np.float32)

        for col_idx, class_val in enumerate(self.model.classes_):
            full_probs[:, class_val] = probs_present[:, col_idx]

        # Row normalization
        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return (full_probs / row_sums).astype(np.float32)

    def predict(self, X: pd.DataFrame) -> List[str]:
        probs = self.predict_proba(X)
        return [IDX_TO_REGIME[i] for i in np.argmax(probs, axis=1)]
