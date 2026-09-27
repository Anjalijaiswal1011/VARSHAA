"""
Primary Supervised Weather Regime Classifier for RAIN-REPAIR X.
Uses LightGBM Multi-Class GBDT with class weighting for rare regimes,
time-aware training compliance, and structured soft probability generation.
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
    from sklearn.ensemble import HistGradientBoostingClassifier

from src.regime.schemas import (
    IDX_TO_REGIME,
    REGIME_CLASSES,
    REGIME_TO_IDX,
    RegimeProbabilityVector,
    WeatherRegime,
)
from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.classifier")


class LightGBMRegimeClassifier:
    """
    Primary Multi-Class Regime Classifier delivering calibrated soft probability vectors
    over the six canonical monsoon regimes.
    """

    def __init__(
        self,
        feature_cols: Optional[List[str]] = None,
        class_weight: str = "balanced",
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_seed: int = 42,
    ):
        self.feature_cols = feature_cols
        self.class_weight = class_weight
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.model: Any = None
        self.is_fitted = False
        self.feature_importances_: Dict[str, float] = {}

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> LightGBMRegimeClassifier:
        """
        Fit multi-class gradient boosted trees on training data.
        """
        cols = self.feature_cols or [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in ["lead_time", "lat", "lon", "obs_precip"]
        ]
        self.feature_cols = cols

        X_tr_num = X_train[cols].fillna(0.0)
        y_tr_int = y_train.map(REGIME_TO_IDX).to_numpy()

        logger.info(
            "Training Primary Regime Classifier on %d samples with %d features. Classes: %s",
            len(X_tr_num),
            len(cols),
            REGIME_CLASSES,
        )

        if HAS_LIGHTGBM:
            self.model = lgb.LGBMClassifier(
                objective="multiclass",
                num_class=len(REGIME_CLASSES),
                class_weight=self.class_weight,
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
                importance_type="gain",
                verbose=-1,
            )
            eval_set = None
            if X_val is not None and y_val is not None:
                val_y_mapped = y_val.map(REGIME_TO_IDX).to_numpy()
                train_classes = set(np.unique(y_tr_int))
                val_mask = np.isin(val_y_mapped, list(train_classes))
                if np.any(val_mask):
                    eval_set = [(X_val[cols].fillna(0.0).iloc[val_mask], val_y_mapped[val_mask])]

            self.model.fit(X_tr_num, y_tr_int, eval_set=eval_set)

            # Store feature importances
            importances = self.model.feature_importances_
            self.feature_importances_ = {
                cols[i]: round(float(importances[i]), 4)
                for i in range(len(cols))
            }
        else:
            logger.warning("LightGBM not installed. Falling back to HistGradientBoostingClassifier.")
            self.model = HistGradientBoostingClassifier(
                class_weight=self.class_weight,
                max_iter=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=self.random_seed,
            )
            self.model.fit(X_tr_num, y_tr_int)

        self.is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict continuous probability distribution over all six regimes.
        Guarantees that columns match exact REGIME_CLASSES indexing.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Classifier must be fitted before predict_proba.")

        X_num = X[self.feature_cols].fillna(0.0)
        raw_probs = self.model.predict_proba(X_num)

        # Handle potential subset of classes if some weren't in training data
        n_samples = len(X)
        probs = np.zeros((n_samples, len(REGIME_CLASSES)), dtype=np.float32)

        for col_idx, class_val in enumerate(self.model.classes_):
            probs[:, class_val] = raw_probs[:, col_idx]

        # Row normalization to guarantee sum = 1.0
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return (probs / row_sums).astype(np.float32)

    def predict_regime_vectors(self, X: pd.DataFrame) -> List[RegimeProbabilityVector]:
        """Convert predictions into structured typed probability vectors."""
        probs = self.predict_proba(X)
        return [RegimeProbabilityVector.from_array(probs[i]) for i in range(len(X))]

    def predict(self, X: pd.DataFrame) -> List[str]:
        """Return dominant predicted regime string per sample."""
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return [IDX_TO_REGIME[idx] for idx in best_indices]

    def save_artifact(
        self,
        artifact_dir: Optional[Path] = None,
        model_version: str = "v1.0.0",
    ) -> Path:
        """Save serialized model, feature list, and metadata manifest."""
        if artifact_dir is None:
            artifact_dir = config.paths.models_checkpoint_dir / "regime_classifier"

        artifact_dir.mkdir(parents=True, exist_ok=True)
        model_file = artifact_dir / f"regime_model_{model_version}.joblib"
        meta_file = artifact_dir / f"metadata_{model_version}.json"

        joblib.dump(self.model, model_file)

        # Top 10 features by importance
        sorted_feats = sorted(self.feature_importances_.items(), key=lambda x: x[1], reverse=True)[:10]

        metadata = {
            "model_version": model_version,
            "model_type": "LightGBMRegimeClassifier" if HAS_LIGHTGBM else "HistGradientBoostingClassifier",
            "class_names": REGIME_CLASSES,
            "feature_columns": self.feature_cols,
            "hyperparameters": {
                "class_weight": self.class_weight,
                "n_estimators": self.n_estimators,
                "learning_rate": self.learning_rate,
                "max_depth": self.max_depth,
            },
            "top_10_features": sorted_feats,
        }

        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info("Saved regime classifier artifact to %s", model_file)
        return model_file

    @classmethod
    def load_artifact(cls, artifact_dir: Path, model_version: str = "v1.0.0") -> LightGBMRegimeClassifier:
        """Load trained model and metadata."""
        model_file = artifact_dir / f"regime_model_{model_version}.joblib"
        meta_file = artifact_dir / f"metadata_{model_version}.json"

        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        instance = cls(
            feature_cols=metadata["feature_columns"],
            class_weight=metadata["hyperparameters"]["class_weight"],
            n_estimators=metadata["hyperparameters"]["n_estimators"],
            learning_rate=metadata["hyperparameters"]["learning_rate"],
            max_depth=metadata["hyperparameters"]["max_depth"],
        )
        instance.model = joblib.load(model_file)
        instance.is_fitted = True
        return instance
