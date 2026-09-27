"""
Extreme Rainfall Threshold Classification & Probability Calibration Engine for RAIN-REPAIR X.
Predicts calibrated exceedance probabilities for standard IMD operational alert categories:
    - Heavy Rainfall:      P(Rain >= 64.5 mm/24h)
    - Very Heavy Rainfall: P(Rain >= 115.6 mm/24h)
    - Extremely Heavy:     P(Rain >= 204.5 mm/24h)

Anti-Leakage Calibration Protocol:
1. TRAIN: Fit base Gradient Boosted Decision Tree (LightGBM).
2. VALIDATION: Fit probability calibrator (Platt scaling / Sigmoid or Isotonic) strictly on hold-out validation set.
3. TEST: Evaluate calibrated probabilities on completely unseen test data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    from sklearn.ensemble import HistGradientBoostingClassifier

from src.postprocessing.constraints import enforce_probability_monotonicity
from src.postprocessing.target import validate_target_isolation
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.thresholds")

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "heavy": 64.5,
    "very_heavy": 115.6,
    "extreme": 204.5,
}

EVENT_MODEL_VERSION: str = "v1.0.0-event"
CALIBRATION_VERSION: str = "v1.0.0-platt"


class ExtremeThresholdClassifier:
    """
    Dedicated classifier bank estimating calibrated exceedance probabilities for extreme precipitation.
    """

    def __init__(
        self,
        thresholds: Optional[Dict[str, float]] = None,
        n_estimators: int = 150,
        learning_rate: float = 0.04,
        max_depth: int = 5,
        random_seed: int = 42,
        model_version: str = EVENT_MODEL_VERSION,
        calibration_version: str = CALIBRATION_VERSION,
    ):
        self.thresholds = thresholds or DEFAULT_THRESHOLDS
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.random_seed = random_seed
        self.model_version = model_version
        self.calibration_version = calibration_version

        # Dict mapping threshold_key -> (mode, model, optional_calibrator)
        self.classifiers: Dict[str, Tuple[str, Any, Optional[Any]]] = {}
        self.feature_cols: List[str] = []
        self.is_fitted = False
        self.is_calibrated = False

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train_precip: Union[np.ndarray, pd.Series],
        X_val: Optional[pd.DataFrame] = None,
        y_val_precip: Optional[Union[np.ndarray, pd.Series]] = None,
        feature_cols: Optional[List[str]] = None,
        calibration_method: str = "sigmoid",  # Platt scaling
    ) -> ExtremeThresholdClassifier:
        """
        Two-stage fit:
        Stage 1: Fits base classifiers on X_train.
        Stage 2: If X_val and y_val_precip are provided, fits probability calibration
                 mapping strictly on validation data to prevent calibration leakage.
        """
        validate_target_isolation(X_train)
        if X_val is not None:
            validate_target_isolation(X_val)

        cols = feature_cols or [
            c for c in X_train.select_dtypes(include=[np.number]).columns
            if c not in ["obs_precip", "signed_error", "absolute_error"]
        ]
        self.feature_cols = cols

        X_tr = X_train[cols].fillna(0.0).to_numpy(dtype=np.float32)
        y_tr = np.asarray(y_train_precip, dtype=np.float32)

        has_val = X_val is not None and y_val_precip is not None
        if has_val:
            X_v = X_val[cols].fillna(0.0).to_numpy(dtype=np.float32)
            y_v = np.asarray(y_val_precip, dtype=np.float32)

        for name, thresh in self.thresholds.items():
            binary_labels_tr = (y_tr >= thresh).astype(int)
            n_pos = int(np.sum(binary_labels_tr))
            n_total = len(binary_labels_tr)

            if n_pos < 3:
                # Sparse event prior
                logger.warning(
                    "Threshold %s (>= %.1f mm) has only %d positive training samples. Initializing empirical fallback.",
                    name,
                    thresh,
                    n_pos,
                )
                self.classifiers[name] = ("constant", float(n_pos / max(n_total, 1)), None)
                continue

            if HAS_LIGHTGBM:
                clf = lgb.LGBMClassifier(
                    n_estimators=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    class_weight="balanced",
                    random_state=self.random_seed,
                    verbose=-1,
                )
            else:
                clf = HistGradientBoostingClassifier(
                    class_weight="balanced",
                    max_iter=self.n_estimators,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    random_state=self.random_seed,
                )

            clf.fit(X_tr, binary_labels_tr)

            calibrator = None
            if has_val:
                binary_labels_v = (y_v >= thresh).astype(int)
                if np.sum(binary_labels_v) >= 2 and np.sum(binary_labels_v == 0) >= 2:
                    val_logits = clf.predict_proba(X_v)[:, 1].reshape(-1, 1)
                    # Platt scaling via LogisticRegression on validation logits
                    calibrator = LogisticRegression(C=1.0, solver="lbfgs")
                    calibrator.fit(val_logits, binary_labels_v)
                    self.is_calibrated = True

            self.classifiers[name] = ("model", clf, calibrator)
            logger.info("Trained threshold classifier for %s (>= %.1f mm). Calibrated on Val: %s.", name, thresh, calibrator is not None)

        self.is_fitted = True
        return self

    def predict_probabilities(
        self,
        X: pd.DataFrame,
        enforce_monotonicity: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Predicts calibrated probabilities and enforces cumulative threshold monotonicity:
            P(extreme) <= P(very_heavy) <= P(heavy)
        """
        if not self.is_fitted:
            raise RuntimeError("ExtremeThresholdClassifier must be fitted before predict.")

        X_mat = X[self.feature_cols].fillna(0.0).to_numpy(dtype=np.float32)
        n_samples = len(X)
        raw_probs: Dict[str, np.ndarray] = {}

        for name in ["heavy", "very_heavy", "extreme"]:
            if name not in self.classifiers:
                raw_probs[name] = np.zeros(n_samples, dtype=np.float32)
                continue

            mode, clf, calibrator = self.classifiers[name]
            if mode == "constant":
                raw_probs[name] = np.full(n_samples, clf, dtype=np.float32)
            else:
                probs = clf.predict_proba(X_mat)[:, 1].astype(np.float32)
                if calibrator is not None:
                    # Apply Platt scaling
                    calib_probs = calibrator.predict_proba(probs.reshape(-1, 1))[:, 1]
                    raw_probs[name] = calib_probs.astype(np.float32)
                else:
                    raw_probs[name] = probs

        if enforce_monotonicity:
            p_h, p_vh, p_ext = enforce_probability_monotonicity(
                raw_probs["heavy"],
                raw_probs["very_heavy"],
                raw_probs["extreme"],
            )
        else:
            p_h = np.clip(raw_probs["heavy"], 0.0, 1.0)
            p_vh = np.clip(raw_probs["very_heavy"], 0.0, 1.0)
            p_ext = np.clip(raw_probs["extreme"], 0.0, 1.0)

        return {
            "prob_heavy_rain": p_h,
            "prob_very_heavy_rain": p_vh,
            "prob_extreme_rain": p_ext,
        }
