"""
Probability Calibration Evaluator & Calibrator for Weather Regime Probabilities.
Evaluates multi-class calibration reliability via Multi-Class Brier Score,
Expected Calibration Error (ECE), and Log Loss.
Allows fitting probability calibration on validation data without leakage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

from src.regime.schemas import REGIME_CLASSES, REGIME_TO_IDX
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.calibration")


def compute_multiclass_brier_score(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Compute multi-class Brier Score:
    BS = (1 / N) * sum_i sum_k (p_ik - y_ik)^2
    Lower is better (0.0 is perfect calibration and sharpness).
    """
    n_samples, n_classes = y_prob.shape
    y_one_hot = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i, idx in enumerate(y_true_indices):
        if 0 <= idx < n_classes:
            y_one_hot[i, idx] = 1.0

    brier = np.mean(np.sum((y_prob - y_one_hot) ** 2, axis=1))
    return float(brier)


def compute_expected_calibration_error(
    y_true_indices: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) based on dominant confidence vs accuracy:
    ECE = sum_{m=1}^M (|B_m| / N) * |acc(B_m) - conf(B_m)|
    """
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true_indices).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true_indices)

    if total_samples == 0:
        return 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            bin_acc = np.mean(accuracies[in_bin])
            bin_conf = np.mean(confidences[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)

    return float(ece)


def evaluate_probability_calibration(
    y_true: Union[pd.Series, List[str], np.ndarray],
    y_prob: np.ndarray,
) -> Dict[str, Any]:
    """
    Comprehensive calibration evaluation report for a probability prediction matrix.
    """
    if isinstance(y_true, pd.Series):
        if y_true.dtype == "object" or isinstance(y_true.iloc[0], str):
            y_indices = y_true.map(REGIME_TO_IDX).fillna(0).astype(int).to_numpy()
        else:
            y_indices = y_true.to_numpy()
    elif isinstance(y_true, (list, tuple)) and len(y_true) > 0 and isinstance(y_true[0], str):
        y_indices = np.array([REGIME_TO_IDX.get(s, 0) for s in y_true], dtype=int)
    else:
        y_indices = np.asarray(y_true, dtype=int)

    # Multi-class Brier Score
    brier = compute_multiclass_brier_score(y_indices, y_prob)
    ece = compute_expected_calibration_error(y_indices, y_prob)

    # Log loss (with epsilon clipping to avoid inf)
    clipped_probs = np.clip(y_prob, 1e-7, 1.0 - 1e-7)
    clipped_probs = clipped_probs / clipped_probs.sum(axis=1, keepdims=True)

    try:
        loss = float(log_loss(y_indices, clipped_probs, labels=list(range(len(REGIME_CLASSES)))))
    except Exception as exc:
        logger.warning("Log loss calculation error: %s", exc)
        loss = float("nan")

    # Per-regime Brier score
    per_regime_brier = {}
    for idx, regime_name in enumerate(REGIME_CLASSES):
        y_bin = (y_indices == idx).astype(int)
        p_bin = y_prob[:, idx]
        try:
            per_regime_brier[regime_name] = round(float(brier_score_loss(y_bin, p_bin)), 4)
        except Exception:
            per_regime_brier[regime_name] = float("nan")

    return {
        "multiclass_brier_score": round(brier, 4),
        "expected_calibration_error": round(ece, 4),
        "log_loss": round(loss, 4) if not np.isnan(loss) else None,
        "per_regime_brier": per_regime_brier,
        "n_samples": len(y_indices),
    }


class RegimeCalibrator:
    """
    Fits and evaluates post-hoc probability calibration (Platt scaling / Isotonic)
    strictly on validation data to prevent data leakage.
    """

    def __init__(self, method: str = "sigmoid"):
        """
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        """
        self.method = method
        self.calibrator: Optional[CalibratedClassifierCV] = None
        self.is_calibrated = False

    def fit_calibration(
        self,
        base_estimator: Any,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        feature_cols: List[str],
    ) -> RegimeCalibrator:
        """
        Fit calibrator using validation data with cv='prefit'.
        """
        y_val_int = y_val.map(REGIME_TO_IDX).to_numpy()
        X_val_num = X_val[feature_cols].fillna(0.0)

        logger.info("Fitting probability calibration (%s) on %d validation samples.", self.method, len(X_val))

        self.calibrator = CalibratedClassifierCV(
            estimator=base_estimator,
            method=self.method,
            cv="prefit",
        )
        self.calibrator.fit(X_val_num, y_val_int)
        self.is_calibrated = True
        return self

    def predict_proba(self, X: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        if not self.is_calibrated or self.calibrator is None:
            raise RuntimeError("RegimeCalibrator is not fitted.")

        X_num = X[feature_cols].fillna(0.0)
        raw_probs = self.calibrator.predict_proba(X_num)

        # Ensure shape matches 6 classes
        n_samples = len(X)
        full_probs = np.zeros((n_samples, len(REGIME_CLASSES)), dtype=np.float32)

        for col_idx, class_val in enumerate(self.calibrator.classes_):
            if class_val < len(REGIME_CLASSES):
                full_probs[:, class_val] = raw_probs[:, col_idx]

        row_sums = full_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return (full_probs / row_sums).astype(np.float32)
