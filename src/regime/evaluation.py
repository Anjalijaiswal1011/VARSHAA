"""
Regime Classification Evaluation & Error Analysis for RAIN-REPAIR X.
Computes multi-class classification metrics (Macro F1, Weighted F1, Accuracy, Precision, Recall),
confusion matrices, and granular regime-wise error reports with common confusion pathways.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.regime.schemas import REGIME_CLASSES, REGIME_TO_IDX, WeatherRegime
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.evaluation")


def evaluate_regime_predictions(
    y_true: Union[pd.Series, List[str], np.ndarray],
    y_pred: Union[pd.Series, List[str], np.ndarray],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute rigorous evaluation metrics for 6-class regime predictions.
    All scores are calculated from actual predictions without fabrication.
    """
    # Normalize labels to string list
    if isinstance(y_true, pd.Series):
        y_true_str = y_true.astype(str).tolist()
    else:
        y_true_str = [str(x) for x in y_true]

    if isinstance(y_pred, pd.Series):
        y_pred_str = y_pred.astype(str).tolist()
    else:
        y_pred_str = [str(x) for x in y_pred]

    # Global scores
    acc = float(accuracy_score(y_true_str, y_pred_str))
    macro_f1 = float(f1_score(y_true_str, y_pred_str, labels=REGIME_CLASSES, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true_str, y_pred_str, labels=REGIME_CLASSES, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true_str, y_pred_str, labels=REGIME_CLASSES, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true_str, y_pred_str, labels=REGIME_CLASSES, average="macro", zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true_str, y_pred_str, labels=REGIME_CLASSES)

    # Granular Regime-Wise Metrics & Common Confusions
    regime_breakdown: Dict[str, Dict[str, Any]] = {}
    for idx, regime in enumerate(REGIME_CLASSES):
        # Support
        row = cm[idx, :]
        support = int(np.sum(row))

        # Precision, recall, f1 for this specific class
        tp = row[idx]
        fp = int(np.sum(cm[:, idx]) - tp)
        fn = int(support - tp)

        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        # Find most common confusion (highest non-diagonal element in row)
        off_diag_indices = [j for j in range(len(REGIME_CLASSES)) if j != idx]
        if len(off_diag_indices) > 0 and support > 0:
            max_off_diag_j = off_diag_indices[int(np.argmax([row[j] for j in off_diag_indices]))]
            most_common_mistake = REGIME_CLASSES[max_off_diag_j]
            mistake_count = int(row[max_off_diag_j])
            mistake_pct = round((mistake_count / max(support, 1)) * 100.0, 1)
            confusion_desc = f"{most_common_mistake} ({mistake_count}/{support}, {mistake_pct}%)" if mistake_count > 0 else "None"
        else:
            confusion_desc = "None"

        regime_breakdown[regime] = {
            "samples": support,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "common_confusion": confusion_desc,
        }

    # Western Disturbance specific diagnostics
    wd_metrics = regime_breakdown.get(WeatherRegime.WESTERN_DISTURBANCE.value, {})
    wd_support = wd_metrics.get("samples", 0)
    wd_diagnosis = {
        "samples_present": wd_support,
        "is_sufficient": wd_support >= 30,
        "f1_score": wd_metrics.get("f1", 0.0),
        "notes": (
            "Sufficient representation detected in dataset."
            if wd_support >= 30
            else f"Low representation ({wd_support} samples). Western Disturbance is rare in July monsoon core data. Upper tropospheric shear proxies provide primary signal, but future winter/pre-monsoon seasonal expansion is required for full generalization."
        ),
    }

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "confusion_matrix": cm.tolist(),
        "class_order": REGIME_CLASSES,
        "regime_breakdown": regime_breakdown,
        "western_disturbance_diagnostics": wd_diagnosis,
        "total_samples": len(y_true_str),
    }


def format_regime_metrics_table(evaluation_results: Dict[str, Any]) -> str:
    """Format evaluation results into a clean markdown table."""
    lines = [
        "| Regime | Samples | Precision | Recall | F1-Score | Common Confusion |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
    ]
    breakdown = evaluation_results.get("regime_breakdown", {})
    for regime in REGIME_CLASSES:
        metrics = breakdown.get(regime, {})
        lines.append(
            f"| {regime} | {metrics.get('samples', 0)} | "
            f"{metrics.get('precision', 0.0):.4f} | {metrics.get('recall', 0.0):.4f} | "
            f"{metrics.get('f1', 0.0):.4f} | {metrics.get('common_confusion', 'None')} |"
        )
    return "\n".join(lines)
