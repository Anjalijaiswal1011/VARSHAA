"""
Operational Model Performance & Calibration Monitoring Engine for RAAP-X (Phase 7 Section 10 & 11).
Computes continuous, quantile, and categorical event verification metrics when ground truth observations arrive.
Evaluates degradation against configurable thresholds without automatic retraining.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.mlops.alerts import AlertManager
from src.postprocessing.quantiles import compute_pinball_loss
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.performance_monitor")

# IMD Standard Rainfall Warning Thresholds (mm/24h)
THRESHOLD_HEAVY: float = 64.5
THRESHOLD_VERY_HEAVY: float = 115.6
THRESHOLD_EXTREME: float = 204.5


@dataclass
class ContinuousMetrics:
    mae: float
    rmse: float
    bias: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class QuantileMetrics:
    pinball_p50: float
    pinball_p75: float
    pinball_p90: float
    mean_pinball: float
    coverage_p50: float
    coverage_p75: float
    coverage_p90: float
    coverage_error: float  # Mean |observed_coverage - nominal_tau|

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class EventMetrics:
    threshold_mm: float
    hits: int
    misses: int
    false_alarms: int
    correct_negatives: int
    csi: float
    pod: float        # Probability of Detection / Recall
    far: float        # False Alarm Ratio
    precision: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceEvaluationReport:
    """Standardized performance monitoring report conforming to Phase 7 Section 10."""
    report_id: str
    pipeline_run_id: str
    cycle_date: str
    model_version: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    continuous: ContinuousMetrics = field(default_factory=lambda: ContinuousMetrics(0.0, 0.0, 0.0))
    quantiles: QuantileMetrics = field(default_factory=lambda: QuantileMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    events: Dict[str, EventMetrics] = field(default_factory=dict)
    reference_comparison: Dict[str, Any] = field(default_factory=dict)
    degradation_detected: bool = False
    retraining_review_required: bool = False
    alerts_triggered: List[str] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["continuous"] = self.continuous.to_dict()
        d["quantiles"] = self.quantiles.to_dict()
        d["events"] = {k: v.to_dict() for k, v in self.events.items()}
        return d


class ModelPerformanceMonitor:
    """
    Automated verification evaluation comparing forecasts with observed ground truth.
    Triggers structured alerts upon degradation while preserving model stability.
    """

    def __init__(
        self,
        mae_threshold: float = 14.0,
        rmse_threshold: float = 22.0,
        coverage_error_threshold: float = 0.12,
        min_csi_heavy_threshold: float = 0.30,
        alert_manager: Optional[AlertManager] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.mae_threshold = mae_threshold
        self.rmse_threshold = rmse_threshold
        self.coverage_error_threshold = coverage_error_threshold
        self.min_csi_heavy_threshold = min_csi_heavy_threshold
        self.alert_manager = alert_manager or AlertManager.get_default_manager()
        self.storage_dir = storage_dir or Path("logs/performance_reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def compute_continuous(y_true: np.ndarray, y_pred: np.ndarray) -> ContinuousMetrics:
        """Computes continuous verification metrics: MAE, RMSE, Bias."""
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        t = y_true[mask]
        p = y_pred[mask]
        if len(t) == 0:
            return ContinuousMetrics(0.0, 0.0, 0.0)

        err = p - t
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        bias = float(np.mean(err))
        return ContinuousMetrics(round(mae, 3), round(rmse, 3), round(bias, 3))

    @staticmethod
    def compute_quantiles(
        y_true: np.ndarray,
        p50: np.ndarray,
        p75: np.ndarray,
        p90: np.ndarray,
    ) -> QuantileMetrics:
        """Computes quantile verification metrics: pinball loss, coverage, and calibration error."""
        mask = (~np.isnan(y_true)) & (~np.isnan(p50)) & (~np.isnan(p75)) & (~np.isnan(p90))
        t = y_true[mask]
        if len(t) == 0:
            return QuantileMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Pinball loss
        l50 = float(compute_pinball_loss(t, p50[mask], 0.50))
        l75 = float(compute_pinball_loss(t, p75[mask], 0.75))
        l90 = float(compute_pinball_loss(t, p90[mask], 0.90))
        mean_pb = float((l50 + l75 + l90) / 3.0)

        # Empirical coverage
        c50 = float(np.mean(t <= p50[mask]))
        c75 = float(np.mean(t <= p75[mask]))
        c90 = float(np.mean(t <= p90[mask]))

        # Mean calibration error
        cov_err = float((abs(c50 - 0.50) + abs(c75 - 0.75) + abs(c90 - 0.90)) / 3.0)

        return QuantileMetrics(
            pinball_p50=round(l50, 3),
            pinball_p75=round(l75, 3),
            pinball_p90=round(l90, 3),
            mean_pinball=round(mean_pb, 3),
            coverage_p50=round(c50, 4),
            coverage_p75=round(c75, 4),
            coverage_p90=round(c90, 4),
            coverage_error=round(cov_err, 4),
        )

    @staticmethod
    def compute_event(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> EventMetrics:
        """Computes 2x2 contingency table and categorical metrics (CSI, POD, FAR, Precision)."""
        mask = (~np.isnan(y_true)) & (~np.isnan(y_pred))
        t = y_true[mask] >= threshold
        p = y_pred[mask] >= threshold

        hits = int(np.sum(t & p))
        misses = int(np.sum(t & ~p))
        false_alarms = int(np.sum(~t & p))
        correct_negatives = int(np.sum(~t & ~p))

        denom_csi = hits + misses + false_alarms
        csi = float(hits / denom_csi) if denom_csi > 0 else (1.0 if not t.any() and not p.any() else 0.0)

        denom_pod = hits + misses
        pod = float(hits / denom_pod) if denom_pod > 0 else 0.0

        denom_far = hits + false_alarms
        far = float(false_alarms / denom_far) if denom_far > 0 else 0.0
        prec = float(hits / denom_far) if denom_far > 0 else 0.0

        return EventMetrics(
            threshold_mm=threshold,
            hits=hits,
            misses=misses,
            false_alarms=false_alarms,
            correct_negatives=correct_negatives,
            csi=round(csi, 4),
            pod=round(pod, 4),
            far=round(far, 4),
            precision=round(prec, 4),
        )

    def evaluate_observations(
        self,
        y_true: np.ndarray,
        p50: np.ndarray,
        p75: np.ndarray,
        p90: np.ndarray,
        cycle_date: str,
        pipeline_run_id: str,
        model_version: str,
        reference_metrics: Optional[Dict[str, float]] = None,
    ) -> PerformanceEvaluationReport:
        """
        Executes full evaluation when verified observations become available.
        Checks for degradation against configured thresholds and historical reference.
        """
        import uuid
        report_id = f"pfr_{uuid.uuid4().hex[:10]}"
        alerts_triggered: List[str] = []

        cont = self.compute_continuous(y_true, p50)
        quant = self.compute_quantiles(y_true, p50, p75, p90)

        events: Dict[str, EventMetrics] = {
            "heavy_64_5": self.compute_event(y_true, p90, THRESHOLD_HEAVY),
            "very_heavy_115_6": self.compute_event(y_true, p90, THRESHOLD_VERY_HEAVY),
            "extreme_204_5": self.compute_event(y_true, p90, THRESHOLD_EXTREME),
        }

        # Reference comparison & degradation checks
        ref = reference_metrics or {"mae": 9.70, "rmse": 16.20, "csi_heavy": 0.44, "coverage_error": 0.05}
        degradations: List[str] = []

        if cont.mae > self.mae_threshold:
            degradations.append(f"MAE ({cont.mae:.2f}mm) exceeds threshold ({self.mae_threshold:.2f}mm).")

        if cont.rmse > self.rmse_threshold:
            degradations.append(f"RMSE ({cont.rmse:.2f}mm) exceeds threshold ({self.rmse_threshold:.2f}mm).")

        if quant.coverage_error > self.coverage_error_threshold:
            degradations.append(f"Quantile coverage error ({quant.coverage_error:.3f}) exceeds threshold ({self.coverage_error_threshold:.3f}).")

        csi_heavy = events["heavy_64_5"].csi
        if csi_heavy < self.min_csi_heavy_threshold:
            degradations.append(f"Heavy rain CSI ({csi_heavy:.3f}) below minimum ({self.min_csi_heavy_threshold:.3f}).")

        # Relative change check vs reference
        ref_mae = ref.get("mae", 9.70)
        mae_degradation_pct = ((cont.mae - ref_mae) / ref_mae * 100.0) if ref_mae > 0 else 0.0

        degradation_detected = len(degradations) > 0
        retraining_review = degradation_detected

        if degradation_detected:
            alt1 = self.alert_manager.record_alert(
                category="MODEL_PERFORMANCE_DEGRADATION",
                message=f"Model performance degraded for cycle {cycle_date}: {'; '.join(degradations)}",
                severity="WARNING",
                pipeline_run_id=pipeline_run_id,
                model_version=model_version,
                recommended_action="Flag RETRAINING_REVIEW_REQUIRED. Evaluate model against holdout verification data; consider rollback.",
            )
            alerts_triggered.append(alt1.alert_id)

        if quant.coverage_error > self.coverage_error_threshold:
            alt2 = self.alert_manager.record_alert(
                category="CALIBRATION_DEGRADATION",
                message=f"Calibration error {quant.coverage_error:.3f} exceeded threshold {self.coverage_error_threshold:.3f}.",
                severity="WARNING",
                pipeline_run_id=pipeline_run_id,
                model_version=model_version,
                recommended_action="Inspect Platt scaling parameters and PIT histograms before retraining.",
            )
            alerts_triggered.append(alt2.alert_id)

        report = PerformanceEvaluationReport(
            report_id=report_id,
            pipeline_run_id=pipeline_run_id,
            cycle_date=cycle_date,
            model_version=model_version,
            continuous=cont,
            quantiles=quant,
            events=events,
            reference_comparison={
                "reference_mae": ref_mae,
                "mae_degradation_pct": round(mae_degradation_pct, 2),
                "reference_csi_heavy": ref.get("csi_heavy", 0.44),
                "csi_difference": round(csi_heavy - ref.get("csi_heavy", 0.44), 4),
            },
            degradation_detected=degradation_detected,
            retraining_review_required=retraining_review,
            alerts_triggered=alerts_triggered,
            diagnostics={"degradation_reasons": degradations},
        )

        self._persist_report(report)
        return report

    def _persist_report(self, report: PerformanceEvaluationReport) -> None:
        p = self.storage_dir / f"{report.report_id}.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to persist performance report: %s", e)
