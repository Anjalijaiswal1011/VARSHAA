"""
Model Promotion Quality Gates & Validation Invariant Engine for RAIN-REPAIR X (PART 10).
Enforces physical invariants, calibration thresholds, regime-wise performance, and schema compatibility
before any candidate model can be promoted to staging or production.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.quality_gates")

MIN_REGIME_SAMPLE_SIZE: int = 25


@dataclass
class QualityGateResult:
    """Detailed summary of all quality gate checks."""
    gate_passed: bool
    status: str  # "PASSED", "FAILED", "CONDITIONAL"
    checks: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    failure_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.gate_passed

    @property
    def failures(self) -> List[str]:
        return self.failure_reasons

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        d["failures"] = self.failures
        return d


class ModelQualityGate:
    """
    Evaluates candidate models against predefined operational promotion criteria.
    Never automatically promotes a model merely because it is newer.
    """

    def __init__(
        self,
        max_rmse_threshold: float = 20.0,
        max_brier_threshold: float = 0.15,
        min_heavy_csi_threshold: float = 0.35,
        min_regime_sample_size: int = MIN_REGIME_SAMPLE_SIZE,
    ) -> None:
        self.max_rmse_threshold = max_rmse_threshold
        self.max_brier_threshold = max_brier_threshold
        self.min_heavy_csi_threshold = min_heavy_csi_threshold
        self.min_regime_sample_size = min_regime_sample_size

    def evaluate(
        self,
        candidate_metrics: Dict[str, Any],
        production_metrics: Optional[Dict[str, Any]] = None,
        target_status: str = "production",
        val_predictions: Optional[pd.DataFrame] = None,
    ) -> QualityGateResult:
        """Alias for evaluate_candidate with target status support."""
        regime_breakdown = candidate_metrics.get("regime_metrics")
        return self.evaluate_candidate(
            candidate_metrics=candidate_metrics,
            production_metrics=production_metrics,
            regime_breakdown=regime_breakdown,
            val_predictions=val_predictions,
            target_status=target_status,
        )

    def evaluate_candidate(
        self,
        candidate_metrics: Dict[str, Any],
        production_metrics: Optional[Dict[str, Any]] = None,
        regime_breakdown: Optional[Dict[str, Dict[str, Any]]] = None,
        val_predictions: Optional[pd.DataFrame] = None,
        target_status: str = "production",
    ) -> QualityGateResult:
        """
        Executes comprehensive quality gate audit.
        """
        checks: Dict[str, bool] = {}
        diagnostics: Dict[str, Any] = {}
        failure_reasons: List[str] = []
        warnings: List[str] = []

        # 1. Performance vs Production Baseline Gate
        cand_rmse = float(candidate_metrics.get("rmse", 999.0))
        checks["rmse_within_limit"] = cand_rmse <= self.max_rmse_threshold
        if not checks["rmse_within_limit"]:
            failure_reasons.append(f"Candidate RMSE ({cand_rmse:.2f}) exceeds max allowed threshold ({self.max_rmse_threshold:.2f})")

        if production_metrics:
            prod_rmse = float(production_metrics.get("rmse", cand_rmse))
            # No major regression: must not be worse than production by > 5%
            max_allowed = prod_rmse * 1.05
            checks["no_regression_vs_prod"] = cand_rmse <= max_allowed
            if not checks["no_regression_vs_prod"]:
                failure_reasons.append(f"RMSE degraded against production ({cand_rmse:.2f} > {max_allowed:.2f}) by > 5%")
        else:
            checks["no_regression_vs_prod"] = True

        # 2. Probability Calibration Gate
        brier = float(candidate_metrics.get("brier_score", candidate_metrics.get("brier_score_heavy", 0.10)))
        checks["calibration_acceptable"] = brier <= self.max_brier_threshold
        if not checks["calibration_acceptable"]:
            failure_reasons.append(f"Candidate Brier score ({brier:.4f}) degraded above threshold ({self.max_brier_threshold:.4f})")

        # 3. Extreme Event Behavior Gate
        csi_heavy = float(candidate_metrics.get("csi_heavy", candidate_metrics.get("heavy_rain_csi", 0.40)))
        checks["extreme_event_behavior_acceptable"] = csi_heavy >= self.min_heavy_csi_threshold
        if not checks["extreme_event_behavior_acceptable"]:
            failure_reasons.append(f"Candidate Heavy Rain CSI ({csi_heavy:.3f}) below minimum threshold ({self.min_heavy_csi_threshold:.3f})")

        # 4. Metric-level invariant checks
        q_violations = candidate_metrics.get("quantile_monotonicity_violations", 0)
        checks["quantile_monotonicity_satisfied"] = q_violations == 0
        if q_violations > 0:
            failure_reasons.append(f"Found {q_violations} quantile monotonicity violations.")

        neg_rainfall = candidate_metrics.get("negative_rainfall_count", 0)
        checks["non_negativity_satisfied"] = neg_rainfall == 0
        if neg_rainfall > 0:
            failure_reasons.append(f"Found {neg_rainfall} negative rainfall predictions.")

        # 5. Regime-Wise Performance Gate
        regime_checks: Dict[str, str] = {}
        if regime_breakdown:
            for r_name, r_stats in regime_breakdown.items():
                sample_count = r_stats.get("sample_size", r_stats.get("sample_count", 0))
                if sample_count < self.min_regime_sample_size:
                    msg = f"Regime '{r_name}': INSUFFICIENT_SAMPLE (N={sample_count} < {self.min_regime_sample_size})"
                    regime_checks[r_name] = msg
                    warnings.append(msg)
                else:
                    r_rmse = r_stats.get("rmse", 0.0)
                    regime_checks[r_name] = "PASSED" if r_rmse <= (self.max_rmse_threshold * 1.25) else "FAILED"
        diagnostics["regime_audit"] = regime_checks
        checks["regime_wise_acceptable"] = not any("FAILED" in status for status in regime_checks.values())

        # 6. Physical Invariant Enforcement on Sample Predictions (if provided)
        if val_predictions is not None and not val_predictions.empty:
            has_negative = False
            for col in ["p50_rainfall", "corrected_p50", "corrected_rainfall", "raw_nwp_rainfall"]:
                if col in val_predictions and (val_predictions[col] < 0.0).any():
                    has_negative = True
                    break
            if has_negative:
                checks["non_negativity_satisfied"] = False
                failure_reasons.append("Non-negativity invariant violated: found negative rainfall predictions.")

            crossing_detected = False
            if "p50_rainfall" in val_predictions and "p75_rainfall" in val_predictions and "p90_rainfall" in val_predictions:
                c1 = (val_predictions["p50_rainfall"] > val_predictions["p75_rainfall"] + 1e-4).any()
                c2 = (val_predictions["p75_rainfall"] > val_predictions["p90_rainfall"] + 1e-4).any()
                crossing_detected = c1 or c2

            if crossing_detected:
                checks["quantile_monotonicity_satisfied"] = False
                failure_reasons.append("Quantile crossing invariant violated: P50 > P75 or P75 > P90.")

        all_passed = all(checks.values())
        return QualityGateResult(
            gate_passed=all_passed,
            status="PASSED" if all_passed else "FAILED",
            checks=checks,
            diagnostics=diagnostics,
            failure_reasons=failure_reasons,
            warnings=warnings,
        )

