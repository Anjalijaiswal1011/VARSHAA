"""
Evaluation and Validation Module for RAIN-REPAIR X.
Provides rigorous Phase 4 validation, calibration checks, and failure mode guardrails.
"""

from src.evaluation.failure_modes import (
    FailureConditionAudit,
    FailureSeverity,
    OperationalFailureGuardrails,
)
from src.evaluation.phase4_validator import (
    IMD_RAINFALL_TIERS,
    Phase4ModelValidator,
    ProductionReadinessGateResult,
)
from src.evaluation.regime_metrics import (
    evaluate_regime_predictions,
    format_regime_metrics_table,
)

__all__ = [
    "Phase4ModelValidator",
    "ProductionReadinessGateResult",
    "IMD_RAINFALL_TIERS",
    "OperationalFailureGuardrails",
    "FailureConditionAudit",
    "FailureSeverity",
    "evaluate_regime_predictions",
    "format_regime_metrics_table",
]
