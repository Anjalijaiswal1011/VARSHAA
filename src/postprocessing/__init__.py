"""
RAIN-REPAIR X Post-Processing Layer:
Deterministic NWP Error DNA Repair, Multi-Scale Error Memory, Historical Analog Memory,
Multi-Quantile Regression Plumes (P10/P50/P75/P90/P95), Calibrated Extreme Thresholds,
Extreme Value Theory (EVT-GPD), Physical Realizability, and IMD Risk Alert Generation.
"""

from src.postprocessing.ablation import ModelSelectionExperimentRunner
from src.postprocessing.analog import AnalogMemory, KNNAnalogMemory
from src.postprocessing.artifacts import ModelArtifactManager
from src.postprocessing.baselines import (
    GenericMLErrorCorrectionBaseline,
    RawNWPBaseline,
    StatisticalBiasCorrectionBaseline,
)
from src.postprocessing.constraints import (
    CANONICAL_QUANTILES,
    enforce_probability_monotonicity,
    enforce_quantile_monotonicity,
    validate_physical_realizability,
)
from src.postprocessing.engine import RainRepairEngine
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.evaluation import (
    compute_contingency_table_metrics,
    compute_deterministic_metrics,
    evaluate_lead_time_performance,
    evaluate_regime_wise_performance,
    evaluate_residual_correction,
    evaluate_spatial_error_summary,
)
from src.postprocessing.evt import EVTGPDTailModel
from src.postprocessing.fallback import (
    FallbackExecutionStatus,
    FallbackLevel,
    FallbackRepairController,
)
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.probabilistic_ablation import ProbabilisticAblationRunner
from src.postprocessing.probabilistic_eval import (
    compute_brier_score,
    compute_crps_from_quantiles,
    evaluate_quantile_calibration,
)
from src.postprocessing.quantiles import (
    MultiQuantileRegressor,
    compute_pinball_loss,
)
from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.postprocessing.risk import (
    CalibratedRiskEngine,
    IMDAlertLevel,
    assign_imd_alert_level,
)
from src.postprocessing.target import (
    apply_error_correction,
    compute_nwp_error_target,
    validate_target_isolation,
)
from src.postprocessing.thresholds import ExtremeThresholdClassifier

__all__ = [
    # Target & Repair
    "compute_nwp_error_target",
    "apply_error_correction",
    "validate_target_isolation",
    # Memory & DNA
    "LeakageSafeErrorMemoryStore",
    "AnalogMemory",
    "KNNAnalogMemory",
    "NWPErrorDNAExtractor",
    # Baselines
    "RawNWPBaseline",
    "StatisticalBiasCorrectionBaseline",
    "GenericMLErrorCorrectionBaseline",
    # Deterministic Repair Models & Fallback
    "SharedRegimeLightGBMRepair",
    "SoftMixtureOfExpertsRepair",
    "FallbackLevel",
    "FallbackExecutionStatus",
    "FallbackRepairController",
    "RainRepairEngine",
    # Verification & Evaluation
    "compute_deterministic_metrics",
    "compute_contingency_table_metrics",
    "evaluate_residual_correction",
    "evaluate_regime_wise_performance",
    "evaluate_lead_time_performance",
    "evaluate_spatial_error_summary",
    "ModelSelectionExperimentRunner",
    "ModelArtifactManager",
    # Part 7: Probabilistic, Quantiles & EVT
    "CANONICAL_QUANTILES",
    "enforce_quantile_monotonicity",
    "enforce_probability_monotonicity",
    "validate_physical_realizability",
    "MultiQuantileRegressor",
    "compute_pinball_loss",
    "ExtremeThresholdClassifier",
    "EVTGPDTailModel",
    "CalibratedRiskEngine",
    "IMDAlertLevel",
    "assign_imd_alert_level",
    "compute_crps_from_quantiles",
    "compute_brier_score",
    "evaluate_quantile_calibration",
    "ProbabilisticAblationRunner",
]
