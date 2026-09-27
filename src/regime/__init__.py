"""
Weather Regime Intelligence Layer for RAIN-REPAIR X.
Provides:
- Six canonical weather regimes + uncertain state
- Soft regime probability vectors
- Baseline heuristic and linear classifiers
- Primary LightGBM gradient boosted regime classifier
- Validation-fitted probability calibration
- Chronological regime transition engine
- Unified inference facade with defensive fallback
"""

from src.regime.baseline import (
    MultinomialLogisticRegimeBaseline,
    RuleBasedRegimeClassifier,
)
from src.regime.calibration import (
    RegimeCalibrator,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    evaluate_probability_calibration,
)
from src.regime.classifier import LightGBMRegimeClassifier
from src.regime.evaluation import (
    evaluate_regime_predictions,
    format_regime_metrics_table,
)
from src.regime.inference import RegimeInferenceEngine
from src.regime.labels import (
    derive_provisional_regime_labels,
    get_regime_distribution_report,
)
from src.regime.schemas import (
    IDX_TO_REGIME,
    REGIME_CLASSES,
    REGIME_TO_IDX,
    RegimeProbabilityVector,
    RegimeTransitionOutput,
    TransitionStatus,
    WeatherRegime,
)
from src.regime.transition import (
    RegimeHistoryBuffer,
    RegimeTransitionEngine,
    compute_normalized_entropy,
    compute_total_variation_distance,
)

__all__ = [
    # Schemas
    "WeatherRegime",
    "REGIME_CLASSES",
    "REGIME_TO_IDX",
    "IDX_TO_REGIME",
    "TransitionStatus",
    "RegimeProbabilityVector",
    "RegimeTransitionOutput",
    # Labels
    "derive_provisional_regime_labels",
    "get_regime_distribution_report",
    # Baselines
    "RuleBasedRegimeClassifier",
    "MultinomialLogisticRegimeBaseline",
    # Primary Model
    "LightGBMRegimeClassifier",
    # Calibration
    "RegimeCalibrator",
    "compute_multiclass_brier_score",
    "compute_expected_calibration_error",
    "evaluate_probability_calibration",
    # Transition Engine
    "RegimeTransitionEngine",
    "RegimeHistoryBuffer",
    "compute_total_variation_distance",
    "compute_normalized_entropy",
    # Evaluation
    "evaluate_regime_predictions",
    "format_regime_metrics_table",
    # High-level Inference
    "RegimeInferenceEngine",
]
