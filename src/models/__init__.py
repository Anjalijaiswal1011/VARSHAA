"""
Machine Learning & AI Modeling Module for RAIN-REPAIR X (VARSHAA).
Trains, evaluates, and applies regime-conditioned quantile post-processing models.
"""

from src.models.explainability import (
    FEATURE_METEOROLOGICAL_DESCRIPTIONS,
    ModelTransparencyEngine,
)
from src.models.raapx_corrector import (
    DEFAULT_QUANTILES,
    QUANTILE_NAMES,
    QuantileEvaluationMetrics,
    RAAPXQuantileCorrector,
)
from src.models.raapx_pipeline import (
    RAAPXTrainingPipeline,
    generate_synthetic_multicyle_data,
)

__all__ = [
    # Phase 3 Core
    "RAAPXQuantileCorrector",
    "QuantileEvaluationMetrics",
    "DEFAULT_QUANTILES",
    "QUANTILE_NAMES",
    "RAAPXTrainingPipeline",
    "generate_synthetic_multicyle_data",
    # Transparency
    "ModelTransparencyEngine",
    "FEATURE_METEOROLOGICAL_DESCRIPTIONS",
]
