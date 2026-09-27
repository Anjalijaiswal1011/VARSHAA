"""
Domain-Specific Exception Hierarchy for RAIN-REPAIR X (VARSHAA).
Enforces structured, typed error handling across all pipeline stages.
"""

from typing import Any, Dict, Optional


class RainRepairError(Exception):
    """Base exception for all RAIN-REPAIR X domain errors."""

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details,
        }


# ==============================================================================
# 1. Data Layer Exceptions
# ==============================================================================


class DataIngestionError(RainRepairError):
    """Raised when loading or reading a raw meteorological file fails."""
    pass


class InvalidCoordinateError(RainRepairError):
    """Raised when spatial coordinates, CRS, or bounding box do not match EPSG:4326 standards."""
    pass


class TemporalAlignmentError(RainRepairError):
    """Raised when observation timestamps or forecast lead times fail alignment checks."""
    pass


class TemporalLeakageError(RainRepairError):
    """Raised when future observations or post-forecast data leak into feature extraction or memory."""
    pass


class DataQualityError(RainRepairError):
    """Raised when data values exceed physical atmospheric limits or exceed NaN thresholds."""
    pass


# ==============================================================================
# 2. ML & Regime Exceptions
# ==============================================================================


class RegimeClassificationError(RainRepairError):
    """Raised when monsoon regime classifier receives invalid atmospheric inputs."""
    pass


class ModelInferenceError(RainRepairError):
    """Raised when ML model fails during prediction or receives incompatible feature tensors."""
    pass


class MissingModelCheckpointError(RainRepairError):
    """Raised when a required serialized model file cannot be found in checkpoints directory."""
    pass


# ==============================================================================
# 3. Post-Processing & Physical Realizability Exceptions
# ==============================================================================


class PhysicalConstraintViolationError(RainRepairError):
    """Raised when output forecasts violate physical laws (negative rain, quantile crossing)."""
    pass


class DistrictAggregationError(RainRepairError):
    """Raised when GIS zonal statistics or administrative polygon intersection fails."""
    pass


# ==============================================================================
# 4. Configuration & API Exceptions
# ==============================================================================


class ConfigurationError(RainRepairError):
    """Raised when invalid or missing configuration parameters are encountered."""
    pass


class APIValidationError(RainRepairError):
    """Raised when client API requests contain invalid query parameters."""
    pass
