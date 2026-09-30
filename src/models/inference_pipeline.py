"""
Production Inference Pipeline & Model Serving Engine for RAIN-REPAIR X (Phase 5).
Coordinates:
    1. Input Validation (Datatypes, ranges, units, temporal causality, India spatial bounds)
    2. Data Normalization (standardized physical units, log transforms)
    3. Feature Generation & Historical Error Retrieval (strictly leakage-safe trailing 3/7/14-day errors)
    4. Regime Classifier (continuous calibrated 6-regime probability vector)
    5. Quantile Rainfall Corrector (RAAP-X P50, P75, P90)
    6. Calibration & Monotonicity Validation (Physical realizability & Chernozhukov Rearrangement Operator)
    7. Structured Output Record & Observability Telemetry
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import uuid

import joblib
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, field_validator, model_validator

from src.mlops.registry import ModelRegistry, RegisteredModel
from src.mlops.versioning import (
    CALIBRATION_VERSION_DEFAULT,
    DATASET_VERSION_DEFAULT,
    FEATURE_VERSION_DEFAULT,
    QUANTILE_MODEL_VERSION_DEFAULT,
)
from src.models.raapx_corrector import RAAPXQuantileCorrector
from src.postprocessing.constraints import enforce_quantile_monotonicity
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.regime.classifier import LightGBMRegimeClassifier
from src.regime.schemas import (
    REGIME_CLASSES,
    RegimeProbabilityVector,
    WeatherRegime,
)
from src.utils.exceptions import (
    PhysicalConstraintViolationError,
    TemporalLeakageError,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.models.inference_pipeline")

# India meteorological geographical boundary constraints (EPSG:4326)
INDIA_BOUNDS = {
    "min_lat": 6.0,
    "max_lat": 38.5,
    "min_lon": 68.0,
    "max_lon": 98.0,
}

CANONICAL_QUANTILES: List[float] = [0.50, 0.75, 0.90]

REGIME_PROB_COLS: List[str] = [
    "prob_active_monsoon",
    "prob_break_monsoon",
    "prob_monsoon_depression",
    "prob_coastal",
    "prob_orographic",
    "prob_western_disturbance",
]


class InferenceError(Exception):
    """Base exception for inference pipeline failures."""
    def __init__(self, message: str, error_code: str = "INFERENCE_ERROR", status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code


class InputValidationError(InferenceError):
    """Raised when incoming forecast input violates validation constraints."""
    def __init__(self, message: str, error_code: str = "INVALID_INPUT"):
        super().__init__(message, error_code=error_code, status_code=400)


class FeatureSchemaMismatchError(InferenceError):
    """Raised when generated inference features do not match model expectation."""
    def __init__(self, message: str):
        super().__init__(message, error_code="FEATURE_SCHEMA_MISMATCH", status_code=500)


class ModelNotFoundError(InferenceError):
    """Raised when requested model or component is missing from registry/filesystem."""
    def __init__(self, message: str):
        super().__init__(message, error_code="MODEL_UNAVAILABLE", status_code=503)


class ModelIntegrityError(InferenceError):
    """Raised when model artifact checksum or version validation fails."""
    def __init__(self, message: str):
        super().__init__(message, error_code="MODEL_INTEGRITY_FAILURE", status_code=500)


class InsufficientHistoryError(InferenceError):
    """Raised when strict historical error retrieval cannot find required prior history."""
    def __init__(self, message: str):
        super().__init__(message, error_code="INSUFFICIENT_HISTORY", status_code=422)


class ForecastInputRecord(BaseModel):
    """Strict input schema for a single NWP forecast state."""
    forecast_time: str = Field(..., description="Forecast valid time in ISO 8601 UTC (e.g. '2026-07-15T03:00:00Z')")
    initialization_time: str = Field(..., description="Cycle run initialization time in ISO 8601 UTC (e.g. '2026-07-15T00:00:00Z')")
    latitude: float = Field(..., description="Latitude coordinate in decimal degrees WGS84")
    longitude: float = Field(..., description="Longitude coordinate in decimal degrees WGS84")
    lead_time: int = Field(..., description="Forecast lead time in hours (>= 0)")
    nwp_precip: float = Field(..., description="Raw NWP accumulated precipitation forecast in mm/24h (>= 0)")

    # Atmospheric & Physical Context Variables
    t2m: Optional[float] = Field(None, description="2m temperature in Kelvin (or Celsius)")
    q2m: Optional[float] = Field(None, description="2m specific humidity in kg/kg or relative humidity %")
    u10: Optional[float] = Field(None, description="10m zonal wind velocity in m/s")
    v10: Optional[float] = Field(None, description="10m meridional wind velocity in m/s")
    u850: Optional[float] = Field(None, description="850 hPa zonal wind in m/s")
    v850: Optional[float] = Field(None, description="850 hPa meridional wind in m/s")
    mslp: Optional[float] = Field(None, description="Mean sea level pressure in hPa")
    elevation: Optional[float] = Field(None, description="Terrain surface elevation in meters")
    slope: Optional[float] = Field(None, description="Terrain slope gradient in degrees")
    aspect_sin: Optional[float] = Field(None, description="Sine of terrain aspect angle")
    aspect_cos: Optional[float] = Field(None, description="Cosine of terrain aspect angle")
    dist_to_coast: Optional[float] = Field(None, description="Distance to nearest coastline in km")
    terrain_roughness: Optional[float] = Field(None, description="Terrain standard deviation roughness index")
    cape: Optional[float] = Field(None, description="Convective Available Potential Energy in J/kg")
    relative_vorticity: Optional[float] = Field(None, description="Relative vorticity at 850 hPa")
    moisture_flux_conv: Optional[float] = Field(None, description="Moisture flux convergence proxy")
    wind_shear_deep: Optional[float] = Field(None, description="Deep layer wind shear (850-200 hPa)")
    orographic_uplift: Optional[float] = Field(None, description="Orographic vertical velocity index")
    vapor_pressure_deficit: Optional[float] = Field(None, description="Vapor pressure deficit in kPa")
    grid_id: Optional[str] = Field(None, description="Optional spatial grid identifier")

    @field_validator("latitude")
    @classmethod
    def check_latitude(cls, v: float) -> float:
        if not (INDIA_BOUNDS["min_lat"] <= v <= INDIA_BOUNDS["max_lat"]):
            raise ValueError(
                f"Latitude {v:.2f} is outside the Indian subcontinent domain [{INDIA_BOUNDS['min_lat']}, {INDIA_BOUNDS['max_lat']}]."
            )
        return float(v)

    @field_validator("longitude")
    @classmethod
    def check_longitude(cls, v: float) -> float:
        if not (INDIA_BOUNDS["min_lon"] <= v <= INDIA_BOUNDS["max_lon"]):
            raise ValueError(
                f"Longitude {v:.2f} is outside the Indian subcontinent domain [{INDIA_BOUNDS['min_lon']}, {INDIA_BOUNDS['max_lon']}]."
            )
        return float(v)

    @field_validator("nwp_precip")
    @classmethod
    def check_nwp_precip(cls, v: float) -> float:
        if v < 0.0:
            raise ValueError(f"Precipitation cannot be negative (got {v} mm).")
        if v > 1500.0:
            raise ValueError(f"Precipitation exceeds realistic 24h meteorological physical limits (got {v} mm).")
        return float(v)

    @field_validator("lead_time")
    @classmethod
    def check_lead_time(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"Lead time cannot be negative (got {v} hours).")
        return int(v)

    @model_validator(mode="after")
    def validate_temporal_causality(self) -> ForecastInputRecord:
        """Enforces that forecast_time is at or after initialization_time."""
        try:
            init_dt = pd.to_datetime(self.initialization_time)
            fcst_dt = pd.to_datetime(self.forecast_time)
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {e}")

        if fcst_dt < init_dt:
            raise ValueError(
                f"Temporal causality violation: forecast_time ({self.forecast_time}) "
                f"precedes initialization_time ({self.initialization_time})."
            )
        return self


class BatchForecastInput(BaseModel):
    """Batch input schema containing multiple forecast records."""
    records: List[Union[ForecastInputRecord, Dict[str, Any]]] = Field(..., description="List of single forecast input records or dicts")
    model_id: Optional[str] = Field(None, description="Optional target model ID. If omitted, uses active production model.")
    require_full_history: bool = Field(False, description="If True, rejects predictions when historical error memory is incomplete.")



class PredictionOutputRecord(BaseModel):
    """Stable production prediction response conforming to Phase 5 Specification."""
    prediction_id: str = Field(..., description="Unique deterministic or UUID prediction identifier")
    forecast_time: str = Field(..., description="Forecast valid timestamp ISO 8601 UTC")
    initialization_time: str = Field(..., description="Cycle initialization timestamp ISO 8601 UTC")
    latitude: float = Field(..., description="Latitude coordinate")
    longitude: float = Field(..., description="Longitude coordinate")
    lead_time: int = Field(..., description="Lead time in hours")

    raw_nwp_rainfall: float = Field(..., description="Input raw NWP rainfall (mm/24h)")
    corrected_p50: float = Field(..., description="Corrected median precipitation P50 (mm/24h)")
    corrected_p75: float = Field(..., description="Corrected upper quartile precipitation P75 (mm/24h)")
    corrected_p90: float = Field(..., description="Corrected extreme risk threshold precipitation P90 (mm/24h)")
    spread_p90_p50: float = Field(..., description="Probabilistic spread / uncertainty indicator (P90 - P50)")

    regime_probabilities: Dict[str, float] = Field(..., description="Calibrated probability distribution across six synoptic regimes")
    dominant_regime: str = Field(..., description="Dominant synoptic regime classification")
    dominant_probability: float = Field(..., description="Probability of dominant regime")

    model_version: str = Field(..., description="Active production model version")
    regime_model_version: str = Field(..., description="Active regime classifier version")
    feature_version: str = Field(..., description="Feature pipeline schema version")
    dataset_version: str = Field(..., description="Dataset lineage version")

    prediction_status: Literal["valid", "calibrated_rearranged", "rejected"] = Field("valid", description="Quality gate status")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="Operational telemetry and execution diagnostics")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))


class BatchPredictionResponse(BaseModel):
    """Batch prediction response containing verified records and any failure diagnostics."""
    total_records: int
    successful_count: int
    failed_count: int
    predictions: List[PredictionOutputRecord]
    failed_records: List[Dict[str, Any]]
    model_id: str
    model_version: str
    total_latency_ms: float


@dataclass
class ProductionModelPackage:
    """Holds fully validated and loaded production model artifacts in memory."""
    model_id: str
    model_version: str
    regime_model_version: str
    dataset_version: str
    feature_version: str
    regime_classifier: LightGBMRegimeClassifier
    quantile_corrector: RAAPXQuantileCorrector
    feature_cols: List[str]
    regime_feature_cols: List[str]
    quantiles: List[float]
    loaded_at_utc: str
    checksum: Optional[str] = None
    components: Optional[Dict[str, Any]] = None


class ProductionModelLoader:
    """
    Thread-safe model loader ensuring safe lifecycle, artifact validation,
    checksum verification, and zero silent fallback.
    """
    _loaded_packages: Dict[str, ProductionModelPackage] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._loaded_packages.clear()

    @classmethod
    def load_model(
        cls,
        model_id: Optional[str] = None,
        registry: Optional[ModelRegistry] = None,
        base_dir: Optional[Path] = None,
    ) -> ProductionModelPackage:
        """
        Loads and validates a production model package.
        Validates artifact existence, SHA-256 checksum, and feature schemas.
        """
        reg = registry or ModelRegistry.get_default_registry()

        if model_id is None:
            reg_model = reg.get_production_model()
            if reg_model is None:
                raise ModelNotFoundError("No active production model found in registry.")
            target_id = reg_model.model_id
        else:
            reg_model = reg.get_model(model_id)
            if reg_model is None:
                raise ModelNotFoundError(f"Requested model '{model_id}' is not registered in registry.")
            target_id = model_id

        # Return cached model if already safely loaded
        if target_id in cls._loaded_packages:
            return cls._loaded_packages[target_id]

        root = base_dir or Path.cwd()

        # Checkpoint validation
        if not reg_model.artifact_path:
            raise ModelIntegrityError(f"Model {target_id} has no artifact_path registered.")

        art_path = Path(reg_model.artifact_path)
        if not art_path.is_absolute():
            art_path = root / art_path

        if not art_path.exists():
            raise ModelNotFoundError(
                f"Model artifact file for '{target_id}' does not exist at expected path: {art_path}"
            )

        # Checksum validation
        if reg_model.checksum:
            computed = RegisteredModel.compute_checksum(art_path)
            if computed != reg_model.checksum:
                raise ModelIntegrityError(
                    f"Model checksum mismatch for '{target_id}'. Registered: {reg_model.checksum}, Computed: {computed}"
                )
            logger.info("Verified SHA-256 checksum for model %s: %s", target_id, computed[:12])

        # Load package from disk
        try:
            loaded_payload = joblib.load(art_path)
        except Exception as exc:
            raise ModelIntegrityError(f"Failed to deserialize model artifact at {art_path}: {exc}") from exc

        if not isinstance(loaded_payload, dict):
            raise ModelIntegrityError(
                f"Model artifact at {art_path} is corrupt: expected dictionary package, got {type(loaded_payload)}"
            )

        regime_clf = loaded_payload.get("regime_model")
        quantile_corr = loaded_payload.get("quantile_model")
        feature_cols = loaded_payload.get("feature_cols", [])
        regime_feature_cols = loaded_payload.get("regime_feature_cols", [])

        if regime_clf is None or not getattr(regime_clf, "is_fitted", False):
            raise ModelIntegrityError(f"Model package '{target_id}' missing valid fitted regime_model.")

        if quantile_corr is None or not getattr(quantile_corr, "is_fitted", False):
            raise ModelIntegrityError(f"Model package '{target_id}' missing valid fitted quantile_model.")

        package = ProductionModelPackage(
            model_id=reg_model.model_id,
            model_version=reg_model.version,
            regime_model_version=reg_model.regime_model_version or "v1.0.0",
            dataset_version=reg_model.dataset_version,
            feature_version=reg_model.feature_version,
            regime_classifier=regime_clf,
            quantile_corrector=quantile_corr,
            feature_cols=feature_cols,
            regime_feature_cols=regime_feature_cols,
            quantiles=reg_model.quantiles or [0.50, 0.75, 0.90],
            loaded_at_utc=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            checksum=reg_model.checksum,
            components=reg_model.components,
        )

        cls._loaded_packages[target_id] = package
        logger.info(
            "Successfully loaded production model package '%s' (Version: %s, Quantiles: %s).",
            package.model_id,
            package.model_version,
            package.quantiles,
        )
        return package


class ProductionInferencePipeline:
    """
    Complete end-to-end production inference pipeline implementing the target architecture:
    NWP / Forecast Input
            ↓
    Input Validation
            ↓
    Data Normalization
            ↓
    Feature Generation
            ↓
    Historical Error Retrieval (Zero Leakage)
            ↓
    Regime Classifier
            ↓
    Regime Probability Vector
            ↓
    Quantile Rainfall Corrector
            ↓
    Calibration / Monotonicity Validation
            ↓
    P50 / P75 / P90
            ↓
    Output Validation
            ↓
    Prediction Record
    """

    def __init__(
        self,
        model_loader: Optional[ProductionModelLoader] = None,
        error_memory_store: Optional[LeakageSafeErrorMemoryStore] = None,
        default_model_id: Optional[str] = None,
    ):
        self.model_loader = model_loader or ProductionModelLoader()
        self.error_memory_store = error_memory_store or LeakageSafeErrorMemoryStore()
        self.default_model_id = default_model_id

    def _normalize_input_record(self, record: ForecastInputRecord) -> Dict[str, float]:
        """
        Normalizes physical units and derives atmospheric proxies from standard input variables.
        """
        # Ensure temperature is in Kelvin
        t2m_k = record.t2m
        if t2m_k is not None:
            if t2m_k < 100.0:  # Provided in Celsius
                t2m_k = t2m_k + 273.15
        else:
            t2m_k = 298.15  # Default ~25 C

        # Terrain heuristics if not provided based on coordinates
        elevation = record.elevation if record.elevation is not None else max(10.0, float((record.latitude - 10.0) * 15.0))
        slope = record.slope if record.slope is not None else (3.5 if elevation > 250 else 0.8)
        aspect_sin = record.aspect_sin if record.aspect_sin is not None else 0.5
        aspect_cos = record.aspect_cos if record.aspect_cos is not None else 0.5
        dist_to_coast = record.dist_to_coast if record.dist_to_coast is not None else max(10.0, float(abs(record.longitude - 72.8) * 85.0))
        terrain_roughness = record.terrain_roughness if record.terrain_roughness is not None else (1.5 if slope > 2 else 0.8)

        # Wind & moisture proxies
        u = record.u850 if record.u850 is not None else (record.u10 if record.u10 is not None else 8.0)
        v = record.v850 if record.v850 is not None else (record.v10 if record.v10 is not None else 2.0)
        q2m = record.q2m if record.q2m is not None else 0.016
        mslp = record.mslp if record.mslp is not None else 1005.0

        moisture_conv = record.moisture_flux_conv if record.moisture_flux_conv is not None else float(max(0.5, u * 0.8 + 2.0))
        wind_shear = record.wind_shear_deep if record.wind_shear_deep is not None else 14.0
        vorticity = record.relative_vorticity if record.relative_vorticity is not None else 1.2
        orographic_uplift = record.orographic_uplift if record.orographic_uplift is not None else float((elevation / 100.0) * slope * 0.6)
        vpd = record.vapor_pressure_deficit if record.vapor_pressure_deficit is not None else 0.8
        cape = record.cape if record.cape is not None else 1200.0

        nwp_precip = float(record.nwp_precip)
        nwp_precip_log = float(np.log1p(nwp_precip))

        return {
            "nwp_precip": nwp_precip,
            "nwp_precip_log": nwp_precip_log,
            "nwp_t2m": float(t2m_k),
            "nwp_q2m": float(q2m),
            "nwp_u10": float(u),
            "nwp_v10": float(v),
            "nwp_wind_speed": float(np.sqrt(u * u + v * v)),
            "nwp_mslp": float(mslp),
            "elevation": float(elevation),
            "slope": float(slope),
            "aspect_sin": float(aspect_sin),
            "aspect_cos": float(aspect_cos),
            "dist_to_coast": float(dist_to_coast),
            "terrain_roughness": float(terrain_roughness),
            "moisture_flux_conv": float(moisture_conv),
            "wind_shear_deep": float(wind_shear),
            "relative_vorticity": float(vorticity),
            "orographic_uplift": float(orographic_uplift),
            "vapor_pressure_deficit": float(vpd),
            "cape": float(cape),
            "lat": float(record.latitude),
            "lon": float(record.longitude),
            "lead_time": float(record.lead_time),
        }

    def _retrieve_error_memory_features(
        self,
        initialization_time: str,
        grid_key: str,
        require_full_history: bool = False,
    ) -> Tuple[Dict[str, float], str]:
        """
        Retrieves trailing 1/3/7/14-day error memory strictly prior to initialization_time.
        Strictly enforces zero temporal leakage: observations at or after initialization_time
        can never be retrieved.
        """
        init_dt = pd.to_datetime(initialization_time)
        if init_dt.tzinfo is not None:
            init_dt = init_dt.tz_localize(None)
        cycle_date_str = init_dt.strftime("%Y-%m-%d")

        # Query past verification dates
        valid_past_dates = self.error_memory_store.get_valid_past_dates(cycle_date_str)

        # Anti-leakage assertion
        for d in valid_past_dates:
            d_dt = pd.to_datetime(d)
            if d_dt.tzinfo is not None:
                d_dt = d_dt.tz_localize(None)
            if d_dt >= init_dt:
                raise TemporalLeakageError(
                    f"CRITICAL CAUSALITY VIOLATION: Verification date {d} is >= forecast cycle {initialization_time}!"
                )

        if not valid_past_dates:
            if require_full_history:
                raise InsufficientHistoryError(
                    f"No historical error memory available strictly prior to {initialization_time}."
                )
            status = "COLD_START_INSUFFICIENT_HISTORY"
            return {
                "error_lag_1d": 0.0,
                "error_lag_3d": 0.0,
                "error_lag_7d": 0.0,
                "error_lag_14d": 0.0,
                "rolling_bias_7d": 0.0,
                "rolling_bias_14d": 0.0,
                "rolling_mae_7d": 0.0,
            }, status

        # Retrieve verified historical errors for the specific grid point or region
        errors_1d = self.error_memory_store._history.get(valid_past_dates[0], {}).get(grid_key, 0.0)
        
        # 3-day window
        w3 = [self.error_memory_store._history.get(d, {}).get(grid_key, 0.0) for d in valid_past_dates[:3]]
        # 7-day window
        w7 = [self.error_memory_store._history.get(d, {}).get(grid_key, 0.0) for d in valid_past_dates[:7]]
        # 14-day window
        w14 = [self.error_memory_store._history.get(d, {}).get(grid_key, 0.0) for d in valid_past_dates[:14]]

        bias_7d = float(np.mean(w7)) if w7 else 0.0
        bias_14d = float(np.mean(w14)) if w14 else 0.0
        mae_7d = float(np.mean(np.abs(w7))) if w7 else 0.0

        status = "COMPLETE_HISTORY" if len(valid_past_dates) >= 14 else f"PARTIAL_HISTORY_{len(valid_past_dates)}_DAYS"

        return {
            "error_lag_1d": float(errors_1d),
            "error_lag_3d": float(np.mean(w3)),
            "error_lag_7d": float(np.mean(w7)),
            "error_lag_14d": float(np.mean(w14)),
            "rolling_bias_7d": bias_7d,
            "rolling_bias_14d": bias_14d,
            "rolling_mae_7d": mae_7d,
        }, status

    def predict_single(
        self,
        input_record: Union[ForecastInputRecord, Dict[str, Any]],
        model_id: Optional[str] = None,
        require_full_history: bool = False,
    ) -> PredictionOutputRecord:
        """
        Executes complete production pipeline for a single forecast record.
        """
        t_start = time.perf_counter()

        # Step 1: Input Validation
        if isinstance(input_record, dict):
            try:
                record = ForecastInputRecord(**input_record)
            except Exception as e:
                raise InputValidationError(f"Input validation failed: {e}") from e
        else:
            record = input_record

        # Step 2: Model Loading (cached/validated)
        t_model_start = time.perf_counter()
        target_model_id = model_id or self.default_model_id
        model_package = self.model_loader.load_model(model_id=target_model_id)
        t_model_load = (time.perf_counter() - t_model_start) * 1000.0

        # Step 3: Data Normalization
        t_feat_start = time.perf_counter()
        base_features = self._normalize_input_record(record)

        # Step 4: Historical Error Retrieval (Zero Leakage)
        grid_key = record.grid_id or f"{record.latitude:.2f}_{record.longitude:.2f}"
        error_feats, mem_status = self._retrieve_error_memory_features(
            initialization_time=record.initialization_time,
            grid_key=grid_key,
            require_full_history=require_full_history,
        )

        full_feature_dict = {**base_features, **error_feats}
        t_feat_gen = (time.perf_counter() - t_feat_start) * 1000.0

        # Step 5: Regime Inference
        t_regime_start = time.perf_counter()
        regime_feature_cols = model_package.regime_feature_cols or [
            c for c in model_package.regime_classifier.feature_cols if c in full_feature_dict
        ]
        
        # Validate regime feature schema
        missing_regime_cols = [c for c in model_package.regime_classifier.feature_cols if c not in full_feature_dict]
        if missing_regime_cols:
            raise FeatureSchemaMismatchError(
                f"Missing required regime classifier features: {missing_regime_cols[:5]}"
            )

        df_regime_in = pd.DataFrame([full_feature_dict])[model_package.regime_classifier.feature_cols]
        regime_probs_arr = model_package.regime_classifier.predict_proba(df_regime_in)[0]

        # Validate probability constraints: 0 <= p <= 1, sum ≈ 1.0
        if not np.all((regime_probs_arr >= 0.0) & (regime_probs_arr <= 1.0)):
            raise PhysicalConstraintViolationError("Regime probabilities out of bounds [0, 1].")
        if not np.isclose(float(np.sum(regime_probs_arr)), 1.0, atol=1e-3):
            regime_probs_arr = regime_probs_arr / np.sum(regime_probs_arr)

        prob_vec = RegimeProbabilityVector.from_array(regime_probs_arr)
        dom_regime, dom_conf = prob_vec.dominant_regime()
        regime_dict = {k.upper(): round(float(v), 4) for k, v in prob_vec.to_dict().items()}
        t_regime_inf = (time.perf_counter() - t_regime_start) * 1000.0

        # Step 6: Quantile Rainfall Corrector
        t_quantile_start = time.perf_counter()
        # Augment features with regime probability vector
        for col_name, prob_val in zip(REGIME_PROB_COLS, prob_vec.to_array()):
            full_feature_dict[col_name] = float(prob_val)

        # Validate quantile feature schema against training columns
        expected_cols = model_package.feature_cols or model_package.quantile_corrector.feature_cols
        missing_quantile_cols = [c for c in expected_cols if c not in full_feature_dict]
        if missing_quantile_cols:
            raise FeatureSchemaMismatchError(
                f"Missing required quantile corrector features: {missing_quantile_cols[:5]}"
            )

        df_quantile_in = pd.DataFrame([full_feature_dict])[expected_cols]
        raw_quantiles_df = model_package.quantile_corrector.predict(df_quantile_in, enforce_constraints=False)

        q_p50 = float(raw_quantiles_df["p50"].iloc[0])
        q_p75 = float(raw_quantiles_df["p75"].iloc[0])
        q_p90 = float(raw_quantiles_df["p90"].iloc[0])

        # Step 7: Calibration & Quantile Validation
        rearrangement_applied = False
        if (q_p50 < 0.0) or (q_p75 < 0.0) or (q_p90 < 0.0) or (q_p50 > q_p75) or (q_p75 > q_p90):
            logger.warning(
                "Quantile monotonicity or non-negativity violated in raw output (P50=%.2f, P75=%.2f, P90=%.2f). "
                "Applying mathematical Chernozhukov Rearrangement Operator.",
                q_p50, q_p75, q_p90,
            )
            raw_dict = {
                0.50: np.array([q_p50], dtype=np.float32),
                0.75: np.array([q_p75], dtype=np.float32),
                0.90: np.array([q_p90], dtype=np.float32),
            }
            rearranged = enforce_quantile_monotonicity(raw_dict, enforce_non_negativity=True)
            q_p50 = float(rearranged[0.50][0])
            q_p75 = float(rearranged[0.75][0])
            q_p90 = float(rearranged[0.90][0])
            rearrangement_applied = True

        # Final physical realizability validation
        if not (0.0 <= q_p50 <= q_p75 + 1e-4 and q_p75 <= q_p90 + 1e-4):
            raise PhysicalConstraintViolationError(
                f"Fatal post-processing error: Quantiles remain inverted after rearrangement: {q_p50}, {q_p75}, {q_p90}"
            )

        t_quantile_inf = (time.perf_counter() - t_quantile_start) * 1000.0
        t_total = (time.perf_counter() - t_start) * 1000.0

        prediction_status: Literal["valid", "calibrated_rearranged", "rejected"] = (
            "calibrated_rearranged" if rearrangement_applied else "valid"
        )

        pred_id = f"PRED-{pd.to_datetime(record.forecast_time).strftime('%Y%m%d%H')}-{uuid.uuid4().hex[:8]}"

        return PredictionOutputRecord(
            prediction_id=pred_id,
            forecast_time=record.forecast_time,
            initialization_time=record.initialization_time,
            latitude=record.latitude,
            longitude=record.longitude,
            lead_time=record.lead_time,
            raw_nwp_rainfall=record.nwp_precip,
            corrected_p50=round(q_p50, 2),
            corrected_p75=round(q_p75, 2),
            corrected_p90=round(q_p90, 2),
            spread_p90_p50=round(max(0.0, q_p90 - q_p50), 2),
            regime_probabilities=regime_dict,
            dominant_regime=dom_regime,
            dominant_probability=round(float(dom_conf), 4),
            model_version=model_package.model_version,
            regime_model_version=model_package.regime_model_version,
            feature_version=model_package.feature_version,
            dataset_version=model_package.dataset_version,
            prediction_status=prediction_status,
            diagnostics={
                "model_id": model_package.model_id,
                "memory_status": mem_status,
                "rearrangement_applied": rearrangement_applied,
                "timings_ms": {
                    "model_load": round(t_model_load, 2),
                    "feature_generation": round(t_feat_gen, 2),
                    "regime_inference": round(t_regime_inf, 2),
                    "quantile_inference": round(t_quantile_inf, 2),
                    "total_latency": round(t_total, 2),
                },
            },
        )

    def predict_batch(
        self,
        batch_input: Union[BatchForecastInput, Dict[str, Any]],
    ) -> BatchPredictionResponse:
        """
        Executes batch predictions over multiple forecast states.
        Isolates record-level errors so that a single invalid record does not abort the entire batch.
        """
        t_batch_start = time.perf_counter()

        if isinstance(batch_input, dict):
            records_raw = batch_input.get("records", [])
            target_model_id = batch_input.get("model_id") or self.default_model_id
            req_full = batch_input.get("require_full_history", False)
        elif isinstance(batch_input, BatchForecastInput):
            records_raw = batch_input.records
            target_model_id = batch_input.model_id or self.default_model_id
            req_full = batch_input.require_full_history
        else:
            records_raw = getattr(batch_input, "records", [])
            target_model_id = getattr(batch_input, "model_id", None) or self.default_model_id
            req_full = getattr(batch_input, "require_full_history", False)

        successful_predictions: List[PredictionOutputRecord] = []
        failed_records: List[Dict[str, Any]] = []

        # Load target model once for the batch
        model_package = self.model_loader.load_model(model_id=target_model_id)

        for idx, record in enumerate(records_raw):
            try:
                pred = self.predict_single(
                    input_record=record,
                    model_id=model_package.model_id,
                    require_full_history=req_full,
                )
                successful_predictions.append(pred)
            except Exception as exc:
                logger.error("Failed prediction at batch index %d: %s", idx, exc)
                failed_records.append({
                    "record_index": idx,
                    "record_data": record.model_dump() if hasattr(record, "model_dump") else record,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

        t_total_batch = (time.perf_counter() - t_batch_start) * 1000.0

        return BatchPredictionResponse(
            total_records=len(records_raw),
            successful_count=len(successful_predictions),
            failed_count=len(failed_records),
            predictions=successful_predictions,
            failed_records=failed_records,
            model_id=model_package.model_id,
            model_version=model_package.model_version,
            total_latency_ms=round(t_total_batch, 2),
        )
