"""
FastAPI Request and Response Pydantic Schemas for RAIN-REPAIR X (PART 8).
Strictly adheres to CONTRACT-API-003, RFC 7807 Problem Details, and PART 8 specifications.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# RFC 7807 Problem Details for Error Responses
# ---------------------------------------------------------------------------
class ProblemDetails(BaseModel):
    status: int = Field(..., description="HTTP status code")
    error_code: str = Field(..., description="Application specific error code")
    message: str = Field(..., description="Human-readable error description")
    timestamp: str = Field(..., description="ISO 8601 UTC timestamp")


# ---------------------------------------------------------------------------
# Section 2.1: Health & Metadata
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "down"] = "healthy"
    version: str = "1.0.0"
    environment: str = "development"
    latest_available_cycle: str = "2026-07-15"
    boundary_dataset_version: str = "IMD-LGD-2026.1"


# ---------------------------------------------------------------------------
# Section 2.2: Latest Cycle Forecast Summary
# ---------------------------------------------------------------------------
class GridSummary(BaseModel):
    min_corrected_mm: float = Field(..., description="Minimum corrected rainfall across domain")
    max_corrected_mm: float = Field(..., description="Maximum corrected rainfall across domain")
    mean_corrected_mm: float = Field(..., description="Mean corrected rainfall across domain")
    heavy_rain_points_count: int = Field(..., description="Count of grid points with heavy rain risk")


class ForecastLatestResponse(BaseModel):
    cycle_date: str = Field(..., description="Cycle run date YYYY-MM-DD")
    lead_time_hours: int = Field(..., description="Lead time in hours (24, 48, 72, 96, 120)")
    valid_time_utc: str = Field(..., description="Forecast valid timestamp ISO 8601 UTC")
    active_synoptic_regime: str = Field(..., description="Dominant synoptic regime")
    regime_confidence: float = Field(..., description="Confidence score [0, 1]")
    grid_summary: GridSummary


# ---------------------------------------------------------------------------
# Section 2.3 & PART 8: Grid Point & Area Forecast
# ---------------------------------------------------------------------------
class QueryCoords(BaseModel):
    lat: float
    lon: float


class NearestGridCoords(BaseModel):
    lat: float
    lon: float


class GridTimeSeriesEntry(BaseModel):
    lead_time: int
    valid_date: str
    raw_nwp_mm: float
    corrected_p10_mm: float
    corrected_p50_mm: float
    corrected_p75_mm: Optional[float] = None
    corrected_p90_mm: float
    corrected_p95_mm: Optional[float] = None
    delta_mm: float
    prob_heavy_rain: float
    prob_very_heavy_rain: float
    prob_extreme_rain: Optional[float] = None
    uncertainty_indicator: Optional[float] = None
    active_regime: str
    quality_flag: Optional[str] = "VALID"


class GridPointForecastResponse(BaseModel):
    query_coords: QueryCoords
    nearest_grid_coords: NearestGridCoords
    cycle_date: str
    time_series: List[GridTimeSeriesEntry]


# ---------------------------------------------------------------------------
# PART 8: Hotspot Information
# ---------------------------------------------------------------------------
class HotspotInfo(BaseModel):
    hotspot_grid_id: str
    hotspot_latitude: float
    hotspot_longitude: float
    hotspot_value_mm: float
    hotspot_heavy_prob: float
    hotspot_metric: str


# ---------------------------------------------------------------------------
# PART 8: Machine-Readable Explanation Object
# ---------------------------------------------------------------------------
class ExplanationDriver(BaseModel):
    category: str
    feature: str
    contribution_mm: float
    description: str


class DistrictExplanation(BaseModel):
    district_id: str
    district_name: str
    dominant_regime: str
    regime_confidence: float
    recent_error_signal: str
    forecast_spread: float
    heavy_rain_risk: float
    very_heavy_rain_risk: float
    extreme_risk: float
    main_drivers: List[ExplanationDriver]
    user_friendly_summary: str


# ---------------------------------------------------------------------------
# PART 8 Section 19: District Detailed Forecast Object
# ---------------------------------------------------------------------------
class DistrictForecastDetail(BaseModel):
    district_id: str
    district_name: str
    state: str
    forecast_time: str
    valid_time: str
    lead_time: int

    raw_nwp_rainfall: float
    corrected_rainfall: float
    area_weighted_rainfall: Optional[float] = None
    max_rainfall: float
    p50_rainfall: float
    p75_rainfall: float
    p90_rainfall: float

    difference: float
    relative_change_pct: float

    heavy_probability: float
    district_max_heavy_probability: Optional[float] = None
    very_heavy_probability: float
    district_max_very_heavy_probability: Optional[float] = None
    extreme_probability: float
    district_max_extreme_probability: Optional[float] = None

    heavy_risk_level: str
    very_heavy_risk_level: str
    extreme_risk_level: str

    dominant_regime: str
    regime_probabilities: Dict[str, float]

    forecast_spread: float
    hotspot: HotspotInfo
    explanation: Optional[DistrictExplanation] = None

    status: str
    coverage_percentage: float
    missing_grid_count: int
    total_grid_count: int

    model_version: str
    boundary_dataset_version: str


# ---------------------------------------------------------------------------
# Section 2.4: District Administrative Forecast (GeoJSON Layer)
# ---------------------------------------------------------------------------
class DistrictProperties(BaseModel):
    district_id: str
    district_name: str
    state_name: str
    lead_time: int
    mean_rainfall_mm: float
    max_rainfall_mm: float
    p90_rainfall_mm: Optional[float] = None
    prob_heavy_rain: float
    prob_very_heavy_rain: float
    prob_extreme_rain: Optional[float] = None
    warning_level: Literal["GREEN", "YELLOW", "ORANGE", "RED"]
    color_code: Optional[str] = None
    action_required: Optional[str] = None
    active_regime: str
    uncertainty_range_mm: Optional[float] = None
    forecast_spread: Optional[float] = None
    hotspot: Optional[HotspotInfo] = None
    status: Optional[str] = "complete"
    coverage_percentage: Optional[float] = 100.0
    grid_points_count: Optional[int] = None
    boundary_version: Optional[str] = None


class GeoJSONGeometry(BaseModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: Any


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    id: str
    properties: DistrictProperties
    geometry: GeoJSONGeometry


class DistrictGeoJSONResponse(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: List[GeoJSONFeature]


# ---------------------------------------------------------------------------
# PART 8 Section 29: Map Layer Contract
# ---------------------------------------------------------------------------
class MapLayerMetadataSchema(BaseModel):
    layer_id: str
    layer_name: str
    metric: str
    unit: str
    min_value: float
    max_value: float
    timestamp: str
    model_version: str
    description: str


class MapLayersResponse(BaseModel):
    timestamp: str
    layers: List[MapLayerMetadataSchema]


# ---------------------------------------------------------------------------
# PART 8 Section 31: Raw NWP vs Corrected Comparison
# ---------------------------------------------------------------------------
class ComparisonItem(BaseModel):
    target_id: str
    target_name: str
    raw_nwp_rainfall: float
    corrected_rainfall: float
    difference: float
    relative_change_pct: float
    heavy_probability_raw: Optional[float] = None
    heavy_probability_corrected: float
    forecast_spread: float
    active_regime: str


class ForecastCompareResponse(BaseModel):
    level: Literal["district", "grid"]
    lead_time: int
    cycle_date: str
    comparisons: List[ComparisonItem]


# ---------------------------------------------------------------------------
# Metadata: Models & Regimes
# ---------------------------------------------------------------------------
class MetadataModelsResponse(BaseModel):
    model_version: str
    boundary_dataset_version: str
    feature_version: str
    calibration_version: str
    evt_version: str
    supported_lead_times: List[int]
    crs_geographic: str
    crs_projected: str


class RegimeMetadataItem(BaseModel):
    regime_code: str
    regime_name: str
    description: str
    key_synoptic_indicators: List[str]


class MetadataRegimesResponse(BaseModel):
    total_regimes: int
    regimes: List[RegimeMetadataItem]


# ---------------------------------------------------------------------------
# Section 2.5: Verification & Accuracy Metrics
# ---------------------------------------------------------------------------
class VerificationMetrics(BaseModel):
    raw_nwp_rmse: float
    corrected_rmse: float
    rmse_improvement_pct: float
    raw_nwp_mae: float
    corrected_mae: float
    heavy_rain_csi_raw: float
    heavy_rain_csi_corrected: float
    crps_raw: float
    crps_corrected: float


class VerificationSummaryResponse(BaseModel):
    evaluation_period: str
    metrics: VerificationMetrics


# ---------------------------------------------------------------------------
# Section 2.6: Model Transparency & Explainability
# ---------------------------------------------------------------------------
class ContributingFeature(BaseModel):
    feature: str
    mean_abs_shap: float
    description: str


class ExplainabilitySummaryResponse(BaseModel):
    cycle_date: str
    top_contributing_features: List[ContributingFeature]


# ---------------------------------------------------------------------------
# PART 10: MLOps & Model Lifecycle Schemas
# ---------------------------------------------------------------------------
class ModelRegistryResponse(BaseModel):
    active_production_model: Optional[Dict[str, Any]]
    total_models: int
    models: List[Dict[str, Any]]


class ExperimentsListResponse(BaseModel):
    total_experiments: int
    experiments: List[Dict[str, Any]]
    baseline_comparison: List[Dict[str, Any]]
    ablation_summary: List[Dict[str, Any]]


class DriftReportResponse(BaseModel):
    timestamp: str
    overall_state: str
    feature_drift_summary: Dict[str, Dict[str, Any]]
    performance_drift_summary: Dict[str, Any]
    error_memory_health: Dict[str, Any]
    analog_memory_health: Dict[str, Any]
    action_required: bool
    recommended_action: str


class PromoteRequest(BaseModel):
    model_id: str
    target_status: Literal["validated", "staging", "production", "rejected", "archived"] = "production"


class PromoteResponse(BaseModel):
    status: str
    message: str
    quality_gate_result: Optional[Dict[str, Any]] = None


class RollbackResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# PHASE 5: Production Inference Pipeline & Model Serving Schemas
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """Production single forecast prediction request."""
    forecast_time: str = Field(..., description="Forecast valid time in ISO 8601 UTC (e.g. '2026-07-15T03:00:00Z')")
    initialization_time: str = Field(..., description="Cycle run initialization time in ISO 8601 UTC (e.g. '2026-07-15T00:00:00Z')")
    latitude: float = Field(..., description="Latitude coordinate in decimal degrees WGS84 [6.0, 38.5]")
    longitude: float = Field(..., description="Longitude coordinate in decimal degrees WGS84 [68.0, 98.0]")
    lead_time: int = Field(..., description="Forecast lead time in hours (>= 0)")
    nwp_precip: float = Field(..., description="Raw NWP accumulated precipitation in mm/24h (>= 0)")

    # Atmospheric & Physical Context Variables
    t2m: Optional[float] = Field(None, description="2m temperature (Kelvin or Celsius)")
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

    model_id: Optional[str] = Field(None, description="Optional target model ID (defaults to active production model)")
    require_full_history: bool = Field(False, description="Whether to reject predictions when historical error memory is incomplete")


class PredictionResponse(BaseModel):
    """Stable production prediction response."""
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
    created_at: str = Field(..., description="Prediction creation timestamp UTC")


class BatchPredictionRequest(BaseModel):
    """Batch prediction request payload."""
    records: List[Union[PredictionRequest, Dict[str, Any]]] = Field(..., description="List of individual forecast state inputs")
    model_id: Optional[str] = Field(None, description="Optional target model ID override")
    require_full_history: bool = Field(False, description="Whether to reject predictions when historical error memory is incomplete")


class BatchPredictionResponseSchema(BaseModel):
    """Batch prediction response schema."""
    total_records: int
    successful_count: int
    failed_count: int
    predictions: List[PredictionResponse]
    failed_records: List[Dict[str, Any]]
    model_id: str
    model_version: str
    total_latency_ms: float


# ---------------------------------------------------------------------------
# PHASE 7: MLOps Orchestration, Verification & Monitoring Schemas
# ---------------------------------------------------------------------------
class PipelineTriggerRequest(BaseModel):
    cycle_date: str = Field(..., description="Forecast cycle initialization date YYYY-MM-DD")
    lead_time_hours: int = Field(24, description="Forecast lead time (24, 48, 72, 96, 120)")
    force_rerun: bool = Field(False, description="Whether to bypass idempotency cache and generate versioned rerun")
    source: str = Field("IMD_GFS_OPERATIONAL", description="NWP source model identifier")


class AlertResponseSchema(BaseModel):
    alert_id: str
    timestamp: str
    severity: str
    category: str
    message: str
    pipeline_run_id: Optional[str] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    recommended_action: str
    resolved: bool = False


class ComprehensiveHealthResponse(BaseModel):
    status: str
    overall_status: str
    version: str
    timestamp: str
    checks_total: int
    checks_passed: int
    checks_failed: int
    components: Dict[str, Any]


# ---------------------------------------------------------------------------
# PHASE 8: Canonical Forecast Product Data Models & API Schemas
# ---------------------------------------------------------------------------
ForecastProductStatus = Literal["VALID", "PARTIAL", "STALE", "INVALID", "UNAVAILABLE"]


class DistrictIdentity(BaseModel):
    id: str = Field(..., description="Unique district administrative code (e.g. 'MH_PUNE')")
    name: str = Field(..., description="Official district name")
    state: Optional[str] = Field(None, description="State or Union Territory name")


class RainfallQuantiles(BaseModel):
    raw_nwp: float = Field(..., description="Uncorrected raw NWP rainfall in mm")
    p50: float = Field(..., description="Corrected expected / central estimate median P50 in mm")
    p75: float = Field(..., description="Corrected upper uncertainty estimate P75 in mm")
    p90: float = Field(..., description="Corrected high uncertainty / peak hotspot estimate P90 in mm")
    spread: Optional[float] = Field(None, description="Probabilistic spread (P90 - P50) in mm")
    difference: Optional[float] = Field(None, description="Post-processing delta (P50 - raw NWP) in mm")


class RegimeState(BaseModel):
    dominant: str = Field(..., description="Dominant synoptic meteorological regime identifier")
    dominant_probability: float = Field(1.0, description="Calibrated confidence of dominant regime [0.0, 1.0]")
    probabilities: Dict[str, float] = Field(..., description="Probability vector over all 6 canonical regimes summing to ~1.0")


class RiskAssessment(BaseModel):
    heavy_rainfall_probability: float = Field(..., description="Exceedance probability for heavy rainfall (>= 64.5 mm/24h)")
    extreme_rainfall_probability: float = Field(..., description="Exceedance probability for extreme rainfall (>= 204.5 mm/24h)")
    warning_level: Optional[str] = Field(None, description="IMD alert tier: GREEN, YELLOW, ORANGE, RED")


class ForecastMetadata(BaseModel):
    model_version: str = Field(..., description="Active production quantile regression model version")
    regime_model_version: str = Field(..., description="Active regime classification model version")
    feature_version: str = Field(..., description="Atmospheric feature engineering pipeline version")
    dataset_version: str = Field(..., description="Training and reference dataset lineage version")
    boundary_version: str = Field(..., description="Administrative boundary dataset version (e.g. IMD-LGD-2026.1)")
    pipeline_run_id: Optional[str] = Field(None, description="Orchestration pipeline execution run ID")


class PaginationMeta(BaseModel):
    total_count: int = Field(..., description="Total matching records across whole query domain")
    limit: int = Field(..., description="Page size limit applied")
    offset: int = Field(..., description="Offset index applied")
    has_more: bool = Field(..., description="Whether additional records exist beyond current page")


class CanonicalGridForecastRecord(BaseModel):
    """Canonical grid-level forecast product record conforming to Phase 8 Section 3 & 4."""
    prediction_id: str = Field(..., description="Unique deterministic prediction record identifier")
    forecast_time: str = Field(..., description="Valid forecast timestamp ISO 8601 UTC")
    initialization_time: str = Field(..., description="Cycle run initialization timestamp ISO 8601 UTC")
    lead_time: int = Field(..., description="Lead time in hours (24, 48, 72, 96, 120)")
    latitude: float = Field(..., description="Grid cell latitude coordinate in WGS84")
    longitude: float = Field(..., description="Grid cell longitude coordinate in WGS84")
    grid_id: Optional[str] = Field(None, description="Spatial grid point identifier")
    district_id: Optional[str] = Field(None, description="Associated district ID if mapped")
    district_name: Optional[str] = Field(None, description="Associated district name if mapped")

    raw_nwp_rainfall: float = Field(..., description="Input raw NWP rainfall in mm")
    corrected_p50: float = Field(..., description="Corrected median precipitation P50 in mm")
    corrected_p75: float = Field(..., description="Corrected upper quartile precipitation P75 in mm")
    corrected_p90: float = Field(..., description="Corrected extreme risk threshold precipitation P90 in mm")
    spread_p90_p50: float = Field(..., description="Probabilistic spread / uncertainty indicator (P90 - P50) in mm")

    regime_probabilities: Dict[str, float] = Field(..., description="Regime probability distribution across canonical regimes")
    dominant_regime: str = Field(..., description="Dominant synoptic regime classification")
    dominant_probability: float = Field(..., description="Probability of dominant regime")

    heavy_rainfall_probability: float = Field(..., description="Heavy rainfall exceedance probability")
    extreme_rainfall_probability: float = Field(..., description="Extreme rainfall exceedance probability")

    model_version: str = Field(..., description="Model version")
    regime_model_version: str = Field(..., description="Regime model version")
    feature_version: str = Field(..., description="Feature pipeline version")
    dataset_version: str = Field(..., description="Dataset lineage version")
    boundary_version: str = Field(..., description="Boundary version (IMD-LGD-2026.1)")
    pipeline_run_id: Optional[str] = Field(None, description="Pipeline run ID")

    prediction_status: ForecastProductStatus = Field("VALID", description="Product verification status")
    created_at: str = Field(..., description="Record generation timestamp ISO 8601 UTC")

    # Structured nested sections conforming to Phase 8 Section 15
    rainfall: Optional[RainfallQuantiles] = None
    regime: Optional[RegimeState] = None
    risk: Optional[RiskAssessment] = None
    metadata: Optional[ForecastMetadata] = None


class GridForecastProductResponse(BaseModel):
    """Product response payload for GET /api/v1/forecasts/grid."""
    pagination: PaginationMeta
    forecast_time: str
    lead_time: int
    boundary_version: str
    model_version: str
    records: List[CanonicalGridForecastRecord]


class CanonicalDistrictForecastRecord(BaseModel):
    """Canonical district-level forecast product record conforming to Phase 8 Section 3, 5, 6, 15."""
    district_id: str = Field(..., description="District administrative code (e.g. 'MH_PUNE')")
    district_name: str = Field(..., description="District official name")
    state: Optional[str] = Field(None, description="State or Union Territory name")
    forecast_time: str = Field(..., description="Forecast valid timestamp ISO 8601 UTC")
    initialization_time: str = Field(..., description="Forecast initialization timestamp ISO 8601 UTC")
    lead_time: int = Field(..., description="Lead time in hours")

    raw_nwp_rainfall: float = Field(..., description="District spatial mean raw NWP rainfall in mm")
    corrected_p50: float = Field(..., description="District area-weighted median rainfall P50 in mm")
    corrected_p75: float = Field(..., description="District area-weighted upper quartile P75 in mm")
    corrected_p90: float = Field(..., description="District peak hotspot upper quantile P90 in mm")
    spread_p90_p50: float = Field(..., description="Probabilistic spread / uncertainty indicator (P90 - P50) in mm")

    heavy_rainfall_probability: float = Field(..., description="District area-weighted heavy rainfall probability")
    extreme_rainfall_probability: float = Field(..., description="District area-weighted extreme rainfall probability")

    dominant_regime: str = Field(..., description="Dominant synoptic regime over district")
    regime_probabilities: Dict[str, float] = Field(..., description="District area-weighted normalized regime probabilities")

    model_version: str = Field(..., description="Model version")
    regime_model_version: str = Field(..., description="Regime model version")
    feature_version: str = Field(..., description="Feature pipeline version")
    dataset_version: str = Field(..., description="Dataset lineage version")
    boundary_version: str = Field(..., description="Boundary version (IMD-LGD-2026.1)")
    pipeline_run_id: Optional[str] = Field(None, description="Pipeline run ID")

    prediction_status: ForecastProductStatus = Field("VALID", description="Product verification status")
    created_at: str = Field(..., description="Record creation timestamp ISO 8601 UTC")

    # Structured nested representations conforming to Phase 8 Section 15
    district: Optional[DistrictIdentity] = None
    rainfall: Optional[RainfallQuantiles] = None
    regime: Optional[RegimeState] = None
    risk: Optional[RiskAssessment] = None
    recent_error_memory: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    metadata: Optional[ForecastMetadata] = None
    status: Optional[str] = None


class DistrictForecastProductResponse(BaseModel):
    """Product response payload for GET /api/v1/forecasts/districts."""
    pagination: PaginationMeta
    forecast_time: str
    lead_time: int
    boundary_version: str
    model_version: str
    records: List[CanonicalDistrictForecastRecord]



