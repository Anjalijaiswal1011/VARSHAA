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
