"""
FastAPI REST API Routes for RAIN-REPAIR X (PART 8).
Standard Base URL: /api/v1
Conforms strictly to CONTRACT-API-003, RFC 7807 Problem Details, and PART 8 specifications.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import JSONResponse

from backend.auth import (
    DEFAULT_USERS,
    TokenResponse,
    UserCredentials,
    UserPayload,
    create_access_token,
    get_current_user,
    require_role,
    verify_password,
)
from backend.cache import cache_manager
from backend.schemas import (
    DistrictForecastDetail,
    DistrictGeoJSONResponse,
    DriftReportResponse,
    ExperimentsListResponse,
    ExplainabilitySummaryResponse,
    ForecastCompareResponse,
    ForecastLatestResponse,
    GridPointForecastResponse,
    HealthResponse,
    MapLayersResponse,
    MetadataModelsResponse,
    MetadataRegimesResponse,
    ModelRegistryResponse,
    ProblemDetails,
    PromoteRequest,
    PromoteResponse,
    RollbackResponse,
    VerificationSummaryResponse,
)
from backend.services import BOUNDARY_DATASET_VERSION, ForecastService
from backend.validation import (
    validate_coordinates,
    validate_district_id,
    validate_iso_date,
    validate_lead_time,
)
from src.mlops.drift import DriftMonitor
from src.mlops.experiments import ExperimentTracker
from src.mlops.registry import ModelRegistry
from src.mlops.retraining import RetrainingPipeline
from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.routes")

router = APIRouter(prefix="/api/v1", tags=["Operational Rainfall Intelligence"])

# Shared service singleton
_service: Optional[ForecastService] = None


def get_service() -> ForecastService:
    global _service
    if _service is None:
        _service = ForecastService()
    return _service


def create_rfc7807_error(status_code: int, error_code: str, message: str) -> JSONResponse:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    content = {
        "status": status_code,
        "error_code": error_code,
        "message": message,
        "timestamp": now_utc,
    }
    return JSONResponse(status_code=status_code, content=content)


# ---------------------------------------------------------------------------
# Tier 2: Authentication & User Profile Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/auth/token",
    response_model=TokenResponse,
    summary="User Authentication & JWT Issuance",
    description="Authenticates credentials and returns a signed bearer JWT access token with role claims.",
)
def login_for_access_token(credentials: UserCredentials) -> TokenResponse:
    user = DEFAULT_USERS.get(credentials.username)
    if not user or not verify_password(credentials.password, user["password_hash"], user["salt"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated.",
        )

    token = create_access_token(user_id=user["user_id"], username=user["username"], role=user["role"])
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in_seconds=7200,
        user_id=user["user_id"],
        username=user["username"],
        role=user["role"],
    )


@router.get(
    "/auth/me",
    response_model=UserPayload,
    summary="Current User Profile & Permissions",
    description="Returns current authenticated user identity and assigned RBAC role.",
)
def get_current_user_profile(user: UserPayload = Depends(get_current_user)) -> UserPayload:
    return user


# ---------------------------------------------------------------------------
# Tier 6: Cache Management Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/cache/stats",
    summary="Operational Cache Telemetry",
    description="Returns cache metrics including hit ratio, active entries, and total hits/misses.",
)
def get_cache_statistics() -> Dict[str, Any]:
    return cache_manager.get_stats()


@router.post(
    "/cache/clear",
    summary="Purge Operational Cache (Admin Only)",
    description="Clears all cached forecast outputs. Requires administrative privileges.",
)
def clear_cache(admin_user: UserPayload = Depends(require_role("admin"))) -> Dict[str, Any]:
    cleared_count = cache_manager.clear()
    return {
        "status": "success",
        "message": f"Cleared {cleared_count} cached entries.",
        "cleared_by": admin_user.username,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


# ---------------------------------------------------------------------------
# Section 2.1: Health & System Status
# ---------------------------------------------------------------------------
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health & Readiness Probe",
    description="Returns server status, software version, environment, boundary version, and latest cycle timestamp.",
)
def get_health() -> HealthResponse:
    service = get_service()
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment="development",
        latest_available_cycle=service.latest_cycle_date,
        boundary_dataset_version=BOUNDARY_DATASET_VERSION,
    )


# ---------------------------------------------------------------------------
# Section 2.2: Latest Cycle Forecast Summary
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/latest",
    response_model=ForecastLatestResponse,
    summary="Latest Forecast Cycle Summary",
    description="Returns synoptic regime state, domain extrema, and heavy rainfall point count for a specified lead time.",
)
def get_latest_forecast(
    lead_time: int = Query(24, description="Forecast lead time in hours (24, 48, 72, 96, 120)"),
) -> ForecastLatestResponse:
    if lead_time not in [24, 48, 72, 96, 120]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid lead time {lead_time}. Supported lead times: [24, 48, 72, 96, 120].",
        )
    service = get_service()
    data = service.get_latest_forecast_summary(lead_time=lead_time)
    return ForecastLatestResponse(**data)


# ---------------------------------------------------------------------------
# Section 2.3: Grid Point & Area Forecast
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/grid",
    response_model=GridPointForecastResponse,
    responses={
        200: {"description": "Point forecast meteogram returned successfully."},
        400: {"model": ProblemDetails, "description": "Coordinates outside India meteorological domain."},
    },
    summary="Grid Point Forecast (Meteogram Plume Query)",
    description="Returns 5-day lead-time quantile plumes and exceedance probabilities for the nearest grid point to requested coordinates.",
)
def get_grid_point_forecast(
    lat: float = Query(..., description="Latitude coordinate (WGS84, EPSG:4326)"),
    lon: float = Query(..., description="Longitude coordinate (WGS84, EPSG:4326)"),
    cycle_date: Optional[str] = Query(None, description="Optional cycle date (YYYY-MM-DD)"),
):
    service = get_service()
    success, result = service.get_grid_point_forecast(lat=lat, lon=lon, cycle_date=cycle_date)
    if not success:
        return create_rfc7807_error(
            status_code=400,
            error_code="INVALID_COORDINATES",
            message=str(result),
        )
    return result


# ---------------------------------------------------------------------------
# Section 2.4: District Administrative Forecast (GeoJSON Layer)
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/districts",
    response_model=DistrictGeoJSONResponse,
    summary="District Administrative Forecast (GeoJSON Layer)",
    description="Returns standard GeoJSON FeatureCollection of districts with IMD warning levels and risk attributes for interactive map rendering.",
)
def get_district_forecast(
    lead_time: int = Query(24, description="Lead time in hours (24, 48, 72, 96, 120)"),
    state: Optional[str] = Query(None, description="Optional state name filter (e.g. 'Maharashtra')"),
):
    if lead_time not in [24, 48, 72, 96, 120]:
        return create_rfc7807_error(
            status_code=400,
            error_code="INVALID_LEAD_TIME",
            message=f"Unsupported lead time {lead_time}. Supported lead times: [24, 48, 72, 96, 120].",
        )
    service = get_service()
    geojson_data = service.get_district_alerts_geojson(lead_time=lead_time, state=state)
    return JSONResponse(
        content=geojson_data,
        media_type="application/geo+json",
    )


# ---------------------------------------------------------------------------
# PART 8 Section 19: District Detailed Forecast Object
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/district/{district_id}",
    response_model=DistrictForecastDetail,
    responses={
        200: {"description": "Detailed district forecast object."},
        404: {"model": ProblemDetails, "description": "District ID not found."},
    },
    summary="Single District Forecast Detail",
    description="Returns detailed meteorological risk statistics, hotspot extraction, and evidence-grounded explanation for a specific district.",
)
def get_single_district_forecast(
    district_id: str,
    lead_time: int = Query(24, description="Lead time in hours (24, 48, 72, 96, 120)"),
):
    service = get_service()
    success, data = service.get_district_forecast_detail(district_id=district_id, lead_time=lead_time)
    if not success:
        return create_rfc7807_error(
            status_code=404,
            error_code="DISTRICT_NOT_FOUND",
            message=str(data),
        )
    return data


# ---------------------------------------------------------------------------
# PART 8 Section 28 & 29: Standardized Map Layer Metadata Contract
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/map",
    response_model=MapLayersResponse,
    summary="Standardized Map Layer Catalog",
    description="Returns metadata catalog of all supported choropleth and risk map layers for frontend GIS rendering.",
)
def get_map_layers(
    timestamp: Optional[str] = Query(None, description="Optional ISO timestamp"),
) -> MapLayersResponse:
    service = get_service()
    data = service.get_map_layers_catalog(timestamp=timestamp)
    return MapLayersResponse(**data)


# ---------------------------------------------------------------------------
# PART 8 Section 31: Raw NWP vs Corrected Comparison
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/compare",
    response_model=ForecastCompareResponse,
    summary="Raw NWP vs RAIN-REPAIR X Comparative Evaluation",
    description="Returns side-by-side comparison data between raw NWP and AI post-processed forecasts at district or grid level.",
)
def get_forecast_comparison(
    level: Literal["district", "grid"] = Query("district", description="Aggregation level: 'district' or 'grid'"),
    lead_time: int = Query(24, description="Lead time in hours (24, 48, 72, 96, 120)"),
) -> ForecastCompareResponse:
    if lead_time not in [24, 48, 72, 96, 120]:
        raise HTTPException(status_code=400, detail=f"Unsupported lead time {lead_time}")
    service = get_service()
    data = service.get_forecast_comparison(level=level, lead_time=lead_time)
    return ForecastCompareResponse(**data)


# ---------------------------------------------------------------------------
# PART 8 Section 13-15: Evidence-Grounded District Explanation
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/explanation/{district_id}",
    summary="District Forecast Explanation & Driver Attribution",
    description="Returns machine-readable drivers, synoptic regime influences, and a user-friendly summary grounded in actual model evidence.",
)
def get_district_explanation(
    district_id: str,
    lead_time: int = Query(24, description="Lead time in hours (24, 48, 72, 96, 120)"),
):
    service = get_service()
    success, data = service.get_district_explanation(district_id=district_id, lead_time=lead_time)
    if not success:
        return create_rfc7807_error(
            status_code=404,
            error_code="DISTRICT_NOT_FOUND",
            message=str(data),
        )
    return data


# ---------------------------------------------------------------------------
# District Risk Ranking Table
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/table",
    summary="District Risk Ranking Table",
    description="Returns tabular rows of districts sorted by IMD alert severity and peak rainfall for tabular dashboard display.",
)
def get_district_table(
    lead_time: int = Query(24, description="Lead time in hours (24, 48, 72, 96, 120)"),
    state: Optional[str] = Query(None, description="Optional state filter"),
) -> List[Dict[str, Any]]:
    service = get_service()
    table_df = service.get_district_summary_table(lead_time=lead_time, state=state)
    return table_df.to_dict(orient="records")


# ---------------------------------------------------------------------------
# PART 8 Section 20 & 26: Metadata Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/metadata/models",
    response_model=MetadataModelsResponse,
    summary="Model Architecture, Calibration & Versioning Metadata",
    description="Returns system versions, active components, and CRS standards for full operational auditability.",
)
def get_metadata_models() -> MetadataModelsResponse:
    service = get_service()
    data = service.get_metadata_models()
    return MetadataModelsResponse(**data)


@router.get(
    "/metadata/regimes",
    response_model=MetadataRegimesResponse,
    summary="Meteorological Synoptic Weather Regime Catalog",
    description="Returns descriptions and key synoptic indicators for all 6 canonical monsoon weather regimes.",
)
def get_metadata_regimes() -> MetadataRegimesResponse:
    service = get_service()
    data = service.get_metadata_regimes()
    return MetadataRegimesResponse(**data)


# ---------------------------------------------------------------------------
# Section 2.5: Verification & Accuracy Metrics
# ---------------------------------------------------------------------------
@router.get(
    "/verification/summary",
    response_model=VerificationSummaryResponse,
    summary="Verification & Accuracy Benchmark Metrics",
    description="Returns operational verification scorecards comparing Raw NWP against AI-repaired forecasts (RMSE, MAE, CSI, CRPS).",
)
def get_verification_summary(
    season: str = Query("monsoon_2026", description="Monsoon evaluation season tag"),
) -> VerificationSummaryResponse:
    service = get_service()
    data = service.get_verification_summary(season=season)
    return VerificationSummaryResponse(**data)


# ---------------------------------------------------------------------------
# Section 2.6: Model Transparency & Explainability
# ---------------------------------------------------------------------------
@router.get(
    "/explainability/summary",
    response_model=ExplainabilitySummaryResponse,
    summary="Model Transparency & Global Attribution Summary",
    description="Returns top contributing features and mean absolute SHAP attributions for the active cycle.",
)
def get_explainability_summary(
    cycle_date: Optional[str] = Query(None, description="Optional cycle date"),
) -> ExplainabilitySummaryResponse:
    service = get_service()
    data = service.get_explainability_summary(cycle_date=cycle_date)
    return ExplainabilitySummaryResponse(**data)


# ---------------------------------------------------------------------------
# PART 10: MLOps & Model Lifecycle Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/mlops/registry",
    response_model=ModelRegistryResponse,
    summary="Model Registry Catalog & Production State",
    description="Returns all registered candidate, validated, staging, production, and archived models.",
)
def get_model_registry() -> ModelRegistryResponse:
    registry = ModelRegistry.get_default_registry()
    models = [m.to_dict() for m in registry.list_models()]
    prod = registry.get_production_model()
    return ModelRegistryResponse(
        active_production_model=prod.to_dict() if prod else None,
        total_models=len(models),
        models=models,
    )


@router.get(
    "/mlops/experiments",
    response_model=ExperimentsListResponse,
    summary="Experiment Tracking & Baselines A-F",
    description="Returns tracked experiments, baseline comparisons, and ablation contributions with time-aware partitions.",
)
def get_mlops_experiments() -> ExperimentsListResponse:
    tracker = ExperimentTracker.get_default_tracker()
    experiments = [e.to_dict() for e in tracker.list_experiments()]
    baselines = tracker.get_baseline_comparison()
    ablations = tracker.get_ablation_summary()
    return ExperimentsListResponse(
        total_experiments=len(experiments),
        experiments=experiments,
        baseline_comparison=baselines,
        ablation_summary=ablations,
    )


@router.get(
    "/mlops/drift/status",
    response_model=DriftReportResponse,
    summary="Data Drift & Concept Drift Health Monitor",
    description="Returns PSI/KS feature drift, performance drift, error-memory status, and analog memory health.",
)
def get_drift_status() -> DriftReportResponse:
    monitor = DriftMonitor.get_default_monitor()
    report = monitor.generate_drift_report()
    return DriftReportResponse(
        timestamp=report.timestamp,
        overall_state=report.overall_state,
        feature_drift_summary=report.feature_drift_summary,
        performance_drift_summary=report.performance_drift_summary,
        error_memory_health=report.error_memory_health,
        analog_memory_health=report.analog_memory_health,
        action_required=report.action_required,
        recommended_action=report.recommended_action,
    )


@router.post(
    "/mlops/promote",
    response_model=PromoteResponse,
    summary="Promote Model Through Quality Gates (Admin Only)",
    description="Executes rigorous quality gates (RMSE, calibration, heavy rain CSI, regime sufficiency) before promoting to target status.",
)
def promote_model(
    payload: PromoteRequest,
    current_user: UserPayload = Depends(require_role("admin")),
) -> PromoteResponse:
    registry = ModelRegistry.get_default_registry()
    success, msg, qg_result = registry.promote_model(
        model_id=payload.model_id,
        target_status=payload.target_status,
        promoted_by=current_user.username,
    )
    if not success:
        return JSONResponse(
            status_code=400,
            content={
                "status": "failed",
                "message": msg,
                "quality_gate_result": qg_result.to_dict() if qg_result else None,
            },
        )
    return PromoteResponse(
        status="success",
        message=msg,
        quality_gate_result=qg_result.to_dict() if qg_result else None,
    )


@router.post(
    "/mlops/rollback",
    response_model=RollbackResponse,
    summary="Rollback Production Model (Admin Only)",
    description="Safely rolls back active production deployment to previously validated production model artifact without retraining.",
)
def rollback_production_model(
    reason: str = Query("Operational rollback triggered by operator", description="Reason for rollback"),
    current_user: UserPayload = Depends(require_role("admin")),
) -> RollbackResponse:
    registry = ModelRegistry.get_default_registry()
    success, msg = registry.rollback_to_previous_production(
        reason=reason,
        triggered_by=current_user.username,
    )
    if not success:
        raise HTTPException(status_code=400, detail=msg)
    return RollbackResponse(status="success", message=msg)


@router.post(
    "/mlops/retrain/evaluate",
    summary="Controlled Retraining Pipeline Trigger & Evaluation (Forecaster/Admin)",
    description="Evaluates retraining criteria (drift, degradation, scheduled review), builds candidate, validates quality gates.",
)
def trigger_retraining_evaluation(
    force: bool = Query(False, description="Force retraining even if drift triggers are not tripped"),
    current_user: UserPayload = Depends(require_role("forecaster")),
) -> Dict[str, Any]:
    pipeline = RetrainingPipeline.get_default_pipeline()
    result = pipeline.execute_retraining_pipeline(
        force=force,
        triggered_by=current_user.username,
    )
    return result
