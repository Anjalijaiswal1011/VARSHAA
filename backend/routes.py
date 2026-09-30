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
    BatchPredictionRequest,
    BatchPredictionResponseSchema,
    CanonicalDistrictForecastRecord,
    CanonicalGridForecastRecord,
    DistrictForecastDetail,
    DistrictForecastProductResponse,
    DistrictGeoJSONResponse,
    DriftReportResponse,
    ExperimentsListResponse,
    ExplainabilitySummaryResponse,
    ForecastCompareResponse,
    ForecastLatestResponse,
    GridForecastProductResponse,
    GridPointForecastResponse,
    HealthResponse,
    MapLayersResponse,
    MetadataModelsResponse,
    MetadataRegimesResponse,
    ModelRegistryResponse,
    PredictionRequest,
    PredictionResponse,
    ProblemDetails,
    PromoteRequest,
    PromoteResponse,
    RollbackResponse,
    VerificationSummaryResponse,
)
from backend.services import BOUNDARY_DATASET_VERSION, ForecastService
from src.models.inference_pipeline import (
    FeatureSchemaMismatchError,
    InferenceError,
    InputValidationError,
    InsufficientHistoryError,
    ModelIntegrityError,
    ModelNotFoundError,
)
from backend.validation import (
    validate_coordinates,
    validate_district_id,
    validate_iso_date,
    validate_lead_time,
)
from src.mlops.alerts import AlertManager
from src.mlops.data_quality import DataQualityMonitor
from src.mlops.drift import DriftMonitor
from src.mlops.experiments import ExperimentTracker
from src.mlops.health import HealthProbeService
from src.mlops.pipeline import OperationalPipeline
from src.mlops.registry import ModelRegistry
from src.mlops.retraining import RetrainingPipeline
from src.mlops.scheduler import OperationalScheduler
from backend.schemas import (
    AlertResponseSchema,
    ComprehensiveHealthResponse,
    PipelineTriggerRequest,
)
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
# Section 2.5: Continuous Spatial Rainfall Field (GIS Canvas / Raster)
# ---------------------------------------------------------------------------
@router.get(
    "/forecast/rainfall-grid",
    summary="Continuous Gridded Spatial Rainfall Field",
    description="Returns high-resolution gridded AI post-processed rainfall predictions across India for GIS map canvas and raster rendering.",
)
def get_spatial_rainfall_grid(
    lead_time: int = Query(24, description="Forecast lead time in hours (24, 48, 72, 96, 120)"),
) -> Dict[str, Any]:
    if lead_time not in [24, 48, 72, 96, 120]:
        raise HTTPException(status_code=400, detail=f"Unsupported lead time {lead_time}")
    service = get_service()
    return service.get_spatial_rainfall_grid(lead_time=lead_time)


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


# ---------------------------------------------------------------------------
# PHASE 5: Production Inference Pipeline & Model Serving Endpoints
# ---------------------------------------------------------------------------
@router.post(
    "/predictions",
    response_model=PredictionResponse,
    responses={
        200: {"description": "Calibrated probabilistic rainfall prediction."},
        400: {"model": ProblemDetails, "description": "Invalid input or domain bounds violation."},
        422: {"model": ProblemDetails, "description": "Insufficient required historical data."},
        503: {"model": ProblemDetails, "description": "Target model artifact unavailable."},
        500: {"model": ProblemDetails, "description": "Inference pipeline execution error."},
    },
    summary="Production Rainfall Quantile Inference (RAAP-X)",
    description=(
        "Executes the full calibrated inference pipeline: "
        "Input Validation -> Normalization -> Feature Generation -> Zero-Leakage Error Memory -> "
        "Regime Classification -> RAAP-X Quantile Prediction (P50/P75/P90) -> "
        "Monotonicity / Non-Negativity Verification -> Validated Prediction Record."
    ),
)
def predict_rainfall(
    payload: PredictionRequest,
) -> Any:
    service = get_service()
    try:
        req_dict = payload.model_dump(exclude_unset=True)
        pred_record = service.run_prediction(
            request_dict=req_dict,
            model_id=payload.model_id,
            require_full_history=payload.require_full_history,
        )
        return pred_record.model_dump()
    except InputValidationError as exc:
        return create_rfc7807_error(400, exc.error_code, exc.message)
    except InsufficientHistoryError as exc:
        return create_rfc7807_error(422, exc.error_code, exc.message)
    except ModelNotFoundError as exc:
        return create_rfc7807_error(503, exc.error_code, exc.message)
    except ModelIntegrityError as exc:
        return create_rfc7807_error(500, exc.error_code, exc.message)
    except FeatureSchemaMismatchError as exc:
        return create_rfc7807_error(500, exc.error_code, exc.message)
    except Exception as exc:
        logger.error("Unhandled inference error: %s", exc)
        return create_rfc7807_error(500, "INFERENCE_FAILURE", "An internal error occurred during prediction.")


@router.post(
    "/predictions/batch",
    response_model=BatchPredictionResponseSchema,
    responses={
        200: {"description": "Batch predictions completed with isolated record results."},
        400: {"model": ProblemDetails, "description": "Batch request schema invalid."},
        503: {"model": ProblemDetails, "description": "Target model artifact unavailable."},
        500: {"model": ProblemDetails, "description": "Batch processing error."},
    },
    summary="Batch Production Rainfall Inference",
    description="Processes multiple weather states in batch with fault isolation across records.",
)
def predict_rainfall_batch(
    payload: BatchPredictionRequest,
) -> Any:
    service = get_service()
    try:
        batch_dict = payload.model_dump(exclude_unset=True)
        batch_resp = service.run_batch_prediction(batch_dict=batch_dict)
        return batch_resp.model_dump()
    except ModelNotFoundError as exc:
        return create_rfc7807_error(503, exc.error_code, exc.message)
    except Exception as exc:
        logger.error("Unhandled batch inference error: %s", exc)
        return create_rfc7807_error(500, "BATCH_INFERENCE_FAILURE", "An internal error occurred during batch processing.")


# ---------------------------------------------------------------------------
# PHASE 7: MLOps Orchestration, Verification & Monitoring Endpoints
# ---------------------------------------------------------------------------
@router.get(
    "/health/detailed",
    response_model=ComprehensiveHealthResponse,
    summary="9-Subsystem Diagnostic Health Probes",
    description="Probes data source, model registry, checkpoints, feature pipeline, regime model, quantile models, GIS boundaries, storage, and database.",
)
def get_detailed_health() -> ComprehensiveHealthResponse:
    probe_service = HealthProbeService()
    report = probe_service.run_comprehensive_health_check()
    return ComprehensiveHealthResponse(
        status="healthy" if report.overall_status == "HEALTHY" else report.overall_status.lower(),
        overall_status=report.overall_status,
        version=report.version,
        timestamp=report.timestamp,
        checks_total=report.checks_total,
        checks_passed=report.checks_passed,
        checks_failed=report.checks_failed,
        components={k: v.to_dict() for k, v in report.components.items()},
    )


@router.post(
    "/mlops/pipeline/trigger",
    summary="Trigger Operational 11-Stage ML Pipeline",
    description="Executes idempotent operational pipeline: Ingestion -> QC -> Alignment -> Features -> Regime -> Quantiles -> Calibration -> Grid -> District -> Validation -> Publish -> Monitor.",
)
def trigger_operational_pipeline(
    payload: PipelineTriggerRequest,
) -> Dict[str, Any]:
    pipeline = OperationalPipeline.get_default_pipeline()
    try:
        job = pipeline.execute_cycle(
            cycle_date=payload.cycle_date,
            lead_time_hours=payload.lead_time_hours,
            source=payload.source,
            force_rerun=payload.force_rerun,
        )
        return job.to_dict()
    except Exception as e:
        logger.error("Operational pipeline trigger error: %s", e)
        return create_rfc7807_error(500, "PIPELINE_EXECUTION_FAILURE", str(e))


@router.get(
    "/mlops/runs",
    summary="List Operational Pipeline Execution Runs",
    description="Returns chronological audit log of all operational pipeline runs, statuses, and performance indicators.",
)
def list_pipeline_runs(
    limit: int = Query(50, description="Maximum number of historical runs to retrieve"),
) -> List[Dict[str, Any]]:
    pipeline = OperationalPipeline.get_default_pipeline()
    runs = pipeline.list_jobs(limit=limit)
    return [r.to_dict() for r in runs]


@router.get(
    "/mlops/runs/{job_id}",
    summary="Get Pipeline Run Audit Trail & Lineage",
    description="Returns full machine-readable stage execution records, inputs, outputs, verification report, and data lineage for a specific run.",
)
def get_pipeline_run(
    job_id: str,
) -> Any:
    pipeline = OperationalPipeline.get_default_pipeline()
    job = pipeline.get_job(job_id)
    if not job:
        return create_rfc7807_error(404, "RUN_NOT_FOUND", f"Pipeline run '{job_id}' not found in registry.")
    return job.to_dict()


@router.get(
    "/mlops/alerts",
    response_model=List[AlertResponseSchema],
    summary="Query Operational Telemetry Alerts",
    description="Lists active and historical operational alerts filterable by severity, category, and resolution status.",
)
def list_operational_alerts(
    severity: Optional[str] = Query(None, description="Filter by severity: INFO, WARNING, CRITICAL"),
    category: Optional[str] = Query(None, description="Filter by category: DATA_MISSING, FEATURE_DRIFT, etc."),
    resolved: Optional[bool] = Query(None, description="Filter by resolution status"),
    limit: int = Query(100, description="Max alerts to return"),
) -> List[AlertResponseSchema]:
    manager = AlertManager.get_default_manager()
    alerts = manager.list_alerts(severity=severity, category=category, resolved=resolved, limit=limit)
    return [
        AlertResponseSchema(
            alert_id=a.alert_id,
            timestamp=a.timestamp,
            severity=a.severity,
            category=a.category,
            message=a.message,
            pipeline_run_id=a.pipeline_run_id,
            dataset_version=a.dataset_version,
            model_version=a.model_version,
            recommended_action=a.recommended_action,
            resolved=a.resolved,
        )
        for a in alerts
    ]


@router.get(
    "/mlops/models",
    summary="Model Registry Lifecycle Inventory",
    description="Lists all registered models with their formal lifecycle states (TRAINED, VALIDATED, CANDIDATE, PRODUCTION, RETIRED, FAILED).",
)
def list_registry_models(
    status_filter: Optional[str] = Query(None, description="Filter by lifecycle status"),
) -> List[Dict[str, Any]]:
    registry = ModelRegistry.get_default_registry()
    models = registry.list_models(status_filter=status_filter)
    return [m.to_dict() for m in models]


@router.get(
    "/mlops/data-quality",
    summary="Latest Data Quality Audit Report",
    description="Returns the most recent 4-dimensional data quality audit report (completeness, validity, timeliness, spatial).",
)
def get_latest_data_quality() -> Any:
    monitor = DataQualityMonitor()
    report = monitor.get_latest_report()
    if not report:
        return {"status": "no_reports_available", "message": "No data quality reports recorded yet."}
    return report.to_dict()


@router.get(
    "/mlops/verification/{job_id}",
    summary="Get Automated Verification Report for Pipeline Run",
    description="Returns the detailed automated verification report and publication gating status for a run.",
)
def get_verification_report(
    job_id: str,
) -> Any:
    pipeline = OperationalPipeline.get_default_pipeline()
    job = pipeline.get_job(job_id)
    if not job:
        return create_rfc7807_error(404, "RUN_NOT_FOUND", f"Pipeline run '{job_id}' not found.")
    if not job.validation_report:
        return create_rfc7807_error(404, "VERIFICATION_NOT_FOUND", f"Run '{job_id}' has no verification report.")
    return job.validation_report


@router.get(
    "/mlops/scheduler/status",
    summary="Operational Pipeline Scheduler Telemetry",
    description="Returns scheduler status, interval, next scheduled run, and total completed runs.",
)
def get_scheduler_status() -> Dict[str, Any]:
    scheduler = OperationalScheduler.get_default_scheduler()
    return scheduler.get_status().to_dict()


# ---------------------------------------------------------------------------
# PHASE 8: Canonical Forecast Product Endpoints (Sections 4, 5, 6)
# ---------------------------------------------------------------------------
@router.get(
    "/forecasts/grid",
    response_model=GridForecastProductResponse,
    responses={
        200: {"description": "Validated grid forecast product records returned successfully."},
        400: {"model": ProblemDetails, "description": "Invalid query parameters or coordinates."},
        422: {"model": ProblemDetails, "description": "Request validation error."},
    },
    summary="Canonical Grid Forecast Product API",
    description="Returns validated grid predictions filtered by lead time, bounding box, spatial proximity, or grid ID with pagination and uncertainty preservation.",
)
def get_product_grid_forecasts(
    lead_time: int = Query(24, description="Forecast lead time in hours (24, 48, 72, 96, 120)"),
    forecast_time: Optional[str] = Query(None, description="Optional forecast initialization/cycle date (YYYY-MM-DD)"),
    min_lat: Optional[float] = Query(None, description="South bounding box latitude in WGS84 [6.0, 38.5]"),
    max_lat: Optional[float] = Query(None, description="North bounding box latitude in WGS84 [6.0, 38.5]"),
    min_lon: Optional[float] = Query(None, description="West bounding box longitude in WGS84 [68.0, 98.0]"),
    max_lon: Optional[float] = Query(None, description="East bounding box longitude in WGS84 [68.0, 98.0]"),
    latitude: Optional[float] = Query(None, description="Query point latitude coordinate in WGS84"),
    longitude: Optional[float] = Query(None, description="Query point longitude coordinate in WGS84"),
    radius_km: Optional[float] = Query(None, description="Search radius in kilometers around query point"),
    grid_id: Optional[str] = Query(None, description="Specific grid point identifier (e.g. 'G_18.50_73.75')"),
    limit: int = Query(50, ge=1, le=1000, description="Pagination page limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    service = get_service()
    success, result = service.get_product_grid_forecasts(
        lead_time=lead_time,
        forecast_time=forecast_time,
        min_lat=min_lat,
        max_lat=max_lat,
        min_lon=min_lon,
        max_lon=max_lon,
        latitude=latitude,
        longitude=longitude,
        radius_km=radius_km,
        grid_id=grid_id,
        limit=limit,
        offset=offset,
    )
    if not success:
        return create_rfc7807_error(
            status_code=400,
            error_code="INVALID_FORECAST_PARAMETERS",
            message=str(result),
        )
    return result


@router.get(
    "/forecasts/districts",
    response_model=DistrictForecastProductResponse,
    responses={
        200: {"description": "District administrative forecast product records returned successfully."},
        400: {"model": ProblemDetails, "description": "Invalid query parameters or unsupported lead time."},
    },
    summary="Canonical District Forecast Product API",
    description="Returns validated district-level forecasts aggregated from high-resolution grids using IMD-LGD-2026.1 boundaries with uncertainty preservation.",
)
def get_product_district_forecasts(
    lead_time: int = Query(24, description="Forecast lead time in hours (24, 48, 72, 96, 120)"),
    forecast_time: Optional[str] = Query(None, description="Optional forecast initialization/cycle date (YYYY-MM-DD)"),
    district_id: Optional[str] = Query(None, description="Optional district administrative identifier (e.g. 'MH_PUNE')"),
    state: Optional[str] = Query(None, description="Optional state or union territory name filter"),
    limit: int = Query(50, ge=1, le=1000, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    service = get_service()
    success, result = service.get_product_district_forecasts(
        lead_time=lead_time,
        forecast_time=forecast_time,
        district_id=district_id,
        state=state,
        limit=limit,
        offset=offset,
    )
    if not success:
        return create_rfc7807_error(
            status_code=400,
            error_code="INVALID_FORECAST_PARAMETERS",
            message=str(result),
        )
    return result


@router.get(
    "/forecasts/districts/{district_id}",
    response_model=CanonicalDistrictForecastRecord,
    responses={
        200: {"description": "Single district forecast product returned successfully."},
        400: {"model": ProblemDetails, "description": "Invalid parameters."},
        404: {"model": ProblemDetails, "description": "District not found or forecast unavailable."},
    },
    summary="Single District Forecast Product API",
    description="Returns the latest valid forecast product for a specific district, including explanation metadata, uncertainty quantiles, and full version lineage.",
)
def get_product_single_district_forecast(
    district_id: str,
    lead_time: int = Query(24, description="Forecast lead time in hours (24, 48, 72, 96, 120)"),
    forecast_time: Optional[str] = Query(None, description="Optional forecast cycle date (YYYY-MM-DD)"),
):
    service = get_service()
    success, error_code, result = service.get_product_single_district_forecast(
        district_id=district_id,
        lead_time=lead_time,
        forecast_time=forecast_time,
    )
    if not success:
        status_code = 404 if error_code in ["DISTRICT_NOT_FOUND", "FORECAST_UNAVAILABLE"] else 400
        return create_rfc7807_error(
            status_code=status_code,
            error_code=error_code,
            message=str(result),
        )
    return result



