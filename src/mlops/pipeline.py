"""
Operational ML Pipeline Orchestrator & Job Lifecycle Engine for RAAP-X (Phase 7 Sections 1, 3, 4, 6, 18).
Executes the reproducible, observable, failure-safe 11-stage pipeline:
DATA ARRIVES -> INGESTION -> QUALITY CHECK -> DATA ALIGNMENT -> FEATURE GENERATION ->
REGIME INFERENCE -> QUANTILE CORRECTION -> CALIBRATION -> GRID PREDICTION ->
DISTRICT AGGREGATION -> VALIDATION -> PUBLISH OUTPUT -> MONITOR
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.gis.alerts import IMDAlertEngine
from src.gis.districts import BOUNDARY_DATASET_VERSION, DistrictGeometryManager
from src.gis.zonal import SpatialZonalStatisticsEngine
from src.mlops.alerts import AlertManager
from src.mlops.data_quality import DataQualityMonitor, DataQualityReport
from src.mlops.drift import DriftMonitor
from src.mlops.performance_monitor import ModelPerformanceMonitor
from src.mlops.registry import ModelRegistry
from src.mlops.verification import AutomatedVerificationEngine, VerificationReport
from src.mlops.versioning import SystemVersionManifest, VersionManager
from src.models.inference_pipeline import (
    ForecastInputRecord,
    PredictionOutputRecord,
    ProductionInferencePipeline,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.pipeline")

JobStatus = Literal["QUEUED", "RUNNING", "SUCCESS", "PARTIAL_SUCCESS", "FAILED"]
StageStatus = Literal["QUEUED", "RUNNING", "SUCCESS", "FAILED", "SKIPPED"]

STAGES_ORDER = [
    "INGESTION",
    "QUALITY_CHECK",
    "DATA_ALIGNMENT",
    "FEATURE_GENERATION",
    "REGIME_INFERENCE",
    "QUANTILE_CORRECTION",
    "CALIBRATION",
    "GRID_PREDICTION",
    "DISTRICT_AGGREGATION",
    "VALIDATION",
    "PUBLISH_OUTPUT",
    "MONITOR",
]


@dataclass
class StageExecutionRecord:
    """Telemetry record for a single pipeline stage."""
    stage_name: str
    status: StageStatus = "QUEUED"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    duration_ms: float = 0.0
    input_summary: Dict[str, Any] = field(default_factory=dict)
    output_summary: Dict[str, Any] = field(default_factory=dict)
    version_metadata: Dict[str, str] = field(default_factory=dict)
    error_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineJobRecord:
    """Standardized Pipeline Job Model conforming to Phase 7 Section 3 & 18."""
    job_id: str
    pipeline_version: str = "1.0.0"
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    completed_at: Optional[str] = None
    status: JobStatus = "QUEUED"
    dataset_version: str = "IMD-ERA5-v2.1"
    feature_version: str = "v1.0.0-phys"
    model_versions: Dict[str, str] = field(default_factory=dict)
    boundary_version: str = BOUNDARY_DATASET_VERSION
    input_period: Dict[str, str] = field(default_factory=dict)
    output_period: Dict[str, str] = field(default_factory=dict)
    records_processed: int = 0
    records_failed: int = 0
    error_summary: Optional[str] = None
    idempotency_key: str = ""
    reused_existing: bool = False
    reused_from_job_id: Optional[str] = None
    stage_states: Dict[str, StageExecutionRecord] = field(default_factory=dict)
    quality_report: Optional[Dict[str, Any]] = None
    validation_report: Optional[Dict[str, Any]] = None
    published_output_summary: Optional[Dict[str, Any]] = None
    performance_report: Optional[Dict[str, Any]] = None
    drift_summary: Optional[Dict[str, Any]] = None
    alerts_triggered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["stage_states"] = {k: v.to_dict() if isinstance(v, StageExecutionRecord) else v for k, v in self.stage_states.items()}
        return d


class OperationalPipeline:
    """
    Master Operational Pipeline coordinating ingestion, ML inference, district aggregation,
    validation, output publishing, and monitoring with strict idempotency and audit trails.
    """

    _default_instance: Optional[OperationalPipeline] = None

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        data_quality_monitor: Optional[DataQualityMonitor] = None,
        verification_engine: Optional[AutomatedVerificationEngine] = None,
        performance_monitor: Optional[ModelPerformanceMonitor] = None,
        drift_monitor: Optional[DriftMonitor] = None,
        alert_manager: Optional[AlertManager] = None,
        geometry_manager: Optional[DistrictGeometryManager] = None,
        zonal_engine: Optional[SpatialZonalStatisticsEngine] = None,
        alert_engine: Optional[IMDAlertEngine] = None,
        inference_pipeline: Optional[ProductionInferencePipeline] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.registry = registry or ModelRegistry.get_default_registry()
        self.alert_manager = alert_manager or AlertManager.get_default_manager()
        self.data_quality_monitor = data_quality_monitor or DataQualityMonitor(alert_manager=self.alert_manager)
        self.verification_engine = verification_engine or AutomatedVerificationEngine(alert_manager=self.alert_manager)
        self.performance_monitor = performance_monitor or ModelPerformanceMonitor(alert_manager=self.alert_manager)
        self.drift_monitor = drift_monitor or DriftMonitor(alert_manager=self.alert_manager)
        self.geometry_manager = geometry_manager or DistrictGeometryManager()
        self.zonal_engine = zonal_engine or SpatialZonalStatisticsEngine(geometry_manager=self.geometry_manager)
        self.alert_engine = alert_engine or IMDAlertEngine()
        self.inference_pipeline = inference_pipeline or ProductionInferencePipeline()
        self.storage_dir = storage_dir or Path("logs/pipeline_runs")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._jobs: Dict[str, PipelineJobRecord] = {}
        self._idempotency_index: Dict[str, str] = {}
        self._load_persisted_jobs()

    @classmethod
    def get_default_pipeline(cls) -> OperationalPipeline:
        if cls._default_instance is None:
            cls._default_instance = OperationalPipeline()
        return cls._default_instance

    def _load_persisted_jobs(self) -> None:
        for f in self.storage_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    stages_dict = {
                        k: StageExecutionRecord(**v) if isinstance(v, dict) else v
                        for k, v in data.get("stage_states", {}).items()
                    }
                    d = dict(data)
                    d["stage_states"] = stages_dict
                    job = PipelineJobRecord(**d)
                    self._jobs[job.job_id] = job
                    if job.idempotency_key and job.status in ["SUCCESS", "PARTIAL_SUCCESS"]:
                        self._idempotency_index[job.idempotency_key] = job.job_id
            except Exception as e:
                logger.warning("Could not load job log %s: %s", f, e)

    def _persist_job(self, job: PipelineJobRecord) -> None:
        p = self.storage_dir / f"{job.job_id}.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(job.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to persist job record %s: %s", job.job_id, e)

    @staticmethod
    def compute_idempotency_key(
        source: str,
        cycle_date: str,
        lead_time_hours: int,
        model_version: str,
    ) -> str:
        """
        Computes deterministic SHA-256 idempotency key from source, cycle, lead time, and model version.
        Conforms strictly to Phase 7 Section 4.
        """
        raw = f"{source.strip().lower()}:{cycle_date.strip()}:{lead_time_hours}:{model_version.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:20]

    def execute_cycle(
        self,
        cycle_date: str,
        lead_time_hours: int = 24,
        source: str = "IMD_GFS_OPERATIONAL",
        raw_input_data: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
        ground_truth_obs: Optional[np.ndarray] = None,
        force_rerun: bool = False,
        model_id: Optional[str] = None,
    ) -> PipelineJobRecord:
        """
        Executes end-to-end 11-stage operational cycle with idempotency, fault containment,
        and machine-readable stage tracking.
        """
        manifest = VersionManager.get_active_manifest()
        prod_model = self.registry.get_production_model()
        active_model_version = prod_model.version if prod_model else manifest.repair_model_version

        # -------------------------------------------------------------
        # 1. Idempotency Check (Phase 7 Section 4)
        # -------------------------------------------------------------
        idem_key = self.compute_idempotency_key(source, cycle_date, lead_time_hours, active_model_version)

        if idem_key in self._idempotency_index and not force_rerun:
            existing_job_id = self._idempotency_index[idem_key]
            existing_job = self._jobs.get(existing_job_id)
            if existing_job and existing_job.status in ["SUCCESS", "PARTIAL_SUCCESS"]:
                logger.info(
                    "Idempotency hit for cycle %s lead %dh: reusing existing valid run %s.",
                    cycle_date, lead_time_hours, existing_job_id,
                )
                # Return existing run without duplicate output or overwrite
                reused = PipelineJobRecord(**existing_job.to_dict())
                reused.reused_existing = True
                reused.reused_from_job_id = existing_job_id
                return reused

        # Create new versioned job
        import uuid
        now_dt = datetime.now(timezone.utc)
        run_tag = f"v{int(time.time())}" if force_rerun and idem_key in self._idempotency_index else uuid.uuid4().hex[:6]
        job_id = f"job_{cycle_date.replace('-', '')}_lt{lead_time_hours}_{run_tag}"

        valid_time_dt = pd.to_datetime(cycle_date) + pd.Timedelta(hours=lead_time_hours)
        valid_time_str = valid_time_dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        job = PipelineJobRecord(
            job_id=job_id,
            pipeline_version="1.0.0",
            started_at=now_dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
            status="RUNNING",
            dataset_version=manifest.dataset_version,
            feature_version=manifest.feature_version,
            model_versions={
                "regime_model": manifest.regime_model_version,
                "repair_model": manifest.repair_model_version,
                "quantile_model": manifest.quantile_model_version,
                "calibration": manifest.calibration_version,
            },
            boundary_version=manifest.boundary_version,
            input_period={"cycle_date": cycle_date, "lead_time_hours": str(lead_time_hours)},
            output_period={"valid_time": valid_time_str},
            idempotency_key=idem_key,
        )

        # Initialize all stages to QUEUED
        for st in STAGES_ORDER:
            job.stage_states[st] = StageExecutionRecord(
                stage_name=st,
                status="QUEUED",
                version_metadata={"pipeline_version": "1.0.0", "dataset_version": manifest.dataset_version},
            )

        self._jobs[job_id] = job
        self._persist_job(job)

        # Context passing through stages
        context: Dict[str, Any] = {
            "cycle_date": cycle_date,
            "lead_time_hours": lead_time_hours,
            "valid_time_str": valid_time_str,
            "source": source,
            "raw_input_data": raw_input_data,
            "ground_truth_obs": ground_truth_obs,
            "manifest": manifest,
            "job_id": job_id,
        }

        # -------------------------------------------------------------
        # Sequential Stage Execution with Fault Propagation Guard
        # -------------------------------------------------------------
        stage_runners = [
            ("INGESTION", self._stage_ingestion),
            ("QUALITY_CHECK", self._stage_quality_check),
            ("DATA_ALIGNMENT", self._stage_alignment),
            ("FEATURE_GENERATION", self._stage_feature_generation),
            ("REGIME_INFERENCE", self._stage_regime_inference),
            ("QUANTILE_CORRECTION", self._stage_quantile_correction),
            ("CALIBRATION", self._stage_calibration),
            ("GRID_PREDICTION", self._stage_grid_prediction),
            ("DISTRICT_AGGREGATION", self._stage_district_aggregation),
            ("VALIDATION", self._stage_validation),
            ("PUBLISH_OUTPUT", self._stage_publish_output),
            ("MONITOR", self._stage_monitor),
        ]

        pipeline_failed = False

        for st_name, runner in stage_runners:
            st_rec = job.stage_states[st_name]
            if pipeline_failed:
                st_rec.status = "SKIPPED"
                continue

            st_rec.status = "RUNNING"
            st_rec.started_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            t0 = time.perf_counter()

            try:
                success, in_sum, out_sum, err_msg = runner(context, job)
                st_rec.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                st_rec.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                st_rec.input_summary = in_sum
                st_rec.output_summary = out_sum

                if success:
                    st_rec.status = "SUCCESS"
                else:
                    st_rec.status = "FAILED"
                    st_rec.error_detail = err_msg
                    pipeline_failed = True
                    job.error_summary = f"Stage {st_name} failed: {err_msg}"
                    logger.error("Job %s: Stage %s failed: %s", job_id, st_name, err_msg)
            except Exception as exc:
                st_rec.duration_ms = round((time.perf_counter() - t0) * 1000.0, 2)
                st_rec.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                st_rec.status = "FAILED"
                st_rec.error_detail = str(exc)
                pipeline_failed = True
                job.error_summary = f"Stage {st_name} unhandled exception: {exc}"
                logger.error("Job %s: Exception in stage %s: %s", job_id, st_name, exc, exc_info=True)
                # Trigger alert for unhandled failure
                alt = self.alert_manager.record_alert(
                    category="PIPELINE_FAILURE",
                    message=f"Pipeline job {job_id} encountered unhandled exception in {st_name}: {exc}",
                    severity="CRITICAL",
                    pipeline_run_id=job_id,
                )
                job.alerts_triggered.append(alt.alert_id)

        # -------------------------------------------------------------
        # Final Job Status Resolution
        # -------------------------------------------------------------
        job.completed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        if pipeline_failed:
            job.status = "FAILED"
        elif job.validation_report and job.validation_report.get("publication_state") == "PARTIAL":
            job.status = "PARTIAL_SUCCESS"
            self._idempotency_index[idem_key] = job_id
        else:
            job.status = "SUCCESS"
            self._idempotency_index[idem_key] = job_id

        self._persist_job(job)
        return job

    # -----------------------------------------------------------------
    # Stage 1: INGESTION
    # -----------------------------------------------------------------
    def _stage_ingestion(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        raw_in = ctx.get("raw_input_data")
        cycle_date = ctx["cycle_date"]

        if raw_in is not None:
            if isinstance(raw_in, pd.DataFrame):
                df = raw_in.copy()
            else:
                df = pd.DataFrame(raw_in)
        else:
            # Generate realistic multi-point synthetic meteorological cycle
            from src.models.raapx_pipeline import generate_synthetic_multicyle_data
            full_df = generate_synthetic_multicyle_data(n_days=15, n_grid_points=15, random_seed=42)
            df = full_df.iloc[:15].copy()
            df["cycle_date"] = cycle_date
            df["forecast_time"] = ctx["valid_time_str"]
            if "t2m" not in df.columns:
                df["t2m"] = 298.15 + (df["lat"] - 18.0) * (-0.3)
            if "u10" not in df.columns:
                df["u10"] = 6.5
            if "v10" not in df.columns:
                df["v10"] = 3.2
            if "mslp" not in df.columns:
                df["mslp"] = 1006.0
            if "latitude" not in df.columns and "lat" in df.columns:
                df["latitude"] = df["lat"]
            if "longitude" not in df.columns and "lon" in df.columns:
                df["longitude"] = df["lon"]


        if len(df) == 0:
            return False, {"source": ctx["source"]}, {}, "Ingestion returned 0 records."

        ctx["ingested_df"] = df
        return True, {"source": ctx["source"], "cycle_date": cycle_date}, {"records_ingested": len(df), "columns": list(df.columns)[:8]}, None

    # -----------------------------------------------------------------
    # Stage 2: QUALITY CHECK (QC)
    # -----------------------------------------------------------------
    def _stage_quality_check(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        df = ctx["ingested_df"]
        report = self.data_quality_monitor.audit_ingested_cycle(
            df=df,
            cycle_date=ctx["cycle_date"],
            pipeline_run_id=job.job_id,
            dataset_version=job.dataset_version,
            source_timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        job.quality_report = report.to_dict()
        ctx["quality_report"] = report

        if report.status == "FAILED":
            return False, {"records": len(df)}, report.to_dict(), f"Data quality check failed: {report.validity.failures + report.completeness.failures}"

        return True, {"records": len(df)}, {"status": report.status, "valid_records": report.valid_records}, None

    # -----------------------------------------------------------------
    # Stage 3: DATA ALIGNMENT
    # -----------------------------------------------------------------
    def _stage_alignment(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        df = ctx["ingested_df"].copy()
        # Ensure standard column naming and coordinate precision
        if "lat" in df.columns and "latitude" not in df.columns:
            df["latitude"] = df["lat"]
        if "lon" in df.columns and "longitude" not in df.columns:
            df["longitude"] = df["lon"]

        df["lead_time"] = ctx["lead_time_hours"]
        df["initialization_time"] = f"{ctx['cycle_date']}T00:00:00Z"
        df["forecast_time"] = ctx["valid_time_str"]

        ctx["aligned_df"] = df
        return True, {"input_records": len(df)}, {"aligned_records": len(df), "lead_time": ctx["lead_time_hours"]}, None

    # -----------------------------------------------------------------
    # Stage 4: FEATURE GENERATION
    # -----------------------------------------------------------------
    def _stage_feature_generation(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        df = ctx["aligned_df"]
        pipe = self.inference_pipeline

        feature_records: List[Dict[str, float]] = []
        for _, row in df.iterrows():
            rec_in = ForecastInputRecord(
                cycle_date=ctx["cycle_date"],
                initialization_time=f"{ctx['cycle_date']}T00:00:00Z",
                forecast_time=ctx["valid_time_str"],
                latitude=float(row.get("latitude", 18.5)),
                longitude=float(row.get("longitude", 73.8)),
                lead_time=ctx["lead_time_hours"],
                nwp_precip=float(row.get("nwp_precip", row.get("raw_nwp_rainfall", 20.0))),
                t2m=float(row.get("t2m", row.get("nwp_t2m", 298.15))),
                q2m=float(row.get("q2m", row.get("nwp_q2m", 0.016))),
                u10=float(row.get("u10", row.get("nwp_u10", 5.0))),
                v10=float(row.get("v10", row.get("nwp_v10", 5.0))),
                mslp=float(row.get("mslp", row.get("nwp_mslp", 1008.0))),
                cape=float(row.get("cape", 1200.0)),
            )
            feats = pipe._normalize_input_record(rec_in)
            # Add zero-leakage error memory features
            err_feats, _ = pipe._retrieve_error_memory_features(f"{ctx['cycle_date']}T00:00:00Z", f"pt_{rec_in.latitude:.2f}_{rec_in.longitude:.2f}")
            feats.update(err_feats)
            feature_records.append(feats)

        ctx["feature_df"] = pd.DataFrame(feature_records)
        return True, {"records": len(df)}, {"features_extracted": len(feature_records[0]) if feature_records else 0}, None

    # -----------------------------------------------------------------
    # Stage 5: REGIME INFERENCE
    # -----------------------------------------------------------------
    def _stage_regime_inference(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        feat_df = ctx["feature_df"]
        pipe = self.inference_pipeline
        pkg = pipe.model_loader.load_model(base_dir=Path.cwd())
        from src.regime.schemas import RegimeProbabilityVector

        in_feat = feat_df.copy()
        for c in pkg.regime_classifier.feature_cols:
            if c not in in_feat.columns:
                in_feat[c] = 0.0

        probs_matrix = pkg.regime_classifier.predict_proba(in_feat[pkg.regime_classifier.feature_cols])

        regime_records = []
        for i in range(len(probs_matrix)):
            p_arr = probs_matrix[i]
            if not np.isclose(float(np.sum(p_arr)), 1.0, atol=1e-3):
                p_arr = p_arr / max(1e-6, float(np.sum(p_arr)))
            vec = RegimeProbabilityVector.from_array(p_arr)
            dom, p_dom = vec.dominant_regime()
            regime_records.append({
                "dominant_regime": dom.name if hasattr(dom, "name") else str(dom),
                "dominant_probability": round(float(p_dom), 4),
                "regime_probabilities": {k.upper(): round(float(v), 4) for k, v in vec.to_dict().items()},
            })

        ctx["regime_df"] = pd.DataFrame(regime_records)
        dom_counts = ctx["regime_df"]["dominant_regime"].value_counts().to_dict()
        return True, {"feature_rows": len(feat_df)}, {"regimes_assigned": len(regime_records), "dominant_distribution": dom_counts}, None

    # -----------------------------------------------------------------
    # Stage 6: QUANTILE CORRECTION
    # -----------------------------------------------------------------
    def _stage_quantile_correction(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        feat_df = ctx["feature_df"]
        reg_df = ctx["regime_df"]
        pipe = self.inference_pipeline
        pkg = pipe.model_loader.load_model(base_dir=Path.cwd())
        from src.models.inference_pipeline import REGIME_PROB_COLS

        # Combine features with regime probabilities
        combined_rows = []
        for i in range(len(feat_df)):
            f_dict = feat_df.iloc[i].to_dict()
            r_dict = reg_df.iloc[i]["regime_probabilities"]
            for c_reg in REGIME_PROB_COLS:
                key_upper = c_reg.replace("prob_", "").upper()
                f_dict[c_reg] = float(r_dict.get(key_upper, 0.0))
            combined_rows.append(f_dict)

        combined_df = pd.DataFrame(combined_rows)
        for c in pkg.quantile_corrector.feature_cols:
            if c not in combined_df.columns:
                combined_df[c] = 0.0

        quant_preds = pkg.quantile_corrector.predict_quantiles_dict(combined_df)
        ctx["quantile_preds"] = quant_preds
        return True, {"records": len(feat_df)}, {"quantiles_predicted": [str(k) for k in quant_preds.keys()]}, None


    # -----------------------------------------------------------------
    # Stage 7: CALIBRATION & MONOTONICITY
    # -----------------------------------------------------------------
    def _stage_calibration(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        q_preds = ctx["quantile_preds"]
        from src.postprocessing.constraints import enforce_quantile_monotonicity

        calibrated_dict = enforce_quantile_monotonicity(q_preds, enforce_non_negativity=True)
        ctx["calibrated_dict"] = calibrated_dict

        p50 = calibrated_dict.get(0.50, np.array([0.0]))
        p75 = calibrated_dict.get(0.75, np.array([0.0]))
        p90 = calibrated_dict.get(0.90, np.array([0.0]))

        calibrated_mono = []
        for i in range(len(p50)):
            calibrated_mono.append((float(p50[i]), float(p75[i]), float(p90[i])))

        ctx["calibrated_quantiles"] = calibrated_mono
        return True, {"input_quantiles": len(q_preds)}, {"calibrated_records": len(calibrated_mono), "monotonicity_enforced": True}, None


    # -----------------------------------------------------------------
    # Stage 8: GRID PREDICTION ASSEMBLY
    # -----------------------------------------------------------------
    def _stage_grid_prediction(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        df = ctx["aligned_df"]
        calibrated = ctx["calibrated_quantiles"]
        reg_df = ctx["regime_df"]

        grid_records: List[Dict[str, Any]] = []
        for i in range(len(df)):
            c50, c75, c90 = calibrated[i]
            r_info = reg_df.iloc[i]
            row = df.iloc[i]
            grid_records.append({
                "latitude": float(row.get("latitude", 18.5)),
                "longitude": float(row.get("longitude", 73.8)),
                "lead_time": ctx["lead_time_hours"],
                "initialization_time": f"{ctx['cycle_date']}T00:00:00Z",
                "forecast_time": ctx["valid_time_str"],
                "raw_nwp_rainfall": float(row.get("nwp_precip", row.get("raw_nwp_rainfall", 20.0))),
                "corrected_p50": float(c50),
                "corrected_p75": float(c75),
                "corrected_p90": float(c90),
                "spread_p90_p50": float(round(c90 - c50, 2)),
                "dominant_regime": r_info["dominant_regime"],
                "dominant_probability": r_info["dominant_probability"],
                "regime_probabilities": r_info["regime_probabilities"],
                "model_version": job.model_versions.get("repair_model", "1.0.0"),
                "feature_version": job.feature_version,
                "dataset_version": job.dataset_version,
            })

        ctx["grid_records"] = grid_records
        job.records_processed = len(grid_records)
        return True, {"points": len(df)}, {"assembled_grid_points": len(grid_records)}, None

    # -----------------------------------------------------------------
    # Stage 9: DISTRICT AGGREGATION
    # -----------------------------------------------------------------
    def _stage_district_aggregation(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        grid_df = pd.DataFrame(ctx["grid_records"])
        zonal_stats = self.zonal_engine.aggregate_grid_to_districts(
            grid_df=grid_df,
            lead_time=ctx["lead_time_hours"],
            model_version=job.model_versions.get("repair_model", "1.0.0"),
        )
        ctx["district_summaries"] = zonal_stats
        return True, {"grid_points": len(grid_df)}, {"districts_aggregated": len(zonal_stats)}, None

    # -----------------------------------------------------------------
    # Stage 10: AUTOMATED VALIDATION
    # -----------------------------------------------------------------
    def _stage_validation(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        grid_recs = ctx["grid_records"]
        dist_recs = ctx.get("district_summaries", [])

        v_report = self.verification_engine.verify_outputs(
            grid_predictions=grid_recs,
            district_summaries=dist_recs,
            manifest=ctx["manifest"],
            cycle_date=ctx["cycle_date"],
            lead_time_hours=ctx["lead_time_hours"],
            pipeline_run_id=job.job_id,
        )

        job.validation_report = v_report.to_dict()
        ctx["validation_report"] = v_report

        if not v_report.passed or v_report.publication_state == "INVALID":
            job.records_failed = v_report.invalid_grid_points
            return False, {"records": len(grid_recs)}, v_report.to_dict(), f"Verification failed: {v_report.failures}"

        return True, {"records": len(grid_recs)}, {"publication_state": v_report.publication_state, "valid_points": v_report.valid_grid_points}, None

    # -----------------------------------------------------------------
    # Stage 11: PUBLISH OUTPUT
    # -----------------------------------------------------------------
    def _stage_publish_output(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        v_report: VerificationReport = ctx["validation_report"]
        if v_report.publication_state == "INVALID":
            return False, {}, {}, "Output publishing blocked: Verification publication state is INVALID."

        # Publish to cache and database
        from backend.cache import cache_manager
        cache_key = f"forecast:{ctx['cycle_date']}:{ctx['lead_time_hours']}"
        cache_data = {
            "job_id": job.job_id,
            "cycle_date": ctx["cycle_date"],
            "lead_time": ctx["lead_time_hours"],
            "publication_state": v_report.publication_state,
            "grid_count": len(ctx["grid_records"]),
            "district_count": len(ctx.get("district_summaries", [])),
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "grid_records": ctx["grid_records"],
            "district_summaries": ctx.get("district_summaries", []),
            "model_version": job.model_versions.get("repair_model", "1.0.0"),
            "feature_version": job.feature_version,
            "dataset_version": job.dataset_version,
            "boundary_version": job.boundary_version,
        }
        cache_manager.set(cache_key, cache_data, ttl_seconds=86400)
        cache_manager.set(f"forecast:latest:{ctx['lead_time_hours']}:published", cache_data, ttl_seconds=86400)

        pub_summary = {
            "published": True,
            "publication_state": v_report.publication_state,
            "cache_key": cache_key,
            "grid_records_published": len(ctx["grid_records"]),
            "district_summaries_published": len(ctx.get("district_summaries", [])),
            "missing_areas": v_report.missing_areas,
        }
        job.published_output_summary = pub_summary
        return True, {"state": v_report.publication_state}, pub_summary, None

    # -----------------------------------------------------------------
    # Stage 12: MONITOR (Drift & Optional Performance)
    # -----------------------------------------------------------------
    def _stage_monitor(self, ctx: Dict[str, Any], job: PipelineJobRecord) -> Tuple[bool, Dict[str, Any], Dict[str, Any], Optional[str]]:
        # Feature & Regime Drift Evaluation
        feat_df = ctx.get("feature_df")
        reg_df = ctx.get("regime_df")

        drift_rep = self.drift_monitor.generate_drift_report(
            current_df=feat_df,
            pipeline_run_id=job.job_id,
            current_regimes=reg_df["dominant_regime"] if reg_df is not None else None,
        )
        job.drift_summary = drift_rep.to_dict()

        # Performance monitoring if ground truth observations are supplied
        obs = ctx.get("ground_truth_obs")
        if obs is not None and len(obs) == len(ctx["grid_records"]):
            p50 = np.array([r["corrected_p50"] for r in ctx["grid_records"]])
            p75 = np.array([r["corrected_p75"] for r in ctx["grid_records"]])
            p90 = np.array([r["corrected_p90"] for r in ctx["grid_records"]])

            perf_rep = self.performance_monitor.evaluate_observations(
                y_true=obs,
                p50=p50,
                p75=p75,
                p90=p90,
                cycle_date=ctx["cycle_date"],
                pipeline_run_id=job.job_id,
                model_version=job.model_versions.get("repair_model", "1.0.0"),
            )
            job.performance_report = perf_rep.to_dict()

        return True, {"drift_features": len(feat_df.columns) if feat_df is not None else 0}, {"monitoring_state": drift_rep.overall_state}, None

    def get_job(self, job_id: str) -> Optional[PipelineJobRecord]:
        return self._jobs.get(job_id)

    def list_jobs(self, limit: int = 50) -> List[PipelineJobRecord]:
        return list(self._jobs.values())[-limit:]

    # Operational alias
    run_pipeline = execute_cycle

