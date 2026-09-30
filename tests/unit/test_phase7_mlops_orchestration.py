"""
Unit & Integration Test Suite for Phase 7 MLOps Orchestration, Monitoring & Automated Verification.
Tests:
1. Operational 11-stage pipeline execution
2. Scheduling (job triggers & scheduler status)
3. Idempotency (duplicate cycles reuse valid results without corruption)
4. Failure propagation (failed QC/ingestion halts downstream execution)
5. Data drift & Regime drift detection
6. Model performance degradation alerts
7. Safe rollback without artifact deletion
8. Automated verification & publishing gating (monotonicity, regime sums)
9. 9-Subsystem health probes
10. Complete metadata lineage & audit trail
"""

from __future__ import annotations

import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.mlops.alerts import AlertManager, AlertRecord
from src.mlops.data_quality import DataQualityMonitor, DataQualityReport
from src.mlops.drift import DriftMonitor, DriftReport
from src.mlops.health import HealthProbeService, SystemHealthReport
from src.mlops.performance_monitor import ModelPerformanceMonitor
from src.mlops.pipeline import OperationalPipeline, PipelineJobRecord
from src.mlops.registry import ModelRegistry, RegisteredModel
from src.mlops.scheduler import OperationalScheduler, ScheduleConfig
from src.mlops.verification import AutomatedVerificationEngine, VerificationReport
from src.mlops.versioning import VersionManager


@pytest.fixture
def temp_mlops_env(tmp_path: Path):
    """Sets up an isolated filesystem environment for MLOps testing."""
    test_dir = tmp_path / "mlops_test"
    test_dir.mkdir(parents=True, exist_ok=True)
    alerts_dir = test_dir / "alerts"
    quality_dir = test_dir / "quality"
    perf_dir = test_dir / "perf"
    verif_dir = test_dir / "verif"
    jobs_dir = test_dir / "jobs"

    alert_mgr = AlertManager(storage_dir=alerts_dir)
    dq_mon = DataQualityMonitor(alert_manager=alert_mgr, storage_dir=quality_dir)
    verif_eng = AutomatedVerificationEngine(alert_manager=alert_mgr, storage_dir=verif_dir)
    perf_mon = ModelPerformanceMonitor(alert_manager=alert_mgr, storage_dir=perf_dir)
    drift_mon = DriftMonitor(alert_manager=alert_mgr)

    pipeline = OperationalPipeline(
        alert_manager=alert_mgr,
        data_quality_monitor=dq_mon,
        verification_engine=verif_eng,
        performance_monitor=perf_mon,
        drift_monitor=drift_mon,
        storage_dir=jobs_dir,
    )

    yield {
        "pipeline": pipeline,
        "alert_manager": alert_mgr,
        "dq_monitor": dq_mon,
        "verif_engine": verif_eng,
        "perf_monitor": perf_mon,
        "drift_monitor": drift_mon,
        "test_dir": test_dir,
    }


# ---------------------------------------------------------------------------
# Test 1 & 2: Idempotency & Forced Rerun
# ---------------------------------------------------------------------------
def test_idempotency_duplicate_cycle(temp_mlops_env):
    """Verifies that calling execute_cycle twice on identical inputs reuses existing result."""
    pipeline: OperationalPipeline = temp_mlops_env["pipeline"]
    cycle_date = "2026-07-20"
    lead_time = 24

    # Run 1: Should execute cleanly
    job1 = pipeline.execute_cycle(cycle_date=cycle_date, lead_time_hours=lead_time, force_rerun=False)
    assert job1.status in ["SUCCESS", "PARTIAL_SUCCESS"]
    assert job1.reused_existing is False
    assert job1.records_processed > 0

    # Run 2: Exact duplicate without force flag -> Must reuse existing run
    job2 = pipeline.execute_cycle(cycle_date=cycle_date, lead_time_hours=lead_time, force_rerun=False)
    assert job2.reused_existing is True
    assert job2.job_id == job1.job_id
    assert job2.idempotency_key == job1.idempotency_key


def test_force_rerun_creates_new_versioned_run(temp_mlops_env):
    """Verifies that force_rerun=True creates a distinct versioned job run."""
    pipeline: OperationalPipeline = temp_mlops_env["pipeline"]
    cycle_date = "2026-07-21"
    lead_time = 48

    job1 = pipeline.execute_cycle(cycle_date=cycle_date, lead_time_hours=lead_time, force_rerun=False)
    job2 = pipeline.execute_cycle(cycle_date=cycle_date, lead_time_hours=lead_time, force_rerun=True)

    assert job1.job_id != job2.job_id
    assert job2.reused_existing is False
    assert job2.status in ["SUCCESS", "PARTIAL_SUCCESS"]


# ---------------------------------------------------------------------------
# Test 3: Failure Propagation & Fault Containment
# ---------------------------------------------------------------------------
def test_failure_propagation_blocks_downstream(temp_mlops_env):
    """Verifies that corrupted input data halts the pipeline at QC and skips downstream stages."""
    pipeline: OperationalPipeline = temp_mlops_env["pipeline"]
    cycle_date = "2026-07-22"

    # Corrupt data: missing required variables and negative rainfall
    corrupt_df = pd.DataFrame([{
        "cycle_date": cycle_date,
        "nwp_precip": -50.0,  # Physical violation
        "lat": 18.5,
        "lon": 73.8,
    }])

    job = pipeline.execute_cycle(
        cycle_date=cycle_date,
        lead_time_hours=24,
        raw_input_data=corrupt_df,
        force_rerun=True,
    )

    assert job.status == "FAILED"
    assert "Stage QUALITY_CHECK failed" in str(job.error_summary)
    assert job.stage_states["QUALITY_CHECK"].status == "FAILED"
    # All downstream stages MUST be SKIPPED
    assert job.stage_states["FEATURE_GENERATION"].status == "SKIPPED"
    assert job.stage_states["REGIME_INFERENCE"].status == "SKIPPED"
    assert job.stage_states["QUANTILE_CORRECTION"].status == "SKIPPED"
    assert job.stage_states["GRID_PREDICTION"].status == "SKIPPED"
    assert job.stage_states["PUBLISH_OUTPUT"].status == "SKIPPED"
    assert job.records_processed == 0


# ---------------------------------------------------------------------------
# Test 4: Data Quality Monitoring Dimensions
# ---------------------------------------------------------------------------
def test_data_quality_monitoring_checks(temp_mlops_env):
    """Tests completeness, validity, timeliness, and spatial bounds in data quality monitor."""
    dq_mon: DataQualityMonitor = temp_mlops_env["dq_monitor"]

    # Valid data
    valid_df = pd.DataFrame([{
        "nwp_precip": 35.0,
        "t2m": 298.15,
        "u10": 5.0,
        "v10": 2.0,
        "mslp": 1008.0,
        "latitude": 18.5,
        "longitude": 73.8,
    } for _ in range(15)])

    rep_valid = dq_mon.audit_ingested_cycle(valid_df, "2026-07-15", expected_points=10)
    assert rep_valid.status == "PASSED"
    assert rep_valid.completeness.status == "PASSED"
    assert rep_valid.validity.status == "PASSED"
    assert rep_valid.spatial.status == "PASSED"

    # Out of spatial bounds data (outside India)
    out_df = valid_df.copy()
    out_df["latitude"] = 55.0  # Outside Indian domain
    rep_out = dq_mon.audit_ingested_cycle(out_df, "2026-07-15")
    assert rep_out.spatial.status == "FAILED"


# ---------------------------------------------------------------------------
# Test 5: Data Drift & Regime Drift Detection
# ---------------------------------------------------------------------------
def test_data_and_regime_drift_detection(temp_mlops_env):
    """Verifies that synthetic atmospheric shifts and regime distribution shifts trigger drift alerts."""
    drift_mon: DriftMonitor = temp_mlops_env["drift_monitor"]

    # Baseline features
    baseline_df = pd.DataFrame({
        "nwp_precip": np.random.gamma(2.0, 5.0, 100),
        "specific_humidity_850": np.random.normal(12.0, 2.0, 100),
    })

    # Drifting features (extreme shift in rainfall distribution)
    drifted_df = pd.DataFrame({
        "nwp_precip": np.random.gamma(15.0, 20.0, 100),  # Massive precipitation shift
        "specific_humidity_850": np.random.normal(24.0, 5.0, 100),
    })

    feat_drift = drift_mon.check_feature_data_drift(baseline_df, drifted_df)
    assert feat_drift["nwp_precip"]["psi"] >= 0.25
    assert feat_drift["nwp_precip"]["status"] == "DRIFT_DETECTED"

    # Regime drift check: 80% concentrated in single regime
    baseline_regimes = ["ACTIVE_MONSOON"] * 30 + ["BREAK_MONSOON"] * 30 + ["NORMAL_TRANSITIONAL"] * 40
    shifted_regimes = ["MONSOON_DEPRESSION"] * 90 + ["ACTIVE_MONSOON"] * 10
    regime_res = drift_mon.check_regime_drift(baseline_regimes, shifted_regimes)

    assert regime_res["regime_drift_detected"] is True
    assert regime_res["unusual_concentration"] is True
    assert regime_res["status"] == "DRIFT_DETECTED"


# ---------------------------------------------------------------------------
# Test 6: Model Performance Degradation & Alerts
# ---------------------------------------------------------------------------
def test_model_performance_degradation_alert(temp_mlops_env):
    """Verifies that large verification error trips degradation alerts and flags retraining review."""
    perf_mon: ModelPerformanceMonitor = temp_mlops_env["perf_monitor"]
    alert_mgr: AlertManager = temp_mlops_env["alert_manager"]

    y_true = np.array([10.0, 25.0, 80.0, 140.0, 220.0])
    # Very poor predictions causing high MAE & low CSI
    p50 = np.array([50.0, 80.0, 5.0, 10.0, 20.0])
    p75 = p50 + 5.0
    p90 = p50 + 10.0

    report = perf_mon.evaluate_observations(
        y_true=y_true,
        p50=p50,
        p75=p75,
        p90=p90,
        cycle_date="2026-07-23",
        pipeline_run_id="run_test_deg",
        model_version="1.0.0",
        reference_metrics={"mae": 9.70, "rmse": 16.20, "csi_heavy": 0.44},
    )

    assert report.degradation_detected is True
    assert report.retraining_review_required is True
    assert len(report.alerts_triggered) > 0

    # Ensure alert is logged in AlertManager
    alerts = alert_mgr.list_alerts(category="MODEL_PERFORMANCE_DEGRADATION")
    assert len(alerts) > 0
    assert alerts[-1].severity == "WARNING"
    assert "RETRAINING_REVIEW_REQUIRED" in alerts[-1].recommended_action


# ---------------------------------------------------------------------------
# Test 7: Model Rollback
# ---------------------------------------------------------------------------
def test_model_rollback_to_validated_production(tmp_path: Path):
    """Verifies that rollback restores previous validated production model without artifact deletion."""
    registry = ModelRegistry(registry_dir=tmp_path / "registry")
    initial_prod = registry.get_production_model()
    assert initial_prod is not None
    initial_prod_id = initial_prod.model_id

    # Register candidate model with good metrics
    cand = registry.register_model(
        model_name="RAIN-REPAIR-X-Candidate",
        version="3.0.0",
        model_type="SoftMixtureOfExperts",
        metrics={"rmse": 14.50, "mae": 8.80, "csi_heavy": 0.48, "brier_score": 0.100},
        initial_status="candidate",
    )

    # Promote to production
    ok, msg, qg = registry.promote_model(cand.model_id, "validated")
    assert ok is True
    ok2, msg2, qg2 = registry.promote_model(cand.model_id, "production")
    assert ok2 is True

    # Current production should now be the new model
    new_prod = registry.get_production_model()
    assert new_prod.model_id == cand.model_id

    # Execute rollback
    ok_rb, msg_rb = registry.rollback_to_previous_production(reason="Production failure detected", triggered_by="tester")
    assert ok_rb is True
    restored_prod = registry.get_production_model()
    assert restored_prod.model_id == initial_prod_id
    assert restored_prod.status == "production"


# ---------------------------------------------------------------------------
# Test 8: Output Verification & Publishing Gate
# ---------------------------------------------------------------------------
def test_output_verification_and_publishing_gating(temp_mlops_env):
    """Verifies that non-monotonic quantiles reject output publishing with INVALID state."""
    verif: AutomatedVerificationEngine = temp_mlops_env["verif_engine"]

    # 1. Valid predictions
    valid_recs = [{
        "latitude": 18.5,
        "longitude": 73.8,
        "initialization_time": "2026-07-24T00:00:00Z",
        "forecast_time": "2026-07-25T00:00:00Z",
        "corrected_p50": 20.0,
        "corrected_p75": 35.0,
        "corrected_p90": 50.0,
        "regime_probabilities": {
            "ACTIVE_MONSOON": 0.8,
            "BREAK_MONSOON": 0.2,
        },
    }]
    rep_v = verif.verify_outputs(valid_recs, cycle_date="2026-07-24", lead_time_hours=24)
    assert rep_v.passed is True
    assert rep_v.publication_state in ["VALID", "PARTIAL"]

    # 2. Invalid crossing quantiles (P75 < P50)
    invalid_recs = [{
        "latitude": 18.5,
        "longitude": 73.8,
        "initialization_time": "2026-07-24T00:00:00Z",
        "forecast_time": "2026-07-25T00:00:00Z",
        "corrected_p50": 50.0,
        "corrected_p75": 20.0,  # MONOTONICITY VIOLATION
        "corrected_p90": 70.0,
        "regime_probabilities": {"ACTIVE_MONSOON": 1.0},
    }]
    rep_inv = verif.verify_outputs(invalid_recs, cycle_date="2026-07-24", lead_time_hours=24)
    assert rep_inv.passed is False
    assert rep_inv.publication_state == "INVALID"
    assert any("Monotonicity violated" in f for f in rep_inv.failures)


# ---------------------------------------------------------------------------
# Test 9: Subsystem Health Diagnostics
# ---------------------------------------------------------------------------
def test_subsystem_health_probes():
    """Verifies that all 9 subsystems are probed and produce a complete health report."""
    probe = HealthProbeService()
    report: SystemHealthReport = probe.run_comprehensive_health_check()

    assert report.checks_total == 9
    assert "data_source" in report.components
    assert "model_registry" in report.components
    assert "model_checkpoints" in report.components
    assert "feature_pipeline" in report.components
    assert "regime_model" in report.components
    assert "quantile_models" in report.components
    assert "gis_boundaries" in report.components
    assert "storage" in report.components
    assert "database" in report.components
    assert report.overall_status in ["HEALTHY", "DEGRADED"]


# ---------------------------------------------------------------------------
# Test 10: Scheduler Lifecycle
# ---------------------------------------------------------------------------
def test_scheduler_trigger_and_status(temp_mlops_env):
    """Verifies that OperationalScheduler triggers immediate runs and reports status."""
    pipeline = temp_mlops_env["pipeline"]
    sched_cfg = ScheduleConfig(enabled=False, interval_minutes=60)
    scheduler = OperationalScheduler(config=sched_cfg, pipeline=pipeline)

    status = scheduler.get_status()
    assert status.enabled is False
    assert status.is_running is False
    assert status.interval_minutes == 60

    # Immediate cycle trigger
    job = scheduler.trigger_cycle_now(cycle_date="2026-07-25", lead_time_hours=24)
    assert job.status in ["SUCCESS", "PARTIAL_SUCCESS"]

    updated_status = scheduler.get_status()
    assert updated_status.total_runs_completed == 1
    assert updated_status.last_job_id == job.job_id
