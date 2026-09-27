"""
Unit Tests for PART 10 MLOps & Model Lifecycle Management.
Covers:
- Immutable versioning (dataset, feature, models, boundaries, API)
- Experiment tracking & Baselines A-F & Ablation summaries
- Time-aware validation and temporal split verification
- Quality gates (RMSE, calibration, Heavy CSI, regime sample sufficiency)
- Model registry lifecycle (candidate -> staging -> production -> rollback)
- Data drift (PSI, KS-test) & performance drift
- Error memory (3/7/14-day) and analog memory health monitoring
- Retraining pipeline triggers & validation execution
"""

import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.mlops.versioning import VersionManager, SystemVersionManifest
from src.mlops.experiments import ExperimentTracker, ExperimentRecord, TemporalSplitInfo
from src.mlops.quality_gates import ModelQualityGate, QualityGateResult
from src.mlops.registry import ModelRegistry, RegisteredModel
from src.mlops.drift import DriftMonitor, DriftReport
from src.mlops.retraining import RetrainingPipeline


class TestMLOpsVersioning:
    """Tests for immutable version tracking across all subsystems."""

    def test_version_manifest_immutability(self):
        manifest = VersionManager.get_active_manifest()
        assert manifest.dataset_version.startswith("IMD-")
        assert manifest.boundary_version == "IMD-LGD-2026.1"
        assert "v1.0.0" in manifest.api_version

        # Check deterministic SHA-256 fingerprint
        fingerprint = manifest.compute_manifest_hash()
        assert len(fingerprint) == 16
        # Same manifest yields identical hash
        assert fingerprint == manifest.compute_manifest_hash()


    def test_version_bump_immutability(self):
        base_version = "FEAT-2.4.0"
        next_patch = VersionManager.bump_version(base_version, "patch")
        assert next_patch == "FEAT-2.4.1"
        next_minor = VersionManager.bump_version(base_version, "minor")
        assert next_minor == "FEAT-2.5.0"
        next_major = VersionManager.bump_version(base_version, "major")
        assert next_major == "FEAT-3.0.0"


class TestMLOpsExperiments:
    """Tests for Experiment Tracking, Baselines A-F, and Ablations."""

    def test_experiment_tracker_baselines(self, tmp_path):
        tracker = ExperimentTracker(store_path=tmp_path / "experiments.json")
        exps = tracker.list_experiments()
        assert len(exps) >= 6

        exp_ids = [e.experiment_id for e in exps]
        assert "EXP_A_RAW_NWP" in exp_ids
        assert "EXP_B_TRAD_BIAS" in exp_ids
        assert "EXP_C_GENERIC_ML" in exp_ids
        assert "EXP_D_REGIME_AWARE" in exp_ids
        assert "EXP_E_RAIN_REPAIR_X" in exp_ids
        assert "EXP_F_RAIN_REPAIR_EVT" in exp_ids

        # Baseline comparison table
        table = tracker.get_baseline_comparison()
        assert len(table) >= 6
        # EXP E / F should have superior CSI / lower RMSE than Raw NWP
        raw_nwp = next(item for item in table if item["experiment_id"] == "EXP_A_RAW_NWP")
        repair_x = next(item for item in table if item["experiment_id"] == "EXP_E_RAIN_REPAIR_X")
        assert repair_x["rmse"] < raw_nwp["rmse"]
        assert repair_x["heavy_rain_csi"] > raw_nwp["heavy_rain_csi"]

    def test_temporal_split_integrity(self, tmp_path):
        tracker = ExperimentTracker(store_path=tmp_path / "experiments.json")
        for exp in tracker.list_experiments():
            split = exp.temporal_split
            # Train period must precede validation period
            assert split.train_end <= split.validation_start
            # Validation period must precede test period
            assert split.validation_end <= split.test_start

    def test_ablation_summary(self, tmp_path):
        tracker = ExperimentTracker(store_path=tmp_path / "experiments.json")
        ablations = tracker.get_ablation_summary()
        assert len(ablations) >= 5
        for item in ablations:
            assert "component" in item
            assert "metric_before" in item
            assert "metric_after" in item
            assert "delta" in item


class TestMLOpsQualityGates:
    """Tests for rigorous model promotion quality gates."""

    def test_quality_gate_passes_for_superior_model(self):
        gate = ModelQualityGate()
        prod_metrics = {
            "rmse": 12.0,
            "brier_score_heavy": 0.15,
            "heavy_rain_csi": 0.45,
            "quantile_monotonicity_violations": 0,
            "negative_rainfall_count": 0,
            "regime_metrics": {
                "Active Monsoon": {"rmse": 14.0, "sample_size": 150},
                "Break Monsoon": {"rmse": 9.0, "sample_size": 100},
            },
        }
        cand_metrics = {
            "rmse": 10.8,
            "brier_score_heavy": 0.13,
            "heavy_rain_csi": 0.49,
            "quantile_monotonicity_violations": 0,
            "negative_rainfall_count": 0,
            "regime_metrics": {
                "Active Monsoon": {"rmse": 12.5, "sample_size": 150},
                "Break Monsoon": {"rmse": 8.5, "sample_size": 100},
            },
        }
        result = gate.evaluate(
            candidate_metrics=cand_metrics,
            production_metrics=prod_metrics,
            target_status="production",
        )
        assert result.passed is True
        assert len(result.failures) == 0

    def test_quality_gate_fails_on_rmse_regression(self):
        gate = ModelQualityGate()
        prod_metrics = {"rmse": 10.0, "brier_score_heavy": 0.14, "heavy_rain_csi": 0.45}
        cand_metrics = {"rmse": 12.5, "brier_score_heavy": 0.14, "heavy_rain_csi": 0.45}  # 25% regression
        result = gate.evaluate(cand_metrics, prod_metrics, target_status="production")
        assert result.passed is False
        assert any("RMSE degraded" in f for f in result.failures)

    def test_quality_gate_fails_on_quantile_inversion(self):
        gate = ModelQualityGate()
        prod_metrics = {"rmse": 10.0, "brier_score_heavy": 0.14, "heavy_rain_csi": 0.45}
        cand_metrics = {
            "rmse": 9.5,
            "brier_score_heavy": 0.14,
            "heavy_rain_csi": 0.45,
            "quantile_monotonicity_violations": 5,
        }
        result = gate.evaluate(cand_metrics, prod_metrics, target_status="production")
        assert result.passed is False
        assert any("monotonicity" in f for f in result.failures)

    def test_quality_gate_regime_insufficient_sample(self):
        gate = ModelQualityGate()
        cand_metrics = {
            "rmse": 10.0,
            "brier_score_heavy": 0.14,
            "heavy_rain_csi": 0.45,
            "regime_metrics": {
                "Western Disturbance": {"rmse": 15.0, "sample_size": 8},  # < 25
            },
        }
        result = gate.evaluate(cand_metrics, target_status="staging")
        # Should not fail, but should issue warning and mark INSUFFICIENT_SAMPLE
        assert any("INSUFFICIENT_SAMPLE" in w for w in result.warnings)


class TestMLOpsModelRegistry:
    """Tests for model registration, promotion, and rollback."""

    def test_registry_lifecycle_and_promotion(self, tmp_path):
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        models = registry.list_models()
        assert len(models) >= 2

        # Check default active production model
        prod = registry.get_production_model()
        assert prod is not None
        assert prod.status == "production"

        # Register candidate
        manifest = VersionManager.get_active_manifest()
        candidate = registry.register_model(
            model_name="rain_repair_candidate",
            version="REPAIR-2.4.1",
            model_type="regime_aware_lightgbm_quantile",
            manifest=manifest,
            temporal_split=TemporalSplitInfo(
                train_start="2020-06-01",
                train_end="2024-09-30",
                validation_start="2025-06-01",
                validation_end="2025-09-30",
                test_start="2026-06-01",
                test_end="2026-09-30",
            ),
            metrics={
                "rmse": 10.2,
                "brier_score_heavy": 0.12,
                "heavy_rain_csi": 0.50,
                "quantile_monotonicity_violations": 0,
                "negative_rainfall_count": 0,
            },
            status="candidate",
        )
        assert candidate.status == "candidate"

        # Promote candidate to staging
        ok, msg, qg = registry.promote_model(candidate.model_id, "staging")
        assert ok is True

        # Promote to production
        ok, msg, qg = registry.promote_model(candidate.model_id, "production")
        assert ok is True
        assert registry.get_production_model().model_id == candidate.model_id

    def test_registry_safe_rollback(self, tmp_path):
        registry = ModelRegistry(registry_dir=tmp_path / "registry")
        initial_prod = registry.get_production_model()

        # Register and promote a candidate
        manifest = VersionManager.get_active_manifest()
        cand = registry.register_model(
            model_name="new_challenger",
            version="REPAIR-2.5.0",
            model_type="regime_aware_lightgbm_quantile",
            manifest=manifest,
            temporal_split=TemporalSplitInfo(
                train_start="2020-06-01",
                train_end="2024-09-30",
                validation_start="2025-06-01",
                validation_end="2025-09-30",
                test_start="2026-06-01",
                test_end="2026-09-30",
            ),
            metrics={"rmse": 9.8, "brier_score_heavy": 0.11, "heavy_rain_csi": 0.52},
            status="validated",
        )
        registry.promote_model(cand.model_id, "production")
        assert registry.get_production_model().model_id == cand.model_id

        # Now execute safe rollback
        ok, msg = registry.rollback_to_previous_production(reason="Sudden calibration drift in field test")
        assert ok is True
        # Production should be restored to initial_prod
        restored = registry.get_production_model()
        assert restored.model_id == initial_prod.model_id


class TestMLOpsDriftMonitoring:
    """Tests for PSI, KS-test, concept drift, and memory health."""

    def test_psi_and_ks_computation(self):
        np.random.seed(42)
        ref_data = np.random.normal(loc=15.0, scale=5.0, size=200)
        curr_data_nodrift = np.random.normal(loc=15.1, scale=5.1, size=200)
        curr_data_drift = np.random.normal(loc=25.0, scale=8.0, size=200)

        # No drift case
        psi_low, status_low = DriftMonitor.calculate_psi(ref_data, curr_data_nodrift)
        assert psi_low < 0.20
        assert status_low in ["STABLE", "MODERATE_SHIFT"]

        # Drift case
        psi_high, status_high = DriftMonitor.calculate_psi(ref_data, curr_data_drift)
        assert psi_high >= 0.20
        assert status_high == "SIGNIFICANT_DRIFT"

        # KS test
        ks_stat, ks_p, ks_drift = DriftMonitor.calculate_ks_test(ref_data, curr_data_drift)
        assert ks_drift is True
        assert ks_p < 0.05

    def test_drift_report_generation(self, tmp_path):
        monitor = DriftMonitor(baseline_store=tmp_path / "baseline_dist.json")
        report = monitor.generate_drift_report()
        assert isinstance(report, DriftReport)
        assert report.overall_state in [
            "HEALTHY",
            "WARNING",
            "DRIFT_DETECTED",
            "PERFORMANCE_DEGRADED",
            "CALIBRATION_DEGRADED",
        ]
        assert "raw_nwp_rainfall" in report.feature_drift_summary
        assert "recent_rmse" in report.performance_drift_summary
        assert "window_3d" in report.error_memory_health
        assert "window_7d" in report.error_memory_health
        assert "window_14d" in report.error_memory_health
        assert "analogs_available" in report.analog_memory_health


class TestMLOpsRetrainingPipeline:
    """Tests for controlled retraining workflow."""

    def test_retraining_execution(self, tmp_path):
        pipeline = RetrainingPipeline(working_dir=tmp_path / "retraining")
        # Run review (force=False should return no_action if metrics are healthy)
        res = pipeline.execute_retraining_pipeline(force=False)
        assert "status" in res
        assert res["status"] in ["no_action_required", "retraining_completed"]

        # Forced run should trigger candidate creation and quality gating
        res_forced = pipeline.execute_retraining_pipeline(force=True, triggered_by="test_admin")
        assert res_forced["status"] == "retraining_completed"
        assert "candidate_model_id" in res_forced
        assert "quality_gate" in res_forced
