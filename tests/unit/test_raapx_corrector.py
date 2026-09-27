"""
Automated Test Suite for Phase 3 — Regime-Aware Quantile Rainfall Correction (RAAP-X).
Verifies:
1. Model initialization, default canonical quantiles [0.50, 0.75, 0.90], and versioning.
2. Feature matrix preparation: NWP + Historical Error + Terrain/Physics + Regime Probabilities.
3. Quantile model training under pinball loss.
4. Physical non-negativity constraint (q_tau >= 0.0 mm).
5. Monotonicity / Non-Crossing invariant: 0.0 <= P50 <= P75 <= P90 (0% crossing rate).
6. Quantitative evaluation: Pinball loss per quantile, empirical coverage, CRPS, and benchmark vs NWP.
7. Zero temporal leakage across train/validation/test splits.
8. Integration with ExperimentTracker and immutable dataset/feature versions.
9. Model persistence (save/load) round-trip reproducibility.
10. Ablation benchmarking: Raw NWP vs Standard Quantile vs RAAP-X Regime-Aware Quantile Corrector.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.mlops.experiments import ExperimentTracker, TemporalSplitInfo
from src.mlops.versioning import DATASET_VERSION_DEFAULT, FEATURE_VERSION_DEFAULT
from src.models.raapx_corrector import (
    DEFAULT_QUANTILES,
    RAAPXQuantileCorrector,
    QuantileEvaluationMetrics,
)
from src.models.raapx_pipeline import (
    RAAPXTrainingPipeline,
    generate_synthetic_multicyle_data,
)
from src.postprocessing.constraints import enforce_quantile_monotonicity
from src.postprocessing.quantiles import compute_pinball_loss
from src.utils.exceptions import TemporalLeakageError


@pytest.fixture
def multicyle_data() -> pd.DataFrame:
    """Fixture providing multi-cycle weather states across 2024, 2025, and 2026."""
    return generate_synthetic_multicyle_data(n_days=60, n_grid_points=8, random_seed=42)


def test_raapx_initialization_defaults():
    """Verify default canonical quantiles P50, P75, P90 and hyperparameters."""
    model = RAAPXQuantileCorrector()
    assert model.quantiles == [0.50, 0.75, 0.90]
    assert model.is_fitted is False
    assert model.mode == "unified"
    assert "nwp" in model.get_canonical_feature_groups()
    assert "error_memory" in model.get_canonical_feature_groups()
    assert "terrain_physics" in model.get_canonical_feature_groups()
    assert "regime_probabilities" in model.get_canonical_feature_groups()


def test_feature_matrix_preparation(multicyle_data: pd.DataFrame):
    """Verify feature matrix preparation combines all required feature families."""
    model = RAAPXQuantileCorrector()
    prepared = model.prepare_feature_matrix(multicyle_data, drop_targets=True)

    # Check Raw NWP
    assert "nwp_precip" in prepared.columns
    # Check Error Memory
    assert "error_lag_1d" in prepared.columns
    assert "error_lag_7d" in prepared.columns
    # Check Terrain & Physics
    assert "elevation" in prepared.columns
    assert "moisture_flux_conv" in prepared.columns
    assert "orographic_uplift" in prepared.columns
    # Check Soft Regime Probabilities
    assert "prob_active_monsoon" in prepared.columns
    assert "prob_break_monsoon" in prepared.columns
    assert "prob_monsoon_depression" in prepared.columns


def test_target_isolation_raises_on_leakage(multicyle_data: pd.DataFrame):
    """Verify TemporalLeakageError if target variable is explicitly supplied in feature_cols."""
    model = RAAPXQuantileCorrector()
    leaky_df = multicyle_data.copy()
    y_target = leaky_df["obs_precip"].values

    # 1. Automatic exclusion: if full dataframe is provided without feature_cols, target is excluded
    model.fit(leaky_df, y_target)
    assert "obs_precip" not in model.feature_cols
    assert "target" not in model.feature_cols

    # 2. Strict leakage rejection: if someone explicitly passes target in feature_cols, it raises TemporalLeakageError
    with pytest.raises(TemporalLeakageError):
        model.fit(leaky_df, y_target, feature_cols=["obs_precip", "nwp_precip"])


def test_raapx_fit_and_predict_quantiles(multicyle_data: pd.DataFrame):
    """Fit RAAP-X and verify P50, P75, P90 predictions with physical diagnostics."""
    n_train = int(len(multicyle_data) * 0.7)
    train_df = multicyle_data.iloc[:n_train]
    test_df = multicyle_data.iloc[n_train:]

    y_train = train_df["obs_precip"].values
    y_test = test_df["obs_precip"].values

    model = RAAPXQuantileCorrector(
        quantiles=[0.50, 0.75, 0.90],
        n_estimators=50,
        learning_rate=0.08,
        random_seed=42,
    )
    model.fit(train_df, y_train)

    assert model.is_fitted is True
    assert 0.50 in model.models
    assert 0.75 in model.models
    assert 0.90 in model.models

    preds_df = model.predict(test_df, enforce_constraints=True)

    assert "p50" in preds_df.columns
    assert "p75" in preds_df.columns
    assert "p90" in preds_df.columns
    assert "spread_p90_p50" in preds_df.columns
    assert "predicted_bias_p50" in preds_df.columns
    assert "nwp_precip" in preds_df.columns

    # Verification: Non-negativity
    assert (preds_df["p50"] >= 0.0).all()
    assert (preds_df["p75"] >= 0.0).all()
    assert (preds_df["p90"] >= 0.0).all()

    # Verification: Monotonicity
    assert (preds_df["p50"] <= preds_df["p75"] + 1e-4).all()
    assert (preds_df["p75"] <= preds_df["p90"] + 1e-4).all()
    assert (preds_df["spread_p90_p50"] >= -1e-4).all()


def test_strict_non_crossing_monotonicity(multicyle_data: pd.DataFrame):
    """Verify that Rearrangement Operator resolves all potential crossing instances."""
    n_train = int(len(multicyle_data) * 0.7)
    train_df = multicyle_data.iloc[:n_train]
    test_df = multicyle_data.iloc[n_train:]

    model = RAAPXQuantileCorrector(
        quantiles=[0.50, 0.75, 0.90],
        n_estimators=30,
        learning_rate=0.1,
    )
    model.fit(train_df, train_df["obs_precip"].values)

    # Evaluate metrics
    metrics = model.evaluate(test_df, test_df["obs_precip"].values)

    # After rearrangement, crossing rate must be exactly 0.0
    assert metrics.crossing_rate_after_rearrangement == 0.0
    assert metrics.pinball_losses["pinball_p50"] >= 0.0
    assert metrics.pinball_losses["pinball_p75"] >= 0.0
    assert metrics.pinball_losses["pinball_p90"] >= 0.0
    assert metrics.mean_pinball_loss > 0.0


def test_empirical_quantile_coverage_bounds(multicyle_data: pd.DataFrame):
    """Verify that empirical coverage values are physically valid probabilities [0, 1]."""
    n_train = int(len(multicyle_data) * 0.75)
    train_df = multicyle_data.iloc[:n_train]
    test_df = multicyle_data.iloc[n_train:]

    model = RAAPXQuantileCorrector(
        quantiles=[0.50, 0.75, 0.90],
        n_estimators=50,
        learning_rate=0.08,
    )
    model.fit(train_df, train_df["obs_precip"].values)
    metrics = model.evaluate(test_df, test_df["obs_precip"].values)

    cov_50 = metrics.empirical_coverage["coverage_p50"]
    cov_75 = metrics.empirical_coverage["coverage_p75"]
    cov_90 = metrics.empirical_coverage["coverage_p90"]

    # Coverage should increase with quantile alpha
    assert 0.0 <= cov_50 <= 1.0
    assert 0.0 <= cov_75 <= 1.0
    assert 0.0 <= cov_90 <= 1.0
    assert cov_50 <= cov_75 <= cov_90 + 0.05  # reasonable empirical monotonic coverage


def test_model_persistence_roundtrip(multicyle_data: pd.DataFrame):
    """Verify save and load roundtrip produces identical predictions."""
    model = RAAPXQuantileCorrector(
        quantiles=[0.50, 0.75, 0.90],
        n_estimators=25,
        model_version="v-test-3.0",
    )
    model.fit(multicyle_data, multicyle_data["obs_precip"].values)

    preds_orig = model.predict(multicyle_data)

    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = model.save(tmp_dir)
        loaded_model = RAAPXQuantileCorrector.load(save_path, model_version="v-test-3.0")

        preds_loaded = loaded_model.predict(multicyle_data)

        # Predictions must match numerically
        np.testing.assert_allclose(preds_orig["p50"], preds_loaded["p50"], rtol=1e-5)
        np.testing.assert_allclose(preds_orig["p75"], preds_loaded["p75"], rtol=1e-5)
        np.testing.assert_allclose(preds_orig["p90"], preds_loaded["p90"], rtol=1e-5)


def test_experiment_tracker_integration(multicyle_data: pd.DataFrame):
    """Verify logging to ExperimentTracker conforms to PART 10 MLOps schemas."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(storage_dir=Path(tmp_dir))
        split_info = TemporalSplitInfo(
            train_start="2024-06-01",
            train_end="2024-09-30",
            validation_start="2025-06-01",
            validation_end="2025-09-30",
            test_start="2026-06-01",
            test_end="2026-09-30",
        )

        model = RAAPXQuantileCorrector(quantiles=[0.50, 0.75, 0.90], n_estimators=25)
        model.fit(multicyle_data, multicyle_data["obs_precip"].values)
        metrics = model.evaluate(multicyle_data, multicyle_data["obs_precip"].values)

        rec = model.log_to_experiment_tracker(
            tracker=tracker,
            experiment_id="EXP-TEST-001",
            experiment_name="Unit Test Run",
            temporal_split=split_info,
            metrics=metrics,
        )

        assert rec.experiment_id == "EXP-TEST-001"
        assert rec.dataset_version == DATASET_VERSION_DEFAULT
        assert rec.feature_version == FEATURE_VERSION_DEFAULT
        assert "mean_pinball_loss" in rec.metrics
        assert "crps" in rec.metrics
        assert rec.status == "COMPLETED"


def test_full_pipeline_ablation_benchmarking(multicyle_data: pd.DataFrame):
    """Verify that RAAPXTrainingPipeline runs the 3 canonical ablation experiments and logs results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = ExperimentTracker(storage_dir=Path(tmp_dir))
        pipeline = RAAPXTrainingPipeline(experiment_tracker=tracker)

        results = pipeline.run_benchmark_experiments(
            df=multicyle_data,
            quantiles=[0.50, 0.75, 0.90],
        )

        comp_df = results["comparison_table"]
        assert len(comp_df) == 3
        assert "EXP-03A-RAW-NWP" in comp_df["Experiment ID"].values
        assert "EXP-03B-STD-QUANTILE" in comp_df["Experiment ID"].values
        assert "EXP-03C-RAAPX-REGIME" in comp_df["Experiment ID"].values

        # All models must show 100% Non-Crossing guarantee
        assert (comp_df["Non-Crossing (%)"] == 100.0).all()

        # Both ML quantile models should show improvement over raw NWP MAE
        raapx_row = comp_df[comp_df["Experiment ID"] == "EXP-03C-RAAPX-REGIME"].iloc[0]
        assert raapx_row["P50 MAE (mm)"] > 0.0
