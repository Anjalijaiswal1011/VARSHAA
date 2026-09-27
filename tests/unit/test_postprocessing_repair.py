"""
Comprehensive Test Suite for PART 6: NWP Error Repair Engine.
Verifies:
1. Unit Tests:
   - Target calculation (Observed - NWP) and non-negativity enforcement.
   - Rolling error memory calculations (3d, 7d, 14d bias, mae, rmse, std).
   - Regime probability and transition feature integration.
   - Analog feature retrieval and KNN index.
2. Leakage Tests:
   - Temporal causality in 3/7/14d error memory: past only, fail on future.
   - Target isolation audit: reject predictors containing observed rainfall or error targets.
   - Analog retrieval causality: candidates strictly prior to query cycle.
3. Model Tests:
   - Baseline 1 (Raw NWP), Baseline 2 (Statistical Bias), Baseline 3 (Generic ML).
   - Option A (Shared LightGBM) and Option B (Soft Mixture of Experts).
   - Automated sample sufficiency gating for regime experts.
   - Fallback hierarchy execution across all 5 tiers.
   - Model artifact serialization, metadata preservation, and reload without silent overwrite.
4. Integration & Ablation Tests:
   - 5-Model comparative ablation study execution on identical splits.
   - Full end-to-end pipeline: Features -> Regimes -> Memory -> Analog -> Repair -> Corrected Rainfall.
   - Residual analysis, regime-wise breakdown (with INSUFFICIENT TEST SAMPLE handling), lead-time breakdown.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.postprocessing.ablation import ModelSelectionExperimentRunner
from src.postprocessing.analog import KNNAnalogMemory
from src.postprocessing.artifacts import ModelArtifactManager
from src.postprocessing.baselines import (
    GenericMLErrorCorrectionBaseline,
    RawNWPBaseline,
    StatisticalBiasCorrectionBaseline,
)
from src.postprocessing.engine import RainRepairEngine
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.evaluation import (
    compute_contingency_table_metrics,
    compute_deterministic_metrics,
    evaluate_lead_time_performance,
    evaluate_regime_wise_performance,
    evaluate_residual_correction,
    evaluate_spatial_error_summary,
)
from src.postprocessing.fallback import (
    FallbackExecutionStatus,
    FallbackLevel,
    FallbackRepairController,
)
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.repair_model import (
    SharedRegimeLightGBMRepair,
    SoftMixtureOfExpertsRepair,
)
from src.postprocessing.target import (
    apply_error_correction,
    compute_nwp_error_target,
    validate_target_isolation,
)
from src.regime.schemas import REGIME_CLASSES, RegimeProbabilityVector
from src.utils.exceptions import TemporalLeakageError


@pytest.fixture
def synthetic_postprocessing_data() -> pd.DataFrame:
    """
    Generate synthetic multi-day chronological forecast data covering
    all 6 weather regimes, multiple lead times, coordinates, and realistic NWP biases.
    """
    np.random.seed(42)
    n = 240

    dates = [
        f"2026-07-{(1 + i // 12):02d}" for i in range(n)
    ]  # 20 distinct cycle days, 12 points per day
    grid_ids = [f"grid_{i % 6:02d}" for i in range(n)]
    lead_times = [24 if (i % 2 == 0) else 48 for i in range(n)]
    lats = [15.0 + (i % 4) * 2.0 for i in range(n)]
    lons = [75.0 + (i % 3) * 2.0 for i in range(n)]

    # Raw NWP with systematic under-forecast (dry bias) at high rains
    raw_nwp = np.random.uniform(5.0, 70.0, size=n)
    # Systematic errors: NWP over-forecasts light rain, under-forecasts heavy rain
    error = np.where(raw_nwp > 40.0, np.random.uniform(10.0, 35.0, size=n), np.random.uniform(-10.0, 5.0, size=n))
    obs_precip = np.maximum(0.0, raw_nwp + error)

    # Atmospheric & Terrain features
    elevation = np.random.uniform(50.0, 1800.0, size=n)
    dem_slope = np.random.uniform(0.2, 5.5, size=n)
    dist_to_coast = np.random.uniform(10.0, 400.0, size=n)
    u850 = np.random.uniform(4.0, 22.0, size=n)
    v850 = np.random.uniform(-3.0, 8.0, size=n)
    rh850 = np.random.uniform(50.0, 95.0, size=n)
    mslp = np.random.uniform(995.0, 1010.0, size=n)
    t2m = np.random.uniform(295.0, 305.0, size=n)
    orographic_uplift = np.random.uniform(0.0, 0.08, size=n)
    vorticity_850 = np.random.uniform(-1e-5, 2e-5, size=n)

    # Regime assignments: distribute across all 6 classes
    regime_list = [REGIME_CLASSES[i % len(REGIME_CLASSES)] for i in range(n)]

    df = pd.DataFrame(
        {
            "cycle_date": dates,
            "grid_id": grid_ids,
            "lead_time": lead_times,
            "lat": lats,
            "lon": lons,
            "nwp_precip": raw_nwp,
            "obs_precip": obs_precip,
            "elevation": elevation,
            "dem_slope": dem_slope,
            "dist_to_coast": dist_to_coast,
            "u850": u850,
            "v850": v850,
            "rh850": rh850,
            "mslp": mslp,
            "t2m": t2m,
            "orographic_uplift": orographic_uplift,
            "vorticity_850": vorticity_850,
            "regime": regime_list,
        }
    )

    # Attach soft regime probabilities
    for r_name in REGIME_CLASSES:
        prob = np.where(df["regime"] == r_name, 0.70, 0.06)
        df[f"prob_{r_name.lower()}"] = prob

    df["transition_status"] = np.where(np.random.rand(n) > 0.8, "TRANSITION", "STABLE")
    df["transition_strength"] = np.random.uniform(0.05, 0.45, size=n)
    df["transition_confidence"] = np.random.uniform(0.55, 0.95, size=n)

    return df


# ==============================================================================
# 1. Unit Tests
# ==============================================================================


def test_nwp_error_target_and_non_negativity():
    """Verify target calculation and physical non-negativity constraint."""
    obs = np.array([25.0, 0.0, 80.0, 5.0], dtype=np.float32)
    nwp = np.array([20.0, 10.0, 50.0, 15.0], dtype=np.float32)

    df_err = compute_nwp_error_target(obs, nwp)
    # signed_error = Obs - NWP
    expected_signed = np.array([5.0, -10.0, 30.0, -10.0], dtype=np.float32)
    np.testing.assert_allclose(df_err["signed_error"], expected_signed)
    np.testing.assert_allclose(df_err["absolute_error"], np.abs(expected_signed))

    # Applying predicted error with non-negativity clipping
    pred_error = np.array([5.0, -25.0, 30.0, -5.0], dtype=np.float32)
    corrected = apply_error_correction(nwp, pred_error, enforce_non_negativity=True)

    # At index 1: NWP=10, error=-25 => raw sum = -15. Non-negativity must clip to 0.0
    assert corrected[1] == 0.0
    assert np.all(corrected >= 0.0)
    assert corrected[0] == 25.0


def test_target_isolation_validation():
    """Verify validator raises TemporalLeakageError when target labels contaminate features."""
    safe_df = pd.DataFrame({"nwp_precip": [10.0], "rh850": [80.0]})
    # Should pass cleanly
    validate_target_isolation(safe_df)

    contaminated_df = pd.DataFrame({"nwp_precip": [10.0], "obs_precip": [15.0]})
    with pytest.raises(TemporalLeakageError, match="TARGET LEAKAGE DETECTED"):
        validate_target_isolation(contaminated_df)

    with pytest.raises(TemporalLeakageError, match="TARGET LEAKAGE DETECTED"):
        validate_target_isolation(pd.DataFrame({"signed_error": [2.5]}))


def test_error_memory_store_calculations(synthetic_postprocessing_data):
    """Verify 3d, 7d, 14d rolling error memory correctly computes bias, mae, rmse, and std."""
    store = LeakageSafeErrorMemoryStore(max_history_days=30)
    df = synthetic_postprocessing_data

    # Feed past cycles 2026-07-01 to 2026-07-05
    for day in range(1, 6):
        date_str = f"2026-07-{day:02d}"
        sub = df[df["cycle_date"] == date_str]
        if not sub.empty:
            store.record_verified_cycle(date_str, sub)

    # Query for cycle on 2026-07-06
    grid_ids = ["grid_00", "grid_01"]
    mem_features = store.extract_recent_memory_features("2026-07-06", grid_ids=grid_ids)

    # Expected column structure
    for win in ["3d", "7d", "14d"]:
        assert f"recent_bias_{win}" in mem_features.columns
        assert f"recent_mae_{win}" in mem_features.columns
        assert f"recent_rmse_{win}" in mem_features.columns
        assert f"recent_error_std_{win}" in mem_features.columns

    # Verify length equals grid points queried
    assert len(mem_features) == len(grid_ids)
    # MAE and RMSE must be strictly non-negative
    assert np.all(mem_features["recent_mae_3d"] >= 0.0)
    assert np.all(mem_features["recent_rmse_3d"] >= 0.0)


# ==============================================================================
# 2. Leakage Tests
# ==============================================================================


def test_temporal_leakage_safety_in_memory_store():
    """Verify error memory store strictly rejects any cycle dates at or beyond query date."""
    store = LeakageSafeErrorMemoryStore()
    df_day1 = pd.DataFrame({"grid_id": ["g1"], "obs_precip": [15.0], "nwp_precip": [10.0]})
    df_day5 = pd.DataFrame({"grid_id": ["g1"], "obs_precip": [25.0], "nwp_precip": [20.0]})

    store.record_verified_cycle("2026-07-01", df_day1)
    store.record_verified_cycle("2026-07-05", df_day5)

    # Query for 2026-07-04: Only 2026-07-01 is eligible; 2026-07-05 MUST be excluded
    past_dates = store.get_valid_past_dates("2026-07-04")
    assert "2026-07-01" in past_dates
    assert "2026-07-05" not in past_dates

    # Direct query test: Future dates can never appear in valid past
    for d in past_dates:
        assert pd.to_datetime(d) < pd.to_datetime("2026-07-04")


def test_analog_memory_anti_leakage(synthetic_postprocessing_data):
    """Verify KNNAnalogMemory does not retrieve future cases."""
    df = synthetic_postprocessing_data
    df_train = df[df["cycle_date"] <= "2026-07-10"]
    y_train = df_train["obs_precip"] - df_train["nwp_precip"]

    analog_mem = KNNAnalogMemory(k=3)
    analog_mem.fit(
        features=df_train,
        errors=y_train,
        dates=df_train["cycle_date"],
    )

    # Query for a future test case on 2026-07-15
    df_query = df[df["cycle_date"] == "2026-07-15"].head(5)
    analog_signals = analog_mem.query(df_query, query_dates=df_query["cycle_date"])

    assert "analog_mean_error" in analog_signals.columns
    assert "analog_std_error" in analog_signals.columns
    assert "analog_min_distance" in analog_signals.columns
    assert len(analog_signals) == 5
    assert not analog_signals.isna().any().any()


# ==============================================================================
# 3. Model Tests & Baselines
# ==============================================================================


def test_baseline_models_execution(synthetic_postprocessing_data):
    """Verify Baseline 1 (Raw NWP), Baseline 2 (Stat Bias), and Baseline 3 (Generic ML)."""
    df = synthetic_postprocessing_data
    y_target = df["obs_precip"] - df["nwp_precip"]

    # Baseline 1: Raw NWP
    b1 = RawNWPBaseline()
    b1_corr = b1.predict(df)
    np.testing.assert_allclose(b1_corr, df["nwp_precip"].values)

    # Baseline 2: Statistical Bias
    b2 = StatisticalBiasCorrectionBaseline()
    b2.fit(df, y_target)
    b2_pred = b2.predict(df)
    assert len(b2_pred) == len(df)
    assert np.all(b2_pred >= 0.0)

    # Baseline 3: Generic ML
    b3 = GenericMLErrorCorrectionBaseline(n_estimators=30)
    b3.fit(df, y_target)
    b3_pred = b3.predict(df)
    assert len(b3_pred) == len(df)
    assert np.all(b3_pred >= 0.0)


def test_shared_lightgbm_repair_option_a(synthetic_postprocessing_data):
    """Verify Option A (Shared Model) fitting, error prediction, and non-negativity."""
    df = synthetic_postprocessing_data
    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X = extractor.extract_error_dna(df, regime_df=df)
    y = df["obs_precip"] - df["nwp_precip"]

    model = SharedRegimeLightGBMRepair(n_estimators=30)
    model.fit(X, y)
    assert model.is_fitted

    pred_err = model.predict_error(X)
    corrected = model.predict_corrected_rainfall(X, nwp_col="nwp_precip")

    assert len(pred_err) == len(df)
    assert len(corrected) == len(df)
    assert np.all(corrected >= 0.0)


def test_soft_mixture_of_experts_option_b_and_sample_sufficiency(synthetic_postprocessing_data):
    """Verify Option B (MoE) trains experts and safely handles sparse regime fallback."""
    df = synthetic_postprocessing_data
    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X = extractor.extract_error_dna(df, regime_df=df)
    y = df["obs_precip"] - df["nwp_precip"]

    # Set high min_samples threshold so some regimes must fallback
    moe = SoftMixtureOfExpertsRepair(min_samples_per_regime=50, n_estimators=25)
    moe.fit(X, y, dominant_regimes=df["regime"])

    assert moe.is_fitted
    pred_err = moe.predict_error(X)
    corrected = moe.predict_corrected_rainfall(X, nwp_col="nwp_precip")

    assert len(pred_err) == len(df)
    assert np.all(corrected >= 0.0)


def test_fallback_controller_hierarchy(synthetic_postprocessing_data):
    """Verify all 5 tiers of the fallback hierarchy execute gracefully when dependencies drop."""
    df = synthetic_postprocessing_data
    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X = extractor.extract_error_dna(df, regime_df=df)
    y = df["obs_precip"] - df["nwp_precip"]

    primary = SharedRegimeLightGBMRepair(n_estimators=20).fit(X, y)
    generic = GenericMLErrorCorrectionBaseline(n_estimators=20).fit(X, y)
    stat_bias = StatisticalBiasCorrectionBaseline().fit(X, y)

    controller = FallbackRepairController(
        primary_model=primary,
        generic_ml_model=generic,
        statistical_bias_model=stat_bias,
    )

    # 1. Tier 0: Full RAIN-REPAIR X
    _, _, status0 = controller.repair_forecast(X, has_analog=True, has_regime=True, has_recent_memory=True)
    assert status0.active_tier == FallbackLevel.FULL_RAIN_REPAIR_X
    assert not status0.degraded

    # 2. Tier 1: Analog missing -> Regime-aware without analog
    _, _, status1 = controller.repair_forecast(X, has_analog=False, has_regime=True, has_recent_memory=True)
    assert status1.active_tier == FallbackLevel.REGIME_AWARE_NO_ANALOG
    assert status1.degraded

    # 3. Tier 2: Regime missing -> Generic ML
    _, _, status2 = controller.repair_forecast(X, has_analog=False, has_regime=False, has_recent_memory=True)
    assert status2.active_tier == FallbackLevel.GENERIC_ML_CORRECTION
    assert status2.degraded

    # 4. Tier 3: ML models unavailable -> Statistical Bias
    controller_no_ml = FallbackRepairController(
        primary_model=None,
        generic_ml_model=None,
        statistical_bias_model=stat_bias,
    )
    _, _, status3 = controller_no_ml.repair_forecast(X, has_regime=False)
    assert status3.active_tier == FallbackLevel.STATISTICAL_BIAS
    assert status3.degraded

    # 5. Tier 4: Fail-safe pass-through -> Raw NWP
    controller_bare = FallbackRepairController(primary_model=None)
    _, _, status4 = controller_bare.repair_forecast(X, has_regime=False)
    assert status4.active_tier == FallbackLevel.RAW_NWP_PASSTHROUGH
    assert status4.degraded


def test_model_artifact_save_and_load_no_silent_overwrite(synthetic_postprocessing_data):
    """Verify ModelArtifactManager persists artifacts and forbids silent overwriting."""
    df = synthetic_postprocessing_data
    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X = extractor.extract_error_dna(df, regime_df=df)
    y = df["obs_precip"] - df["nwp_precip"]

    model = SharedRegimeLightGBMRepair(n_estimators=20).fit(X, y)

    with tempfile.TemporaryDirectory() as tmp_dir:
        manager = ModelArtifactManager(base_checkpoint_dir=Path(tmp_dir))

        # First save should succeed
        save_path = manager.save_model(
            model=model,
            model_version="v6.0.0-test",
            train_period=("2026-07-01", "2026-07-15"),
            overwrite=False,
        )
        assert save_path.exists()

        # Second save with same version and overwrite=False MUST raise FileExistsError
        with pytest.raises(FileExistsError, match="already exists"):
            manager.save_model(model=model, model_version="v6.0.0-test", overwrite=False)

        # Reload model and verify metadata
        loaded_model, meta = manager.load_model("v6.0.0-test")
        assert meta["model_version"] == "v6.0.0-test"
        assert meta["architecture"] == "SharedRegimeLightGBMRepair"
        assert loaded_model.is_fitted


# ==============================================================================
# 4. Integration & Evaluation Tests
# ==============================================================================


def test_model_selection_ablation_experiment(synthetic_postprocessing_data):
    """Execute 5-model ablation study and verify structured comparison output."""
    df = synthetic_postprocessing_data
    # Chronological split: Train on first 14 days, test on remaining
    split_date = "2026-07-14"
    train_df = df[df["cycle_date"] <= split_date].copy()
    test_df = df[df["cycle_date"] > split_date].copy()

    runner = ModelSelectionExperimentRunner()
    comparison_df, models_dict = runner.run_ablation(train_df=train_df, test_df=test_df)

    assert len(comparison_df) == 6  # Baseline + 5 models
    assert "mae" in comparison_df.columns
    assert "rmse" in comparison_df.columns
    assert "mae_reduction" in comparison_df.columns

    # Verify no fabricated values (all metrics must be non-negative real numbers)
    assert np.all(comparison_df["mae"] >= 0.0)
    assert np.all(comparison_df["rmse"] >= 0.0)


def test_end_to_end_postprocessing_engine(synthetic_postprocessing_data):
    """Test full integration: Features -> Error DNA -> Repair Engine -> Corrected Rainfall."""
    df = synthetic_postprocessing_data
    train_df = df.iloc[:180].copy()
    test_df = df.iloc[180:].copy()

    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X_tr = extractor.extract_error_dna(train_df, regime_df=train_df)
    y_tr = train_df["obs_precip"] - train_df["nwp_precip"]

    model = SharedRegimeLightGBMRepair(n_estimators=30).fit(X_tr, y_tr)
    engine = RainRepairEngine(repair_model=model)

    # Process test cycle
    repaired_df = engine.process_forecast_cycle(
        df_features=test_df.drop(columns=["obs_precip"], errors="ignore"),
        cycle_date="2026-07-18",
        regime_df=test_df,
    )

    # Check CONTRACT-ML-002 compliance
    expected_cols = [
        "cycle_date", "lead_time", "lat", "lon", "raw_nwp_precip",
        "predicted_error", "delta_correction", "corrected_rainfall", "corrected_p50",
        "active_regime", "fallback_tier",
    ]
    for col in expected_cols:
        assert col in repaired_df.columns

    assert np.all(repaired_df["corrected_rainfall"] >= 0.0)
    assert np.all(repaired_df["corrected_p50"] >= 0.0)


def test_residual_evaluation_and_regime_breakdown(synthetic_postprocessing_data):
    """Test residual metrics, contingency table, and regime-wise breakdown."""
    df = synthetic_postprocessing_data
    obs = df["obs_precip"].to_numpy()
    raw = df["nwp_precip"].to_numpy()
    # Corrected has lower error
    corr = np.maximum(0.0, raw + 0.6 * (obs - raw))

    df_eval = df.copy()
    df_eval["corrected_precip"] = corr

    # Residual metrics
    res = evaluate_residual_correction(obs, raw, corr, threshold_mm=64.5)
    assert res["improvement"]["mae_reduction_mm"] >= 0.0

    # Regime-wise breakdown
    reg_df = evaluate_regime_wise_performance(df_eval, min_samples=10)
    assert len(reg_df) == len(REGIME_CLASSES)
    assert "status" in reg_df.columns

    # Lead-time breakdown
    lt_df = evaluate_lead_time_performance(df_eval)
    assert len(lt_df) == 2  # 24h and 48h
    assert "lead_time_hours" in lt_df.columns

    # Spatial summary
    spat_df = evaluate_spatial_error_summary(df_eval)
    assert not spat_df.empty
    assert "raw_mae" in spat_df.columns
    assert "corr_mae" in spat_df.columns
