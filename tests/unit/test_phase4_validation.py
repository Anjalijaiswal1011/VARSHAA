"""
Automated Test Suite for Phase 4 — Model Validation, Calibration & Error Analysis.
Verifies all 17 core requirements of Phase 4:
1. Locked test set definition (evaluation-only guarantee)
2. Four-way baseline comparison (Raw NWP, Bias-Corrected, Non-Regime ML, RAAP-X)
3. Quantile calibration (P50, P75, P90 coverage and coverage error)
4. Reliability analysis and post-hoc calibration investigation
5. Quantile crossings and 0% post-rearrangement verification
6. Regime-wise performance and probability entropy inspection
7. Lead-time error trajectories
8. Official IMD rainfall intensity stratification
9. Spatial terrain and elevation group analysis
10. Temporal drift and seasonal analysis
11. Top error stratification and failure mode post-mortem
12. Moving block bootstrap 95% confidence intervals
13. Operational failure guardrails (Input, Regime, Quantile, Data, Model, Shift)
14. Production readiness gates (Data, Model, Leakage, Calibration, Error, Integration)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.failure_modes import (
    FailureConditionAudit,
    FailureSeverity,
    OperationalFailureGuardrails,
)
from src.evaluation.phase4_validator import (
    IMD_RAINFALL_TIERS,
    Phase4ModelValidator,
    ProductionReadinessGateResult,
)
from src.mlops.experiments import ExperimentTracker, TemporalSplitInfo
from src.models.raapx_corrector import RAAPXQuantileCorrector
from src.models.raapx_pipeline import generate_synthetic_multicyle_data


@pytest.fixture
def fitted_raapx_system() -> Tuple[RAAPXQuantileCorrector, pd.DataFrame, pd.DataFrame]:
    """Provides a fitted RAAP-X model and strictly separated train and locked test datasets."""
    full_data = generate_synthetic_multicyle_data(n_days=90, n_grid_points=8, random_seed=42)

    # Strict chronological partition
    train_df = full_data[full_data["date"] <= "2024-12-31"].copy()
    test_df = full_data[full_data["date"] >= "2026-01-01"].copy()

    # If small sample partition, fallback to clean chronological split
    if len(test_df) < 20:
        n = len(full_data)
        train_df = full_data.iloc[: int(n * 0.7)].copy()
        test_df = full_data.iloc[int(n * 0.7) :].copy()

    model = RAAPXQuantileCorrector(
        quantiles=[0.50, 0.75, 0.90],
        n_estimators=40,
        learning_rate=0.08,
        random_seed=42,
    )
    y_train = train_df["obs_precip"].values
    model.fit(train_df, y_train)

    return model, train_df, test_df


def test_lock_test_set(fitted_raapx_system):
    """Verify test set locking and evaluation-only status."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)

    locked = validator.lock_test_set(test_df, test_start="2026-06-01", test_end="2026-09-30")

    assert validator.test_period_info is not None
    assert validator.test_period_info["status"] == "LOCKED_FOR_EVALUATION_ONLY"
    assert validator.test_period_info["sample_count"] == len(locked)
    assert len(locked) > 0


def test_baseline_comparison_four_way(fitted_raapx_system):
    """Verify 4-way baseline comparison emits numerical differences for exact test samples."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    comp_df = validator.compare_baselines()

    assert len(comp_df) == 4
    systems = comp_df["Model System"].values
    assert "A. Raw NWP" in systems
    assert "B. Statistical Bias Correction" in systems
    assert "C. Non-Regime ML Correction" in systems
    assert "D. RAAP-X (Regime-Aware)" in systems

    # RAAP-X should report numerical improvement over raw NWP
    raapx_row = comp_df[comp_df["Model System"] == "D. RAAP-X (Regime-Aware)"].iloc[0]
    nwp_row = comp_df[comp_df["Model System"] == "A. Raw NWP"].iloc[0]

    assert raapx_row["MAE (mm)"] < nwp_row["MAE (mm)"]
    assert raapx_row["Relative Gain vs NWP (%)"] > 0.0


def test_quantile_calibration_coverage(fitted_raapx_system):
    """Verify P50, P75, P90 nominal vs observed coverage calculation."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    calib_df = validator.evaluate_quantile_calibration()

    assert len(calib_df) == 3
    quantiles = calib_df["Quantile"].values
    assert "P50" in quantiles
    assert "P75" in quantiles
    assert "P90" in quantiles

    # Ensure coverage values are reported as percentages
    for _, row in calib_df.iterrows():
        assert "%" in row["Nominal Coverage"]
        assert "%" in row["Observed Coverage"]
        assert "%" in row["Coverage Error"]
        assert row["Pinball Loss"] >= 0.0


def test_reliability_and_calibration_investigation(fitted_raapx_system):
    """Verify reliability diagnostics (underprediction, spread-error correlation, recalibration indicators)."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    rel_dict = validator.evaluate_reliability_and_calibration_methods()

    assert "underprediction_rate_gt_p90" in rel_dict
    assert "overprediction_rate_lt_p50" in rel_dict
    assert "mean_uncertainty_spread_p90_p50" in rel_dict
    assert "conformal_calibration_indicated" in rel_dict
    assert isinstance(rel_dict["conformal_calibration_indicated"], bool)
    assert "calibration_assessment" in rel_dict


def test_quantile_crossing_verification(fitted_raapx_system):
    """Verify raw crossings are audited and post-rearrangement crossings are strictly 0%."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    cross_dict = validator.evaluate_quantile_crossings()

    assert "total_test_samples" in cross_dict
    assert "raw_crossing_rate_pct" in cross_dict
    assert "post_rearrangement_crossings" in cross_dict
    assert cross_dict["post_rearrangement_crossings"] == 0
    assert cross_dict["rearrangement_operator_verified"] is True


def test_regime_wise_performance_breakdown(fitted_raapx_system):
    """Verify regime-stratified evaluation across all active monsoon regimes."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    reg_df = validator.evaluate_regime_wise_errors()

    assert len(reg_df) > 0
    assert "Regime" in reg_df.columns
    assert "MAE (mm)" in reg_df.columns
    assert "P50 Loss" in reg_df.columns
    assert "Heavy Rain CSI" in reg_df.columns

    for _, row in reg_df.iterrows():
        assert row["Sample Count"] > 0
        assert row["MAE (mm)"] >= 0.0


def test_lead_time_error_trajectories(fitted_raapx_system):
    """Verify lead time trajectory (24h to 120h) calculation."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    lead_df = validator.evaluate_lead_time_errors()

    assert len(lead_df) >= 3
    assert "Lead Horizon" in lead_df.columns
    assert "Uncertainty Spread (mm)" in lead_df.columns

    # Spread should be positive
    for _, row in lead_df.iterrows():
        assert row["Uncertainty Spread (mm)"] >= 0.0


def test_rainfall_intensity_imd_tiers(fitted_raapx_system):
    """Verify evaluation across official IMD rainfall intensity tiers."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    tier_df = validator.evaluate_rainfall_intensity_tiers()

    assert len(tier_df) == len(IMD_RAINFALL_TIERS)
    categories = tier_df["Intensity Category"].values
    assert "No Rain" in categories
    assert "Light Rain" in categories
    assert "Moderate Rain" in categories
    assert "Heavy Rain" in categories
    assert "Very Heavy / Extreme Rain" in categories


def test_spatial_and_temporal_error_analysis(fitted_raapx_system):
    """Verify geographic elevation zones and monthly temporal progression."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    spat_df = validator.evaluate_spatial_distribution()
    assert len(spat_df) > 0
    assert "Geographical Zone" in spat_df.columns

    temp_df = validator.evaluate_temporal_patterns()
    assert len(temp_df) > 0
    assert "Monsoon Month" in temp_df.columns


def test_top_prediction_errors_stratification(fitted_raapx_system):
    """Verify error stratification table extracts the top prediction failures."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    top_err = validator.stratify_top_prediction_errors(top_n=5)

    assert len(top_err) == 5
    assert "Raw NWP (mm)" in top_err.columns
    assert "Observed Rain (mm)" in top_err.columns
    assert "Absolute Error (mm)" in top_err.columns
    # Top errors must be sorted descending by error
    errors = top_err["Absolute Error (mm)"].values
    assert errors[0] >= errors[-1]


def test_block_bootstrap_confidence_intervals(fitted_raapx_system):
    """Verify moving block bootstrap generates 95% Confidence Intervals for MAE and losses."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    ci_dict = validator.compute_block_bootstrap_confidence_intervals(n_bootstraps=50, block_size=3)

    assert "p50_mae" in ci_dict
    assert "p50_pinball_loss" in ci_dict
    assert "p90_pinball_loss" in ci_dict

    mae_ci = ci_dict["p50_mae"]
    assert mae_ci["ci_lower_95"] <= mae_ci["mean"] <= mae_ci["ci_upper_95"]


def test_operational_failure_guardrails(fitted_raapx_system):
    """Verify all 6 operational failure condition audits."""
    model, train_df, test_df = fitted_raapx_system

    # 1. Input Failure (missing nwp_precip)
    corrupted_df = test_df.drop(columns=["nwp_precip"])
    input_audit = OperationalFailureGuardrails.check_input_failure(
        corrupted_df,
        required_features=["nwp_precip", "elevation", "slope"],
    )
    assert input_audit.is_triggered is True
    assert input_audit.severity == FailureSeverity.CRITICAL

    # 2. Regime Failure (sum != 1.0)
    invalid_probs = {"active": 0.2, "break": 0.2}  # sum = 0.4
    regime_audit = OperationalFailureGuardrails.check_regime_failure(invalid_probs)
    assert regime_audit.is_triggered is True

    # 3. Quantile Failure (crossing)
    p50_cross = np.array([50.0, 10.0])
    p75_cross = np.array([30.0, 20.0])  # sample 0 has 50 > 30 crossing
    p90_cross = np.array([70.0, 40.0])
    quantile_audit = OperationalFailureGuardrails.check_quantile_failure(p50_cross, p75_cross, p90_cross)
    assert quantile_audit.is_triggered is True
    assert quantile_audit.diagnostic_evidence["crossings_p50_gt_p75"] == 1

    # 4. Data Failure (out of domain coordinates)
    bad_lats = np.array([45.0, 20.0])  # 45N is outside 6-38N
    bad_lons = np.array([75.0, 75.0])
    data_audit = OperationalFailureGuardrails.check_data_failure(bad_lats, bad_lons, dates=["2026-07-01"])
    assert data_audit.is_triggered is True
    assert data_audit.diagnostic_evidence["out_of_bounds_lat_count"] == 1

    # 5. Model Failure (nonexistent checkpoint)
    model_audit = OperationalFailureGuardrails.check_model_failure(Path("nonexistent/checkpoint.joblib"))
    assert model_audit.is_triggered is True
    assert model_audit.severity == FailureSeverity.CRITICAL

    # 6. Distribution Shift Check
    shift_audit = OperationalFailureGuardrails.check_distribution_shift(
        train_df,
        test_df,
        features=["nwp_precip", "elevation"],
    )
    assert shift_audit.condition_name == "Distribution_Shift"


def test_production_readiness_gates(fitted_raapx_system):
    """Verify all 6 operational production gates and overall model status."""
    model, train_df, test_df = fitted_raapx_system
    validator = Phase4ModelValidator(trained_model=model)
    validator.lock_test_set(test_df)

    gates = validator.audit_production_readiness_gates()

    assert "DATA_GATE" in gates
    assert "MODEL_GATE" in gates
    assert "LEAKAGE_GATE" in gates
    assert "CALIBRATION_GATE" in gates
    assert "ERROR_GATE" in gates
    assert "INTEGRATION_GATE" in gates

    for g_name, gate in gates.items():
        assert isinstance(gate, ProductionReadinessGateResult)
        assert len(gate.status_summary) > 5

    full_results = validator.run_full_phase4_validation(test_df)
    assert full_results["final_model_status"] in ["VALIDATED", "PARTIALLY VALIDATED"]
