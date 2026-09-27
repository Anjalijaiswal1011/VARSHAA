"""
Unit and Integration Test Suite for PART 7: Probabilistic Post-Processing,
Multi-Quantile Regression, Extreme Value Theory (EVT-GPD), and IMD Risk Alerts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.postprocessing.constraints import (
    enforce_probability_monotonicity,
    enforce_quantile_monotonicity,
    validate_physical_realizability,
)
from src.postprocessing.engine import RainRepairEngine
from src.postprocessing.error_dna import NWPErrorDNAExtractor
from src.postprocessing.evaluation import (
    compute_contingency_table_metrics,
    compute_deterministic_metrics,
)
from src.postprocessing.evt import EVTGPDTailModel
from src.postprocessing.probabilistic_eval import (
    compute_brier_score,
    compute_crps_from_quantiles,
    evaluate_quantile_calibration,
)
from src.postprocessing.quantiles import (
    MultiQuantileRegressor,
    compute_pinball_loss,
)
from src.postprocessing.repair_model import SharedRegimeLightGBMRepair
from src.postprocessing.risk import (
    CalibratedRiskEngine,
    IMDAlertLevel,
    assign_imd_alert_level,
)
from src.postprocessing.thresholds import ExtremeThresholdClassifier
from src.regime.schemas import REGIME_CLASSES
from src.utils.exceptions import PhysicalConstraintViolationError


@pytest.fixture
def synthetic_probabilistic_data() -> pd.DataFrame:
    """Generates synthetic dataset for probabilistic and EVT tail testing."""
    np.random.seed(42)
    n = 200

    dates = [f"2026-07-{(1 + i // 10):02d}" for i in range(n)]
    lead_times = [24 if i % 2 == 0 else 48 for i in range(n)]
    lats = [18.0 + (i % 5) * 1.5 for i in range(n)]
    lons = [73.0 + (i % 4) * 1.5 for i in range(n)]

    # Raw NWP
    raw_nwp = np.random.uniform(5.0, 80.0, size=n)
    # Target observed rainfall with extreme convective deluges
    obs_precip = np.random.exponential(scale=20.0, size=n)
    # Inject rare extreme rainfall cases (> 64.5 mm and > 115.6 mm)
    obs_precip[0:15] = np.random.uniform(70.0, 110.0, size=15)
    obs_precip[15:25] = np.random.uniform(120.0, 220.0, size=10)

    u850 = np.random.uniform(5.0, 20.0, size=n)
    v850 = np.random.uniform(-4.0, 8.0, size=n)
    rh850 = np.random.uniform(55.0, 95.0, size=n)
    mslp = np.random.uniform(995.0, 1008.0, size=n)
    t2m = np.random.uniform(296.0, 304.0, size=n)
    elevation = np.random.uniform(20.0, 1500.0, size=n)
    dem_slope = np.random.uniform(0.2, 5.0, size=n)
    dist_to_coast = np.random.uniform(10.0, 300.0, size=n)
    orographic_uplift = np.random.uniform(0.0, 0.08, size=n)
    vorticity_850 = np.random.uniform(-1e-5, 2e-5, size=n)

    df = pd.DataFrame(
        {
            "cycle_date": dates,
            "lead_time": lead_times,
            "lat": lats,
            "lon": lons,
            "nwp_precip": raw_nwp,
            "obs_precip": obs_precip,
            "u850": u850,
            "v850": v850,
            "rh850": rh850,
            "mslp": mslp,
            "t2m": t2m,
            "elevation": elevation,
            "dem_slope": dem_slope,
            "dist_to_coast": dist_to_coast,
            "orographic_uplift": orographic_uplift,
            "vorticity_850": vorticity_850,
        }
    )

    for r_name in REGIME_CLASSES:
        df[f"prob_{r_name.lower()}"] = 1.0 / len(REGIME_CLASSES)

    return df


def test_rearrangement_operator_guarantees_monotonicity():
    """Verify Rearrangement Operator sorts scrambled quantiles and enforces non-negativity."""
    raw_quantiles = {
        0.10: np.array([25.0, -5.0, 10.0], dtype=np.float32),  # Contains negative
        0.50: np.array([15.0, 12.0, 8.0], dtype=np.float32),   # Crosses Q10
        0.75: np.array([30.0, 18.0, 12.0], dtype=np.float32),
        0.90: np.array([28.0, 22.0, 15.0], dtype=np.float32),  # Crosses Q75
        0.95: np.array([45.0, 30.0, 20.0], dtype=np.float32),
    }

    rearranged = enforce_quantile_monotonicity(raw_quantiles, enforce_non_negativity=True)

    # Check non-negativity
    for alpha, vals in rearranged.items():
        assert np.all(vals >= 0.0)

    # Check strict non-crossing: Q10 <= Q50 <= Q75 <= Q90 <= Q95
    assert np.all(rearranged[0.10] <= rearranged[0.50])
    assert np.all(rearranged[0.50] <= rearranged[0.75])
    assert np.all(rearranged[0.75] <= rearranged[0.90])
    assert np.all(rearranged[0.90] <= rearranged[0.95])


def test_probability_monotonicity():
    """Verify cumulative upper bounds for heavy, very heavy, and extreme rainfall."""
    # Inverted probabilities
    p_heavy = np.array([0.30, 0.50])
    p_vheavy = np.array([0.45, 0.40])  # Index 0 exceeds heavy (0.45 > 0.30)
    p_extreme = np.array([0.55, 0.10]) # Index 0 exceeds very heavy (0.55 > 0.45)

    adj_h, adj_vh, adj_ext = enforce_probability_monotonicity(p_heavy, p_vheavy, p_extreme)

    assert np.all(adj_ext <= adj_vh)
    assert np.all(adj_vh <= adj_h)
    assert np.all(adj_h <= 1.0)
    assert np.all(adj_ext >= 0.0)
    # At index 0, both vheavy and extreme must be capped at heavy (0.30)
    assert adj_vh[0] == 0.30
    assert adj_ext[0] == 0.30


def test_multi_quantile_regressor_fit_and_pinball_loss(synthetic_probabilistic_data):
    """Verify MultiQuantileRegressor fits and produces monotonic plumes with valid pinball loss."""
    df = synthetic_probabilistic_data
    X = df.drop(columns=["obs_precip"])
    y = df["obs_precip"]

    mqr = MultiQuantileRegressor(n_estimators=25)
    mqr.fit(X, y)
    assert mqr.is_fitted

    preds = mqr.predict_quantiles(X, enforce_monotonicity=True)
    expected_keys = ["corrected_p10", "corrected_p50", "corrected_p75", "corrected_p90", "corrected_p95"]
    for k in expected_keys:
        assert k in preds
        assert len(preds[k]) == len(df)
        assert np.all(preds[k] >= 0.0)

    # Monotonicity check
    assert np.all(preds["corrected_p10"] <= preds["corrected_p50"])
    assert np.all(preds["corrected_p50"] <= preds["corrected_p75"])
    assert np.all(preds["corrected_p75"] <= preds["corrected_p90"])
    assert np.all(preds["corrected_p90"] <= preds["corrected_p95"])

    losses = mqr.evaluate_pinball_losses(X, y)
    for alpha in [10, 50, 75, 90, 95]:
        assert f"pinball_loss_p{alpha}" in losses
        assert losses[f"pinball_loss_p{alpha}"] >= 0.0


def test_extreme_threshold_classifier(synthetic_probabilistic_data):
    """Verify ExtremeThresholdClassifier estimates calibrated exceedance probabilities."""
    df = synthetic_probabilistic_data
    X = df.drop(columns=["obs_precip"])
    y = df["obs_precip"]

    etc = ExtremeThresholdClassifier(n_estimators=25)
    etc.fit(X, y)
    assert etc.is_fitted

    probs = etc.predict_probabilities(X, enforce_monotonicity=True)
    assert "prob_heavy_rain" in probs
    assert "prob_very_heavy_rain" in probs
    assert "prob_extreme_rain" in probs

    # Bounds & Monotonicity
    assert np.all(probs["prob_extreme_rain"] <= probs["prob_very_heavy_rain"])
    assert np.all(probs["prob_very_heavy_rain"] <= probs["prob_heavy_rain"])
    assert np.all(probs["prob_heavy_rain"] <= 1.0)
    assert np.all(probs["prob_extreme_rain"] >= 0.0)


def test_evt_gpd_peaks_over_threshold(synthetic_probabilistic_data):
    """Verify EVT-GPD model fitting, tail survival probabilities, and return level calculation."""
    df = synthetic_probabilistic_data
    y = df["obs_precip"]

    evt = EVTGPDTailModel(threshold_mm=64.5)
    evt.fit(y)
    assert evt.is_fitted
    assert evt.scale_sigma > 0.0
    assert -0.2 <= evt.shape_xi <= 0.6

    # Probability of exceeding 100mm and 200mm
    p_100 = evt.compute_exceedance_probability(100.0)
    p_200 = evt.compute_exceedance_probability(200.0)
    assert 0.0 <= p_200 <= p_100 <= 1.0

    # Rare return level (P99)
    p99_val = evt.compute_return_level(0.99)
    assert p99_val >= 64.5


def test_imd_alert_level_assignment():
    """Verify IMD 4-tier alert level classification rules."""
    # Green: low probabilities
    lvl1, _ = assign_imd_alert_level(prob_heavy=0.10, prob_very_heavy=0.02, prob_extreme=0.0, p90_rain=25.0)
    assert lvl1 == IMDAlertLevel.GREEN

    # Yellow: moderate heavy rain
    lvl2, _ = assign_imd_alert_level(prob_heavy=0.35, prob_very_heavy=0.05, prob_extreme=0.0, p90_rain=68.0)
    assert lvl2 == IMDAlertLevel.YELLOW

    # Orange: very heavy rain likely
    lvl3, _ = assign_imd_alert_level(prob_heavy=0.75, prob_very_heavy=0.45, prob_extreme=0.05, p90_rain=130.0)
    assert lvl3 == IMDAlertLevel.ORANGE

    # Red: extreme rain danger
    lvl4, _ = assign_imd_alert_level(prob_heavy=0.95, prob_very_heavy=0.80, prob_extreme=0.40, p90_rain=220.0)
    assert lvl4 == IMDAlertLevel.RED


def test_end_to_end_calibrated_risk_engine(synthetic_probabilistic_data):
    """Verify full Part 6 + Part 7 workflow emitting CONTRACT-ML-002 calibrated risk matrix."""
    df = synthetic_probabilistic_data
    X = df.drop(columns=["obs_precip"])
    y = df["obs_precip"]

    # 1. Part 6 Deterministic Engine
    extractor = NWPErrorDNAExtractor(include_recent_memory=False, include_analog=False)
    X_dna = extractor.extract_error_dna(X)
    repair_model = SharedRegimeLightGBMRepair(n_estimators=20).fit(X_dna, y - df["nwp_precip"])
    part6_engine = RainRepairEngine(repair_model=repair_model)

    df_repaired = part6_engine.process_forecast_cycle(
        df_features=X,
        cycle_date="2026-07-15",
    )

    # 2. Part 7 Calibrated Risk Engine
    risk_engine = CalibratedRiskEngine()
    risk_engine.fit(X_dna, y)

    final_risk_df = risk_engine.generate_risk_output(df_repaired, X_dna)

    # Verify CONTRACT-ML-002 schema completeness
    contract_cols = [
        "cycle_date", "lead_time", "lat", "lon", "raw_nwp_precip",
        "corrected_p10", "corrected_p50", "corrected_p75", "corrected_p90", "corrected_p95",
        "prob_heavy_rain", "prob_very_heavy_rain", "prob_extreme_rain",
        "imd_alert_level", "imd_alert_advisory", "evt_tail_p99",
    ]
    for col in contract_cols:
        assert col in final_risk_df.columns

    # Verify physical invariants pass
    assert validate_physical_realizability(final_risk_df, strict=True)


def test_probabilistic_eval_metrics(synthetic_probabilistic_data):
    """Verify CRPS, Brier Score, and Quantile Reliability evaluation functions."""
    df = synthetic_probabilistic_data
    y = df["obs_precip"].to_numpy()

    quantiles_dict = {
        0.10: np.maximum(0.0, y * 0.4),
        0.50: y * 0.9,
        0.75: y * 1.2,
        0.90: y * 1.6,
        0.95: y * 2.0,
    }

    # CRPS
    crps = compute_crps_from_quantiles(y, quantiles_dict)
    assert crps >= 0.0

    # Brier Score
    probs = np.where(y > 64.5, 0.8, 0.1)
    bs_dict = compute_brier_score(y, probs, threshold_mm=64.5)
    assert "brier_score" in bs_dict
    assert 0.0 <= bs_dict["brier_score"] <= 1.0

    # Quantile Reliability
    rel_df = evaluate_quantile_calibration(y, quantiles_dict)
    assert len(rel_df) == 5
    assert "empirical_coverage" in rel_df.columns


def test_probabilistic_ablation_a0_to_a7(synthetic_probabilistic_data):
    """Verify execution of the 8-stage ablation experiment from A0 to A7."""
    from src.postprocessing.probabilistic_ablation import ProbabilisticAblationRunner

    df = synthetic_probabilistic_data
    # 3-way chronological split: Train (12 days), Val (3 days), Test (5 days)
    train_df = df[df["cycle_date"] <= "2026-07-12"].copy()
    val_df = df[(df["cycle_date"] > "2026-07-12") & (df["cycle_date"] <= "2026-07-15")].copy()
    test_df = df[df["cycle_date"] > "2026-07-15"].copy()

    runner = ProbabilisticAblationRunner()
    ablation_df = runner.run_full_ablation(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        heavy_threshold=64.5,
    )

    assert len(ablation_df) == 8  # A0 to A7
    assert set(ablation_df["stage_id"]) == {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7"}
    assert "mae" in ablation_df.columns
    assert "rmse" in ablation_df.columns
    assert "crps" in ablation_df.columns
    assert "brier_score" in ablation_df.columns


def test_two_stage_calibration_leakage_safety(synthetic_probabilistic_data):
    """Verify threshold classifier fits on train and calibrates on val with zero test leakage."""
    df = synthetic_probabilistic_data
    train_df = df.iloc[:120]
    val_df = df.iloc[120:160]
    test_df = df.iloc[160:]

    X_tr = train_df.drop(columns=["obs_precip"])
    y_tr = train_df["obs_precip"]
    X_v = val_df.drop(columns=["obs_precip"])
    y_v = val_df["obs_precip"]
    X_te = test_df.drop(columns=["obs_precip"])

    etc = ExtremeThresholdClassifier(n_estimators=20)
    etc.fit(X_train=X_tr, y_train_precip=y_tr, X_val=X_v, y_val_precip=y_v)

    assert etc.is_fitted
    probs_test = etc.predict_probabilities(X_te, enforce_monotonicity=True)
    assert len(probs_test["prob_heavy_rain"]) == len(test_df)
    assert np.all(probs_test["prob_heavy_rain"] >= 0.0)
    assert np.all(probs_test["prob_heavy_rain"] <= 1.0)


def test_evt_unavailable_fallback_behavior():
    """Verify that when extreme samples are sparse (<15), EVT marks unavailable and falls back cleanly."""
    sparse_precip = np.array([5.0, 10.0, 12.0, 15.0, 20.0, 70.0, 85.0])  # Only 2 exceedances > 64.5
    evt = EVTGPDTailModel(threshold_mm=64.5, min_exceedances=15)
    evt.fit(sparse_precip)

    assert evt.evt_status == "unavailable"
    assert not evt.is_fitted

    # Downstream should still return valid bounded return level without throwing exception
    ret_val = evt.compute_return_level(0.99)
    assert ret_val > 0.0

    exc_prob = evt.compute_exceedance_probability(100.0)
    assert 0.0 <= exc_prob <= 1.0


def test_extreme_event_evaluation(synthetic_probabilistic_data):
    """Verify evaluation on known extreme rainfall subset (>64.5 mm)."""
    df = synthetic_probabilistic_data
    extreme_subset = df[df["obs_precip"] >= 64.5]
    assert len(extreme_subset) >= 10

    # Contingency evaluation
    contingency = compute_contingency_table_metrics(
        observed=df["obs_precip"].to_numpy(),
        predicted=df["nwp_precip"].to_numpy(),
        threshold_mm=64.5,
    )
    assert "pod" in contingency
    assert "far" in contingency
    assert "csi" in contingency
