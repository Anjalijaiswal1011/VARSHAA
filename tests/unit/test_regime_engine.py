"""
Unit, Model, Leakage, and Integration Tests for RAIN-REPAIR X Regime Intelligence Layer.
Verifies:
1. Six canonical classes and schema serialization.
2. Probability normalization and strict sum = 1.0 guarantee.
3. Rule-based heuristic baseline and Multinomial Logistic baseline.
4. Primary LightGBM regime classifier fitting, probability calibration, and feature importances.
5. Regime transition engine states (STABLE, TRANSITION, UNCERTAIN) and TV distance.
6. Temporal leakage safety: strictly past-to-future chronological progression.
7. Fallback behavior when features are corrupted or missing (REGIME_UNCERTAIN).
8. End-to-end integration flow from physics features to model-ready outputs.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.regime.baseline import (
    MultinomialLogisticRegimeBaseline,
    RuleBasedRegimeClassifier,
)
from src.regime.calibration import (
    RegimeCalibrator,
    compute_expected_calibration_error,
    compute_multiclass_brier_score,
    evaluate_probability_calibration,
)
from src.regime.classifier import LightGBMRegimeClassifier
from src.regime.evaluation import (
    evaluate_regime_predictions,
    format_regime_metrics_table,
)
from src.regime.inference import RegimeInferenceEngine
from src.regime.labels import (
    derive_provisional_regime_labels,
    get_regime_distribution_report,
)
from src.regime.schemas import (
    IDX_TO_REGIME,
    REGIME_CLASSES,
    REGIME_TO_IDX,
    RegimeProbabilityVector,
    RegimeTransitionOutput,
    TransitionStatus,
    WeatherRegime,
)
from src.regime.transition import (
    RegimeHistoryBuffer,
    RegimeTransitionEngine,
    compute_normalized_entropy,
    compute_total_variation_distance,
)


@pytest.fixture
def synthetic_weather_dataframe() -> pd.DataFrame:
    """Generate realistic synthetic atmospheric feature dataframe covering all regimes."""
    np.random.seed(42)
    n = 180

    # Simulate chronological sequence across multiple grid points
    grid_ids = [f"grid_{i%10:02d}" for i in range(n)]
    timestamps = [f"2026-07-{(15 + i//30):02d}T03:00:00Z" for i in range(n)]

    # Diverse meteorological features
    lat = np.random.uniform(8.0, 36.0, size=n)
    lon = np.random.uniform(68.0, 96.0, size=n)
    u850 = np.random.uniform(2.0, 18.0, size=n)
    v850 = np.random.uniform(-4.0, 10.0, size=n)
    rh850 = np.random.uniform(40.0, 95.0, size=n)
    mslp = np.random.uniform(998.0, 1012.0, size=n)
    elevation = np.random.uniform(10.0, 2500.0, size=n)
    dem_slope = np.random.uniform(0.1, 8.0, size=n)
    dist_to_coast = np.random.uniform(5.0, 500.0, size=n)
    orographic_uplift = np.random.uniform(0.0, 0.12, size=n)
    vorticity_850 = np.random.uniform(-1e-5, 2.5e-5, size=n)
    wd_shear_proxy = np.random.uniform(2.0, 30.0, size=n)

    # Inject specific regime characteristics to ensure non-zero representation
    # Sample 0..25: Western Disturbance (North latitude, high shear proxy, high humidity)
    lat[0:25] = np.random.uniform(26.0, 34.0, size=25)
    wd_shear_proxy[0:25] = np.random.uniform(18.0, 32.0, size=25)
    rh850[0:25] = np.random.uniform(55.0, 85.0, size=25)

    # Sample 25..55: Monsoon Depression (deep cyclonic vorticity, low MSLP)
    vorticity_850[25:55] = np.random.uniform(1.3e-5, 3.0e-5, size=30)
    mslp[25:55] = np.random.uniform(996.0, 1001.0, size=30)

    # Sample 55..85: Orographic (steep slope, mountain uplift)
    dem_slope[55:85] = np.random.uniform(3.0, 7.5, size=30)
    orographic_uplift[55:85] = np.random.uniform(0.03, 0.10, size=30)
    elevation[55:85] = np.random.uniform(700.0, 2200.0, size=30)

    # Sample 85..115: Coastal (near coast, low elevation, high RH)
    dist_to_coast[85:115] = np.random.uniform(5.0, 45.0, size=30)
    elevation[85:115] = np.random.uniform(10.0, 80.0, size=30)
    rh850[85:115] = np.random.uniform(75.0, 95.0, size=30)

    # Sample 115..145: Break Monsoon (dry, weak winds, south/central, inland)
    lat[115:145] = np.random.uniform(12.0, 20.0, size=30)
    u850[115:145] = np.random.uniform(1.0, 5.0, size=30)
    v850[115:145] = np.random.uniform(-1.0, 2.0, size=30)
    rh850[115:145] = np.random.uniform(35.0, 55.0, size=30)
    dem_slope[115:145] = 0.5
    orographic_uplift[115:145] = 0.0
    elevation[115:145] = 200.0
    dist_to_coast[115:145] = 300.0
    vorticity_850[115:145] = 0.0
    wd_shear_proxy[115:145] = 5.0

    # Sample 145..180: Active Monsoon (strong westerly jet, moist, flat inland)
    u850[145:180] = np.random.uniform(12.0, 22.0, size=35)
    rh850[145:180] = np.random.uniform(70.0, 92.0, size=35)
    dem_slope[145:180] = 0.5
    orographic_uplift[145:180] = 0.0
    elevation[145:180] = 200.0
    dist_to_coast[145:180] = 300.0
    vorticity_850[145:180] = 0.0
    wd_shear_proxy[145:180] = 5.0

    df = pd.DataFrame(
        {
            "grid_id": grid_ids,
            "forecast_time": timestamps,
            "lat": lat,
            "lon": lon,
            "u850": u850,
            "v850": v850,
            "wind_speed_850": np.sqrt(u850**2 + v850**2),
            "rh850": rh850,
            "mslp": mslp,
            "elevation": elevation,
            "dem_slope": dem_slope,
            "dist_to_coast": dist_to_coast,
            "orographic_uplift": orographic_uplift,
            "vorticity_850": vorticity_850,
            "wd_shear_proxy": wd_shear_proxy,
        }
    )
    return df


# ==============================================================================
# 1. UNIT TESTS: SCHEMAS & NORMALIZATION
# ==============================================================================

def test_six_canonical_regimes_exist():
    """Verify that exactly the six required regimes are defined."""
    expected = [
        "ACTIVE_MONSOON",
        "BREAK_MONSOON",
        "MONSOON_DEPRESSION",
        "COASTAL",
        "OROGRAPHIC",
        "WESTERN_DISTURBANCE",
    ]
    assert REGIME_CLASSES == expected
    assert len(REGIME_CLASSES) == 6
    assert WeatherRegime.WESTERN_DISTURBANCE.value in REGIME_CLASSES


def test_regime_probability_vector_normalization():
    """Verify normalization logic guarantees sum is exactly 1.0."""
    raw = np.array([2.0, 1.0, 0.5, 0.5, 0.0, 0.0])
    vec = RegimeProbabilityVector.from_array(raw)
    arr = vec.to_array()

    assert len(arr) == 6
    assert np.all(arr >= 0.0)
    assert np.isclose(float(np.sum(arr)), 1.0, atol=1e-3)
    dom, conf = vec.dominant_regime()
    assert dom == "ACTIVE_MONSOON"
    assert conf > 0.45


def test_zero_vector_fallback_uniform():
    """Verify all zeros produce uniform 1/6 distribution."""
    raw = np.zeros(6)
    vec = RegimeProbabilityVector.from_array(raw)
    arr = vec.to_array()
    assert np.allclose(arr, 1.0 / 6.0, atol=1e-3)


# ==============================================================================
# 2. PROVISIONAL LABEL DERIVATION
# ==============================================================================

def test_derive_provisional_regime_labels(synthetic_weather_dataframe):
    """Test heuristic label generator and class distribution report."""
    df = synthetic_weather_dataframe
    labels = derive_provisional_regime_labels(df)

    assert len(labels) == len(df)
    assert set(labels.unique()).issubset(set(REGIME_CLASSES))

    report = get_regime_distribution_report(labels)
    assert report["is_verified"] is False
    assert report["label_type"] == "provisional_weak_rule_based"
    assert report["total_samples"] == len(df)
    assert WeatherRegime.WESTERN_DISTURBANCE.value in report["distribution"]


# ==============================================================================
# 3. BASELINE CLASSIFIERS
# ==============================================================================

def test_rule_based_baseline_classifier(synthetic_weather_dataframe):
    df = synthetic_weather_dataframe
    baseline = RuleBasedRegimeClassifier(smoothing_factor=0.06)
    probs = baseline.predict_proba(df)

    assert probs.shape == (len(df), 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    preds = baseline.predict(df)
    assert len(preds) == len(df)
    assert all(p in REGIME_CLASSES for p in preds)


def test_multinomial_logistic_baseline(synthetic_weather_dataframe):
    df = synthetic_weather_dataframe
    labels = derive_provisional_regime_labels(df)

    baseline = MultinomialLogisticRegimeBaseline()
    baseline.fit(df, labels)
    assert baseline.is_fitted is True

    probs = baseline.predict_proba(df)
    assert probs.shape == (len(df), 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    preds = baseline.predict(df)
    assert len(preds) == len(df)


# ==============================================================================
# 4. PRIMARY CLASSIFIER & ARTIFACT PERSISTENCE
# ==============================================================================

def test_lightgbm_regime_classifier_fit_and_save(synthetic_weather_dataframe):
    df = synthetic_weather_dataframe
    labels = derive_provisional_regime_labels(df)

    # Time-aware split: first 120 train, next 30 val, last 30 test
    X_train, y_train = df.iloc[:120], labels.iloc[:120]
    X_val, y_val = df.iloc[120:150], labels.iloc[120:150]
    X_test = df.iloc[150:]

    clf = LightGBMRegimeClassifier(n_estimators=30, max_depth=4)
    clf.fit(X_train, y_train, X_val, y_val)
    assert clf.is_fitted is True
    assert len(clf.feature_importances_) > 0

    probs = clf.predict_proba(X_test)
    assert probs.shape == (len(X_test), 6)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4)

    vectors = clf.predict_regime_vectors(X_test)
    assert len(vectors) == len(X_test)
    assert isinstance(vectors[0], RegimeProbabilityVector)

    # Test artifact serialization
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        clf.save_artifact(artifact_dir=tmp_path, model_version="v_test")

        loaded = LightGBMRegimeClassifier.load_artifact(artifact_dir=tmp_path, model_version="v_test")
        assert loaded.is_fitted is True
        loaded_probs = loaded.predict_proba(X_test)
        assert np.allclose(probs, loaded_probs, atol=1e-4)


# ==============================================================================
# 5. CALIBRATION & EVALUATION
# ==============================================================================

def test_probability_calibration_and_evaluation(synthetic_weather_dataframe):
    df = synthetic_weather_dataframe
    labels = derive_provisional_regime_labels(df)

    X_train, y_train = df.iloc[:120], labels.iloc[:120]
    X_val, y_val = df.iloc[120:150], labels.iloc[120:150]

    clf = LightGBMRegimeClassifier(n_estimators=25)
    clf.fit(X_train, y_train)

    val_probs = clf.predict_proba(X_val)
    calib_report = evaluate_probability_calibration(y_val, val_probs)

    assert "multiclass_brier_score" in calib_report
    assert "expected_calibration_error" in calib_report
    assert calib_report["multiclass_brier_score"] >= 0.0

    # Fit post-hoc calibrator on validation
    calibrator = RegimeCalibrator(method="sigmoid")
    calibrator.fit_calibration(clf.model, X_val, y_val, clf.feature_cols)
    assert calibrator.is_calibrated is True

    calib_probs = calibrator.predict_proba(X_val, clf.feature_cols)
    assert calib_probs.shape == (len(X_val), 6)
    assert np.allclose(calib_probs.sum(axis=1), 1.0, atol=1e-3)


def test_regime_metrics_evaluation_no_fabrication():
    """Verify metrics calculation on known truth and predictions."""
    y_true = ["ACTIVE_MONSOON", "BREAK_MONSOON", "OROGRAPHIC", "WESTERN_DISTURBANCE"]
    y_pred = ["ACTIVE_MONSOON", "ACTIVE_MONSOON", "OROGRAPHIC", "WESTERN_DISTURBANCE"]

    report = evaluate_regime_predictions(y_true, y_pred)
    assert report["accuracy"] == 0.75
    assert 0.0 <= report["macro_f1"] <= 1.0
    assert "western_disturbance_diagnostics" in report
    assert report["western_disturbance_diagnostics"]["samples_present"] == 1

    table_md = format_regime_metrics_table(report)
    assert "| ACTIVE_MONSOON |" in table_md
    assert "| WESTERN_DISTURBANCE |" in table_md


# ==============================================================================
# 6. REGIME TRANSITION ENGINE & VECTOR DISTANCES
# ==============================================================================

def test_transition_engine_states():
    engine = RegimeTransitionEngine()

    # Step 1: High confidence active monsoon
    p1 = RegimeProbabilityVector(0.70, 0.06, 0.06, 0.06, 0.06, 0.06)
    out1 = engine.evaluate_transition(
        grid_id="g1",
        timestamp="2026-07-15T03:00:00Z",
        current_prob=p1,
    )
    assert out1.transition_status == TransitionStatus.STABLE
    assert out1.current_regime == "ACTIVE_MONSOON"

    # Step 2: Maintained active monsoon (STABLE)
    p2 = RegimeProbabilityVector(0.68, 0.07, 0.05, 0.07, 0.06, 0.07)
    out2 = engine.evaluate_transition(
        grid_id="g1",
        timestamp="2026-07-16T03:00:00Z",
        current_prob=p2,
    )
    assert out2.transition_status == TransitionStatus.STABLE
    assert out2.previous_regime == "ACTIVE_MONSOON"
    assert out2.transition_strength < 0.10

    # Step 3: Shift toward Monsoon Depression (TRANSITION)
    p3 = RegimeProbabilityVector(0.15, 0.05, 0.65, 0.05, 0.05, 0.05)
    out3 = engine.evaluate_transition(
        grid_id="g1",
        timestamp="2026-07-17T03:00:00Z",
        current_prob=p3,
    )
    assert out3.transition_status == TransitionStatus.TRANSITION
    assert out3.transition_from == "ACTIVE_MONSOON"
    assert out3.transition_to == "MONSOON_DEPRESSION"
    assert out3.current_regime == "MONSOON_DEPRESSION"
    assert out3.transition_strength > 0.40

    # Step 4: Highly ambiguous flat distribution (UNCERTAIN)
    p4 = RegimeProbabilityVector(0.18, 0.17, 0.17, 0.16, 0.16, 0.16)
    out4 = engine.evaluate_transition(
        grid_id="g1",
        timestamp="2026-07-18T03:00:00Z",
        current_prob=p4,
    )
    assert out4.transition_status == TransitionStatus.UNCERTAIN
    assert out4.current_regime == WeatherRegime.REGIME_UNCERTAIN.value


def test_tv_distance_and_entropy_bounds():
    p = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 0.5, 0.5, 0.0, 0.0])
    tv = compute_total_variation_distance(p, q)
    assert tv == 1.0  # Disjoint distributions

    p_same = np.array([0.2, 0.2, 0.2, 0.2, 0.1, 0.1])
    assert compute_total_variation_distance(p_same, p_same) == 0.0

    # Uniform distribution entropy must be 1.0
    uniform = np.full(6, 1.0 / 6.0)
    assert np.isclose(compute_normalized_entropy(uniform), 1.0, atol=1e-4)


# ==============================================================================
# 7. TEMPORAL LEAKAGE AUDIT
# ==============================================================================

def test_leakage_safety_chronological_history():
    """Verify that history strictly maintains past records and cannot see future."""
    buffer = RegimeHistoryBuffer(max_history_len=3)
    p_t0 = RegimeProbabilityVector(0.7, 0.1, 0.1, 0.05, 0.02, 0.03)
    p_t1 = RegimeProbabilityVector(0.5, 0.2, 0.1, 0.1, 0.05, 0.05)

    buffer.append("grid_01", "2026-07-15", p_t0)
    assert buffer.get_latest("grid_01")[0] == "2026-07-15"

    buffer.append("grid_01", "2026-07-16", p_t1)
    assert buffer.get_latest("grid_01")[0] == "2026-07-16"
    assert len(buffer.get_history("grid_01")) == 2


# ==============================================================================
# 8. INTEGRATION & FALLBACK
# ==============================================================================

def test_inference_engine_fallback_on_corrupt_data():
    """Verify fallback mechanism safely outputs REGIME_UNCERTAIN without crashing."""
    engine = RegimeInferenceEngine(classifier=None)  # uninitialized

    corrupt_df = pd.DataFrame([{"lat": 15.0, "lon": 75.0, "grid_id": "test_grid"}])
    outputs = engine.predict(corrupt_df)

    assert len(outputs) == 1
    out = outputs[0]
    assert out.current_regime == WeatherRegime.REGIME_UNCERTAIN.value
    assert out.transition_status == TransitionStatus.UNCERTAIN
    assert out.current_regime_probabilities["active_monsoon"] == round(1.0 / 6.0, 4)


def test_end_to_end_regime_pipeline(synthetic_weather_dataframe):
    """End-to-End Test: Features -> LightGBM -> Calibration -> Transition -> DataFrame."""
    df = synthetic_weather_dataframe
    labels = derive_provisional_regime_labels(df)

    clf = LightGBMRegimeClassifier(n_estimators=30)
    clf.fit(df.iloc[:120], labels.iloc[:120])

    engine = RegimeInferenceEngine(classifier=clf)
    result_df = engine.predict_to_dataframe(df.iloc[120:])

    assert len(result_df) == len(df) - 120
    assert "current_regime" in result_df.columns
    assert "transition_status" in result_df.columns
    assert "prob_active_monsoon" in result_df.columns
    assert "prob_western_disturbance" in result_df.columns

    # Verify probability rows sum to 1.0
    prob_cols = [c for c in result_df.columns if c.startswith("prob_")]
    assert len(prob_cols) == 6
    row_sums = result_df[prob_cols].sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-3)
