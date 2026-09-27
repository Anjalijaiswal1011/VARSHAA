"""
Comprehensive Unit & Integration Test Suite for Feature Engineering Layer.
Validates physics features, spatial context, temporal harmonics, climatology,
leakage auditing, schema conformance, and end-to-end pipeline execution.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.climatology import ClimatologyEngine
from src.features.diagnostics import generate_feature_diagnostics_report
from src.features.leakage_audit import audit_feature_dataframe_leakage
from src.features.physics import (
    compute_deep_layer_shear,
    compute_moisture_convergence_proxy,
    compute_orographic_uplift,
    compute_regime_support_diagnostics,
    compute_relative_vorticity,
    compute_vapor_pressure_proxy,
    compute_wind_speed_and_direction,
    compute_windward_leeward_index,
)
from src.features.pipeline import FeatureEngineeringPipeline
from src.features.scaling import RobustFeatureScaler, handle_missing_features, sanitize_domain_outliers
from src.features.schema import (
    FEATURE_REGISTRY,
    FeatureGroup,
    LeakageRisk,
    get_feature_group_names,
    get_feature_names_for_model,
    validate_dataframe_against_schema,
)
from src.features.spatial import compute_spatial_neighborhood_context
from src.features.temporal import compute_temporal_features
from src.utils.exceptions import TemporalAlignmentError


def test_feature_schema_completeness():
    """Verify that every feature in the registry has complete, valid metadata and zero future risk."""
    assert len(FEATURE_REGISTRY) >= 30, "Feature registry should define at least 30 features"

    for name, meta in FEATURE_REGISTRY.items():
        assert meta.feature_name == name
        assert isinstance(meta.feature_group, FeatureGroup)
        assert meta.leakage_risk == LeakageRisk.SAFE, f"Predictor {name} must be classified as SAFE"
        assert meta.data_type in ["float32", "int32", "string"]
        assert len(meta.description) > 5
        assert len(meta.physical_meaning) > 5


def test_spatial_neighborhood_context():
    """Verify 3x3 neighborhood calculations: mean, max, std, gradient, and relative exposure."""
    lats = np.array([18.0, 18.25, 18.5], dtype=np.float32)
    lons = np.array([73.0, 73.25, 73.5], dtype=np.float32)

    # 3x3 precipitation grid with a peak center
    precip = np.array(
        [
            [10.0, 20.0, 10.0],
            [20.0, 100.0, 20.0],
            [10.0, 20.0, 10.0],
        ],
        dtype=np.float32,
    )
    elev = np.array(
        [
            [500.0, 600.0, 500.0],
            [600.0, 1200.0, 600.0],
            [500.0, 600.0, 500.0],
        ],
        dtype=np.float32,
    )

    ctx = compute_spatial_neighborhood_context(precip, elev, lats, lons, kernel_size=3)

    assert "nwp_precip_spatial_mean_3x3" in ctx
    assert "nwp_precip_spatial_max_3x3" in ctx
    assert "nwp_precip_spatial_std_3x3" in ctx
    assert "nwp_precip_spatial_gradient" in ctx
    assert "relative_elevation_exposure" in ctx

    # Center cell max should be 100.0
    assert ctx["nwp_precip_spatial_max_3x3"][1, 1] == 100.0
    # Peak center elevation exposure should be positive (> 0 m above surroundings)
    assert ctx["relative_elevation_exposure"][1, 1] > 0.0


def test_temporal_cyclical_encoding():
    """Verify cyclical harmonics lie in [-1, 1] and satisfy sin^2 + cos^2 = 1."""
    temp_july = compute_temporal_features("2026-07-15", lead_time_hours=48)
    assert -1.0 <= temp_july["doy_sin"] <= 1.0
    assert -1.0 <= temp_july["doy_cos"] <= 1.0
    np.testing.assert_allclose(temp_july["doy_sin"] ** 2 + temp_july["doy_cos"] ** 2, 1.0, atol=1e-4)

    # July is Peak Monsoon (phase code 3)
    assert temp_july["monsoon_phase_code"] == 3
    assert temp_july["lead_time_days"] == 2.0


def test_physics_wind_speed_and_direction():
    """Verify meteorological wind speed and direction conversion."""
    # Pure Westerly wind (u > 0, v = 0) blows FROM 270 degrees (West)
    u = np.array([[10.0]], dtype=np.float32)
    v = np.array([[0.0]], dtype=np.float32)
    speed, direction = compute_wind_speed_and_direction(u, v)

    assert speed[0, 0] == 10.0
    np.testing.assert_allclose(direction[0, 0], 270.0, atol=1e-2)

    # Pure Southerly wind (u = 0, v > 0) blows FROM 180 degrees (South)
    u_south = np.array([[0.0]], dtype=np.float32)
    v_south = np.array([[15.0]], dtype=np.float32)
    speed_s, dir_s = compute_wind_speed_and_direction(u_south, v_south)
    assert speed_s[0, 0] == 15.0
    np.testing.assert_allclose(dir_s[0, 0], 180.0, atol=1e-2)


def test_physics_vapor_pressure_proxy():
    """Verify Tetens equation vapor pressure behaves physically with temperature and humidity."""
    t2m = np.array([[293.15, 303.15]], dtype=np.float32)  # 20°C and 30°C
    rh = np.array([[50.0, 50.0]], dtype=np.float32)        # 50% relative humidity

    vp = compute_vapor_pressure_proxy(t2m, rh)
    # Warmer air holds more water vapor at identical relative humidity
    assert vp[0, 1] > vp[0, 0]
    assert 5.0 < vp[0, 0] < 20.0
    assert 10.0 < vp[0, 1] < 35.0


def test_climatology_engine():
    """Verify climatology engine returns expected baselines and anomaly deltas."""
    engine = ClimatologyEngine()
    lats, lons = engine.lats, engine.lons
    precip = np.full((len(lats), len(lons)), 25.0, dtype=np.float32)

    clim_dict = engine.get_climatology_features("2026-07-25", precip)
    assert "clim_mean_doy" in clim_dict
    assert "nwp_clim_anomaly" in clim_dict
    # Anomaly = NWP - Climatology
    np.testing.assert_allclose(
        clim_dict["nwp_clim_anomaly"],
        precip - clim_dict["clim_mean_doy"],
        atol=1e-4,
    )


def test_domain_outlier_sanitization_preserves_extreme_weather():
    """Verify invalid artifacts are clipped while valid extreme rainfall is preserved."""
    test_df = pd.DataFrame(
        {
            "nwp_precip": [-10.0, 15.0, 350.0],  # -10 is invalid, 350 is valid extreme cloudburst
            "rh850": [-5.0, 80.0, 120.0],         # -5 and 120 are physically impossible
            "t2m": [150.0, 298.0, 400.0],
            "mslp": [850.0, 1004.0, 1100.0],
        }
    )

    clean_df = sanitize_domain_outliers(test_df, preserve_extreme_weather=True)

    # -10 clipped to 0.0
    assert clean_df["nwp_precip"].iloc[0] == 0.0
    # 350 mm preserved without truncation
    assert clean_df["nwp_precip"].iloc[2] == 350.0
    # Relative humidity clipped to [0, 100]
    assert clean_df["rh850"].iloc[0] == 0.0
    assert clean_df["rh850"].iloc[2] == 100.0


def test_missing_feature_imputation():
    """Verify domain-aware imputation fills missing values with sensible defaults."""
    df_missing = pd.DataFrame(
        {
            "nwp_precip": [np.nan, 25.0],
            "rh850": [80.0, np.nan],
            "t2m": [np.nan, 300.0],
        }
    )

    df_imputed = handle_missing_features(df_missing)
    assert df_imputed["nwp_precip"].iloc[0] == 0.0
    assert df_imputed["rh850"].iloc[1] == 70.0
    assert df_imputed["t2m"].iloc[0] == 298.15


def test_leakage_audit_catches_target_label():
    """Verify leakage audit raises TemporalAlignmentError if target column is passed in feature set."""
    dirty_feature_df = pd.DataFrame(
        {
            "nwp_precip": [10.0, 20.0],
            "rh850": [75.0, 80.0],
            "obs_precip": [12.0, 22.0],  # FORBIDDEN PREDICTOR
        }
    )

    with pytest.raises(TemporalAlignmentError):
        audit_feature_dataframe_leakage(dirty_feature_df, cycle_date="2026-07-15", strict=True)


def test_feature_diagnostics_report():
    """Verify feature diagnostics compute missingness, variance, and correlation correctly."""
    df = pd.DataFrame(
        {
            "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feat_b": [2.0, 4.0, 6.0, 8.0, 10.0],  # Exactly collinear with feat_a
            "feat_constant": [5.0, 5.0, 5.0, 5.0, 5.0], # Near-zero variance
        }
    )

    report = generate_feature_diagnostics_report(df)
    assert "feat_constant" in report["near_zero_variance_features"]
    assert len(report["high_correlation_pairs"]) >= 1
    pair = report["high_correlation_pairs"][0]
    assert pair["feature_1"] == "feat_a" and pair["feature_2"] == "feat_b"
    assert pair["abs_correlation"] == 1.0


def test_end_to_end_feature_pipeline_execution():
    """Verify that FeatureEngineeringPipeline produces model-ready features conforming to schema."""
    sample_nwp = Path("data/raw/nwp_gfs_2026-07-15.parquet")
    sample_obs = Path("data/raw/obs_imd_2026-07-15.parquet")

    if not sample_nwp.exists():
        from scripts.download_sample_data import generate_synthetic_monsoon_fixture
        generate_synthetic_monsoon_fixture("2026-07-15")

    pipeline = FeatureEngineeringPipeline(spatial_kernel_size=3)
    X, y, metadata = pipeline.extract_features(
        nwp_data_or_path=sample_nwp,
        cycle_date="2026-07-15",
        lead_time_hours=24,
        obs_data_or_path=sample_obs,
        execute_leakage_audit=True,
    )

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)
    assert len(X) == 129 * 121  # 15,609 points
    assert len(y) == 15609

    # Verify zero leakage
    assert "obs_precip" not in X.columns
    assert metadata["leakage_audit"]["status"] == "PASSED"
    assert metadata["schema_validation"]["is_valid"] is True
    assert metadata["total_features"] >= 28

    # Verify specific derived features exist
    for expected_feat in [
        "nwp_precip_log", "wind_speed_850", "wind_direction_850",
        "vapor_pressure_proxy", "orographic_uplift", "moisture_flux_conv",
        "vorticity_850", "bulk_wind_shear", "windward_leeward_index",
        "relative_elevation_exposure", "nwp_precip_spatial_mean_3x3",
        "clim_mean_doy", "nwp_clim_anomaly", "doy_sin", "doy_cos",
        "wd_shear_proxy", "monsoon_trough_mslp_gradient",
    ]:
        assert expected_feat in X.columns, f"Feature {expected_feat} missing from output X"
