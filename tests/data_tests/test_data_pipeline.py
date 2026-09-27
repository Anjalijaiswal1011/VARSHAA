"""
Automated Test Suite for RAIN-REPAIR X Data Pipeline.
Validates ingestion, validation, regridding, physics feature calculation,
anti-leakage error memory buffers, and end-to-end dataset assembly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.ingest import load_nwp_forecast_file, load_observation_file, load_static_terrain_data
from src.data.pipeline import DataPipeline
from src.data.standardize import (
    align_observation_time_window,
    generate_target_grid_coords,
    regrid_2d_field,
)
from src.data.validate import (
    PHYSICAL_VARIABLE_BOUNDS,
    validate_missing_data_rate,
    validate_physical_bounds,
    validate_spatial_coordinates,
)
from src.features.error_memory import ErrorMemoryBuffer
from src.features.physics import (
    compute_deep_layer_shear,
    compute_moisture_convergence_proxy,
    compute_orographic_uplift,
    compute_relative_vorticity,
)
from src.features.terrain import compute_terrain_derivatives
from src.utils.exceptions import (
    DataQualityError,
    InvalidCoordinateError,
    TemporalAlignmentError,
)


def test_generate_target_grid_dimensions():
    """Verify target grid generates exact dimensions for Indian domain at 0.25 deg."""
    lats, lons = generate_target_grid_coords()
    assert lats[0] == 6.0
    assert lats[-1] == 38.0
    assert len(lats) == 129

    assert lons[0] == 68.0
    assert lons[-1] == 98.0
    assert len(lons) == 121


def test_spatial_coordinate_validation_raises():
    """Test that out-of-bounds coordinates raise InvalidCoordinateError."""
    # North of India
    with pytest.raises(InvalidCoordinateError):
        validate_spatial_coordinates([45.0], [75.0])

    # West of India
    with pytest.raises(InvalidCoordinateError):
        validate_spatial_coordinates([20.0], [55.0])


def test_physical_bounds_validation():
    """Verify physical validator catches negative rainfall and temperature spikes."""
    valid_df = pd.DataFrame(
        {
            "nwp_precip": [0.0, 15.5, 120.0],
            "t2m": [280.0, 298.0, 315.0],
            "rh850": [10.0, 75.0, 95.0],
        }
    )
    # Should not raise
    validate_physical_bounds(valid_df, strict=True)

    # Negative rainfall must raise DataQualityError
    invalid_df = pd.DataFrame({"nwp_precip": [-5.0, 10.0]})
    with pytest.raises(DataQualityError):
        validate_physical_bounds(invalid_df, strict=True)


def test_missing_data_rate_validation():
    """Verify missing data validator threshold behavior."""
    arr = np.array([1.0, 2.0, np.nan, 4.0, 5.0])  # 20% NaN
    with pytest.raises(DataQualityError):
        validate_missing_data_rate(arr, max_missing_ratio=0.05)

    arr_clean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert validate_missing_data_rate(arr_clean, max_missing_ratio=0.05) == 0.0


def test_regrid_2d_field_spatial_interpolation():
    """Verify 2D bilinear regridding produces expected output shape and preserves range."""
    src_lats = np.linspace(10.0, 25.0, 30)
    src_lons = np.linspace(70.0, 85.0, 30)
    field = np.ones((len(src_lats), len(src_lons)), dtype=np.float32) * 50.0

    target_lats, target_lons = generate_target_grid_coords()
    regridded = regrid_2d_field(field, src_lats, src_lons, target_lats, target_lons)

    assert regridded.shape == (len(target_lats), len(target_lons))
    # Interior should interpolate close to 50.0
    interior_mask = (
        (target_lats[:, None] >= 11.0) & (target_lats[:, None] <= 24.0) &
        (target_lons[None, :] >= 71.0) & (target_lons[None, :] <= 84.0)
    )
    np.testing.assert_allclose(regridded[interior_mask], 50.0, rtol=1e-3)


def test_physics_orographic_uplift():
    """Verify orographic uplift gives positive values on windward ascending terrain."""
    lats = np.array([15.0, 15.25], dtype=np.float32)
    lons = np.array([73.0, 73.25], dtype=np.float32)

    # Elevation rising to the east (like Western Ghats)
    elev = np.array([[100.0, 800.0], [100.0, 800.0]], dtype=np.float32)

    # Westerly wind (u > 0 blowing eastward towards mountain)
    u_wind = np.array([[15.0, 15.0], [15.0, 15.0]], dtype=np.float32)
    v_wind = np.zeros_like(u_wind)

    uplift = compute_orographic_uplift(u_wind, v_wind, elev, lats, lons)
    assert np.all(uplift > 0.0), "Westerly flow against rising topography must produce positive uplift"


def test_physics_vorticity_calculation():
    """Verify cyclonic circulation produces positive relative vorticity."""
    lats = np.array([20.0, 20.25], dtype=np.float32)
    lons = np.array([80.0, 80.25], dtype=np.float32)

    # Cyclonic shear: v increasing with x, u decreasing with y
    u = np.array([[10.0, 10.0], [-10.0, -10.0]], dtype=np.float32)
    v = np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=np.float32)

    vort = compute_relative_vorticity(u, v, lats, lons)
    assert np.all(vort > 0.0), "Cyclonic vortex must have positive relative vorticity"


def test_error_memory_causal_anti_leakage():
    """Verify error memory strictly prevents future observation queries."""
    buf = ErrorMemoryBuffer()
    dummy_grid = np.ones((5, 5), dtype=np.float32)

    # Record past dates
    buf.record_historical_error("2026-07-12", dummy_grid * 20, dummy_grid * 10)
    buf.record_historical_error("2026-07-13", dummy_grid * 25, dummy_grid * 15)
    buf.record_historical_error("2026-07-14", dummy_grid * 30, dummy_grid * 20)

    # Query for cycle 2026-07-15
    res = buf.get_multi_timescale_error_memory("2026-07-15", (5, 5))
    assert "error_memory_lag1" in res
    assert "error_memory_lag3_mean" in res
    # Lag 1 error on 2026-07-14 was 30 - 20 = 10.0
    np.testing.assert_allclose(res["error_memory_lag1"], 10.0)

    # If we record a future error (e.g. 2026-07-16), query for 2026-07-15 must ignore it
    buf.record_historical_error("2026-07-16", dummy_grid * 100, dummy_grid * 0)
    res_causal = buf.get_multi_timescale_error_memory("2026-07-15", (5, 5))
    # Lag 1 should STILL be 10.0, not 100.0
    np.testing.assert_allclose(res_causal["error_memory_lag1"], 10.0)


def test_end_to_end_data_pipeline_execution():
    """Verify complete end-to-end DataPipeline process_cycle on generated fixtures."""
    sample_nwp = Path("data/raw/nwp_gfs_2026-07-15.parquet")
    sample_obs = Path("data/raw/obs_imd_2026-07-15.parquet")

    if not sample_nwp.exists():
        from scripts.download_sample_data import generate_synthetic_monsoon_fixture
        generate_synthetic_monsoon_fixture("2026-07-15")

    pipeline = DataPipeline()
    processed_df = pipeline.process_cycle(
        nwp_data_or_path=sample_nwp,
        cycle_date="2026-07-15",
        lead_time_hours=24,
        obs_data_or_path=sample_obs,
    )

    assert isinstance(processed_df, pd.DataFrame)
    assert len(processed_df) == 129 * 121  # 15,609 grid cells

    # Verify all CONTRACT-DATA-001 required columns exist
    expected_cols = [
        "cycle_date", "lead_time", "lat", "lon", "nwp_precip",
        "t2m", "rh850", "u850", "v850", "mslp", "elevation",
        "dem_slope", "dem_aspect", "dist_to_coast", "orographic_uplift",
        "moisture_flux_conv", "vorticity_850", "bulk_wind_shear",
        "error_memory_lag1", "error_memory_lag3_mean", "obs_precip",
    ]
    for col in expected_cols:
        assert col in processed_df.columns, f"Missing expected column: {col}"

    # Verify physical sanity
    assert processed_df["nwp_precip"].min() >= 0.0
    assert processed_df["obs_precip"].min() >= 0.0
    assert processed_df["elevation"].max() > 1000.0
