"""
Unit tests for Configuration Loading and Environment Overrides.
"""

from pathlib import Path
from src.utils.config import AppConfig, load_config


def test_default_config_loading(test_config: AppConfig):
    """Test that default configuration loads with correct project metadata."""
    assert test_config.project_name == "RAIN-REPAIR X (VARSHAA)"
    assert test_config.version == "1.0.0"
    assert test_config.random_seed == 42


def test_spatial_domain_bounds(test_config: AppConfig):
    """Verify that spatial coordinates reflect Indian Subcontinent boundaries."""
    spatial = test_config.spatial
    assert spatial.crs == "EPSG:4326"
    assert spatial.lat_min == 6.0
    assert spatial.lat_max == 38.0
    assert spatial.lon_min == 68.0
    assert spatial.lon_max == 98.0
    assert spatial.resolution_deg == 0.25


def test_meteorological_thresholds(test_config: AppConfig):
    """Verify IMD rainfall category thresholds."""
    thresholds = test_config.thresholds
    assert thresholds.heavy_rain_mm == 64.5
    assert thresholds.very_heavy_rain_mm == 115.6
    assert thresholds.extreme_rain_mm == 204.5
    assert 0.50 in thresholds.quantiles
    assert 0.90 in thresholds.quantiles


def test_directory_paths_are_path_instances(test_config: AppConfig):
    """Verify paths are properly typed as pathlib.Path."""
    paths = test_config.paths
    assert isinstance(paths.data_raw_dir, Path)
    assert isinstance(paths.data_interim_dir, Path)
    assert isinstance(paths.data_processed_dir, Path)
    assert isinstance(paths.models_checkpoint_dir, Path)
