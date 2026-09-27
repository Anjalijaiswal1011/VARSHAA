"""
Spatial and Temporal Standardization Module for RAIN-REPAIR X (VARSHAA).
Provides spatial regridding to standard 0.25 deg grid (EPSG:4326) and
temporal accumulation window alignment (03:00 to 03:00 UTC).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from src.utils.config import SpatialDomainConfig, config
from src.utils.exceptions import InvalidCoordinateError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.data.standardize")


def generate_target_grid_coords(
    domain: Optional[SpatialDomainConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 1D coordinate arrays for standard Indian domain grid at target resolution.

    Returns:
        (lats, lons) tuple where:
        lats: 1D array from lat_min to lat_max with step resolution_deg
        lons: 1D array from lon_min to lon_max with step resolution_deg
    """
    if domain is None:
        domain = config.spatial

    lats = np.arange(domain.lat_min, domain.lat_max + (domain.resolution_deg / 2.0), domain.resolution_deg, dtype=np.float32)
    lons = np.arange(domain.lon_min, domain.lon_max + (domain.resolution_deg / 2.0), domain.resolution_deg, dtype=np.float32)

    # Clean floating precision noise (e.g. 18.250002 -> 18.25)
    lats = np.round(lats, decimals=2)
    lons = np.round(lons, decimals=2)

    return lats, lons


def regrid_2d_field(
    source_field: np.ndarray,
    source_lats: np.ndarray,
    source_lons: np.ndarray,
    target_lats: Optional[np.ndarray] = None,
    target_lons: Optional[np.ndarray] = None,
    method: str = "linear",
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Interpolate a 2D spatial grid (e.g. precipitation or temperature) to target grid coordinates.

    Args:
        source_field: 2D array of shape (len(source_lats), len(source_lons)).
        source_lats: 1D monotonic array of source latitude coordinates.
        source_lons: 1D monotonic array of source longitude coordinates.
        target_lats: 1D target latitudes. If None, uses standard domain.
        target_lons: 1D target longitudes. If None, uses standard domain.
        method: Interpolation method ('linear', 'nearest', 'slinear', 'cubic').
        fill_value: Value used for points outside source convex hull.

    Returns:
        2D array of shape (len(target_lats), len(target_lons)).
    """
    if target_lats is None or target_lons is None:
        target_lats, target_lons = generate_target_grid_coords()

    source_lats = np.asarray(source_lats, dtype=np.float64)
    source_lons = np.asarray(source_lons, dtype=np.float64)
    source_field = np.asarray(source_field, dtype=np.float64)

    # Check ascending order requirement for RegularGridInterpolator
    flip_lat = False
    if len(source_lats) > 1 and source_lats[0] > source_lats[-1]:
        source_lats = source_lats[::-1]
        source_field = np.flipud(source_field)
        flip_lat = True

    if source_field.shape != (len(source_lats), len(source_lons)):
        raise InvalidCoordinateError(
            f"Shape mismatch: field shape {source_field.shape} does not match coords "
            f"({len(source_lats)}, {len(source_lons)})"
        )

    # Create interpolator
    interpolator = RegularGridInterpolator(
        (source_lats, source_lons),
        source_field,
        method=method,
        bounds_error=False,
        fill_value=fill_value,
    )

    # Construct target query points meshgrid
    tgt_lat_grid, tgt_lon_grid = np.meshgrid(target_lats, target_lons, indexing="ij")
    target_points = np.stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()], axis=-1)

    regridded = interpolator(target_points).reshape(len(target_lats), len(target_lons))

    return regridded.astype(np.float32)


def align_observation_time_window(
    observation_date: str,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Standardize the 24-hour observation accumulation window matching IMD operational practice.
    IMD daily rainfall recorded on Day T (08:30 IST) corresponds to accumulation from
    03:00 UTC (Day T - 1) to 03:00 UTC (Day T).

    Args:
        observation_date: ISO date string 'YYYY-MM-DD'.

    Returns:
        (window_start_utc, window_end_utc) timestamps.
    """
    obs_day = pd.to_datetime(observation_date).tz_localize("UTC") if pd.to_datetime(observation_date).tzinfo is None else pd.to_datetime(observation_date)

    window_end_utc = obs_day.replace(hour=3, minute=0, second=0)
    window_start_utc = window_end_utc - pd.Timedelta(days=1)

    return window_start_utc, window_end_utc
