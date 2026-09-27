"""
Spatial Context and Neighborhood Feature Extraction Module for RAIN-REPAIR X.
Computes configurable 3x3 or 5x5 neighborhood metrics: spatial mean, spatial maximum,
spatial variability, gradient magnitude, and relative topographical exposure.
"""

from __future__ import annotations

from typing import Dict, Tuple
import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter

EARTH_RADIUS_METERS = 6371000.0


def compute_spatial_neighborhood_context(
    precip_grid: np.ndarray,
    elevation_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    kernel_size: int = 3,
) -> Dict[str, np.ndarray]:
    """
    Extract spatial neighborhood features around each grid cell.

    Args:
        precip_grid: 2D array of precipitation in mm (n_lats, n_lons).
        elevation_grid: 2D array of elevation in meters (n_lats, n_lons).
        lats: 1D array of latitude coordinates.
        lons: 1D array of longitude coordinates.
        kernel_size: Neighborhood window size (odd integer, typically 3 or 5).

    Returns:
        Dictionary of 2D feature arrays:
        - 'nwp_precip_spatial_mean_3x3': Mean precipitation in neighborhood
        - 'nwp_precip_spatial_max_3x3': Maximum precipitation in neighborhood
        - 'nwp_precip_spatial_std_3x3': Standard deviation in neighborhood
        - 'nwp_precip_spatial_gradient': Spatial gradient magnitude (mm/km)
        - 'relative_elevation_exposure': Elevation minus local mean elevation (meters)
    """
    p = np.asarray(precip_grid, dtype=np.float64)
    elev = np.asarray(elevation_grid, dtype=np.float64)

    if kernel_size % 2 == 0:
        kernel_size += 1

    # 1. Local spatial mean (uniform filter with reflect boundary condition)
    p_mean = uniform_filter(p, size=kernel_size, mode="reflect")

    # 2. Local spatial maximum
    p_max = maximum_filter(p, size=kernel_size, mode="reflect")

    # 3. Local spatial variability / standard deviation: sqrt(E[x^2] - E[x]^2)
    p_sq_mean = uniform_filter(p**2, size=kernel_size, mode="reflect")
    variance = np.maximum(p_sq_mean - p_mean**2, 0.0)
    p_std = np.sqrt(variance)

    # 4. Spatial gradient magnitude in mm/km
    d_lat_rad = np.radians(np.abs(np.diff(lats)[0]))
    d_lon_rad = np.radians(np.abs(np.diff(lons)[0]))
    dy_km = (EARTH_RADIUS_METERS * d_lat_rad) / 1000.0

    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    dx_km = (EARTH_RADIUS_METERS * d_lon_rad * np.cos(np.radians(lat_grid))) / 1000.0
    dx_km = np.clip(dx_km, 1.0, None)

    dp_dy = np.gradient(p, dy_km, axis=0)
    dp_dx = np.gradient(p, axis=1) / dx_km
    p_gradient = np.sqrt(dp_dx**2 + dp_dy**2)

    # 5. Relative topographical exposure: elevation - local mean elevation
    elev_mean = uniform_filter(elev, size=kernel_size, mode="reflect")
    rel_exposure = elev - elev_mean

    return {
        f"nwp_precip_spatial_mean_{kernel_size}x{kernel_size}": p_mean.astype(np.float32),
        f"nwp_precip_spatial_max_{kernel_size}x{kernel_size}": p_max.astype(np.float32),
        f"nwp_precip_spatial_std_{kernel_size}x{kernel_size}": p_std.astype(np.float32),
        "nwp_precip_spatial_gradient": p_gradient.astype(np.float32),
        "relative_elevation_exposure": rel_exposure.astype(np.float32),
    }
