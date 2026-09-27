"""
Terrain and Orographic Feature Extraction Module for RAIN-REPAIR X.
Computes elevation, slope gradient, aspect azimuth, and coastal proximity indices.
"""

from __future__ import annotations

from typing import Tuple
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("rain_repair.features.terrain")

# Earth radius in meters for latitude/longitude coordinate distance conversions
EARTH_RADIUS_METERS = 6371000.0


def compute_terrain_derivatives(
    elevation_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate spatial terrain slope (degrees) and aspect (degrees) from elevation raster.

    Args:
        elevation_grid: 2D array of elevations in meters (n_lats, n_lons).
        lats: 1D array of latitude coordinates.
        lons: 1D array of longitude coordinates.

    Returns:
        (slope_deg, aspect_deg) tuple of 2D arrays.
    """
    elev = np.asarray(elevation_grid, dtype=np.float64)

    # Resolution in radians
    d_lat_rad = np.radians(np.abs(np.diff(lats)[0]))
    d_lon_rad = np.radians(np.abs(np.diff(lons)[0]))

    # Grid cell dimensions in meters
    dy = EARTH_RADIUS_METERS * d_lat_rad

    # dx varies with cosine of latitude
    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    dx = EARTH_RADIUS_METERS * d_lon_rad * np.cos(np.radians(lat_grid))
    dx = np.clip(dx, 1000.0, None)  # Prevent division by zero near poles

    # Central difference spatial gradients: dz/dy (north-south) and dz/dx (east-west)
    dz_dy, dz_dx = np.gradient(elev, dy, axis=0), np.gradient(elev, axis=1) / dx

    # Gradient magnitude
    grad_mag = np.sqrt(dz_dx**2 + dz_dy**2)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # Aspect: compass direction that the slope faces (0° North, 90° East, 180° South, 270° West)
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    aspect_deg = np.where(aspect_deg < 0, aspect_deg + 360.0, aspect_deg)

    return slope_deg.astype(np.float32), aspect_deg.astype(np.float32)


def compute_distance_to_coast(
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Compute approximate Euclidean distance to the Indian coastline in kilometers.
    Uses reference coastline vertices for Arabian Sea and Bay of Bengal.
    """
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")
    n_points = lat_grid.size

    # Simplified Indian coastline outline coordinates (lat, lon)
    coast_vertices = np.array(
        [
            [23.5, 68.5], [22.0, 69.0], [20.5, 70.0], [21.0, 72.5], [19.0, 72.8], # Gujarat & Mumbai
            [15.5, 73.8], [12.9, 74.8], [9.9, 76.2], [8.1, 77.5],                   # Konkan & Kerala
            [9.2, 79.2], [10.8, 79.8], [13.1, 80.3], [16.5, 82.0],                 # Tamil Nadu & Andhra
            [19.8, 85.8], [21.5, 87.0], [22.0, 89.0]                                 # Odisha & Bengal
        ],
        dtype=np.float32,
    )

    pts = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])

    # Approximate minimum distance in km using degree distance (~ 111 km/deg)
    dists = np.zeros(n_points, dtype=np.float32)
    for i in range(n_points):
        lat_diff = coast_vertices[:, 0] - pts[i, 0]
        lon_diff = (coast_vertices[:, 1] - pts[i, 1]) * np.cos(np.radians(pts[i, 0]))
        min_deg = np.min(np.sqrt(lat_diff**2 + lon_diff**2))
        dists[i] = min_deg * 111.0

    return dists.reshape(lat_grid.shape).astype(np.float32)
