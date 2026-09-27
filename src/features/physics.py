"""
Physics-Guided Meteorological Feature Extraction Module for RAIN-REPAIR X.
Computes domain-informed physical diagnostics: orographic uplift velocity,
moisture flux convergence, relative vorticity, and deep-layer wind shear.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np

EARTH_RADIUS_METERS = 6371000.0


def compute_orographic_uplift(
    u_wind_850: np.ndarray,
    v_wind_850: np.ndarray,
    elevation_grid: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Calculate mechanical orographic vertical uplift velocity:
        w_orog = V_850 · ∇h = u * (dh/dx) + v * (dh/dy)

    Positive values denote forced windward ascending motion (condensation/precipitation).
    Negative values denote leeward descending rain-shadow subsidence.

    Args:
        u_wind_850: 2D array of zonal wind (m/s).
        v_wind_850: 2D array of meridional wind (m/s).
        elevation_grid: 2D array of terrain elevation in meters.
        lats: 1D array of latitude coordinates.
        lons: 1D array of longitude coordinates.

    Returns:
        2D array of orographic uplift velocity in m/s.
    """
    u = np.asarray(u_wind_850, dtype=np.float64)
    v = np.asarray(v_wind_850, dtype=np.float64)
    elev = np.asarray(elevation_grid, dtype=np.float64)

    # Grid spacings in meters
    d_lat_rad = np.radians(np.abs(np.diff(lats)[0]))
    d_lon_rad = np.radians(np.abs(np.diff(lons)[0]))
    dy = EARTH_RADIUS_METERS * d_lat_rad

    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    dx = EARTH_RADIUS_METERS * d_lon_rad * np.cos(np.radians(lat_grid))
    dx = np.clip(dx, 1000.0, None)

    # Elevation spatial gradients: dh/dy and dh/dx
    dh_dy = np.gradient(elev, dy, axis=0)
    dh_dx = np.gradient(elev, axis=1) / dx

    # Dot product: u * (dh/dx) + v * (dh/dy)
    w_orog = (u * dh_dx) + (v * dh_dy)

    return w_orog.astype(np.float32)


def compute_moisture_convergence_proxy(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    rh_field: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Calculate 2D horizontal moisture flux convergence proxy:
        MFC_proxy = - ∇ · (RH * V) = - [ d(u*RH)/dx + d(v*RH)/dy ]

    Positive values indicate moisture accumulation fueling convective precipitation.
    """
    u = np.asarray(u_wind, dtype=np.float64)
    v = np.asarray(v_wind, dtype=np.float64)
    rh = np.asarray(rh_field, dtype=np.float64) / 100.0  # Normalize to [0, 1]

    d_lat_rad = np.radians(np.abs(np.diff(lats)[0]))
    d_lon_rad = np.radians(np.abs(np.diff(lons)[0]))
    dy = EARTH_RADIUS_METERS * d_lat_rad

    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    dx = EARTH_RADIUS_METERS * d_lon_rad * np.cos(np.radians(lat_grid))
    dx = np.clip(dx, 1000.0, None)

    flux_x = u * rh
    flux_y = v * rh

    d_flux_x_dx = np.gradient(flux_x, axis=1) / dx
    d_flux_y_dy = np.gradient(flux_y, dy, axis=0)

    # Convergence = - Divergence
    mfc = -(d_flux_x_dx + d_flux_y_dy)

    return mfc.astype(np.float32)


def compute_relative_vorticity(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
) -> np.ndarray:
    """
    Calculate 2D relative vorticity (zeta):
        zeta = dv/dx - du/dy

    Positive in Northern Hemisphere denotes cyclonic rotation (Monsoon Depressions / Lows).
    """
    u = np.asarray(u_wind, dtype=np.float64)
    v = np.asarray(v_wind, dtype=np.float64)

    d_lat_rad = np.radians(np.abs(np.diff(lats)[0]))
    d_lon_rad = np.radians(np.abs(np.diff(lons)[0]))
    dy = EARTH_RADIUS_METERS * d_lat_rad

    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]
    dx = EARTH_RADIUS_METERS * d_lon_rad * np.cos(np.radians(lat_grid))
    dx = np.clip(dx, 1000.0, None)

    dv_dx = np.gradient(v, axis=1) / dx
    du_dy = np.gradient(u, dy, axis=0)

    zeta = dv_dx - du_dy

    return zeta.astype(np.float32)


def compute_deep_layer_shear(
    u_lower: np.ndarray,
    v_lower: np.ndarray,
    u_upper: np.ndarray,
    v_upper: np.ndarray,
) -> np.ndarray:
    """
    Calculate bulk vertical wind shear magnitude between lower (e.g. 850 hPa)
    and upper troposphere (e.g. 200 hPa):
        Shear = sqrt( (u_upper - u_lower)^2 + (v_upper - v_lower)^2 )
    """
    du = np.asarray(u_upper, dtype=np.float32) - np.asarray(u_lower, dtype=np.float32)
    dv = np.asarray(v_upper, dtype=np.float32) - np.asarray(v_lower, dtype=np.float32)

    shear = np.sqrt(du**2 + dv**2)
    return shear.astype(np.float32)
