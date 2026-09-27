"""
Physics-Guided Meteorological Feature Extraction Module for RAIN-REPAIR X.
Computes domain-informed physical diagnostics: orographic uplift velocity,
moisture flux convergence, relative vorticity, deep-layer wind shear,
vapor pressure proxy, windward/leeward exposure, and synoptic regime indicators.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple
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


def compute_wind_speed_and_direction(
    u_wind: np.ndarray,
    v_wind: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate wind speed (m/s) and meteorological direction (degrees from North).

    Returns:
        (speed, direction_deg)
    """
    u = np.asarray(u_wind, dtype=np.float32)
    v = np.asarray(v_wind, dtype=np.float32)

    speed = np.sqrt(u**2 + v**2)

    # Meteorological direction: direction FROM which wind blows
    # 0 deg = North, 90 deg = East, 180 deg = South, 270 deg = West
    rad = np.arctan2(-u, -v)
    deg = np.degrees(rad)
    deg = np.where(deg < 0, deg + 360.0, deg)

    return speed.astype(np.float32), deg.astype(np.float32)


def compute_vapor_pressure_proxy(
    t2m: np.ndarray,
    rh850: np.ndarray,
) -> np.ndarray:
    """
    Estimate column actual vapor pressure (hPa) via Tetens equation:
        e_sat(T) = 6.1078 * exp( 17.27 * (T - 273.15) / (T - 35.86) )
        e_actual = (RH / 100.0) * e_sat
    """
    t_celsius = np.asarray(t2m, dtype=np.float64) - 273.15
    rh = np.asarray(rh850, dtype=np.float64) / 100.0

    # Tetens formula for saturation vapor pressure over liquid water (hPa)
    e_sat = 6.1078 * np.exp((17.27 * t_celsius) / (t_celsius + 237.3))
    e_actual = rh * e_sat

    return np.clip(e_actual, 0.0, 70.0).astype(np.float32)


def compute_windward_leeward_index(
    wind_direction_deg: np.ndarray,
    aspect_deg: np.ndarray,
) -> np.ndarray:
    """
    Calculate windward / leeward alignment index in [-1.0, 1.0].
        index = cos( wind_direction - (aspect + 180) )

    +1.0: Flow blows directly into the facing mountainside (maximum mechanical uplift).
    -1.0: Flow blows downslope away from the face (subsidence / rain shadow).
    """
    w_dir = np.radians(np.asarray(wind_direction_deg, dtype=np.float32))
    asp = np.radians(np.asarray(aspect_deg, dtype=np.float32))

    # Aspect faces downhill; windward is opposite of aspect
    facing_dir = asp + np.pi
    alignment = np.cos(w_dir - facing_dir)

    return np.clip(alignment, -1.0, 1.0).astype(np.float32)


def compute_regime_support_diagnostics(
    u850: np.ndarray,
    v850: np.ndarray,
    u200: np.ndarray,
    mslp: np.ndarray,
    lats: np.ndarray,
    lons: np.ndarray,
    dist_to_coast: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Derive synoptic diagnostic indicators directly supporting the 6 monsoon regimes:
    1. Western Disturbance (WD) jet shear proxy
    2. Monsoon Trough meridional MSLP gradient
    3. Offshore trough coastal wind convergence
    """
    lat_grid = np.meshgrid(lats, lons, indexing="ij")[0]

    # 1. Western Disturbance: Upper westerly jet core (>24N)
    wd_shear = np.maximum(u200, 0.0) * (lat_grid >= 24.0)

    # 2. Monsoon Trough MSLP Gradient: d(MSLP)/d(Lat)
    d_lat_deg = np.abs(np.diff(lats)[0])
    d_mslp_dlat = np.gradient(mslp, d_lat_deg, axis=0)

    # 3. Offshore Trough: Coastal shear along West Coast (lat 10-18N, dist_to_coast < 80km)
    coastal_west_mask = (lat_grid >= 10.0) & (lat_grid <= 18.0) & (dist_to_coast < 80.0)
    offshore_shear = v850 * coastal_west_mask.astype(np.float32)

    return {
        "wd_shear_proxy": wd_shear.astype(np.float32),
        "monsoon_trough_mslp_gradient": d_mslp_dlat.astype(np.float32),
        "offshore_trough_coastal_shear": offshore_shear.astype(np.float32),
    }
