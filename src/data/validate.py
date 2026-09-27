"""
Data Validation and Quality Assurance Module for RAIN-REPAIR X (VARSHAA).
Enforces CONTRACT-DATA-001 schema standards, coordinate reference integrity,
and atmospheric physical bounds.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import xarray as xr

from src.utils.config import SpatialDomainConfig, config
from src.utils.exceptions import (
    DataQualityError,
    InvalidCoordinateError,
    TemporalAlignmentError,
)
from src.utils.logging import get_logger

logger = get_logger("rain_repair.data.validate")

# Standard physical bounds for meteorological variables in Indian domain
PHYSICAL_VARIABLE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "nwp_precip": (0.0, 1500.0),      # mm / 24h
    "obs_precip": (0.0, 1500.0),      # mm / 24h
    "t2m": (220.0, 335.0),             # Kelvin (~ -53°C to +62°C)
    "rh850": (0.0, 100.0),             # Percentage
    "u850": (-100.0, 100.0),           # m/s
    "v850": (-100.0, 100.0),           # m/s
    "u200": (-150.0, 150.0),           # m/s
    "v200": (-150.0, 150.0),           # m/s
    "mslp": (900.0, 1050.0),           # hPa
    "elevation": (-50.0, 8848.0),      # meters (Dead Sea to Everest)
    "dem_slope": (0.0, 90.0),          # degrees
}


def validate_spatial_coordinates(
    lats: Union[np.ndarray, List[float], pd.Series],
    lons: Union[np.ndarray, List[float], pd.Series],
    domain: Optional[SpatialDomainConfig] = None,
    tolerance: float = 0.05,
) -> bool:
    """
    Validate that geographic coordinates fall strictly within the defined Indian domain.

    Args:
        lats: Array or series of latitude coordinates.
        lons: Array or series of longitude coordinates.
        domain: SpatialDomainConfig defining bounds. If None, uses app config.
        tolerance: Coordinate tolerance margin in degrees.

    Returns:
        True if all coordinates are valid.

    Raises:
        InvalidCoordinateError: If coordinates are out of bounds or malformed.
    """
    if domain is None:
        domain = config.spatial

    lats_arr = np.asarray(lats, dtype=np.float32)
    lons_arr = np.asarray(lons, dtype=np.float32)

    if lats_arr.size == 0 or lons_arr.size == 0:
        raise InvalidCoordinateError("Coordinate arrays cannot be empty.")

    min_lat, max_lat = float(np.min(lats_arr)), float(np.max(lats_arr))
    min_lon, max_lon = float(np.min(lons_arr)), float(np.max(lons_arr))

    if min_lat < (domain.lat_min - tolerance) or max_lat > (domain.lat_max + tolerance):
        raise InvalidCoordinateError(
            f"Latitude bounds [{min_lat:.2f}, {max_lat:.2f}] exceed domain "
            f"limits [{domain.lat_min}, {domain.lat_max}]."
        )

    if min_lon < (domain.lon_min - tolerance) or max_lon > (domain.lon_max + tolerance):
        raise InvalidCoordinateError(
            f"Longitude bounds [{min_lon:.2f}, {max_lon:.2f}] exceed domain "
            f"limits [{domain.lon_min}, {domain.lon_max}]."
        )

    return True


def validate_physical_bounds(
    data: Union[pd.DataFrame, Dict[str, np.ndarray], xr.Dataset],
    strict: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Verify that numerical values adhere to physical atmospheric and terrestrial bounds.
    Detects invalid sentinel values (e.g. -999.0, 9999).

    Args:
        data: DataFrame, dictionary of arrays, or xarray Dataset to validate.
        strict: If True, raises DataQualityError upon violation; if False, returns report.

    Returns:
        Dictionary of variable violation statistics.

    Raises:
        DataQualityError: If strict is True and any physical bound is violated.
    """
    violations: Dict[str, Dict[str, float]] = {}

    for var_name, (lower_bound, upper_bound) in PHYSICAL_VARIABLE_BOUNDS.items():
        arr: Optional[np.ndarray] = None

        if isinstance(data, pd.DataFrame):
            if var_name in data.columns:
                arr = data[var_name].dropna().to_numpy()
        elif isinstance(data, dict):
            if var_name in data:
                arr = np.asarray(data[var_name])
                arr = arr[~np.isnan(arr)]
        elif isinstance(data, xr.Dataset):
            if var_name in data.data_vars:
                arr = data[var_name].values
                arr = arr[~np.isnan(arr)]

        if arr is None or arr.size == 0:
            continue

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))

        if min_val < lower_bound or max_val > upper_bound:
            n_below = int(np.sum(arr < lower_bound))
            n_above = int(np.sum(arr > upper_bound))
            violations[var_name] = {
                "min": min_val,
                "max": max_val,
                "expected_lower": lower_bound,
                "expected_upper": upper_bound,
                "count_below": n_below,
                "count_above": n_above,
            }

            msg = (
                f"Variable '{var_name}' violates physical bounds: range [{min_val:.2f}, {max_val:.2f}], "
                f"expected [{lower_bound:.2f}, {upper_bound:.2f}]."
            )
            if strict:
                logger.error(msg)
                raise DataQualityError(msg, details=violations[var_name])
            else:
                logger.warning(msg)

    return violations


def validate_missing_data_rate(
    data: Union[pd.DataFrame, np.ndarray],
    max_missing_ratio: float = 0.05,
    variable_name: str = "dataset",
) -> float:
    """
    Verify that missing values (NaN) do not exceed acceptable threshold ratio.

    Args:
        data: Data to inspect.
        max_missing_ratio: Maximum allowed ratio of NaNs (default 0.05 = 5%).
        variable_name: Name of variable for reporting.

    Returns:
        Observed missing ratio.

    Raises:
        DataQualityError: If missing ratio exceeds max_missing_ratio.
    """
    if isinstance(data, pd.DataFrame):
        missing_count = int(data.isna().sum().sum())
        total_count = data.size
    elif isinstance(data, np.ndarray):
        missing_count = int(np.isnan(data).sum())
        total_count = data.size
    else:
        raise ValueError(f"Unsupported data type for missing check: {type(data)}")

    missing_ratio = missing_count / max(total_count, 1)

    if missing_ratio > max_missing_ratio:
        raise DataQualityError(
            f"Missing data ratio for '{variable_name}' is {missing_ratio:.2%}, "
            f"exceeding threshold of {max_missing_ratio:.2%}.",
            details={"missing_count": missing_count, "total_count": total_count, "ratio": missing_ratio},
        )

    return missing_ratio


def validate_lead_times(
    lead_times: Union[List[int], np.ndarray, pd.Series],
    expected_lead_times: Optional[List[int]] = None,
) -> bool:
    """
    Ensure forecast lead times match standard IMD/NWP 24h accumulation cycles.
    """
    if expected_lead_times is None:
        expected_lead_times = config.temporal.lead_times_hours

    unique_leads = sorted(list(set(np.asarray(lead_times, dtype=int).tolist())))

    for lt in unique_leads:
        if lt not in expected_lead_times:
            raise TemporalAlignmentError(
                f"Unexpected lead time {lt}h. Allowed lead times: {expected_lead_times}"
            )

    return True
