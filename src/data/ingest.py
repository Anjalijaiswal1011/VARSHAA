"""
Data Ingestion Module for RAIN-REPAIR X (VARSHAA).
Handles ingestion of raw NWP forecasts, ground-truth observations,
static terrain DEM rasters, and climatology files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
import xarray as xr

from src.data.standardize import generate_target_grid_coords, regrid_2d_field
from src.data.validate import (
    validate_missing_data_rate,
    validate_physical_bounds,
    validate_spatial_coordinates,
)
from src.utils.config import config
from src.utils.exceptions import DataIngestionError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.data.ingest")


def load_nwp_forecast_file(
    file_path: Union[str, Path],
    expected_lead_times: Optional[List[int]] = None,
    regrid_to_standard: bool = True,
) -> Dict[str, Any]:
    """
    Ingest a raw NWP forecast file (NetCDF4, Parquet, or HDF5/GRIB).

    Args:
        file_path: Path to raw forecast file.
        expected_lead_times: List of lead times in hours. Default from config.
        regrid_to_standard: If True, regrids fields to standard 0.25 deg grid.

    Returns:
        Standardized dictionary containing:
        - 'lats': 1D array
        - 'lons': 1D array
        - 'lead_times': 1D array
        - 'cycle_date': str
        - 'variables': Dict[str, np.ndarray] e.g. 'nwp_precip', 'rh850', 'u850', 'v850', 'mslp'
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"NWP forecast file not found: {path}")

    if expected_lead_times is None:
        expected_lead_times = config.temporal.lead_times_hours

    logger.info("Ingesting NWP forecast file: %s", path.name)

    try:
        if path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
            validate_physical_bounds(df, strict=False)
            return _parse_tabular_forecast(df, expected_lead_times, regrid_to_standard)

        elif path.suffix in [".nc", ".nc4", ".h5", ".hdf5"]:
            # Load via xarray
            with xr.open_dataset(path) as ds:
                return _parse_xarray_forecast(ds, expected_lead_times, regrid_to_standard)

        elif path.suffix in [".csv"]:
            df = pd.read_csv(path)
            validate_physical_bounds(df, strict=False)
            return _parse_tabular_forecast(df, expected_lead_times, regrid_to_standard)

        else:
            raise DataIngestionError(f"Unsupported file format '{path.suffix}' for NWP ingestion.")

    except Exception as exc:
        if isinstance(exc, DataIngestionError):
            raise
        logger.error("Failed to ingest NWP file %s: %s", path, exc)
        raise DataIngestionError(f"Corrupted or invalid NWP forecast file: {exc}") from exc


def load_observation_file(
    file_path: Union[str, Path],
    observation_date: str,
    regrid_to_standard: bool = True,
) -> Dict[str, Any]:
    """
    Ingest a ground truth observation file (e.g. IMD gridded daily rainfall).

    Args:
        file_path: Path to observation dataset.
        observation_date: ISO date string 'YYYY-MM-DD'.
        regrid_to_standard: If True, regrids onto canonical 0.25 deg grid.

    Returns:
        Dictionary with 'lats', 'lons', 'obs_precip', 'observation_date'.
    """
    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"Observation file not found: {path}")

    logger.info("Ingesting ground observation file: %s for date %s", path.name, observation_date)

    try:
        if path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
            if "date" in df.columns:
                df = df[df["date"] == observation_date]
            if df.empty:
                raise DataIngestionError(f"No records found for date {observation_date} in {path}")
            return _parse_tabular_observation(df, observation_date, regrid_to_standard)

        elif path.suffix in [".nc", ".nc4"]:
            with xr.open_dataset(path) as ds:
                return _parse_xarray_observation(ds, observation_date, regrid_to_standard)

        elif path.suffix in [".csv"]:
            df = pd.read_csv(path)
            if "date" in df.columns:
                df = df[df["date"] == observation_date]
            return _parse_tabular_observation(df, observation_date, regrid_to_standard)

        else:
            raise DataIngestionError(f"Unsupported observation file extension: {path.suffix}")

    except Exception as exc:
        if isinstance(exc, DataIngestionError):
            raise
        logger.error("Failed to ingest observation file %s: %s", path, exc)
        raise DataIngestionError(f"Corrupted or invalid observation file: {exc}") from exc


def load_static_terrain_data(
    file_path: Optional[Union[str, Path]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load or initialize static topography (DEM elevation, slope, aspect) on standard 0.25 deg grid.
    If file_path is None or does not exist, generates terrain based on Indian topography.
    """
    tgt_lats, tgt_lons = generate_target_grid_coords()

    if file_path is not None and Path(file_path).exists():
        path = Path(file_path)
        logger.info("Loading terrain from: %s", path.name)
        if path.suffix in [".parquet", ".pq"]:
            df = pd.read_parquet(path)
            return {
                "elevation": df["elevation"].to_numpy().reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32),
                "dem_slope": df.get("dem_slope", pd.Series(0.0)).to_numpy().reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32),
                "dem_aspect": df.get("dem_aspect", pd.Series(0.0)).to_numpy().reshape(len(tgt_lats), len(tgt_lons)).astype(np.float32),
            }

    # Generate reference topography model for Western Ghats and Himalayas
    lat_grid, lon_grid = np.meshgrid(tgt_lats, tgt_lons, indexing="ij")
    elevation = np.zeros_like(lat_grid, dtype=np.float32)

    # Western Ghats Ridge (~ 8°N to 21°N, 73°E to 76°E, ~ 1000m-2000m)
    wg_mask = (lat_grid >= 8.0) & (lat_grid <= 21.0) & (lon_grid >= 73.0) & (lon_grid <= 76.5)
    elevation[wg_mask] = 1200.0 * np.exp(-((lon_grid[wg_mask] - 74.0) ** 2) / 0.5)

    # Himalayan Foothills & Plateau (~ 28°N to 36°N, 75°E to 95°E, ~ 2500m-6000m)
    him_mask = (lat_grid >= 28.0)
    elevation[him_mask] += np.clip(1000.0 * (lat_grid[him_mask] - 27.0) ** 1.5, 0.0, 6500.0)

    # Central Deccan Plateau (~ 12°N to 24°N, 76°E to 82°E, ~ 400m-600m)
    deccan_mask = (lat_grid >= 12.0) & (lat_grid <= 24.0) & (lon_grid >= 76.0) & (lon_grid <= 82.0)
    elevation[deccan_mask] += 450.0

    return {
        "elevation": elevation.astype(np.float32),
        "dem_slope": np.clip(np.gradient(elevation, axis=0) * 0.1, 0.0, 45.0).astype(np.float32),
        "dem_aspect": (np.arctan2(np.gradient(elevation, axis=1), np.gradient(elevation, axis=0)) * 180.0 / np.pi).astype(np.float32),
    }


# ==============================================================================
# Helper Parsers
# ==============================================================================


def _parse_tabular_forecast(
    df: pd.DataFrame,
    expected_lead_times: List[int],
    regrid_to_standard: bool,
) -> Dict[str, Any]:
    required_cols = ["lat", "lon", "lead_time", "nwp_precip"]
    for col in required_cols:
        if col not in df.columns:
            raise DataIngestionError(f"Missing required forecast column: '{col}'")

    validate_spatial_coordinates(df["lat"], df["lon"])
    cycle_date = str(df["cycle_date"].iloc[0]) if "cycle_date" in df.columns else "2026-07-15"

    tgt_lats, tgt_lons = generate_target_grid_coords()
    variables: Dict[str, np.ndarray] = {}

    for var in ["nwp_precip", "t2m", "rh850", "u850", "v850", "mslp"]:
        if var in df.columns:
            variables[var] = df[var].to_numpy(dtype=np.float32)

    return {
        "lats": tgt_lats if regrid_to_standard else np.sort(df["lat"].unique()),
        "lons": tgt_lons if regrid_to_standard else np.sort(df["lon"].unique()),
        "lead_times": np.array(expected_lead_times, dtype=np.int32),
        "cycle_date": cycle_date,
        "variables": variables,
        "dataframe": df,
    }


def _parse_xarray_forecast(
    ds: xr.Dataset,
    expected_lead_times: List[int],
    regrid_to_standard: bool,
) -> Dict[str, Any]:
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"

    src_lats = ds[lat_key].values
    src_lons = ds[lon_key].values
    validate_spatial_coordinates(src_lats, src_lons)

    tgt_lats, tgt_lons = generate_target_grid_coords()
    cycle_date = str(ds.attrs.get("cycle_date", "2026-07-15"))

    variables: Dict[str, np.ndarray] = {}
    precip_key = "nwp_precip" if "nwp_precip" in ds else "tp"
    if precip_key in ds:
        raw_precip = ds[precip_key].values
        if regrid_to_standard and (len(src_lats) != len(tgt_lats) or len(src_lons) != len(tgt_lons)):
            regridded = regrid_2d_field(raw_precip, src_lats, src_lons, tgt_lats, tgt_lons)
            variables["nwp_precip"] = regridded
        else:
            variables["nwp_precip"] = raw_precip.astype(np.float32)

    return {
        "lats": tgt_lats if regrid_to_standard else src_lats,
        "lons": tgt_lons if regrid_to_standard else src_lons,
        "lead_times": np.array(expected_lead_times, dtype=np.int32),
        "cycle_date": cycle_date,
        "variables": variables,
        "dataset": ds,
    }


def _parse_tabular_observation(
    df: pd.DataFrame,
    observation_date: str,
    regrid_to_standard: bool,
) -> Dict[str, Any]:
    if "obs_precip" not in df.columns and "precip" in df.columns:
        df = df.rename(columns={"precip": "obs_precip"})

    if "obs_precip" not in df.columns:
        raise DataIngestionError("Observation table must contain 'obs_precip' column.")

    validate_spatial_coordinates(df["lat"], df["lon"])
    tgt_lats, tgt_lons = generate_target_grid_coords()

    return {
        "lats": tgt_lats if regrid_to_standard else np.sort(df["lat"].unique()),
        "lons": tgt_lons if regrid_to_standard else np.sort(df["lon"].unique()),
        "observation_date": observation_date,
        "obs_precip": df["obs_precip"].to_numpy(dtype=np.float32),
        "dataframe": df,
    }


def _parse_xarray_observation(
    ds: xr.Dataset,
    observation_date: str,
    regrid_to_standard: bool,
) -> Dict[str, Any]:
    lat_key = "lat" if "lat" in ds.coords else "latitude"
    lon_key = "lon" if "lon" in ds.coords else "longitude"
    src_lats = ds[lat_key].values
    src_lons = ds[lon_key].values

    tgt_lats, tgt_lons = generate_target_grid_coords()
    obs_key = "obs_precip" if "obs_precip" in ds else "rainfall"
    raw_obs = ds[obs_key].values

    if regrid_to_standard and (len(src_lats) != len(tgt_lats) or len(src_lons) != len(tgt_lons)):
        regridded = regrid_2d_field(raw_obs, src_lats, src_lons, tgt_lats, tgt_lons)
        obs_arr = regridded
    else:
        obs_arr = raw_obs.astype(np.float32)

    return {
        "lats": tgt_lats if regrid_to_standard else src_lats,
        "lons": tgt_lons if regrid_to_standard else src_lons,
        "observation_date": observation_date,
        "obs_precip": obs_arr,
        "dataset": ds,
    }
