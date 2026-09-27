"""
Master Data Pipeline Orchestrator for RAIN-REPAIR X (VARSHAA).
Executes end-to-end data processing:
Ingestion -> Validation -> Standardization -> Physics/Terrain Features -> Error Memory -> Dataset Assembly.
Emits matrices conforming strictly to CONTRACT-DATA-001.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.ingest import load_nwp_forecast_file, load_observation_file, load_static_terrain_data
from src.data.standardize import generate_target_grid_coords, regrid_2d_field
from src.data.validate import validate_lead_times, validate_physical_bounds, validate_spatial_coordinates
from src.features.error_memory import ErrorMemoryBuffer
from src.features.physics import (
    compute_deep_layer_shear,
    compute_moisture_convergence_proxy,
    compute_orographic_uplift,
    compute_relative_vorticity,
)
from src.features.terrain import compute_distance_to_coast, compute_terrain_derivatives
from src.utils.config import config
from src.utils.exceptions import DataQualityError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.data.pipeline")


class DataPipeline:
    """
    Unified Data Processing Pipeline for model training and real-time inference.
    """

    def __init__(
        self,
        error_buffer: Optional[ErrorMemoryBuffer] = None,
        terrain_path: Optional[Union[str, Path]] = None,
    ):
        self.lats, self.lons = generate_target_grid_coords()
        self.error_buffer = error_buffer or ErrorMemoryBuffer()

        # Preload static terrain data
        terrain_data = load_static_terrain_data(terrain_path)
        self.elevation = terrain_data["elevation"]
        self.slope, self.aspect = compute_terrain_derivatives(self.elevation, self.lats, self.lons)
        self.dist_to_coast = compute_distance_to_coast(self.lats, self.lons)

        logger.info(
            "Initialized DataPipeline over Indian Domain: Lat [%.2f, %.2f], Lon [%.2f, %.2f], Shape %s",
            self.lats[0], self.lats[-1], self.lons[0], self.lons[-1], self.elevation.shape,
        )

    def process_cycle(
        self,
        nwp_data_or_path: Union[str, Path, Dict[str, Any]],
        cycle_date: str,
        lead_time_hours: int = 24,
        obs_data_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """
        Process a single forecast cycle:
        1. Ingest/standardize raw NWP fields.
        2. Ingest observations if available (training mode).
        3. Extract physics, terrain, and error memory features.
        4. Return standardized DataFrame adhering to CONTRACT-DATA-001.

        Args:
            nwp_data_or_path: Path to NWP file or pre-loaded dictionary of arrays.
            cycle_date: ISO date string 'YYYY-MM-DD'.
            lead_time_hours: Forecast lead time (24, 48, 72, 96, 120).
            obs_data_or_path: Optional ground truth observations for training/validation.

        Returns:
            pd.DataFrame containing feature columns and optional target label 'obs_precip'.
        """
        logger.info("Processing cycle %s at lead time %dh", cycle_date, lead_time_hours)
        validate_lead_times([lead_time_hours])

        grid_shape = (len(self.lats), len(self.lons))

        # 1. Ingest NWP
        if isinstance(nwp_data_or_path, (str, Path)):
            nwp_dict = load_nwp_forecast_file(nwp_data_or_path)
            if "dataframe" in nwp_dict:
                df_nwp = nwp_dict["dataframe"]
                if "lead_time" in df_nwp.columns:
                    df_filtered = df_nwp[df_nwp["lead_time"] == lead_time_hours]
                    if df_filtered.empty:
                        df_filtered = df_nwp.iloc[: len(self.lats) * len(self.lons)]
                else:
                    df_filtered = df_nwp.iloc[: len(self.lats) * len(self.lons)]

                raw_precip = df_filtered["nwp_precip"].to_numpy().reshape(grid_shape).astype(np.float32)
                vars_dict = {
                    col: df_filtered[col].to_numpy().reshape(grid_shape).astype(np.float32)
                    for col in ["t2m", "rh850", "u850", "v850", "mslp"]
                    if col in df_filtered.columns
                }
            else:
                raw_precip = nwp_dict["variables"]["nwp_precip"]
                vars_dict = nwp_dict["variables"]
        elif isinstance(nwp_data_or_path, dict):
            raw_precip = nwp_data_or_path["nwp_precip"]
            vars_dict = nwp_data_or_path
        else:
            raise ValueError(f"Unsupported NWP input type: {type(nwp_data_or_path)}")

        if raw_precip.shape != grid_shape:
            # Regrid if dimension mismatch
            src_lats = vars_dict.get("lats", self.lats)
            src_lons = vars_dict.get("lons", self.lons)
            raw_precip = regrid_2d_field(raw_precip, src_lats, src_lons, self.lats, self.lons)

        # Retrieve or compute default atmospheric variables
        u850 = vars_dict.get("u850", np.full(grid_shape, 12.0, dtype=np.float32))
        v850 = vars_dict.get("v850", np.full(grid_shape, 4.0, dtype=np.float32))
        rh850 = vars_dict.get("rh850", np.full(grid_shape, 75.0, dtype=np.float32))
        mslp = vars_dict.get("mslp", np.full(grid_shape, 1004.0, dtype=np.float32))
        t2m = vars_dict.get("t2m", np.full(grid_shape, 298.15, dtype=np.float32))
        u200 = vars_dict.get("u200", np.full(grid_shape, -15.0, dtype=np.float32))
        v200 = vars_dict.get("v200", np.full(grid_shape, 0.0, dtype=np.float32))

        # 2. Physics-guided Feature Calculations
        orog_uplift = compute_orographic_uplift(u850, v850, self.elevation, self.lats, self.lons)
        mfc_proxy = compute_moisture_convergence_proxy(u850, v850, rh850, self.lats, self.lons)
        vort_850 = compute_relative_vorticity(u850, v850, self.lats, self.lons)
        wind_shear = compute_deep_layer_shear(u850, v850, u200, v200)

        # 3. Retrieve Multi-Scale Error Memory
        error_mem = self.error_buffer.get_multi_timescale_error_memory(cycle_date, grid_shape)

        # 4. Construct Coordinate Meshgrids
        lat_grid, lon_grid = np.meshgrid(self.lats, self.lons, indexing="ij")

        # Calendar harmonics
        doy = pd.to_datetime(cycle_date).dayofyear
        doy_sin = float(np.sin(2 * np.pi * doy / 365.25))
        doy_cos = float(np.cos(2 * np.pi * doy / 365.25))

        # 5. Assemble Flat Tabular DataFrame
        df = pd.DataFrame(
            {
                "cycle_date": cycle_date,
                "lead_time": lead_time_hours,
                "lat": lat_grid.ravel(),
                "lon": lon_grid.ravel(),
                "nwp_precip": raw_precip.ravel(),
                "t2m": t2m.ravel(),
                "rh850": rh850.ravel(),
                "u850": u850.ravel(),
                "v850": v850.ravel(),
                "mslp": mslp.ravel(),
                "elevation": self.elevation.ravel(),
                "dem_slope": self.slope.ravel(),
                "dem_aspect": self.aspect.ravel(),
                "dist_to_coast": self.dist_to_coast.ravel(),
                "orographic_uplift": orog_uplift.ravel(),
                "moisture_flux_conv": mfc_proxy.ravel(),
                "vorticity_850": vort_850.ravel(),
                "bulk_wind_shear": wind_shear.ravel(),
                "error_memory_lag1": error_mem["error_memory_lag1"].ravel(),
                "error_memory_lag3_mean": error_mem["error_memory_lag3_mean"].ravel(),
                "error_memory_lag7_mean": error_mem["error_memory_lag7_mean"].ravel(),
                "error_memory_lag14_mean": error_mem["error_memory_lag14_mean"].ravel(),
                "doy_sin": doy_sin,
                "doy_cos": doy_cos,
            }
        )

        # 6. Ingest Optional Ground Observations
        if obs_data_or_path is not None:
            if isinstance(obs_data_or_path, (str, Path)):
                obs_dict = load_observation_file(obs_data_or_path, cycle_date)
                obs_precip = obs_dict["obs_precip"]
            elif isinstance(obs_data_or_path, dict):
                obs_precip = obs_data_or_path["obs_precip"]
            elif isinstance(obs_data_or_path, np.ndarray):
                obs_precip = obs_data_or_path
            else:
                raise ValueError(f"Unsupported observation format: {type(obs_data_or_path)}")

            if obs_precip.shape != grid_shape:
                if obs_precip.size == grid_shape[0] * grid_shape[1]:
                    obs_precip = obs_precip.reshape(grid_shape)
                else:
                    obs_precip = regrid_2d_field(obs_precip, self.lats, self.lons, self.lats, self.lons)

            df["obs_precip"] = obs_precip.ravel()

            # Record into error memory buffer for subsequent cycles
            self.error_buffer.record_historical_error(cycle_date, raw_precip, obs_precip)

        # Final quality assurance check
        validate_physical_bounds(df, strict=False)

        return df
