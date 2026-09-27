"""
Unified Feature Engineering Pipeline Orchestrator for RAIN-REPAIR X (VARSHAA).
Assembles all feature groups:
Base NWP -> Atmospheric & Physics -> Terrain -> Spatial Context -> Climatology -> Temporal -> Error Memory -> Regime Support.
Validates against FEATURE_REGISTRY and executes automated temporal leakage audits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.standardize import generate_target_grid_coords
from src.features.climatology import ClimatologyEngine
from src.features.diagnostics import generate_feature_diagnostics_report
from src.features.error_memory import ErrorMemoryBuffer
from src.features.leakage_audit import audit_feature_dataframe_leakage
from src.features.physics import (
    compute_deep_layer_shear,
    compute_moisture_convergence_proxy,
    compute_orographic_uplift,
    compute_regime_support_diagnostics,
    compute_relative_vorticity,
    compute_vapor_pressure_proxy,
    compute_wind_speed_and_direction,
    compute_windward_leeward_index,
)
from src.features.scaling import handle_missing_features, sanitize_domain_outliers
from src.features.schema import FEATURE_REGISTRY, validate_dataframe_against_schema
from src.features.spatial import compute_spatial_neighborhood_context
from src.features.temporal import compute_temporal_features
from src.features.terrain import compute_distance_to_coast, compute_terrain_derivatives
from src.utils.config import config
from src.utils.logging import get_logger

logger = get_logger("rain_repair.features.pipeline")


class FeatureEngineeringPipeline:
    """
    End-to-end Feature Engineering Pipeline transforming aligned data into
    scientifically validated, leakage-safe feature matrices.
    """

    def __init__(
        self,
        climatology_engine: Optional[ClimatologyEngine] = None,
        error_buffer: Optional[ErrorMemoryBuffer] = None,
        spatial_kernel_size: int = 3,
        data_pipeline: Optional[Any] = None,
    ):
        from src.data.pipeline import DataPipeline

        self.lats, self.lons = generate_target_grid_coords()
        self.grid_shape = (len(self.lats), len(self.lons))
        self.spatial_kernel_size = spatial_kernel_size
        self.climatology_engine = climatology_engine or ClimatologyEngine()
        self.error_buffer = error_buffer or ErrorMemoryBuffer()
        self.data_pipeline = data_pipeline or DataPipeline(error_buffer=self.error_buffer)

        logger.info(
            "Initialized FeatureEngineeringPipeline with spatial kernel %dx%d.",
            self.spatial_kernel_size,
            self.spatial_kernel_size,
        )

    def extract_features(
        self,
        nwp_data_or_path: Union[str, Path, Dict[str, Any]],
        cycle_date: str,
        lead_time_hours: int = 24,
        obs_data_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        execute_leakage_audit: bool = True,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
        """
        Execute end-to-end feature extraction for a forecast cycle.

        Args:
            nwp_data_or_path: NWP forecast input.
            cycle_date: Forecast cycle date string 'YYYY-MM-DD'.
            lead_time_hours: Forecast lead time (24, 48, 72, 96, 120).
            obs_data_or_path: Optional ground truth observations.
            execute_leakage_audit: If True, validates zero leakage before returning.

        Returns:
            Tuple of:
            - X: pd.DataFrame of feature predictors
            - y: Optional pd.Series of target observation labels (None in inference mode)
            - metadata: Dict with diagnostics, schema validation, and leakage reports
        """
        logger.info("Extracting features for cycle %s at lead time %dh", cycle_date, lead_time_hours)

        # 1. Base Data Processing via DataPipeline
        base_df = self.data_pipeline.process_cycle(
            nwp_data_or_path=nwp_data_or_path,
            cycle_date=cycle_date,
            lead_time_hours=lead_time_hours,
            obs_data_or_path=obs_data_or_path,
        )

        # Reshape primary grids for 2D spatial & physics operations
        raw_precip_2d = base_df["nwp_precip"].to_numpy().reshape(self.grid_shape)
        elev_2d = base_df["elevation"].to_numpy().reshape(self.grid_shape)
        u850_2d = base_df["u850"].to_numpy().reshape(self.grid_shape)
        v850_2d = base_df["v850"].to_numpy().reshape(self.grid_shape)
        rh850_2d = base_df["rh850"].to_numpy().reshape(self.grid_shape)
        t2m_2d = base_df["t2m"].to_numpy().reshape(self.grid_shape)
        mslp_2d = base_df["mslp"].to_numpy().reshape(self.grid_shape)
        u200_2d = np.full(self.grid_shape, -15.0, dtype=np.float32)
        dist_to_coast_2d = base_df["dist_to_coast"].to_numpy().reshape(self.grid_shape)

        # 2. Physics Derivatives
        wind_speed_850, wind_dir_850 = compute_wind_speed_and_direction(u850_2d, v850_2d)
        vapor_pressure = compute_vapor_pressure_proxy(t2m_2d, rh850_2d)
        aspect_2d = base_df["dem_aspect"].to_numpy().reshape(self.grid_shape)
        windward_leeward = compute_windward_leeward_index(wind_dir_850, aspect_2d)

        # Regime support diagnostics
        regime_diag = compute_regime_support_diagnostics(
            u850=u850_2d,
            v850=v850_2d,
            u200=u200_2d,
            mslp=mslp_2d,
            lats=self.lats,
            lons=self.lons,
            dist_to_coast=dist_to_coast_2d,
        )

        # 3. Spatial Context & Neighborhood Features
        spatial_ctx = compute_spatial_neighborhood_context(
            precip_grid=raw_precip_2d,
            elevation_grid=elev_2d,
            lats=self.lats,
            lons=self.lons,
            kernel_size=self.spatial_kernel_size,
        )

        # 4. Climatology Baseline & Anomaly
        clim_features = self.climatology_engine.get_climatology_features(
            cycle_date=cycle_date,
            nwp_precip_grid=raw_precip_2d,
        )

        # 5. Temporal Harmonics & Lead Time
        temp_dict = compute_temporal_features(cycle_date, lead_time_hours)

        # 6. Assemble Comprehensive Feature Table
        X = pd.DataFrame(
            {
                # Identification / Coordinates
                "cycle_date": cycle_date,
                "lead_time": lead_time_hours,
                "lat": base_df["lat"],
                "lon": base_df["lon"],
                # Group A: NWP Features
                "nwp_precip": base_df["nwp_precip"],
                "nwp_precip_log": np.log1p(base_df["nwp_precip"].clip(lower=0.0)),
                "lead_time_days": temp_dict["lead_time_days"],
                # Group B: Atmospheric Features
                "t2m": base_df["t2m"],
                "rh850": base_df["rh850"],
                "mslp": base_df["mslp"],
                "u850": base_df["u850"],
                "v850": base_df["v850"],
                "wind_speed_850": wind_speed_850.ravel(),
                "wind_direction_850": wind_dir_850.ravel(),
                "vapor_pressure_proxy": vapor_pressure.ravel(),
                # Group C: Physics-Guided Features
                "orographic_uplift": base_df["orographic_uplift"],
                "moisture_flux_conv": base_df["moisture_flux_conv"],
                "vorticity_850": base_df["vorticity_850"],
                "bulk_wind_shear": base_df["bulk_wind_shear"],
                "windward_leeward_index": windward_leeward.ravel(),
                # Group D: Terrain Features
                "elevation": base_df["elevation"],
                "dem_slope": base_df["dem_slope"],
                "dem_aspect": base_df["dem_aspect"],
                "dist_to_coast": base_df["dist_to_coast"],
                "relative_elevation_exposure": spatial_ctx["relative_elevation_exposure"].ravel(),
                # Group E: Spatial Context Features
                "nwp_precip_spatial_mean_3x3": spatial_ctx[f"nwp_precip_spatial_mean_{self.spatial_kernel_size}x{self.spatial_kernel_size}"].ravel(),
                "nwp_precip_spatial_max_3x3": spatial_ctx[f"nwp_precip_spatial_max_{self.spatial_kernel_size}x{self.spatial_kernel_size}"].ravel(),
                "nwp_precip_spatial_std_3x3": spatial_ctx[f"nwp_precip_spatial_std_{self.spatial_kernel_size}x{self.spatial_kernel_size}"].ravel(),
                "nwp_precip_spatial_gradient": spatial_ctx["nwp_precip_spatial_gradient"].ravel(),
                # Group F: Climatology Features
                "clim_mean_doy": clim_features["clim_mean_doy"].ravel(),
                "nwp_clim_anomaly": clim_features["nwp_clim_anomaly"].ravel(),
                # Group G: Temporal Features
                "doy_sin": temp_dict["doy_sin"],
                "doy_cos": temp_dict["doy_cos"],
                "month_sin": temp_dict["month_sin"],
                "month_cos": temp_dict["month_cos"],
                # Group H: Multi-Scale Error Memory
                "error_memory_lag1": base_df["error_memory_lag1"],
                "error_memory_lag3_mean": base_df["error_memory_lag3_mean"],
                "error_memory_lag7_mean": base_df["error_memory_lag7_mean"],
                "error_memory_lag14_mean": base_df["error_memory_lag14_mean"],
                # Group I: Regime Support Diagnostics
                "wd_shear_proxy": regime_diag["wd_shear_proxy"].ravel(),
                "monsoon_trough_mslp_gradient": regime_diag["monsoon_trough_mslp_gradient"].ravel(),
                "offshore_trough_coastal_shear": regime_diag["offshore_trough_coastal_shear"].ravel(),
            }
        )

        # 7. Quality Sanitization & Missing Value Imputation
        X = sanitize_domain_outliers(X, preserve_extreme_weather=True)
        X = handle_missing_features(X)

        # 8. Target Label Extraction (Separated from X to guarantee zero leakage)
        y: Optional[pd.Series] = None
        if "obs_precip" in base_df.columns:
            y = base_df["obs_precip"].copy()

        # 9. Automated Leakage Audit & Schema Validation
        leakage_report = {}
        if execute_leakage_audit:
            leakage_report = audit_feature_dataframe_leakage(X, cycle_date, strict=True)

        schema_report = validate_dataframe_against_schema(X, allow_missing_cols=False)
        diagnostics_report = generate_feature_diagnostics_report(X)

        metadata = {
            "cycle_date": cycle_date,
            "lead_time_hours": lead_time_hours,
            "total_grid_points": len(X),
            "total_features": len(X.columns) - 4,  # Minus coordinate/id cols
            "leakage_audit": leakage_report,
            "schema_validation": schema_report,
            "diagnostics": diagnostics_report,
        }

        logger.info(
            "Feature extraction complete for cycle %s: %d points x %d features.",
            cycle_date,
            len(X),
            metadata["total_features"],
        )

        return X, y, metadata
