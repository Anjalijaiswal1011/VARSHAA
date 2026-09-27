"""
NWP Error DNA Feature Representation and Assembly Engine for RAIN-REPAIR X.
Builds the unified feature representation capturing the contextual fingerprint of NWP errors:
    Error DNA = f(
        lead_time,
        location & terrain,
        atmospheric context,
        spatial neighborhood,
        soft regime probabilities,
        regime transition state,
        recent multi-scale error memory (3/7/14d),
        historical analog memory
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.postprocessing.analog import AnalogMemory
from src.postprocessing.memory import LeakageSafeErrorMemoryStore
from src.postprocessing.target import validate_target_isolation
from src.regime.schemas import REGIME_CLASSES
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.error_dna")


class NWPErrorDNAExtractor:
    """
    Transforms meteorological contexts, regime outputs, and historical memory
    into a structured Error DNA feature table for post-processing regressors.
    """

    # 1. Base NWP & Lead Time
    BASE_NWP_COLS: List[str] = [
        "nwp_precip",
        "nwp_precip_log",
        "lead_time",
        "lead_time_days",
    ]

    # 2. Location & Terrain
    TERRAIN_COLS: List[str] = [
        "lat",
        "lon",
        "elevation",
        "dem_slope",
        "dem_aspect",
        "dist_to_coast",
    ]

    # 3. Atmospheric Dynamics & Physics
    ATMOSPHERIC_COLS: List[str] = [
        "t2m",
        "rh850",
        "mslp",
        "u850",
        "v850",
        "wind_speed_850",
        "wind_direction_850",
        "vapor_pressure_proxy",
        "orographic_uplift",
        "moisture_flux_conv",
        "vorticity_850",
        "bulk_wind_shear",
        "windward_leeward_index",
    ]

    # 4. Spatial & Climatology
    SPATIAL_CLIM_COLS: List[str] = [
        "nwp_precip_spatial_mean_3x3",
        "nwp_precip_spatial_max_3x3",
        "nwp_precip_spatial_std_3x3",
        "nwp_precip_spatial_gradient",
        "clim_mean_doy",
        "nwp_clim_anomaly",
        "doy_sin",
        "doy_cos",
    ]

    # 5. Soft Regime Probabilities (6 canonical classes)
    REGIME_PROB_COLS: List[str] = [
        f"prob_{cls_name.lower()}" for cls_name in REGIME_CLASSES
    ]

    # 6. Regime Transition Signals
    TRANSITION_COLS: List[str] = [
        "transition_strength",
        "transition_confidence",
        "is_transition_state",
    ]

    # 7. Recent Multi-Scale Error Memory (3d, 7d, 14d)
    RECENT_MEMORY_COLS: List[str] = [
        "recent_bias_3d", "recent_mae_3d", "recent_rmse_3d", "recent_error_std_3d",
        "recent_bias_7d", "recent_mae_7d", "recent_rmse_7d", "recent_error_std_7d",
        "recent_bias_14d", "recent_mae_14d", "recent_rmse_14d", "recent_error_std_14d",
    ]

    # 8. Historical Analog Error Signals
    ANALOG_COLS: List[str] = [
        "analog_mean_error",
        "analog_std_error",
        "analog_min_distance",
    ]

    def __init__(
        self,
        memory_store: Optional[LeakageSafeErrorMemoryStore] = None,
        analog_memory: Optional[AnalogMemory] = None,
        include_regimes: bool = True,
        include_transition: bool = True,
        include_recent_memory: bool = True,
        include_analog: bool = True,
    ):
        self.memory_store = memory_store
        self.analog_memory = analog_memory
        self.include_regimes = include_regimes
        self.include_transition = include_transition
        self.include_recent_memory = include_recent_memory
        self.include_analog = include_analog

    def get_feature_column_names(self, available_columns: Optional[List[str]] = None) -> List[str]:
        """Return the active feature subset based on configuration and availability."""
        cols: List[str] = []
        cols.extend(self.BASE_NWP_COLS)
        cols.extend(self.TERRAIN_COLS)
        cols.extend(self.ATMOSPHERIC_COLS)
        cols.extend(self.SPATIAL_CLIM_COLS)

        if self.include_regimes:
            cols.extend(self.REGIME_PROB_COLS)

        if self.include_transition:
            cols.extend(self.TRANSITION_COLS)

        if self.include_recent_memory:
            cols.extend(self.RECENT_MEMORY_COLS)

        if self.include_analog:
            cols.extend(self.ANALOG_COLS)

        if available_columns is not None:
            cols = [c for c in cols if c in available_columns]

        return cols

    def extract_error_dna(
        self,
        features_df: pd.DataFrame,
        regime_df: Optional[pd.DataFrame] = None,
        query_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Assemble the full Error DNA feature matrix for a forecast batch or cycle.
        Safeguards against target leakage.
        """
        # Strictly isolate and drop any target columns so they never enter feature extraction
        prohibited = [
            "obs_precip",
            "observation",
            "ground_truth",
            "target",
            "signed_error",
            "nwp_error",
            "absolute_error",
            "residual_error",
        ]
        drop_targets = [c for c in features_df.columns if c.lower() in prohibited]
        df = features_df.drop(columns=drop_targets, errors="ignore").copy()

        # 1. Ensure log transform and lead-time day conversion exist
        if "nwp_precip" in df.columns and "nwp_precip_log" not in df.columns:
            df["nwp_precip_log"] = np.log1p(np.maximum(df["nwp_precip"].to_numpy(), 0.0))

        if "lead_time" in df.columns and "lead_time_days" not in df.columns:
            df["lead_time_days"] = df["lead_time"] / 24.0

        # 2. Integrate Regime Intelligence & Transition Features if provided
        if regime_df is not None:
            for p_col in self.REGIME_PROB_COLS:
                if p_col in regime_df.columns:
                    df[p_col] = regime_df[p_col].to_numpy()
                elif p_col not in df.columns:
                    df[p_col] = 1.0 / len(REGIME_CLASSES)

            # Transition features
            if "transition_strength" in regime_df.columns:
                df["transition_strength"] = regime_df["transition_strength"].to_numpy()
            elif "transition_strength" not in df.columns:
                df["transition_strength"] = 0.0

            if "transition_confidence" in regime_df.columns:
                df["transition_confidence"] = regime_df["transition_confidence"].to_numpy()
            elif "transition_confidence" not in df.columns:
                df["transition_confidence"] = 0.5

            if "transition_status" in regime_df.columns:
                df["is_transition_state"] = (
                    regime_df["transition_status"].astype(str) == "TRANSITION"
                ).astype(np.float32)
            elif "is_transition_state" not in df.columns:
                df["is_transition_state"] = 0.0
        else:
            # Impute defaults if regime info not provided
            if self.include_regimes:
                for p_col in self.REGIME_PROB_COLS:
                    if p_col not in df.columns:
                        df[p_col] = 1.0 / len(REGIME_CLASSES)
            if self.include_transition:
                for t_col in self.TRANSITION_COLS:
                    if t_col not in df.columns:
                        df[t_col] = 0.0

        # 3. Integrate Multi-Scale Recent Error Memory if store is provided
        if self.include_recent_memory and self.memory_store is not None:
            cycle_date = query_date or str(df["cycle_date"].iloc[0] if "cycle_date" in df.columns else "2099-01-01")
            if "grid_id" in df.columns:
                grid_ids = df["grid_id"].astype(str).tolist()
            elif "lat" in df.columns and "lon" in df.columns:
                grid_ids = [f"{lat:.2f}_{lon:.2f}" for lat, lon in zip(df["lat"], df["lon"])]
            else:
                grid_ids = [f"point_{i}" for i in range(len(df))]
            mem_df = self.memory_store.extract_recent_memory_features(cycle_date, grid_ids=grid_ids)
            for m_col in self.RECENT_MEMORY_COLS:
                if m_col in mem_df.columns:
                    df[m_col] = mem_df[m_col].to_numpy()
        else:
            for m_col in self.RECENT_MEMORY_COLS:
                if m_col not in df.columns:
                    df[m_col] = 0.0

        # 4. Integrate Historical Analog Memory if analog index is provided
        if self.include_analog and self.analog_memory is not None:
            cycle_date_series = df["cycle_date"] if "cycle_date" in df.columns else ([query_date] * len(df) if query_date else None)
            analog_df = self.analog_memory.query(df, query_dates=cycle_date_series)
            for a_col in self.ANALOG_COLS:
                df[a_col] = analog_df[a_col].to_numpy()
        else:
            for a_col in self.ANALOG_COLS:
                if a_col not in df.columns:
                    df[a_col] = 0.0

        # Final column filtering based on requested active features
        active_cols = self.get_feature_column_names(list(df.columns))
        out_df = df[active_cols].fillna(0.0)
        validate_target_isolation(out_df)
        return out_df
