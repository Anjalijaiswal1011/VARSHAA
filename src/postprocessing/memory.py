"""
Multi-Scale Recent Error Memory Engine for RAIN-REPAIR X.
Computes strictly leakage-safe 3-day, 7-day, and 14-day rolling error memory:
    recent_bias (mean signed error: Obs - NWP)
    recent_mae  (mean absolute error)
    recent_rmse (root mean squared error)
    recent_error_std (error standard deviation / volatility)

Enforces strict causal guardrails: for forecast cycle T_0, only error verifications
from dates strictly prior to T_0 (t < T_0) may enter memory.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.exceptions import TemporalLeakageError
from src.utils.logging import get_logger

logger = get_logger("rain_repair.postprocessing.memory")


class LeakageSafeErrorMemoryStore:
    """
    Chronological verification store maintaining historical NWP error records:
        error = observed_precip - raw_nwp_precip

    Keyed by verification date ('YYYY-MM-DD').
    Provides point-level and grid-level multi-timescale trailing error statistics.
    """

    def __init__(self, max_history_days: int = 30):
        self.max_history_days = max_history_days
        # date_str -> Dict[grid_id or (lat, lon), float or np.ndarray error]
        self._history: Dict[str, Dict[str, float]] = {}
        # date_str -> pd.DataFrame of verified records
        self._dataframe_history: Dict[str, pd.DataFrame] = {}

    def record_verified_cycle(
        self,
        verification_date: str,
        df_cycle: pd.DataFrame,
        grid_id_col: str = "grid_id",
        obs_col: str = "obs_precip",
        nwp_col: str = "nwp_precip",
    ) -> None:
        """
        Record verified errors for a historical cycle after ground truth observations are received.
        verification_date: date string 'YYYY-MM-DD' of the verified forecast.
        """
        date_str = str(pd.to_datetime(verification_date).strftime("%Y-%m-%d"))

        if obs_col not in df_cycle.columns or nwp_col not in df_cycle.columns:
            raise ValueError(f"Columns {obs_col} and {nwp_col} are required to record verified error.")

        errors = (df_cycle[obs_col].to_numpy() - df_cycle[nwp_col].to_numpy()).astype(np.float32)

        if grid_id_col in df_cycle.columns:
            grid_keys = df_cycle[grid_id_col].astype(str).tolist()
        elif "lat" in df_cycle.columns and "lon" in df_cycle.columns:
            grid_keys = [f"{lat:.2f}_{lon:.2f}" for lat, lon in zip(df_cycle["lat"], df_cycle["lon"])]
        else:
            grid_keys = [f"point_{i}" for i in range(len(df_cycle))]

        grid_error_map = {grid_keys[i]: float(errors[i]) for i in range(len(grid_keys))}
        self._history[date_str] = grid_error_map

        # Retain minimal columns for regime/lead-time conditioned memory if available
        keep_cols = [c for c in [grid_id_col, "lat", "lon", "lead_time", "regime"] if c in df_cycle.columns]
        stored_df = df_cycle[keep_cols].copy() if keep_cols else pd.DataFrame(index=range(len(df_cycle)))
        stored_df["signed_error"] = errors
        self._dataframe_history[date_str] = stored_df

        # Prune oldest if exceeds capacity
        sorted_dates = sorted(self._history.keys())
        if len(sorted_dates) > self.max_history_days:
            for old_date in sorted_dates[:-self.max_history_days]:
                del self._history[old_date]
                if old_date in self._dataframe_history:
                    del self._dataframe_history[old_date]

    def get_valid_past_dates(self, forecast_cycle_date: str) -> List[str]:
        """
        Retrieve strictly past verification dates:
            verification_date < forecast_cycle_date

        Raises TemporalLeakageError if any returned date is >= forecast_cycle_date.
        """
        curr_dt = pd.to_datetime(forecast_cycle_date)
        valid_dates = sorted(
            [d for d in self._history.keys() if pd.to_datetime(d) < curr_dt],
            reverse=True,
        )

        for d in valid_dates:
            if pd.to_datetime(d) >= curr_dt:
                raise TemporalLeakageError(
                    f"CRITICAL CAUSALITY VIOLATION: Verification date {d} is >= forecast cycle date {forecast_cycle_date}! "
                    f"Future observations must never enter error memory."
                )

        return valid_dates

    def extract_recent_memory_features(
        self,
        forecast_cycle_date: str,
        grid_ids: Optional[List[str]] = None,
        default_error: float = 0.0,
    ) -> pd.DataFrame:
        """
        Extract 3-day, 7-day, and 14-day rolling error memory statistics:
            - recent_bias_{3,7,14}d
            - recent_mae_{3,7,14}d
            - recent_rmse_{3,7,14}d
            - recent_error_std_{3,7,14}d

        Strictly enforces that dates >= forecast_cycle_date are inaccessible.
        """
        past_dates = self.get_valid_past_dates(forecast_cycle_date)
        n_points = len(grid_ids) if grid_ids is not None else 1

        windows = {"3d": 3, "7d": 7, "14d": 14}
        features: Dict[str, np.ndarray] = {}

        for win_name, win_len in windows.items():
            win_dates = past_dates[:win_len]
            if not win_dates:
                # Cold start: neutral zeros
                features[f"recent_bias_{win_name}"] = np.full(n_points, default_error, dtype=np.float32)
                features[f"recent_mae_{win_name}"] = np.full(n_points, default_error, dtype=np.float32)
                features[f"recent_rmse_{win_name}"] = np.full(n_points, default_error, dtype=np.float32)
                features[f"recent_error_std_{win_name}"] = np.full(n_points, 0.0, dtype=np.float32)
                continue

            # Assemble error matrix: (len(win_dates), n_points)
            matrix = np.zeros((len(win_dates), n_points), dtype=np.float32)
            for row_idx, d in enumerate(win_dates):
                day_map = self._history[d]
                if grid_ids is not None:
                    matrix[row_idx, :] = [day_map.get(gid, default_error) for gid in grid_ids]
                else:
                    matrix[row_idx, :] = list(day_map.values())[0] if day_map else default_error

            # Compute statistics across trailing time dimension
            bias = np.mean(matrix, axis=0)
            mae = np.mean(np.abs(matrix), axis=0)
            rmse = np.sqrt(np.mean(matrix**2, axis=0))
            std = np.std(matrix, axis=0)

            features[f"recent_bias_{win_name}"] = bias.astype(np.float32)
            features[f"recent_mae_{win_name}"] = mae.astype(np.float32)
            features[f"recent_rmse_{win_name}"] = rmse.astype(np.float32)
            features[f"recent_error_std_{win_name}"] = std.astype(np.float32)

        return pd.DataFrame(features)

    def attach_memory_to_dataframe(
        self,
        df: pd.DataFrame,
        cycle_date_col: str = "cycle_date",
        grid_id_col: str = "grid_id",
    ) -> pd.DataFrame:
        """
        Attach 3/7/14-day memory features to each row in a forecast DataFrame.
        Maintains strict causality row-by-row or group-by-cycle.
        """
        df_out = df.copy()

        # Generate grid identifier if not present
        if grid_id_col not in df_out.columns:
            if "lat" in df_out.columns and "lon" in df_out.columns:
                grid_keys = [f"{lat:.2f}_{lon:.2f}" for lat, lon in zip(df_out["lat"], df_out["lon"])]
            else:
                grid_keys = [f"grid_{i}" for i in range(len(df_out))]
        else:
            grid_keys = df_out[grid_id_col].astype(str).tolist()

        if cycle_date_col in df_out.columns:
            unique_dates = df_out[cycle_date_col].unique()
            # Initialize empty memory columns
            mem_cols = [
                "recent_bias_3d", "recent_mae_3d", "recent_rmse_3d", "recent_error_std_3d",
                "recent_bias_7d", "recent_mae_7d", "recent_rmse_7d", "recent_error_std_7d",
                "recent_bias_14d", "recent_mae_14d", "recent_rmse_14d", "recent_error_std_14d",
            ]
            for col in mem_cols:
                df_out[col] = 0.0

            for u_date in unique_dates:
                mask = df_out[cycle_date_col] == u_date
                sub_grid_ids = [grid_keys[i] for i in range(len(df_out)) if mask.iloc[i]]
                mem_sub = self.extract_recent_memory_features(str(u_date), sub_grid_ids)
                for col in mem_cols:
                    df_out.loc[mask, col] = mem_sub[col].to_numpy()
        else:
            # Fallback to today / cold-start
            mem_df = self.extract_recent_memory_features("2099-01-01", grid_keys)
            for col in mem_df.columns:
                df_out[col] = mem_df[col]

        return df_out
