"""
Standardized Grid-Level Forecast Schema and Scientific Quality Check Engine (PART 8).
Validates raw and post-processed grid predictions before geospatial aggregation.
Does NOT silently repair invalid scientific output — flags invalid records explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("rain_repair.gis.grid_schema")

SUPPORTED_LEAD_TIMES: Set[int] = {24, 48, 72, 96, 120}
MIN_LAT: float = 6.0
MAX_LAT: float = 38.5
MIN_LON: float = 68.0
MAX_LON: float = 98.0


@dataclass
class GridForecastRecord:
    """
    Standardized Grid-Level Forecast Record.
    Conforms strictly to PART 8 Section 3 specification.
    """
    forecast_time: str
    valid_time: str
    grid_id: str
    latitude: float
    longitude: float
    lead_time: int
    raw_nwp_rainfall: float
    corrected_rainfall: float
    p50_rainfall: float
    p75_rainfall: float
    p90_rainfall: float
    heavy_probability: float
    very_heavy_probability: float
    extreme_probability: float
    regime: str
    regime_probabilities: Dict[str, float]
    uncertainty_indicator: float
    model_version: str
    evt_status: str
    quality_flag: str = "VALID"
    quality_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GridForecastQualityValidator:
    """
    Scientific Quality Validator for Grid Forecast DataFrames and Records.
    Flags invalid records rather than performing silent data manipulation.
    """

    @classmethod
    def validate_record(cls, record: GridForecastRecord) -> Tuple[bool, List[str]]:
        """
        Validates a single GridForecastRecord against meteorological domain constraints.
        """
        errors: List[str] = []

        # 1. Coordinate spatial bounds
        if not (MIN_LAT <= record.latitude <= MAX_LAT):
            errors.append(f"Latitude {record.latitude} outside India meteorological domain [{MIN_LAT}, {MAX_LAT}]")
        if not (MIN_LON <= record.longitude <= MAX_LON):
            errors.append(f"Longitude {record.longitude} outside India meteorological domain [{MIN_LON}, {MAX_LON}]")

        # 2. Non-negative rainfall constraints
        for field_name, val in [
            ("raw_nwp_rainfall", record.raw_nwp_rainfall),
            ("corrected_rainfall", record.corrected_rainfall),
            ("p50_rainfall", record.p50_rainfall),
            ("p75_rainfall", record.p75_rainfall),
            ("p90_rainfall", record.p90_rainfall),
        ]:
            if val < 0.0 or np.isnan(val):
                errors.append(f"{field_name} must be non-negative real, got {val}")

        # 3. Probability bounds [0, 1]
        for prob_name, p_val in [
            ("heavy_probability", record.heavy_probability),
            ("very_heavy_probability", record.very_heavy_probability),
            ("extreme_probability", record.extreme_probability),
        ]:
            if not (0.0 <= p_val <= 1.0) or np.isnan(p_val):
                errors.append(f"{prob_name} must be in [0, 1], got {p_val}")

        # 4. Quantile ordering: P50 <= P75 <= P90
        # Allow small floating point tolerance
        if record.p50_rainfall > record.p75_rainfall + 1e-4:
            errors.append(f"Quantile crossing: p50 ({record.p50_rainfall}) > p75 ({record.p75_rainfall})")
        if record.p75_rainfall > record.p90_rainfall + 1e-4:
            errors.append(f"Quantile crossing: p75 ({record.p75_rainfall}) > p90 ({record.p90_rainfall})")

        # 5. Timestamp validation
        for time_field, t_str in [("forecast_time", record.forecast_time), ("valid_time", record.valid_time)]:
            try:
                # Accept ISO 8601 strings
                if "T" in t_str:
                    datetime.fromisoformat(t_str.replace("Z", "+00:00"))
                else:
                    datetime.strptime(t_str, "%Y-%m-%d")
            except Exception:
                errors.append(f"Invalid timestamp format for {time_field}: {t_str}")

        # 6. Lead time validation
        if record.lead_time not in SUPPORTED_LEAD_TIMES:
            errors.append(f"Unsupported lead time {record.lead_time}. Supported: {sorted(SUPPORTED_LEAD_TIMES)}")

        is_valid = len(errors) == 0
        return is_valid, errors

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates an entire grid forecast DataFrame.
        Adds 'quality_flag' ('VALID' | 'INVALID') and 'quality_errors' columns.
        Checks for duplicate (grid_id, lead_time, forecast_time) records.
        """
        out_df = df.copy()
        flags: List[str] = []
        error_lists: List[List[str]] = []

        # Check duplicates
        dup_mask = out_df.duplicated(subset=["grid_id", "lead_time", "forecast_time"], keep=False) if {"grid_id", "lead_time", "forecast_time"}.issubset(out_df.columns) else pd.Series(False, index=out_df.index)

        for idx, row in out_df.iterrows():
            row_errors: List[str] = []
            if dup_mask.iloc[idx]:
                row_errors.append("Duplicate grid_id, lead_time, and forecast_time record")

            lat = float(row.get("latitude", row.get("lat", 0.0)))
            lon = float(row.get("longitude", row.get("lon", 0.0)))
            if not (MIN_LAT <= lat <= MAX_LAT):
                row_errors.append(f"Latitude {lat} out of bounds")
            if not (MIN_LON <= lon <= MAX_LON):
                row_errors.append(f"Longitude {lon} out of bounds")

            # Rainfall non-negativity
            for col in ["raw_nwp_rainfall", "raw_nwp_precip", "corrected_rainfall", "p50_rainfall", "corrected_p50"]:
                if col in row:
                    val = float(row[col])
                    if val < 0.0 or np.isnan(val):
                        row_errors.append(f"{col} is negative or NaN: {val}")

            # Quantiles ordering
            p50 = float(row.get("p50_rainfall", row.get("corrected_p50", 0.0)))
            p75 = float(row.get("p75_rainfall", row.get("corrected_p75", p50)))
            p90 = float(row.get("p90_rainfall", row.get("corrected_p90", p75)))
            if p50 > p75 + 1e-4:
                row_errors.append(f"Quantile crossing: p50 ({p50}) > p75 ({p75})")
            if p75 > p90 + 1e-4:
                row_errors.append(f"Quantile crossing: p75 ({p75}) > p90 ({p90})")

            # Probability bounds
            for p_col in ["heavy_probability", "prob_heavy_rain", "very_heavy_probability", "prob_very_heavy_rain"]:
                if p_col in row:
                    pval = float(row[p_col])
                    if not (0.0 <= pval <= 1.0) or np.isnan(pval):
                        row_errors.append(f"{p_col} out of [0, 1]: {pval}")

            if row_errors:
                flags.append("INVALID")
                error_lists.append(row_errors)
            else:
                flags.append("VALID")
                error_lists.append([])

        out_df["quality_flag"] = flags
        out_df["quality_errors"] = error_lists

        valid_count = sum(1 for f in flags if f == "VALID")
        invalid_count = len(flags) - valid_count

        summary = {
            "total_records": len(df),
            "valid_records": valid_count,
            "invalid_records": invalid_count,
            "validation_passed": invalid_count == 0,
        }
        return out_df, summary
