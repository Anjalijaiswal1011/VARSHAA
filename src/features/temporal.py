"""
Temporal and Calendar Feature Extraction Module for RAIN-REPAIR X.
Computes smooth cyclical harmonics (sin/cos) for Day of Year and Month,
and standardizes monsoon seasonal phase encodings.
"""

from __future__ import annotations

from typing import Dict, Union
import numpy as np
import pandas as pd


def compute_temporal_features(
    date_str: Union[str, pd.Timestamp],
    lead_time_hours: int = 24,
) -> Dict[str, float]:
    """
    Generate cyclical calendar and forecast lead-time features.

    Args:
        date_str: ISO date string 'YYYY-MM-DD' or pd.Timestamp.
        lead_time_hours: Forecast lead time in hours (24, 48, 72, 96, 120).

    Returns:
        Dictionary of temporal features:
        - 'doy_sin', 'doy_cos'
        - 'month_sin', 'month_cos'
        - 'lead_time': int
        - 'lead_time_days': float
        - 'monsoon_phase_code': int (0=Dry/Winter, 1=Pre-Monsoon, 2=Early Monsoon, 3=Peak Monsoon, 4=Late/Retreating)
    """
    dt = pd.to_datetime(date_str)
    doy = dt.dayofyear
    month = dt.month

    # Cyclical day of year harmonics
    doy_rad = 2.0 * np.pi * (doy / 365.25)
    doy_sin = float(np.sin(doy_rad))
    doy_cos = float(np.cos(doy_rad))

    # Cyclical month harmonics
    month_rad = 2.0 * np.pi * (month / 12.0)
    month_sin = float(np.sin(month_rad))
    month_cos = float(np.cos(month_rad))

    # Monsoon phase categorization (Indian Subcontinent Climatology)
    if month in [11, 12, 1, 2]:
        phase_code = 0  # Winter / Northeast dry season
    elif month in [3, 4, 5]:
        phase_code = 1  # Pre-Monsoon convective thunder / Western Disturbance season
    elif month == 6:
        phase_code = 2  # Southwest Monsoon Onset (Early)
    elif month in [7, 8]:
        phase_code = 3  # Peak Southwest Monsoon
    else:  # month in [9, 10]
        phase_code = 4  # Late Monsoon / Withdrawal phase

    return {
        "doy_sin": np.round(doy_sin, 6),
        "doy_cos": np.round(doy_cos, 6),
        "month_sin": np.round(month_sin, 6),
        "month_cos": np.round(month_cos, 6),
        "lead_time": lead_time_hours,
        "lead_time_days": float(lead_time_hours) / 24.0,
        "monsoon_phase_code": phase_code,
    }
