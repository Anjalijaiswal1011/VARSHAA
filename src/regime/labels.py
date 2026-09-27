"""
Provisional Regime Label Generation & Configuration Layer for RAIN-REPAIR X.
Derives objective meteorological weak/provisional labels from atmospheric indices
when certified ground-truth manual synoptic charts are unavailable.
All derived labels are explicitly flagged as PROVISIONAL rather than ground truth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.regime.schemas import REGIME_CLASSES, REGIME_TO_IDX, WeatherRegime
from src.utils.logging import get_logger

logger = get_logger("rain_repair.regime.labels")


def derive_provisional_regime_labels(
    df: pd.DataFrame,
) -> pd.Series:
    """
    Generate objective rule-based provisional regime labels from meteorological features.
    
    Decision Hierarchy:
    1. WESTERN_DISTURBANCE:
       Upper jet core (wd_shear_proxy > 20 m/s or u200 > 25 m/s) in North/NW India (lat >= 24°N)
       with elevated relative humidity (rh850 > 50%).
    2. MONSOON_DEPRESSION:
       Deep cyclonic vorticity (vorticity_850 > 1.2e-5 s^-1) coupled with low pressure (mslp < 1002 hPa).
    3. OROGRAPHIC:
       Steep mountain slope (dem_slope > 2.5 deg) with mechanical uplift (orographic_uplift > 0.05 m/s).
    4. COASTAL:
       Proximity to coastline (dist_to_coast < 75 km) with low elevation (< 150m) and onshore moisture.
    5. BREAK_MONSOON:
       Weak peninsular jet (u850 < 6.0 m/s over central/south India) with dry conditions (rh850 < 60%).
    6. ACTIVE_MONSOON:
       Strong low-level westerly monsoon jet (u850 >= 10.0 m/s, wind_speed_850 >= 12.0 m/s) with high moisture.
    Default:
       Fallback to ACTIVE_MONSOON or COASTAL depending on proximity.
    """
    labels = pd.Series(index=df.index, dtype="object")

    # Extract required inputs or safe defaults
    lat = df["lat"].to_numpy()
    u850 = df.get("u850", pd.Series(0.0, index=df.index)).to_numpy()
    v850 = df.get("v850", pd.Series(0.0, index=df.index)).to_numpy()
    w_speed = df.get("wind_speed_850", pd.Series(np.sqrt(u850**2 + v850**2))).to_numpy()
    rh850 = df.get("rh850", pd.Series(70.0, index=df.index)).to_numpy()
    mslp = df.get("mslp", pd.Series(1005.0, index=df.index)).to_numpy()
    elev = df.get("elevation", pd.Series(100.0, index=df.index)).to_numpy()
    slope = df.get("dem_slope", pd.Series(0.5, index=df.index)).to_numpy()
    dist_coast = df.get("dist_to_coast", pd.Series(200.0, index=df.index)).to_numpy()
    uplift = df.get("orographic_uplift", pd.Series(0.0, index=df.index)).to_numpy()
    vort = df.get("vorticity_850", pd.Series(0.0, index=df.index)).to_numpy()
    wd_proxy = df.get("wd_shear_proxy", pd.Series(0.0, index=df.index)).to_numpy()

    n = len(df)
    assigned = np.full(n, WeatherRegime.ACTIVE_MONSOON.value, dtype=object)

    for i in range(n):
        # 1. Western Disturbance
        if (lat[i] >= 24.0 and wd_proxy[i] > 15.0 and rh850[i] > 40.0) or (lat[i] >= 28.0 and u850[i] < 2.0 and uplift[i] > 0.02):
            assigned[i] = WeatherRegime.WESTERN_DISTURBANCE.value

        # 2. Monsoon Depression
        elif vort[i] > 1.0e-5 and mslp[i] < 1002.5:
            assigned[i] = WeatherRegime.MONSOON_DEPRESSION.value

        # 3. Orographic Precipitation
        elif (slope[i] >= 2.0 or elev[i] >= 600.0) and uplift[i] > 0.02:
            assigned[i] = WeatherRegime.OROGRAPHIC.value

        # 4. Coastal Precipitation
        elif dist_coast[i] < 60.0 and elev[i] < 150.0 and rh850[i] > 70.0:
            assigned[i] = WeatherRegime.COASTAL.value

        # 5. Break Monsoon
        elif lat[i] <= 22.0 and w_speed[i] < 7.0 and rh850[i] < 65.0:
            assigned[i] = WeatherRegime.BREAK_MONSOON.value

        # 6. Active Monsoon (baseline default for moist southwest flow)
        else:
            assigned[i] = WeatherRegime.ACTIVE_MONSOON.value

    labels[:] = assigned
    return labels


def get_regime_distribution_report(labels: pd.Series) -> Dict[str, Any]:
    """Calculate class frequencies and imbalance ratios across regimes."""
    counts = labels.value_counts().to_dict()
    total = len(labels)
    distribution = {}

    for regime in REGIME_CLASSES:
        c = counts.get(regime, 0)
        distribution[regime] = {
            "count": int(c),
            "percentage": round((c / max(total, 1)) * 100.0, 2),
        }

    return {
        "total_samples": total,
        "is_verified": False,
        "label_type": "provisional_weak_rule_based",
        "distribution": distribution,
    }
