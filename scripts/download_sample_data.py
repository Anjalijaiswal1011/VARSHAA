"""
Sample Meteorological Dataset & Benchmark Fixture Generator for RAIN-REPAIR X.
Generates physically consistent synthetic NWP and Observation datasets across the
Indian subcontinent for development, automated testing, and CI verification.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.standardize import generate_target_grid_coords
from src.features.terrain import compute_terrain_derivatives
from src.utils.config import config
from src.utils.logging import setup_logger

logger = setup_logger("rain_repair.scripts.sample_data")


def generate_synthetic_monsoon_fixture(
    cycle_date: str = "2026-07-15",
    output_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Generate realistic synthetic NWP forecast and paired observation files
    representing an Active Monsoon state over India.

    Features modeled:
    - Strong southwesterly monsoon flow (westerlies over Arabian Sea)
    - Heavy orographic precipitation along the Western Ghats (lat 8°-19°N, lon 73°-75°E)
    - Monsoon Trough rainband stretching across Central India
    - Dry rain shadow over eastern Tamil Nadu / interior peninsula
    - Realistic NWP displacement and drizzle biases.
    """
    if output_dir is None:
        output_dir = config.paths.data_raw_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    lats, lons = generate_target_grid_coords()
    lat_grid, lon_grid = np.meshgrid(lats, lons, indexing="ij")

    # 1. Base Atmospheric State
    # Southwesterly monsoon winds (u > 0, v > 0)
    u850 = 12.0 + 4.0 * np.sin(np.radians(lat_grid))
    v850 = 5.0 + 3.0 * np.cos(np.radians(lon_grid))
    rh850 = np.clip(80.0 - 0.5 * (lat_grid - 15.0), 30.0, 98.0)
    mslp = 1008.0 - 8.0 * np.exp(-((lat_grid - 22.0) ** 2 + (lon_grid - 82.0) ** 2) / 60.0) # Monsoon low
    t2m = 299.15 - 0.5 * (lat_grid - 10.0)

    # 2. Synthetic Ground-Truth Rainfall
    # Orographic peak along Western Ghats
    wg_pattern = 95.0 * np.exp(-((lon_grid - 74.2) ** 2) / 0.8) * np.exp(-((lat_grid - 14.5) ** 2) / 30.0)

    # Monsoon trough band across Central India
    trough_pattern = 65.0 * np.exp(-((lat_grid - 22.5) ** 2) / 12.0) * np.exp(-((lon_grid - 83.0) ** 2) / 70.0)

    # True rainfall
    true_rain = wg_pattern + trough_pattern
    true_rain = np.where(true_rain < 1.0, 0.0, true_rain).astype(np.float32)

    # 3. Modeled NWP Forecast with Realistic Biases:
    # - Spatial displacement (shifted 0.3° east of crest)
    # - Drizzle bias (added 2-4 mm over dry areas)
    # - Under-prediction of extreme convective core
    nwp_wg = 75.0 * np.exp(-((lon_grid - 74.6) ** 2) / 1.0) * np.exp(-((lat_grid - 14.5) ** 2) / 30.0)
    nwp_trough = 55.0 * np.exp(-((lat_grid - 23.0) ** 2) / 14.0) * np.exp(-((lon_grid - 83.5) ** 2) / 70.0)
    nwp_drizzle = 2.5 * (rh850 > 65.0)  # NWP drizzle bias

    raw_nwp_rain = nwp_wg + nwp_trough + nwp_drizzle
    raw_nwp_rain = np.clip(raw_nwp_rain, 0.0, 500.0).astype(np.float32)

    # 4. Construct Multi-Lead Times (Day 1 to Day 5 with error growth)
    lead_times = [24, 48, 72]
    records = []

    for lt in lead_times:
        error_scale = 1.0 + (lt - 24) * 0.08  # Forecast degrades with lead time
        perturbed_nwp = np.clip(raw_nwp_rain * error_scale + np.random.normal(0, 1.5, raw_nwp_rain.shape), 0.0, 600.0)

        for i in range(len(lats)):
            for j in range(len(lons)):
                records.append(
                    {
                        "cycle_date": cycle_date,
                        "lead_time": lt,
                        "lat": float(lats[i]),
                        "lon": float(lons[j]),
                        "nwp_precip": float(np.round(perturbed_nwp[i, j], 2)),
                        "t2m": float(np.round(t2m[i, j], 2)),
                        "rh850": float(np.round(rh850[i, j], 1)),
                        "u850": float(np.round(u850[i, j], 2)),
                        "v850": float(np.round(v850[i, j], 2)),
                        "mslp": float(np.round(mslp[i, j], 1)),
                    }
                )

    nwp_df = pd.DataFrame(records)
    nwp_file = output_dir / f"nwp_gfs_{cycle_date}.parquet"
    nwp_df.to_parquet(nwp_file, index=False)
    logger.info("Saved synthetic NWP forecast fixture to %s (%d rows)", nwp_file.name, len(nwp_df))

    # 5. Save Paired Observation File (24h accumulation for Day 1 valid time)
    obs_records = []
    for i in range(len(lats)):
        for j in range(len(lons)):
            obs_records.append(
                {
                    "date": cycle_date,
                    "lat": float(lats[i]),
                    "lon": float(lons[j]),
                    "obs_precip": float(np.round(true_rain[i, j], 2)),
                }
            )

    obs_df = pd.DataFrame(obs_records)
    obs_file = output_dir / f"obs_imd_{cycle_date}.parquet"
    obs_df.to_parquet(obs_file, index=False)
    logger.info("Saved synthetic Observation fixture to %s (%d rows)", obs_file.name, len(obs_df))

    return nwp_file, obs_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sample meteorological fixtures")
    parser.add_argument("--date", default="2026-07-15", help="Cycle date YYYY-MM-DD")
    args = parser.parse_args()
    generate_synthetic_monsoon_fixture(cycle_date=args.date)
