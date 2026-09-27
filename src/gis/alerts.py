"""
IMD Alert Level Assignment & GIS Risk Layer Engine for RAIN-REPAIR X.
Standardizes district-level risk warnings in full conformance with:
    - CONTRACT-API-003 (Section 2.4 and Section 3)
    - IMD Four-Tier Color-Coded Severe Weather Early Warning Standards:
        GREEN  (#22c55e): No advisory / Normal monitoring
        YELLOW (#eab308): Watch & stay updated
        ORANGE (#f97316): Be prepared / Action alerts
        RED    (#ef4444): Take action / Severe warning
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger("rain_repair.gis.alerts")


class AlertLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ORANGE = "ORANGE"
    RED = "RED"


ALERT_COLORS: Dict[AlertLevel, str] = {
    AlertLevel.GREEN: "#22c55e",
    AlertLevel.YELLOW: "#eab308",
    AlertLevel.ORANGE: "#f97316",
    AlertLevel.RED: "#ef4444",
}

ALERT_ACTIONS: Dict[AlertLevel, str] = {
    AlertLevel.GREEN: "No advisory / Normal monitoring",
    AlertLevel.YELLOW: "Watch & stay updated",
    AlertLevel.ORANGE: "Be prepared / Action alerts",
    AlertLevel.RED: "Take action / Severe warning",
}

ALERT_SEVERITY_ORDER: Dict[AlertLevel, int] = {
    AlertLevel.RED: 4,
    AlertLevel.ORANGE: 3,
    AlertLevel.YELLOW: 2,
    AlertLevel.GREEN: 1,
}


def classify_imd_alert(
    prob_heavy: float,
    max_rainfall_mm: float,
) -> Tuple[AlertLevel, str, str]:
    """
    Classifies warning level according to IMD Standard Matrix (CONTRACT-API-003 Section 3):
        - RED:    P(Heavy Rain) >= 0.75 OR Max >= 204.5 mm
        - ORANGE: 0.50 <= P(Heavy Rain) < 0.75 OR 115.6 <= Max < 204.5 mm
        - YELLOW: 0.25 <= P(Heavy Rain) < 0.50 OR 64.5 <= Max < 115.6 mm
        - GREEN:  P(Heavy Rain) < 0.25 AND Max < 64.5 mm
    """
    p_h = float(prob_heavy)
    m_r = float(max_rainfall_mm)

    if p_h >= 0.75 or m_r >= 204.5:
        level = AlertLevel.RED
    elif p_h >= 0.50 or m_r >= 115.6:
        level = AlertLevel.ORANGE
    elif p_h >= 0.25 or m_r >= 64.5:
        level = AlertLevel.YELLOW
    else:
        level = AlertLevel.GREEN

    return level, ALERT_COLORS[level], ALERT_ACTIONS[level]


class IMDAlertEngine:
    """
    Orchestrates district alert tagging, GeoJSON generation, and risk table reporting.
    """

    def __init__(self) -> None:
        pass

    def enrich_district_records(
        self,
        district_records: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Enriches raw zonal statistics records with warning level, color codes,
        action guidelines, and uncertainty metrics.
        """
        enriched: List[Dict[str, Any]] = []

        for rec in district_records:
            item = dict(rec)
            prob_h = float(item.get("prob_heavy_rain", 0.0))
            max_r = float(item.get("max_rainfall_mm", 0.0))
            mean_r = float(item.get("mean_rainfall_mm", 0.0))

            level, color, action = classify_imd_alert(prob_h, max_r)

            item["warning_level"] = level.value
            item["color_code"] = color
            item["action_required"] = action
            item["severity_rank"] = ALERT_SEVERITY_ORDER[level]

            # Uncertainty indicator: peak to mean spread
            item["uncertainty_range_mm"] = round(max(0.0, max_r - mean_r), 2)

            enriched.append(item)

        # Sort by severity descending, then max_rainfall descending
        enriched.sort(key=lambda x: (x["severity_rank"], x.get("max_rainfall_mm", 0.0)), reverse=True)
        return enriched

    def build_geojson_feature_collection(
        self,
        enriched_records: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Constructs a standard GeoJSON FeatureCollection matching CONTRACT-API-003 Section 2.4.
        """
        features: List[Dict[str, Any]] = []

        for rec in enriched_records:
            geometry = rec.get("geometry")
            if not geometry:
                continue

            properties = {
                "district_id": rec.get("district_id"),
                "district_name": rec.get("district_name"),
                "state_name": rec.get("state_name"),
                "lead_time": rec.get("lead_time", 24),
                "mean_rainfall_mm": rec.get("mean_rainfall_mm", 0.0),
                "max_rainfall_mm": rec.get("max_rainfall_mm", 0.0),
                "prob_heavy_rain": rec.get("prob_heavy_rain", 0.0),
                "prob_very_heavy_rain": rec.get("prob_very_heavy_rain", 0.0),
                "prob_extreme_rain": rec.get("prob_extreme_rain", 0.0),
                "warning_level": rec.get("warning_level", AlertLevel.GREEN.value),
                "color_code": rec.get("color_code", ALERT_COLORS[AlertLevel.GREEN]),
                "action_required": rec.get("action_required", ALERT_ACTIONS[AlertLevel.GREEN]),
                "active_regime": rec.get("active_regime", "NORMAL_TRANSITIONAL"),
                "uncertainty_range_mm": rec.get("uncertainty_range_mm", 0.0),
                "grid_points_count": rec.get("grid_points_count", 1),
            }

            feature = {
                "type": "Feature",
                "id": rec.get("district_id"),
                "properties": properties,
                "geometry": geometry,
            }
            features.append(feature)

        return {
            "type": "FeatureCollection",
            "features": features,
        }

    def build_district_summary_table(
        self,
        enriched_records: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        """
        Creates a structured summary table for the UI and operational reporting.
        """
        rows = []
        for rec in enriched_records:
            rows.append(
                {
                    "District ID": rec.get("district_id"),
                    "District Name": rec.get("district_name"),
                    "State": rec.get("state_name"),
                    "Lead Time (h)": rec.get("lead_time", 24),
                    "Mean Rain (mm)": rec.get("mean_rainfall_mm", 0.0),
                    "Max Rain (mm)": rec.get("max_rainfall_mm", 0.0),
                    "P(>=64.5mm)": rec.get("prob_heavy_rain", 0.0),
                    "P(>=115.6mm)": rec.get("prob_very_heavy_rain", 0.0),
                    "Warning Level": rec.get("warning_level", AlertLevel.GREEN.value),
                    "Action Required": rec.get("action_required", ""),
                    "Active Regime": rec.get("active_regime", ""),
                }
            )

        df = pd.DataFrame(rows)
        return df
