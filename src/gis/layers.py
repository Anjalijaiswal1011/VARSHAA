"""
Standardized Map Layer Contract & Metadata Management for RAIN-REPAIR X (PART 8).
Defines operational choropleth and risk map layer definitions for Leaflet/MapLibre map rendering.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger("rain_repair.gis.layers")


@dataclass
class MapLayerMetadata:
    """
    Standardized Map Layer Metadata conforming to PART 8 Section 29.
    """
    layer_id: str
    layer_name: str
    metric: str
    unit: str
    min_value: float
    max_value: float
    timestamp: str
    model_version: str
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MapLayerContractManager:
    """
    Manages operational map layers and generates standardized layer catalogs.
    """

    def __init__(self, model_version: str = "v1.0.0-prob") -> None:
        self.model_version = model_version

    def get_supported_layers(self, timestamp: str = "2026-07-15T00:00:00Z") -> List[MapLayerMetadata]:
        """
        Returns list of standardized operational map layers.
        """
        return [
            MapLayerMetadata(
                layer_id="layer_corrected_rainfall",
                layer_name="Corrected Rainfall (P50)",
                metric="corrected_rainfall",
                unit="mm/day",
                min_value=0.0,
                max_value=300.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="Median post-processed 24h accumulated rainfall expectation.",
            ),
            MapLayerMetadata(
                layer_id="layer_p90_rainfall",
                layer_name="High-Impact Rainfall (P90)",
                metric="p90_rainfall",
                unit="mm/day",
                min_value=0.0,
                max_value=450.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="90th percentile upper-bound rainfall scenario for emergency planning.",
            ),
            MapLayerMetadata(
                layer_id="layer_heavy_probability",
                layer_name="Heavy Rain Probability (>= 64.5 mm)",
                metric="heavy_probability",
                unit="probability [0-1]",
                min_value=0.0,
                max_value=1.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="Calibrated probability of exceeding IMD Heavy Rainfall threshold.",
            ),
            MapLayerMetadata(
                layer_id="layer_very_heavy_probability",
                layer_name="Very Heavy Rain Probability (>= 115.6 mm)",
                metric="very_heavy_probability",
                unit="probability [0-1]",
                min_value=0.0,
                max_value=1.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="Calibrated probability of exceeding IMD Very Heavy Rainfall threshold.",
            ),
            MapLayerMetadata(
                layer_id="layer_extreme_probability",
                layer_name="Extreme Rain Probability (>= 204.5 mm)",
                metric="extreme_probability",
                unit="probability [0-1]",
                min_value=0.0,
                max_value=1.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="Calibrated probability of exceeding IMD Extremely Heavy Rainfall threshold.",
            ),
            MapLayerMetadata(
                layer_id="layer_forecast_spread",
                layer_name="Forecast Spread (Uncertainty)",
                metric="forecast_spread",
                unit="mm",
                min_value=0.0,
                max_value=150.0,
                timestamp=timestamp,
                model_version=self.model_version,
                description="Upper-quantile uncertainty indicator computed as P90 - P50 spread.",
            ),
        ]
