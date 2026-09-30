"""
District Administrative Boundary & GeoJSON Normalization Engine for RAIN-REPAIR X (PART 8).
Standardizes Indian district administrative boundaries in EPSG:4326.
Tracks boundary dataset versioning, geodetic area calculations, and ray-casting polygon containment.
Supports built-in monsoon-vulnerable benchmark districts and external GeoJSON ingestion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from src.gis.crs import CRS_GEOGRAPHIC_WGS84, CRSManager
from src.utils.logging import get_logger

logger = get_logger("rain_repair.gis.districts")

BOUNDARY_DATASET_VERSION: str = "IMD-LGD-2026.1"

# Representative reference districts across meteorologically critical Indian zones:
# Western Ghats, Himalayan foothills, Indo-Gangetic plains, Bay of Bengal coast, Central India.
BENCHMARK_DISTRICTS: List[Dict[str, Any]] = [
    {
        "district_id": "MH_PUNE",
        "district_name": "Pune",
        "state_name": "Maharashtra",
        "meteorological_subdivision": "Madhya Maharashtra",
        "bbox": [73.30, 18.10, 74.30, 19.10],
        "centroid": [18.5204, 73.8567],
        "polygon_coords": [
            [73.35, 18.25], [73.55, 18.12], [73.95, 18.15], [74.28, 18.45],
            [74.25, 18.85], [73.95, 19.08], [73.60, 19.05], [73.38, 18.70], [73.35, 18.25]
        ],
    },
    {
        "district_id": "MH_MUMBAI",
        "district_name": "Mumbai",
        "state_name": "Maharashtra",
        "meteorological_subdivision": "Konkan & Goa",
        "bbox": [72.75, 18.88, 73.05, 19.30],
        "centroid": [19.0760, 72.8777],
        "polygon_coords": [
            [72.76, 18.89], [72.88, 18.90], [72.95, 19.05], [73.04, 19.22],
            [72.96, 19.29], [72.82, 19.28], [72.78, 19.12], [72.76, 18.89]
        ],
    },
    {
        "district_id": "KL_WAYANAD",
        "district_name": "Wayanad",
        "state_name": "Kerala",
        "meteorological_subdivision": "Kerala & Mahe",
        "bbox": [75.80, 11.45, 76.45, 11.95],
        "centroid": [11.6854, 76.1320],
        "polygon_coords": [
            [75.82, 11.60], [75.95, 11.48], [76.25, 11.46], [76.44, 11.72],
            [76.38, 11.94], [76.10, 11.93], [75.85, 11.78], [75.82, 11.60]
        ],
    },
    {
        "district_id": "UK_DEHRADUN",
        "district_name": "Dehradun",
        "state_name": "Uttarakhand",
        "meteorological_subdivision": "Uttarakhand",
        "bbox": [77.60, 29.95, 78.35, 30.75],
        "centroid": [30.3165, 78.0322],
        "polygon_coords": [
            [77.65, 30.15], [77.90, 29.98], [78.25, 30.10], [78.34, 30.45],
            [78.18, 30.72], [77.85, 30.74], [77.62, 30.48], [77.65, 30.15]
        ],
    },
    {
        "district_id": "HP_SHIMLA",
        "district_name": "Shimla",
        "state_name": "Himachal Pradesh",
        "meteorological_subdivision": "Himachal Pradesh",
        "bbox": [77.00, 30.90, 77.80, 31.50],
        "centroid": [31.1048, 77.1734],
        "polygon_coords": [
            [77.02, 31.10], [77.25, 30.92], [77.60, 30.98], [77.78, 31.25],
            [77.68, 31.48], [77.30, 31.45], [77.05, 31.28], [77.02, 31.10]
        ],
    },
    {
        "district_id": "OR_PURI",
        "district_name": "Puri",
        "state_name": "Odisha",
        "meteorological_subdivision": "Odisha",
        "bbox": [85.10, 19.60, 86.10, 20.20],
        "centroid": [19.8135, 85.8312],
        "polygon_coords": [
            [85.12, 19.78], [85.45, 19.62], [85.90, 19.70], [86.08, 19.95],
            [85.92, 20.18], [85.50, 20.12], [85.18, 19.98], [85.12, 19.78]
        ],
    },
    {
        "district_id": "TN_CHENNAI",
        "district_name": "Chennai",
        "state_name": "Tamil Nadu",
        "meteorological_subdivision": "Tamil Nadu, Puducherry & Karaikal",
        "bbox": [80.10, 12.90, 80.35, 13.20],
        "centroid": [13.0827, 80.2707],
        "polygon_coords": [
            [80.12, 12.92], [80.28, 12.91], [80.34, 13.05], [80.32, 13.19],
            [80.20, 13.18], [80.13, 13.08], [80.12, 12.92]
        ],
    },
    {
        "district_id": "KA_BENGALURU",
        "district_name": "Bengaluru Urban",
        "state_name": "Karnataka",
        "meteorological_subdivision": "South Interior Karnataka",
        "bbox": [77.40, 12.80, 77.80, 13.20],
        "centroid": [12.9716, 77.5946],
        "polygon_coords": [
            [77.42, 12.95], [77.58, 12.82], [77.78, 12.88], [77.79, 13.12],
            [77.65, 13.19], [77.45, 13.12], [77.42, 12.95]
        ],
    },
    {
        "district_id": "BR_PATNA",
        "district_name": "Patna",
        "state_name": "Bihar",
        "meteorological_subdivision": "Bihar",
        "bbox": [84.70, 25.20, 85.50, 25.75],
        "centroid": [25.5941, 85.1376],
        "polygon_coords": [
            [84.72, 25.42], [85.05, 25.22], [85.45, 25.35], [85.48, 25.68],
            [85.15, 25.74], [84.78, 25.62], [84.72, 25.42]
        ],
    },
    {
        "district_id": "AS_GUWAHATI",
        "district_name": "Kamrup Metropolitan",
        "state_name": "Assam",
        "meteorological_subdivision": "Assam & Meghalaya",
        "bbox": [91.50, 26.00, 92.00, 26.35],
        "centroid": [26.1445, 91.7362],
        "polygon_coords": [
            [91.52, 26.12], [91.75, 26.02], [91.98, 26.15], [91.95, 26.34],
            [91.70, 26.32], [91.55, 26.25], [91.52, 26.12]
        ],
    },
    {
        "district_id": "MP_DHAR",
        "district_name": "Dhar",
        "state_name": "Madhya Pradesh",
        "meteorological_subdivision": "West Madhya Pradesh",
        "bbox": [74.60, 22.00, 75.85, 23.15],
        "centroid": [22.5976, 75.3034],
        "polygon_coords": [
            [74.65, 22.15], [74.95, 22.02], [75.45, 22.10], [75.80, 22.45],
            [75.75, 22.95], [75.35, 23.12], [74.80, 23.05], [74.65, 22.60], [74.65, 22.15]
        ],
    },
]


def point_in_polygon_ray_casting(lon: float, lat: float, poly_coords: List[List[float]]) -> bool:
    """
    Standard Ray Casting algorithm (Jordan curve theorem) to determine if a (lon, lat)
    point is strictly inside a polygon boundary in EPSG:4326.
    """
    n = len(poly_coords)
    inside = False
    p1x, p1y = poly_coords[0]
    for i in range(1, n + 1):
        p2x, p2y = poly_coords[i % n]
        if lat > min(p1y, p2y):
            if lat <= max(p1y, p2y):
                if lon <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (lat - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or lon <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    return inside


class DistrictGeometryManager:
    """
    Manages standardized district geometries, spatial indexing, versioning,
    and polygon containment tests.
    """

    def __init__(
        self,
        external_geojson_path: Optional[Union[str, Path]] = None,
        dataset_version: str = BOUNDARY_DATASET_VERSION,
        crs: str = CRS_GEOGRAPHIC_WGS84,
    ) -> None:
        self.dataset_version = dataset_version
        self.crs = crs
        self.districts: List[Dict[str, Any]] = []

        if external_geojson_path and Path(external_geojson_path).exists():
            self._load_from_geojson(Path(external_geojson_path))
        else:
            self._load_builtins()

    def _load_builtins(self) -> None:
        """Load benchmark canonical district geometries with computed geodetic areas."""
        self.districts = []
        for d in BENCHMARK_DISTRICTS:
            if "polygon_coords" in d:
                coords = [d["polygon_coords"]]
            else:
                bbox = d["bbox"]
                coords = [
                    [
                        [bbox[0], bbox[1]],
                        [bbox[2], bbox[1]],
                        [bbox[2], bbox[3]],
                        [bbox[0], bbox[3]],
                        [bbox[0], bbox[1]],
                    ]
                ]
            # Compute geodesic area approximation
            area_km2 = CRSManager.compute_polygon_geodesic_area_km2(coords[0])

            feature = {
                "type": "Feature",
                "id": d["district_id"],
                "properties": {
                    "district_id": d["district_id"],
                    "district_name": d["district_name"],
                    "state_name": d["state_name"],
                    "subdivision": d.get("meteorological_subdivision", ""),
                    "centroid_lat": d["centroid"][0],
                    "centroid_lon": d["centroid"][1],
                    "area_km2": round(area_km2, 2),
                    "boundary_version": self.dataset_version,
                    "crs": self.crs,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": coords,
                },
            }
            self.districts.append(feature)
        logger.info(
            "Loaded %d benchmark district boundaries (Version: %s, CRS: %s).",
            len(self.districts),
            self.dataset_version,
            self.crs,
        )

    def _load_from_geojson(self, geojson_path: Path) -> None:
        """Load external district boundaries from GeoJSON file."""
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        features = data.get("features", [])
        self.districts = features
        logger.info(
            "Loaded %d district boundaries from %s (Version: %s).",
            len(self.districts),
            geojson_path,
            self.dataset_version,
        )

    def get_all_districts(self, state_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """Return list of district features, optionally filtered by state."""
        if state_filter is None:
            return list(self.districts)
        sf_clean = state_filter.lower().strip()
        return [
            d for d in self.districts
            if d["properties"].get("state_name", "").lower().strip() == sf_clean
        ]

    def find_district_by_id(self, district_id: str) -> Optional[Dict[str, Any]]:
        """Finds a district feature by its unique district_id."""
        for d in self.districts:
            if d["properties"].get("district_id") == district_id or d.get("id") == district_id:
                return d
        return None

    def update_district_boundary(self, district_id: str, new_feature: Dict[str, Any]) -> bool:
        """
        Updates an existing district boundary or appends a new one, maintaining version traceability.
        """
        for i, d in enumerate(self.districts):
            if d["properties"].get("district_id") == district_id or d.get("id") == district_id:
                new_feature["properties"]["boundary_version"] = self.dataset_version
                self.districts[i] = new_feature
                logger.info("Updated district boundary for %s.", district_id)
                return True

        # Append new
        new_feature["properties"]["boundary_version"] = self.dataset_version
        self.districts.append(new_feature)
        logger.info("Appended new district boundary for %s.", district_id)
        return True

    def get_metadata(self) -> Dict[str, Any]:
        """Returns boundary dataset metadata."""
        return {
            "boundary_dataset_version": self.dataset_version,
            "district_count": len(self.districts),
            "crs": self.crs,
            "states": sorted(list({d["properties"].get("state_name", "") for d in self.districts})),
        }
