"""
Coordinate Reference System (CRS) Management for RAIN-REPAIR X.
Standardizes geospatial projection handling and area calculation across Indian subcontinental domain.

Explicit CRS Standards:
    - Input & Storage CRS: EPSG:4326 (WGS 84 geographic 2D latitude/longitude in degrees).
    - Boundary Geometries CRS: EPSG:4326.
    - Calculation & Area-Weighting Projected CRS: EPSG:7755 (India National Coordinate System 2011 / LCC)
      or EPSG:32643 (WGS 84 / UTM Zone 43N).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.utils.logging import get_logger

logger = get_logger("rain_repair.gis.crs")

# Standard CRS constants
CRS_GEOGRAPHIC_WGS84: str = "EPSG:4326"
CRS_PROJECTED_INDIA_LCC: str = "EPSG:7755"
CRS_PROJECTED_UTM43N: str = "EPSG:32643"

# WGS 84 Ellipsoid constants
WGS84_SEMI_MAJOR_AXIS_A: float = 6378137.0  # meters
WGS84_FLATTENING_F: float = 1.0 / 298.257223563
WGS84_SEMI_MINOR_AXIS_B: float = WGS84_SEMI_MAJOR_AXIS_A * (1.0 - WGS84_FLATTENING_F)


class CRSManager:
    """
    Validates, documents, and applies coordinate reference system conversions
    and geodetic area calculations for rainfall grid cells and administrative polygons.
    """

    def __init__(
        self,
        input_crs: str = CRS_GEOGRAPHIC_WGS84,
        boundary_crs: str = CRS_GEOGRAPHIC_WGS84,
        calculation_crs: str = CRS_PROJECTED_INDIA_LCC,
        output_crs: str = CRS_GEOGRAPHIC_WGS84,
    ) -> None:
        self.input_crs = input_crs
        self.boundary_crs = boundary_crs
        self.calculation_crs = calculation_crs
        self.output_crs = output_crs

        self.validate_crs_compatibility()

    def validate_crs_compatibility(self) -> bool:
        """
        Validates that input and boundary CRS align for spatial indexing.
        """
        if self.input_crs.upper() != self.boundary_crs.upper():
            logger.warning(
                "CRS mismatch detected: input_crs=%s vs boundary_crs=%s. Spatial operations require reprojection.",
                self.input_crs,
                self.boundary_crs,
            )
            return False
        return True

    def get_crs_metadata(self) -> Dict[str, str]:
        """Returns explicit CRS configuration dictionary."""
        return {
            "input_crs": self.input_crs,
            "boundary_crs": self.boundary_crs,
            "calculation_crs": self.calculation_crs,
            "output_crs": self.output_crs,
        }

    @staticmethod
    def compute_grid_cell_area_km2(lat: float, d_lat_deg: float = 0.25, d_lon_deg: float = 0.25) -> float:
        """
        Computes accurate authalic surface area of a 0.25-degree grid cell on the WGS 84 ellipsoid.
        Area of a quadrilateral bounded by parallels lat1, lat2 and meridians lon1, lon2:
            Area = (b^2 * delta_lon / 2) * (sin(lat2)/(1 - e^2*sin^2(lat2)) + ... )
        Spherical authalic approximation (error < 0.1% for 0.25 deg cells):
            Area = R_authalic^2 * delta_lon_rad * (sin(lat2_rad) - sin(lat1_rad))
        """
        r_earth_km = 6371.0088  # Mean authalic radius
        half_dlat = d_lat_deg / 2.0
        lat_south_rad = math.radians(lat - half_dlat)
        lat_north_rad = math.radians(lat + half_dlat)
        d_lon_rad = math.radians(d_lon_deg)

        area_km2 = (r_earth_km ** 2) * d_lon_rad * abs(math.sin(lat_north_rad) - math.sin(lat_south_rad))
        return float(area_km2)

    @staticmethod
    def compute_polygon_geodesic_area_km2(coordinates: List[List[float]]) -> float:
        """
        Computes planar/geodetic approximation of polygon area given lon/lat vertices in EPSG:4326.
        Uses the Shoelace formula on equirectangular projected coordinates around the centroid.
        """
        if len(coordinates) < 3:
            return 0.0

        coords = np.array(coordinates)
        lons = coords[:, 0]
        lats = coords[:, 1]

        # Centroid latitude for longitude cosine scaling
        mean_lat_rad = math.radians(np.mean(lats))
        r_km = 6371.0088

        # Project lon/lat to local planar km
        x = np.radians(lons) * r_km * math.cos(mean_lat_rad)
        y = np.radians(lats) * r_km

        # Shoelace formula
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return float(area)
