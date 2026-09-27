"""
Unit tests for GIS Zonal Aggregation, CRS Management, Grid Quality Checks,
IMD Alerts, and Machine-Readable Explanations (PART 8).
"""

import numpy as np
import pandas as pd
import pytest

from src.gis.alerts import (
    AlertLevel,
    IMDAlertEngine,
    classify_imd_alert,
)
from src.gis.crs import (
    CRS_GEOGRAPHIC_WGS84,
    CRS_PROJECTED_INDIA_LCC,
    CRSManager,
)
from src.gis.districts import (
    BOUNDARY_DATASET_VERSION,
    DistrictGeometryManager,
    point_in_polygon_ray_casting,
)
from src.gis.grid_schema import (
    GridForecastQualityValidator,
    GridForecastRecord,
)
from src.gis.layers import MapLayerContractManager
from src.gis.zonal import SpatialZonalStatisticsEngine
from src.models.explainability import ModelTransparencyEngine


def test_crs_manager_and_geodetic_areas():
    """Validates CRS consistency, projection definitions, and geodetic area calculations."""
    crs_mgr = CRSManager(
        input_crs=CRS_GEOGRAPHIC_WGS84,
        boundary_crs=CRS_GEOGRAPHIC_WGS84,
        calculation_crs=CRS_PROJECTED_INDIA_LCC,
    )
    assert crs_mgr.validate_crs_compatibility() is True
    meta = crs_mgr.get_crs_metadata()
    assert meta["input_crs"] == "EPSG:4326"
    assert meta["calculation_crs"] == "EPSG:7755"

    # Test cell area computation for 0.25 deg grid cell
    area_equator = CRSManager.compute_grid_cell_area_km2(lat=0.0)
    area_pune = CRSManager.compute_grid_cell_area_km2(lat=18.5)
    area_himalaya = CRSManager.compute_grid_cell_area_km2(lat=31.0)

    # Authalic cell area decreases toward the pole as meridians converge
    assert area_equator > area_pune > area_himalaya > 0.0
    # At ~18.5 N, a 0.25 deg cell is roughly 27.8 km x 26.3 km ~ 730-760 km2
    assert 700.0 < area_pune < 800.0

    # Test polygon geodesic area
    poly_box = [[73.0, 18.0], [74.0, 18.0], [74.0, 19.0], [73.0, 19.0], [73.0, 18.0]]
    poly_area = CRSManager.compute_polygon_geodesic_area_km2(poly_box)
    assert 10000.0 < poly_area < 13000.0


def test_grid_forecast_quality_validator_clean_and_corrupt():
    """Validates GridForecastQualityValidator on scientific invariants and corruption flags."""
    # 1. Clean valid record
    clean_record = GridForecastRecord(
        forecast_time="2026-07-15T00:00:00Z",
        valid_time="2026-07-16T03:00:00Z",
        grid_id="G_18.50_73.75",
        latitude=18.50,
        longitude=73.75,
        lead_time=24,
        raw_nwp_rainfall=45.0,
        corrected_rainfall=52.0,
        p50_rainfall=52.0,
        p75_rainfall=68.0,
        p90_rainfall=88.0,
        heavy_probability=0.45,
        very_heavy_probability=0.15,
        extreme_probability=0.02,
        regime="ACTIVE_MONSOON",
        regime_probabilities={"ACTIVE_MONSOON": 0.8},
        uncertainty_indicator=36.0,
        model_version="v1.0.0-prob",
        evt_status="CONVERGED_NORMAL",
    )
    is_valid, errors = GridForecastQualityValidator.validate_record(clean_record)
    assert is_valid is True
    assert len(errors) == 0

    # 2. Corrupt record: negative rainfall, quantile crossing, out of bounds coordinates
    corrupt_record = GridForecastRecord(
        forecast_time="invalid-date",
        valid_time="2026-07-16T03:00:00Z",
        grid_id="G_corrupt",
        latitude=55.0,  # Out of India
        longitude=120.0,
        lead_time=33,   # Unsupported lead time
        raw_nwp_rainfall=-10.0,  # Negative
        corrected_rainfall=50.0,
        p50_rainfall=80.0,  # Crossing P50 > P75
        p75_rainfall=60.0,
        p90_rainfall=90.0,
        heavy_probability=1.5,  # > 1.0
        very_heavy_probability=-0.2,  # < 0.0
        extreme_probability=0.0,
        regime="ACTIVE_MONSOON",
        regime_probabilities={},
        uncertainty_indicator=10.0,
        model_version="v1.0.0",
        evt_status="FAIL",
    )
    is_valid_c, errors_c = GridForecastQualityValidator.validate_record(corrupt_record)
    assert is_valid_c is False
    assert len(errors_c) >= 5

    # 3. DataFrame validator with duplicates
    df = pd.DataFrame([
        clean_record.to_dict(),
        clean_record.to_dict(),  # Duplicate
    ])
    validated_df, summary = GridForecastQualityValidator.validate_dataframe(df)
    assert "quality_flag" in validated_df.columns
    assert summary["total_records"] == 2
    # Duplicates flagged
    assert validated_df["quality_flag"].iloc[0] == "INVALID"


def test_district_geometry_manager_and_ray_casting():
    """Validates DistrictGeometryManager dataset versioning, boundary updates, and ray-casting."""
    mgr = DistrictGeometryManager()
    assert mgr.dataset_version == BOUNDARY_DATASET_VERSION
    districts = mgr.get_all_districts()
    assert len(districts) >= 10

    # Test ray casting point in polygon
    # Pune bounding box is approx [73.30, 18.10, 74.30, 19.10]
    pune = mgr.find_district_by_id("MH_PUNE")
    assert pune is not None
    poly_coords = pune["geometry"]["coordinates"][0]

    # Inside Pune
    assert point_in_polygon_ray_casting(73.85, 18.52, poly_coords) is True
    # Outside Pune (e.g. Mumbai or Delhi)
    assert point_in_polygon_ray_casting(77.20, 28.61, poly_coords) is False

    # Test dynamic boundary update
    updated_pune = dict(pune)
    updated_pune["properties"]["test_tag"] = "custom_update"
    assert mgr.update_district_boundary("MH_PUNE", updated_pune) is True
    re_pune = mgr.find_district_by_id("MH_PUNE")
    assert re_pune["properties"]["test_tag"] == "custom_update"


def test_classify_imd_alert_thresholds():
    """Validates IMD Standard Alert Matrix (CONTRACT-API-003 Section 3)."""
    # 1. GREEN
    lvl, col, act = classify_imd_alert(prob_heavy=0.10, max_rainfall_mm=30.0)
    assert lvl == AlertLevel.GREEN
    assert col == "#22c55e"

    # 2. YELLOW
    lvl, col, act = classify_imd_alert(prob_heavy=0.30, max_rainfall_mm=40.0)
    assert lvl == AlertLevel.YELLOW
    assert col == "#eab308"

    # 3. ORANGE
    lvl, col, act = classify_imd_alert(prob_heavy=0.60, max_rainfall_mm=70.0)
    assert lvl == AlertLevel.ORANGE
    assert col == "#f97316"

    # 4. RED
    lvl, col, act = classify_imd_alert(prob_heavy=0.85, max_rainfall_mm=50.0)
    assert lvl == AlertLevel.RED
    assert col == "#ef4444"


def test_spatial_zonal_aggregation_full_part8():
    """Tests comprehensive Part 8 zonal statistics: dual probabilities, hotspot, spread, and comparison."""
    geom_mgr = DistrictGeometryManager()
    zonal_engine = SpatialZonalStatisticsEngine(geometry_manager=geom_mgr)

    # Synthetic grid DataFrame covering Pune
    grid_df = pd.DataFrame(
        [
            {
                "grid_id": "G_18.50_73.75",
                "latitude": 18.50,
                "longitude": 73.75,
                "lead_time": 24,
                "raw_nwp_rainfall": 45.0,
                "corrected_rainfall": 55.0,
                "p50_rainfall": 55.0,
                "p75_rainfall": 75.0,
                "p90_rainfall": 125.0,
                "heavy_probability": 0.65,
                "very_heavy_probability": 0.30,
                "extreme_probability": 0.05,
                "regime": "ACTIVE_MONSOON",
                "forecast_time": "2026-07-15T00:00:00Z",
                "valid_time": "2026-07-16T03:00:00Z",
            },
            {
                "grid_id": "G_18.60_73.80",
                "latitude": 18.60,
                "longitude": 73.80,
                "lead_time": 24,
                "raw_nwp_rainfall": 60.0,
                "corrected_rainfall": 70.0,
                "p50_rainfall": 70.0,
                "p75_rainfall": 95.0,
                "p90_rainfall": 150.0,
                "heavy_probability": 0.85,
                "very_heavy_probability": 0.55,
                "extreme_probability": 0.12,
                "regime": "ACTIVE_MONSOON",
                "forecast_time": "2026-07-15T00:00:00Z",
                "valid_time": "2026-07-16T03:00:00Z",
            },
        ]
    )

    records = zonal_engine.aggregate_grid_to_districts(grid_df=grid_df, lead_time=24)
    assert len(records) > 0

    pune_rec = next(r for r in records if r["district_id"] == "MH_PUNE")
    # Verify dual event probabilities
    assert 0.0 <= pune_rec["heavy_probability"] <= 1.0
    assert 0.0 <= pune_rec["district_max_heavy_probability"] <= 1.0
    assert pune_rec["district_max_heavy_probability"] >= pune_rec["heavy_probability"]

    # Verify Hotspot extraction
    assert "hotspot" in pune_rec
    hotspot = pune_rec["hotspot"]
    assert hotspot["hotspot_grid_id"] in ["G_18.50_73.75", "G_18.60_73.80"]
    assert hotspot["hotspot_value_mm"] >= 125.0

    # Verify uncertainty forecast spread
    assert pune_rec["forecast_spread"] >= 0.0
    assert pune_rec["forecast_spread"] == round(pune_rec["p90_rainfall"] - pune_rec["p50_rainfall"], 2)

    # Verify comparison fields
    assert "difference" in pune_rec
    assert "relative_change_pct" in pune_rec

    # Verify status and coverage
    assert pune_rec["status"] in ["complete", "partial"]
    assert pune_rec["coverage_percentage"] >= 50.0


def test_map_layers_contract_manager():
    """Validates MapLayerContractManager metadata."""
    layer_mgr = MapLayerContractManager(model_version="v1.0.0-prob")
    layers = layer_mgr.get_supported_layers()
    assert len(layers) >= 5
    layer_ids = [l.layer_id for l in layers]
    assert "layer_corrected_rainfall" in layer_ids
    assert "layer_p90_rainfall" in layer_ids
    assert "layer_heavy_probability" in layer_ids
    assert "layer_forecast_spread" in layer_ids


def test_machine_readable_district_explanation():
    """Validates evidence-grounded machine-readable district explanation."""
    transparency_engine = ModelTransparencyEngine()
    district_data = {
        "district_id": "MH_PUNE",
        "district_name": "Pune",
        "raw_nwp_rainfall": 45.0,
        "corrected_rainfall": 58.0,
        "difference": 13.0,
        "dominant_regime": "ACTIVE_MONSOON",
        "regime_probabilities": {"ACTIVE_MONSOON": 0.85},
        "heavy_probability": 0.62,
        "very_heavy_probability": 0.28,
        "extreme_probability": 0.05,
        "forecast_spread": 42.0,
    }

    explanation = transparency_engine.explain_district_forecast(district_data)
    assert explanation["district_id"] == "MH_PUNE"
    assert explanation["dominant_regime"] == "ACTIVE_MONSOON"
    assert "NWP persistent underprediction" in explanation["recent_error_signal"]
    assert len(explanation["main_drivers"]) >= 3
    assert "user_friendly_summary" in explanation
    assert "Pune" in explanation["user_friendly_summary"]
    assert "ACTIVE_MONSOON" in explanation["user_friendly_summary"]
