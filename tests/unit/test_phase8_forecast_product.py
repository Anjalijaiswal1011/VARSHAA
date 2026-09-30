"""
Unit tests for Phase 8: District/Grid Forecast Product Schemas, GIS Mapping,
Uncertainty Preservation, and Status Engine.
"""

import pytest
import numpy as np
import pandas as pd
from pydantic import ValidationError

from backend.schemas import (
    CanonicalDistrictForecastRecord,
    CanonicalGridForecastRecord,
    DistrictForecastProductResponse,
    GridForecastProductResponse,
    PaginationMeta,
    RainfallQuantiles,
    RegimeState,
    RiskAssessment,
    ForecastMetadata,
)
from backend.services import (
    BOUNDARY_DATASET_VERSION,
    DATASET_VERSION,
    FEATURE_VERSION,
    MODEL_VERSION,
    REGIME_MODEL_VERSION,
    ForecastService,
)
from src.gis.districts import DistrictGeometryManager, point_in_polygon_ray_casting
from src.gis.zonal import CANONICAL_REGIMES, SpatialZonalStatisticsEngine


def test_canonical_grid_forecast_schema_validation():
    """Verify CanonicalGridForecastRecord validates strict product fields and nested structures."""
    valid_record = {
        "prediction_id": "pred_G_18.50_73.75_24_2026-07-15",
        "forecast_time": "2026-07-16T03:00:00Z",
        "initialization_time": "2026-07-15T00:00:00Z",
        "lead_time": 24,
        "latitude": 18.50,
        "longitude": 73.75,
        "grid_id": "G_18.50_73.75",
        "district_id": "MH_PUNE",
        "district_name": "Pune",
        "raw_nwp_rainfall": 45.2,
        "corrected_p50": 52.4,
        "corrected_p75": 67.0,
        "corrected_p90": 82.8,
        "spread_p90_p50": 30.4,
        "regime_probabilities": {r: 0.1667 for r in CANONICAL_REGIMES},
        "dominant_regime": "ACTIVE_MONSOON",
        "dominant_probability": 0.72,
        "heavy_rainfall_probability": 0.54,
        "extreme_rainfall_probability": 0.12,
        "model_version": MODEL_VERSION,
        "regime_model_version": REGIME_MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset_version": DATASET_VERSION,
        "boundary_version": BOUNDARY_DATASET_VERSION,
        "pipeline_run_id": "run_test_001",
        "prediction_status": "VALID",
        "created_at": "2026-07-15T01:00:00Z",
        "rainfall": {
            "raw_nwp": 45.2,
            "p50": 52.4,
            "p75": 67.0,
            "p90": 82.8,
            "spread": 30.4,
            "difference": 7.2,
        },
        "regime": {
            "dominant": "ACTIVE_MONSOON",
            "dominant_probability": 0.72,
            "probabilities": {r: 0.1667 for r in CANONICAL_REGIMES},
        },
        "risk": {
            "heavy_rainfall_probability": 0.54,
            "extreme_rainfall_probability": 0.12,
            "warning_level": "ORANGE",
        },
        "metadata": {
            "model_version": MODEL_VERSION,
            "regime_model_version": REGIME_MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "dataset_version": DATASET_VERSION,
            "boundary_version": BOUNDARY_DATASET_VERSION,
            "pipeline_run_id": "run_test_001",
        },
    }

    parsed = CanonicalGridForecastRecord(**valid_record)
    assert parsed.prediction_id == "pred_G_18.50_73.75_24_2026-07-15"
    assert parsed.corrected_p50 <= parsed.corrected_p75 <= parsed.corrected_p90
    assert parsed.boundary_version == "IMD-LGD-2026.1"
    assert parsed.prediction_status == "VALID"
    assert parsed.rainfall.p50 == 52.4


def test_canonical_district_forecast_schema_validation():
    """Verify CanonicalDistrictForecastRecord validates full district properties and Section 15 nested objects."""
    district_data = {
        "district_id": "MH_PUNE",
        "district_name": "Pune",
        "state": "Maharashtra",
        "forecast_time": "2026-07-16T03:00:00Z",
        "initialization_time": "2026-07-15T00:00:00Z",
        "lead_time": 24,
        "raw_nwp_rainfall": 45.2,
        "corrected_p50": 52.4,
        "corrected_p75": 67.0,
        "corrected_p90": 82.8,
        "spread_p90_p50": 30.4,
        "heavy_rainfall_probability": 0.54,
        "extreme_rainfall_probability": 0.12,
        "dominant_regime": "ACTIVE_MONSOON",
        "regime_probabilities": {
            "ACTIVE_MONSOON": 0.70,
            "BREAK_MONSOON": 0.05,
            "MONSOON_DEPRESSION": 0.10,
            "WESTERN_DISTURBANCE": 0.05,
            "OFFSHORE_TROUGH": 0.05,
            "NORMAL_TRANSITIONAL": 0.05,
        },
        "model_version": MODEL_VERSION,
        "regime_model_version": REGIME_MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "dataset_version": DATASET_VERSION,
        "boundary_version": BOUNDARY_DATASET_VERSION,
        "pipeline_run_id": "pipe_001",
        "prediction_status": "VALID",
        "created_at": "2026-07-15T01:00:00Z",
        "status": "VALID",
        "district": {
            "id": "MH_PUNE",
            "name": "Pune",
            "state": "Maharashtra",
        },
        "rainfall": {
            "raw_nwp": 45.2,
            "p50": 52.4,
            "p75": 67.0,
            "p90": 82.8,
            "spread": 30.4,
            "difference": 7.2,
        },
        "regime": {
            "dominant": "ACTIVE_MONSOON",
            "dominant_probability": 0.70,
            "probabilities": {
                "ACTIVE_MONSOON": 0.70,
                "BREAK_MONSOON": 0.05,
                "MONSOON_DEPRESSION": 0.10,
                "WESTERN_DISTURBANCE": 0.05,
                "OFFSHORE_TROUGH": 0.05,
                "NORMAL_TRANSITIONAL": 0.05,
            },
        },
        "risk": {
            "heavy_rainfall_probability": 0.54,
            "extreme_rainfall_probability": 0.12,
            "warning_level": "ORANGE",
        },
        "metadata": {
            "model_version": MODEL_VERSION,
            "regime_model_version": REGIME_MODEL_VERSION,
            "feature_version": FEATURE_VERSION,
            "dataset_version": DATASET_VERSION,
            "boundary_version": BOUNDARY_DATASET_VERSION,
            "pipeline_run_id": "pipe_001",
        },
    }

    parsed = CanonicalDistrictForecastRecord(**district_data)
    assert parsed.district_id == "MH_PUNE"
    assert parsed.district.name == "Pune"
    assert parsed.rainfall.p50 == 52.4
    assert parsed.rainfall.p75 == 67.0
    assert parsed.rainfall.p90 == 82.8
    assert parsed.risk.heavy_rainfall_probability == 0.54
    assert sum(parsed.regime.probabilities.values()) == pytest.approx(1.0, abs=1e-3)


def test_gis_boundary_integration_and_version():
    """Verify geometry manager returns expected benchmark districts and confirms boundary version."""
    geom_mgr = DistrictGeometryManager()
    assert geom_mgr.dataset_version == "IMD-LGD-2026.1"

    districts = geom_mgr.get_all_districts()
    assert len(districts) >= 10

    # Ensure key benchmark districts exist
    district_ids = [d["properties"]["district_id"] for d in districts]
    assert "MH_PUNE" in district_ids
    assert "MH_MUMBAI" in district_ids
    assert "KL_WAYANAD" in district_ids
    assert "UK_DEHRADUN" in district_ids

    # Validate coordinate reference system
    pune = geom_mgr.find_district_by_id("MH_PUNE")
    assert pune is not None
    coords = pune["geometry"]["coordinates"][0]
    for pt in coords:
        lon, lat = pt[0], pt[1]
        assert 68.0 <= lon <= 98.0
        assert 6.0 <= lat <= 38.5


def test_district_aggregation_uncertainty_and_regime_sum():
    """Verify district aggregation preserves uncertainty (P50 <= P75 <= P90) and normalized regime vector."""
    service = ForecastService()
    ok, result = service.get_product_district_forecasts(lead_time=24)
    assert ok is True
    assert isinstance(result, dict)

    records = result["records"]
    assert len(records) > 0

    for rec in records:
        # 1. Uncertainty preservation
        assert rec["corrected_p50"] >= 0.0
        assert rec["corrected_p75"] >= rec["corrected_p50"]
        assert rec["corrected_p90"] >= rec["corrected_p75"]
        assert rec["spread_p90_p50"] == pytest.approx(rec["corrected_p90"] - rec["corrected_p50"], abs=0.05)

        # 2. Regime probability vector validity: 0 <= P <= 1 and sum(P) ~ 1.0
        reg_probs = rec["regime_probabilities"]
        assert len(reg_probs) == 6
        for r_name, p_val in reg_probs.items():
            assert 0.0 <= p_val <= 1.0
        assert sum(reg_probs.values()) == pytest.approx(1.0, abs=1e-2)

        # 3. Version metadata
        assert rec["boundary_version"] == "IMD-LGD-2026.1"
        assert rec["model_version"] == MODEL_VERSION


def test_forecast_status_stale_detection():
    """Verify STALE status is set when querying a non-fresh forecast cycle date."""
    service = ForecastService()
    ok, result = service.get_product_district_forecasts(
        lead_time=24,
        forecast_time="2025-01-01",  # Old date
    )
    assert ok is True
    recs = result["records"]
    assert len(recs) > 0
    for r in recs:
        assert r["prediction_status"] == "STALE"


def test_grid_forecast_filtering_and_pagination():
    """Verify grid forecast filtering by bounding box, radius, and pagination."""
    service = ForecastService()

    # 1. Pagination check
    ok, res_p1 = service.get_product_grid_forecasts(lead_time=24, limit=3, offset=0)
    assert ok is True
    assert len(res_p1["records"]) == 3
    assert res_p1["pagination"]["total_count"] >= 10
    assert res_p1["pagination"]["has_more"] is True

    ok, res_p2 = service.get_product_grid_forecasts(lead_time=24, limit=3, offset=3)
    assert ok is True
    assert len(res_p2["records"]) == 3
    assert res_p1["records"][0]["grid_id"] != res_p2["records"][0]["grid_id"]

    # 2. Bounding box query
    ok, res_bbox = service.get_product_grid_forecasts(
        lead_time=24,
        min_lat=18.0,
        max_lat=20.0,
        min_lon=72.0,
        max_lon=74.5,
    )
    assert ok is True
    assert len(res_bbox["records"]) >= 1
    for r in res_bbox["records"]:
        assert 18.0 <= r["latitude"] <= 20.0
        assert 72.0 <= r["longitude"] <= 74.5

    # 3. Grid ID filter
    ok, res_gid = service.get_product_grid_forecasts(lead_time=24, grid_id="G_18.50_73.75")
    assert ok is True
    assert len(res_gid["records"]) == 1
    assert res_gid["records"][0]["grid_id"] == "G_18.50_73.75"


def test_single_district_lookup_and_not_found():
    """Verify single district returns explanation metadata or RFC 7807 error when missing."""
    service = ForecastService()

    # Known district
    ok, code, result = service.get_product_single_district_forecast(district_id="MH_PUNE", lead_time=24)
    assert ok is True
    assert code == "OK"
    assert result["district_id"] == "MH_PUNE"
    assert "explanation" in result
    assert "dominant_regime" in result["explanation"]
    assert "top_features" in result["explanation"]
    assert len(result["explanation"]["top_features"]) > 0

    # Nonexistent district
    ok_miss, code_miss, err_msg = service.get_product_single_district_forecast(district_id="NON_EXISTENT_999", lead_time=24)
    assert ok_miss is False
    assert code_miss == "DISTRICT_NOT_FOUND"
    assert "not found" in err_msg.lower()
