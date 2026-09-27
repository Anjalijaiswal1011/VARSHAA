"""
Integration tests for RAIN-REPAIR X FastAPI Backend API Service (PART 8).
Validates all endpoints against CONTRACT-API-003, RFC 7807 problem details, and PART 8 specifications.
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Validates GET /api/v1/health."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "latest_available_cycle" in data
    assert "boundary_dataset_version" in data


def test_api_forecast_latest_endpoint():
    """Validates GET /api/v1/forecast/latest."""
    response = client.get("/api/v1/forecast/latest?lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["lead_time_hours"] == 24
    assert "active_synoptic_regime" in data
    assert "grid_summary" in data
    assert "mean_corrected_mm" in data["grid_summary"]

    # Invalid lead time should return 400
    invalid_resp = client.get("/api/v1/forecast/latest?lead_time=13")
    assert invalid_resp.status_code == 400


def test_api_forecast_grid_point_query():
    """Validates GET /api/v1/forecast/grid with valid coordinates (Pune)."""
    response = client.get("/api/v1/forecast/grid?lat=18.5204&lon=73.8567")
    assert response.status_code == 200
    data = response.json()
    assert "query_coords" in data
    assert "nearest_grid_coords" in data
    assert len(data["time_series"]) > 0

    ts0 = data["time_series"][0]
    assert "raw_nwp_mm" in ts0
    assert "corrected_p50_mm" in ts0
    assert "corrected_p90_mm" in ts0
    assert "prob_heavy_rain" in ts0
    assert ts0["quality_flag"] == "VALID"


def test_api_forecast_grid_out_of_bounds():
    """Validates RFC 7807 Problem Details on invalid coordinates."""
    response = client.get("/api/v1/forecast/grid?lat=55.0&lon=12.0")
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == 400
    assert data["error_code"] == "INVALID_COORDINATES"
    assert "outside India meteorological domain" in data["message"]
    assert "timestamp" in data


def test_api_forecast_districts_geojson():
    """Validates GET /api/v1/forecast/districts GeoJSON layer."""
    response = client.get("/api/v1/forecast/districts?lead_time=24")
    assert response.status_code == 200
    assert "application/geo+json" in response.headers.get("content-type", "")

    data = response.json()
    assert data["type"] == "FeatureCollection"
    assert len(data["features"]) > 0

    feat = data["features"][0]
    props = feat["properties"]
    assert "district_id" in props
    assert "warning_level" in props
    assert props["warning_level"] in ["GREEN", "YELLOW", "ORANGE", "RED"]
    assert "color_code" in props
    assert "mean_rainfall_mm" in props
    assert "max_rainfall_mm" in props


def test_api_single_district_forecast_detail():
    """Validates GET /api/v1/forecast/district/{district_id}."""
    response = client.get("/api/v1/forecast/district/MH_PUNE?lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["district_id"] == "MH_PUNE"
    assert "district_name" in data
    assert "raw_nwp_rainfall" in data
    assert "corrected_rainfall" in data
    assert "p50_rainfall" in data
    assert "p75_rainfall" in data
    assert "p90_rainfall" in data
    assert "hotspot" in data
    assert "hotspot_grid_id" in data["hotspot"]
    assert "explanation" in data
    assert "forecast_spread" in data
    assert data["status"] in ["complete", "partial"]

    # Invalid district ID returns 404 RFC 7807
    inv_resp = client.get("/api/v1/forecast/district/NONEXISTENT_DISTRICT_XYZ")
    assert inv_resp.status_code == 404
    inv_data = inv_resp.json()
    assert inv_data["status"] == 404
    assert inv_data["error_code"] == "DISTRICT_NOT_FOUND"


def test_api_map_layers_catalog():
    """Validates GET /api/v1/forecast/map."""
    response = client.get("/api/v1/forecast/map")
    assert response.status_code == 200
    data = response.json()
    assert "layers" in data
    assert len(data["layers"]) >= 5
    layer0 = data["layers"][0]
    assert "layer_id" in layer0
    assert "metric" in layer0
    assert "unit" in layer0
    assert "min_value" in layer0
    assert "max_value" in layer0


def test_api_forecast_compare():
    """Validates GET /api/v1/forecast/compare."""
    # District level
    resp_dist = client.get("/api/v1/forecast/compare?level=district&lead_time=24")
    assert resp_dist.status_code == 200
    data_dist = resp_dist.json()
    assert data_dist["level"] == "district"
    assert len(data_dist["comparisons"]) > 0
    c0 = data_dist["comparisons"][0]
    assert "raw_nwp_rainfall" in c0
    assert "corrected_rainfall" in c0
    assert "difference" in c0
    assert "relative_change_pct" in c0

    # Grid level
    resp_grid = client.get("/api/v1/forecast/compare?level=grid&lead_time=24")
    assert resp_grid.status_code == 200
    data_grid = resp_grid.json()
    assert data_grid["level"] == "grid"
    assert len(data_grid["comparisons"]) > 0


def test_api_district_explanation():
    """Validates GET /api/v1/forecast/explanation/{district_id}."""
    response = client.get("/api/v1/forecast/explanation/MH_MUMBAI?lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert data["district_id"] == "MH_MUMBAI"
    assert "dominant_regime" in data
    assert "recent_error_signal" in data
    assert "main_drivers" in data
    assert len(data["main_drivers"]) >= 3
    assert "user_friendly_summary" in data

    # Missing district
    inv_resp = client.get("/api/v1/forecast/explanation/NON_EXISTENT")
    assert inv_resp.status_code == 404


def test_api_metadata_models_and_regimes():
    """Validates GET /api/v1/metadata/models and GET /api/v1/metadata/regimes."""
    resp_m = client.get("/api/v1/metadata/models")
    assert resp_m.status_code == 200
    m_data = resp_m.json()
    assert "model_version" in m_data
    assert "boundary_dataset_version" in m_data
    assert "crs_geographic" in m_data
    assert m_data["crs_geographic"] == "EPSG:4326"

    resp_r = client.get("/api/v1/metadata/regimes")
    assert resp_r.status_code == 200
    r_data = resp_r.json()
    assert r_data["total_regimes"] == 6
    assert len(r_data["regimes"]) == 6
    assert r_data["regimes"][0]["regime_code"] == "ACTIVE_MONSOON"


def test_api_forecast_table():
    """Validates GET /api/v1/forecast/table."""
    response = client.get("/api/v1/forecast/table?lead_time=24")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    assert "District ID" in data[0]
    assert "Warning Level" in data[0]


def test_api_verification_summary():
    """Validates GET /api/v1/verification/summary."""
    response = client.get("/api/v1/verification/summary")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    m = data["metrics"]
    assert "raw_nwp_rmse" in m
    assert "corrected_rmse" in m
    assert m["rmse_improvement_pct"] > 0
    assert "crps_corrected" in m


def test_api_explainability_summary():
    """Validates GET /api/v1/explainability/summary."""
    response = client.get("/api/v1/explainability/summary")
    assert response.status_code == 200
    data = response.json()
    assert "top_contributing_features" in data
    assert len(data["top_contributing_features"]) >= 4
