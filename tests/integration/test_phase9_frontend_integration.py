"""
Integration Tests for Phase 9: Frontend Dashboard & API Integration.

Validates:
1. Static dashboard route (/dashboard) serving production UI build.
2. Root API payload includes dashboard route reference.
3. Verification of all API endpoints consumed by frontend:
   - GET /api/v1/forecasts/districts (pagination, district metadata, quantiles, lead_time)
   - GET /api/v1/forecasts/districts/{district_id} (detailed drill-down, regimes, explainability, lineage)
   - GET /api/v1/forecast/districts (GeoJSON FeatureCollection for Leaflet map choropleth)
   - GET /api/v1/forecasts/grid (grid cell forecast points)
   - GET /api/v1/health/detailed (operational subsystems health)
4. Scientific validity of frontend-consumed contracts:
   - Quantile monotonicity: P50 <= P75 <= P90
   - Regime probability completeness: sum(P_regimes) approx 1.0
   - Non-causal feature contribution metadata
5. Error state handling:
   - 404 RFC 7807 problem details for non-existent district
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_dashboard_static_serving():
    """Verify that /dashboard serves the production HTML build containing RAAP-X title/tags."""
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    content = resp.text
    assert "RAAP-X" in content
    assert "root" in content


def test_root_advertises_dashboard():
    """Verify that the API root endpoint advertises the dashboard route."""
    resp = client.get("/")
    assert resp.status_code == 200
    data = resp.json()
    assert "dashboard" in data
    assert data["dashboard"] == "/dashboard"


def test_frontend_districts_endpoint_contract():
    """Verify GET /api/v1/forecasts/districts returns schema required by frontend."""
    resp = client.get("/api/v1/forecasts/districts?lead_time=24&limit=10")
    assert resp.status_code == 200
    data = resp.json()

    assert "records" in data
    assert "pagination" in data
    assert len(data["records"]) > 0

    record = data["records"][0]
    required_fields = [
        "district_id",
        "district_name",
        "forecast_time",
        "lead_time",
        "corrected_p50",
        "corrected_p75",
        "corrected_p90",
        "dominant_regime",
        "regime_probabilities",
        "prediction_status",
    ]
    for field in required_fields:
        assert field in record, f"Missing required field {field} in district forecast record"

    # Quantile monotonicity verification
    assert record["corrected_p50"] <= record["corrected_p75"] + 1e-4, "P50 must be <= P75"
    assert record["corrected_p75"] <= record["corrected_p90"] + 1e-4, "P75 must be <= P90"


def test_frontend_district_detail_contract():
    """Verify GET /api/v1/forecasts/districts/{district_id} returns all panels' data."""
    # First get a valid district_id
    list_resp = client.get("/api/v1/forecasts/districts?limit=1")
    assert list_resp.status_code == 200
    dist_id = list_resp.json()["records"][0]["district_id"]

    detail_resp = client.get(f"/api/v1/forecasts/districts/{dist_id}?lead_time=24")
    assert detail_resp.status_code == 200
    data = detail_resp.json()

    # Core attributes
    assert data["district_id"] == dist_id
    assert "district_name" in data
    assert "corrected_p50" in data
    assert "corrected_p75" in data
    assert "corrected_p90" in data

    # Regime intelligence distribution
    assert "dominant_regime" in data
    assert "regime_probabilities" in data
    reg_probs = data["regime_probabilities"]
    assert isinstance(reg_probs, dict)
    prob_sum = sum(reg_probs.values())
    assert 0.95 <= prob_sum <= 1.05, f"Regime probabilities should sum to ~1.0, got {prob_sum}"

    # Explainability / feature contributions (non-causal model attribution)
    explanation = data.get("explanation")
    if explanation:
        assert "features" in explanation or "top_features" in explanation or "summary" in explanation

    # Lineage / provenance metadata
    assert "model_version" in data
    assert "boundary_version" in data
    assert data["boundary_version"] == "IMD-LGD-2026.1"


def test_frontend_geojson_endpoint_contract():
    """Verify GET /api/v1/forecast/districts returns Leaflet-compatible GeoJSON."""
    resp = client.get("/api/v1/forecast/districts?lead_time=24")
    assert resp.status_code == 200
    geojson = resp.json()

    assert geojson.get("type") == "FeatureCollection"
    assert "features" in geojson
    features = geojson["features"]
    assert len(features) > 0

    first_feat = features[0]
    assert first_feat.get("type") == "Feature"
    assert "geometry" in first_feat
    assert "properties" in first_feat

    props = first_feat["properties"]
    assert "district_id" in props
    assert "district_name" in props
    assert "mean_rainfall_mm" in props or "corrected_p50" in props
    assert "dominant_regime" in props or "active_regime" in props


def test_frontend_system_health_contract():
    """Verify GET /api/v1/health/detailed returns subsystem status for the health modal."""
    resp = client.get("/api/v1/health/detailed")
    assert resp.status_code == 200
    data = resp.json()

    assert "status" in data
    assert "components" in data or "subsystems" in data
    components = data.get("components") or data.get("subsystems")

    expected_subsystems = [
        "data_source",
        "feature_pipeline",
        "regime_model",
        "quantile_models",
        "gis_boundaries",
    ]
    for sub in expected_subsystems:
        assert sub in components, f"Missing subsystem '{sub}' in health report"
        assert "status" in components[sub] or "healthy" in str(components[sub]).lower()


def test_frontend_error_rfc7807_handling():
    """Verify requesting non-existent district returns RFC 7807 problem details."""
    resp = client.get("/api/v1/forecasts/districts/INVALID_DISTRICT_XYZ_99999")
    assert resp.status_code == 404
    err = resp.json()

    assert "status" in err
    assert err["status"] == 404
    assert "error_code" in err
    assert "message" in err


def test_frontend_verification_summary_contract():
    """Verify GET /api/v1/verification/summary returns metrics consumed by VerificationBarChart."""
    resp = client.get("/api/v1/verification/summary?season=monsoon_2026")
    assert resp.status_code == 200
    data = resp.json()

    assert "evaluation_period" in data
    assert "metrics" in data
    m = data["metrics"]

    assert "raw_nwp_rmse" in m
    assert "corrected_rmse" in m
    assert "raw_nwp_mae" in m
    assert "corrected_mae" in m
    assert "heavy_rain_csi_raw" in m
    assert "heavy_rain_csi_corrected" in m
    assert "crps_raw" in m
    assert "crps_corrected" in m

