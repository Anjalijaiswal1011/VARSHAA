"""
Integration Tests for Phase 8: District/Grid Forecast Product APIs.
Validates:
- GET /api/v1/forecasts/grid (with spatial/bounding box/point/grid_id filters & pagination)
- GET /api/v1/forecasts/districts (with state & district filters & pagination)
- GET /api/v1/forecasts/districts/{district_id} (with uncertainty quantiles & explanation metadata)
- End-to-end operational pipeline -> publication -> API retrieval lineage trace
- RFC 7807 problem details error handling
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.mlops.pipeline import OperationalPipeline

client = TestClient(app)


def test_grid_forecast_product_endpoint_basic():
    """Validates GET /api/v1/forecasts/grid basic query, pagination, and canonical schema."""
    resp = client.get("/api/v1/forecasts/grid?lead_time=24&limit=5&offset=0")
    assert resp.status_code == 200
    data = resp.json()

    assert "pagination" in data
    assert data["pagination"]["limit"] == 5
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["total_count"] >= 5
    assert "records" in data
    assert len(data["records"]) <= 5
    assert data["boundary_version"] == "IMD-LGD-2026.1"

    rec = data["records"][0]
    # Check top-level product fields
    assert "prediction_id" in rec
    assert "forecast_time" in rec
    assert "latitude" in rec
    assert "longitude" in rec
    assert "corrected_p50" in rec
    assert "corrected_p75" in rec
    assert "corrected_p90" in rec
    assert "dominant_regime" in rec
    assert "regime_probabilities" in rec
    assert "model_version" in rec
    assert "boundary_version" in rec
    assert rec["prediction_status"] == "VALID"

    # Check Section 15 nested objects
    assert "rainfall" in rec
    assert "p50" in rec["rainfall"]
    assert "p75" in rec["rainfall"]
    assert "p90" in rec["rainfall"]
    assert "regime" in rec
    assert "risk" in rec
    assert "metadata" in rec


def test_grid_forecast_bounding_box_filter():
    """Validates GET /api/v1/forecasts/grid bounding box filtering."""
    resp = client.get("/api/v1/forecasts/grid?min_lat=18.0&max_lat=20.0&min_lon=72.0&max_lon=74.5&lead_time=24")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) >= 1
    for r in data["records"]:
        assert 18.0 <= r["latitude"] <= 20.0
        assert 72.0 <= r["longitude"] <= 74.5


def test_grid_forecast_invalid_bbox_rfc7807():
    """Validates that invalid bounding box returns RFC 7807 problem details with 400 status."""
    resp = client.get("/api/v1/forecasts/grid?min_lat=25.0&max_lat=15.0&min_lon=70.0&max_lon=80.0")
    assert resp.status_code == 400
    data = resp.json()
    assert data["status"] == 400
    assert data["error_code"] == "INVALID_FORECAST_PARAMETERS"
    assert "min_lat" in data["message"]
    assert "timestamp" in data


def test_district_forecast_product_endpoint_basic():
    """Validates GET /api/v1/forecasts/districts list, pagination, and canonical fields."""
    resp = client.get("/api/v1/forecasts/districts?lead_time=24&limit=4&offset=0")
    assert resp.status_code == 200
    data = resp.json()

    assert "pagination" in data
    assert data["pagination"]["limit"] == 4
    assert data["pagination"]["offset"] == 0
    assert data["pagination"]["total_count"] >= 10
    assert data["boundary_version"] == "IMD-LGD-2026.1"
    assert len(data["records"]) == 4

    rec = data["records"][0]
    assert "district_id" in rec
    assert "district_name" in rec
    assert "corrected_p50" in rec
    assert "corrected_p75" in rec
    assert "corrected_p90" in rec
    assert rec["corrected_p50"] <= rec["corrected_p75"] <= rec["corrected_p90"]
    assert "dominant_regime" in rec
    assert "regime_probabilities" in rec
    assert rec["prediction_status"] == "VALID"

    # Nested Section 15 objects
    assert "district" in rec
    assert "rainfall" in rec
    assert "regime" in rec
    assert "risk" in rec
    assert "metadata" in rec


def test_district_forecast_state_filter():
    """Validates GET /api/v1/forecasts/districts state filter."""
    resp = client.get("/api/v1/forecasts/districts?state=Maharashtra&lead_time=24")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["records"]) >= 2
    for r in data["records"]:
        assert r["state"] == "Maharashtra"


def test_single_district_forecast_product_known():
    """Validates GET /api/v1/forecasts/districts/{district_id} for a known district."""
    resp = client.get("/api/v1/forecasts/districts/MH_PUNE?lead_time=24")
    assert resp.status_code == 200
    data = resp.json()

    assert data["district_id"] == "MH_PUNE"
    assert data["district_name"] == "Pune"
    assert data["state"] == "Maharashtra"
    assert data["lead_time"] == 24
    assert data["corrected_p50"] <= data["corrected_p75"] <= data["corrected_p90"]
    assert data["boundary_version"] == "IMD-LGD-2026.1"
    assert data["prediction_status"] == "VALID"

    # Verify model explanation metadata is present and non-causal
    assert "explanation" in data
    exp = data["explanation"]
    assert "dominant_regime" in exp
    assert "top_features" in exp
    assert "feature_contributions" in exp
    assert len(exp["top_features"]) > 0


def test_single_district_forecast_product_unknown_rfc7807():
    """Validates GET /api/v1/forecasts/districts/{district_id} returns 404 RFC 7807 for unknown district."""
    resp = client.get("/api/v1/forecasts/districts/UNKNOWN_DISTRICT_XYZ?lead_time=24")
    assert resp.status_code == 404
    data = resp.json()
    assert data["status"] == 404
    assert data["error_code"] == "DISTRICT_NOT_FOUND"
    assert "not found" in data["message"].lower()
    assert "timestamp" in data


def test_end_to_end_operational_pipeline_to_product_api():
    """
    Phase 8 Section 21: Full End-to-End Pipeline Verification Test:
    Real-world input -> OperationalPipeline execution -> Grid predictions ->
    GIS mapping -> District aggregation -> Verification Gate -> Publish Output ->
    GET /api/v1/forecasts/districts/{district_id} -> Verify Lineage & Values.
    """
    cycle_date = "2026-08-01"
    lead_time = 24

    # 1. Execute Operational Pipeline
    pipeline = OperationalPipeline.get_default_pipeline()
    job = pipeline.run_pipeline(
        cycle_date=cycle_date,
        lead_time_hours=lead_time,
        source="IMD_GFS_OPERATIONAL_E2E",
        force_rerun=True,
    )

    assert job.status in ["SUCCESS", "PARTIAL_SUCCESS"]
    assert job.validation_report is not None
    assert job.validation_report["publication_state"] in ["VALID", "PARTIAL"]

    # 2. Query Single District Product API for Pune
    resp = client.get(f"/api/v1/forecasts/districts/MH_PUNE?lead_time={lead_time}")
    assert resp.status_code == 200
    prod_data = resp.json()

    # 3. Verify complete traceability to model and pipeline lineage
    assert prod_data["district_id"] == "MH_PUNE"
    assert prod_data["lead_time"] == lead_time
    assert prod_data["boundary_version"] == "IMD-LGD-2026.1"
    assert prod_data["model_version"] is not None
    assert prod_data["feature_version"] is not None
    assert prod_data["dataset_version"] is not None
    assert prod_data["prediction_status"] == "VALID"

    # 4. Verify Uncertainty Quantiles are strictly preserved
    p50 = prod_data["corrected_p50"]
    p75 = prod_data["corrected_p75"]
    p90 = prod_data["corrected_p90"]
    assert 0.0 <= p50 <= p75 <= p90
    assert prod_data["spread_p90_p50"] == pytest.approx(p90 - p50, abs=0.05)

    # 5. Verify Regime Probabilities Sum to 1.0
    r_probs = prod_data["regime_probabilities"]
    assert len(r_probs) == 6
    assert sum(r_probs.values()) == pytest.approx(1.0, abs=1e-2)
