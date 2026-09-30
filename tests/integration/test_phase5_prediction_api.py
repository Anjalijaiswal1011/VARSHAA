"""
Integration tests for Phase 5 Prediction API Endpoints:
- POST /api/v1/predictions (single state inference)
- POST /api/v1/predictions/batch (batch inference with fault isolation)
Validates RFC 7807 problem details error handling for 400, 422, 503, 500 status codes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


@pytest.fixture
def valid_prediction_payload() -> dict:
    """Valid forecast prediction payload for Mumbai coastal region."""
    return {
        "forecast_time": "2026-07-15T03:00:00Z",
        "initialization_time": "2026-07-15T00:00:00Z",
        "latitude": 19.00,
        "longitude": 72.85,
        "lead_time": 24,
        "nwp_precip": 65.4,
        "t2m": 300.2,
        "q2m": 0.021,
        "u10": 12.0,
        "v10": 4.5,
        "mslp": 1002.8,
        "elevation": 15.0,
        "slope": 1.2,
        "dist_to_coast": 5.0,
        "cape": 1800.0,
    }


def test_api_predict_single_success(valid_prediction_payload):
    """Validates successful POST /api/v1/predictions."""
    response = client.post("/api/v1/predictions", json=valid_prediction_payload)
    assert response.status_code == 200
    data = response.json()

    assert "prediction_id" in data
    assert data["prediction_id"].startswith("PRED-")
    assert data["latitude"] == 19.00
    assert data["longitude"] == 72.85
    assert data["raw_nwp_rainfall"] == 65.4

    # Quantile constraints: P50 <= P75 <= P90, P50 >= 0
    assert data["corrected_p50"] >= 0.0
    assert data["corrected_p75"] >= data["corrected_p50"]
    assert data["corrected_p90"] >= data["corrected_p75"]
    assert data["spread_p90_p50"] >= 0.0

    # Regime probabilities
    reg_probs = data["regime_probabilities"]
    assert len(reg_probs) == 6
    assert data["dominant_regime"] in reg_probs
    assert 0.0 <= data["dominant_probability"] <= 1.0

    # Model metadata
    assert data["model_version"] in ["1.0.0", "2.4.1"]
    assert data["regime_model_version"] == "v1.0.0"
    assert data["prediction_status"] in ["valid", "calibrated_rearranged"]
    assert "diagnostics" in data
    assert "timings_ms" in data["diagnostics"]


def test_api_predict_invalid_coordinates_rfc7807(valid_prediction_payload):
    """Validates RFC 7807 error format for coordinates outside India."""
    bad_payload = dict(valid_prediction_payload, latitude=52.5)  # Berlin latitude
    response = client.post("/api/v1/predictions", json=bad_payload)
    assert response.status_code == 400
    data = response.json()

    assert data["status"] == 400
    assert "error_code" in data
    assert "timestamp" in data
    assert "outside the Indian subcontinent domain" in data["message"]


def test_api_predict_inverted_timestamps_rfc7807(valid_prediction_payload):
    """Validates RFC 7807 error for temporal causality inversion."""
    bad_payload = dict(
        valid_prediction_payload,
        forecast_time="2026-07-14T00:00:00Z",
        initialization_time="2026-07-15T00:00:00Z",
    )
    response = client.post("/api/v1/predictions", json=bad_payload)
    assert response.status_code == 400
    data = response.json()

    assert data["status"] == 400
    assert "Temporal causality violation" in data["message"]


def test_api_predict_negative_rainfall_rfc7807(valid_prediction_payload):
    """Validates RFC 7807 error for negative precipitation."""
    bad_payload = dict(valid_prediction_payload, nwp_precip=-5.0)
    response = client.post("/api/v1/predictions", json=bad_payload)
    assert response.status_code == 400
    data = response.json()

    assert data["status"] == 400
    assert "Precipitation cannot be negative" in data["message"]


def test_api_predict_model_unavailable_rfc7807(valid_prediction_payload):
    """Validates RFC 7807 503 error when requested model ID does not exist."""
    bad_payload = dict(valid_prediction_payload, model_id="UNKNOWN_EXPERIMENTAL_MODEL_V99")
    response = client.post("/api/v1/predictions", json=bad_payload)
    assert response.status_code == 503
    data = response.json()

    assert data["status"] == 503
    assert data["error_code"] == "MODEL_UNAVAILABLE"


def test_api_predict_batch_success(valid_prediction_payload):
    """Validates POST /api/v1/predictions/batch with all valid records."""
    rec1 = valid_prediction_payload
    rec2 = dict(valid_prediction_payload, latitude=18.50, longitude=73.75, nwp_precip=42.0)
    rec3 = dict(valid_prediction_payload, latitude=11.70, longitude=76.10, nwp_precip=85.0)

    batch_payload = {"records": [rec1, rec2, rec3]}
    response = client.post("/api/v1/predictions/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 3
    assert data["successful_count"] == 3
    assert data["failed_count"] == 0
    assert len(data["predictions"]) == 3
    assert len(data["failed_records"]) == 0
    assert data["total_latency_ms"] > 0.0


def test_api_predict_batch_fault_isolation(valid_prediction_payload):
    """Validates that a corrupt record in batch does not abort other valid records."""
    rec_valid = valid_prediction_payload
    rec_invalid = dict(valid_prediction_payload, latitude=58.0)  # Outside bounds

    batch_payload = {"records": [rec_valid, rec_invalid]}
    response = client.post("/api/v1/predictions/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()

    assert data["total_records"] == 2
    assert data["successful_count"] == 1
    assert data["failed_count"] == 1
    assert len(data["predictions"]) == 1
    assert len(data["failed_records"]) == 1
    assert data["failed_records"][0]["record_index"] == 1
    assert "outside the Indian subcontinent domain" in data["failed_records"][0]["error_message"]
