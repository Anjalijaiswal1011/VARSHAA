"""
Integration Tests for Phase 7 MLOps REST Endpoints in RAAP-X Backend.
Validates:
- GET /api/v1/health & /api/v1/health/detailed
- POST /api/v1/mlops/pipeline/trigger
- GET /api/v1/mlops/runs & /api/v1/mlops/runs/{job_id}
- GET /api/v1/mlops/alerts
- GET /api/v1/mlops/models
- GET /api/v1/mlops/data-quality
- GET /api/v1/mlops/verification/{job_id}
- GET /api/v1/mlops/scheduler/status
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


def test_detailed_health_endpoint():
    """Validates GET /api/v1/health/detailed."""
    resp = client.get("/api/v1/health/detailed")
    assert resp.status_code == 200
    data = resp.json()
    assert "overall_status" in data
    assert data["checks_total"] == 9
    assert "components" in data
    assert "model_registry" in data["components"]
    assert "feature_pipeline" in data["components"]
    assert "regime_model" in data["components"]
    assert "quantile_models" in data["components"]


def test_trigger_pipeline_endpoint():
    """Validates POST /api/v1/mlops/pipeline/trigger and GET /api/v1/mlops/runs/{job_id}."""
    payload = {
        "cycle_date": "2026-07-28",
        "lead_time_hours": 24,
        "force_rerun": True,
        "source": "IMD_GFS_TEST",
    }
    resp = client.post("/api/v1/mlops/pipeline/trigger", json=payload)
    assert resp.status_code == 200
    job_data = resp.json()
    assert "job_id" in job_data
    assert job_data["status"] in ["SUCCESS", "PARTIAL_SUCCESS"]
    assert job_data["records_processed"] > 0
    job_id = job_data["job_id"]

    # Retrieve specific run
    run_resp = client.get(f"/api/v1/mlops/runs/{job_id}")
    assert run_resp.status_code == 200
    run_data = run_resp.json()
    assert run_data["job_id"] == job_id
    assert "stage_states" in run_data
    assert "INGESTION" in run_data["stage_states"]
    assert "VALIDATION" in run_data["stage_states"]

    # Verification endpoint
    verif_resp = client.get(f"/api/v1/mlops/verification/{job_id}")
    assert verif_resp.status_code == 200
    verif_data = verif_resp.json()
    assert "publication_state" in verif_data


def test_list_runs_endpoint():
    """Validates GET /api/v1/mlops/runs."""
    resp = client.get("/api/v1/mlops/runs?limit=10")
    assert resp.status_code == 200
    runs = resp.json()
    assert isinstance(runs, list)
    assert len(runs) > 0


def test_alerts_endpoint():
    """Validates GET /api/v1/mlops/alerts."""
    resp = client.get("/api/v1/mlops/alerts")
    assert resp.status_code == 200
    alerts = resp.json()
    assert isinstance(alerts, list)


def test_models_registry_endpoint():
    """Validates GET /api/v1/mlops/models."""
    resp = client.get("/api/v1/mlops/models")
    assert resp.status_code == 200
    models = resp.json()
    assert isinstance(models, list)
    assert len(models) >= 1
    # Check model lifecycle status presence
    for m in models:
        assert "status" in m
        assert m["status"] in ["candidate", "validated", "staging", "production", "retired", "failed", "archived", "trained", "rejected"]


def test_data_quality_endpoint():
    """Validates GET /api/v1/mlops/data-quality."""
    resp = client.get("/api/v1/mlops/data-quality")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data


def test_scheduler_status_endpoint():
    """Validates GET /api/v1/mlops/scheduler/status."""
    resp = client.get("/api/v1/mlops/scheduler/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "interval_minutes" in data
    assert "timezone" in data
    assert "is_running" in data
