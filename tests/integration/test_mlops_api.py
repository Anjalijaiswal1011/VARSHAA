"""
Integration Tests for PART 10 MLOps REST API Endpoints.
Tests:
- GET /api/v1/mlops/registry
- GET /api/v1/mlops/experiments
- GET /api/v1/mlops/drift/status
- POST /api/v1/mlops/promote (RBAC + Quality Gate)
- POST /api/v1/mlops/rollback (RBAC + Safe Rollback)
- POST /api/v1/mlops/retrain/evaluate (Forecaster trigger)
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth import create_access_token


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def admin_headers():
    token = create_access_token(
        user_id="usr_admin_001", username="admin", role="admin"
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def forecaster_headers():
    token = create_access_token(
        user_id="usr_forecaster_001", username="forecaster", role="forecaster"
    )
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def public_headers():
    token = create_access_token(
        user_id="usr_public_001", username="public_user", role="public"
    )
    return {"Authorization": f"Bearer {token}"}



class TestMLOpsAPI:
    """Test suite for MLOps operational endpoints."""

    def test_get_model_registry(self, client):
        response = client.get("/api/v1/mlops/registry")
        assert response.status_code == 200
        data = response.json()
        assert "total_models" in data
        assert data["total_models"] >= 1
        assert "active_production_model" in data
        assert data["active_production_model"]["status"] == "production"
        assert "models" in data

    def test_get_experiments(self, client):
        response = client.get("/api/v1/mlops/experiments")
        assert response.status_code == 200
        data = response.json()
        assert "total_experiments" in data
        assert data["total_experiments"] >= 6
        assert "baseline_comparison" in data
        assert len(data["baseline_comparison"]) >= 6
        assert "ablation_summary" in data
        assert len(data["ablation_summary"]) >= 5

    def test_get_drift_status(self, client):
        response = client.get("/api/v1/mlops/drift/status")
        assert response.status_code == 200
        data = response.json()
        assert "overall_state" in data
        assert data["overall_state"] in [
            "HEALTHY",
            "WARNING",
            "DRIFT_DETECTED",
            "PERFORMANCE_DEGRADED",
            "CALIBRATION_DEGRADED",
        ]
        assert "feature_drift_summary" in data
        assert "raw_nwp_rainfall" in data["feature_drift_summary"]
        assert "performance_drift_summary" in data
        assert "error_memory_health" in data
        assert "analog_memory_health" in data

    def test_promote_unauthorized_and_forbidden(self, client, public_headers, forecaster_headers):
        payload = {"model_id": "MOD_REPAIR_PROD_V2.4", "target_status": "production"}

        # No auth
        res_no_auth = client.post("/api/v1/mlops/promote", json=payload)
        assert res_no_auth.status_code in [401, 403]

        # Public role -> Forbidden
        res_public = client.post("/api/v1/mlops/promote", json=payload, headers=public_headers)
        assert res_public.status_code == 403

        # Forecaster role -> Forbidden (admin required)
        res_forecaster = client.post("/api/v1/mlops/promote", json=payload, headers=forecaster_headers)
        assert res_forecaster.status_code == 403

    def test_promote_and_rollback_as_admin(self, client, admin_headers):
        # Promote candidate to staging
        payload = {"model_id": "MOD_REPAIR_CANDIDATE_V2.4.1", "target_status": "staging"}
        res = client.post("/api/v1/mlops/promote", json=payload, headers=admin_headers)
        assert res.status_code in [200, 400]
        if res.status_code == 200:
            assert res.json()["status"] == "success"

        # Rollback
        res_rb = client.post("/api/v1/mlops/rollback?reason=Field+trial+rollback", headers=admin_headers)
        assert res_rb.status_code in [200, 400]

    def test_retraining_evaluate_as_forecaster(self, client, forecaster_headers):
        res = client.post("/api/v1/mlops/retrain/evaluate?force=false", headers=forecaster_headers)
        assert res.status_code == 200
        data = res.json()
        assert "status" in data
        assert "timestamp" in data
