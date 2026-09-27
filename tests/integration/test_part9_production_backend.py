"""
Integration Tests for PART 9: Production Backend & API Gateway.
Validates:
    - API Gateway (Rate Limiter, Security Headers, Request Tracing, Latency)
    - Authentication (JWT Issuance, API Key Authentication, Invalid Credentials)
    - RBAC Authorization (Admin vs Public vs Forecaster Roles)
    - Request Validation & Malicious Input Sanitization
    - High-Performance Caching (Hit/Miss Telemetry, Invalidation)
    - Database Persistence & Benchmark Seeding
"""

import pytest
from fastapi.testclient import TestClient

from backend.auth import DEFAULT_USERS, create_access_token
from backend.cache import cache_manager
from backend.db.init_db import init_db
from backend.db.models import DistrictModel, UserModel
from backend.db.session import SessionLocal
from backend.main import app

# Ensure database is initialized
init_db()

client = TestClient(app)


def test_gateway_security_headers_and_tracing():
    """Validates API Gateway OWASP security headers and X-Request-ID/X-Response-Time headers."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200

    # 1. Security Headers
    assert response.headers.get("x-content-type-options") == "nosniff"
    assert response.headers.get("x-frame-options") == "DENY"
    assert "max-age=" in response.headers.get("strict-transport-security", "")

    # 2. Request Tracing Headers
    assert "x-request-id" in response.headers
    assert "x-response-time" in response.headers
    assert "ms" in response.headers["x-response-time"]


def test_gateway_rate_limiting():
    """Validates API Gateway rate limiter headers."""
    response = client.get("/api/v1/forecast/latest?lead_time=24")
    assert response.status_code == 200
    assert "x-ratelimit-limit" in response.headers
    assert "x-ratelimit-remaining" in response.headers
    assert int(response.headers["x-ratelimit-remaining"]) >= 0


def test_auth_token_issuance_and_validation():
    """Validates JWT token issuance and invalid credential handling."""
    # 1. Valid credentials for forecaster
    resp_valid = client.post(
        "/api/v1/auth/token",
        json={"username": "forecaster", "password": "Forecaster@2026"},
    )
    assert resp_valid.status_code == 200
    data = resp_valid.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"
    assert data["role"] == "forecaster"
    token = data["access_token"]

    # 2. Access /auth/me with Bearer token
    resp_me = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp_me.status_code == 200
    me_data = resp_me.json()
    assert me_data["username"] == "forecaster"
    assert me_data["role"] == "forecaster"

    # 3. Invalid credentials
    resp_invalid = client.post(
        "/api/v1/auth/token",
        json={"username": "forecaster", "password": "WrongPassword"},
    )
    assert resp_invalid.status_code == 401


def test_api_key_authentication():
    """Validates authentication using X-API-Key header."""
    admin_api_key = DEFAULT_USERS["admin"]["api_key"]
    response = client.get(
        "/api/v1/auth/me",
        headers={"X-API-Key": admin_api_key},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["username"] == "admin"
    assert data["role"] == "admin"


def test_rbac_admin_cache_purge_permissions():
    """Validates that admin-only endpoints reject non-admin users and accept admin users."""
    admin_token = create_access_token(user_id="u_admin", username="admin", role="admin")
    forecaster_token = create_access_token(user_id="u_forecaster", username="forecaster", role="forecaster")

    # 1. Forecaster attempts admin cache clear -> 403 Forbidden
    resp_denied = client.post(
        "/api/v1/cache/clear",
        headers={"Authorization": f"Bearer {forecaster_token}"},
    )
    assert resp_denied.status_code == 403
    assert "Insufficient permissions" in resp_denied.json()["message"]

    # 2. Admin attempts cache clear -> 200 OK
    resp_allowed = client.post(
        "/api/v1/cache/clear",
        headers={"Authorization": f"Bearer {admin_token}"},
    )
    assert resp_allowed.status_code == 200
    assert resp_allowed.json()["status"] == "success"


def test_cache_telemetry_and_hit_performance():
    """Validates cache stats endpoint and caching performance."""
    # Seed cache by requesting districts
    resp1 = client.get("/api/v1/forecast/districts?lead_time=24")
    assert resp1.status_code == 200

    # Second request should hit cache
    resp2 = client.get("/api/v1/forecast/districts?lead_time=24")
    assert resp2.status_code == 200

    stats_resp = client.get("/api/v1/cache/stats")
    assert stats_resp.status_code == 200
    stats = stats_resp.json()
    assert stats["hits"] >= 1
    assert stats["active_entries"] >= 1


def test_database_persistence_and_seed_data():
    """Validates that SQLite database has seeded users and benchmark districts."""
    db = SessionLocal()
    try:
        user_count = db.query(UserModel).count()
        assert user_count >= 3

        district_count = db.query(DistrictModel).count()
        assert district_count >= 10

        pune = db.query(DistrictModel).filter(DistrictModel.district_id == "MH_PUNE").first()
        assert pune is not None
        assert pune.state_name == "Maharashtra"
        assert pune.area_km2 > 0
    finally:
        db.close()


def test_request_validation_and_sanitization():
    """Validates request parameter bounds and malicious input rejection."""
    # 1. Out of bounds coordinates -> 400
    resp_oob = client.get("/api/v1/forecast/grid?lat=89.0&lon=150.0")
    assert resp_oob.status_code == 400

    # 2. Invalid district ID with SQL injection pattern -> 404/400
    resp_sqli = client.get("/api/v1/forecast/district/MH_PUNE;DROP TABLE users;--")
    assert resp_sqli.status_code in [400, 404]

    # 3. Path traversal attack in district ID -> 400
    resp_trav = client.get("/api/v1/forecast/district/../../etc/passwd")
    assert resp_trav.status_code in [400, 404]
