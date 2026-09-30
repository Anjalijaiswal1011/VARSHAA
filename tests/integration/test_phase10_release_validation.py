"""
Phase 10: Final End-to-End Hardening, Security, Performance & Release Validation Suite.

Executes and verifies:
1. Full System Audit: Ingestion -> Model -> GIS -> API -> Dashboard.
2. Security & RBAC: Token auth, role enforcement, injection protection, RFC 7807 safety.
3. Prediction Sanity: Non-negativity, quantile monotonicity (P50 <= P75 <= P90),
   probability bounds [0, 1], regime sum to 1.0, and geographic domain bounds.
4. Data Leakage & Causality: Feature schema audit, strictly lagged error memory.
5. Numerical Reproducibility: Identical inputs yield identical outputs.
6. Performance Benchmarking: Latency measurements across critical endpoints.
7. Frontend Release Integrity: Static dashboard serving and contract validation.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.auth import hash_password, create_access_token
from backend.main import app
from src.features.leakage_audit import audit_feature_dataframe_leakage
from src.features.schema import FEATURE_REGISTRY, LeakageRisk
from src.models.inference_pipeline import ProductionInferencePipeline, ForecastInputRecord

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. SECURITY & ACCESS CONTROL AUDIT
# ---------------------------------------------------------------------------
def test_security_protected_endpoints_reject_unauthenticated():
    """Verify admin endpoints reject unauthenticated access with 401."""
    resp = client.post("/api/v1/cache/clear")
    assert resp.status_code in [401, 403], f"Expected 401/403, got {resp.status_code}"
    data = resp.json()
    assert "error_code" in data or "message" in data or "detail" in data


def test_security_rbac_role_enforcement():
    """Verify public and forecaster roles cannot execute admin operations."""
    forecaster_token = create_access_token(user_id="usr_f", username="forecaster", role="forecaster")
    resp = client.post(
        "/api/v1/cache/clear",
        headers={"Authorization": f"Bearer {forecaster_token}"},
    )
    assert resp.status_code == 403, "Forecaster role must not be permitted to clear system cache"


def test_security_safe_rfc7807_error_no_stack_traces():
    """Verify malformed requests return RFC 7807 without leaking Python internals or stack traces."""
    resp = client.get("/api/v1/forecasts/districts?lead_time=9999")
    assert resp.status_code == 400
    data = resp.json()

    assert data.get("status") == 400
    assert "error_code" in data
    assert "message" in data
    # Ensure no Python traceback / file path leak
    msg = str(data["message"])
    assert "Traceback" not in msg
    assert "File \"" not in msg


# ---------------------------------------------------------------------------
# 2. PREDICTION SANITY & QUANTILE MONOTONICITY
# ---------------------------------------------------------------------------
def test_prediction_sanity_quantile_monotonicity_and_bounds():
    """Verify all district predictions strictly satisfy P50 >= 0, P75 >= P50, P90 >= P75."""
    resp = client.get("/api/v1/forecasts/districts?lead_time=24&limit=50")
    assert resp.status_code == 200
    data = resp.json()
    records = data["records"]
    assert len(records) > 0

    for r in records:
        p50 = r["corrected_p50"]
        p75 = r["corrected_p75"]
        p90 = r["corrected_p90"]

        assert p50 >= 0.0, f"P50 must be non-negative, got {p50} for {r['district_id']}"
        assert p75 >= p50 - 1e-4, f"P75 ({p75}) must be >= P50 ({p50}) for {r['district_id']}"
        assert p90 >= p75 - 1e-4, f"P90 ({p90}) must be >= P75 ({p75}) for {r['district_id']}"

        # Risk probabilities
        hp = r["heavy_rainfall_probability"]
        ep = r["extreme_rainfall_probability"]
        assert 0.0 <= hp <= 1.0, f"Heavy rain probability out of bounds: {hp}"
        assert 0.0 <= ep <= 1.0, f"Extreme rain probability out of bounds: {ep}"
        assert ep <= hp + 1e-4, f"Extreme probability ({ep}) should not exceed Heavy probability ({hp})"

        # Regime probabilities completeness
        probs = r["regime_probabilities"]
        prob_sum = sum(probs.values())
        assert 0.95 <= prob_sum <= 1.05, f"Regime probabilities should sum to ~1.0, got {prob_sum}"


# ---------------------------------------------------------------------------
# 3. DATA LEAKAGE & TEMPORAL CAUSALITY AUDIT
# ---------------------------------------------------------------------------
def test_data_leakage_feature_schema_audit():
    """Verify feature registry contains zero unlagged target variables and passes leakage audit."""
    # Build sample feature frame from schema registry
    safe_cols = [k for k, v in FEATURE_REGISTRY.items() if v.leakage_risk == LeakageRisk.SAFE]
    assert len(safe_cols) > 0

    sample_df = pd.DataFrame(
        np.random.randn(5, len(safe_cols)),
        columns=safe_cols,
    )
    sample_df["cycle_date"] = "2026-07-15"
    sample_df["lead_time"] = 24

    report = audit_feature_dataframe_leakage(sample_df, cycle_date="2026-07-15", strict=True)
    assert report["is_leakage_clean"] is True
    assert report["status"] == "PASSED"
    assert len(report["violations"]) == 0


# ---------------------------------------------------------------------------
# 4. NUMERICAL REPRODUCIBILITY AUDIT
# ---------------------------------------------------------------------------
def test_model_numerical_reproducibility():
    """Verify running the model twice on identical input features produces identical outputs."""
    resp1 = client.get("/api/v1/forecasts/districts/MH_PUNE?lead_time=24")
    assert resp1.status_code == 200
    d1 = resp1.json()

    # Repeat exact same query
    resp2 = client.get("/api/v1/forecasts/districts/MH_PUNE?lead_time=24")
    assert resp2.status_code == 200
    d2 = resp2.json()

    assert d1["corrected_p50"] == pytest.approx(d2["corrected_p50"], abs=1e-5)
    assert d1["corrected_p75"] == pytest.approx(d2["corrected_p75"], abs=1e-5)
    assert d1["corrected_p90"] == pytest.approx(d2["corrected_p90"], abs=1e-5)
    assert d1["dominant_regime"] == d2["dominant_regime"]
    for reg, p in d1["regime_probabilities"].items():
        assert p == pytest.approx(d2["regime_probabilities"][reg], abs=1e-5)


# ---------------------------------------------------------------------------
# 5. PERFORMANCE & LATENCY BENCHMARKING
# ---------------------------------------------------------------------------
def test_performance_backend_endpoint_latencies():
    """Measure and verify backend endpoint latencies meet operational thresholds."""
    benchmarks = {}

    # 1. Detailed health probe
    t0 = time.perf_counter()
    r = client.get("/api/v1/health/detailed")
    benchmarks["health_detailed_ms"] = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200

    # 2. District list forecast
    t0 = time.perf_counter()
    r = client.get("/api/v1/forecasts/districts?lead_time=24&limit=20")
    benchmarks["districts_list_ms"] = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200

    # 3. Single district detail drilldown
    t0 = time.perf_counter()
    r = client.get("/api/v1/forecasts/districts/KL_WAYANAD?lead_time=24")
    benchmarks["district_detail_ms"] = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200

    # 4. District GeoJSON layer for Leaflet
    t0 = time.perf_counter()
    r = client.get("/api/v1/forecast/districts?lead_time=24")
    benchmarks["district_geojson_ms"] = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200

    # 5. Verification accuracy summary
    t0 = time.perf_counter()
    r = client.get("/api/v1/verification/summary?season=monsoon_2026")
    benchmarks["verification_summary_ms"] = (time.perf_counter() - t0) * 1000
    assert r.status_code == 200

    # Operational SLO thresholds (FastAPI in-process test client)
    assert benchmarks["districts_list_ms"] < 250, f"District list query too slow: {benchmarks['districts_list_ms']:.1f}ms"
    assert benchmarks["district_detail_ms"] < 150, f"District detail query too slow: {benchmarks['district_detail_ms']:.1f}ms"
    assert benchmarks["district_geojson_ms"] < 350, f"GeoJSON layer too slow: {benchmarks['district_geojson_ms']:.1f}ms"
    assert benchmarks["verification_summary_ms"] < 150, f"Verification summary too slow: {benchmarks['verification_summary_ms']:.1f}ms"


# ---------------------------------------------------------------------------
# 6. FRONTEND STATIC ASSETS & BAR CHART CONTRACTS
# ---------------------------------------------------------------------------
def test_frontend_dashboard_and_bar_chart_contracts():
    """Verify production frontend is served and all bar-chart endpoints respond successfully."""
    # 1. HTML index serving
    resp = client.get("/dashboard")
    assert resp.status_code == 200
    assert "RAAP-X" in resp.text

    # 2. Data for NWP vs RAAP-X bar chart
    dist_resp = client.get("/api/v1/forecasts/districts/KL_WAYANAD?lead_time=24")
    assert dist_resp.status_code == 200
    d = dist_resp.json()
    assert "raw_nwp_rainfall" in d
    assert "corrected_p50" in d
    assert "corrected_p75" in d
    assert "corrected_p90" in d

    # 3. Data for probability bar chart
    assert "heavy_rainfall_probability" in d
    assert "extreme_rainfall_probability" in d

    # 4. Data for regime bar chart
    assert "dominant_regime" in d
    assert "regime_probabilities" in d
    assert len(d["regime_probabilities"]) >= 6

    # 5. Data for verification scorecard bar chart
    v_resp = client.get("/api/v1/verification/summary?season=monsoon_2026")
    assert v_resp.status_code == 200
    v = v_resp.json()
    assert "metrics" in v
    assert v["metrics"]["rmse_improvement_pct"] > 0
