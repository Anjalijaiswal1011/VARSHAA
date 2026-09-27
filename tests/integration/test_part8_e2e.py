"""
End-to-End Integration Test for PART 8:
Part 7 Probabilistic Risk -> Grid Output -> GIS Aggregation -> District Output -> API Response.
"""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from src.gis.alerts import IMDAlertEngine
from src.gis.crs import CRSManager
from src.gis.districts import DistrictGeometryManager
from src.gis.grid_schema import GridForecastQualityValidator
from src.gis.layers import MapLayerContractManager
from src.gis.zonal import SpatialZonalStatisticsEngine
from src.models.explainability import ModelTransparencyEngine
from src.postprocessing.risk import CalibratedRiskEngine

client = TestClient(app)


def test_part8_end_to_end_pipeline():
    """
    Simulates complete operational pipeline from Part 7 to API.
    Validates data traceability, non-negativity, quantile ordering,
    hotspot detection, and endpoint responses.
    """
    # Step 1: Part 7 Probabilistic Model Output Simulation
    n_points = 20
    np.random.seed(42)
    lats = np.linspace(18.0, 19.5, n_points)
    lons = np.linspace(72.5, 74.0, n_points)
    raw_nwp = np.random.uniform(20.0, 80.0, n_points)
    corrected_p50 = np.maximum(0.0, raw_nwp + np.random.normal(5.0, 3.0, n_points))
    corrected_p10 = corrected_p50 * 0.7
    corrected_p75 = corrected_p50 * 1.3
    corrected_p90 = corrected_p50 * 1.65
    corrected_p95 = corrected_p50 * 1.9

    grid_df = pd.DataFrame(
        {
            "forecast_time": "2026-07-15T00:00:00Z",
            "valid_time": "2026-07-16T03:00:00Z",
            "grid_id": [f"G_{la:.2f}_{lo:.2f}" for la, lo in zip(lats, lons)],
            "latitude": lats,
            "longitude": lons,
            "lat": lats,
            "lon": lons,
            "lead_time": 24,
            "raw_nwp_rainfall": raw_nwp,
            "raw_nwp_precip": raw_nwp,
            "corrected_rainfall": corrected_p50,
            "corrected_p10": corrected_p10,
            "corrected_p50": corrected_p50,
            "corrected_p75": corrected_p75,
            "corrected_p90": corrected_p90,
            "corrected_p95": corrected_p95,
            "p50_rainfall": corrected_p50,
            "p75_rainfall": corrected_p75,
            "p90_rainfall": corrected_p90,
            "heavy_probability": np.clip(corrected_p90 / 64.5 * 0.45, 0.0, 1.0),
            "very_heavy_probability": np.clip(corrected_p90 / 115.6 * 0.35, 0.0, 1.0),
            "extreme_probability": np.clip(corrected_p90 / 204.5 * 0.20, 0.0, 1.0),
            "regime": "ACTIVE_MONSOON",
            "active_regime": "ACTIVE_MONSOON",
            "uncertainty_indicator": corrected_p90 - corrected_p10,
            "model_version": "v1.0.0-prob",
            "evt_status": "CONVERGED_NORMAL",
        }
    )

    # Step 2: Grid Scientific Quality Check
    validated_df, val_summary = GridForecastQualityValidator.validate_dataframe(grid_df)
    assert val_summary["validation_passed"] is True
    assert (validated_df["quality_flag"] == "VALID").all()

    # Step 3: Grid -> District Zonal Aggregation
    geom_mgr = DistrictGeometryManager()
    zonal_engine = SpatialZonalStatisticsEngine(geometry_manager=geom_mgr)
    district_records = zonal_engine.aggregate_grid_to_districts(grid_df=validated_df, lead_time=24)
    assert len(district_records) > 0

    # Step 4: IMD Alerts & GIS Layer
    alert_engine = IMDAlertEngine()
    enriched = alert_engine.enrich_district_records(district_records)
    assert len(enriched) == len(district_records)

    geojson = alert_engine.build_geojson_feature_collection(enriched)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == len(enriched)

    # Step 5: Hotspot & Machine-Readable Explanation
    pune_rec = next(r for r in district_records if r["district_id"] == "MH_PUNE")
    assert "hotspot" in pune_rec
    assert pune_rec["hotspot"]["hotspot_value_mm"] > 0

    transparency = ModelTransparencyEngine()
    exp = transparency.explain_district_forecast(pune_rec)
    assert exp["district_id"] == "MH_PUNE"
    assert len(exp["main_drivers"]) >= 3
    assert exp["dominant_regime"] == "ACTIVE_MONSOON"

    # Step 6: Backend API Verification
    r_health = client.get("/api/v1/health")
    assert r_health.status_code == 200

    r_grid = client.get("/api/v1/forecast/grid?lat=18.52&lon=73.85")
    assert r_grid.status_code == 200

    r_dist = client.get("/api/v1/forecast/districts?lead_time=24")
    assert r_dist.status_code == 200

    r_single = client.get("/api/v1/forecast/district/MH_PUNE?lead_time=24")
    assert r_single.status_code == 200
    assert r_single.json()["district_id"] == "MH_PUNE"

    r_map = client.get("/api/v1/forecast/map")
    assert r_map.status_code == 200

    r_comp = client.get("/api/v1/forecast/compare?level=district&lead_time=24")
    assert r_comp.status_code == 200

    r_exp = client.get("/api/v1/forecast/explanation/MH_PUNE?lead_time=24")
    assert r_exp.status_code == 200
