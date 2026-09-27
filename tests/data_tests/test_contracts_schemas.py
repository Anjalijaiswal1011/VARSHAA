"""
Data & Interface Contract Schema Validation Tests.
Tests adherence to CONTRACT-DATA-001 and CONTRACT-ML-002.
"""

import pytest
from src.utils.config import AppConfig


def test_spatial_coordinate_validation(test_config: AppConfig, sample_valid_grid_coords, sample_invalid_grid_coords):
    """Ensure coordinates inside Indian domain pass and outside coordinates fail bounds check."""
    bounds = test_config.spatial

    def is_in_domain(lat: float, lon: float) -> bool:
        return bounds.lat_min <= lat <= bounds.lat_max and bounds.lon_min <= lon <= bounds.lon_max

    for city, coord in sample_valid_grid_coords.items():
        assert is_in_domain(coord["lat"], coord["lon"]), f"{city} should be inside India domain"

    for city, coord in sample_invalid_grid_coords.items():
        assert not is_in_domain(coord["lat"], coord["lon"]), f"{city} should be outside India domain"


def test_ml_contract_physical_realizability(sample_prediction_record):
    """
    Verify CONTRACT-ML-002 physical invariants:
    1. Rainfall >= 0.0 mm
    2. Quantile monotonicity: P10 <= P50 <= P75 <= P90 <= P95
    3. Probability bounds [0.0, 1.0]
    """
    rec = sample_prediction_record

    # 1. Non-negativity
    assert rec["raw_nwp_precip"] >= 0.0
    assert rec["corrected_p10"] >= 0.0
    assert rec["corrected_p50"] >= 0.0
    assert rec["corrected_p75"] >= 0.0
    assert rec["corrected_p90"] >= 0.0
    assert rec["corrected_p95"] >= 0.0

    # 2. Quantile Monotonicity
    assert rec["corrected_p10"] <= rec["corrected_p50"]
    assert rec["corrected_p50"] <= rec["corrected_p75"]
    assert rec["corrected_p75"] <= rec["corrected_p90"]
    assert rec["corrected_p90"] <= rec["corrected_p95"]

    # 3. Probability bounds
    for prob_key in ["prob_heavy_rain", "prob_very_heavy_rain", "prob_extreme_rain"]:
        assert 0.0 <= rec[prob_key] <= 1.0

    # 4. Delta calculation consistency
    expected_delta = round(rec["corrected_p50"] - rec["raw_nwp_precip"], 2)
    assert round(rec["delta_correction"], 2) == expected_delta
