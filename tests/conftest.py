"""
Pytest Fixtures and Global Test Configuration for RAIN-REPAIR X (VARSHAA).
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path for test discovery
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import AppConfig, load_config


@pytest.fixture(scope="session")
def test_config() -> AppConfig:
    """Session-scoped application configuration for tests."""
    return load_config()


@pytest.fixture
def sample_valid_grid_coords():
    """Returns typical valid India subcontinent bounding coordinates."""
    return {
        "pune": {"lat": 18.5204, "lon": 73.8567},
        "delhi": {"lat": 28.6139, "lon": 77.2090},
        "guwahati": {"lat": 26.1445, "lon": 91.7362},
    }


@pytest.fixture
def sample_invalid_grid_coords():
    """Returns out-of-bounds coordinates (outside India domain)."""
    return {
        "london": {"lat": 51.5074, "lon": -0.1278},
        "sydney": {"lat": -33.8688, "lon": 151.2093},
    }


@pytest.fixture
def sample_prediction_record():
    """Returns a valid dictionary adhering to CONTRACT-ML-002."""
    return {
        "cycle_date": "2026-07-15",
        "lead_time": 24,
        "lat": 18.50,
        "lon": 73.75,
        "raw_nwp_precip": 45.2,
        "corrected_p10": 38.0,
        "corrected_p50": 52.4,
        "corrected_p75": 61.0,
        "corrected_p90": 68.1,
        "corrected_p95": 74.5,
        "prob_heavy_rain": 0.35,
        "prob_very_heavy_rain": 0.08,
        "prob_extreme_rain": 0.01,
        "active_regime": "ACTIVE_MONSOON",
        "regime_confidence": 0.88,
        "delta_correction": 7.2,
    }
