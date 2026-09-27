"""
Integration tests for Pipeline Skeleton Components.
"""

from src.utils.config import load_config
from src.utils.exceptions import (
    DataQualityError,
    InvalidCoordinateError,
    PhysicalConstraintViolationError,
    RainRepairError,
)
from src.utils.logging import setup_logger


def test_exception_serialization_integrity():
    """Verify custom domain exceptions serialize structured metadata properly."""
    exc = InvalidCoordinateError(
        message="Latitude 51.5 is outside Indian domain",
        details={"lat": 51.5, "allowed_min": 6.0, "allowed_max": 38.0},
    )
    assert isinstance(exc, RainRepairError)
    d = exc.to_dict()
    assert d["error_type"] == "InvalidCoordinateError"
    assert "51.5" in d["message"]
    assert d["details"]["allowed_min"] == 6.0


def test_pipeline_configuration_and_logging_integration(tmp_path):
    """Verify configuration loads and writes to logger without side effects."""
    cfg = load_config()
    logger = setup_logger("test_pipeline_integration", log_level="DEBUG", log_dir=tmp_path)
    logger.info("Initializing test pipeline cycle for domain: %s", cfg.spatial.name)

    log_file = tmp_path / "rain_repair.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Initializing test pipeline cycle" in content
    assert "India_Subcontinent" in content
