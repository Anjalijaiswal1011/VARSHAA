"""
Unit tests for Structured Logging and Secret Sanitization.
"""

import logging
from src.utils.logging import SecretMaskingFormatter, setup_logger


def test_secret_masking_formatter():
    """Verify that credentials and tokens are redacted from log outputs."""
    formatter = SecretMaskingFormatter()

    record = logging.LogRecord(
        name="test_logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Connecting to service with api_key='sk-secret12345' and password=\"supersecret\"",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)
    assert "sk-secret12345" not in formatted
    assert "supersecret" not in formatted
    assert "***REDACTED***" in formatted


def test_setup_logger_creates_instance():
    """Verify setup_logger returns a valid logger."""
    logger = setup_logger(name="test_rain_repair_instance")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test_rain_repair_instance"
