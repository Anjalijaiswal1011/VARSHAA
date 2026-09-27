"""
Structured Logging Module for RAIN-REPAIR X (VARSHAA).
Provides uniform console & file logging with secret masking.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Optional

# Regular expression to mask common sensitive tokens/keys in log strings
SENSITIVE_PATTERNS = [
    re.compile(r"(?i)(api[_-]?key|token|password|secret)\s*[:=]\s*['\"]?([^'\"\s]+)['\"]?"),
]


class SecretMaskingFormatter(logging.Formatter):
    """Custom logging formatter that scrubs sensitive credentials from log records."""

    def format(self, record: logging.LogRecord) -> str:
        original = super().format(record)
        sanitized = original
        for pattern in SENSITIVE_PATTERNS:
            sanitized = pattern.sub(r"\1=***REDACTED***", sanitized)
        return sanitized


def setup_logger(
    name: str = "rain_repair",
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and retrieve a structured application logger.

    Args:
        name: Name of the logger module.
        log_level: Minimum logging level (DEBUG, INFO, WARNING, ERROR).
        log_dir: Path to directory where application log files should be written.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    formatter = SecretMaskingFormatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_path = log_dir / "rain_repair.log"
        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rain_repair") -> logging.Logger:
    """Convenience getter for existing logger."""
    return logging.getLogger(name)
