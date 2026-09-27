"""
Request Validation & Input Sanitization Engine for RAIN-REPAIR X (PART 9).
Defines domain-specific validators and protection guards against malicious queries.
"""

from __future__ import annotations

from datetime import datetime
import re
from typing import Optional, Set
from fastapi import HTTPException, status

SUPPORTED_LEAD_TIMES: Set[int] = {24, 48, 72, 96, 120}
MIN_LAT: float = 6.0
MAX_LAT: float = 38.5
MIN_LON: float = 68.0
MAX_LON: float = 98.0

DISTRICT_ID_REGEX = re.compile(r"^[A-Z]{2}_[A-Z0-9_]{2,32}$", re.IGNORECASE)
DANGEROUS_SQL_PATTERNS = re.compile(r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|OR|AND)\b|--|;|\/\*)", re.IGNORECASE)


def validate_coordinates(lat: float, lon: float) -> None:
    """
    Validates that latitude and longitude fall strictly within the Indian meteorological domain.
    """
    if not (MIN_LAT <= lat <= MAX_LAT) or not (MIN_LON <= lon <= MAX_LON):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Coordinates ({lat}, {lon}) outside India meteorological domain. "
                f"Latitude must be within [{MIN_LAT}, {MAX_LAT}], Longitude within [{MIN_LON}, {MAX_LON}]."
            ),
        )


def validate_lead_time(lead_time: int) -> None:
    """
    Validates that forecast lead time is one of the supported standard horizons.
    """
    if lead_time not in SUPPORTED_LEAD_TIMES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported lead time {lead_time}h. Supported lead times are: {sorted(SUPPORTED_LEAD_TIMES)}.",
        )


def validate_district_id(district_id: str) -> str:
    """
    Sanitizes and validates district identifier format (e.g. 'MH_PUNE').
    """
    clean_id = district_id.strip()
    if not clean_id or len(clean_id) > 64:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="District ID must be a non-empty string with maximum 64 characters.",
        )
    # Check for path traversal or malicious SQL injection tokens
    if ".." in clean_id or "/" in clean_id or "\\" in clean_id or DANGEROUS_SQL_PATTERNS.search(clean_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid characters in district identifier.",
        )
    return clean_id


def validate_iso_date(date_str: Optional[str]) -> Optional[str]:
    """
    Validates standard ISO 8601 date string format (YYYY-MM-DD).
    """
    if not date_str:
        return None
    try:
        if "T" in date_str:
            datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        else:
            datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid date format '{date_str}'. Expected ISO format YYYY-MM-DD or YYYY-MM-DDTHH:MM:SSZ.",
        )
