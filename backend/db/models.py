"""
SQLAlchemy Database Models for RAIN-REPAIR X (PART 9).
Defines persistence schemas for districts, forecast runs, audit logs, and users.
"""

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text
from backend.db.session import Base


class UserModel(Base):
    __tablename__ = "users"

    user_id = Column(String(64), primary_key=True, index=True)
    username = Column(String(64), unique=True, index=True, nullable=False)
    password_hash = Column(String(128), nullable=False)
    salt = Column(String(64), nullable=False)
    role = Column(String(32), default="public", nullable=False)
    api_key = Column(String(64), unique=True, index=True, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class DistrictModel(Base):
    __tablename__ = "districts"

    district_id = Column(String(64), primary_key=True, index=True)
    district_name = Column(String(128), nullable=False, index=True)
    state_name = Column(String(128), nullable=False, index=True)
    subdivision = Column(String(128), nullable=True)
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    area_km2 = Column(Float, nullable=True)
    geometry_json = Column(Text, nullable=False)
    boundary_version = Column(String(64), nullable=False, default="IMD-LGD-2026.1")
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class ForecastRunModel(Base):
    __tablename__ = "forecast_runs"

    run_id = Column(String(64), primary_key=True, index=True)
    cycle_date = Column(String(32), nullable=False, index=True)
    lead_time = Column(Integer, nullable=False, index=True)
    model_version = Column(String(64), nullable=False)
    boundary_version = Column(String(64), nullable=False)
    status = Column(String(32), default="complete", nullable=False)
    summary_json = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    log_id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
    request_id = Column(String(64), index=True, nullable=True)
    user_id = Column(String(64), nullable=True)
    client_ip = Column(String(64), nullable=True)
    endpoint = Column(String(256), nullable=False)
    method = Column(String(16), nullable=False)
    status_code = Column(Integer, nullable=False)
    response_time_ms = Column(Float, nullable=False)
