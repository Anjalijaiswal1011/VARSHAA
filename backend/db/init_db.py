"""
Database Initialization & Seeding Module for RAIN-REPAIR X (PART 9).
Creates required tables and populates default administrative users and benchmark districts.
"""

from __future__ import annotations

import json
from backend.auth import DEFAULT_USERS
from backend.db.models import AuditLogModel, DistrictModel, ForecastRunModel, UserModel
from backend.db.session import Base, SessionLocal, engine
from src.gis.districts import BENCHMARK_DISTRICTS, DistrictGeometryManager
from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.db.init")


def init_db() -> None:
    """Creates database tables and seeds initial users and benchmark district boundaries."""
    logger.info("Initializing database schema...")
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    try:
        # 1. Seed Users
        existing_users = db.query(UserModel).count()
        if existing_users == 0:
            logger.info("Seeding default authentication users...")
            for u in DEFAULT_USERS.values():
                user_obj = UserModel(
                    user_id=u["user_id"],
                    username=u["username"],
                    password_hash=u["password_hash"],
                    salt=u["salt"],
                    role=u["role"],
                    api_key=u["api_key"],
                    is_active=u["is_active"],
                )
                db.add(user_obj)
            db.commit()
            logger.info("Seeded %d default users.", len(DEFAULT_USERS))

        # 2. Seed Districts
        existing_districts = db.query(DistrictModel).count()
        if existing_districts == 0:
            logger.info("Seeding benchmark district boundaries...")
            geom_mgr = DistrictGeometryManager()
            for d in geom_mgr.get_all_districts():
                props = d["properties"]
                geom_json = json.dumps(d["geometry"])
                dist_obj = DistrictModel(
                    district_id=props["district_id"],
                    district_name=props["district_name"],
                    state_name=props["state_name"],
                    subdivision=props.get("subdivision", ""),
                    centroid_lat=props["centroid_lat"],
                    centroid_lon=props["centroid_lon"],
                    area_km2=props.get("area_km2", 0.0),
                    geometry_json=geom_json,
                    boundary_version=props.get("boundary_version", "IMD-LGD-2026.1"),
                )
                db.add(dist_obj)
            db.commit()
            logger.info("Seeded %d benchmark districts.", len(geom_mgr.get_all_districts()))

    except Exception as e:
        db.rollback()
        logger.error("Error during database initialization: %s", e, exc_info=True)
    finally:
        db.close()
