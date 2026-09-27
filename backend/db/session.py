"""
Database Session & Connection Engine for RAIN-REPAIR X (PART 9).
Provides SQLite/Spatial persistence engine and transactional session management.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.utils.config import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger("rain_repair.backend.db")

DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "varshaa.db"

# SQLite connection URL with check_same_thread=False for multithreaded ASGI workers
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db() -> Generator:
    """Dependency that yields a managed database session and closes it on exit."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
