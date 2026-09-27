"""
Configuration Management Module for RAIN-REPAIR X (VARSHAA).
Loads YAML configuration, validates settings, and applies environment variable overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


def find_project_root() -> Path:
    """Traverse upwards from the current file to locate the project root containing configs/."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "configs" / "base_config.yaml").exists() or (parent / ".git").exists():
            return parent
    return Path.cwd()


PROJECT_ROOT = find_project_root()


@dataclass(frozen=True)
class SpatialDomainConfig:
    name: str = "India_Subcontinent"
    crs: str = "EPSG:4326"
    lat_min: float = 6.0
    lat_max: float = 38.0
    lon_min: float = 68.0
    lon_max: float = 98.0
    resolution_deg: float = 0.25


@dataclass(frozen=True)
class PathsConfig:
    data_raw_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "raw")
    data_interim_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "interim")
    data_processed_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "processed")
    data_external_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "data" / "external")
    models_checkpoint_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "models" / "checkpoints")
    logs_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "logs")


@dataclass(frozen=True)
class TemporalConfig:
    accumulation_hours: int = 24
    cycle_hour_utc: str = "00:00"
    lead_times_hours: List[int] = field(default_factory=lambda: [24, 48, 72, 96, 120])


@dataclass(frozen=True)
class ThresholdsConfig:
    heavy_rain_mm: float = 64.5
    very_heavy_rain_mm: float = 115.6
    extreme_rain_mm: float = 204.5
    quantiles: List[float] = field(default_factory=lambda: [0.10, 0.50, 0.75, 0.90, 0.95])


@dataclass(frozen=True)
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    prefix: str = "/api/v1"
    cors_origins: List[str] = field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
        ]
    )


@dataclass
class AppConfig:
    project_name: str = "RAIN-REPAIR X (VARSHAA)"
    version: str = "1.0.0"
    environment: str = "development"
    random_seed: int = 42
    paths: PathsConfig = field(default_factory=PathsConfig)
    spatial: SpatialDomainConfig = field(default_factory=SpatialDomainConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    regimes: List[str] = field(
        default_factory=lambda: [
            "ACTIVE_MONSOON",
            "BREAK_MONSOON",
            "MONSOON_DEPRESSION",
            "WESTERN_DISTURBANCE",
            "OFFSHORE_TROUGH",
            "NORMAL_TRANSITIONAL",
        ]
    )


def load_config(config_path: Optional[Path] = None) -> AppConfig:
    """
    Load configuration from YAML file and apply environment variable overrides.
    """
    if config_path is None:
        config_path = PROJECT_ROOT / "configs" / "base_config.yaml"

    yaml_data: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    project_data = yaml_data.get("project", {})
    paths_data = yaml_data.get("paths", {})
    spatial_data = yaml_data.get("spatial_domain", {})
    temporal_data = yaml_data.get("temporal", {})
    thresholds_data = yaml_data.get("thresholds", {})
    api_data = yaml_data.get("api", {})
    regimes_data = yaml_data.get("regimes", {}).get("taxonomy", [])

    env = os.getenv("ENVIRONMENT", project_data.get("environment", "development"))
    seed = int(os.getenv("DEFAULT_RANDOM_SEED", project_data.get("random_seed", 42)))

    # Path overrides
    def resolve_path(env_key: str, yaml_val: str, default_sub: str) -> Path:
        val = os.getenv(env_key, yaml_val or default_sub)
        p = Path(val)
        return p if p.is_absolute() else PROJECT_ROOT / p

    paths = PathsConfig(
        data_raw_dir=resolve_path("DATA_RAW_DIR", paths_data.get("data_raw_dir"), "data/raw"),
        data_interim_dir=resolve_path("DATA_INTERIM_DIR", paths_data.get("data_interim_dir"), "data/interim"),
        data_processed_dir=resolve_path("DATA_PROCESSED_DIR", paths_data.get("data_processed_dir"), "data/processed"),
        data_external_dir=resolve_path("DATA_EXTERNAL_DIR", paths_data.get("data_external_dir"), "data/external"),
        models_checkpoint_dir=resolve_path("MODELS_CHECKPOINT_DIR", paths_data.get("models_checkpoint_dir"), "models/checkpoints"),
        logs_dir=resolve_path("LOGS_DIR", paths_data.get("logs_dir"), "logs"),
    )

    spatial = SpatialDomainConfig(
        name=spatial_data.get("name", "India_Subcontinent"),
        crs=spatial_data.get("crs", "EPSG:4326"),
        lat_min=float(os.getenv("LAT_MIN", spatial_data.get("lat_min", 6.0))),
        lat_max=float(os.getenv("LAT_MAX", spatial_data.get("lat_max", 38.0))),
        lon_min=float(os.getenv("LON_MIN", spatial_data.get("lon_min", 68.0))),
        lon_max=float(os.getenv("LON_MAX", spatial_data.get("lon_max", 98.0))),
        resolution_deg=float(os.getenv("GRID_RESOLUTION_DEG", spatial_data.get("resolution_deg", 0.25))),
    )

    temporal = TemporalConfig(
        accumulation_hours=temporal_data.get("accumulation_hours", 24),
        cycle_hour_utc=temporal_data.get("cycle_hour_utc", "00:00"),
        lead_times_hours=temporal_data.get("lead_times_hours", [24, 48, 72, 96, 120]),
    )

    thresholds = ThresholdsConfig(
        heavy_rain_mm=float(os.getenv("HEAVY_RAIN_THRESHOLD_MM", thresholds_data.get("heavy_rain_mm", 64.5))),
        very_heavy_rain_mm=float(os.getenv("VERY_HEAVY_RAIN_THRESHOLD_MM", thresholds_data.get("very_heavy_rain_mm", 115.6))),
        extreme_rain_mm=float(os.getenv("EXTREME_RAIN_THRESHOLD_MM", thresholds_data.get("extreme_rain_mm", 204.5))),
        quantiles=thresholds_data.get("quantiles", [0.10, 0.50, 0.75, 0.90, 0.95]),
    )

    api = ApiConfig(
        host=os.getenv("API_HOST", api_data.get("host", "0.0.0.0")),
        port=int(os.getenv("API_PORT", api_data.get("port", 8000))),
        prefix=os.getenv("API_PREFIX", api_data.get("prefix", "/api/v1")),
        cors_origins=api_data.get("cors_origins", ["http://localhost:3000", "http://localhost:5173"]),
    )

    return AppConfig(
        project_name=project_data.get("name", "RAIN-REPAIR X (VARSHAA)"),
        version=project_data.get("version", "1.0.0"),
        environment=env,
        random_seed=seed,
        paths=paths,
        spatial=spatial,
        temporal=temporal,
        thresholds=thresholds,
        api=api,
        regimes=regimes_data or AppConfig().regimes,
    )


# Singleton instance for direct import
config = load_config()
