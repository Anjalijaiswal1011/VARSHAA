"""
Comprehensive Immutable Versioning Engine for RAIN-REPAIR X (PART 10).
Tracks distinct semantic version tags and integrity checksums across the entire ML lifecycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.versioning")

# Canonical System Component Versions
DATASET_VERSION_DEFAULT: str = "IMD-ERA5-v2.1"
FEATURE_VERSION_DEFAULT: str = "v1.0.0-phys"
REGIME_MODEL_VERSION_DEFAULT: str = "v1.0.0-regime-lgb"
REPAIR_MODEL_VERSION_DEFAULT: str = "v1.0.0-repair-smoe"
QUANTILE_MODEL_VERSION_DEFAULT: str = "v1.0.0-quantile-pinball"
CALIBRATION_VERSION_DEFAULT: str = "v1.0.0-platt"
EVT_VERSION_DEFAULT: str = "v1.0.0-evt-gpd"
BOUNDARY_VERSION_DEFAULT: str = "IMD-LGD-2026.1"
API_VERSION_DEFAULT: str = "v1.0.0"
DEPLOYMENT_VERSION_DEFAULT: str = "1.0.0-prod"


@dataclass(frozen=True)
class SystemVersionManifest:
    """
    Immutable Version Manifest capturing all dependent sub-components.
    Avoids vague 'latest_model' pointers in favor of fully auditable version hashes.
    """
    dataset_version: str = DATASET_VERSION_DEFAULT
    feature_version: str = FEATURE_VERSION_DEFAULT
    regime_model_version: str = REGIME_MODEL_VERSION_DEFAULT
    repair_model_version: str = REPAIR_MODEL_VERSION_DEFAULT
    quantile_model_version: str = QUANTILE_MODEL_VERSION_DEFAULT
    calibration_version: str = CALIBRATION_VERSION_DEFAULT
    evt_version: str = EVT_VERSION_DEFAULT
    boundary_version: str = BOUNDARY_VERSION_DEFAULT
    api_version: str = API_VERSION_DEFAULT
    deployment_version: str = DEPLOYMENT_VERSION_DEFAULT
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def compute_manifest_hash(self) -> str:
        """Computes deterministic SHA-256 fingerprint of the complete version manifest."""
        manifest_dict = {
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "regime_model_version": self.regime_model_version,
            "repair_model_version": self.repair_model_version,
            "quantile_model_version": self.quantile_model_version,
            "calibration_version": self.calibration_version,
            "evt_version": self.evt_version,
            "boundary_version": self.boundary_version,
            "api_version": self.api_version,
            "deployment_version": self.deployment_version,
        }
        raw_bytes = json.dumps(manifest_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw_bytes).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["manifest_hash"] = self.compute_manifest_hash()
        return d


class VersionManager:
    """
    Manages and persists system version manifests for reproducibility and governance.
    """

    _default_instance: Optional[VersionManager] = None

    def __init__(self, manifests_dir: Optional[Path] = None) -> None:
        self.manifests_dir = manifests_dir or Path("models/manifests")
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.current_manifest = SystemVersionManifest()

    @classmethod
    def get_default_manager(cls) -> VersionManager:
        if cls._default_instance is None:
            cls._default_instance = VersionManager()
        return cls._default_instance

    @classmethod
    def get_active_manifest(cls) -> SystemVersionManifest:
        return cls.get_default_manager().get_current_manifest()

    def get_current_manifest(self) -> SystemVersionManifest:
        return self.current_manifest

    @staticmethod
    def bump_version(base_version: str, bump_type: str = "patch") -> str:
        """
        Increments semantic version string while maintaining component prefix.
        e.g. 'FEAT-2.4.0' with 'minor' -> 'FEAT-2.5.0'
        """
        prefix = ""
        version_part = base_version
        if "-" in base_version:
            parts = base_version.split("-")
            prefix = parts[0] + "-"
            version_part = parts[1]

        semver = version_part.split(".")
        major = int(semver[0]) if len(semver) > 0 else 1
        minor = int(semver[1]) if len(semver) > 1 else 0
        patch = int(semver[2]) if len(semver) > 2 else 0

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"{prefix}{major}.{minor}.{patch}"

    def save_manifest(self, manifest: SystemVersionManifest) -> Path:
        m_hash = manifest.compute_manifest_hash()
        file_path = self.manifests_dir / f"manifest_{m_hash}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, indent=2)
        logger.info("Saved system version manifest to %s (Hash: %s).", file_path, m_hash)
        return file_path

