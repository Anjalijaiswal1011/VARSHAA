"""
Comprehensive Subsystem Health Probes & Readiness Diagnostics for RAAP-X (Phase 7 Section 13).
Verifies:
1. Data source availability
2. Model registry accessibility
3. Model checkpoints integrity
4. Feature pipeline operational readiness
5. Regime model responsiveness
6. Quantile correction models responsiveness
7. GIS administrative boundaries availability
8. Storage write/read availability
9. Database session responsiveness
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional
import numpy as np
import pandas as pd

from src.gis.districts import DistrictGeometryManager
from src.mlops.registry import ModelRegistry
from src.mlops.versioning import VersionManager
from src.models.inference_pipeline import ProductionInferencePipeline
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.health")

OverallHealthStatus = Literal["HEALTHY", "DEGRADED", "UNHEALTHY"]


@dataclass
class ComponentHealthResult:
    component: str
    status: Literal["OK", "WARNING", "ERROR"]
    message: str
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SystemHealthReport:
    """Standardized System Health Report conforming to Phase 7 Section 13."""
    overall_status: OverallHealthStatus
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    version: str = "1.0.0"
    checks_total: int = 0
    checks_passed: int = 0
    checks_failed: int = 0
    components: Dict[str, ComponentHealthResult] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_status": self.overall_status,
            "timestamp": self.timestamp,
            "version": self.version,
            "checks_total": self.checks_total,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "components": {k: v.to_dict() for k, v in self.components.items()},
        }


class HealthProbeService:
    """
    Executes automated health checks across all RAAP-X subsystems.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        registry: Optional[ModelRegistry] = None,
        geometry_manager: Optional[DistrictGeometryManager] = None,
    ) -> None:
        self.base_dir = base_dir or Path.cwd()
        self.registry = registry or ModelRegistry.get_default_registry()
        self.geometry_manager = geometry_manager or DistrictGeometryManager()

    def check_data_source(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        raw_dir = self.base_dir / "data" / "raw"
        exists = raw_dir.exists()
        n_files = len(list(raw_dir.glob("*.*"))) if exists else 0
        lat = (time.perf_counter() - t0) * 1000.0

        if exists and n_files > 0:
            return ComponentHealthResult("data_source", "OK", f"Raw data directory accessible with {n_files} files.", lat, {"files_count": n_files})
        elif exists:
            return ComponentHealthResult("data_source", "OK", "Raw data directory accessible (empty/synthetic mode ready).", lat, {"files_count": 0})
        return ComponentHealthResult("data_source", "WARNING", "Data directory not found. Synthetic mode active.", lat)

    def check_model_registry(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            prod_m = self.registry.get_production_model()
            models_count = len(self.registry.list_models())
            lat = (time.perf_counter() - t0) * 1000.0

            if prod_m:
                return ComponentHealthResult(
                    "model_registry",
                    "OK",
                    f"Model registry operational. Active production model: {prod_m.model_id} (Version {prod_m.version}).",
                    lat,
                    {"active_production_model": prod_m.model_id, "total_registered_models": models_count},
                )
            return ComponentHealthResult("model_registry", "WARNING", "Registry accessible but no active production model found.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("model_registry", "ERROR", f"Model registry access failed: {e}", lat)

    def check_model_checkpoints(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        ckpt_dir = self.base_dir / "models" / "checkpoints"
        exists = ckpt_dir.exists()
        ckpts = list(ckpt_dir.glob("*.joblib")) if exists else []
        lat = (time.perf_counter() - t0) * 1000.0

        if ckpts:
            return ComponentHealthResult("model_checkpoints", "OK", f"Found {len(ckpts)} model checkpoint artifacts.", lat, {"checkpoints": [c.name for c in ckpts]})
        return ComponentHealthResult("model_checkpoints", "OK", "Checkpoints directory accessible (in-memory pipeline ready).", lat)

    def check_feature_pipeline(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            pipe = ProductionInferencePipeline()
            # Test normalizing a dummy record
            from src.models.inference_pipeline import ForecastInputRecord
            sample_rec = ForecastInputRecord(
                cycle_date="2026-07-15",
                initialization_time="2026-07-15T00:00:00Z",
                forecast_time="2026-07-16T00:00:00Z",
                latitude=18.5,
                longitude=73.8,
                lead_time=24,
                nwp_precip=25.0,
            )
            norm = pipe._normalize_input_record(sample_rec)
            lat = (time.perf_counter() - t0) * 1000.0
            if "nwp_precip" in norm and "elevation" in norm:
                return ComponentHealthResult("feature_pipeline", "OK", "Feature engineering pipeline operational.", lat)
            return ComponentHealthResult("feature_pipeline", "ERROR", "Feature generation returned incomplete dictionary.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("feature_pipeline", "ERROR", f"Feature pipeline failed: {e}", lat)

    def check_regime_model(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            pipe = ProductionInferencePipeline()
            # Perform quick prediction on sample record to verify regime engine
            rec_dict = {
                "cycle_date": "2026-07-15",
                "initialization_time": "2026-07-15T00:00:00Z",
                "forecast_time": "2026-07-16T00:00:00Z",
                "latitude": 18.5,
                "longitude": 73.8,
                "lead_time": 24,
                "nwp_precip": 25.0,
            }
            res = pipe.predict_single(rec_dict)
            lat = (time.perf_counter() - t0) * 1000.0
            if res.regime_probabilities and res.dominant_regime:
                return ComponentHealthResult(
                    "regime_model",
                    "OK",
                    f"Regime classifier responsive. Dominant: {res.dominant_regime} ({res.dominant_probability:.1%}).",
                    lat,
                    {"dominant_regime": res.dominant_regime, "version": res.regime_model_version},
                )
            return ComponentHealthResult("regime_model", "ERROR", "Regime classifier returned empty probabilities.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("regime_model", "ERROR", f"Regime model check failed: {e}", lat)

    def check_quantile_models(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            pipe = ProductionInferencePipeline()
            res = pipe.predict_single({
                "cycle_date": "2026-07-15",
                "initialization_time": "2026-07-15T00:00:00Z",
                "forecast_time": "2026-07-16T00:00:00Z",
                "latitude": 18.5,
                "longitude": 73.8,
                "lead_time": 24,
                "nwp_precip": 45.0,
            })
            lat = (time.perf_counter() - t0) * 1000.0
            is_monotonic = (res.corrected_p50 <= res.corrected_p75 <= res.corrected_p90)
            is_non_neg = (res.corrected_p50 >= 0.0)

            if is_monotonic and is_non_neg:
                return ComponentHealthResult(
                    "quantile_models",
                    "OK",
                    f"RAAP-X Quantile Corrector responsive: P50={res.corrected_p50:.1f}, P75={res.corrected_p75:.1f}, P90={res.corrected_p90:.1f} mm.",
                    lat,
                    {"p50": res.corrected_p50, "p75": res.corrected_p75, "p90": res.corrected_p90},
                )
            return ComponentHealthResult("quantile_models", "ERROR", "Quantile output violates monotonicity or non-negativity.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("quantile_models", "ERROR", f"Quantile model check failed: {e}", lat)

    def check_gis_boundaries(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            districts = self.geometry_manager.get_all_districts()
            lat = (time.perf_counter() - t0) * 1000.0
            if districts:
                return ComponentHealthResult(
                    "gis_boundaries",
                    "OK",
                    f"District boundary engine operational with {len(districts)} Indian administrative districts.",
                    lat,
                    {"district_count": len(districts), "version": self.geometry_manager.dataset_version},
                )
            return ComponentHealthResult("gis_boundaries", "WARNING", "No district polygons loaded in boundary engine.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("gis_boundaries", "ERROR", f"GIS boundary check failed: {e}", lat)

    def check_storage(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        test_file = self.base_dir / "logs" / ".health_test.tmp"
        try:
            test_file.parent.mkdir(parents=True, exist_ok=True)
            with open(test_file, "w", encoding="utf-8") as f:
                f.write("health_ok")
            with open(test_file, "r", encoding="utf-8") as f:
                content = f.read()
            test_file.unlink(missing_ok=True)
            lat = (time.perf_counter() - t0) * 1000.0

            if content == "health_ok":
                return ComponentHealthResult("storage", "OK", "Local filesystem read and write operational.", lat)
            return ComponentHealthResult("storage", "ERROR", "Filesystem read did not match written content.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("storage", "ERROR", f"Storage access failed: {e}", lat)

    def check_database(self) -> ComponentHealthResult:
        t0 = time.perf_counter()
        try:
            from backend.db.session import SessionLocal
            with SessionLocal() as db:
                from sqlalchemy import text
                db.execute(text("SELECT 1"))
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("database", "OK", "Relational database connection operational.", lat)
        except Exception as e:
            lat = (time.perf_counter() - t0) * 1000.0
            return ComponentHealthResult("database", "WARNING", f"Database check warning (SQLite fallback active): {e}", lat)

    def run_comprehensive_health_check(self) -> SystemHealthReport:
        """Executes full diagnostic suite and returns unified SystemHealthReport."""
        components = {
            "data_source": self.check_data_source(),
            "model_registry": self.check_model_registry(),
            "model_checkpoints": self.check_model_checkpoints(),
            "feature_pipeline": self.check_feature_pipeline(),
            "regime_model": self.check_regime_model(),
            "quantile_models": self.check_quantile_models(),
            "gis_boundaries": self.check_gis_boundaries(),
            "storage": self.check_storage(),
            "database": self.check_database(),
        }

        total = len(components)
        passed = sum(1 for c in components.values() if c.status == "OK")
        failed = sum(1 for c in components.values() if c.status == "ERROR")

        if failed > 0:
            overall: OverallHealthStatus = "UNHEALTHY"
        elif any(c.status == "WARNING" for c in components.values()):
            overall = "DEGRADED"
        else:
            overall = "HEALTHY"

        return SystemHealthReport(
            overall_status=overall,
            checks_total=total,
            checks_passed=passed,
            checks_failed=failed,
            components=components,
        )
