"""
Structured Alert Management & Notification Engine for RAAP-X MLOps (Phase 7).
Tracks, persists, and queries operational alerts across all pipeline lifecycle stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
import uuid

from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.alerts")

AlertSeverity = Literal["INFO", "WARNING", "CRITICAL"]

AlertCategory = Literal[
    "DATA_MISSING",
    "DATA_STALE",
    "DATA_INVALID",
    "FEATURE_DRIFT",
    "REGIME_DRIFT",
    "MODEL_PERFORMANCE_DEGRADATION",
    "CALIBRATION_DEGRADATION",
    "PIPELINE_FAILURE",
    "MODEL_CHECKPOINT_FAILURE",
    "GIS_MAPPING_FAILURE",
]

DEFAULT_RECOMMENDED_ACTIONS: Dict[str, str] = {
    "DATA_MISSING": "Verify upstream NWP server sync, check data ingest directory, and inspect file availability.",
    "DATA_STALE": "Check NWP cycle dissemination schedule and verify system clock synchronization.",
    "DATA_INVALID": "Inspect raw input fields for corrupt values, unit discrepancies, or NaN spikes.",
    "FEATURE_DRIFT": "Audit atmospheric predictor distributions. Review feature engineering pipeline; flag for domain evaluation.",
    "REGIME_DRIFT": "Inspect synoptic circulation patterns. Unusual weather regime concentration requires meteorological review.",
    "MODEL_PERFORMANCE_DEGRADATION": "Flag RETRAINING_REVIEW_REQUIRED. Evaluate model against holdout verification data; consider rollback.",
    "CALIBRATION_DEGRADATION": "Check reliability diagrams and Platt scaling/quantile mapping parameters.",
    "PIPELINE_FAILURE": "Check stage failure logs, ensure dependencies exist, and inspect input data validity.",
    "MODEL_CHECKPOINT_FAILURE": "Verify artifact checksums on disk and roll back to previous validated model checkpoint.",
    "GIS_MAPPING_FAILURE": "Verify administrative boundary GeoJSON files and check coordinate reference system (CRS).",
}


@dataclass
class AlertRecord:
    """Standardized Alert Record conforming to Phase 7 Section 12."""
    alert_id: str = field(default_factory=lambda: f"alt_{uuid.uuid4().hex[:12]}")
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    severity: AlertSeverity = "WARNING"
    category: AlertCategory = "PIPELINE_FAILURE"
    message: str = ""
    pipeline_run_id: Optional[str] = None
    dataset_version: Optional[str] = None
    model_version: Optional[str] = None
    recommended_action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.recommended_action and self.category in DEFAULT_RECOMMENDED_ACTIONS:
            self.recommended_action = DEFAULT_RECOMMENDED_ACTIONS[self.category]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertManager:
    """
    In-memory and file-backed Alert Store for RAAP-X MLOps telemetry.
    """

    _default_instance: Optional[AlertManager] = None

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir or Path("logs/alerts")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.alerts: List[AlertRecord] = []
        self._load_persisted_alerts()

    @classmethod
    def get_default_manager(cls) -> AlertManager:
        if cls._default_instance is None:
            cls._default_instance = AlertManager()
        return cls._default_instance

    def _load_persisted_alerts(self) -> None:
        alert_file = self.storage_dir / "active_alerts.json"
        if alert_file.exists():
            try:
                with open(alert_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.alerts = [AlertRecord(**d) for d in data]
            except Exception as e:
                logger.warning("Could not load persisted alerts from %s: %s", alert_file, e)

    def _persist_alerts(self) -> None:
        alert_file = self.storage_dir / "active_alerts.json"
        try:
            with open(alert_file, "w", encoding="utf-8") as f:
                json.dump([a.to_dict() for a in self.alerts], f, indent=2)
        except Exception as e:
            logger.error("Failed to persist alerts: %s", e)

    def record_alert(
        self,
        category: AlertCategory,
        message: str,
        severity: AlertSeverity = "WARNING",
        pipeline_run_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        model_version: Optional[str] = None,
        recommended_action: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> AlertRecord:
        """Creates, logs, and stores a new operational alert."""
        alert = AlertRecord(
            category=category,
            message=message,
            severity=severity,
            pipeline_run_id=pipeline_run_id,
            dataset_version=dataset_version,
            model_version=model_version,
            recommended_action=recommended_action or DEFAULT_RECOMMENDED_ACTIONS.get(category, ""),
            details=details or {},
        )
        self.alerts.append(alert)
        self._persist_alerts()

        log_fn = logger.error if severity == "CRITICAL" else (logger.warning if severity == "WARNING" else logger.info)
        log_fn(
            "[%s ALERT] %s - Run: %s - Action: %s",
            category,
            message,
            pipeline_run_id or "N/A",
            alert.recommended_action,
        )
        return alert

    def list_alerts(
        self,
        severity: Optional[str] = None,
        category: Optional[str] = None,
        resolved: Optional[bool] = None,
        limit: int = 100,
    ) -> List[AlertRecord]:
        """Queries stored alerts with optional filtering."""
        res = self.alerts
        if severity:
            res = [a for a in res if a.severity == severity]
        if category:
            res = [a for a in res if a.category == category]
        if resolved is not None:
            res = [a for a in res if a.resolved == resolved]
        return res[-limit:]

    def resolve_alert(self, alert_id: str) -> bool:
        """Marks an alert as resolved."""
        for a in self.alerts:
            if a.alert_id == alert_id:
                a.resolved = True
                a.resolved_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                self._persist_alerts()
                return True
        return False

    def clear(self) -> None:
        """Clears in-memory and persisted alerts (useful for testing)."""
        self.alerts.clear()
        alert_file = self.storage_dir / "active_alerts.json"
        if alert_file.exists():
            try:
                alert_file.unlink()
            except Exception:
                pass
