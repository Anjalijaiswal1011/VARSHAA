"""
Data Quality Monitoring & Telemetry Engine for RAAP-X (Phase 7 Section 7).
Audits completeness, validity, timeliness, and spatial coverage for every ingestion cycle.
Persists quality check reports and triggers structured alerts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.mlops.alerts import AlertManager
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.data_quality")

QualityStatus = Literal["PASSED", "WARNING", "FAILED"]

# Canonical expected meteorological variables
EXPECTED_VARIABLES = [
    "nwp_precip",
    "t2m",
    "u10",
    "v10",
    "mslp",
]

# Physical plausible limits for meteorological variables over Indian Subcontinent
PHYSICAL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "nwp_precip": (0.0, 1500.0),       # mm/24h (Cherrapunji/Mawsynram record max ~1040mm)
    "t2m": (210.0, 340.0),             # Kelvin (~ -63 C to +67 C)
    "u10": (-100.0, 100.0),            # m/s
    "v10": (-100.0, 100.0),            # m/s
    "q2m": (0.0, 0.045),               # kg/kg specific humidity
    "mslp": (850.0, 1060.0),           # hPa (low for severe cyclonic storms ~890 hPa)
    "cape": (0.0, 10000.0),            # J/kg
    "latitude": (6.0, 38.5),           # India latitude EPSG:4326
    "longitude": (68.0, 98.0),         # India longitude EPSG:4326
}


@dataclass
class QualityDimensionResult:
    """Audit result for a single quality dimension."""
    status: QualityStatus
    checks_passed: int
    checks_total: int
    metrics: Dict[str, Any] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DataQualityReport:
    """Comprehensive Data Quality Report conforming to Phase 7 Section 7."""
    report_id: str
    cycle_date: str
    pipeline_run_id: Optional[str]
    status: QualityStatus
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    completeness: QualityDimensionResult = field(default_factory=lambda: QualityDimensionResult("PASSED", 0, 0))
    validity: QualityDimensionResult = field(default_factory=lambda: QualityDimensionResult("PASSED", 0, 0))
    timeliness: QualityDimensionResult = field(default_factory=lambda: QualityDimensionResult("PASSED", 0, 0))
    spatial: QualityDimensionResult = field(default_factory=lambda: QualityDimensionResult("PASSED", 0, 0))
    total_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    action_required: bool = False
    alerts_triggered: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DataQualityMonitor:
    """
    Automated data quality auditing service for real-world and synthetic NWP ingestion.
    """

    def __init__(
        self,
        max_missing_ratio_threshold: float = 0.05,
        max_stale_hours_threshold: float = 48.0,
        alert_manager: Optional[AlertManager] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.max_missing_ratio_threshold = max_missing_ratio_threshold
        self.max_stale_hours_threshold = max_stale_hours_threshold
        self.alert_manager = alert_manager or AlertManager.get_default_manager()
        self.storage_dir = storage_dir or Path("logs/quality_reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def audit_ingested_cycle(
        self,
        df: pd.DataFrame,
        cycle_date: str,
        pipeline_run_id: Optional[str] = None,
        dataset_version: Optional[str] = None,
        source_timestamp: Optional[str] = None,
        expected_points: int = 15,
    ) -> DataQualityReport:
        """
        Executes four-dimensional audit: Completeness, Validity, Timeliness, Spatial.
        """
        import uuid
        report_id = f"dqr_{uuid.uuid4().hex[:10]}"
        alerts_triggered: List[str] = []

        # -------------------------------------------------------------
        # 1. Completeness Audit
        # -------------------------------------------------------------
        comp_failures: List[str] = []
        comp_warnings: List[str] = []
        comp_checks = 0
        comp_passed = 0

        # Check expected columns
        comp_checks += 1
        missing_vars = [v for v in EXPECTED_VARIABLES if v not in df.columns]
        if missing_vars:
            comp_failures.append(f"Missing expected variables: {missing_vars}")
        else:
            comp_passed += 1

        # Check record count vs expected points
        comp_checks += 1
        n_records = len(df)
        if n_records == 0:
            comp_failures.append("Dataset is completely empty (0 records).")
        elif n_records < expected_points:
            comp_warnings.append(f"Fewer records ({n_records}) than expected minimum ({expected_points}).")
            comp_passed += 1
        else:
            comp_passed += 1

        # Check null ratios
        comp_checks += 1
        missing_cell_ratio = float(df.isna().sum().sum()) / (n_records * max(1, len(df.columns))) if n_records > 0 else 1.0
        if missing_cell_ratio > self.max_missing_ratio_threshold:
            comp_failures.append(f"Missing cell ratio ({missing_cell_ratio:.2%}) exceeds limit ({self.max_missing_ratio_threshold:.2%}).")
        else:
            comp_passed += 1

        comp_status: QualityStatus = "FAILED" if comp_failures else ("WARNING" if comp_warnings else "PASSED")
        completeness = QualityDimensionResult(
            status=comp_status,
            checks_passed=comp_passed,
            checks_total=comp_checks,
            metrics={"missing_cell_ratio": round(missing_cell_ratio, 4), "record_count": n_records},
            failures=comp_failures,
            warnings=comp_warnings,
        )

        # -------------------------------------------------------------
        # 2. Validity Audit
        # -------------------------------------------------------------
        val_failures: List[str] = []
        val_warnings: List[str] = []
        val_checks = 0
        val_passed = 0

        # Infinite values check
        val_checks += 1
        inf_count = 0
        for col in df.select_dtypes(include=[np.number]).columns:
            inf_count += int(np.isinf(df[col]).sum())
        if inf_count > 0:
            val_failures.append(f"Found {inf_count} infinite values in numeric columns.")
        else:
            val_passed += 1

        # Physical plausibility and units
        invalid_rows_set = set()
        for col, (b_min, b_max) in PHYSICAL_BOUNDS.items():
            if col in df.columns:
                val_checks += 1
                col_vals = df[col].dropna()
                # Check for negative precipitation
                if col == "nwp_precip" and (col_vals < 0.0).any():
                    neg_count = int((col_vals < 0.0).sum())
                    val_failures.append(f"Negative precipitation values detected: {neg_count} records.")
                    invalid_rows_set.update(df[df[col] < 0.0].index.tolist())
                    continue

                # Plausible range
                out_of_bounds = (col_vals < b_min) | (col_vals > b_max)
                if out_of_bounds.any():
                    count = int(out_of_bounds.sum())
                    val_failures.append(f"Variable '{col}' has {count} values outside [{b_min}, {b_max}].")
                    invalid_rows_set.update(df[out_of_bounds].index.tolist())
                else:
                    val_passed += 1

        val_status: QualityStatus = "FAILED" if val_failures else ("WARNING" if val_warnings else "PASSED")
        validity = QualityDimensionResult(
            status=val_status,
            checks_passed=val_passed,
            checks_total=val_checks,
            metrics={"invalid_records": len(invalid_rows_set), "infinite_values": inf_count},
            failures=val_failures,
            warnings=val_warnings,
        )

        # -------------------------------------------------------------
        # 3. Timeliness Audit
        # -------------------------------------------------------------
        time_failures: List[str] = []
        time_warnings: List[str] = []
        time_checks = 1
        time_passed = 0

        now_utc = datetime.now(timezone.utc)
        if source_timestamp:
            try:
                src_dt = pd.to_datetime(source_timestamp)
                if src_dt.tzinfo is None:
                    src_dt = src_dt.tz_localize(timezone.utc)
                data_age_hours = float((now_utc - src_dt).total_seconds() / 3600.0)
                if data_age_hours > self.max_stale_hours_threshold:
                    time_failures.append(f"Data is stale: age {data_age_hours:.1f}h exceeds threshold {self.max_stale_hours_threshold}h.")
                elif data_age_hours > 24.0:
                    time_warnings.append(f"Data age {data_age_hours:.1f}h exceeds normal 24h operational cycle.")
                    time_passed += 1
                else:
                    time_passed += 1
            except Exception as e:
                time_warnings.append(f"Could not parse source timestamp: {e}")
                data_age_hours = 0.0
        else:
            data_age_hours = 0.0
            time_passed += 1

        time_status: QualityStatus = "FAILED" if time_failures else ("WARNING" if time_warnings else "PASSED")
        timeliness = QualityDimensionResult(
            status=time_status,
            checks_passed=time_passed,
            checks_total=time_checks,
            metrics={"data_age_hours": round(data_age_hours, 2), "source_timestamp": source_timestamp},
            failures=time_failures,
            warnings=time_warnings,
        )

        # -------------------------------------------------------------
        # 4. Spatial Audit
        # -------------------------------------------------------------
        spa_failures: List[str] = []
        spa_warnings: List[str] = []
        spa_checks = 2
        spa_passed = 0

        lat_col = "latitude" if "latitude" in df.columns else ("lat" if "lat" in df.columns else None)
        lon_col = "longitude" if "longitude" in df.columns else ("lon" if "lon" in df.columns else None)

        if lat_col and lon_col:
            out_lat = ((df[lat_col] < 6.0) | (df[lat_col] > 38.5)).sum()
            out_lon = ((df[lon_col] < 68.0) | (df[lon_col] > 98.0)).sum()
            if out_lat > 0 or out_lon > 0:
                spa_failures.append(f"Coordinates outside Indian Subcontinent bounds: {out_lat} lat, {out_lon} lon out.")
            else:
                spa_passed += 1
            spa_passed += 1
        else:
            spa_failures.append("Missing latitude/longitude spatial coordinates.")

        spa_status: QualityStatus = "FAILED" if spa_failures else ("WARNING" if spa_warnings else "PASSED")
        spatial = QualityDimensionResult(
            status=spa_status,
            checks_passed=spa_passed,
            checks_total=spa_checks,
            metrics={"has_coordinates": bool(lat_col and lon_col)},
            failures=spa_failures,
            warnings=spa_warnings,
        )

        # -------------------------------------------------------------
        # Overall Status & Alert Dispatch
        # -------------------------------------------------------------
        has_failure = any(s == "FAILED" for s in [comp_status, val_status, time_status, spa_status])
        has_warning = any(s == "WARNING" for s in [comp_status, val_status, time_status, spa_status])
        overall_status: QualityStatus = "FAILED" if has_failure else ("WARNING" if has_warning else "PASSED")

        invalid_count = len(invalid_rows_set)
        valid_count = max(0, n_records - invalid_count)

        # Trigger Alerts if required
        if comp_status == "FAILED":
            alt = self.alert_manager.record_alert(
                category="DATA_MISSING",
                message=f"Completeness audit failed for cycle {cycle_date}: {comp_failures}",
                severity="CRITICAL",
                pipeline_run_id=pipeline_run_id,
                dataset_version=dataset_version,
            )
            alerts_triggered.append(alt.alert_id)

        if val_status == "FAILED":
            alt = self.alert_manager.record_alert(
                category="DATA_INVALID",
                message=f"Validity audit failed for cycle {cycle_date}: {val_failures}",
                severity="CRITICAL",
                pipeline_run_id=pipeline_run_id,
                dataset_version=dataset_version,
            )
            alerts_triggered.append(alt.alert_id)

        if time_status == "FAILED":
            alt = self.alert_manager.record_alert(
                category="DATA_STALE",
                message=f"Data timeliness threshold breached for cycle {cycle_date}: {time_failures}",
                severity="WARNING",
                pipeline_run_id=pipeline_run_id,
                dataset_version=dataset_version,
            )
            alerts_triggered.append(alt.alert_id)

        report = DataQualityReport(
            report_id=report_id,
            cycle_date=cycle_date,
            pipeline_run_id=pipeline_run_id,
            status=overall_status,
            completeness=completeness,
            validity=validity,
            timeliness=timeliness,
            spatial=spatial,
            total_records=n_records,
            valid_records=valid_count,
            invalid_records=invalid_count,
            action_required=has_failure,
            alerts_triggered=alerts_triggered,
        )

        # Persist report
        self._persist_report(report)
        return report

    def _persist_report(self, report: DataQualityReport) -> None:
        report_file = self.storage_dir / f"{report.report_id}.json"
        try:
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to save quality report %s: %s", report.report_id, e)

    def get_latest_report(self) -> Optional[DataQualityReport]:
        reports = list(self.storage_dir.glob("*.json"))
        if not reports:
            return None
        latest = max(reports, key=lambda p: p.stat().st_mtime)
        try:
            with open(latest, "r", encoding="utf-8") as f:
                data = json.load(f)
                # Reconstruct DataQualityReport
                comp = QualityDimensionResult(**data.get("completeness", {}))
                val = QualityDimensionResult(**data.get("validity", {}))
                tim = QualityDimensionResult(**data.get("timeliness", {}))
                spa = QualityDimensionResult(**data.get("spatial", {}))
                d = dict(data)
                d["completeness"] = comp
                d["validity"] = val
                d["timeliness"] = tim
                d["spatial"] = spa
                return DataQualityReport(**d)
        except Exception as e:
            logger.warning("Could not read quality report: %s", e)
            return None
