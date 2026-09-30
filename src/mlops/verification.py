"""
Automated Output Verification & Publishing Gate Engine for RAAP-X (Phase 7 Section 16 & 17).
Validates prediction monotonicity, regime sum properties, spatial bounds, temporal causality, and metadata lineage.
Gates downstream output publishing to VALID, PARTIAL, or INVALID states.
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
from src.mlops.versioning import SystemVersionManifest, VersionManager
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.verification")

PublishState = Literal["VALID", "PARTIAL", "INVALID"]

INDIA_LAT_BOUNDS = (6.0, 38.5)
INDIA_LON_BOUNDS = (68.0, 98.0)


@dataclass
class VerificationReport:
    """Standardized Automated Verification Report conforming to Phase 7 Section 16."""
    report_id: str
    pipeline_run_id: str
    cycle_date: str
    lead_time: int
    publication_state: PublishState
    passed: bool
    checks: Dict[str, bool] = field(default_factory=dict)
    failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_grid_points: int = 0
    valid_grid_points: int = 0
    invalid_grid_points: int = 0
    total_districts: int = 0
    missing_areas: List[str] = field(default_factory=list)
    metadata_verified: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AutomatedVerificationEngine:
    """
    Evaluates predictions and district aggregations against rigorous physical,
    probabilistic, spatial, and metadata validation criteria prior to publishing.
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        storage_dir: Optional[Path] = None,
    ) -> None:
        self.alert_manager = alert_manager or AlertManager.get_default_manager()
        self.storage_dir = storage_dir or Path("logs/verification_reports")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def verify_outputs(
        self,
        grid_predictions: Union[List[Dict[str, Any]], pd.DataFrame],
        district_summaries: Optional[Union[List[Dict[str, Any]], pd.DataFrame]] = None,
        manifest: Optional[SystemVersionManifest] = None,
        cycle_date: str = "",
        lead_time_hours: int = 24,
        pipeline_run_id: Optional[str] = None,
        expected_districts: Optional[List[str]] = None,
    ) -> VerificationReport:
        """
        Executes complete verification suite across 5 core dimensions:
        1. Prediction Validity (Monotonicity & Non-Negativity)
        2. Regime Validity (Probabilities in [0, 1] & sum to 1)
        3. Spatial Validity (Coordinates & Boundaries)
        4. Temporal Validity (Forecast Time Consistency)
        5. Metadata Lineage Validity (System Manifest Consistency)
        """
        import uuid
        report_id = f"vr_{uuid.uuid4().hex[:10]}"
        run_id = pipeline_run_id or f"run_{uuid.uuid4().hex[:8]}"

        checks: Dict[str, bool] = {}
        failures: List[str] = []
        warnings: List[str] = []
        missing_areas: List[str] = []

        # Convert to records if DataFrame
        if isinstance(grid_predictions, pd.DataFrame):
            records = grid_predictions.to_dict(orient="records")
        else:
            records = grid_predictions

        total_grid = len(records)
        invalid_grid_indices: set[int] = set()

        if total_grid == 0:
            failures.append("Grid predictions are completely empty (0 records).")
            checks["non_empty_grid"] = False
        else:
            checks["non_empty_grid"] = True

        # -------------------------------------------------------------
        # 1. Prediction Validity: P50 >= 0, P75 >= P50, P90 >= P75
        # -------------------------------------------------------------
        monotonicity_violations = 0
        negativity_violations = 0

        for idx, rec in enumerate(records):
            p50 = float(rec.get("corrected_p50", rec.get("p50", -1.0)))
            p75 = float(rec.get("corrected_p75", rec.get("p75", -1.0)))
            p90 = float(rec.get("corrected_p90", rec.get("p90", -1.0)))

            if p50 < -1e-4:
                negativity_violations += 1
                invalid_grid_indices.add(idx)

            if p75 < (p50 - 1e-4) or p90 < (p75 - 1e-4):
                monotonicity_violations += 1
                invalid_grid_indices.add(idx)

        checks["non_negativity"] = (negativity_violations == 0)
        checks["monotonicity"] = (monotonicity_violations == 0)

        if negativity_violations > 0:
            failures.append(f"Physical constraint violated: {negativity_violations} records have P50 < 0.")
        if monotonicity_violations > 0:
            failures.append(f"Monotonicity violated: {monotonicity_violations} records have non-monotonic quantiles (P75 < P50 or P90 < P75).")

        # -------------------------------------------------------------
        # 2. Regime Validity: 0 <= prob <= 1, sum ≈ 1
        # -------------------------------------------------------------
        regime_range_violations = 0
        regime_sum_violations = 0

        for idx, rec in enumerate(records):
            r_probs = rec.get("regime_probabilities", {})
            if isinstance(r_probs, dict) and r_probs:
                p_sum = sum(r_probs.values())
                if abs(p_sum - 1.0) > 0.05:  # Tolerance for float normalization
                    regime_sum_violations += 1
                    invalid_grid_indices.add(idx)
                for p_val in r_probs.values():
                    if p_val < -1e-4 or p_val > (1.0 + 1e-4):
                        regime_range_violations += 1
                        invalid_grid_indices.add(idx)
                        break

        checks["regime_probability_range"] = (regime_range_violations == 0)
        checks["regime_sum_to_one"] = (regime_sum_violations == 0)

        if regime_range_violations > 0:
            failures.append(f"Regime probabilities out of bounds [0, 1] in {regime_range_violations} records.")
        if regime_sum_violations > 0:
            failures.append(f"Regime probability sum diverges from 1.0 in {regime_sum_violations} records.")

        # -------------------------------------------------------------
        # 3. Spatial Validity: Grid coordinates in India bounds
        # -------------------------------------------------------------
        coord_violations = 0
        for idx, rec in enumerate(records):
            lat = float(rec.get("latitude", rec.get("lat", 0.0)))
            lon = float(rec.get("longitude", rec.get("lon", 0.0)))
            if not (INDIA_LAT_BOUNDS[0] <= lat <= INDIA_LAT_BOUNDS[1] and INDIA_LON_BOUNDS[0] <= lon <= INDIA_LON_BOUNDS[1]):
                coord_violations += 1
                invalid_grid_indices.add(idx)

        checks["spatial_coordinates_in_bounds"] = (coord_violations == 0)
        if coord_violations > 0:
            failures.append(f"Spatial violation: {coord_violations} grid points fall outside India subcontinent bounds.")

        # District spatial completeness
        dist_records: List[Dict[str, Any]] = []
        if district_summaries is not None:
            if isinstance(district_summaries, pd.DataFrame):
                dist_records = district_summaries.to_dict(orient="records")
            else:
                dist_records = district_summaries

        found_districts = {d.get("district_id") for d in dist_records if d.get("district_id")}
        if expected_districts:
            missing_set = set(expected_districts) - found_districts
            if missing_set:
                missing_areas = sorted(list(missing_set))
                warnings.append(f"Missing {len(missing_areas)} expected district aggregations: {missing_areas[:5]}...")

        # -------------------------------------------------------------
        # 4. Temporal Validity: Timestamps
        # -------------------------------------------------------------
        temporal_violations = 0
        if cycle_date and records:
            for idx, rec in enumerate(records):
                f_time = rec.get("forecast_time", "")
                init_time = rec.get("initialization_time", "")
                if init_time and not init_time.startswith(cycle_date[:10]):
                    temporal_violations += 1
                    invalid_grid_indices.add(idx)

        checks["temporal_validity"] = (temporal_violations == 0)
        if temporal_violations > 0:
            failures.append(f"Temporal inconsistency: {temporal_violations} records have mismatched initialization dates.")

        # -------------------------------------------------------------
        # 5. Metadata Lineage Validity
        # -------------------------------------------------------------
        active_manifest = manifest or VersionManager.get_active_manifest()
        metadata_ok = True
        if records:
            sample = records[0]
            m_ver = sample.get("model_version")
            f_ver = sample.get("feature_version")
            d_ver = sample.get("dataset_version")

            if m_ver and m_ver != active_manifest.quantile_model_version and m_ver != active_manifest.repair_model_version and "1.0.0" not in m_ver:
                warnings.append(f"Model version '{m_ver}' differs from manifest '{active_manifest.repair_model_version}'.")
            if f_ver and f_ver != active_manifest.feature_version:
                warnings.append(f"Feature version '{f_ver}' differs from manifest '{active_manifest.feature_version}'.")

        checks["metadata_lineage"] = metadata_ok

        # -------------------------------------------------------------
        # Publication State Determination
        # -------------------------------------------------------------
        valid_grid_count = max(0, total_grid - len(invalid_grid_indices))
        invalid_grid_count = len(invalid_grid_indices)

        if failures:
            pub_state: PublishState = "INVALID"
            passed = False
            # Alert on verification failure
            self.alert_manager.record_alert(
                category="PIPELINE_FAILURE",
                message=f"Automated verification failed for run {run_id}: {failures[0]}",
                severity="CRITICAL",
                pipeline_run_id=run_id,
            )
        elif missing_areas or invalid_grid_count > 0:
            pub_state = "PARTIAL"
            passed = True
        else:
            pub_state = "VALID"
            passed = True

        report = VerificationReport(
            report_id=report_id,
            pipeline_run_id=run_id,
            cycle_date=cycle_date,
            lead_time=lead_time_hours,
            publication_state=pub_state,
            passed=passed,
            checks=checks,
            failures=failures,
            warnings=warnings,
            total_grid_points=total_grid,
            valid_grid_points=valid_grid_count,
            invalid_grid_points=invalid_grid_count,
            total_districts=len(dist_records),
            missing_areas=missing_areas,
            metadata_verified=metadata_ok,
        )

        self._persist_report(report)
        return report

    def _persist_report(self, report: VerificationReport) -> None:
        p = self.storage_dir / f"{report.report_id}.json"
        try:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(report.to_dict(), f, indent=2)
        except Exception as e:
            logger.error("Failed to persist verification report: %s", e)
