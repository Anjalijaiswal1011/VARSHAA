"""
Operational Pipeline CLI & Automated Replay Utility for RAAP-X (Phase 7).
Enables operational runs, historical replays, health diagnostics, drift audits, and safe rollback from the command line.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.alerts import AlertManager
from src.mlops.drift import DriftMonitor
from src.mlops.health import HealthProbeService
from src.mlops.pipeline import OperationalPipeline
from src.mlops.registry import ModelRegistry
from src.mlops.scheduler import OperationalScheduler, ScheduleConfig
from src.utils.logging import get_logger

logger = get_logger("rain_repair.scripts.operational_pipeline")


def run_single_cycle(
    cycle_date: str,
    lead_time: int = 24,
    source: str = "IMD_GFS_OPERATIONAL",
    force_rerun: bool = False,
) -> None:
    pipeline = OperationalPipeline.get_default_pipeline()
    print(f"=== Starting Operational Pipeline Run: Cycle {cycle_date}, Lead Time {lead_time}h ===")
    job = pipeline.execute_cycle(
        cycle_date=cycle_date,
        lead_time_hours=lead_time,
        source=source,
        force_rerun=force_rerun,
    )

    print("\n--- Pipeline Execution Summary ---")
    print(f"Job ID:             {job.job_id}")
    print(f"Status:             {job.status}")
    print(f"Idempotency Key:    {job.idempotency_key}")
    print(f"Reused Existing:    {job.reused_existing}")
    print(f"Records Processed:  {job.records_processed}")
    print(f"Records Failed:     {job.records_failed}")
    print(f"Started At:         {job.started_at}")
    print(f"Completed At:       {job.completed_at}")
    if job.error_summary:
        print(f"Error Summary:      {job.error_summary}")

    print("\n--- Stage States ---")
    for st_name, st in job.stage_states.items():
        dur_val = st.get("duration_ms", 0.0) if isinstance(st, dict) else st.duration_ms
        st_status = st.get("status", "UNKNOWN") if isinstance(st, dict) else st.status
        dur = f"{dur_val:.1f}ms" if dur_val > 0 else "-"
        print(f"  [{st_status:7s}] {st_name:22s} ({dur})")


    if job.validation_report:
        pub = job.validation_report.get("publication_state", "UNKNOWN")
        print(f"\nPublication Gating State: {pub}")


def run_historical_replay(
    start_date: str,
    end_date: str,
    lead_time: int = 24,
    source: str = "IMD_GFS_HISTORICAL_REPLAY",
) -> None:
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    pipeline = OperationalPipeline.get_default_pipeline()

    print(f"=== Commencing Historical Operational Replay ({len(dates)} cycles from {start_date} to {end_date}) ===")
    total_processed = 0
    total_failed = 0
    success_runs = 0
    start_t = datetime.now(timezone.utc)

    for d in dates:
        c_date = d.strftime("%Y-%m-%d")
        print(f"\n>> Processing cycle {c_date} lead {lead_time}h...")
        job = pipeline.execute_cycle(
            cycle_date=c_date,
            lead_time_hours=lead_time,
            source=source,
            force_rerun=True,
        )
        total_processed += job.records_processed
        total_failed += job.records_failed
        if job.status in ["SUCCESS", "PARTIAL_SUCCESS"]:
            success_runs += 1
        print(f"   Status: {job.status} (Processed {job.records_processed} grid points, {len(job.alerts_triggered)} alerts)")

    elapsed = (datetime.now(timezone.utc) - start_t).total_seconds()
    print("\n================ Replay Complete ================")
    print(f"Total Cycles:        {len(dates)}")
    print(f"Successful Cycles:   {success_runs}/{len(dates)}")
    print(f"Total Records:       {total_processed}")
    print(f"Total Failed:        {total_failed}")
    print(f"Total Elapsed Time:  {elapsed:.2f}s (Avg {elapsed/max(1, len(dates)):.2f}s per cycle)")


def run_health_check() -> None:
    probe = HealthProbeService()
    report = probe.run_comprehensive_health_check()

    print("\n================ RAAP-X 9-Subsystem Health Probe ================")
    print(f"Overall Status: {report.overall_status}")
    print(f"Timestamp:      {report.timestamp}")
    print(f"Passed Checks:  {report.checks_passed}/{report.checks_total}")
    print("-----------------------------------------------------------------")
    for name, c in report.components.items():
        icon = "[OK]" if c.status == "OK" else ("[WARN]" if c.status == "WARNING" else "[FAIL]")
        print(f"{icon:7s} {name:20s} ({c.latency_ms:6.1f}ms): {c.message}")


def run_drift_audit() -> None:
    monitor = DriftMonitor.get_default_monitor()
    report = monitor.generate_drift_report()

    print("\n================ Data & Regime Drift Monitoring ================")
    print(f"Overall Monitoring State: {report.overall_state}")
    print(f"Action Required:          {report.action_required}")
    print(f"Recommended Action:       {report.recommended_action}")
    print("\nFeature Drift Summary (PSI & KS-test):")
    for feat, d in report.feature_drift_summary.items():
        print(f"  - {feat:25s}: PSI={d.get('psi', 0.0):.4f}, KS p-val={d.get('ks_pvalue', 1.0):.4f} [{d.get('status')}]")


def run_safe_rollback() -> None:
    reg = ModelRegistry.get_default_registry()
    print("\n=== Executing Operational Model Rollback ===")
    ok, msg = reg.rollback_to_previous_production(
        reason="Manual operator rollback from CLI",
        triggered_by="operator",
    )
    print(f"Result: {msg}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAAP-X Operational MLOps Pipeline Orchestrator CLI")
    parser.add_argument("--cycle-date", type=str, default=None, help="Forecast cycle date (YYYY-MM-DD)")
    parser.add_argument("--lead-time", type=int, default=24, help="Lead time in hours (24, 48, 72, 96, 120)")
    parser.add_argument("--force-rerun", action="store_true", help="Bypass idempotency cache and rerun cycle")
    parser.add_argument("--replay-start", type=str, default=None, help="Start date for historical replay")
    parser.add_argument("--replay-end", type=str, default=None, help="End date for historical replay")
    parser.add_argument("--check-health", action="store_true", help="Execute 9-subsystem health diagnostics")
    parser.add_argument("--check-drift", action="store_true", help="Execute statistical data and regime drift audit")
    parser.add_argument("--rollback", action="store_true", help="Safely roll back to previous validated model")

    args = parser.parse_args()

    if args.check_health:
        run_health_check()
    elif args.check_drift:
        run_drift_audit()
    elif args.rollback:
        run_safe_rollback()
    elif args.replay_start and args.replay_end:
        run_historical_replay(args.replay_start, args.replay_end, args.lead_time)
    else:
        c_date = args.cycle_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        run_single_cycle(c_date, args.lead_time, force_rerun=args.force_rerun)


if __name__ == "__main__":
    main()
