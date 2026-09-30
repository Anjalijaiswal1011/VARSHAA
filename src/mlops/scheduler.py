"""
Configurable Operational Pipeline Scheduler for RAAP-X (Phase 7 Section 5).
Coordinates scheduled automated ingestion, quality checking, ML inference, district aggregation,
and verification according to operational NWP cycle availability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import threading
import time
from typing import Any, Dict, List, Optional

from src.mlops.pipeline import OperationalPipeline, PipelineJobRecord
from src.utils.logging import get_logger

logger = get_logger("rain_repair.mlops.scheduler")


@dataclass
class ScheduleConfig:
    """Configurable scheduler settings conforming to Phase 7 Section 5."""
    enabled: bool = False
    interval_minutes: int = 360  # Default 6-hour cycle (00Z, 06Z, 12Z, 18Z GFS/NCUM cycles)
    timezone: str = "UTC"
    lead_times_hours: List[int] = field(default_factory=lambda: [24, 48, 72, 96, 120])
    auto_verify_on_obs: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SchedulerStatus:
    is_running: bool
    enabled: bool
    interval_minutes: int
    timezone: str
    last_run_utc: Optional[str]
    next_run_utc: Optional[str]
    total_runs_completed: int
    last_job_id: Optional[str]
    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OperationalScheduler:
    """
    Lightweight, Python-native operational background scheduler.
    Avoids heavyweight external orchestrators while delivering robust, configurable,
    and observable job execution.
    """

    _default_instance: Optional[OperationalScheduler] = None

    def __init__(
        self,
        config: Optional[ScheduleConfig] = None,
        pipeline: Optional[OperationalPipeline] = None,
    ) -> None:
        self.config = config or ScheduleConfig()
        self.pipeline = pipeline or OperationalPipeline.get_default_pipeline()
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        self.last_run_utc: Optional[str] = None
        self.next_run_utc: Optional[str] = None
        self.total_runs: int = 0
        self.last_job_id: Optional[str] = None
        self.last_error: Optional[str] = None

    @classmethod
    def get_default_scheduler(cls) -> OperationalScheduler:
        if cls._default_instance is None:
            cls._default_instance = OperationalScheduler()
        return cls._default_instance

    def get_status(self) -> SchedulerStatus:
        is_alive = self._thread is not None and self._thread.is_alive()
        return SchedulerStatus(
            is_running=is_alive,
            enabled=self.config.enabled,
            interval_minutes=self.config.interval_minutes,
            timezone=self.config.timezone,
            last_run_utc=self.last_run_utc,
            next_run_utc=self.next_run_utc,
            total_runs_completed=self.total_runs,
            last_job_id=self.last_job_id,
            last_error=self.last_error,
        )

    def trigger_cycle_now(
        self,
        cycle_date: Optional[str] = None,
        lead_time_hours: int = 24,
        force_rerun: bool = False,
    ) -> PipelineJobRecord:
        """Manually triggers an immediate operational cycle."""
        c_date = cycle_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Executing immediate operational cycle for %s lead %dh...", c_date, lead_time_hours)
        try:
            job = self.pipeline.execute_cycle(
                cycle_date=c_date,
                lead_time_hours=lead_time_hours,
                force_rerun=force_rerun,
            )
            self.last_run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            self.last_job_id = job.job_id
            self.total_runs += 1
            self.last_error = None
            return job
        except Exception as e:
            self.last_error = str(e)
            logger.error("Error executing triggered operational cycle: %s", e)
            raise

    def start(self) -> None:
        """Starts the scheduler thread if enabled."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Scheduler already running.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="OperationalSchedulerThread")
        self._thread.start()
        logger.info("OperationalScheduler background thread started (Interval: %d min).", self.config.interval_minutes)

    def stop(self) -> None:
        """Signals background thread to stop."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("OperationalScheduler stopped.")

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            if self.config.enabled:
                now_utc = datetime.now(timezone.utc)
                self.next_run_utc = (now_utc + timedelta(minutes=self.config.interval_minutes)).strftime("%Y-%m-%dT%H:%M:%SZ")

                cycle_date = now_utc.strftime("%Y-%m-%d")
                for lt in self.config.lead_times_hours:
                    if self._stop_event.is_set():
                        break
                    try:
                        self.trigger_cycle_now(cycle_date=cycle_date, lead_time_hours=lt)
                    except Exception as e:
                        logger.error("Failed scheduled run for %s lt %d: %s", cycle_date, lt, e)

            # Sleep in small increments to be responsive to stop_event
            sleep_seconds = self.config.interval_minutes * 60
            elapsed = 0
            while elapsed < sleep_seconds and not self._stop_event.is_set():
                time.sleep(1.0)
                elapsed += 1
