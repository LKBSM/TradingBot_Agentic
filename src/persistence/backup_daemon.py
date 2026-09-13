"""BKP-1 — The daily trigger, inside the backend process.

Why in-process, and not a Render Cron Job
-----------------------------------------
Render's persistent disk is the constraint: *"You can't add a disk to a cron job
service"*, and *"A persistent disk is accessible by only a single service
instance"*. A cron container would run beside ``mia-backend`` and never see
``/app/data``. So the only process that can read the databases is the backend
itself, and the scheduler has to live there.

That is not a new mechanism: ``src/intelligence/scheduler.py`` already runs an
APScheduler ``BackgroundScheduler`` in this process, and ``APScheduler`` is
already a dependency. This adds one job to a second, tiny scheduler owned by
this module — separate so that a backup misfire can never disturb the market
reading cadence, and so it can be started and stopped on its own.

Boot catch-up
-------------
A deploy landing on the scheduled minute would otherwise cost a whole day. So on
start, if the last successful backup is older than a day, one runs immediately
(in the background, never blocking startup).
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Optional

from src.persistence.backup_service import (
    is_backup_enabled,
    read_state,
    run_backup,
    schedule_utc,
)
from src.persistence.backup_storage import destination_configured

logger = logging.getLogger(__name__)

_CATCHUP_AFTER_HOURS = 24.0


class BackupDaemon:
    """Owns the daily job. Start/stop are idempotent and never raise."""

    def __init__(self, env: Optional[dict] = None) -> None:
        self._env = env
        self._scheduler: Any = None
        self._lock = threading.Lock()
        self._started = False

    # -- lifecycle -------------------------------------------------------- #

    def start(self) -> bool:
        """Start the daily job. Returns True when it is actually scheduled."""
        with self._lock:
            if self._started:
                return True
            if not is_backup_enabled(self._env):
                logger.info("backup daemon: disabled (BACKUP_ENABLED is not set)")
                return False
            if not destination_configured(self._env):
                # Loud on purpose: "backups are on" with nowhere to write is the
                # worst of both worlds — it looks configured and stores nothing.
                logger.error(
                    "backup daemon: BACKUP_ENABLED is set but no destination is "
                    "configured (BACKUP_S3_BUCKET / BACKUP_S3_ENDPOINT / "
                    "BACKUP_S3_ACCESS_KEY_ID / BACKUP_S3_SECRET_ACCESS_KEY). "
                    "NOTHING WILL BE BACKED UP."
                )
                return False
            try:
                from apscheduler.schedulers.background import BackgroundScheduler
                from apscheduler.triggers.cron import CronTrigger
            except ImportError:  # pragma: no cover - APScheduler is in requirements.txt
                logger.exception("backup daemon: APScheduler unavailable")
                return False

            hour, minute = schedule_utc(self._env)
            try:
                self._scheduler = BackgroundScheduler(timezone="UTC")
                self._scheduler.add_job(
                    self._job,
                    CronTrigger(hour=hour, minute=minute, timezone="UTC"),
                    id="daily-sqlite-backup",
                    name="daily SQLite backup",
                    max_instances=1,
                    coalesce=True,
                    # A backup that missed its slot by an hour is still worth
                    # running; two hours late and the next daily one is closer.
                    misfire_grace_time=3600,
                )
                self._scheduler.start()
                self._started = True
            except Exception:
                logger.exception("backup daemon: failed to start")
                return False

        logger.info("backup daemon: daily backup scheduled at %02d:%02d UTC", hour, minute)
        self._maybe_catch_up()
        return True

    def stop(self) -> None:
        with self._lock:
            sched, self._scheduler, self._started = self._scheduler, None, False
        if sched is None:
            return
        try:
            if sched.running:
                sched.shutdown(wait=False)
        except Exception:
            logger.warning("backup daemon: shutdown raised", exc_info=True)

    @property
    def running(self) -> bool:
        sched = self._scheduler
        try:
            return bool(sched is not None and sched.running)
        except Exception:  # pragma: no cover - defensive
            return False

    # -- work ------------------------------------------------------------- #

    def _job(self) -> None:
        """The scheduled call. ``run_backup`` never raises; this only logs."""
        result = run_backup(env=self._env)
        if not result.ok:
            logger.error("backup daemon: scheduled backup failed — %s", result.error)

    def _maybe_catch_up(self) -> None:
        """Run now if the last success is over a day old (never blocks boot)."""
        try:
            state = read_state(self._env)
            last = state.get("last_success_utc")
            due = True
            if last:
                then = datetime.fromisoformat(last)
                if then.tzinfo is None:
                    then = then.replace(tzinfo=timezone.utc)
                age_h = (datetime.now(timezone.utc) - then).total_seconds() / 3600.0
                due = age_h >= _CATCHUP_AFTER_HOURS
                if due:
                    logger.warning(
                        "backup daemon: last backup is %.1fh old — catching up now", age_h
                    )
            else:
                logger.info("backup daemon: no backup on record — running one now")
            if due:
                threading.Thread(
                    target=self._job, name="backup-catchup", daemon=True
                ).start()
        except Exception:
            logger.warning("backup daemon: catch-up check failed", exc_info=True)


_singleton: Optional[BackupDaemon] = None
_singleton_lock = threading.Lock()


def start_backup_daemon(env: Optional[dict] = None) -> Optional[BackupDaemon]:
    """Start the process-wide daemon once. ``None`` when it did not start."""
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = BackupDaemon(env)
        daemon = _singleton
    return daemon if daemon.start() else None


def stop_backup_daemon() -> None:
    """Stop the process-wide daemon, if any. Used by the shutdown coordinator."""
    global _singleton
    with _singleton_lock:
        daemon, _singleton = _singleton, None
    if daemon is not None:
        daemon.stop()


__all__ = ["BackupDaemon", "start_backup_daemon", "stop_backup_daemon"]
