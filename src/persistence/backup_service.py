"""BKP-1 — Orchestration: run a backup, prune to 30 days, judge the result.

One run is: snapshot every ``*.db`` → one ``.tar.gz`` → upload → verify what is
actually in the store → prune older than the retention window → record the
outcome in a state file.

The order matters. Pruning happens **after** the upload has been confirmed by a
listing, never before, so a failed upload can never be the reason yesterday's
backup disappeared.

Judging the result
------------------
``verify_backups`` is the part that has to fail loudly. It answers three
questions against the store itself — not against our own memory of what we
think we uploaded:

1. Is there a backup at all?
2. Is the most recent one younger than ``BACKUP_MAX_AGE_HOURS`` (26h by
   default — a full day plus the slack of a redeploy)?
3. Is its size within ``BACKUP_SIZE_TOLERANCE`` of the median of the recent
   ones? A backup that suddenly weighs a tenth of yesterday's is the classic
   silent failure: the job "succeeded" and shipped an empty or truncated
   database.

The same function backs the daemon's ERROR log, the ``/health`` field, and the
external GitHub Actions verifier — one definition of "the backup is fine", so
the three can never disagree.
"""

from __future__ import annotations

import json
import logging
import os
import statistics
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.persistence.backup_storage import (
    BackupDestination,
    RemoteObject,
    destination_from_env,
)
from src.persistence.sqlite_backup import (
    ArchiveResult,
    BackupError,
    create_archive,
    discover_databases,
    free_space_bytes,
    restore_archive,
    verify_archive,
)

logger = logging.getLogger(__name__)

ENV_ENABLED = "BACKUP_ENABLED"
ENV_HOUR = "BACKUP_HOUR_UTC"
ENV_MINUTE = "BACKUP_MINUTE_UTC"
ENV_RETENTION_DAYS = "BACKUP_RETENTION_DAYS"
ENV_STAGING_DIR = "BACKUP_STAGING_DIR"
ENV_MAX_AGE_HOURS = "BACKUP_MAX_AGE_HOURS"
ENV_SIZE_TOLERANCE = "BACKUP_SIZE_TOLERANCE"
ENV_DATA_DIR = "DATA_DIR"

DEFAULT_RETENTION_DAYS = 30
DEFAULT_HOUR_UTC = 3
DEFAULT_MINUTE_UTC = 17          # off the hour: the top of the hour is crowded
DEFAULT_MAX_AGE_HOURS = 26.0
DEFAULT_SIZE_TOLERANCE = 0.40    # ±40% around the median of the recent backups
STATE_FILENAME = "backup_state.json"

#: Free space demanded before staging, as a multiple of the databases' total
#: size. The snapshots (1×) and the archive (~0.2×) live side by side; 1.6×
#: leaves room and refuses early rather than dying mid-copy on a full disk.
STAGING_HEADROOM = 1.6


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _env(name: str, default: str = "", env: Optional[dict] = None) -> str:
    return ((env if env is not None else os.environ).get(name) or default).strip()


def _env_int(name: str, default: int, env: Optional[dict] = None) -> int:
    try:
        return int(_env(name, str(default), env))
    except ValueError:
        logger.warning("backup: %s is not an integer — using %d", name, default)
        return default


def _env_float(name: str, default: float, env: Optional[dict] = None) -> float:
    try:
        return float(_env(name, str(default), env))
    except ValueError:
        logger.warning("backup: %s is not a number — using %s", name, default)
        return default


def is_backup_enabled(env: Optional[dict] = None) -> bool:
    """``BACKUP_ENABLED`` — off by default (a dev box must not ship data)."""
    return _env(ENV_ENABLED, "", env).lower() in ("1", "true", "yes", "on")


def data_dir(env: Optional[dict] = None) -> Path:
    return Path(_env(ENV_DATA_DIR, "./data", env))


def retention_days(env: Optional[dict] = None) -> int:
    return max(1, _env_int(ENV_RETENTION_DAYS, DEFAULT_RETENTION_DAYS, env))


def max_age_hours(env: Optional[dict] = None) -> float:
    return _env_float(ENV_MAX_AGE_HOURS, DEFAULT_MAX_AGE_HOURS, env)


def size_tolerance(env: Optional[dict] = None) -> float:
    return _env_float(ENV_SIZE_TOLERANCE, DEFAULT_SIZE_TOLERANCE, env)


def schedule_utc(env: Optional[dict] = None) -> tuple[int, int]:
    hour = _env_int(ENV_HOUR, DEFAULT_HOUR_UTC, env) % 24
    minute = _env_int(ENV_MINUTE, DEFAULT_MINUTE_UTC, env) % 60
    return hour, minute


# --------------------------------------------------------------------------- #
# State file
# --------------------------------------------------------------------------- #

def state_path(env: Optional[dict] = None) -> Path:
    return data_dir(env) / STATE_FILENAME


def read_state(env: Optional[dict] = None) -> dict:
    """Last known outcome. Never raises — a missing or corrupt file reads {}."""
    try:
        p = state_path(env)
        if not p.is_file():
            return {}
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("backup: unreadable state file — treating as absent", exc_info=True)
        return {}


def write_state(patch: dict, env: Optional[dict] = None) -> None:
    """Merge ``patch`` into the state file. Never raises."""
    try:
        p = state_path(env)
        p.parent.mkdir(parents=True, exist_ok=True)
        current = read_state(env)
        current.update(patch)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(p)
    except Exception:
        logger.warning("backup: could not write state file", exc_info=True)


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #

@dataclass
class VerifyReport:
    """The verdict on what is actually in the store."""

    healthy: bool
    problems: list[str] = field(default_factory=list)
    count: int = 0
    latest_key: Optional[str] = None
    latest_size: Optional[int] = None
    latest_age_hours: Optional[float] = None
    median_size: Optional[int] = None
    destination: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def verify_backups(
    destination: BackupDestination,
    *,
    now: Optional[datetime] = None,
    max_age_h: float = DEFAULT_MAX_AGE_HOURS,
    tolerance: float = DEFAULT_SIZE_TOLERANCE,
    min_history: int = 3,
) -> VerifyReport:
    """Judge the store's contents. Never raises: a failure IS the verdict."""
    at = now or datetime.now(timezone.utc)
    report = VerifyReport(healthy=True)
    try:
        report.destination = destination.describe()
        objects = destination.list_backups()
    except Exception as exc:
        report.healthy = False
        report.problems.append(f"cannot list the backup store: {exc}")
        return report

    dated = [(o.stamp, o) for o in objects if o.stamp is not None]
    report.count = len(dated)
    if not dated:
        report.healthy = False
        report.problems.append("no backup found in the store")
        return report

    dated.sort(key=lambda pair: pair[0])
    latest_stamp, latest = dated[-1]
    age_h = (at - latest_stamp).total_seconds() / 3600.0
    report.latest_key = latest.key
    report.latest_size = latest.size
    report.latest_age_hours = round(age_h, 2)

    if age_h > max_age_h:
        report.healthy = False
        report.problems.append(
            f"the most recent backup ({latest.name}) is {age_h:.1f}h old, "
            f"over the {max_age_h:.0f}h limit"
        )

    if latest.size <= 0:
        report.healthy = False
        report.problems.append(f"{latest.name} is empty (0 byte)")

    # Size anomaly — only once there is enough history for a median to mean
    # something. Compared against the PREVIOUS backups, not including today's,
    # so one bad run cannot drag its own reference along with it.
    previous = [o.size for _, o in dated[:-1]][-7:]
    if len(previous) >= min_history:
        median = int(statistics.median(previous))
        report.median_size = median
        if median > 0:
            drift = abs(latest.size - median) / median
            if drift > tolerance:
                report.healthy = False
                report.problems.append(
                    f"{latest.name} weighs {latest.size} bytes, {drift * 100:.0f}% away "
                    f"from the median of the last {len(previous)} ({median} bytes) — "
                    f"over the {tolerance * 100:.0f}% tolerance"
                )

    return report


# --------------------------------------------------------------------------- #
# Retention
# --------------------------------------------------------------------------- #

def prune_old_backups(
    destination: BackupDestination,
    *,
    days: int = DEFAULT_RETENTION_DAYS,
    now: Optional[datetime] = None,
    keep_minimum: int = 2,
) -> list[str]:
    """Delete backups older than ``days``. Returns the keys deleted.

    ``keep_minimum`` is a floor, not a nicety: if the clock is wrong, or the
    window is misconfigured to something absurd, retention must never be able to
    empty the store. The two most recent backups always survive.
    """
    at = now or datetime.now(timezone.utc)
    cutoff = at - timedelta(days=days)
    dated = [(o.stamp, o) for o in destination.list_backups() if o.stamp is not None]
    dated.sort(key=lambda pair: pair[0])
    protected = {o.key for _, o in dated[-keep_minimum:]} if keep_minimum else set()

    deleted: list[str] = []
    for stamp, obj in dated:
        if stamp >= cutoff or obj.key in protected:
            continue
        try:
            destination.delete(obj.key)
            deleted.append(obj.key)
            logger.info("backup: pruned %s (older than %d days)", obj.name, days)
        except Exception:
            logger.warning("backup: could not prune %s", obj.key, exc_info=True)
    return deleted


# --------------------------------------------------------------------------- #
# One full run
# --------------------------------------------------------------------------- #

@dataclass
class BackupRunResult:
    ok: bool
    archive: Optional[str] = None
    size_bytes: int = 0
    source_bytes: int = 0
    databases: list[str] = field(default_factory=list)
    duration_s: float = 0.0
    pruned: list[str] = field(default_factory=list)
    verify: Optional[dict] = None
    error: Optional[str] = None

    def as_dict(self) -> dict:
        return asdict(self)


def run_backup(
    *,
    destination: Optional[BackupDestination] = None,
    env: Optional[dict] = None,
    now: Optional[datetime] = None,
    prune: bool = True,
) -> BackupRunResult:
    """Snapshot → upload → verify → prune. Never raises; returns the outcome.

    A failure is logged at ERROR with the reason, recorded in the state file, and
    returned as ``ok=False``. Callers (daemon, CLI) decide what to do with it.
    """
    at = now or datetime.now(timezone.utc)
    started = at
    dest = destination
    result = BackupRunResult(ok=False)
    write_state({"last_attempt_utc": at.isoformat()}, env)

    try:
        if dest is None:
            dest = destination_from_env(env)

        src_dir = data_dir(env)
        databases = discover_databases(src_dir)
        result.databases = [p.name for p in databases]
        total = sum(p.stat().st_size for p in databases)

        staging = _env(ENV_STAGING_DIR, "", env) or tempfile.gettempdir()
        needed = int(total * STAGING_HEADROOM)
        free = free_space_bytes(staging)
        if free < needed:
            raise BackupError(
                f"not enough room in {staging}: {free} bytes free, {needed} needed "
                f"({total} bytes of databases × {STAGING_HEADROOM}). "
                f"Point {ENV_STAGING_DIR} at a roomier volume."
            )

        with tempfile.TemporaryDirectory(prefix="mia-bkp-out-", dir=staging) as out:
            archive: ArchiveResult = create_archive(
                src_dir, out, databases=databases, now=at, staging_dir=staging
            )
            # Prove the archive before it leaves the machine: a corrupt upload is
            # indistinguishable from a good one once it is in the bucket.
            verify_archive(archive.path)
            remote = dest.upload(archive.path, archive.name)

            result.archive = remote.key
            result.size_bytes = archive.size_bytes
            result.source_bytes = archive.source_bytes
            result.duration_s = archive.duration_s

        # Confirm against the store, not against our own return value.
        listing = {o.name for o in dest.list_backups()}
        if archive.name not in listing:
            raise BackupError(
                f"{archive.name} was uploaded but does not appear in the store listing"
            )

        if prune:
            result.pruned = prune_old_backups(dest, days=retention_days(env), now=at)

        report = verify_backups(
            dest, now=at, max_age_h=max_age_hours(env), tolerance=size_tolerance(env)
        )
        result.verify = report.as_dict()
        result.ok = True

        write_state(
            {
                "last_success_utc": at.isoformat(),
                "last_archive": result.archive,
                "last_size_bytes": result.size_bytes,
                "last_source_bytes": result.source_bytes,
                "last_databases": result.databases,
                "last_duration_s": result.duration_s,
                "last_error": None,
                "destination": dest.describe(),
            },
            env,
        )
        logger.info(
            "backup: run OK — %s (%d bytes, %d databases, %.1fs), %d pruned",
            result.archive,
            result.size_bytes,
            len(result.databases),
            result.duration_s,
            len(result.pruned),
        )
        if not report.healthy:
            # The upload worked and the store still looks wrong — say so loudly.
            logger.error(
                "backup: the store is NOT healthy after a successful run: %s",
                "; ".join(report.problems),
            )
    except Exception as exc:
        result.error = f"{type(exc).__name__}: {exc}"
        result.duration_s = round((datetime.now(timezone.utc) - started).total_seconds(), 2)
        write_state({"last_error": result.error, "last_error_utc": at.isoformat()}, env)
        # ERROR level, with the reason in the message: this is the line that has
        # to be greppable in Render's JSON logs.
        logger.error("backup: run FAILED — %s", result.error, exc_info=True)

    return result


def restore_latest(
    target_dir: str | Path,
    *,
    destination: Optional[BackupDestination] = None,
    env: Optional[dict] = None,
    key: Optional[str] = None,
    force: bool = False,
) -> dict:
    """Download a backup (the most recent by default) and restore it.

    Every restored database is checked against the manifest's SHA-256 and then
    with ``PRAGMA integrity_check``. Raises ``BackupError`` on any doubt —
    a restore is the one place where a silent partial success is unacceptable.
    """
    dest = destination if destination is not None else destination_from_env(env)
    objects = [o for o in dest.list_backups() if o.stamp is not None]
    if not objects:
        raise BackupError(f"no backup to restore in {dest.describe()}")
    if key:
        chosen = next((o for o in objects if o.key == key or o.name == key), None)
        if chosen is None:
            raise BackupError(f"no backup named {key} in {dest.describe()}")
    else:
        chosen = max(objects, key=lambda o: o.stamp)  # type: ignore[arg-type]

    staging = _env(ENV_STAGING_DIR, "", env) or tempfile.gettempdir()
    with tempfile.TemporaryDirectory(prefix="mia-restore-", dir=staging) as tmp:
        local = dest.download(chosen.key, Path(tmp) / chosen.name)
        verify_archive(local)
        report = restore_archive(local, target_dir, force=force)
    report["source_key"] = chosen.key
    logger.info("backup: restored %s into %s", chosen.key, target_dir)
    return report


def health_snapshot(env: Optional[dict] = None) -> Optional[dict]:
    """Compact view for ``/health``. ``None`` when backups are switched off.

    Reads the local state file only — no network call on a health check.
    """
    if not is_backup_enabled(env):
        return None
    state = read_state(env)
    last = state.get("last_success_utc")
    age_h: Optional[float] = None
    if last:
        try:
            then = datetime.fromisoformat(last)
            if then.tzinfo is None:
                then = then.replace(tzinfo=timezone.utc)
            age_h = round((datetime.now(timezone.utc) - then).total_seconds() / 3600.0, 2)
        except ValueError:
            age_h = None
    return {
        "enabled": True,
        "last_success_utc": last,
        "last_backup_age_hours": age_h,
        "last_archive": state.get("last_archive"),
        "last_size_bytes": state.get("last_size_bytes"),
        "last_error": state.get("last_error"),
        # The single number an operator should look at: False means the last
        # backup is missing or stale, whatever the reason.
        "fresh": bool(age_h is not None and age_h <= max_age_hours(env)),
    }


__all__ = [
    "BackupRunResult",
    "DEFAULT_MAX_AGE_HOURS",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_SIZE_TOLERANCE",
    "VerifyReport",
    "data_dir",
    "health_snapshot",
    "is_backup_enabled",
    "max_age_hours",
    "prune_old_backups",
    "read_state",
    "restore_latest",
    "retention_days",
    "run_backup",
    "schedule_utc",
    "size_tolerance",
    "state_path",
    "verify_backups",
    "write_state",
]
