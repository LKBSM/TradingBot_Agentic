"""SQLite-backed cache of scheduled economic-calendar events (NW-1).

Distinct from ``NewsCacheStore`` (which keeps only medium/high events to feed
the MarketReading "events" block): the calendar page needs the FULL impact
range (its "Faible" filter must be honest), the market-attachment, revisions,
and the official-source-shaped provenance fields (source, series code, organism,
unit, publisher timezone, license). Rather than widen the shared news cache
(and risk a MarketReading regression), the calendar owns its own store.

The persisted row is neutral (no provider-specific column). Revision handling
(mission NW-1 §2A): on upsert, if a row already exists for an ``event_id`` and
any published value CHANGED, the row is flagged ``revised=1`` and the prior
``actual`` is kept in ``previous_before_revision`` — a revised value is surfaced
as revised, never overwritten silently.

Env-aware path (``CALENDAR_CACHE_DB_PATH`` → ``./data/calendar_cache.db``).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_iso(ts: Optional[datetime] = None) -> str:
    ts = ts if ts is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_iso(value: str) -> datetime:
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


@dataclass(frozen=True)
class CalendarCacheEvent:
    """A calendar event as persisted — neutral, source-agnostic.

    ``impact`` may be "high" / "medium" / "low" (unlike the news cache).
    ``markets`` is the attached-market list (never empty once stored).
    """

    event_id: str
    source: str
    event: str
    currency: str
    impact: str
    scheduled_at: datetime
    markets: List[str] = field(default_factory=list)
    series_code: Optional[str] = None
    license_label: Optional[str] = None
    organism: Optional[str] = None
    source_timezone: Optional[str] = None
    value_unit: Optional[str] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    revised: bool = False
    previous_before_revision: Optional[float] = None


class CalendarCacheStore:
    SCHEMA_VERSION = 1
    DEFAULT_DB_PATH = "./data/calendar_cache.db"
    DB_PATH_ENV_VAR = "CALENDAR_CACHE_DB_PATH"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("CalendarCacheStore initialised at %s", self._db_path)

    @classmethod
    def _resolve_db_path(cls, db_path: Optional[str]) -> Path:
        if db_path:
            return Path(db_path)
        env_val = os.environ.get(cls.DB_PATH_ENV_VAR)
        if env_val:
            return Path(env_val)
        return Path(cls.DEFAULT_DB_PATH)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_database(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS schema_version "
                    "(version INTEGER PRIMARY KEY)"
                )
                cur = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cur.fetchone()
                current = row["version"] if row else 0
                if current < self.SCHEMA_VERSION:
                    self._migrate(conn, current)
                    conn.execute("DELETE FROM schema_version")
                    conn.execute(
                        "INSERT INTO schema_version (version) VALUES (?)",
                        (self.SCHEMA_VERSION,),
                    )
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_v: int) -> None:
        if from_v < 1:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS calendar_cache (
                    event_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    event TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    markets TEXT NOT NULL,
                    series_code TEXT,
                    license_label TEXT,
                    organism TEXT,
                    source_timezone TEXT,
                    value_unit TEXT,
                    actual REAL,
                    forecast REAL,
                    previous REAL,
                    revised INTEGER NOT NULL DEFAULT 0,
                    previous_before_revision REAL,
                    fetched_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_calendar_cache_window
                    ON calendar_cache(scheduled_at);
                """
            )

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def upsert_events(
        self,
        events: List[CalendarCacheEvent],
        fetched_at: Optional[datetime] = None,
    ) -> int:
        """Upsert events (dedup by ``event_id``). Flags ``revised`` and captures
        ``previous_before_revision`` when a published value changed vs the stored
        row. Returns rows affected."""
        if not events:
            return 0
        fetched_iso = _utc_iso(fetched_at)
        with self._lock:
            conn = self._get_connection()
            try:
                affected = 0
                for e in events:
                    existing = conn.execute(
                        "SELECT actual, forecast, previous, revised, "
                        "previous_before_revision FROM calendar_cache "
                        "WHERE event_id = ?",
                        (e.event_id,),
                    ).fetchone()

                    revised = e.revised
                    prev_before = e.previous_before_revision
                    if existing is not None:
                        changed = (
                            existing["actual"] != e.actual
                            or existing["forecast"] != e.forecast
                            or existing["previous"] != e.previous
                        )
                        revised = revised or bool(existing["revised"]) or changed
                        # Capture the pre-revision published value the first time
                        # ``actual`` changes away from a known value.
                        if prev_before is None:
                            if (
                                changed
                                and existing["actual"] is not None
                                and existing["actual"] != e.actual
                            ):
                                prev_before = existing["actual"]
                            else:
                                prev_before = existing["previous_before_revision"]

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO calendar_cache
                            (event_id, source, event, currency, impact,
                             scheduled_at, markets, series_code, license_label,
                             organism, source_timezone, value_unit, actual,
                             forecast, previous, revised,
                             previous_before_revision, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            e.event_id,
                            e.source,
                            e.event,
                            e.currency,
                            e.impact,
                            _utc_iso(e.scheduled_at),
                            json.dumps(e.markets),
                            e.series_code,
                            e.license_label,
                            e.organism,
                            e.source_timezone,
                            e.value_unit,
                            e.actual,
                            e.forecast,
                            e.previous,
                            1 if revised else 0,
                            prev_before,
                            fetched_iso,
                        ),
                    )
                    affected += 1
                return affected
            finally:
                conn.close()

    def get_events_between(
        self, start: datetime, end: datetime
    ) -> List[CalendarCacheEvent]:
        """Events with ``start <= scheduled_at <= end``, chronological."""
        start_iso = _utc_iso(start)
        end_iso = _utc_iso(end)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT * FROM calendar_cache
                    WHERE scheduled_at >= ? AND scheduled_at <= ?
                    ORDER BY scheduled_at ASC
                    """,
                    (start_iso, end_iso),
                )
                return [self._row_to_event(row) for row in cur.fetchall()]
            finally:
                conn.close()

    def coverage_bounds(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """Earliest and latest ``scheduled_at`` across ALL cached rows, or
        (None, None) when empty. Feeds the honest coverage indicator."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT MIN(scheduled_at) AS lo, MAX(scheduled_at) AS hi "
                    "FROM calendar_cache"
                )
                row = cur.fetchone()
                if row is None or row["lo"] is None:
                    return (None, None)
                return (_parse_iso(row["lo"]), _parse_iso(row["hi"]))
            finally:
                conn.close()

    def last_fetch_at(self) -> Optional[datetime]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT MAX(fetched_at) AS m FROM calendar_cache")
                row = cur.fetchone()
                if row is None or row["m"] is None:
                    return None
                return _parse_iso(row["m"])
            finally:
                conn.close()

    def purge_old_events(self, older_than_days: int) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM calendar_cache "
                    "WHERE scheduled_at < datetime('now', ?)",
                    (f"-{int(older_than_days)} days",),
                )
                return cur.rowcount
            finally:
                conn.close()

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> CalendarCacheEvent:
        try:
            markets = json.loads(row["markets"])
            if not isinstance(markets, list):
                markets = []
        except (ValueError, TypeError):
            markets = []
        return CalendarCacheEvent(
            event_id=row["event_id"],
            source=row["source"],
            event=row["event"],
            currency=row["currency"],
            impact=row["impact"],
            scheduled_at=_parse_iso(row["scheduled_at"]),
            markets=[str(m) for m in markets],
            series_code=row["series_code"],
            license_label=row["license_label"],
            organism=row["organism"],
            source_timezone=row["source_timezone"],
            value_unit=row["value_unit"],
            actual=row["actual"],
            forecast=row["forecast"],
            previous=row["previous"],
            revised=bool(row["revised"]),
            previous_before_revision=row["previous_before_revision"],
        )


__all__ = ["CalendarCacheStore", "CalendarCacheEvent"]
