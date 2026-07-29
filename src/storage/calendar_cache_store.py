"""SQLite-backed cache of scheduled economic-calendar events (NW-1 / NW-1b).

Distinct from ``NewsCacheStore`` (which feeds the MarketReading "events" block):
the calendar owns its own store so widening it never risks a MarketReading
regression. The persisted row is neutral — no provider-specific column.

NW-1b schema (v2): dropped ``impact`` (no organism ranks its releases) and
``forecast`` (no organism publishes a consensus); added ``periodicity``,
``time_confirmed``, ``actual_initial`` (the value first published for a release)
and ``revised_at``. The cache is regenerable, so the v1→v2 migration simply
rebuilds the table.

Revision handling (mission NW-1b §2D): on upsert, if a row exists for an
``event_id`` and ``actual`` CHANGED away from a known value, the row is flagged
``revised=1`` with ``revised_at`` set, the first-published value is preserved in
``actual_initial``, and the new value is stored in ``actual`` — initial and
revised coexist, neither overwrites the other silently.

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
from typing import Dict, List, Optional, Tuple

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


def _opt_iso(ts: Optional[datetime]) -> Optional[str]:
    return _utc_iso(ts) if ts is not None else None


@dataclass(frozen=True)
class CalendarCacheEvent:
    """A calendar event as persisted — neutral, source-agnostic.

    NW-1b: no ``impact`` and no ``forecast``. ``markets`` is the attached-market
    list (never empty once stored)."""

    event_id: str
    source: str
    event: str
    currency: str
    scheduled_at: datetime
    markets: List[str] = field(default_factory=list)
    series_code: Optional[str] = None
    license_label: Optional[str] = None
    organism: Optional[str] = None
    periodicity: Optional[str] = None
    source_timezone: Optional[str] = None
    time_confirmed: bool = True
    value_unit: Optional[str] = None
    actual: Optional[float] = None
    actual_initial: Optional[float] = None
    previous: Optional[float] = None
    revised: bool = False
    revised_at: Optional[datetime] = None


class CalendarCacheStore:
    SCHEMA_VERSION = 2
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
        # The cache is regenerable from providers, so a schema change rebuilds
        # the table rather than ALTER-ing it (the v1 shape carried impact/forecast).
        if from_v < 2:
            conn.executescript(
                """
                DROP TABLE IF EXISTS calendar_cache;
                CREATE TABLE calendar_cache (
                    event_id TEXT PRIMARY KEY,
                    source TEXT NOT NULL,
                    event TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    markets TEXT NOT NULL,
                    series_code TEXT,
                    license_label TEXT,
                    organism TEXT,
                    periodicity TEXT,
                    source_timezone TEXT,
                    time_confirmed INTEGER NOT NULL DEFAULT 1,
                    value_unit TEXT,
                    actual REAL,
                    actual_initial REAL,
                    previous REAL,
                    revised INTEGER NOT NULL DEFAULT 0,
                    revised_at TEXT,
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
        """Upsert events (dedup by ``event_id``). When a stored ``actual`` value
        changes, flags ``revised``, stamps ``revised_at``, and preserves the
        first-published value in ``actual_initial``. Never deletes: a source that
        returns nothing keeps its stored rows. Returns rows affected."""
        if not events:
            return 0
        fetched = fetched_at or datetime.now(timezone.utc)
        fetched_iso = _utc_iso(fetched)
        with self._lock:
            conn = self._get_connection()
            try:
                affected = 0
                for e in events:
                    existing = conn.execute(
                        "SELECT actual, actual_initial, previous, revised, revised_at "
                        "FROM calendar_cache WHERE event_id = ?",
                        (e.event_id,),
                    ).fetchone()

                    revised = e.revised
                    revised_at = e.revised_at
                    actual_initial = e.actual_initial

                    if existing is not None:
                        prior_actual = existing["actual"]
                        prior_initial = existing["actual_initial"]
                        changed = (
                            prior_actual is not None
                            and e.actual is not None
                            and prior_actual != e.actual
                        )
                        # First-published value is locked once known: keep the
                        # prior initial, else the prior actual, else — when this
                        # upsert is the first non-null print — the new value.
                        if actual_initial is None:
                            if prior_initial is not None:
                                actual_initial = prior_initial
                            elif prior_actual is not None:
                                actual_initial = prior_actual
                            else:
                                actual_initial = e.actual
                        if changed:
                            revised = True
                            if revised_at is None:
                                revised_at = fetched
                        else:
                            revised = revised or bool(existing["revised"])
                            if revised_at is None and existing["revised_at"]:
                                revised_at = _parse_iso(existing["revised_at"])
                    else:
                        # First insert: a present value IS the initial print.
                        if actual_initial is None and e.actual is not None:
                            actual_initial = e.actual

                    conn.execute(
                        """
                        INSERT OR REPLACE INTO calendar_cache
                            (event_id, source, event, currency, scheduled_at,
                             markets, series_code, license_label, organism,
                             periodicity, source_timezone, time_confirmed,
                             value_unit, actual, actual_initial, previous,
                             revised, revised_at, fetched_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            e.event_id,
                            e.source,
                            e.event,
                            e.currency,
                            _utc_iso(e.scheduled_at),
                            json.dumps(e.markets),
                            e.series_code,
                            e.license_label,
                            e.organism,
                            e.periodicity,
                            e.source_timezone,
                            1 if e.time_confirmed else 0,
                            e.value_unit,
                            e.actual,
                            actual_initial,
                            e.previous,
                            1 if revised else 0,
                            _opt_iso(revised_at),
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

    def source_last_success(self) -> Dict[str, datetime]:
        """Per-source last successful refresh = MAX(fetched_at) grouped by source.
        A source that failed to refresh keeps its prior timestamp here (its rows
        are never deleted), so the page can say "not refreshed since <date>"."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT source, MAX(fetched_at) AS m FROM calendar_cache "
                    "GROUP BY source"
                )
                return {
                    row["source"]: _parse_iso(row["m"])
                    for row in cur.fetchall()
                    if row["m"] is not None
                }
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
            scheduled_at=_parse_iso(row["scheduled_at"]),
            markets=[str(m) for m in markets],
            series_code=row["series_code"],
            license_label=row["license_label"],
            organism=row["organism"],
            periodicity=row["periodicity"],
            source_timezone=row["source_timezone"],
            time_confirmed=bool(row["time_confirmed"]),
            value_unit=row["value_unit"],
            actual=row["actual"],
            actual_initial=row["actual_initial"],
            previous=row["previous"],
            revised=bool(row["revised"]),
            revised_at=_parse_iso(row["revised_at"]) if row["revised_at"] else None,
        )


__all__ = ["CalendarCacheStore", "CalendarCacheEvent"]
