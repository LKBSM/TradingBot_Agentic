"""SQLite-backed cache of macro economic news events (Chantier 3).

Backs the ForexFactory news pipeline (``src/intelligence/news_pipeline.py``).
One row per ``event_id`` (deterministic hash of title+currency+scheduled_at),
so re-fetching the same FF feed upserts in place — natural deduplication
(architecture doc §3.4 ``news_cache`` table + "logic de déduplication").

Same pattern as ``MarketReadingsStore`` / ``CandlesCacheStore``: WAL mode,
``RLock`` for thread safety, connection-per-call, ``schema_version`` table,
env-aware path (``NEWS_CACHE_DB_PATH`` → ``./data/news_cache.db``).

The ``NormalizedNewsEvent`` value object lives here (not in news_pipeline) so
the pipeline imports the store — never the reverse — avoiding a circular import.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _utc_iso(ts: Optional[datetime] = None) -> str:
    """Render a datetime as an ISO 8601 UTC string with a ``Z`` suffix.

    Naive datetimes are assumed to already represent UTC. ``ts=None`` → now.
    The fixed ``timespec='seconds'`` format makes the strings directly
    comparable lexicographically (chronological order == string order).
    """
    ts = ts if ts is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class NormalizedNewsEvent:
    """A single macro economic event, normalized from the ForexFactory feed.

    ``impact`` is constrained to ``"medium"`` / ``"high"`` by the pipeline
    (low/holiday/non-economic are dropped at normalization for economy).
    ``scheduled_at`` is always timezone-aware UTC.
    """

    event_id: str
    event: str
    currency: str
    impact: str
    scheduled_at: datetime
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None


class NewsCacheStore:
    """Persistent cache for normalized macro news events.

    Path resolution priority:
      1. Explicit ``db_path`` argument
      2. ``NEWS_CACHE_DB_PATH`` env var (if non-empty)
      3. ``DEFAULT_DB_PATH`` (``./data/news_cache.db``)
    """

    SCHEMA_VERSION = 1
    DEFAULT_DB_PATH = "./data/news_cache.db"
    DB_PATH_ENV_VAR = "NEWS_CACHE_DB_PATH"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("NewsCacheStore initialised at %s", self._db_path)

    @classmethod
    def _resolve_db_path(cls, db_path: Optional[str]) -> Path:
        if db_path:
            return Path(db_path)
        env_val = os.environ.get(cls.DB_PATH_ENV_VAR)
        if env_val:
            return Path(env_val)
        return Path(cls.DEFAULT_DB_PATH)

    # ------------------------------------------------------------------ #
    # SQLite helpers
    # ------------------------------------------------------------------ #
    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self._db_path), timeout=30.0, isolation_level=None
        )
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
                CREATE TABLE IF NOT EXISTS news_cache (
                    event_id TEXT PRIMARY KEY,
                    event TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    impact TEXT NOT NULL,
                    scheduled_at TEXT NOT NULL,
                    actual REAL,
                    forecast REAL,
                    previous REAL,
                    fetched_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_news_cache_window
                    ON news_cache(scheduled_at);
                """
            )

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def upsert_events(
        self,
        events: List[NormalizedNewsEvent],
        fetched_at: Optional[datetime] = None,
    ) -> int:
        """Upsert a batch of events (dedup by ``event_id``). Returns rows affected.

        ``fetched_at`` stamps the refresh time; ``last_fetch_at()`` reads it
        back so the pipeline can honour its TTL.
        """
        if not events:
            return 0
        fetched_iso = _utc_iso(fetched_at)
        rows = [
            (
                e.event_id,
                e.event,
                e.currency,
                e.impact,
                _utc_iso(e.scheduled_at),
                e.actual,
                e.forecast,
                e.previous,
                fetched_iso,
            )
            for e in events
        ]
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.executemany(
                    """
                    INSERT OR REPLACE INTO news_cache
                        (event_id, event, currency, impact, scheduled_at,
                         actual, forecast, previous, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                return cur.rowcount
            finally:
                conn.close()

    def get_events_between(
        self, start: datetime, end: datetime
    ) -> List[NormalizedNewsEvent]:
        """Return events with ``start <= scheduled_at <= end``, chronological."""
        start_iso = _utc_iso(start)
        end_iso = _utc_iso(end)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT event_id, event, currency, impact, scheduled_at,
                           actual, forecast, previous
                    FROM news_cache
                    WHERE scheduled_at >= ? AND scheduled_at <= ?
                    ORDER BY scheduled_at ASC
                    """,
                    (start_iso, end_iso),
                )
                return [self._row_to_event(row) for row in cur.fetchall()]
            finally:
                conn.close()

    def last_fetch_at(self) -> Optional[datetime]:
        """Return the most recent ``fetched_at`` across all rows, or None."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT MAX(fetched_at) AS m FROM news_cache")
                row = cur.fetchone()
                if row is None or row["m"] is None:
                    return None
                return _parse_iso(row["m"])
            finally:
                conn.close()

    def purge_old_events(self, older_than_days: int) -> int:
        """Delete events whose ``scheduled_at`` is older than N days. Returns rows deleted."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM news_cache "
                    "WHERE scheduled_at < datetime('now', ?)",
                    (f"-{int(older_than_days)} days",),
                )
                return cur.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> NormalizedNewsEvent:
        return NormalizedNewsEvent(
            event_id=row["event_id"],
            event=row["event"],
            currency=row["currency"],
            impact=row["impact"],
            scheduled_at=_parse_iso(row["scheduled_at"]),
            actual=row["actual"],
            forecast=row["forecast"],
            previous=row["previous"],
        )


def _parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 string (with ``Z`` or offset) into aware UTC."""
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


__all__ = ["NewsCacheStore", "NormalizedNewsEvent"]
