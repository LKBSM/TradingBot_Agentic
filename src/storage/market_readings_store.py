"""Thread-safe SQLite store for MarketReadings + active combinations.

Owns two tables in one .db file (./data/market_readings.db by default):
  - market_readings    : payload JSON per (instrument, timeframe, candle_close_ts)
  - active_combinations: hybrid-mode scheduler state (cf. Chantier 3)

Follows the WAL-mode SQLite pattern used by ``src/api/signal_store.py`` and
``src/persistence/kill_switch_store.py`` (connection-per-call, RLock,
``schema_version`` table, versioned ``_migrate(conn, from_v)``).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_iso(ts: Optional[datetime] = None) -> str:
    """Render a datetime as an ISO 8601 UTC string.

    Naive datetimes are assumed to already represent UTC. Aware datetimes are
    converted. ``ts=None`` means "now".
    """
    ts = ts if ts is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    # ISO-8601 with explicit "Z" suffix for unambiguous UTC marking
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


class MarketReadingsStore:
    """SQLite-backed persistence for MarketReadings and active combinations.

    Usage:
        store = MarketReadingsStore()  # ./data/market_readings.db
        store.save_reading("XAUUSD", "M15", candle_ts, payload_dict)
        latest = store.get_latest_reading("XAUUSD", "M15")
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/market_readings.db") -> None:
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("MarketReadingsStore initialised at %s", self._db_path)

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
                CREATE TABLE IF NOT EXISTS market_readings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    instrument TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    candle_close_ts TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(instrument, timeframe, candle_close_ts)
                );
                CREATE INDEX IF NOT EXISTS idx_market_readings_lookup
                    ON market_readings(instrument, timeframe, candle_close_ts DESC);
                CREATE INDEX IF NOT EXISTS idx_market_readings_cleanup
                    ON market_readings(created_at);
                CREATE TABLE IF NOT EXISTS active_combinations (
                    instrument TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_accessed_at TEXT NOT NULL,
                    last_accessed_at TEXT NOT NULL,
                    PRIMARY KEY(instrument, timeframe)
                );
                """
            )

    # ------------------------------------------------------------------ #
    # MarketReadings CRUD
    # ------------------------------------------------------------------ #
    def save_reading(
        self,
        instrument: str,
        timeframe: str,
        candle_close_ts: datetime,
        payload: Dict[str, Any],
    ) -> int:
        """Persist a MarketReading payload. Returns the SQLite rowid.

        Upsert semantics: a duplicate ``(instrument, timeframe, candle_close_ts)``
        replaces the previous row. Allows safe re-processing of the same candle
        (e.g. recovery after crash, corrected backfill).
        """
        ts_iso = _utc_iso(candle_close_ts)
        payload_json = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    INSERT OR REPLACE INTO market_readings
                        (instrument, timeframe, candle_close_ts, payload_json)
                    VALUES (?, ?, ?, ?)
                    """,
                    (instrument, timeframe, ts_iso, payload_json),
                )
                return int(cur.lastrowid)
            finally:
                conn.close()

    def get_latest_reading(
        self, instrument: str, timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent payload, or ``None`` if no reading exists."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT payload_json FROM market_readings
                    WHERE instrument = ? AND timeframe = ?
                    ORDER BY candle_close_ts DESC
                    LIMIT 1
                    """,
                    (instrument, timeframe),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return json.loads(row["payload_json"])
            finally:
                conn.close()

    def get_readings_history(
        self,
        instrument: str,
        timeframe: str,
        since: datetime,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Return payloads with ``candle_close_ts >= since``, newest first."""
        since_iso = _utc_iso(since)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT payload_json FROM market_readings
                    WHERE instrument = ? AND timeframe = ?
                          AND candle_close_ts >= ?
                    ORDER BY candle_close_ts DESC
                    LIMIT ?
                    """,
                    (instrument, timeframe, since_iso, limit),
                )
                return [json.loads(row["payload_json"]) for row in cur.fetchall()]
            finally:
                conn.close()

    def purge_old_readings(self, older_than_days: int) -> int:
        """Delete readings whose ``created_at`` is older than N days.

        Returns the number of rows deleted. Intended for the retention cron
        (tiered retention: 7 days for Approfondie, 30 days for Intégrale,
        90 days absolute cap per architecture doc §3.5).
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM market_readings "
                    "WHERE created_at < datetime('now', ?)",
                    (f"-{int(older_than_days)} days",),
                )
                return cur.rowcount
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # Active combinations (hybrid-mode scheduler state)
    # ------------------------------------------------------------------ #
    def mark_combination_active(self, instrument: str, timeframe: str) -> None:
        """Record an access. Upsert: preserves ``first_accessed_at``, refreshes ``last_accessed_at``."""
        now_iso = _utc_iso()
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT INTO active_combinations
                        (instrument, timeframe, first_accessed_at, last_accessed_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(instrument, timeframe) DO UPDATE SET
                        last_accessed_at = excluded.last_accessed_at
                    """,
                    (instrument, timeframe, now_iso, now_iso),
                )
            finally:
                conn.close()

    def get_active_combinations(self, since: datetime) -> List[Tuple[str, str]]:
        """Return ``(instrument, timeframe)`` pairs accessed at least once since ``since``."""
        since_iso = _utc_iso(since)
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT instrument, timeframe FROM active_combinations
                    WHERE last_accessed_at >= ?
                    ORDER BY last_accessed_at DESC
                    """,
                    (since_iso,),
                )
                return [(row["instrument"], row["timeframe"]) for row in cur.fetchall()]
            finally:
                conn.close()
