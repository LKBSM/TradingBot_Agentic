"""SQLite-backed cache of raw OHLCV candles fetched from external providers.

Used by ``scripts/seed_twelve_data.py`` and (later) by the live engine for
warm-start / reprocessing. One row per ``(instrument, timeframe, ts)``.

Same pattern as ``MarketReadingsStore``: WAL mode, ``RLock`` for thread
safety, connection-per-call, ``schema_version`` table, env-aware path.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from src.intelligence.data_providers.twelve_data_provider import Candle
from src.persistence.sqlite_pragmas import apply_wal

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CandleCoverage:
    """How much history is actually in the cache for one combo.

    ``oldest_ts`` is the truthful start of coverage — what the UI shows as
    "historique depuis le …". A partial fill must never be presented as complete,
    so this is read straight from the rows, never inferred from a target.
    """

    count: int
    oldest_ts: Optional[datetime]
    newest_ts: Optional[datetime]


class CandlesCacheStore:
    """Persistent cache for raw OHLCV candles.

    Path resolution priority:
      1. Explicit ``db_path`` argument
      2. ``CANDLES_DB_PATH`` env var (if non-empty)
      3. ``DEFAULT_DB_PATH`` (``./data/candles.db``)
    """

    SCHEMA_VERSION = 2
    DEFAULT_DB_PATH = "./data/candles.db"
    DB_PATH_ENV_VAR = "CANDLES_DB_PATH"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("CandlesCacheStore initialised at %s", self._db_path)

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
        apply_wal(conn)
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
                CREATE TABLE IF NOT EXISTS candles_cache (
                    instrument TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL,
                    PRIMARY KEY(instrument, timeframe, ts)
                );
                CREATE INDEX IF NOT EXISTS idx_candles_cache_lookup
                    ON candles_cache(instrument, timeframe, ts DESC);
                """
            )
        if 0 < from_v < 2:
            # One-time wipe (audit 2026-06-12 §T2/T3): rows written before the
            # timezone fix carry exchange-local timestamps mislabelled as UTC
            # (up to +10h in the future) plus overwritten forming-bar values.
            # Keyed by (instrument, timeframe, ts), they would survive as ghost
            # future bars on the chart. The cache is a pure cache — dropping it
            # is safe; the assembler refills it with closed UTC bars on the
            # next generation.
            conn.execute("DELETE FROM candles_cache")

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def upsert_candles(
        self, instrument: str, timeframe: str, candles: List[Candle]
    ) -> int:
        """Upsert a batch of candles. Returns the number of rows affected."""
        if not candles:
            return 0
        rows = [
            (
                instrument,
                timeframe,
                c.ts.isoformat() if hasattr(c.ts, "isoformat") else str(c.ts),
                c.open, c.high, c.low, c.close, c.volume,
            )
            for c in candles
        ]
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.executemany(
                    """
                    INSERT OR REPLACE INTO candles_cache
                        (instrument, timeframe, ts, open, high, low, close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
                return cur.rowcount
            finally:
                conn.close()

    def get_latest_candle(
        self, instrument: str, timeframe: str
    ) -> Optional[Candle]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume FROM candles_cache
                    WHERE instrument = ? AND timeframe = ?
                    ORDER BY ts DESC LIMIT 1
                    """,
                    (instrument, timeframe),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return Candle(
                    ts=datetime.fromisoformat(row["ts"]),
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"] if row["volume"] is not None else 0.0,
                )
            finally:
                conn.close()

    def get_last_n_candles(
        self, instrument: str, timeframe: str, n: int
    ) -> List[Candle]:
        """Return the most recent ``n`` candles in ascending chronological order.

        Read-only window over ``candles_cache`` (already populated by the
        assembler / scheduler). No external provider call is made here — this is
        a pure cache read. Returns an empty list if the combo has no cached
        candles. ``n <= 0`` returns an empty list.
        """
        if n <= 0:
            return []
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume FROM candles_cache
                    WHERE instrument = ? AND timeframe = ?
                    ORDER BY ts DESC LIMIT ?
                    """,
                    (instrument, timeframe, n),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        # Rows come back newest-first; reverse to ascending (chart expects it).
        candles = [
            Candle(
                ts=datetime.fromisoformat(row["ts"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"] if row["volume"] is not None else 0.0,
            )
            for row in reversed(rows)
        ]
        return candles

    def get_candles_before(
        self, instrument: str, timeframe: str, before_ts: datetime, n: int
    ) -> List[Candle]:
        """Return up to ``n`` candles STRICTLY BEFORE ``before_ts``, ascending.

        Backs the chart's history pagination (CHART-1): when the user zooms out
        past what is loaded, the front asks for the ``n`` candles just older than
        its current oldest bar. Pure cache read over ``candles_cache`` — no
        provider call, no quota. Empty list means there is nothing older (the
        floor of available data for this combo). Uses the ``ts DESC`` index.

        ``ts`` is stored as ISO-8601 text; the boundary is compared as ISO text.
        The front sends back the epoch of a candle it received from this same
        store, so the boundary reproduces the stored format (UTC).
        """
        if n <= 0:
            return []
        boundary = (
            before_ts.isoformat() if hasattr(before_ts, "isoformat") else str(before_ts)
        )
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    """
                    SELECT ts, open, high, low, close, volume FROM candles_cache
                    WHERE instrument = ? AND timeframe = ? AND ts < ?
                    ORDER BY ts DESC LIMIT ?
                    """,
                    (instrument, timeframe, boundary, n),
                )
                rows = cur.fetchall()
            finally:
                conn.close()
        # Newest-first from the query; reverse to ascending (chart expects it).
        return [
            Candle(
                ts=datetime.fromisoformat(row["ts"]),
                open=row["open"],
                high=row["high"],
                low=row["low"],
                close=row["close"],
                volume=row["volume"] if row["volume"] is not None else 0.0,
            )
            for row in reversed(rows)
        ]

    def count_candles(self, instrument: str, timeframe: str) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS n FROM candles_cache "
                    "WHERE instrument = ? AND timeframe = ?",
                    (instrument, timeframe),
                )
                return int(cur.fetchone()["n"])
            finally:
                conn.close()

    def get_coverage(self, instrument: str, timeframe: str) -> CandleCoverage:
        """Count + oldest + newest cached candle for a combo, in one query.

        The single source for honest coverage reporting (backfill skip-logic and
        the "historique depuis le …" UI). Empty combo → count 0, timestamps None.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS n, MIN(ts) AS oldest, MAX(ts) AS newest "
                    "FROM candles_cache WHERE instrument = ? AND timeframe = ?",
                    (instrument, timeframe),
                )
                row = cur.fetchone()
            finally:
                conn.close()
        count = int(row["n"]) if row and row["n"] is not None else 0
        oldest = datetime.fromisoformat(row["oldest"]) if count and row["oldest"] else None
        newest = datetime.fromisoformat(row["newest"]) if count and row["newest"] else None
        return CandleCoverage(count=count, oldest_ts=oldest, newest_ts=newest)
