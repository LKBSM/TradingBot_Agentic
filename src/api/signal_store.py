"""Thread-safe signal state manager with SQLite history persistence."""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASS
# =============================================================================

@dataclass
class SignalRecord:
    """A single trading signal."""
    signal_id: str
    action: str
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    created_at: str  # ISO-8601
    outcome: Optional[str] = None
    pnl_pips: Optional[float] = None
    closed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SIGNAL STORE
# =============================================================================

class SignalStore:
    """
    Thread-safe signal state + SQLite history.

    The trading loop calls ``publish()`` to write the current signal.
    The API calls ``get_current()`` and ``get_history()`` to read.

    Follows the same WAL-mode SQLite pattern as
    ``src/persistence/kill_switch_store.py``.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/signals.db"):
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._current: Optional[SignalRecord] = None

        # Ensure directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("SignalStore initialised at %s", self._db_path)

    # --------------------------------------------------------------------- #
    # SQLite helpers
    # --------------------------------------------------------------------- #
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
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_v: int) -> None:
        if from_v < 1:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    action    TEXT NOT NULL,
                    symbol    TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    stop_loss   REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    rr_ratio    REAL NOT NULL,
                    created_at  TEXT NOT NULL,
                    outcome     TEXT,
                    pnl_pips    REAL,
                    closed_at   TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_signals_created
                    ON signals(created_at DESC);
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def publish(self, record: SignalRecord) -> None:
        """Write a new signal (called by the trading loop)."""
        with self._lock:
            self._current = record
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO signals "
                    "(signal_id, action, symbol, entry_price, stop_loss, "
                    " take_profit, rr_ratio, created_at, outcome, pnl_pips, closed_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record.signal_id,
                        record.action,
                        record.symbol,
                        record.entry_price,
                        record.stop_loss,
                        record.take_profit,
                        record.rr_ratio,
                        record.created_at,
                        record.outcome,
                        record.pnl_pips,
                        record.closed_at,
                    ),
                )
            finally:
                conn.close()

    def get_current(self) -> Optional[SignalRecord]:
        """Return the most-recently published signal (API reads)."""
        with self._lock:
            return self._current

    def get_history(
        self, page: int = 1, page_size: int = 20
    ) -> Tuple[List[SignalRecord], int]:
        """
        Paginated signal history.

        Returns:
            (list_of_records, total_count)
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT COUNT(*) AS cnt FROM signals")
                total = cur.fetchone()["cnt"]

                offset = (page - 1) * page_size
                cur = conn.execute(
                    "SELECT * FROM signals ORDER BY created_at DESC "
                    "LIMIT ? OFFSET ?",
                    (page_size, offset),
                )
                rows = cur.fetchall()
                records = [
                    SignalRecord(
                        signal_id=r["signal_id"],
                        action=r["action"],
                        symbol=r["symbol"],
                        entry_price=r["entry_price"],
                        stop_loss=r["stop_loss"],
                        take_profit=r["take_profit"],
                        rr_ratio=r["rr_ratio"],
                        created_at=r["created_at"],
                        outcome=r["outcome"],
                        pnl_pips=r["pnl_pips"],
                        closed_at=r["closed_at"],
                    )
                    for r in rows
                ]
                return records, total
            finally:
                conn.close()

    def update_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl_pips: float,
        closed_at: Optional[str] = None,
    ) -> bool:
        """Mark a signal as closed with its outcome."""
        closed_at = closed_at or datetime.now(tz=None).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE signals SET outcome=?, pnl_pips=?, closed_at=? "
                    "WHERE signal_id=?",
                    (outcome, pnl_pips, closed_at, signal_id),
                )
                if self._current and self._current.signal_id == signal_id:
                    self._current.outcome = outcome
                    self._current.pnl_pips = pnl_pips
                    self._current.closed_at = closed_at
                return cur.rowcount > 0
            finally:
                conn.close()
