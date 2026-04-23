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
    # Smart Sentinel narrative fields
    confluence_score: Optional[float] = None
    narrative: Optional[str] = None
    validation_reason: Optional[str] = None
    key_confluences: Optional[str] = None
    risk_warnings: Optional[str] = None
    market_context: Optional[str] = None
    # Volatility forecast fields (schema v3)
    vol_forecast_atr: Optional[float] = None
    vol_regime: Optional[str] = None
    vol_confidence: Optional[str] = None  # JSON: {"lower": x, "upper": y}

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

    SCHEMA_VERSION = 3

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
        if from_v < 2:
            # Smart Sentinel narrative columns
            for col, col_type in [
                ("confluence_score", "REAL"),
                ("narrative", "TEXT"),
                ("validation_reason", "TEXT"),
                ("key_confluences", "TEXT"),
                ("risk_warnings", "TEXT"),
                ("market_context", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

        if from_v < 3:
            # Volatility forecast columns
            for col, col_type in [
                ("vol_forecast_atr", "REAL"),
                ("vol_regime", "TEXT"),
                ("vol_confidence", "TEXT"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE signals ADD COLUMN {col} {col_type}")
                except sqlite3.OperationalError:
                    pass  # Column already exists

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
                    " take_profit, rr_ratio, created_at, outcome, pnl_pips, closed_at, "
                    " confluence_score, narrative, validation_reason, "
                    " key_confluences, risk_warnings, market_context, "
                    " vol_forecast_atr, vol_regime, vol_confidence) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        record.confluence_score,
                        record.narrative,
                        record.validation_reason,
                        record.key_confluences,
                        record.risk_warnings,
                        record.market_context,
                        record.vol_forecast_atr,
                        record.vol_regime,
                        record.vol_confidence,
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
                records = [self._row_to_record(r) for r in rows]
                return records, total
            finally:
                conn.close()

    def get_by_id(self, signal_id: str) -> Optional[SignalRecord]:
        """Retrieve a single signal by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM signals WHERE signal_id = ?",
                    (signal_id,),
                )
                row = cur.fetchone()
                return self._row_to_record(row) if row else None
            finally:
                conn.close()

    @staticmethod
    def _row_to_record(r: Any) -> SignalRecord:
        """Convert a sqlite3.Row to SignalRecord."""
        # Handle both old (v1) and new (v2) schema
        def _get(key: str, default=None):
            try:
                return r[key]
            except (IndexError, KeyError):
                return default

        return SignalRecord(
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
            confluence_score=_get("confluence_score"),
            narrative=_get("narrative"),
            validation_reason=_get("validation_reason"),
            key_confluences=_get("key_confluences"),
            risk_warnings=_get("risk_warnings"),
            market_context=_get("market_context"),
        )

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
