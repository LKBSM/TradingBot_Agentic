# =============================================================================
# KILL SWITCH PERSISTENCE - SQLite Backend
# =============================================================================
# Persistent storage for Kill Switch state to survive bot restarts.
#
# CRITICAL SECURITY FEATURE:
# Without persistence, a bot crash/restart could bypass kill switch limits:
#   1. Bot hits 10% drawdown limit -> Kill Switch activates
#   2. Bot crashes or is restarted
#   3. Kill Switch state is lost (was in memory only)
#   4. Bot resumes trading, ignoring the limit that was hit
#
# This module prevents that scenario by persisting state to SQLite.
#
# Features:
#   - Atomic writes with WAL mode for crash safety
#   - Automatic schema migration
#   - Heartbeat tracking for crash detection
#   - Full audit trail of halt events
#
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class KillSwitchState:
    """Persisted Kill Switch state."""
    halt_level: int  # HaltLevel enum value
    halt_reason: Optional[str]  # HaltReason enum value or None
    halt_message: str
    halt_time: Optional[str]  # ISO format datetime
    is_manually_halted: bool
    equity: float
    peak_equity: float
    daily_pnl: float
    weekly_pnl: float
    consecutive_losses: int
    last_updated: str  # ISO format datetime

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'KillSwitchState':
        return cls(**data)


@dataclass
class BreakerRecord:
    """Record of a circuit breaker state."""
    name: str
    state: str  # BreakerState enum value
    trip_count: int
    last_trip_time: Optional[str]  # ISO format datetime
    recovery_time: Optional[str]  # ISO format datetime
    threshold: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BreakerRecord':
        return cls(**data)


@dataclass
class HaltEventRecord:
    """Record of a halt event for audit trail."""
    halt_id: str
    reason: str
    level: int
    timestamp: str
    trigger_value: float
    threshold: float
    message: str
    auto_recovery: bool
    recovery_time: Optional[str]
    recovered: bool
    recovery_timestamp: Optional[str]


# =============================================================================
# SQLITE STORE
# =============================================================================

class KillSwitchStore:
    """
    SQLite-backed persistent storage for Kill Switch state.

    This class provides crash-safe persistence for the Kill Switch,
    ensuring that safety limits are enforced even after bot restarts.

    Features:
    - WAL mode for concurrent reads and crash safety
    - Automatic schema creation and migration
    - Heartbeat tracking for crash detection
    - Full audit trail of all halt events

    Example:
        store = KillSwitchStore("./data/kill_switch.db")

        # Save state
        state = KillSwitchState(
            halt_level=5,
            halt_reason="max_drawdown",
            ...
        )
        store.save_state(state)

        # Load state on restart
        loaded_state = store.load_state()
        if loaded_state and loaded_state.halt_level >= 5:
            print("WARNING: Bot was halted before restart!")
    """

    # Schema version for migrations
    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str = "./data/kill_switch.db",
        heartbeat_interval_seconds: int = 30
    ):
        """
        Initialize the Kill Switch store.

        Args:
            db_path: Path to SQLite database file
            heartbeat_interval_seconds: How often to update heartbeat
        """
        self.db_path = Path(db_path)
        self.heartbeat_interval = heartbeat_interval_seconds
        self._logger = logging.getLogger("persistence.kill_switch")
        self._lock = threading.RLock()

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_database()

        self._logger.info(f"KillSwitchStore initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with proper settings."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None  # Autocommit mode
        )
        conn.row_factory = sqlite3.Row

        # Enable WAL mode for crash safety and concurrent access
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")

        return conn

    def _init_database(self) -> None:
        """Initialize database schema."""
        with self._lock:
            conn = self._get_connection()
            try:
                # Create schema version table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_version (
                        version INTEGER PRIMARY KEY
                    )
                """)

                # Check current version
                cursor = conn.execute("SELECT version FROM schema_version LIMIT 1")
                row = cursor.fetchone()
                current_version = row['version'] if row else 0

                if current_version < self.SCHEMA_VERSION:
                    self._migrate_schema(conn, current_version)

            finally:
                conn.close()

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int) -> None:
        """Migrate database schema to current version."""
        self._logger.info(f"Migrating schema from v{from_version} to v{self.SCHEMA_VERSION}")

        if from_version < 1:
            # Initial schema
            conn.executescript("""
                -- Main Kill Switch state
                CREATE TABLE IF NOT EXISTS kill_switch_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    halt_level INTEGER NOT NULL DEFAULT 0,
                    halt_reason TEXT,
                    halt_message TEXT DEFAULT '',
                    halt_time TEXT,
                    is_manually_halted INTEGER NOT NULL DEFAULT 0,
                    equity REAL NOT NULL DEFAULT 0,
                    peak_equity REAL NOT NULL DEFAULT 0,
                    daily_pnl REAL NOT NULL DEFAULT 0,
                    weekly_pnl REAL NOT NULL DEFAULT 0,
                    consecutive_losses INTEGER NOT NULL DEFAULT 0,
                    last_updated TEXT NOT NULL
                );

                -- Circuit breaker states
                CREATE TABLE IF NOT EXISTS circuit_breakers (
                    name TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'closed',
                    trip_count INTEGER NOT NULL DEFAULT 0,
                    last_trip_time TEXT,
                    recovery_time TEXT,
                    threshold REAL NOT NULL DEFAULT 0
                );

                -- Halt event history (audit trail)
                CREATE TABLE IF NOT EXISTS halt_events (
                    halt_id TEXT PRIMARY KEY,
                    reason TEXT NOT NULL,
                    level INTEGER NOT NULL,
                    timestamp TEXT NOT NULL,
                    trigger_value REAL NOT NULL,
                    threshold REAL NOT NULL,
                    message TEXT NOT NULL,
                    auto_recovery INTEGER NOT NULL DEFAULT 0,
                    recovery_time TEXT,
                    recovered INTEGER NOT NULL DEFAULT 0,
                    recovery_timestamp TEXT
                );

                -- Heartbeat for crash detection
                CREATE TABLE IF NOT EXISTS heartbeat (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    last_heartbeat TEXT NOT NULL,
                    pid INTEGER NOT NULL,
                    started_at TEXT NOT NULL
                );

                -- Create indexes
                CREATE INDEX IF NOT EXISTS idx_halt_events_timestamp
                    ON halt_events(timestamp);
                CREATE INDEX IF NOT EXISTS idx_halt_events_reason
                    ON halt_events(reason);
            """)

            # Update version
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,)
            )

        self._logger.info(f"Schema migration complete (now v{self.SCHEMA_VERSION})")

    # =========================================================================
    # STATE PERSISTENCE
    # =========================================================================

    def save_state(self, state: KillSwitchState) -> bool:
        """
        Save Kill Switch state to database.

        Args:
            state: Current Kill Switch state

        Returns:
            True if saved successfully
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO kill_switch_state (
                        id, halt_level, halt_reason, halt_message, halt_time,
                        is_manually_halted, equity, peak_equity, daily_pnl,
                        weekly_pnl, consecutive_losses, last_updated
                    ) VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    state.halt_level,
                    state.halt_reason,
                    state.halt_message,
                    state.halt_time,
                    1 if state.is_manually_halted else 0,
                    state.equity,
                    state.peak_equity,
                    state.daily_pnl,
                    state.weekly_pnl,
                    state.consecutive_losses,
                    state.last_updated
                ))

                self._logger.debug(f"Saved Kill Switch state: halt_level={state.halt_level}")
                return True

            except Exception as e:
                self._logger.error(f"Failed to save state: {e}")
                return False
            finally:
                conn.close()

    def load_state(self) -> Optional[KillSwitchState]:
        """
        Load Kill Switch state from database.

        Returns:
            KillSwitchState if found, None otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM kill_switch_state WHERE id = 1"
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                return KillSwitchState(
                    halt_level=row['halt_level'],
                    halt_reason=row['halt_reason'],
                    halt_message=row['halt_message'] or '',
                    halt_time=row['halt_time'],
                    is_manually_halted=bool(row['is_manually_halted']),
                    equity=row['equity'],
                    peak_equity=row['peak_equity'],
                    daily_pnl=row['daily_pnl'],
                    weekly_pnl=row['weekly_pnl'],
                    consecutive_losses=row['consecutive_losses'],
                    last_updated=row['last_updated']
                )

            except Exception as e:
                self._logger.error(f"Failed to load state: {e}")
                return None
            finally:
                conn.close()

    # =========================================================================
    # CIRCUIT BREAKER PERSISTENCE
    # =========================================================================

    def save_breaker(self, breaker: BreakerRecord) -> bool:
        """Save circuit breaker state."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO circuit_breakers (
                        name, state, trip_count, last_trip_time,
                        recovery_time, threshold
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    breaker.name,
                    breaker.state,
                    breaker.trip_count,
                    breaker.last_trip_time,
                    breaker.recovery_time,
                    breaker.threshold
                ))
                return True
            except Exception as e:
                self._logger.error(f"Failed to save breaker {breaker.name}: {e}")
                return False
            finally:
                conn.close()

    def load_breakers(self) -> List[BreakerRecord]:
        """Load all circuit breaker states."""
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute("SELECT * FROM circuit_breakers")
                rows = cursor.fetchall()

                return [
                    BreakerRecord(
                        name=row['name'],
                        state=row['state'],
                        trip_count=row['trip_count'],
                        last_trip_time=row['last_trip_time'],
                        recovery_time=row['recovery_time'],
                        threshold=row['threshold']
                    )
                    for row in rows
                ]
            except Exception as e:
                self._logger.error(f"Failed to load breakers: {e}")
                return []
            finally:
                conn.close()

    # =========================================================================
    # HALT EVENT AUDIT TRAIL
    # =========================================================================

    def record_halt_event(self, event: HaltEventRecord) -> bool:
        """Record a halt event for audit trail."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO halt_events (
                        halt_id, reason, level, timestamp, trigger_value,
                        threshold, message, auto_recovery, recovery_time,
                        recovered, recovery_timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.halt_id,
                    event.reason,
                    event.level,
                    event.timestamp,
                    event.trigger_value,
                    event.threshold,
                    event.message,
                    1 if event.auto_recovery else 0,
                    event.recovery_time,
                    1 if event.recovered else 0,
                    event.recovery_timestamp
                ))
                return True
            except Exception as e:
                self._logger.error(f"Failed to record halt event: {e}")
                return False
            finally:
                conn.close()

    def get_halt_history(
        self,
        limit: int = 100,
        since: Optional[datetime] = None
    ) -> List[HaltEventRecord]:
        """Get halt event history."""
        with self._lock:
            conn = self._get_connection()
            try:
                if since:
                    cursor = conn.execute(
                        """
                        SELECT * FROM halt_events
                        WHERE timestamp >= ?
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (since.isoformat(), limit)
                    )
                else:
                    cursor = conn.execute(
                        """
                        SELECT * FROM halt_events
                        ORDER BY timestamp DESC
                        LIMIT ?
                        """,
                        (limit,)
                    )

                rows = cursor.fetchall()
                return [
                    HaltEventRecord(
                        halt_id=row['halt_id'],
                        reason=row['reason'],
                        level=row['level'],
                        timestamp=row['timestamp'],
                        trigger_value=row['trigger_value'],
                        threshold=row['threshold'],
                        message=row['message'],
                        auto_recovery=bool(row['auto_recovery']),
                        recovery_time=row['recovery_time'],
                        recovered=bool(row['recovered']),
                        recovery_timestamp=row['recovery_timestamp']
                    )
                    for row in rows
                ]
            except Exception as e:
                self._logger.error(f"Failed to get halt history: {e}")
                return []
            finally:
                conn.close()

    # =========================================================================
    # HEARTBEAT / CRASH DETECTION
    # =========================================================================

    def update_heartbeat(self) -> bool:
        """
        Update heartbeat timestamp.

        Call this regularly to indicate the bot is alive.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                now = datetime.utcnow().isoformat()
                pid = os.getpid()

                # Check if this is first heartbeat
                cursor = conn.execute("SELECT started_at FROM heartbeat WHERE id = 1")
                row = cursor.fetchone()

                if row is None:
                    # First heartbeat - insert
                    conn.execute("""
                        INSERT INTO heartbeat (id, last_heartbeat, pid, started_at)
                        VALUES (1, ?, ?, ?)
                    """, (now, pid, now))
                else:
                    # Update existing
                    conn.execute("""
                        UPDATE heartbeat
                        SET last_heartbeat = ?, pid = ?
                        WHERE id = 1
                    """, (now, pid))

                return True
            except Exception as e:
                self._logger.error(f"Failed to update heartbeat: {e}")
                return False
            finally:
                conn.close()

    def check_previous_crash(self) -> Optional[Dict[str, Any]]:
        """
        Check if the previous instance crashed.

        Returns:
            Dict with crash info if previous crash detected, None otherwise
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "SELECT * FROM heartbeat WHERE id = 1"
                )
                row = cursor.fetchone()

                if row is None:
                    return None

                last_heartbeat = datetime.fromisoformat(row['last_heartbeat'])
                time_since = datetime.utcnow() - last_heartbeat

                # If last heartbeat was more than 2x interval ago, likely crashed
                if time_since > timedelta(seconds=self.heartbeat_interval * 2):
                    return {
                        'last_heartbeat': row['last_heartbeat'],
                        'previous_pid': row['pid'],
                        'started_at': row['started_at'],
                        'time_since_seconds': time_since.total_seconds()
                    }

                return None

            except Exception as e:
                self._logger.error(f"Failed to check previous crash: {e}")
                return None
            finally:
                conn.close()

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def clear_state(self) -> bool:
        """
        Clear all state (use with caution!).

        This should only be called for testing or explicit admin reset.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute("DELETE FROM kill_switch_state")
                conn.execute("DELETE FROM circuit_breakers")
                conn.execute("DELETE FROM heartbeat")
                # Keep halt_events for audit trail

                self._logger.warning("Kill Switch state cleared!")
                return True
            except Exception as e:
                self._logger.error(f"Failed to clear state: {e}")
                return False
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._lock:
            conn = self._get_connection()
            try:
                stats = {}

                # Count halt events
                cursor = conn.execute("SELECT COUNT(*) as count FROM halt_events")
                stats['total_halt_events'] = cursor.fetchone()['count']

                # Count breakers
                cursor = conn.execute("SELECT COUNT(*) as count FROM circuit_breakers")
                stats['breaker_count'] = cursor.fetchone()['count']

                # Get state last update
                cursor = conn.execute(
                    "SELECT last_updated FROM kill_switch_state WHERE id = 1"
                )
                row = cursor.fetchone()
                stats['state_last_updated'] = row['last_updated'] if row else None

                # Get db file size
                stats['db_size_bytes'] = self.db_path.stat().st_size if self.db_path.exists() else 0

                return stats

            except Exception as e:
                self._logger.error(f"Failed to get stats: {e}")
                return {}
            finally:
                conn.close()
