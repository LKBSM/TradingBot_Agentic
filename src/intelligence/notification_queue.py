"""Persistent notification queue with TTL + replay on circuit close.

When the Telegram/Discord notifier circuit breaker opens, we drop
notifications silently (eval 09 finding #3). This queue persists pending
notifications to SQLite WAL so they can be replayed once the circuit
recovers — bounded by ``ttl_seconds`` to avoid replaying signals that
have aged out of relevance (typical: 15-30 min).

Same WAL pattern as ``SignalStore`` / ``SemanticCache`` / ``KeyStore``.

Usage::

    queue = NotificationQueue(db_path="./data/notifications.db", ttl_seconds=900)

    # On circuit-open exception
    queue.enqueue(signal_id="abc", payload={"signal": ..., "narrative": ...})

    # Periodically (e.g., once a minute, or on circuit-close hook)
    queue.replay(notify_fn=lambda payload: notifier.send_signal(...))
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class NotificationQueue:
    """SQLite-backed FIFO queue with TTL-bounded replay."""

    SCHEMA_VERSION = 1
    DEFAULT_TTL_SECONDS = 15 * 60  # 15 minutes

    def __init__(
        self,
        db_path: str = "./data/notifications.db",
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ):
        self._db_path = Path(db_path)
        self._ttl = int(ttl_seconds)
        self._lock = threading.RLock()
        self._enqueued = 0
        self._replayed = 0
        self._dropped_expired = 0

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info(
            "NotificationQueue initialised at %s (TTL=%ds)",
            self._db_path, self._ttl,
        )

    # ------------------------------------------------------------------ #
    # SQLite setup
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
            finally:
                conn.close()

    def _migrate(self, conn: sqlite3.Connection, from_v: int) -> None:
        if from_v < 1:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS pending_notifications (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id   TEXT    NOT NULL,
                    payload     TEXT    NOT NULL,
                    enqueued_at REAL    NOT NULL,
                    attempts    INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(signal_id)
                );
                CREATE INDEX IF NOT EXISTS idx_notif_enqueued
                    ON pending_notifications(enqueued_at);
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def enqueue(self, signal_id: str, payload: Dict[str, Any]) -> bool:
        """Queue a notification. Idempotent on signal_id (UNIQUE constraint).

        Returns True if newly enqueued, False if already pending.
        """
        if not signal_id:
            return False
        now = time.time()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO pending_notifications "
                    "(signal_id, payload, enqueued_at, attempts) "
                    "VALUES (?, ?, ?, 0)",
                    (signal_id, json.dumps(payload, default=str), now),
                )
                inserted = cur.rowcount > 0
            finally:
                conn.close()
        if inserted:
            self._enqueued += 1
            logger.info("Notification enqueued: signal_id=%s", signal_id)
        return inserted

    def replay(
        self,
        notify_fn: Callable[[Dict[str, Any]], None],
        max_per_run: int = 50,
    ) -> Dict[str, int]:
        """Replay pending notifications via ``notify_fn``.

        - Drops entries older than ``ttl_seconds`` (counted as dropped_expired).
        - Calls ``notify_fn(payload)`` for each non-expired entry.
        - On success → row deleted.
        - On exception → row left in queue, ``attempts`` incremented.

        Returns ``{"replayed": N, "dropped_expired": M, "failed": F}``.
        """
        cutoff = time.time() - self._ttl
        out = {"replayed": 0, "dropped_expired": 0, "failed": 0}

        with self._lock:
            conn = self._get_connection()
            try:
                # Drop expired first
                cur = conn.execute(
                    "DELETE FROM pending_notifications WHERE enqueued_at < ?",
                    (cutoff,),
                )
                out["dropped_expired"] = cur.rowcount
                self._dropped_expired += cur.rowcount

                # Fetch fresh batch
                cur = conn.execute(
                    "SELECT id, signal_id, payload, attempts FROM pending_notifications "
                    "ORDER BY enqueued_at LIMIT ?",
                    (max_per_run,),
                )
                rows = cur.fetchall()
            finally:
                conn.close()

            for row in rows:
                try:
                    payload = json.loads(row["payload"])
                except json.JSONDecodeError:
                    logger.error("Corrupt payload for notification id=%s — dropping", row["id"])
                    self._delete(row["id"])
                    out["failed"] += 1
                    continue
                try:
                    notify_fn(payload)
                except Exception as exc:
                    logger.warning(
                        "Replay attempt %d for signal=%s failed: %s",
                        row["attempts"] + 1, row["signal_id"], exc,
                    )
                    self._increment_attempts(row["id"])
                    out["failed"] += 1
                    continue
                self._delete(row["id"])
                out["replayed"] += 1
                self._replayed += 1

        if out["replayed"] or out["dropped_expired"] or out["failed"]:
            logger.info(
                "NotificationQueue replay: replayed=%d expired=%d failed=%d",
                out["replayed"], out["dropped_expired"], out["failed"],
            )
        return out

    def _delete(self, row_id: int) -> None:
        conn = self._get_connection()
        try:
            conn.execute("DELETE FROM pending_notifications WHERE id = ?", (row_id,))
        finally:
            conn.close()

    def _increment_attempts(self, row_id: int) -> None:
        conn = self._get_connection()
        try:
            conn.execute(
                "UPDATE pending_notifications SET attempts = attempts + 1 WHERE id = ?",
                (row_id,),
            )
        finally:
            conn.close()

    def cleanup_expired(self) -> int:
        """Delete TTL-expired entries. Returns count deleted."""
        cutoff = time.time() - self._ttl
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM pending_notifications WHERE enqueued_at < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
            finally:
                conn.close()
        if deleted > 0:
            self._dropped_expired += deleted
            logger.info("NotificationQueue cleanup: removed %d expired entries", deleted)
        return deleted

    def size(self) -> int:
        """Number of pending notifications."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT COUNT(*) AS cnt FROM pending_notifications")
                return cur.fetchone()["cnt"]
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Cumulative counters + current depth."""
        return {
            "enqueued_total": self._enqueued,
            "replayed_total": self._replayed,
            "dropped_expired_total": self._dropped_expired,
            "current_depth": self.size(),
            "ttl_seconds": self._ttl,
        }

    def clear(self) -> int:
        """Remove all pending notifications. Returns count deleted. Use with care."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("DELETE FROM pending_notifications")
                return cur.rowcount
            finally:
                conn.close()
