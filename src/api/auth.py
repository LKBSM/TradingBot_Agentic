"""API key authentication and admin HMAC verification."""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)


# =============================================================================
# KEY STORE — SQLite-backed API key management
# =============================================================================

class KeyStore:
    """
    Thread-safe API key store with SQLite WAL persistence.

    Same pattern as ``SignalStore`` / ``KillSwitchStore``.
    Keys are hashed with SHA-256 before storage — the raw key
    is shown only once at creation time.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/api_keys.db"):
        self._db_path = Path(db_path)
        self._lock = threading.RLock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("KeyStore initialised at %s", self._db_path)

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
                CREATE TABLE IF NOT EXISTS api_keys (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_hash  TEXT    NOT NULL UNIQUE,
                    label     TEXT    NOT NULL,
                    created_at TEXT   NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1
                );
                CREATE TABLE IF NOT EXISTS api_usage (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id    INTEGER NOT NULL,
                    endpoint  TEXT    NOT NULL,
                    timestamp REAL    NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_key_ts
                    ON api_usage(key_id, timestamp);
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _hash_key(raw_key: str) -> str:
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @staticmethod
    def _generate_raw_key() -> str:
        return "sk_" + secrets.token_hex(32)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #

    def create_key(self, label: str) -> Dict[str, Any]:
        """Create a new API key. Returns the raw key (shown once)."""
        raw_key = self._generate_raw_key()
        key_hash = self._hash_key(raw_key)
        now = time.strftime("%Y-%m-%dT%H:%M:%S")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "INSERT INTO api_keys (key_hash, label, created_at, is_active) "
                    "VALUES (?, ?, ?, 1)",
                    (key_hash, label, now),
                )
                key_id = cur.lastrowid
            finally:
                conn.close()

        return {"key_id": key_id, "api_key": raw_key, "label": label}

    def verify_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Verify a raw API key. Returns subscriber dict or None."""
        key_hash = self._hash_key(raw_key)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id, label, created_at, is_active FROM api_keys "
                    "WHERE key_hash = ?",
                    (key_hash,),
                )
                row = cur.fetchone()
            finally:
                conn.close()

        if row is None:
            return None
        if not row["is_active"]:
            return None

        return {
            "key_id": row["id"],
            "label": row["label"],
            "created_at": row["created_at"],
        }

    def revoke_key(self, key_id: int) -> bool:
        """Soft-delete a key (is_active=0). Returns True if found."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE api_keys SET is_active = 0 WHERE id = ?",
                    (key_id,),
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys — metadata only, never exposes hashes."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id, label, created_at, is_active FROM api_keys "
                    "ORDER BY id"
                )
                return [
                    {
                        "key_id": r["id"],
                        "label": r["label"],
                        "created_at": r["created_at"],
                        "is_active": bool(r["is_active"]),
                    }
                    for r in cur.fetchall()
                ]
            finally:
                conn.close()

    def record_usage(self, key_id: int, endpoint: str) -> None:
        """Insert a usage row."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO api_usage (key_id, endpoint, timestamp) "
                    "VALUES (?, ?, ?)",
                    (key_id, endpoint, time.time()),
                )
            finally:
                conn.close()

    def get_usage(self, key_id: int, days: int = 30) -> List[Dict[str, Any]]:
        """Usage stats grouped by endpoint for the last N days."""
        cutoff = time.time() - (days * 86400)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT endpoint, COUNT(*) as count "
                    "FROM api_usage "
                    "WHERE key_id = ? AND timestamp >= ? "
                    "GROUP BY endpoint "
                    "ORDER BY count DESC",
                    (key_id, cutoff),
                )
                return [
                    {"endpoint": r["endpoint"], "count": r["count"]}
                    for r in cur.fetchall()
                ]
            finally:
                conn.close()

    def check_rate_limit(self, key_id: int, max_per_minute: int = 100) -> bool:
        """Return True if under rate limit, False if exceeded."""
        cutoff = time.time() - 60.0

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) as cnt FROM api_usage "
                    "WHERE key_id = ? AND timestamp >= ?",
                    (key_id, cutoff),
                )
                count = cur.fetchone()["cnt"]
                return count < max_per_minute
            finally:
                conn.close()


# =============================================================================
# FastAPI DEPENDENCIES
# =============================================================================

async def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None),
) -> Dict[str, Any]:
    """
    FastAPI dependency — validates X-API-Key header.

    Returns subscriber dict on success.
    Raises 401 (invalid/revoked), 429 (rate-limited), 503 (no KeyStore).
    """
    key_store: Optional[KeyStore] = getattr(
        request.app.state.app_state, "key_store", None
    )

    if key_store is None:
        raise HTTPException(status_code=503, detail="Authentication service unavailable")

    if not x_api_key:
        raise HTTPException(status_code=401, detail="Missing X-API-Key header")

    subscriber = key_store.verify_key(x_api_key)
    if subscriber is None:
        raise HTTPException(status_code=401, detail="Invalid or revoked API key")

    if not key_store.check_rate_limit(subscriber["key_id"]):
        raise HTTPException(status_code=429, detail="Rate limit exceeded (100 req/min)")

    key_store.record_usage(subscriber["key_id"], request.url.path)
    return subscriber


async def require_admin(
    request: Request,
    x_admin_signature: Optional[str] = Header(None),
    x_admin_timestamp: Optional[str] = Header(None),
) -> bool:
    """
    FastAPI dependency — validates HMAC-signed admin requests.

    Reads X-Admin-Signature + X-Admin-Timestamp headers.
    5-minute replay protection window.
    """
    hmac_manager = getattr(request.app.state.app_state, "hmac_manager", None)

    if hmac_manager is None:
        raise HTTPException(status_code=503, detail="Admin auth service unavailable")

    if not x_admin_signature or not x_admin_timestamp:
        raise HTTPException(status_code=401, detail="Missing admin signature headers")

    # Replay protection — 5-minute window
    try:
        ts = float(x_admin_timestamp)
    except (ValueError, TypeError):
        raise HTTPException(status_code=401, detail="Invalid timestamp")

    if abs(time.time() - ts) > 300:
        raise HTTPException(status_code=401, detail="Timestamp expired (5-min window)")

    # Verify HMAC: sign the timestamp bytes
    data = x_admin_timestamp.encode()
    is_valid = hmac_manager.verify(data, x_admin_signature)

    if not is_valid:
        raise HTTPException(status_code=401, detail="Invalid admin signature")

    return True
