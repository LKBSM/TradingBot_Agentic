"""API key authentication and admin HMAC verification."""

from __future__ import annotations

import hashlib
import logging
import os
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Header, HTTPException, Request

logger = logging.getLogger(__name__)

# Auth gate. SENTINEL_TESTING_MODE=1 bypasses API keys and grants
# INSTITUTIONAL-level access for personal testing/dev. Default is now "0"
# (fail-closed) — a deployment without explicit env var configuration
# will require valid API keys, eliminating the silent open-prod risk.
# Flip to "1" only on local dev or in CI where auth bypass is intentional.
TESTING_MODE = os.environ.get("SENTINEL_TESTING_MODE", "0") == "1"
if TESTING_MODE:
    logger.warning(
        "SENTINEL_TESTING_MODE=1 — auth bypassed, all endpoints grant "
        "INSTITUTIONAL access. DO NOT enable in production."
    )


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

    SCHEMA_VERSION = 2  # v2 adds superseded_at for SECURITY-2B.2 key rotation

    # In-memory verify_key cache (key_hash → (subscriber_dict, expiry_ts)).
    # Eliminates ~70% of SQL SELECT api_keys hits on every authenticated
    # request (eval 11). Invalidated on create_key / revoke_key. 60s TTL
    # keeps revocation lag bounded; tune lower for stricter security.
    _CACHE_TTL_S = 60.0

    def __init__(self, db_path: str = "./data/api_keys.db"):
        self._db_path = Path(db_path)
        self._lock = threading.RLock()
        self._verify_cache: Dict[str, tuple[Optional[Dict[str, Any]], float]] = {}

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("KeyStore initialised at %s", self._db_path)

    def _cache_invalidate(self) -> None:
        """Drop the entire verify cache. Called on key create/revoke."""
        with self._lock:
            self._verify_cache.clear()

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
        if from_v < 2:
            # SECURITY-2B.2: rotation flow — superseded_at is the wall-clock
            # Unix timestamp after which a superseded key STOPS verifying.
            # NULL = key has never been rotated. Adding via ALTER preserves
            # every existing row.
            cur = conn.execute(
                "SELECT 1 FROM pragma_table_info('api_keys') "
                "WHERE name = 'superseded_at'"
            )
            if cur.fetchone() is None:
                conn.execute(
                    "ALTER TABLE api_keys ADD COLUMN superseded_at REAL"
                )
            cur = conn.execute(
                "SELECT 1 FROM pragma_table_info('api_keys') "
                "WHERE name = 'superseded_by'"
            )
            if cur.fetchone() is None:
                conn.execute(
                    "ALTER TABLE api_keys ADD COLUMN superseded_by INTEGER"
                )
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
        self._cache_invalidate()
        return {"key_id": key_id, "api_key": raw_key, "label": label}

    def verify_key(self, raw_key: str) -> Optional[Dict[str, Any]]:
        """Verify a raw API key. Returns subscriber dict or None.

        Cached for ``_CACHE_TTL_S`` seconds (60s default) to avoid hitting
        SQLite on every authenticated request. Cache is invalidated on
        create_key / revoke_key, so revocation lag is bounded by TTL.
        """
        key_hash = self._hash_key(raw_key)
        now = time.time()

        # Cache lookup
        with self._lock:
            cached = self._verify_cache.get(key_hash)
            if cached is not None:
                subscriber, expiry = cached
                if now < expiry:
                    return subscriber
                # expired — fall through to DB
                self._verify_cache.pop(key_hash, None)

            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id, label, created_at, is_active, "
                    "       superseded_at, superseded_by "
                    "FROM api_keys WHERE key_hash = ?",
                    (key_hash,),
                )
                row = cur.fetchone()
            finally:
                conn.close()

            subscriber: Optional[Dict[str, Any]] = None
            if row is not None and row["is_active"]:
                # SECURITY-2B.2: a rotated key keeps verifying until the
                # grace deadline so brokers can swap key material without
                # a hard cutover. After the deadline it's effectively
                # revoked even if is_active remained 1.
                superseded_at = row["superseded_at"]
                if superseded_at is None or now < superseded_at:
                    subscriber = {
                        "key_id": row["id"],
                        "label": row["label"],
                        "created_at": row["created_at"],
                    }
                    if superseded_at is not None:
                        subscriber["superseded_at"] = superseded_at
                        subscriber["superseded_by"] = row["superseded_by"]
            self._verify_cache[key_hash] = (subscriber, now + self._CACHE_TTL_S)
            return subscriber

    def rotate_key(
        self, key_id: int, *, grace_seconds: float = 86400.0
    ) -> Optional[Dict[str, Any]]:
        """Issue a new key that supersedes ``key_id``.

        SECURITY-2B.2 — graceful key rotation:

        - mint a fresh raw key + insert a new row with the same label
          plus a "(rotated YYYY-MM-DD)" suffix,
        - mark the old key with ``superseded_at = now + grace_seconds``
          and ``superseded_by = new_key_id``,
        - both keys verify successfully until the grace window expires;
          after that the old key stops authenticating and only the new
          one works.

        Returns ``{old_key_id, new_key_id, new_api_key, superseded_at,
        label}`` on success, ``None`` if the old key doesn't exist
        (404 in the calling route).

        ``grace_seconds`` must be in [0, 30 days]. 24h is a reasonable
        default for brokers running daily deploys; 0 = immediate
        revocation (handy for "the key just leaked" emergency
        rotations); 30 days is the safety ceiling so a forgotten
        rotation can't leave an old key valid forever.
        """
        if grace_seconds < 0:
            raise ValueError("grace_seconds must be >= 0")
        if grace_seconds > 30 * 86400:
            raise ValueError("grace_seconds capped at 30 days")

        now = time.time()
        deadline = now + grace_seconds
        new_raw = self._generate_raw_key()
        new_hash = self._hash_key(new_raw)
        created_at = time.strftime("%Y-%m-%dT%H:%M:%S")

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id, label, is_active, superseded_at "
                    "FROM api_keys WHERE id = ?",
                    (key_id,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                if not row["is_active"]:
                    raise ValueError(
                        f"key {key_id} is already revoked; cannot rotate"
                    )
                # Don't double-rotate: if the key is already inside its
                # grace window, refuse rather than silently extending.
                if row["superseded_at"] is not None:
                    raise ValueError(
                        f"key {key_id} has already been rotated (grace "
                        f"window in progress)"
                    )

                old_label = row["label"]
                today = time.strftime("%Y-%m-%d")
                new_label = f"{old_label} (rotated {today})"

                cur = conn.execute(
                    "INSERT INTO api_keys "
                    "(key_hash, label, created_at, is_active) "
                    "VALUES (?, ?, ?, 1)",
                    (new_hash, new_label, created_at),
                )
                new_id = cur.lastrowid

                conn.execute(
                    "UPDATE api_keys "
                    "SET superseded_at = ?, superseded_by = ? "
                    "WHERE id = ?",
                    (deadline, new_id, key_id),
                )
            finally:
                conn.close()
        self._cache_invalidate()
        return {
            "old_key_id": key_id,
            "new_key_id": new_id,
            "new_api_key": new_raw,
            "superseded_at": deadline,
            "grace_seconds": grace_seconds,
            "label": new_label,
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
                found = cur.rowcount > 0
            finally:
                conn.close()
        self._cache_invalidate()
        return found

    def list_keys(self) -> List[Dict[str, Any]]:
        """List all keys — metadata only, never exposes hashes."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT id, label, created_at, is_active, "
                    "       superseded_at, superseded_by "
                    "FROM api_keys ORDER BY id"
                )
                return [
                    {
                        "key_id": r["id"],
                        "label": r["label"],
                        "created_at": r["created_at"],
                        "is_active": bool(r["is_active"]),
                        "superseded_at": r["superseded_at"],
                        "superseded_by": r["superseded_by"],
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

    In TESTING_MODE, skips auth and returns full-access subscriber.
    """
    if TESTING_MODE:
        subscriber = {
            "key_id": 0,
            "label": "testing",
            "tier": "INSTITUTIONAL",
            "user_id": 0,
            "testing_mode": True,
            "api_key": "testing",
        }
        # Surface for downstream middleware (access log, rate-limit headers)
        request.state.subscriber = subscriber
        return subscriber

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

    # Enrich with tier info if UserTierManager is available
    tier_manager = getattr(request.app.state.app_state, "tier_manager", None)
    if tier_manager is not None:
        user = tier_manager.get_user_by_api_key(subscriber["key_id"])
        if user is not None:
            subscriber["user_id"] = user["user_id"]
            subscriber["tier"] = user["tier"]
            subscriber["telegram_chat_id"] = user.get("telegram_chat_id")
            # Daily quota gate (eval 11 finding: tier_manager.check_rate_limit
            # was implemented but never called — FREE could spend 144 000
            # calls/day via the per-minute KeyStore ceiling). This wires the
            # tier-quotas advertised on the pricing grid.
            try:
                if not tier_manager.check_rate_limit(user["user_id"]):
                    raise HTTPException(
                        status_code=429,
                        detail=f"Daily quota exceeded for tier {user['tier']}",
                    )
                tier_manager.record_usage(user["user_id"], request.url.path)
            except HTTPException:
                raise
            except Exception as exc:
                # Tier check failure must NOT crash auth — log + continue.
                logger.warning("tier rate-limit check failed for user %s: %s", user["user_id"], exc)
        else:
            subscriber["tier"] = "FREE"
    else:
        subscriber["tier"] = "FREE"

    # Keep the raw key around for downstream middleware that needs to
    # key into per-tier counters (rate-limit headers, etc). It's already
    # in memory at this point; stashing it on the request lets us avoid
    # re-deriving from the X-API-Key header in three different places.
    subscriber["api_key"] = x_api_key
    request.state.subscriber = subscriber
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
