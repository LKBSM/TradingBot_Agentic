"""Subscription tier management with per-tier rate limiting.

Tiers:
  FREE         $0     10 API calls/day   Visual only
  ANALYST      $49    100 calls/day      Haiku validation + Telegram
  STRATEGIST   $99    500 calls/day      Full Sonnet narrative + Telegram
  INSTITUTIONAL $149  2000 calls/day     Full + chat + webhooks
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# TIER DEFINITIONS
# =============================================================================

class UserTier(str, Enum):
    FREE = "FREE"
    ANALYST = "ANALYST"
    STRATEGIST = "STRATEGIST"
    INSTITUTIONAL = "INSTITUTIONAL"


TIER_CONFIG: Dict[str, Dict[str, Any]] = {
    UserTier.FREE: {
        "price_usd": 0,
        "api_calls_per_day": 10,
        "narrative_depth": "VISUAL",
        "telegram": False,
        "webhooks": False,
        "chat": False,
    },
    UserTier.ANALYST: {
        "price_usd": 49,
        "api_calls_per_day": 100,
        "narrative_depth": "VALIDATOR",
        "telegram": True,
        "webhooks": False,
        "chat": False,
    },
    UserTier.STRATEGIST: {
        "price_usd": 99,
        "api_calls_per_day": 500,
        "narrative_depth": "NARRATOR",
        "telegram": True,
        "webhooks": False,
        "chat": False,
    },
    UserTier.INSTITUTIONAL: {
        "price_usd": 149,
        "api_calls_per_day": 2000,
        "narrative_depth": "NARRATOR",
        "telegram": True,
        "webhooks": True,
        "chat": True,
    },
}


# =============================================================================
# USER TIER MANAGER
# =============================================================================

class UserTierManager:
    """
    SQLite-backed user tier + rate limiting.

    Links API keys (from KeyStore) to user profiles with subscription tiers.
    Same WAL pattern as KeyStore / SignalStore.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str = "./data/users.db"):
        self._db_path = Path(db_path)
        self._lock = threading.RLock()

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("UserTierManager initialised at %s", self._db_path)

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
                CREATE TABLE IF NOT EXISTS users (
                    user_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    email            TEXT NOT NULL UNIQUE,
                    tier             TEXT NOT NULL DEFAULT 'FREE',
                    api_key_id       INTEGER,
                    telegram_chat_id TEXT,
                    stripe_customer_id TEXT,
                    subscription_expires TEXT,
                    created_at       TEXT NOT NULL,
                    updated_at       TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id   INTEGER NOT NULL,
                    endpoint  TEXT NOT NULL,
                    timestamp REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_usage_user_ts
                    ON usage_log(user_id, timestamp);
                CREATE INDEX IF NOT EXISTS idx_users_api_key
                    ON users(api_key_id);
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    # ------------------------------------------------------------------ #
    # USER CRUD
    # ------------------------------------------------------------------ #

    def create_user(
        self,
        email: str,
        tier: UserTier = UserTier.FREE,
        api_key_id: Optional[int] = None,
        telegram_chat_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new user. Returns user dict."""
        now = datetime.now(tz=None).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "INSERT INTO users (email, tier, api_key_id, telegram_chat_id, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                    (email, tier.value, api_key_id, telegram_chat_id, now, now),
                )
                return {
                    "user_id": cur.lastrowid,
                    "email": email,
                    "tier": tier.value,
                    "api_key_id": api_key_id,
                }
            finally:
                conn.close()

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM users WHERE user_id = ?", (user_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def get_user_by_api_key(self, api_key_id: int) -> Optional[Dict[str, Any]]:
        """Look up user by their linked API key ID."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT * FROM users WHERE api_key_id = ?", (api_key_id,)
                )
                row = cur.fetchone()
                return dict(row) if row else None
            finally:
                conn.close()

    def update_tier(self, user_id: int, tier: UserTier) -> bool:
        """Update a user's subscription tier."""
        now = datetime.now(tz=None).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE users SET tier = ?, updated_at = ? WHERE user_id = ?",
                    (tier.value, now, user_id),
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    def link_api_key(self, user_id: int, api_key_id: int) -> bool:
        """Link an API key to a user."""
        now = datetime.now(tz=None).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE users SET api_key_id = ?, updated_at = ? WHERE user_id = ?",
                    (api_key_id, now, user_id),
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    def link_telegram(self, user_id: int, chat_id: str) -> bool:
        """Link a Telegram chat ID to a user."""
        now = datetime.now(tz=None).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "UPDATE users SET telegram_chat_id = ?, updated_at = ? WHERE user_id = ?",
                    (chat_id, now, user_id),
                )
                return cur.rowcount > 0
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # RATE LIMITING
    # ------------------------------------------------------------------ #

    def check_rate_limit(self, user_id: int) -> bool:
        """
        Check if user is under their daily API call limit.

        Returns True if allowed, False if limit exceeded.
        """
        user = self.get_user(user_id)
        if user is None:
            return False

        tier = UserTier(user["tier"])
        limit = TIER_CONFIG[tier]["api_calls_per_day"]

        # Count calls in the last 24 hours
        cutoff = time.time() - 86400
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM usage_log "
                    "WHERE user_id = ? AND timestamp >= ?",
                    (user_id, cutoff),
                )
                count = cur.fetchone()["cnt"]
                return count < limit
            finally:
                conn.close()

    def record_usage(self, user_id: int, endpoint: str) -> None:
        """Record an API call for rate limiting."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT INTO usage_log (user_id, endpoint, timestamp) "
                    "VALUES (?, ?, ?)",
                    (user_id, endpoint, time.time()),
                )
            finally:
                conn.close()

    def get_daily_usage(self, user_id: int) -> int:
        """Get number of API calls in the last 24 hours."""
        cutoff = time.time() - 86400
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM usage_log "
                    "WHERE user_id = ? AND timestamp >= ?",
                    (user_id, cutoff),
                )
                return cur.fetchone()["cnt"]
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # TIER QUERIES
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_tier_config(tier: UserTier) -> Dict[str, Any]:
        """Get configuration for a tier."""
        return TIER_CONFIG.get(tier, TIER_CONFIG[UserTier.FREE]).copy()

    @staticmethod
    def get_narrative_tier(user_tier: UserTier) -> str:
        """Map user tier to narrative depth."""
        return TIER_CONFIG.get(user_tier, TIER_CONFIG[UserTier.FREE])["narrative_depth"]

    def list_users(self, tier: Optional[UserTier] = None) -> List[Dict[str, Any]]:
        """List all users, optionally filtered by tier."""
        with self._lock:
            conn = self._get_connection()
            try:
                if tier:
                    cur = conn.execute(
                        "SELECT * FROM users WHERE tier = ? ORDER BY user_id",
                        (tier.value,),
                    )
                else:
                    cur = conn.execute("SELECT * FROM users ORDER BY user_id")
                return [dict(r) for r in cur.fetchall()]
            finally:
                conn.close()
