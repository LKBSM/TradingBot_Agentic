"""Hash-based response deduplication for LLM narrative responses.

If multiple users request the same signal (same asset + bar + SMC features),
serve one cached response instead of N API calls.

Cache key: SHA256(symbol + bar_ts + bos + fvg + ob + regime + news)[:16]
Storage: SQLite WAL (same pattern as signal_store.py)
TTL: 24 hours
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """
    SQLite-backed narrative response cache with TTL expiration.

    Same WAL-mode pattern as SignalStore / KeyStore.
    """

    SCHEMA_VERSION = 1

    def __init__(
        self,
        db_path: str = "./data/narrative_cache.db",
        ttl_seconds: int = 86400,  # 24 hours
    ):
        self._db_path = Path(db_path)
        self._ttl = ttl_seconds
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        logger.info("SemanticCache initialised at %s (TTL=%ds)", self._db_path, self._ttl)

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
                CREATE TABLE IF NOT EXISTS narrative_cache (
                    cache_key  TEXT PRIMARY KEY,
                    data_json  TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    hit_count  INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_cache_created
                    ON narrative_cache(created_at);
            """)
            conn.execute(
                "INSERT OR REPLACE INTO schema_version (version) VALUES (?)",
                (self.SCHEMA_VERSION,),
            )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_cache_key(signal: Any) -> str:
        """
        Generate deterministic cache key from signal attributes.

        Key = SHA256(symbol + bar_ts + bos + fvg + ob + regime_type + news_decision)[:16]
        """
        parts = [
            str(getattr(signal, "symbol", "")),
            str(getattr(signal, "bar_timestamp", "")),
        ]

        # Add component scores for determinism
        for comp in getattr(signal, "components", []):
            parts.append(f"{comp.name}={comp.weighted_score:.1f}")

        raw = "|".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Look up a cached narrative.

        Returns:
            Deserialized narrative dict if found and not expired, else None.
        """
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "SELECT data_json, created_at, hit_count FROM narrative_cache "
                    "WHERE cache_key = ?",
                    (cache_key,),
                )
                row = cur.fetchone()

                if row is None:
                    self._misses += 1
                    return None

                age = time.time() - row["created_at"]
                if age > self._ttl:
                    # Expired — delete and return miss
                    conn.execute(
                        "DELETE FROM narrative_cache WHERE cache_key = ?",
                        (cache_key,),
                    )
                    self._misses += 1
                    return None

                # Increment hit count
                conn.execute(
                    "UPDATE narrative_cache SET hit_count = hit_count + 1 "
                    "WHERE cache_key = ?",
                    (cache_key,),
                )
                self._hits += 1
                return json.loads(row["data_json"])
            finally:
                conn.close()

    def put(self, cache_key: str, narrative_data: Dict[str, Any]) -> None:
        """Store a narrative in the cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO narrative_cache "
                    "(cache_key, data_json, created_at, hit_count) "
                    "VALUES (?, ?, ?, 0)",
                    (cache_key, json.dumps(narrative_data), time.time()),
                )
            finally:
                conn.close()

    def cleanup_expired(self) -> int:
        """Delete all expired entries. Returns count deleted."""
        cutoff = time.time() - self._ttl
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    "DELETE FROM narrative_cache WHERE created_at < ?",
                    (cutoff,),
                )
                deleted = cur.rowcount
                if deleted > 0:
                    logger.info("SemanticCache: cleaned %d expired entries", deleted)
                return deleted
            finally:
                conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
            "total_lookups": total,
        }

    def size(self) -> int:
        """Number of entries in cache."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute("SELECT COUNT(*) AS cnt FROM narrative_cache")
                return cur.fetchone()["cnt"]
            finally:
                conn.close()
