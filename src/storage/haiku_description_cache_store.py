"""SQLite-backed cache for Haiku-generated MarketReading descriptions.

Lives in the same SQLite file as ``MarketReadingsStore`` (default
``./data/market_readings.db``) so it inherits the Fly.io volume
persistence configured in Chantier 1. Owns the ``haiku_description_cache``
table only — does not touch any table owned by ``MarketReadingsStore``.

Same WAL + RLock pattern as the Chantier 1 stores.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_iso(ts: Optional[datetime] = None) -> str:
    ts = ts if ts is not None else datetime.now(timezone.utc)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    else:
        ts = ts.astimezone(timezone.utc)
    return ts.isoformat(timespec="seconds").replace("+00:00", "Z")


class HaikuDescriptionCacheStore:
    """Persistent (hash_key → description) cache for Haiku-generated text.

    Path resolution priority:
      1. Explicit ``db_path`` argument
      2. ``MARKET_READINGS_DB_PATH`` env var (shared with MarketReadingsStore
         so the Haiku cache lives alongside the readings, both inheriting
         the Fly.io volume mount)
      3. ``DEFAULT_DB_PATH`` (``./data/market_readings.db``)
    """

    DEFAULT_DB_PATH = "./data/market_readings.db"
    DB_PATH_ENV_VAR = "MARKET_READINGS_DB_PATH"
    TABLE_NAME = "haiku_description_cache"

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = self._resolve_db_path(db_path)
        self._lock = threading.RLock()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_table()
        logger.info("HaikuDescriptionCacheStore initialised at %s", self._db_path)

    @classmethod
    def _resolve_db_path(cls, db_path: Optional[str]) -> Path:
        if db_path:
            return Path(db_path)
        env_val = os.environ.get(cls.DB_PATH_ENV_VAR)
        if env_val:
            return Path(env_val)
        return Path(cls.DEFAULT_DB_PATH)

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_table(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {self.TABLE_NAME} (
                        hash_key TEXT PRIMARY KEY,
                        description TEXT NOT NULL,
                        source TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
            finally:
                conn.close()

    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def get(self, hash_key: str) -> Optional[Tuple[str, str]]:
        """Return ``(description, source)`` for ``hash_key`` or None on miss."""
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(
                    f"SELECT description, source FROM {self.TABLE_NAME} WHERE hash_key = ?",
                    (hash_key,),
                )
                row = cur.fetchone()
                if row is None:
                    return None
                return row["description"], row["source"]
            finally:
                conn.close()

    def put(self, hash_key: str, description: str, source: str) -> None:
        """Insert or replace ``(description, source)`` for ``hash_key``."""
        with self._lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {self.TABLE_NAME}
                        (hash_key, description, source, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (hash_key, description, source, _utc_iso()),
                )
            finally:
                conn.close()

    def size(self) -> int:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(f"SELECT COUNT(*) AS n FROM {self.TABLE_NAME}")
                return int(cur.fetchone()["n"])
            finally:
                conn.close()


__all__ = ["HaikuDescriptionCacheStore"]
