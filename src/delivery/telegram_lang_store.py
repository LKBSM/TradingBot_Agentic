"""Per-chat language store for Telegram delivery — P29 W3 compliance.

Telegram's ``User`` payload carries a ``language_code`` field (ISO-639-1
2-letter code, sometimes followed by a region: ``en``, ``en-US``, ``fr``,
``de``, ``es``…). It is the most reliable signal we have of which
disclaimer language to render in a Telegram push, since Telegram bots
do not carry an ``Accept-Language`` HTTP header.

This store keeps a tiny ``chat_id -> language`` mapping behind a SQLite
file and an in-memory cache. The bot's ``/start`` handler is expected to
populate it once on first contact.

Usage::

    store = TelegramLangStore(db_path="data/telegram_lang.db")

    # in the bot /start handler:
    user = update.effective_user
    store.set(str(update.effective_chat.id), user.language_code)

    # in the notifier:
    lang = store.get(chat_id) or "en"
    TelegramNotifier(...).send_signal(signal, lang=lang)

The store is intentionally minimal — no user model, no email, no PII.
Just the chat id and the language tag.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from typing import Dict, Optional

from src.api.disclaimers import SUPPORTED_LANGS

logger = logging.getLogger(__name__)


class TelegramLangStore:
    """Persistent ``chat_id -> language`` mapping with in-memory cache."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS telegram_lang (
        chat_id    TEXT PRIMARY KEY,
        language   TEXT NOT NULL,
        updated_at REAL NOT NULL
    );
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        self._lock = threading.RLock()
        self._cache: Dict[str, str] = {}

        # When using a file-based DB, ensure the parent directory exists.
        if db_path != ":memory:":
            parent = os.path.dirname(os.path.abspath(db_path))
            if parent:
                os.makedirs(parent, exist_ok=True)

        # We keep ONE persistent connection so ``:memory:`` works across
        # method calls (each ``sqlite3.connect(":memory:")`` opens a fresh
        # private DB; we cannot reopen and find our table). For file DBs the
        # single-connection approach is also fine — reads stay fast and the
        # lock around write methods serialises writers.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        self._init_db()
        self._reload_cache()

    # -- internal helpers ------------------------------------------------

    def _init_db(self) -> None:
        with self._lock:
            self._conn.executescript(self.SCHEMA)
            self._conn.commit()

    def _reload_cache(self) -> None:
        with self._lock:
            rows = self._conn.execute(
                "SELECT chat_id, language FROM telegram_lang"
            ).fetchall()
            self._cache = {row["chat_id"]: row["language"] for row in rows}

    @staticmethod
    def _normalise(language_code: Optional[str]) -> Optional[str]:
        """Return a 2-letter lowercase language code or None.

        Telegram emits codes like ``en``, ``en-US``, ``pt-br``, ``fr``…
        We only retain the primary subtag and only if it is one of the
        languages we render disclaimers for.
        """
        if not language_code:
            return None
        primary = language_code.strip().split("-")[0].lower()
        if primary in SUPPORTED_LANGS:
            return primary
        return None

    # -- public API ------------------------------------------------------

    def set(self, chat_id: str, language_code: Optional[str]) -> None:
        """Store the language for a chat. ``None`` / unsupported codes
        are accepted but stored as a deletion (so the resolver later
        falls back to the default)."""
        normalised = self._normalise(language_code)
        with self._lock:
            if normalised is None:
                self._conn.execute(
                    "DELETE FROM telegram_lang WHERE chat_id = ?",
                    (str(chat_id),),
                )
                self._cache.pop(str(chat_id), None)
            else:
                import time

                self._conn.execute(
                    "INSERT INTO telegram_lang (chat_id, language, updated_at) "
                    "VALUES (?, ?, ?) "
                    "ON CONFLICT(chat_id) DO UPDATE SET language = excluded.language, "
                    "updated_at = excluded.updated_at",
                    (str(chat_id), normalised, time.time()),
                )
                self._cache[str(chat_id)] = normalised
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
            except Exception:  # pragma: no cover — defensive
                pass

    def get(self, chat_id: Optional[str]) -> Optional[str]:
        """Return the stored language for ``chat_id`` or ``None``."""
        if chat_id is None:
            return None
        return self._cache.get(str(chat_id))

    def __len__(self) -> int:
        return len(self._cache)


__all__ = ["TelegramLangStore"]
