"""Tests for TelegramLangStore + TelegramNotifier per-chat language routing.

W3 of P29 compliance — replaces ``Accept-Language`` (which Telegram
bots cannot read) with the Telegram ``User.language_code`` payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

from src.delivery.telegram_lang_store import TelegramLangStore
from src.delivery.telegram_notifier import TelegramNotifier


# ─── Mock signal (small) ───────────────────────────────────────────────────


@dataclass
class _MockSignal:
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    confluence_score: float = 75.0
    tier: str = "STANDARD"
    entry_price: float = 2400.0
    stop_loss: float = 2380.0
    take_profit: float = 2440.0
    rr_ratio: float = 2.0
    atr: float = 10.0
    components: list = None

    def __post_init__(self):
        if self.components is None:
            self.components = []


# ─── Store-level tests ─────────────────────────────────────────────────────


class TestTelegramLangStore:
    def test_set_and_get_round_trip(self):
        store = TelegramLangStore(":memory:")
        store.set("12345", "fr")
        assert store.get("12345") == "fr"

    def test_unknown_chat_returns_none(self):
        store = TelegramLangStore(":memory:")
        assert store.get("99999") is None

    def test_full_locale_is_normalised_to_two_letters(self):
        store = TelegramLangStore(":memory:")
        store.set("12345", "fr-FR")
        assert store.get("12345") == "fr"

    def test_unsupported_language_is_dropped(self):
        # Telegram clients can carry "ja", "zh", "ar", etc. We only render
        # disclaimers in fr/en/de/es so anything else is stored as a no-op
        # and the chat falls back to the notifier's default_lang.
        store = TelegramLangStore(":memory:")
        store.set("12345", "ja")
        assert store.get("12345") is None

    def test_set_none_removes_entry(self):
        store = TelegramLangStore(":memory:")
        store.set("12345", "fr")
        assert store.get("12345") == "fr"
        store.set("12345", None)
        assert store.get("12345") is None

    def test_chat_id_is_stringified(self):
        store = TelegramLangStore(":memory:")
        store.set(12345, "de")          # type: ignore[arg-type]
        assert store.get("12345") == "de"
        assert store.get(12345) == "de"  # type: ignore[arg-type]

    def test_persistence_across_instances(self, tmp_path):
        # A second instance opened on the same DB should see the data.
        db = tmp_path / "lang.db"
        s1 = TelegramLangStore(str(db))
        s1.set("1", "es")
        s1.set("2", "fr")

        s2 = TelegramLangStore(str(db))
        assert s2.get("1") == "es"
        assert s2.get("2") == "fr"
        assert len(s2) == 2

    def test_upsert_overwrites_existing(self):
        store = TelegramLangStore(":memory:")
        store.set("1", "fr")
        store.set("1", "de")
        assert store.get("1") == "de"
        assert len(store) == 1

    def test_supported_languages_match_disclaimers(self):
        from src.api.disclaimers import SUPPORTED_LANGS

        store = TelegramLangStore(":memory:")
        for lang in SUPPORTED_LANGS:
            store.set(f"chat-{lang}", lang)
            assert store.get(f"chat-{lang}") == lang


# ─── Notifier resolution logic ─────────────────────────────────────────────


class TestNotifierLangResolution:
    def test_explicit_lang_wins_over_store(self):
        store = TelegramLangStore(":memory:")
        store.set("42", "fr")
        notifier = TelegramNotifier(lang_store=store, default_lang="en")
        assert notifier._resolve_lang("42", "de") == "de"

    def test_store_used_when_lang_omitted(self):
        store = TelegramLangStore(":memory:")
        store.set("42", "fr")
        notifier = TelegramNotifier(lang_store=store, default_lang="en")
        assert notifier._resolve_lang("42", None) == "fr"

    def test_default_used_when_chat_unknown(self):
        store = TelegramLangStore(":memory:")
        notifier = TelegramNotifier(lang_store=store, default_lang="en")
        assert notifier._resolve_lang("9999", None) == "en"

    def test_default_used_when_no_store(self):
        notifier = TelegramNotifier(default_lang="es")
        assert notifier._resolve_lang("42", None) == "es"

    def test_send_signal_uses_resolved_lang(self):
        # Wire: store says chat 7 wants German. send_signal called with no
        # explicit lang. Resulting message should embed the German footer.
        store = TelegramLangStore(":memory:")
        store.set("7", "de")
        notifier = TelegramNotifier(lang_store=store, default_lang="en")

        # Fake bot — capture the text argument.
        captured = {}

        class _FakeBot:
            def send_message(self, chat_id, text, parse_mode, **kw):
                captured["chat_id"] = chat_id
                captured["text"] = text

        notifier._bot = _FakeBot()

        ok = notifier.send_signal(_MockSignal(), chat_id="7", tier="FREE")
        assert ok
        # German footer fragment.
        assert "Anlageberatung" in captured["text"]

    def test_send_signal_explicit_lang_overrides_store(self):
        store = TelegramLangStore(":memory:")
        store.set("7", "de")
        notifier = TelegramNotifier(lang_store=store, default_lang="en")

        captured = {}

        class _FakeBot:
            def send_message(self, chat_id, text, parse_mode, **kw):
                captured["text"] = text

        notifier._bot = _FakeBot()

        ok = notifier.send_signal(_MockSignal(), chat_id="7", tier="FREE", lang="es")
        assert ok
        assert "asesoramiento" in captured["text"].lower()

    def test_send_signal_falls_back_to_default_when_unknown(self):
        store = TelegramLangStore(":memory:")
        notifier = TelegramNotifier(lang_store=store, default_lang="fr")

        captured = {}

        class _FakeBot:
            def send_message(self, chat_id, text, parse_mode, **kw):
                captured["text"] = text

        notifier._bot = _FakeBot()
        ok = notifier.send_signal(_MockSignal(), chat_id="unknown-chat", tier="FREE")
        assert ok
        # French footer fragment.
        assert "conseil en investissement" in captured["text"]


# ─── send_to_multiple respects per-recipient overrides + store ─────────────


class TestSendToMultiple:
    def test_per_recipient_lang_wins_over_store(self):
        store = TelegramLangStore(":memory:")
        store.set("1", "de")
        store.set("2", "fr")

        notifier = TelegramNotifier(lang_store=store, default_lang="en")

        sent = []

        class _FakeBot:
            def send_message(self, chat_id, text, parse_mode, **kw):
                sent.append((chat_id, text))

        notifier._bot = _FakeBot()

        recipients = [
            {"chat_id": "1", "tier": "FREE"},                    # store -> de
            {"chat_id": "2", "tier": "FREE", "lang": "es"},       # explicit -> es
            {"chat_id": "3", "tier": "FREE"},                    # default -> en
        ]
        n = notifier.send_to_multiple(_MockSignal(), None, recipients)
        assert n == 3

        assert "Anlageberatung" in sent[0][1]
        assert "asesoramiento" in sent[1][1].lower()
        assert "investment advice" in sent[2][1].lower()
