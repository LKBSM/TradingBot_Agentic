"""DG-054 — Telegram retry + dedup tests.

The real ``python-telegram-bot`` v20 client is async; we stub it with a
synchronous mock that mimics the ``send_message`` signature + the
exception hierarchy we care about (``RetryAfter``, ``TimedOut``,
``NetworkError``). The notifier we ship presents a sync surface to its
single caller (the scanner thread), so this is faithful to production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.delivery.telegram_notifier import (
    TelegramNotifier,
    _DedupStore,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class _StubSignal:
    """Minimal subset of ConfluenceSignal used by format_signal_message + dedup."""
    signal_id: str
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    direction: str = "LONG"
    confluence_score: float = 70.0
    tier: str = "WEAK"
    entry_price: float = 2350.0
    stop_loss: float = 2340.0
    take_profit: float = 2370.0
    timestamp: Any = None


class RetryAfter(Exception):
    """Mimics ``telegram.error.RetryAfter`` from python-telegram-bot."""
    def __init__(self, retry_after: float):
        super().__init__(f"Flood control exceeded. Retry in {retry_after}")
        self.retry_after = retry_after


class TimedOut(Exception):
    """Mimics ``telegram.error.TimedOut``."""


def _make_notifier(bot_mock, *, max_retries: int = 3) -> TelegramNotifier:
    n = TelegramNotifier(
        bot_token="x",
        default_chat_id="default",
        max_retries=max_retries,
        base_backoff_s=0.001,
        max_backoff_s=0.01,
        sleep_fn=lambda _s: None,  # no real sleeping in tests
    )
    n._bot = bot_mock
    return n


# ---------------------------------------------------------------------------
# DedupStore unit tests
# ---------------------------------------------------------------------------

def test_dedup_first_use_returns_true():
    s = _DedupStore(ttl_seconds=60.0)
    assert s.check_and_record("chat-1", "sig-1") is True


def test_dedup_replay_returns_false():
    s = _DedupStore(ttl_seconds=60.0)
    s.check_and_record("chat-1", "sig-1")
    assert s.check_and_record("chat-1", "sig-1") is False


def test_dedup_distinct_chats_dont_collide():
    s = _DedupStore(ttl_seconds=60.0)
    assert s.check_and_record("chat-1", "sig-1") is True
    assert s.check_and_record("chat-2", "sig-1") is True


def test_dedup_distinct_signals_dont_collide():
    s = _DedupStore(ttl_seconds=60.0)
    assert s.check_and_record("chat-1", "sig-1") is True
    assert s.check_and_record("chat-1", "sig-2") is True


def test_dedup_ttl_expiry_allows_resend():
    s = _DedupStore(ttl_seconds=60.0)
    base = 1_000.0
    s.check_and_record("chat-1", "sig-1", now=base)
    assert s.check_and_record("chat-1", "sig-1", now=base + 61.0) is True


def test_dedup_missing_signal_id_does_not_block():
    """A signal without an id should not be silently swallowed."""
    s = _DedupStore(ttl_seconds=60.0)
    assert s.check_and_record("chat-1", "") is True
    assert s.check_and_record("", "sig-1") is True


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------

def test_send_signal_retries_on_retry_after():
    bot = MagicMock()
    # Fail with RetryAfter twice, then succeed
    bot.send_message.side_effect = [RetryAfter(0.5), RetryAfter(0.5), None]
    n = _make_notifier(bot, max_retries=3)
    ok = n.send_signal(_StubSignal("sig-1"), chat_id="chat-1")
    assert ok is True
    assert bot.send_message.call_count == 3
    stats = n.get_stats()
    assert stats["retries_total"] == 2
    assert stats["messages_sent"] == 1


def test_send_signal_retries_on_timed_out():
    bot = MagicMock()
    bot.send_message.side_effect = [TimedOut("timed out"), None]
    n = _make_notifier(bot, max_retries=3)
    ok = n.send_signal(_StubSignal("sig-2"), chat_id="chat-1")
    assert ok is True
    assert bot.send_message.call_count == 2


def test_send_signal_does_not_retry_on_400_like_error():
    bot = MagicMock()
    bot.send_message.side_effect = ValueError("bad request: chat_id not found")
    n = _make_notifier(bot, max_retries=5)
    ok = n.send_signal(_StubSignal("sig-3"), chat_id="chat-1")
    assert ok is False
    assert bot.send_message.call_count == 1  # no retry
    assert n.get_stats()["send_failures"] == 1


def test_send_signal_exhausts_retries_and_reports_failure():
    bot = MagicMock()
    bot.send_message.side_effect = RetryAfter(0.5)  # always fails
    n = _make_notifier(bot, max_retries=2)
    ok = n.send_signal(_StubSignal("sig-4"), chat_id="chat-1")
    assert ok is False
    assert bot.send_message.call_count == 3  # 1 try + 2 retries
    assert n.get_stats()["send_failures"] == 1


# ---------------------------------------------------------------------------
# Dedup wired into send_signal
# ---------------------------------------------------------------------------

def test_send_signal_skips_replay_within_window():
    bot = MagicMock()
    bot.send_message.return_value = None
    n = _make_notifier(bot)
    # First call lands
    assert n.send_signal(_StubSignal("sig-7"), chat_id="chat-1") is True
    # Replay should be suppressed
    assert n.send_signal(_StubSignal("sig-7"), chat_id="chat-1") is False
    # Bot saw only the first message
    assert bot.send_message.call_count == 1
    assert n.get_stats()["dedup_skips"] == 1


def test_send_signal_same_signal_different_chats_both_land():
    bot = MagicMock()
    bot.send_message.return_value = None
    n = _make_notifier(bot)
    n.send_signal(_StubSignal("sig-8"), chat_id="chat-A")
    n.send_signal(_StubSignal("sig-8"), chat_id="chat-B")
    assert bot.send_message.call_count == 2


def test_get_stats_surfaces_dedup_store_size():
    bot = MagicMock()
    bot.send_message.return_value = None
    n = _make_notifier(bot)
    n.send_signal(_StubSignal("sig-A"), chat_id="chat-1")
    n.send_signal(_StubSignal("sig-B"), chat_id="chat-2")
    assert n.get_stats()["dedup_store_size"] == 2
