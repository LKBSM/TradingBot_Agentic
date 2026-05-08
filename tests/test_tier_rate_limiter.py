"""Tests for the INFRA-2B.9 per-tier rate limiter."""

from __future__ import annotations

import threading

import pytest

from src.intelligence.rag.tier_rate_limiter import (
    DEFAULT_PER_MINUTE_CAPS,
    TierRateLimiter,
)


# ---------------------------------------------------------------------------
# Caps
# ---------------------------------------------------------------------------


def test_default_caps_in_seniority_order():
    caps = DEFAULT_PER_MINUTE_CAPS
    assert caps["FREE"] < caps["ANALYST"] < caps["STRATEGIST"] < caps["INSTITUTIONAL"]


def test_unknown_tier_uses_unknown_cap():
    rl = TierRateLimiter()
    snap = rl.snapshot("k", "MYSTERY")
    assert snap.cap == DEFAULT_PER_MINUTE_CAPS["unknown"]


# ---------------------------------------------------------------------------
# Allow / cap
# ---------------------------------------------------------------------------


def test_allow_within_cap():
    rl = TierRateLimiter({"FREE": 3})
    assert rl.allow("k", "FREE")
    assert rl.allow("k", "FREE")
    assert rl.allow("k", "FREE")


def test_allow_blocks_at_cap():
    rl = TierRateLimiter({"FREE": 2})
    assert rl.allow("k", "FREE")
    assert rl.allow("k", "FREE")
    assert not rl.allow("k", "FREE")


def test_zero_cap_blocks_everything():
    rl = TierRateLimiter({"BANNED": 0})
    assert not rl.allow("k", "BANNED")


def test_distinct_keys_isolated():
    rl = TierRateLimiter({"FREE": 1})
    assert rl.allow("alice", "FREE")
    assert rl.allow("bob", "FREE")  # different key → fresh bucket
    assert not rl.allow("alice", "FREE")


def test_distinct_tiers_isolated():
    rl = TierRateLimiter({"FREE": 1, "ANALYST": 5})
    assert rl.allow("k", "FREE")
    # Same key, different tier ⇒ separate bucket.
    assert rl.allow("k", "ANALYST")
    assert not rl.allow("k", "FREE")


def test_allow_rejects_empty_api_key():
    rl = TierRateLimiter({"FREE": 5})
    with pytest.raises(ValueError):
        rl.allow("", "FREE")


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reports_remaining():
    rl = TierRateLimiter({"FREE": 5})
    rl.allow("k", "FREE")
    rl.allow("k", "FREE")
    snap = rl.snapshot("k", "FREE")
    assert snap.api_key == "k"
    assert snap.tier == "FREE"
    assert snap.used == 2
    assert snap.cap == 5
    assert snap.remaining == 3
    assert not snap.is_throttled


def test_snapshot_at_cap_marks_throttled():
    rl = TierRateLimiter({"FREE": 1})
    rl.allow("k", "FREE")
    snap = rl.snapshot("k", "FREE")
    assert snap.remaining == 0
    assert snap.is_throttled
    assert snap.reset_in_seconds > 0


# ---------------------------------------------------------------------------
# Sliding window
# ---------------------------------------------------------------------------


def test_window_slides_releasing_old_entries(monkeypatch):
    rl = TierRateLimiter({"FREE": 2}, window_seconds=10.0)
    fake_now = [100.0]
    monkeypatch.setattr(rl, "_now", lambda: fake_now[0])

    assert rl.allow("k", "FREE")
    assert rl.allow("k", "FREE")
    assert not rl.allow("k", "FREE")
    # Advance past the window
    fake_now[0] = 200.0
    assert rl.allow("k", "FREE")


def test_partial_window_release(monkeypatch):
    """Sliding (not fixed) window: advancing partially still releases
    items that crossed the cutoff."""
    rl = TierRateLimiter({"FREE": 3}, window_seconds=10.0)
    fake_now = [100.0]
    monkeypatch.setattr(rl, "_now", lambda: fake_now[0])

    rl.allow("k", "FREE")  # t=100
    fake_now[0] = 105.0
    rl.allow("k", "FREE")  # t=105
    rl.allow("k", "FREE")  # t=105 — at cap
    assert not rl.allow("k", "FREE")
    # Advance past first entry's window; second still active.
    fake_now[0] = 111.0
    assert rl.allow("k", "FREE")  # 1 slot freed


# ---------------------------------------------------------------------------
# Cap mutation + reset
# ---------------------------------------------------------------------------


def test_set_cap_takes_effect_next_call():
    rl = TierRateLimiter({"FREE": 1})
    rl.allow("k", "FREE")  # at cap
    assert not rl.allow("k", "FREE")
    rl.set_cap("FREE", 5)
    assert rl.allow("k", "FREE")  # extra slots now available


def test_set_cap_rejects_negative():
    rl = TierRateLimiter()
    with pytest.raises(ValueError):
        rl.set_cap("FREE", -1)


def test_reset_per_key():
    rl = TierRateLimiter({"FREE": 1})
    rl.allow("alice", "FREE")
    rl.allow("bob", "FREE")
    rl.reset("alice")
    assert rl.allow("alice", "FREE")
    # bob still at cap
    assert not rl.allow("bob", "FREE")


def test_reset_all():
    rl = TierRateLimiter({"FREE": 1})
    rl.allow("alice", "FREE")
    rl.allow("bob", "FREE")
    rl.reset()
    assert rl.allow("alice", "FREE")
    assert rl.allow("bob", "FREE")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_concurrent_allow_respects_cap():
    rl = TierRateLimiter({"FREE": 100})
    successes = {"n": 0}
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            if rl.allow("alice", "FREE"):
                with lock:
                    successes["n"] += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 8 threads × 50 = 400 attempts; cap = 100 ⇒ exactly 100 succeed.
    assert successes["n"] == 100
