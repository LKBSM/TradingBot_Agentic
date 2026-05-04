"""Tests for the INFRA-2B.7 per-tier cost quota enforcer."""

from __future__ import annotations

import threading

import pytest

from src.intelligence.rag.cost_quota import (
    DEFAULT_DAILY_CAPS_USD,
    CostQuotaEnforcer,
    QuotaExceeded,
)


# ---------------------------------------------------------------------------
# Default caps
# ---------------------------------------------------------------------------


def test_default_caps_ordered_by_tier_seniority():
    caps = DEFAULT_DAILY_CAPS_USD
    assert caps["FREE"] < caps["ANALYST"] < caps["STRATEGIST"] < caps["INSTITUTIONAL"]


def test_unknown_tier_falls_back_to_unknown_cap():
    e = CostQuotaEnforcer()
    snap = e.snapshot("MYSTERY-TIER")
    assert snap.cap_usd == DEFAULT_DAILY_CAPS_USD["unknown"]


# ---------------------------------------------------------------------------
# Allow / record
# ---------------------------------------------------------------------------


def test_allow_true_under_cap():
    e = CostQuotaEnforcer()
    assert e.allow("FREE", 0.01)


def test_allow_false_over_cap():
    e = CostQuotaEnforcer()
    e.record("FREE", DEFAULT_DAILY_CAPS_USD["FREE"])
    assert not e.allow("FREE", 0.01)


def test_record_then_used_reflects_spend():
    e = CostQuotaEnforcer()
    e.record("ANALYST", 0.10)
    e.record("ANALYST", 0.05)
    assert e.used("ANALYST") == pytest.approx(0.15)


def test_record_rejects_negative():
    e = CostQuotaEnforcer()
    with pytest.raises(ValueError):
        e.record("FREE", -1.0)


def test_record_zero_is_noop():
    e = CostQuotaEnforcer()
    e.record("FREE", 0.0)
    assert e.used("FREE") == 0.0


# ---------------------------------------------------------------------------
# check_and_record (atomic)
# ---------------------------------------------------------------------------


def test_check_and_record_atomic_when_within_cap():
    e = CostQuotaEnforcer({"FREE": 1.0})
    e.check_and_record("FREE", 0.5)
    e.check_and_record("FREE", 0.4)
    assert e.used("FREE") == pytest.approx(0.9)


def test_check_and_record_raises_when_over_cap():
    e = CostQuotaEnforcer({"FREE": 0.10})
    e.check_and_record("FREE", 0.08)
    with pytest.raises(QuotaExceeded) as exc_info:
        e.check_and_record("FREE", 0.05)
    err = exc_info.value
    assert err.tier == "FREE"
    assert err.cap_usd == 0.10
    # Cost wasn't recorded after the rejection.
    assert e.used("FREE") == pytest.approx(0.08)


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


def test_snapshot_reports_remaining_quota():
    e = CostQuotaEnforcer({"STRATEGIST": 5.0})
    e.record("STRATEGIST", 1.5)
    snap = e.snapshot("STRATEGIST")
    assert snap.tier == "STRATEGIST"
    assert snap.used_usd == 1.5
    assert snap.cap_usd == 5.0
    assert snap.remaining_usd == 3.5
    assert snap.utilisation == pytest.approx(0.30)


def test_snapshot_zero_cap_does_not_divide():
    e = CostQuotaEnforcer({"FROZEN_TIER": 0.0})
    snap = e.snapshot("FROZEN_TIER")
    assert snap.utilisation == 0.0
    assert snap.remaining_usd == 0.0


# ---------------------------------------------------------------------------
# Rolling 24h window
# ---------------------------------------------------------------------------


def test_expired_records_are_purged(monkeypatch):
    """Records older than ``window_seconds`` drop out on next read."""
    e = CostQuotaEnforcer(caps={"FREE": 1.0}, window_seconds=10)

    fake_now = [100.0]
    monkeypatch.setattr(e, "_now", lambda: fake_now[0])

    e.record("FREE", 0.5)
    assert e.used("FREE") == 0.5

    # Advance well past the 10-second window
    fake_now[0] = 200.0
    assert e.used("FREE") == 0.0
    # Quota is fresh again
    assert e.allow("FREE", 0.9)


def test_records_within_window_remain():
    e = CostQuotaEnforcer(caps={"FREE": 1.0}, window_seconds=100)
    e.record("FREE", 0.4)
    assert e.used("FREE") == 0.4


# ---------------------------------------------------------------------------
# Cap mutation
# ---------------------------------------------------------------------------


def test_set_cap_takes_effect_immediately():
    e = CostQuotaEnforcer({"FREE": 1.0})
    e.record("FREE", 0.6)
    assert e.allow("FREE", 0.3)
    e.set_cap("FREE", 0.5)
    assert not e.allow("FREE", 0.01)


def test_set_cap_rejects_negative():
    e = CostQuotaEnforcer()
    with pytest.raises(ValueError):
        e.set_cap("FREE", -1.0)


def test_reset_clears_spend():
    e = CostQuotaEnforcer({"FREE": 1.0})
    e.record("FREE", 0.9)
    e.reset("FREE")
    assert e.used("FREE") == 0.0


def test_reset_all_clears_every_tier():
    e = CostQuotaEnforcer()
    e.record("FREE", 0.01)
    e.record("ANALYST", 0.10)
    e.reset()
    assert e.used("FREE") == 0.0
    assert e.used("ANALYST") == 0.0


# ---------------------------------------------------------------------------
# Concurrent writers
# ---------------------------------------------------------------------------


def test_concurrent_check_and_record_respects_cap():
    """Many parallel writers must not push spend past the cap."""
    cap = 1.00
    per_call = 0.01
    e = CostQuotaEnforcer({"FREE": cap})

    successes = {"n": 0}
    rejections = {"n": 0}
    lock = threading.Lock()

    def worker():
        for _ in range(50):
            try:
                e.check_and_record("FREE", per_call)
                with lock:
                    successes["n"] += 1
            except QuotaExceeded:
                with lock:
                    rejections["n"] += 1

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Cap is 1.00, per-call 0.01 ⇒ at most 100 successes.
    assert successes["n"] <= 100
    # Total spend never overshoots (rolling window keeps every entry here).
    assert e.used("FREE") <= cap + 1e-9
    # Plenty of rejections happened (otherwise the cap was too generous).
    assert rejections["n"] > 0
