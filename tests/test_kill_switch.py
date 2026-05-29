"""Unit tests for ``src.risk.kill_switch.KillSwitch``."""
from __future__ import annotations

import time

import pytest

from src.risk.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    TripReason,
)


def _ks(**overrides) -> KillSwitch:
    cfg = KillSwitchConfig(**overrides)
    return KillSwitch(config=cfg, starting_equity=1000.0)


# ============================================================================ #
# 1. Streak rule
# ============================================================================ #


def test_consecutive_losses_trip_at_limit() -> None:
    ks = _ks(max_consecutive_losses=3)
    for _ in range(2):
        ks.record_trade_outcome(r_multiple=-1.0, pnl_dollars=-10.0)
    assert ks.check() is True  # still under limit
    ks.record_trade_outcome(r_multiple=-1.0, pnl_dollars=-10.0)
    assert ks.check() is False
    assert ks.trip_reason is TripReason.CONSECUTIVE_LOSSES


def test_winning_trade_resets_streak() -> None:
    ks = _ks(max_consecutive_losses=3)
    ks.record_trade_outcome(-1.0)
    ks.record_trade_outcome(-1.0)
    ks.record_trade_outcome(+2.0)  # winner clears streak
    assert ks.consecutive_losses == 0
    ks.record_trade_outcome(-1.0)
    ks.record_trade_outcome(-1.0)
    assert ks.check() is True  # still safe


# ============================================================================ #
# 2. Daily-DD rule
# ============================================================================ #


def test_daily_dd_trip() -> None:
    ks = _ks(daily_dd_limit_pct=0.05, max_consecutive_losses=99)
    # Lose 6 % of starting equity in mixed signs of trades.
    ks.record_trade_outcome(r_multiple=-1.0, pnl_dollars=-30.0)
    assert ks.check() is True
    ks.record_trade_outcome(r_multiple=-1.0, pnl_dollars=-31.0)
    assert ks.check() is False
    assert ks.trip_reason is TripReason.DAILY_DRAWDOWN


# ============================================================================ #
# 3. Vol-spike rule
# ============================================================================ #


def test_volatility_spike_trip() -> None:
    ks = _ks(vol_zscore_limit=3.0, vol_history_window=80)
    # Push 80 calm bars then one spike at z >> 3.
    for _ in range(80):
        ks.update_volatility(1.0)
    assert ks.check() is True
    ks.update_volatility(50.0)  # huge outlier
    assert ks.check() is False
    assert ks.trip_reason is TripReason.VOLATILITY_SPIKE


def test_volatility_does_not_trip_below_window() -> None:
    """Cannot trip if the rolling buffer has too few samples."""
    ks = _ks(vol_zscore_limit=3.0, vol_history_window=200)
    for _ in range(5):
        ks.update_volatility(1.0)
    ks.update_volatility(1000.0)
    # Buffer too small (< window/4 = 50) — must not trip.
    assert ks.check() is True


# ============================================================================ #
# 4. Broker disconnect
# ============================================================================ #


def test_broker_disconnect_trip() -> None:
    ks = _ks(heartbeat_max_silence_s=60.0)
    now = time.time()
    ks.heartbeat(now=now)
    # Simulate 90 s of silence by passing 'now' forward to check().
    assert ks.check(now=now + 90.0) is False
    assert ks.trip_reason is TripReason.BROKER_DISCONNECT


def test_heartbeat_clears_disconnect() -> None:
    ks = _ks(heartbeat_max_silence_s=30.0)
    ks.heartbeat(now=1000.0)
    assert ks.check(now=1100.0) is False  # tripped
    ks.heartbeat(now=1110.0)  # broker resumes
    assert ks.check(now=1115.0) is True


# ============================================================================ #
# Manual override / persistence
# ============================================================================ #


def test_manual_reset_requires_ack_phrase() -> None:
    ks = _ks(max_consecutive_losses=2)
    ks.record_trade_outcome(-1.0)
    ks.record_trade_outcome(-1.0)
    assert ks.is_tripped is True
    assert ks.manual_reset(operator="ops@bot", ack_phrase="oops") is False
    assert ks.is_tripped is True
    assert ks.manual_reset(operator="ops@bot", ack_phrase="I-ACCEPT-RISK") is True
    assert ks.is_tripped is False


def test_manual_reset_refused_for_broker_disconnect() -> None:
    """Operator must NOT be able to override a real broker outage."""
    ks = _ks(heartbeat_max_silence_s=10.0)
    ks.heartbeat(now=1000.0)
    assert ks.check(now=1050.0) is False
    assert ks.trip_reason is TripReason.BROKER_DISCONNECT
    ok = ks.manual_reset(operator="ops@bot", ack_phrase="I-ACCEPT-RISK")
    assert ok is False
    assert ks.is_tripped is True


def test_state_roundtrips_through_dict() -> None:
    ks = _ks(max_consecutive_losses=3)
    ks.record_trade_outcome(-1.0, pnl_dollars=-10.0)
    ks.record_trade_outcome(-1.0, pnl_dollars=-10.0)
    ks.update_volatility(1.5)
    ks.heartbeat(now=2000.0)
    state = ks.to_dict()

    restored = KillSwitch.from_dict(state, starting_equity=1000.0)
    assert restored.consecutive_losses == 2
    assert pytest.approx(restored.daily_pnl_pct, rel=1e-9) == -0.02
    assert restored.is_tripped == ks.is_tripped


def test_status_payload_shape() -> None:
    ks = _ks()
    ks.heartbeat()
    s = ks.status()
    for key in (
        "tripped",
        "reason",
        "consecutive_losses",
        "daily_pnl_pct",
        "last_heartbeat_age_s",
        "vol_buffer_size",
    ):
        assert key in s
