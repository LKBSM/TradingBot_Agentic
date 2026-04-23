"""Tests for SignalStateMachine.

Covers the full contract:
  * Hysteresis (oscillating scores don't flip)
  * Confirmation window (single-bar spike rejected)
  * All 6 exit reasons (target, stop, time, score-decay, regime, opposing)
  * Cooldown enforcement (no re-entry during cooldown bars)
  * Opposing-direction lockout (must pass through HOLD+cooldown)
  * Idempotency (duplicate bar timestamp is a no-op)
  * Out-of-order bar rejection
  * Persistence roundtrip (to_dict / from_dict)
  * Thread safety (concurrent on_bar calls on same machine)
  * Input validation (NaN / negative prices / low>high / close outside range)
  * Stats & transition history
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest

from src.intelligence.signal_state_machine import (
    BarInput,
    Direction,
    ExitReason,
    PublicState,
    SignalStateMachine,
    StateMachineConfig,
    StateTransition,
)


# =============================================================================
# TEST FIXTURES — lightweight stand-in for ConfluenceSignal
# =============================================================================

@dataclass
class FakeSignal:
    """Minimal duck-typed ConfluenceSignal for tests."""
    signal_id: str
    symbol: str
    signal_type: str           # "LONG" or "SHORT"
    confluence_score: float
    entry_price: float
    stop_loss: float
    take_profit: float

    def to_dict(self):
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type,
            "confluence_score": self.confluence_score,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


def mk_signal(direction: str, score: float, entry: float = 100.0,
              sl_dist: float = 2.0, tp_dist: float = 4.0) -> FakeSignal:
    if direction == "LONG":
        return FakeSignal("sid-L", "TEST", "LONG", score, entry,
                          entry - sl_dist, entry + tp_dist)
    return FakeSignal("sid-S", "TEST", "SHORT", score, entry,
                      entry + sl_dist, entry - tp_dist)


def mk_bar(ts: str, signal: Optional[FakeSignal] = None,
           close: float = 100.0, high: Optional[float] = None,
           low: Optional[float] = None, vol_regime: Optional[str] = None,
           structure_broken: bool = False) -> BarInput:
    high = high if high is not None else close + 0.5
    low = low if low is not None else close - 0.5
    return BarInput(
        bar_timestamp=ts,
        high=high,
        low=low,
        close=close,
        signal=signal,
        vol_regime=vol_regime,
        structure_broken=structure_broken,
    )


def ts(i: int) -> str:
    """Build an ISO-ordered bar timestamp so lex comparison == temporal order."""
    return f"2026-04-22T{i // 60:02d}:{i % 60:02d}:00"


# =============================================================================
# CONFIG VALIDATION
# =============================================================================

class TestStateMachineConfig:
    def test_defaults_are_sane(self):
        cfg = StateMachineConfig()
        assert cfg.enter_threshold == 75.0
        assert cfg.exit_threshold == 55.0
        assert cfg.confirm_bars == 2
        assert cfg.cooldown_bars == 2

    def test_exit_must_be_below_enter(self):
        with pytest.raises(ValueError, match="exit_threshold"):
            StateMachineConfig(enter_threshold=70, exit_threshold=80)

    def test_exit_can_equal_zero(self):
        StateMachineConfig(enter_threshold=60, exit_threshold=0)

    def test_thresholds_bounded_to_100(self):
        with pytest.raises(ValueError):
            StateMachineConfig(enter_threshold=101)

    def test_confirm_bars_must_be_positive(self):
        with pytest.raises(ValueError):
            StateMachineConfig(confirm_bars=0)

    def test_cooldown_bars_zero_allowed(self):
        StateMachineConfig(cooldown_bars=0)

    def test_cooldown_bars_negative_rejected(self):
        with pytest.raises(ValueError):
            StateMachineConfig(cooldown_bars=-1)


# =============================================================================
# BAR INPUT VALIDATION
# =============================================================================

class TestBarInputValidation:
    def test_nan_high_rejected(self):
        with pytest.raises(ValueError):
            BarInput(bar_timestamp=ts(1), high=float("nan"), low=99, close=99.5)

    def test_inf_rejected(self):
        with pytest.raises(ValueError):
            BarInput(bar_timestamp=ts(1), high=float("inf"), low=99, close=99.5)

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            BarInput(bar_timestamp=ts(1), high=-1, low=-2, close=-1.5)

    def test_low_gt_high_rejected(self):
        with pytest.raises(ValueError, match="low.*> high"):
            BarInput(bar_timestamp=ts(1), high=99, low=101, close=100)

    def test_close_outside_range_rejected(self):
        with pytest.raises(ValueError, match="must lie within"):
            BarInput(bar_timestamp=ts(1), high=100, low=99, close=105)

    def test_empty_timestamp_rejected(self):
        with pytest.raises(ValueError):
            BarInput(bar_timestamp="", high=100, low=99, close=99.5)


# =============================================================================
# INITIAL STATE
# =============================================================================

class TestInitialState:
    def test_starts_in_hold(self):
        sm = SignalStateMachine()
        snap = sm.snapshot()
        assert snap.state is PublicState.HOLD
        assert snap.direction is None
        assert snap.active_signal is None
        assert snap.confirmation_progress is None

    def test_stats_zeroed(self):
        stats = SignalStateMachine().get_stats()
        assert stats["bars_processed"] == 0
        assert stats["signals_emitted"] == 0
        assert stats["arms_started"] == 0
        assert stats["confirmation_rate"] is None


# =============================================================================
# RULE 1 — HYSTERESIS (the single most important anti-flicker mechanism)
# =============================================================================

class TestHysteresis:
    def test_score_oscillating_around_enter_does_not_flicker(self):
        """Score bouncing 70 ↔ 80 around enter=75 must NOT produce multiple entries."""
        cfg = StateMachineConfig(
            enter_threshold=75, exit_threshold=55, confirm_bars=2, cooldown_bars=1
        )
        sm = SignalStateMachine(cfg)
        transitions: List[StateTransition] = []
        # 20 bars of oscillation 70→80→70→80...
        for i in range(20):
            score = 80 if i % 2 == 0 else 70
            sig = mk_signal("LONG", score) if score >= 75 else None
            _, t = sm.on_bar(mk_bar(ts(i), sig, close=100.0))
            if t is not None:
                transitions.append(t)
        # Arming starts at i=0 (score=80 ≥ 75), aborts at i=1 (score=70 < 75).
        # Then re-arms at i=2, aborts at i=3, etc. Never confirms.
        assert sm.snapshot().state is PublicState.HOLD
        # No ACTIVE transitions — only internal arm/abort cycles (no transition emitted)
        buy_transitions = [t for t in transitions if t.to_state is PublicState.BUY]
        assert buy_transitions == []

    def test_score_oscillating_around_exit_does_not_flicker(self):
        """Once active, a score bouncing around exit_threshold should NOT whipsaw out."""
        cfg = StateMachineConfig(
            enter_threshold=75, exit_threshold=55, confirm_bars=2, cooldown_bars=2
        )
        sm = SignalStateMachine(cfg)
        # Enter cleanly
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 80), close=100))
        _, t = sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 80), close=100))
        assert t is not None and t.to_state is PublicState.BUY
        # Now oscillate score 60↔65 — both ABOVE exit_threshold=55, so stay BUY
        exits = 0
        for i in range(2, 10):
            score = 60 if i % 2 == 0 else 65
            sig = mk_signal("LONG", score)
            _, t = sm.on_bar(mk_bar(ts(i), sig, close=100))
            if t is not None and t.exit_reason is not None:
                exits += 1
        assert exits == 0
        assert sm.snapshot().state is PublicState.BUY


# =============================================================================
# RULE 2 — CONFIRMATION WINDOW
# =============================================================================

class TestConfirmationWindow:
    def test_single_bar_spike_does_not_enter(self):
        cfg = StateMachineConfig(confirm_bars=2)
        sm = SignalStateMachine(cfg)
        # Bar 0: score=85 (arms)
        _, t0 = sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        assert t0 is None  # No transition emitted yet (still arming)
        assert sm.snapshot().state is PublicState.HOLD
        assert sm.snapshot().confirmation_progress == (1, 2)
        # Bar 1: score=50 (aborts)
        _, t1 = sm.on_bar(mk_bar(ts(1), None))
        assert t1 is None
        assert sm.snapshot().state is PublicState.HOLD
        assert sm.snapshot().confirmation_progress is None
        assert sm.get_stats()["arms_aborted"] == 1
        assert sm.get_stats()["arms_confirmed"] == 0

    def test_two_consecutive_bars_enter(self):
        cfg = StateMachineConfig(confirm_bars=2)
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        snap, t = sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 82)))
        assert t is not None
        assert t.to_state is PublicState.BUY
        assert t.direction is Direction.LONG
        assert snap.state is PublicState.BUY

    def test_confirm_bars_one_enters_immediately(self):
        cfg = StateMachineConfig(confirm_bars=1)
        sm = SignalStateMachine(cfg)
        _, t = sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        assert t is not None and t.to_state is PublicState.BUY

    def test_direction_flip_during_arming_resets(self):
        """If arming LONG and bar 2 is a SHORT signal, LONG is aborted — not flipped."""
        cfg = StateMachineConfig(confirm_bars=3)
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        _, t1 = sm.on_bar(mk_bar(ts(1), mk_signal("SHORT", 85)))  # opposite direction
        assert t1 is None
        # LONG arm aborted
        assert sm.get_stats()["arms_aborted"] == 1
        # SHORT arming begins fresh at bar 1? No — we enforce "fresh bar to re-arm"
        assert sm.snapshot().state is PublicState.HOLD
        # The SHORT arm starts on bar 2
        sm.on_bar(mk_bar(ts(2), mk_signal("SHORT", 85)))
        sm.on_bar(mk_bar(ts(3), mk_signal("SHORT", 85)))
        snap, t = sm.on_bar(mk_bar(ts(4), mk_signal("SHORT", 85)))
        assert t is not None
        assert t.to_state is PublicState.SELL


# =============================================================================
# RULE 3 — EXIT REASONS (the 6 paths back to HOLD)
# =============================================================================

class TestExitReasons:
    @staticmethod
    def _enter_long(sm: SignalStateMachine) -> None:
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, sl_dist=2, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85, entry=100, sl_dist=2, tp_dist=4)))

    def test_target_reached_long(self):
        sm = SignalStateMachine()
        self._enter_long(sm)
        # Price prints a high of 104.5 — past TP of 104
        snap, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 80), close=103, high=104.5, low=102))
        assert t is not None
        assert t.exit_reason is ExitReason.TARGET_REACHED
        assert t.exit_price == 104.0
        assert snap.state is PublicState.HOLD
        assert sm.get_stats()["exits_by_reason"]["target_reached"] == 1

    def test_invalidated_stop_hit_long(self):
        sm = SignalStateMachine()
        self._enter_long(sm)
        # Low of 97 pierces stop at 98
        snap, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 80), close=97.5, high=98.5, low=97))
        assert t is not None
        assert t.exit_reason is ExitReason.INVALIDATED
        assert t.exit_price == 98.0

    def test_target_reached_short(self):
        sm = SignalStateMachine()
        sm.on_bar(mk_bar(ts(0), mk_signal("SHORT", 85, entry=100, sl_dist=2, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("SHORT", 85, entry=100, sl_dist=2, tp_dist=4)))
        # TP for short is entry - tp_dist = 96
        snap, t = sm.on_bar(mk_bar(ts(2), mk_signal("SHORT", 80), close=96.5, high=97, low=95.5))
        assert t is not None
        assert t.exit_reason is ExitReason.TARGET_REACHED

    def test_time_expired(self):
        cfg = StateMachineConfig(max_signal_age_bars=3, silent_bars_before_score_exit=99)
        sm = SignalStateMachine(cfg)
        # Enter LONG with confirm_bars=2 (default)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        _, enter = sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85)))
        assert enter.to_state is PublicState.BUY
        # bar 2: active bar 2 of 3
        sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 70)))
        # bar 3: active bar 3 of 3 — time expires
        _, t = sm.on_bar(mk_bar(ts(3), mk_signal("LONG", 70)))
        assert t is not None
        assert t.exit_reason is ExitReason.TIME_EXPIRED

    def test_score_decayed_below_exit_threshold(self):
        sm = SignalStateMachine(StateMachineConfig(exit_threshold=55))
        TestExitReasons._enter_long(sm)
        # Score drops to 50 — below exit — exit
        _, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 50)))
        assert t is not None
        assert t.exit_reason is ExitReason.SCORE_DECAYED

    def test_score_decayed_via_silent_bars(self):
        """Detector returns None for N bars → infer score decay."""
        cfg = StateMachineConfig(silent_bars_before_score_exit=2)
        sm = SignalStateMachine(cfg)
        TestExitReasons._enter_long(sm)
        # 1 silent bar — still active
        _, t1 = sm.on_bar(mk_bar(ts(2), None))
        assert t1 is None
        # 2nd silent bar — exits
        _, t2 = sm.on_bar(mk_bar(ts(3), None))
        assert t2 is not None
        assert t2.exit_reason is ExitReason.SCORE_DECAYED

    def test_silent_counter_resets_on_strong_signal(self):
        cfg = StateMachineConfig(silent_bars_before_score_exit=2)
        sm = SignalStateMachine(cfg)
        TestExitReasons._enter_long(sm)
        sm.on_bar(mk_bar(ts(2), None))                              # silent 1
        sm.on_bar(mk_bar(ts(3), mk_signal("LONG", 80)))             # resets
        _, t = sm.on_bar(mk_bar(ts(4), None))                       # silent 1 again
        assert t is None

    def test_regime_shifted_to_high_exits(self):
        sm = SignalStateMachine()
        TestExitReasons._enter_long(sm)
        _, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 80), vol_regime="high"))
        assert t is not None
        assert t.exit_reason is ExitReason.REGIME_SHIFTED

    def test_regime_high_on_entry_bar_does_not_exit(self):
        """Regime='high' on the entry bar itself should NOT cause exit in that bar."""
        cfg = StateMachineConfig(confirm_bars=1)  # enter on bar 0
        sm = SignalStateMachine(cfg)
        _, t = sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85), vol_regime="high"))
        assert t is not None
        assert t.to_state is PublicState.BUY
        assert t.exit_reason is None

    def test_opposing_signal_exits(self):
        sm = SignalStateMachine(StateMachineConfig(enter_threshold=75))
        TestExitReasons._enter_long(sm)
        _, t = sm.on_bar(mk_bar(ts(2), mk_signal("SHORT", 80)))
        assert t is not None
        assert t.exit_reason is ExitReason.OPPOSING_SIGNAL

    def test_weak_opposing_signal_does_not_exit(self):
        """Opposing direction but below enter_threshold → no exit."""
        sm = SignalStateMachine(StateMachineConfig(enter_threshold=75))
        TestExitReasons._enter_long(sm)
        _, t = sm.on_bar(mk_bar(ts(2), mk_signal("SHORT", 60)))
        assert t is None

    def test_structure_broken_exits(self):
        sm = SignalStateMachine()
        TestExitReasons._enter_long(sm)
        _, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 80), structure_broken=True))
        assert t is not None
        assert t.exit_reason is ExitReason.INVALIDATED


# =============================================================================
# RULE 4 — COOLDOWN
# =============================================================================

class TestCooldown:
    def test_cooldown_blocks_re_entry(self):
        cfg = StateMachineConfig(cooldown_bars=2, confirm_bars=1)
        sm = SignalStateMachine(cfg)
        # Enter, exit by target, then try to re-enter immediately
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, tp_dist=4)))
        # Target hit on bar 1
        _, exit_t = sm.on_bar(
            mk_bar(ts(1), mk_signal("LONG", 85), close=103, high=104.5, low=102)
        )
        assert exit_t.exit_reason is ExitReason.TARGET_REACHED
        # Cooldown bar 1 of 2 — even strong signal ignored
        snap, t = sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 95)))
        assert t is None
        assert snap.state is PublicState.HOLD
        assert snap.cooldown_bars_remaining is not None
        # Cooldown bar 2 of 2
        snap, t = sm.on_bar(mk_bar(ts(3), mk_signal("LONG", 95)))
        assert t is None
        assert snap.state is PublicState.HOLD

    def test_after_cooldown_can_re_enter(self):
        cfg = StateMachineConfig(cooldown_bars=2, confirm_bars=1)
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85), close=103, high=104.5, low=102))
        # 2 cooldown bars
        sm.on_bar(mk_bar(ts(2), None))
        sm.on_bar(mk_bar(ts(3), None))
        # Fresh bar after cooldown — can now arm
        _, t = sm.on_bar(mk_bar(ts(4), mk_signal("LONG", 85)))
        assert t is not None
        assert t.to_state is PublicState.BUY


# =============================================================================
# RULE 5 — OPPOSING-DIRECTION LOCKOUT
# =============================================================================

class TestOpposingLockout:
    def test_long_cannot_flip_directly_to_short(self):
        cfg = StateMachineConfig(cooldown_bars=2, confirm_bars=1)
        sm = SignalStateMachine(cfg)
        # Enter LONG
        _, t0 = sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        assert t0.to_state is PublicState.BUY
        # Strong SHORT signal arrives → must go to HOLD first
        _, t1 = sm.on_bar(mk_bar(ts(1), mk_signal("SHORT", 90)))
        assert t1.to_state is PublicState.HOLD
        assert t1.exit_reason is ExitReason.OPPOSING_SIGNAL
        # Next bar still in cooldown
        _, t2 = sm.on_bar(mk_bar(ts(2), mk_signal("SHORT", 90)))
        assert t2 is None
        assert sm.snapshot().state is PublicState.HOLD
        # Still can't flip — must finish cooldown
        _, t3 = sm.on_bar(mk_bar(ts(3), mk_signal("SHORT", 90)))
        assert t3 is None
        # After cooldown, needs confirmation bars
        _, t4 = sm.on_bar(mk_bar(ts(4), mk_signal("SHORT", 90)))
        assert t4.to_state is PublicState.SELL


# =============================================================================
# IDEMPOTENCY & ORDERING
# =============================================================================

class TestIdempotency:
    def test_duplicate_timestamp_is_no_op(self):
        sm = SignalStateMachine()
        bar = mk_bar(ts(0), mk_signal("LONG", 85))
        sm.on_bar(bar)
        before = sm.snapshot()
        # Replay same bar
        snap, t = sm.on_bar(bar)
        assert t is None
        assert snap == before
        assert sm.get_stats()["bars_rejected_duplicate"] == 1

    def test_out_of_order_bar_rejected(self):
        sm = SignalStateMachine()
        sm.on_bar(mk_bar(ts(5), mk_signal("LONG", 85)))
        # Earlier timestamp now arrives
        snap, t = sm.on_bar(mk_bar(ts(3), mk_signal("LONG", 85)))
        assert t is None
        assert sm.get_stats()["bars_rejected_out_of_order"] == 1

    def test_ordered_bars_increment_counter(self):
        sm = SignalStateMachine()
        for i in range(5):
            sm.on_bar(mk_bar(ts(i), None))
        assert sm.get_stats()["bars_processed"] == 5


# =============================================================================
# PERSISTENCE
# =============================================================================

class TestPersistence:
    def test_roundtrip_from_idle(self):
        sm = SignalStateMachine()
        sm.on_bar(mk_bar(ts(0), None))
        d = sm.to_dict()
        restored = SignalStateMachine.from_dict(d)
        assert restored.snapshot().state is PublicState.HOLD
        assert restored.get_stats()["bars_processed"] == 1

    def test_roundtrip_active_signal(self):
        sm = SignalStateMachine(StateMachineConfig(confirm_bars=1))
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100)))
        d = sm.to_dict()
        restored = SignalStateMachine.from_dict(d)
        snap = restored.snapshot()
        assert snap.state is PublicState.BUY
        assert snap.direction is Direction.LONG
        assert snap.entered_at_bar == ts(0)
        assert snap.entered_at_price == 100.0

    def test_roundtrip_preserves_transition_history(self):
        sm = SignalStateMachine(StateMachineConfig(confirm_bars=1, cooldown_bars=1))
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85), close=105, high=105.5, low=104))
        history_before = sm.transition_history()
        restored = SignalStateMachine.from_dict(sm.to_dict())
        assert restored.transition_history() == history_before

    def test_roundtrip_cooldown_phase(self):
        cfg = StateMachineConfig(confirm_bars=1, cooldown_bars=2)
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85), close=105, high=105.5, low=104))
        # Now in cooldown
        d = sm.to_dict()
        restored = SignalStateMachine.from_dict(d)
        snap = restored.snapshot()
        assert snap.state is PublicState.HOLD
        assert snap.cooldown_bars_remaining == 2
        assert snap.last_exit_reason is ExitReason.TARGET_REACHED

    def test_bad_schema_version_rejected(self):
        with pytest.raises(ValueError, match="schema_version"):
            SignalStateMachine.from_dict({"schema_version": 999, "config": {}})

    def test_bad_payload_type_rejected(self):
        with pytest.raises(TypeError):
            SignalStateMachine.from_dict("not-a-dict")  # type: ignore

    def test_rehydrator_callback_used(self):
        sm = SignalStateMachine(StateMachineConfig(confirm_bars=1))
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100)))
        d = sm.to_dict()

        def rehydrate(payload):
            return FakeSignal(**payload)

        restored = SignalStateMachine.from_dict(d, signal_rehydrator=rehydrate)
        assert isinstance(restored.snapshot().active_signal, FakeSignal)


# =============================================================================
# THREAD SAFETY
# =============================================================================

class TestThreadSafety:
    def test_concurrent_readers_do_not_corrupt(self):
        """Reads happening concurrently with writes should never observe torn state."""
        sm = SignalStateMachine(StateMachineConfig(confirm_bars=1))

        errors: List[Exception] = []

        def writer():
            try:
                for i in range(200):
                    sig = mk_signal("LONG", 85) if i % 2 == 0 else None
                    sm.on_bar(mk_bar(ts(i), sig))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(500):
                    snap = sm.snapshot()
                    # Every snapshot must have a valid state
                    assert snap.state in PublicState
                    # Invariant: if BUY/SELL, must have a direction
                    if snap.state in (PublicState.BUY, PublicState.SELL):
                        assert snap.direction is not None
                    sm.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer)]
        threads += [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == []


# =============================================================================
# STATS & TRANSITION HISTORY
# =============================================================================

class TestObservability:
    def test_signal_lifetime_avg(self):
        cfg = StateMachineConfig(confirm_bars=1, cooldown_bars=1, max_signal_age_bars=5)
        sm = SignalStateMachine(cfg)
        # Enter at bar 0, time expire at bar 4
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        for i in range(1, 5):
            sm.on_bar(mk_bar(ts(i), mk_signal("LONG", 70)))
        stats = sm.get_stats()
        assert stats["signals_emitted"] == 1
        assert stats["avg_signal_lifetime_bars"] == 5.0

    def test_transition_history_captures_entry_and_exit(self):
        cfg = StateMachineConfig(confirm_bars=1, cooldown_bars=1)
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100, tp_dist=4)))
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 85), close=105, high=105.5, low=104))
        h = sm.transition_history()
        assert len(h) == 2
        assert h[0]["to_state"] == "BUY"
        assert h[1]["to_state"] == "HOLD"
        assert h[1]["exit_reason"] == "target_reached"

    def test_confirmation_rate(self):
        cfg = StateMachineConfig(confirm_bars=2)
        sm = SignalStateMachine(cfg)
        # Arm 1, abort
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        sm.on_bar(mk_bar(ts(1), None))
        # Arm 2, confirm
        sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 85)))
        sm.on_bar(mk_bar(ts(3), mk_signal("LONG", 85)))
        stats = sm.get_stats()
        assert stats["arms_started"] == 2
        assert stats["arms_confirmed"] == 1
        assert stats["arms_aborted"] == 1
        assert stats["confirmation_rate"] == 0.5

    def test_history_bounded(self):
        cfg = StateMachineConfig(
            confirm_bars=1, cooldown_bars=0, transition_history_max=3
        )
        sm = SignalStateMachine(cfg)
        # Generate 5 entry/exit cycles
        for i in range(5):
            sm.on_bar(mk_bar(
                ts(2 * i), mk_signal("LONG", 85, entry=100, tp_dist=4)
            ))
            sm.on_bar(mk_bar(
                ts(2 * i + 1), mk_signal("LONG", 85),
                close=105, high=105.5, low=104,
            ))
        # 10 transitions emitted, history kept at 3
        assert len(sm.transition_history()) == 3


# =============================================================================
# SNAPSHOT SEMANTICS
# =============================================================================

class TestSnapshotSemantics:
    def test_bars_remaining_counts_down_during_active(self):
        cfg = StateMachineConfig(
            confirm_bars=1, max_signal_age_bars=5, silent_bars_before_score_exit=99
        )
        sm = SignalStateMachine(cfg)
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85)))
        assert sm.snapshot().bars_remaining == 4  # 5 - 1 (just entered)
        sm.on_bar(mk_bar(ts(1), mk_signal("LONG", 70)))
        assert sm.snapshot().bars_remaining == 3
        sm.on_bar(mk_bar(ts(2), mk_signal("LONG", 70)))
        assert sm.snapshot().bars_remaining == 2

    def test_snapshot_to_dict_is_json_friendly(self):
        import json
        sm = SignalStateMachine(StateMachineConfig(confirm_bars=1))
        sm.on_bar(mk_bar(ts(0), mk_signal("LONG", 85, entry=100)))
        snap = sm.snapshot()
        payload = snap.to_dict()
        # Must be serializable
        json.dumps(payload)
        assert payload["state"] == "BUY"
        assert payload["direction"] == "LONG"
        assert payload["active_signal"]["signal_id"] == "sid-L"


# =============================================================================
# END-TO-END SCENARIO — a realistic sequence
# =============================================================================

class TestRealisticScenario:
    def test_full_session_multiple_cycles(self):
        """
        Simulate a 40-bar session with multiple arm/exit cycles.
        Validates that the state sequence never violates invariants.
        """
        cfg = StateMachineConfig(
            enter_threshold=75, exit_threshold=55,
            confirm_bars=2, cooldown_bars=2, max_signal_age_bars=6,
        )
        sm = SignalStateMachine(cfg)

        # Scripted scenario: (score, direction, close, high, low, regime)
        scenario: List[Tuple[Optional[int], str, float, float, float, Optional[str]]] = [
            (None, "NONE", 100.0, 100.3, 99.7, "normal"),   # 0 HOLD
            (85, "LONG", 100.0, 100.4, 99.8, "normal"),     # 1 arming 1/2
            (82, "LONG", 100.1, 100.5, 99.9, "normal"),     # 2 CONFIRM → BUY
            (78, "LONG", 100.5, 100.9, 100.2, "normal"),    # 3 active
            (76, "LONG", 101.0, 101.4, 100.7, "normal"),    # 4 active
            (75, "LONG", 101.5, 101.9, 101.2, "normal"),    # 5 active
            (None, "NONE", 101.8, 102.2, 101.5, "normal"),  # 6 silent 1
            (None, "NONE", 102.0, 102.5, 101.7, "normal"),  # 7 silent 2 → EXIT score_decayed
            (None, "NONE", 102.0, 102.3, 101.8, "normal"),  # 8 cooldown 1
            (None, "NONE", 101.9, 102.1, 101.6, "normal"),  # 9 cooldown 2
            (None, "NONE", 101.7, 101.9, 101.4, "normal"),  # 10 idle
            (90, "SHORT", 101.5, 101.7, 101.2, "normal"),   # 11 arming 1/2
            (88, "SHORT", 101.3, 101.5, 101.0, "normal"),   # 12 CONFIRM → SELL
        ]
        states = []
        for i, (score, dir_, c, h, l, reg) in enumerate(scenario):
            sig = mk_signal(dir_, score, entry=c, sl_dist=2, tp_dist=4) if score is not None else None
            snap, t = sm.on_bar(mk_bar(ts(i), sig, close=c, high=h, low=l, vol_regime=reg))
            states.append(snap.state)

        # Invariants
        expected = [
            PublicState.HOLD,        # 0
            PublicState.HOLD,        # 1 arming
            PublicState.BUY,         # 2 confirmed
            PublicState.BUY,
            PublicState.BUY,
            PublicState.BUY,
            PublicState.BUY,         # 6 silent 1, still active
            PublicState.HOLD,        # 7 exit
            PublicState.HOLD,
            PublicState.HOLD,
            PublicState.HOLD,
            PublicState.HOLD,        # 11 arming SHORT
            PublicState.SELL,        # 12 confirmed SHORT
        ]
        assert states == expected

        # Sanity: no direct BUY→SELL flip anywhere
        for prev, curr in zip(states, states[1:]):
            if prev is PublicState.BUY:
                assert curr in (PublicState.BUY, PublicState.HOLD)
            if prev is PublicState.SELL:
                assert curr in (PublicState.SELL, PublicState.HOLD)
