"""Tests for state_persistence — save/load roundtrip + staleness guard."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.intelligence.signal_state_machine import (
    BarInput,
    Direction,
    PublicState,
    SignalStateMachine,
    StateMachineConfig,
)
from src.intelligence.state_persistence import (
    load_state_machine,
    save_state_machine,
)


# =============================================================================
# ROUND-TRIP
# =============================================================================


def test_save_and_load_empty_machine_roundtrips(tmp_path: Path) -> None:
    machine = SignalStateMachine()
    target = tmp_path / "state.json"

    assert save_state_machine(machine, target) is True
    assert target.exists()

    restored = load_state_machine(target)
    assert restored is not None
    assert restored.snapshot().state is PublicState.HOLD


def test_save_preserves_active_signal_and_phase(tmp_path: Path) -> None:
    """Drive the machine to an ACTIVE phase, save, reload, expect identity."""
    from src.intelligence.confluence_detector import (
        ConfluenceSignal,
        SignalTier,
        SignalType,
    )

    cfg = StateMachineConfig(
        enter_threshold=50.0, exit_threshold=25.0,
        confirm_bars=1, cooldown_bars=1, max_signal_age_bars=20,
    )
    m = SignalStateMachine(cfg)
    sig = ConfluenceSignal(
        signal_id="abc123",
        symbol="XAUUSD",
        signal_type=SignalType.LONG,
        confluence_score=80.0,
        tier=SignalTier.PREMIUM,
        entry_price=2000.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        rr_ratio=2.0,
        atr=5.0,
    )
    # Arm
    m.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:00:00",
        high=2001.0, low=1999.0, close=2000.0,
        signal=sig, vol_regime="normal",
    ))
    # Confirm (entry)
    m.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:15:00",
        high=2002.0, low=1999.5, close=2001.0,
        signal=sig, vol_regime="normal",
    ))
    pre = m.snapshot()
    assert pre.state is PublicState.BUY
    assert pre.direction is Direction.LONG

    target = tmp_path / "state.json"
    assert save_state_machine(m, target) is True

    restored = load_state_machine(target)
    assert restored is not None
    post = restored.snapshot()
    assert post.state is PublicState.BUY
    assert post.direction is Direction.LONG
    # active_signal is kept as the stored dict unless a rehydrator is supplied
    assert restored._active_signal is not None
    # Stats counters carried over
    assert restored.get_stats()["arms_started"] == m.get_stats()["arms_started"]


# =============================================================================
# FILE ISSUES
# =============================================================================


def test_load_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_state_machine(tmp_path / "does_not_exist.json") is None


def test_load_corrupt_file_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{this is not json", encoding="utf-8")
    assert load_state_machine(p) is None


def test_load_wrong_schema_returns_none(tmp_path: Path) -> None:
    p = tmp_path / "wrong_schema.json"
    p.write_text(json.dumps({"schema_version": 999, "phase": "idle"}), encoding="utf-8")
    assert load_state_machine(p) is None


# =============================================================================
# ATOMIC WRITE
# =============================================================================


def test_atomic_write_does_not_leave_temp_files(tmp_path: Path) -> None:
    machine = SignalStateMachine()
    target = tmp_path / "state.json"
    save_state_machine(machine, target)
    # tempdir should have exactly one file — no ".tmp" leftovers
    files = list(tmp_path.iterdir())
    assert len(files) == 1
    assert files[0] == target


def test_save_overwrites_existing_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    m1 = SignalStateMachine()
    save_state_machine(m1, target)
    bytes_a = target.read_bytes()

    # Advance the machine a little, then save again
    m2 = SignalStateMachine()
    m2.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:00:00",
        high=101.0, low=99.0, close=100.0, signal=None, vol_regime="normal",
    ))
    save_state_machine(m2, target)
    bytes_b = target.read_bytes()
    # The two snapshots differ in last_bar_ts
    assert bytes_a != bytes_b


# =============================================================================
# STALENESS GUARD
# =============================================================================


def test_stale_snapshot_is_discarded(tmp_path: Path) -> None:
    """A snapshot older than ``max_staleness_bars`` is dropped."""
    m = SignalStateMachine()
    # Advance the machine so last_bar_ts is set
    m.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:00:00",
        high=101.0, low=99.0, close=100.0, signal=None, vol_regime="normal",
    ))
    target = tmp_path / "state.json"
    save_state_machine(m, target)

    # Current bar is 10 hours later — that's 40 x 15min bars ahead.
    restored = load_state_machine(
        target,
        current_bar_ts="2026-04-23T20:00:00",
        max_staleness_bars=4,
        bar_minutes=15,
    )
    assert restored is None, "Stale snapshot should be discarded"


def test_fresh_snapshot_is_loaded(tmp_path: Path) -> None:
    """Snapshot only 1 bar behind is kept."""
    m = SignalStateMachine()
    m.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:00:00",
        high=101.0, low=99.0, close=100.0, signal=None, vol_regime="normal",
    ))
    target = tmp_path / "state.json"
    save_state_machine(m, target)

    restored = load_state_machine(
        target,
        current_bar_ts="2026-04-23T10:15:00",  # exactly 1 bar later
        max_staleness_bars=4,
        bar_minutes=15,
    )
    assert restored is not None


def test_staleness_check_skipped_when_max_is_zero(tmp_path: Path) -> None:
    """Cold-start case: no current_bar_ts / max=0 → never discard."""
    m = SignalStateMachine()
    m.on_bar(BarInput(
        bar_timestamp="2020-01-01T00:00:00",  # ancient
        high=101.0, low=99.0, close=100.0, signal=None, vol_regime="normal",
    ))
    target = tmp_path / "state.json"
    save_state_machine(m, target)

    restored = load_state_machine(
        target,
        current_bar_ts="2026-04-23T10:00:00",
        max_staleness_bars=0,
    )
    assert restored is not None


# =============================================================================
# SCANNER INTEGRATION — shutdown writes, next start rehydrates
# =============================================================================


def test_scanner_shutdown_writes_snapshot_next_start_rehydrates(tmp_path: Path) -> None:
    """Full lifecycle: build scanner with a state machine, shutdown, verify
    the snapshot on disk, reload into a fresh scanner, verify state survives."""
    from unittest.mock import MagicMock

    from src.intelligence.sentinel_scanner import SentinelScanner

    target = tmp_path / "state_XAUUSD.json"

    # First scanner — advance the state machine, then shutdown
    sm = SignalStateMachine()
    sm.on_bar(BarInput(
        bar_timestamp="2026-04-23T10:00:00",
        high=101.0, low=99.0, close=100.0, signal=None, vol_regime="normal",
    ))
    scanner = SentinelScanner(
        data_provider=MagicMock(), smc_factory=MagicMock(),
        regime_agent=MagicMock(), news_agent=MagicMock(),
        confluence=MagicMock(), llm_engine=MagicMock(),
        cache=MagicMock(), signal_store=MagicMock(),
        state_machine=sm, persistence_path=target,
    )
    scanner.shutdown()  # should write snapshot
    assert target.exists()

    # Second scanner — fresh machine instance, then start (no blocking) and
    # verify the scanner swaps in the rehydrated machine.
    fresh_sm = SignalStateMachine()
    scanner2 = SentinelScanner(
        data_provider=MagicMock(), smc_factory=MagicMock(),
        regime_agent=MagicMock(), news_agent=MagicMock(),
        confluence=MagicMock(), llm_engine=MagicMock(),
        cache=MagicMock(), signal_store=MagicMock(),
        state_machine=fresh_sm, persistence_path=target,
    )
    scanner2._restore_state_machine()

    # The restored machine is not the same object as the freshly created one
    assert scanner2.state_machine is not fresh_sm
    # And its last_bar_ts matches what was saved
    assert scanner2.state_machine._last_bar_ts == "2026-04-23T10:00:00"


def test_scanner_without_persistence_path_is_noop(tmp_path: Path) -> None:
    """Scanners without ``persistence_path`` must never touch disk."""
    from unittest.mock import MagicMock

    from src.intelligence.sentinel_scanner import SentinelScanner

    sm = SignalStateMachine()
    scanner = SentinelScanner(
        data_provider=MagicMock(), smc_factory=MagicMock(),
        regime_agent=MagicMock(), news_agent=MagicMock(),
        confluence=MagicMock(), llm_engine=MagicMock(),
        cache=MagicMock(), signal_store=MagicMock(),
        state_machine=sm, persistence_path=None,
    )
    scanner.shutdown()  # should not crash, should not create files
    assert list(tmp_path.iterdir()) == []
