"""Tests for MarketReadingScheduler (Chantier 3 Étape 4 — hybrid mode).

The scheduler regenerates MarketReadings for *active* (instrument, timeframe)
combinations whenever a new candle has closed since the stored reading.
Behaviour covered:

  1. ``tick()`` regenerates a combination only when ``_needs_regeneration``
     fires (no stored reading, stored reading older than expected close, or
     stored ``candle_close_ts`` is unparsable).
  2. ``tick()`` is idempotent — calling it twice for the same expected close
     only regenerates once.
  3. One combination failing must NOT abort the whole tick (per-combination
     exception isolation).
  4. A failure inside ``get_active_combinations`` must NOT crash the tick
     itself (global isolation — the BackgroundScheduler thread keeps running).
  5. Inactive combinations (last_accessed_at < now - auto_stop_hours) are
     silently dropped — handled at the store level via ``since=``, the
     scheduler only forwards the cutoff.
  6. ``_parse_iso`` accepts ISO-8601 strings, datetimes (naive→UTC, aware→UTC),
     and returns None for garbage.
  7. Lifecycle: ``start`` registers the job, ``stop`` shuts down cleanly,
     ``stop`` is safe to call when not running, ``running`` reflects state.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import pytest

# Skip the whole module gracefully when APScheduler is not installed in the
# test environment (the scheduler constructor imports it lazily).
pytest.importorskip("apscheduler")

from src.intelligence.scheduler import (  # noqa: E402
    MarketReadingScheduler,
    _parse_iso,
)


CLOCK = datetime(2026, 5, 28, 14, 23, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Mocks
# --------------------------------------------------------------------------- #
class _MockReadingsStore:
    """Minimal in-memory readings store fitting the scheduler's contract."""

    def __init__(
        self,
        active: Optional[list[tuple[str, str]]] = None,
        readings: Optional[dict[tuple[str, str], dict]] = None,
        raise_on_active: bool = False,
    ):
        self._active = active or []
        self._readings = readings or {}
        self._raise_on_active = raise_on_active
        self.active_calls: list[datetime] = []

    def get_active_combinations(self, since):
        self.active_calls.append(since)
        if self._raise_on_active:
            raise RuntimeError("store boom")
        return list(self._active)

    def get_latest_reading(self, instrument, timeframe):
        return self._readings.get((instrument, timeframe))


class _MockAssembler:
    """Records (instrument, timeframe) regeneration calls."""

    def __init__(self, raise_for: Optional[set[tuple[str, str]]] = None):
        self.calls: list[tuple[str, str]] = []
        self._raise_for = raise_for or set()

    def get_or_generate(self, instrument, timeframe):
        self.calls.append((instrument, timeframe))
        if (instrument, timeframe) in self._raise_for:
            raise RuntimeError(f"assembler boom for {instrument}/{timeframe}")
        return None  # return value is ignored by the scheduler


def _reading_payload(candle_close_ts: datetime) -> dict:
    return {"header": {"candle_close_ts": candle_close_ts.isoformat()}}


def _clock():
    return CLOCK


# =========================================================================== #
# _parse_iso
# =========================================================================== #
class TestParseIso:
    def test_accepts_iso_string_with_offset(self):
        dt = _parse_iso("2026-05-28T14:00:00+00:00")
        assert dt == datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc)

    def test_accepts_iso_string_with_z_suffix(self):
        dt = _parse_iso("2026-05-28T14:00:00Z")
        assert dt == datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc)

    def test_accepts_aware_datetime_passthrough(self):
        src = datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc)
        assert _parse_iso(src) == src

    def test_naive_datetime_assumed_utc(self):
        src = datetime(2026, 5, 28, 14, 0, 0)
        dt = _parse_iso(src)
        assert dt == datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc)
        assert dt.tzinfo is not None

    def test_returns_none_on_garbage(self):
        assert _parse_iso("not-a-date") is None
        assert _parse_iso(None) is None
        assert _parse_iso(12345) is None


# =========================================================================== #
# _needs_regeneration
# =========================================================================== #
class TestNeedsRegeneration:
    def _build(self, readings: Optional[dict] = None) -> MarketReadingScheduler:
        return MarketReadingScheduler(
            assembler=_MockAssembler(),
            readings_store=_MockReadingsStore(readings=readings),
            clock=_clock,
        )

    def test_true_when_no_stored_reading(self):
        sched = self._build(readings={})
        assert sched._needs_regeneration("XAUUSD", "M15", CLOCK) is True

    def test_true_when_stored_reading_is_stale(self):
        # Expected close for M15 at 14:23 → 14:15. Stale stored close = 13:30.
        old = datetime(2026, 5, 28, 13, 30, 0, tzinfo=timezone.utc)
        sched = self._build(readings={("XAUUSD", "M15"): _reading_payload(old)})
        assert sched._needs_regeneration("XAUUSD", "M15", CLOCK) is True

    def test_false_when_stored_reading_matches_expected_close(self):
        # Expected close for M15 at 14:23 → 14:15. Fresh stored close = 14:15.
        fresh = datetime(2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc)
        sched = self._build(readings={("XAUUSD", "M15"): _reading_payload(fresh)})
        assert sched._needs_regeneration("XAUUSD", "M15", CLOCK) is False

    def test_true_when_header_missing(self):
        sched = self._build(readings={("XAUUSD", "M15"): {"no_header": True}})
        assert sched._needs_regeneration("XAUUSD", "M15", CLOCK) is True

    def test_true_when_candle_close_ts_unparsable(self):
        sched = self._build(
            readings={("XAUUSD", "M15"): {"header": {"candle_close_ts": "garbage"}}}
        )
        assert sched._needs_regeneration("XAUUSD", "M15", CLOCK) is True


# =========================================================================== #
# tick()
# =========================================================================== #
class TestTick:
    def test_regenerates_stale_combination(self):
        old = datetime(2026, 5, 28, 13, 30, 0, tzinfo=timezone.utc)
        store = _MockReadingsStore(
            active=[("XAUUSD", "M15")],
            readings={("XAUUSD", "M15"): _reading_payload(old)},
        )
        assembler = _MockAssembler()
        sched = MarketReadingScheduler(assembler, store, clock=_clock)

        regen = sched.tick()
        assert regen == 1
        assert assembler.calls == [("XAUUSD", "M15")]

    def test_skips_fresh_combination(self):
        fresh = datetime(2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc)
        store = _MockReadingsStore(
            active=[("XAUUSD", "M15")],
            readings={("XAUUSD", "M15"): _reading_payload(fresh)},
        )
        assembler = _MockAssembler()
        sched = MarketReadingScheduler(assembler, store, clock=_clock)

        regen = sched.tick()
        assert regen == 0
        assert assembler.calls == []

    def test_idempotent_on_second_tick_without_new_candle(self):
        old = datetime(2026, 5, 28, 13, 30, 0, tzinfo=timezone.utc)
        fresh = datetime(2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc)
        readings = {("XAUUSD", "M15"): _reading_payload(old)}
        store = _MockReadingsStore(active=[("XAUUSD", "M15")], readings=readings)

        # Simulate the assembler refreshing the stored reading on regeneration.
        def _refresh(instrument, timeframe):
            readings[(instrument, timeframe)] = _reading_payload(fresh)

        class _RefreshingAssembler:
            calls: list[tuple[str, str]] = []

            def get_or_generate(self, instrument, timeframe):
                _RefreshingAssembler.calls.append((instrument, timeframe))
                _refresh(instrument, timeframe)

        sched = MarketReadingScheduler(_RefreshingAssembler(), store, clock=_clock)

        assert sched.tick() == 1
        # Second tick at the same wall-clock — no new candle has closed.
        assert sched.tick() == 0
        assert _RefreshingAssembler.calls == [("XAUUSD", "M15")]

    def test_exception_in_one_combination_does_not_abort_tick(self):
        # Use an old-enough timestamp to be stale across M15/H1/H4 simultaneously
        # (H4 boundary at CLOCK=14:23 is 12:00 — so "old" must be earlier).
        old = datetime(2026, 5, 28, 8, 0, 0, tzinfo=timezone.utc)
        store = _MockReadingsStore(
            active=[("XAUUSD", "M15"), ("EURUSD", "H1"), ("XAUUSD", "H4")],
            readings={
                ("XAUUSD", "M15"): _reading_payload(old),
                ("EURUSD", "H1"): _reading_payload(old),
                ("XAUUSD", "H4"): _reading_payload(old),
            },
        )
        assembler = _MockAssembler(raise_for={("EURUSD", "H1")})
        sched = MarketReadingScheduler(assembler, store, clock=_clock)

        regen = sched.tick()
        # 2 succeed, 1 fails — tick still returns successes, no exception bubbles.
        assert regen == 2
        # All 3 combinations were attempted in order despite the middle one failing.
        assert assembler.calls == [
            ("XAUUSD", "M15"),
            ("EURUSD", "H1"),
            ("XAUUSD", "H4"),
        ]

    def test_store_failure_returns_zero_and_does_not_raise(self):
        store = _MockReadingsStore(raise_on_active=True)
        assembler = _MockAssembler()
        sched = MarketReadingScheduler(assembler, store, clock=_clock)

        # Must NOT raise — the BackgroundScheduler thread would die otherwise.
        regen = sched.tick()
        assert regen == 0
        assert assembler.calls == []

    def test_no_active_combinations_is_a_noop(self):
        store = _MockReadingsStore(active=[])
        assembler = _MockAssembler()
        sched = MarketReadingScheduler(assembler, store, clock=_clock)
        assert sched.tick() == 0
        assert assembler.calls == []

    def test_active_combinations_queried_with_auto_stop_cutoff(self):
        store = _MockReadingsStore(active=[])
        sched = MarketReadingScheduler(
            assembler=_MockAssembler(),
            readings_store=store,
            auto_stop_hours=24,
            clock=_clock,
        )
        sched.tick()
        assert len(store.active_calls) == 1
        assert store.active_calls[0] == CLOCK - timedelta(hours=24)

    def test_custom_auto_stop_hours_propagates_to_cutoff(self):
        store = _MockReadingsStore(active=[])
        sched = MarketReadingScheduler(
            assembler=_MockAssembler(),
            readings_store=store,
            auto_stop_hours=2,
            clock=_clock,
        )
        sched.tick()
        assert store.active_calls[0] == CLOCK - timedelta(hours=2)


# =========================================================================== #
# Lifecycle: start / stop / running
# =========================================================================== #
class TestLifecycle:
    def _build(self) -> MarketReadingScheduler:
        return MarketReadingScheduler(
            assembler=_MockAssembler(),
            readings_store=_MockReadingsStore(),
            tick_interval_seconds=60,
            clock=_clock,
        )

    def test_not_running_before_start(self):
        sched = self._build()
        assert sched.running is False

    def test_start_registers_job_and_running_becomes_true(self):
        sched = self._build()
        try:
            sched.start()
            assert sched.running is True
            # The job is registered under the constant _JOB_ID
            from src.intelligence.scheduler import _JOB_ID
            assert sched._scheduler.get_job(_JOB_ID) is not None
        finally:
            sched.stop()

    def test_stop_idempotent_when_not_running(self):
        sched = self._build()
        # Must not raise even though start() was never called.
        sched.stop()
        assert sched.running is False

    def test_stop_after_start_brings_running_to_false(self):
        sched = self._build()
        sched.start()
        sched.stop()
        assert sched.running is False
