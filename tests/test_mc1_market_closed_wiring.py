"""MC-1 — end-to-end wiring: when the market is closed the assembler and the
scheduler make ZERO outbound data calls, re-emit NOTHING, and keep serving the
last reading frozen. Plus the holiday-only early-reopen safety probe.

All clocks are frozen (never the system clock).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

import pytest

pytest.importorskip("apscheduler")

from src.intelligence import market_calendar as mc  # noqa: E402
from src.intelligence.market_reading_assembler import MarketReadingAssembler  # noqa: E402
from src.intelligence.scheduler import MarketReadingScheduler  # noqa: E402

NY = ZoneInfo("America/New_York")

# Frozen wall clocks.
SATURDAY = datetime(2026, 7, 25, 12, 0, tzinfo=NY)       # weekend, market closed
WEDNESDAY = datetime(2026, 7, 22, 10, 7, tzinfo=NY)      # market open
XMAS = datetime(2026, 12, 25, 12, 0, tzinfo=NY)          # configured holiday


class _MockCandle:
    def __init__(self, ts: datetime, close: float):
        self.ts = ts
        self.open = close - 0.5
        self.high = close + 1.0
        self.low = close - 1.0
        self.close = close
        self.volume = 100.0


def _friday_candles(n: int = 40, base: float = 2300.0) -> list[_MockCandle]:
    """M15 candles ending exactly at the Friday 17:00 NY close."""
    end = datetime(2026, 7, 24, 17, 0, tzinfo=NY).astimezone(timezone.utc)
    start = end - timedelta(minutes=15 * n)
    return [_MockCandle(start + timedelta(minutes=15 * i), base + i) for i in range(n)]


class _CountingProvider:
    def __init__(self, candles: Sequence[_MockCandle]):
        self._candles = list(candles)
        self.call_count = 0

    def fetch_candles(self, instrument, timeframe, count):
        self.call_count += 1
        return self._candles[-count:]


class _CandlesStore:
    def upsert_candles(self, instrument, timeframe, candles):
        return len(candles)

    def get_last_n_candles(self, instrument, timeframe, n):
        return []


class _ReadingsStore:
    def __init__(self, prepopulated: Optional[dict] = None):
        self._latest = prepopulated
        self.save_calls: list = []

    def get_latest_reading(self, instrument, timeframe):
        return self._latest

    def get_active_combinations(self, since):
        return [("EURUSD", "M15")]

    def save_reading(self, instrument, timeframe, candle_close_ts, payload):
        self.save_calls.append((instrument, timeframe, candle_close_ts))
        self._latest = payload
        return len(self.save_calls)

    def mark_combination_active(self, instrument, timeframe):
        pass


def _stub_pipeline(candles):
    return ({"BOS_SIGNAL": 1.0, "FVG_SIGNAL": 1.0, "OB_STRENGTH_NORM": 0.6, "ATR": 5.0}, None)


def _seed_friday_reading(assembler: MarketReadingAssembler, store: _ReadingsStore) -> None:
    """Generate the Friday reading once (market open at generation time), then
    freeze the clock into the weekend for the assertions."""
    reading = assembler._build_fresh(
        "EURUSD", "M15", mc.market_aware_expected_close("EURUSD", "M15", SATURDAY)
    )
    store._latest = reading.model_dump(mode="json")


def _make(clock_dt, provider):
    store = _ReadingsStore()
    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=store,
        candles_store=_CandlesStore(),
        smc_pipeline=_stub_pipeline,
        clock=lambda: clock_dt.astimezone(timezone.utc),
    )
    return assembler, store


# --------------------------------------------------------------------------- #
# Emission lock — no rebuild, no fetch, no re-emission while closed
# --------------------------------------------------------------------------- #
def test_get_or_generate_makes_no_call_and_no_save_when_closed():
    provider = _CountingProvider(_friday_candles())
    assembler, store = _make(SATURDAY, provider)
    _seed_friday_reading(assembler, store)  # 1 fetch here (market was open)
    calls_after_seed = provider.call_count
    saves_after_seed = len(store.save_calls)

    # Two reads over the weekend: identical frozen reading, ZERO new work.
    r1 = assembler.get_or_generate("EURUSD", "M15")
    r2 = assembler.get_or_generate("EURUSD", "M15")

    assert provider.call_count == calls_after_seed  # no outbound Twelve Data call
    assert len(store.save_calls) == saves_after_seed  # no re-emission (no save)
    assert r1.header.candle_close_ts == r2.header.candle_close_ts
    # And the reading carries the closed status for the UI/agent.
    assert r1.market_status["state"] == "closed_weekend"
    assert r1.market_status["next_open_ts"] is not None


def test_scheduler_tick_makes_zero_provider_calls_on_weekend():
    provider = _CountingProvider(_friday_candles())
    assembler, store = _make(SATURDAY, provider)
    _seed_friday_reading(assembler, store)
    baseline = provider.call_count

    sched = MarketReadingScheduler(
        assembler,
        store,
        always_warm=[("EURUSD", "M15")],
        clock=lambda: SATURDAY.astimezone(timezone.utc),
    )
    for _ in range(5):  # five ticks across the weekend
        assert sched.tick() == 0

    assert provider.call_count == baseline  # not a single outbound call


# --------------------------------------------------------------------------- #
# Open market still works (no regression) + data_lagged surfaces
# --------------------------------------------------------------------------- #
def test_open_market_status_is_open():
    provider = _CountingProvider(_friday_candles())
    assembler, store = _make(WEDNESDAY, provider)
    reading = assembler.get_or_generate("EURUSD", "M15")
    assert reading.market_status["state"] in ("open", "data_lagged")


# --------------------------------------------------------------------------- #
# Holiday safety probe (early reopen) — fires only on holidays, honestly
# --------------------------------------------------------------------------- #
def test_safety_probe_only_on_holiday_not_weekend():
    provider = _CountingProvider(_friday_candles())
    assembler, store = _make(SATURDAY, provider)
    sched = MarketReadingScheduler(
        assembler, store, clock=lambda: SATURDAY.astimezone(timezone.utc)
    )
    # Weekend is deterministic → never probed.
    assert sched._should_safety_probe("EURUSD", "M15", SATURDAY.astimezone(timezone.utc)) is False


def test_safety_probe_rate_limited_on_holiday():
    provider = _CountingProvider(_friday_candles())
    assembler, store = _make(XMAS, provider)
    sched = MarketReadingScheduler(
        assembler,
        store,
        safety_poll_seconds=1800,
        clock=lambda: XMAS.astimezone(timezone.utc),
    )
    now = XMAS.astimezone(timezone.utc)
    assert sched._should_safety_probe("EURUSD", "M15", now) is True   # first probe due
    assert sched._should_safety_probe("EURUSD", "M15", now) is False  # rate-limited
    later = now + timedelta(seconds=1801)
    assert sched._should_safety_probe("EURUSD", "M15", later) is True  # window elapsed
