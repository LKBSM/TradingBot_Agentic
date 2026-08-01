"""PERF-1 — interactive read-through of the local candle cache.

On a cache miss the /app read must never hang on a slow/failing data feed. These
tests pin the contract of ``MarketReadingAssembler._fetch_candles_for_build`` and
its wiring through ``get_or_generate``:

  - provider fast  → provider candles win, the candle cache is NOT consulted;
  - provider fails → read THROUGH candles.db (real bars, staleness badged);
  - provider slow  → bounded, degrades to the cache within the budget (no 20s hang);
  - provider + cache both empty → the failure is surfaced, never a blank reading;
  - background scheduler path (``bound_provider=False``) stays patient and does
    NOT fall back to the cache — it must wait for the feed to advance candles.db;
  - NON-REGRESSION: the SMC pipeline sees byte-identical candles whether they came
    from the provider or the read-through, across all six timeframes — so detection
    output cannot change as a side effect of the sourcing path.
"""

import threading
import time
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Sequence

import pytest

from src.intelligence.market_reading_assembler import (
    MarketReadingAssembler,
    MarketReadingDataUnavailable,
)
from src.intelligence.market_reading_schema import MarketReading

_TF_MINUTES = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


class _Candle:
    """Duck-typed Candle (ts, open, high, low, close, volume)."""

    def __init__(self, ts: datetime, close: float):
        self.ts = ts
        self.open = close - 0.5
        self.high = close + 1.0
        self.low = close - 1.0
        self.close = close
        self.volume = 100.0

    def key(self) -> tuple:
        return (self.ts, self.open, self.high, self.low, self.close, self.volume)


def _candles(n: int, tf: str = "M15", base: float = 2300.0,
             start: Optional[datetime] = None) -> List[_Candle]:
    step = timedelta(minutes=_TF_MINUTES[tf])
    start = start or datetime(2026, 5, 20, 0, 0, 0, tzinfo=timezone.utc)
    return [_Candle(start + step * i, base + i * 2.0) for i in range(n)]


class _Provider:
    """fetch_candles that can return, raise, or block on an Event."""

    def __init__(self, candles: Optional[Sequence[_Candle]] = None,
                 raise_exc: Optional[BaseException] = None,
                 block: Optional[threading.Event] = None):
        self._candles = list(candles or [])
        self._raise = raise_exc
        self._block = block
        self.calls = 0

    def fetch_candles(self, instrument: str, timeframe: str, count: int) -> List[_Candle]:
        self.calls += 1
        if self._block is not None:
            # Wait until released (or a hard ceiling so a bug can't hang the suite).
            self._block.wait(timeout=10.0)
        if self._raise is not None:
            raise self._raise
        return self._candles[-count:]


class _CandlesStore:
    def __init__(self, cached: Optional[Sequence[_Candle]] = None):
        self._cached = list(cached or [])
        self.get_calls = 0
        self.upserts: List[int] = []

    def get_last_n_candles(self, instrument: str, timeframe: str, n: int) -> List[_Candle]:
        self.get_calls += 1
        if n <= 0:
            return []
        return self._cached[-n:]

    def upsert_candles(self, instrument: str, timeframe: str, candles: list) -> int:
        self.upserts.append(len(candles))
        return len(candles)


class _ReadingsStore:
    def __init__(self, prepopulated: Optional[dict] = None):
        self._latest = prepopulated
        self.save_calls: List[tuple] = []
        self.mark_active_calls: List[tuple] = []

    def get_latest_reading(self, instrument: str, timeframe: str) -> Optional[dict]:
        return self._latest

    def save_reading(self, instrument, timeframe, candle_close_ts, payload) -> int:
        self.save_calls.append((instrument, timeframe, candle_close_ts, payload))
        self._latest = payload
        return len(self.save_calls)

    def mark_combination_active(self, instrument, timeframe) -> None:
        self.mark_active_calls.append((instrument, timeframe))


def _stub_pipeline(candles):
    return (
        {"BOS_SIGNAL": 1.0, "BOS_EVENT": 1.0, "FVG_SIGNAL": 1.0,
         "OB_STRENGTH_NORM": 0.6, "ATR": 5.0},
        None,
    )


def _clock():
    # Well AFTER the synthetic candles so the feed lags the wall clock → a miss.
    return datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_assembler(provider, candles_store, readings_store=None,
                    pipeline=_stub_pipeline, timeout=5.0) -> MarketReadingAssembler:
    return MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store or _ReadingsStore(),
        candles_store=candles_store,
        smc_pipeline=pipeline,
        clock=_clock,
        provider_fetch_timeout_s=timeout,
    )


# --------------------------------------------------------------------------- #
# Helper-level contract
# --------------------------------------------------------------------------- #


def test_provider_success_wins_and_cache_not_read():
    provider = _Provider(candles=_candles(60))
    store = _CandlesStore(cached=_candles(60))
    asm = _make_assembler(provider, store)

    out = asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=True)

    assert provider.calls == 1
    assert store.get_calls == 0  # the cache is only a fallback, never the first choice
    assert [c.key() for c in out] == [c.key() for c in provider._candles]


def test_provider_failure_reads_through_cache():
    cached = _candles(40)
    provider = _Provider(raise_exc=RuntimeError("feed down"))
    store = _CandlesStore(cached=cached)
    asm = _make_assembler(provider, store)

    out = asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=True)

    assert provider.calls == 1
    assert store.get_calls == 1
    assert [c.key() for c in out] == [c.key() for c in cached]


def test_provider_empty_response_reads_through_cache():
    cached = _candles(40)
    provider = _Provider(candles=[])  # returns nothing, no exception
    store = _CandlesStore(cached=cached)
    asm = _make_assembler(provider, store)

    out = asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=True)

    assert [c.key() for c in out] == [c.key() for c in cached]


def test_slow_provider_is_bounded_and_degrades_fast():
    release = threading.Event()
    provider = _Provider(candles=_candles(60), block=release)
    store = _CandlesStore(cached=_candles(40))
    asm = _make_assembler(provider, store, timeout=0.2)

    try:
        t0 = time.perf_counter()
        out = asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=True)
        elapsed = time.perf_counter() - t0

        # Bounded to ~0.2s, NOT the provider's 10s block: served from cache.
        assert elapsed < 2.0
        assert [c.key() for c in out] == [c.key() for c in store._cached]
    finally:
        release.set()  # let the background worker exit promptly


def test_provider_and_cache_both_empty_raises_typed_no_data_chaining_cause():
    provider = _Provider(raise_exc=ValueError("no key"))
    store = _CandlesStore(cached=[])  # nothing cached either
    asm = _make_assembler(provider, store)

    # Typed no-data error (route maps it to a distinct 404), with the real provider
    # failure chained as __cause__ so nothing is swallowed from the logs.
    with pytest.raises(MarketReadingDataUnavailable) as excinfo:
        asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=True)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "no key" in str(excinfo.value.__cause__)


def test_background_path_is_patient_and_never_reads_cache():
    provider = _Provider(raise_exc=RuntimeError("feed down"))
    store = _CandlesStore(cached=_candles(40))
    asm = _make_assembler(provider, store)

    # bound_provider=False (scheduler): the error must propagate untouched and the
    # cache must NOT be consulted — the background job's contract is to advance the
    # real feed, and its caller (scheduler tick) already isolates the failure.
    with pytest.raises(RuntimeError, match="feed down"):
        asm._fetch_candles_for_build("XAUUSD", "M15", 500, bound_provider=False)
    assert store.get_calls == 0


# --------------------------------------------------------------------------- #
# End-to-end wiring through get_or_generate
# --------------------------------------------------------------------------- #


def test_get_or_generate_serves_reading_from_cache_when_feed_down():
    cached = _candles(60)
    provider = _Provider(raise_exc=RuntimeError("feed down"))
    store = _CandlesStore(cached=cached)
    asm = _make_assembler(provider, store)

    reading = asm.get_or_generate("XAUUSD", "M15")  # interactive default

    assert isinstance(reading, MarketReading)
    # The header names the last CACHED candle (real data) — the freshness badge,
    # computed from candle_close_ts vs the clock, is what tells the user it lags.
    span = timedelta(minutes=_TF_MINUTES["M15"])
    assert reading.header.candle_close_ts == cached[-1].ts + span
    assert store.upserts  # cached candles were (idempotently) re-persisted


def test_get_or_generate_raises_when_no_data_anywhere():
    provider = _Provider(raise_exc=RuntimeError("feed down"))
    store = _CandlesStore(cached=[])
    asm = _make_assembler(provider, store, readings_store=_ReadingsStore(None))

    # No live feed, no cached candles, no stored reading → honest failure, never a
    # blank/synthetic reading served as if real.
    with pytest.raises(Exception):
        asm.get_or_generate("XAUUSD", "M15")


# --------------------------------------------------------------------------- #
# NON-REGRESSION: detection sees identical candles regardless of source (6 TFs)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
def test_detection_input_identical_provider_vs_readthrough(tf):
    """The SMC pipeline must receive byte-identical candles whether they arrived
    from the provider (unchanged path) or the read-through fallback — so PERF-1
    cannot alter what BOS/CHOCH/OB/FVG/liquidity detection sees."""
    seen: List[list] = []

    def _spy_pipeline(candles):
        seen.append([c.key() for c in candles])
        return _stub_pipeline(candles)

    data = _candles(120, tf=tf)

    # Path A: provider succeeds (the pre-PERF-1 behaviour).
    asm_a = _make_assembler(_Provider(candles=data), _CandlesStore(cached=[]),
                            pipeline=_spy_pipeline)
    asm_a.get_or_generate("XAUUSD", tf)

    # Path B: provider down → read-through the same candles from the cache.
    asm_b = _make_assembler(_Provider(raise_exc=RuntimeError("feed down")),
                            _CandlesStore(cached=data), pipeline=_spy_pipeline)
    asm_b.get_or_generate("XAUUSD", tf)

    assert len(seen) == 2
    assert seen[0] == seen[1], f"{tf}: detection input differs between provider and read-through"
