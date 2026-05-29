"""Tests for MarketReadingAssembler (Chantier 2 Étape 4)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pytest

from src.intelligence.market_reading_assembler import (
    MarketReadingAssembler,
    expected_last_candle_close,
)
from src.intelligence.market_reading_schema import MarketReading


class _MockCandle:
    """Duck-typed Candle (ts, open, high, low, close, volume)."""

    def __init__(self, ts: datetime, close: float):
        self.ts = ts
        self.open = close - 0.5
        self.high = close + 1.0
        self.low = close - 1.0
        self.close = close
        self.volume = 100.0


def _build_candles(n: int = 30, base: float = 2300.0) -> list[_MockCandle]:
    start = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)
    return [_MockCandle(start + timedelta(minutes=15 * i), base + i * 2.0) for i in range(n)]


class _MockDataProvider:
    def __init__(self, candles: Sequence[_MockCandle]):
        self._candles = list(candles)
        self.call_count = 0

    def fetch_candles(self, instrument: str, timeframe: str, count: int) -> list[_MockCandle]:
        self.call_count += 1
        return self._candles[-count:]


class _MockCandlesStore:
    def __init__(self):
        self.upsert_calls: list[tuple[str, str, int]] = []

    def upsert_candles(self, instrument: str, timeframe: str, candles: list[Any]) -> int:
        self.upsert_calls.append((instrument, timeframe, len(candles)))
        return len(candles)


class _MockReadingsStore:
    def __init__(self, prepopulated: Optional[dict] = None):
        self._latest = prepopulated
        self.save_calls: list[tuple[str, str, datetime, dict]] = []
        self.mark_active_calls: list[tuple[str, str]] = []
        self.get_latest_calls = 0

    def get_latest_reading(self, instrument: str, timeframe: str) -> Optional[dict]:
        self.get_latest_calls += 1
        return self._latest

    def save_reading(
        self, instrument: str, timeframe: str, candle_close_ts: datetime, payload: dict
    ) -> int:
        self.save_calls.append((instrument, timeframe, candle_close_ts, payload))
        # Mirror the persistence: subsequent get_latest_reading returns this.
        self._latest = payload
        return len(self.save_calls)

    def mark_combination_active(self, instrument: str, timeframe: str) -> None:
        self.mark_active_calls.append((instrument, timeframe))


def _stub_smc_pipeline(candles):
    return (
        {
            "BOS_SIGNAL": 1.0,
            "BOS_EVENT": 1.0,
            "FVG_SIGNAL": 1.0,
            "OB_STRENGTH_NORM": 0.6,
            "ATR": 5.0,
        },
        None,
    )


@pytest.fixture
def fixed_clock():
    """A clock locked at 2026-05-28 14:23:00Z (last M15 close = 14:15)."""

    def _clock() -> datetime:
        return datetime(2026, 5, 28, 14, 23, 0, tzinfo=timezone.utc)

    return _clock


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------


def test_expected_last_candle_close_m15():
    now = datetime(2026, 5, 28, 14, 23, 0, tzinfo=timezone.utc)
    assert expected_last_candle_close("M15", now) == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )


def test_expected_last_candle_close_h1():
    now = datetime(2026, 5, 28, 14, 59, 0, tzinfo=timezone.utc)
    assert expected_last_candle_close("H1", now) == datetime(
        2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc
    )


def test_expected_last_candle_close_h4():
    now = datetime(2026, 5, 28, 14, 23, 0, tzinfo=timezone.utc)
    # H4 boundaries at 00, 04, 08, 12, 16, 20 UTC → last passed = 12:00
    assert expected_last_candle_close("H4", now) == datetime(
        2026, 5, 28, 12, 0, 0, tzinfo=timezone.utc
    )


def test_expected_last_candle_close_naive_datetime_assumed_utc():
    naive = datetime(2026, 5, 28, 14, 23, 0)
    assert expected_last_candle_close("M15", naive) == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )


def test_expected_last_candle_close_unsupported_tf():
    now = datetime(2026, 5, 28, tzinfo=timezone.utc)
    with pytest.raises(ValueError):
        expected_last_candle_close("X42", now)


# ---------------------------------------------------------------------------
# Lazy cache miss — full pipeline
# ---------------------------------------------------------------------------


def test_lazy_cache_miss_runs_full_pipeline(fixed_clock):
    candles = _build_candles(30)
    provider = _MockDataProvider(candles)
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=None)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )

    reading = assembler.get_or_generate("XAUUSD", "M15")

    assert isinstance(reading, MarketReading)
    assert reading.header.instrument == "XAUUSD"
    assert reading.header.timeframe == "M15"
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )
    assert reading.header.close_price == candles[-1].close
    assert reading.structure.bos is not None  # populated from stub_smc_pipeline
    assert reading.conditions.description_source == "template_fallback"

    # Provider called exactly once (fetched fresh candles)
    assert provider.call_count == 1
    # Candles persisted
    assert candles_store.upsert_calls == [("XAUUSD", "M15", 30)]
    # Reading persisted
    assert len(readings_store.save_calls) == 1
    # Combination marked active
    assert readings_store.mark_active_calls == [("XAUUSD", "M15")]


# ---------------------------------------------------------------------------
# Lazy cache hit — no fetch, no scan, returns stored
# ---------------------------------------------------------------------------


def test_lazy_cache_hit_returns_stored_without_fetch(fixed_clock):
    # Pre-build a valid MarketReading whose candle_close_ts matches the expected
    # M15 close at 14:23:00Z → 14:15:00Z.
    seed_assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(30)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    seed_reading = seed_assembler.get_or_generate("XAUUSD", "M15")

    # Now build a fresh assembler whose store is pre-populated with that payload.
    payload = seed_reading.model_dump(mode="json")
    provider = _MockDataProvider(_build_candles(30))
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=payload)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )

    reading = assembler.get_or_generate("XAUUSD", "M15")

    assert isinstance(reading, MarketReading)
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )
    # Critical: NO fetch, NO candles upsert, NO save (cache hit)
    assert provider.call_count == 0
    assert candles_store.upsert_calls == []
    assert readings_store.save_calls == []
    # But mark_combination_active is still called (lazy hybrid mode: every
    # access keeps the combination warm for Chantier 3 scheduler).
    assert readings_store.mark_active_calls == [("XAUUSD", "M15")]


# ---------------------------------------------------------------------------
# Stale cache — regeneration
# ---------------------------------------------------------------------------


def test_stale_cache_triggers_regeneration(fixed_clock):
    # Stored reading has an OLD candle_close_ts (one bar before expected)
    stale_payload = {
        "schema_version": "2.0.0",
        "header": {
            "instrument": "XAUUSD",
            "timeframe": "M15",
            "candle_close_ts": "2026-05-28T14:00:00Z",  # 1 bar older than expected 14:15
            "close_price": 2300.0,
        },
        "structure": {
            "bos": None, "choch": None, "order_blocks": [], "fair_value_gaps": [],
            "retest_in_progress": None,
        },
        "regime": {
            "trend": "neutral", "volatility_observed": "normal",
            "market_phase": "accumulation", "mtf_confluence": {},
        },
        "events": {
            "news_upcoming": [], "news_just_published": [], "technical_triggers_recent": [],
        },
        "conditions": {
            "tags": ["stale"], "description": "Stale.", "description_source": "template_fallback",
        },
    }
    candles = _build_candles(30)
    provider = _MockDataProvider(candles)
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=stale_payload)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )

    reading = assembler.get_or_generate("XAUUSD", "M15")

    # Regenerated — fresh candle_close_ts
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )
    assert provider.call_count == 1
    assert len(candles_store.upsert_calls) == 1
    assert len(readings_store.save_calls) == 1
    assert readings_store.mark_active_calls == [("XAUUSD", "M15")]


# ---------------------------------------------------------------------------
# Pydantic validation passes on assembled output
# ---------------------------------------------------------------------------


def test_assembled_output_validates_against_pydantic_schema(fixed_clock):
    candles = _build_candles(30)
    provider = _MockDataProvider(candles)
    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")

    # Roundtrip through JSON to exercise full Pydantic validation
    serialized = reading.model_dump_json()
    reparsed = MarketReading.model_validate_json(serialized)
    assert reparsed == reading


# ---------------------------------------------------------------------------
# Description engine injection (Étape 5 wiring contract)
# ---------------------------------------------------------------------------


class _StubDescriptionEngine:
    def __init__(self, description: str = "Stub description from engine.", source: str = "haiku_generated"):
        self._description = description
        self._source = source
        self.calls: list[tuple[list[str], Any]] = []

    def generate(self, tags, regime):
        self.calls.append((list(tags), regime))
        return self._description, self._source


def test_description_engine_used_when_injected(fixed_clock):
    engine = _StubDescriptionEngine()
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(30)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        description_engine=engine,
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")
    assert reading.conditions.description == "Stub description from engine."
    assert reading.conditions.description_source == "haiku_generated"
    assert len(engine.calls) == 1


def test_description_engine_failure_falls_back_to_template(fixed_clock):
    class _FailingEngine:
        def generate(self, tags, regime):
            raise RuntimeError("LLM down")

    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(30)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        description_engine=_FailingEngine(),
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")
    # Falls back: still produces a valid reading via template
    assert reading.conditions.description_source == "template_fallback"
    assert len(reading.conditions.description) > 0
