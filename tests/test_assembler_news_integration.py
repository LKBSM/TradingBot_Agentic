"""Tests for news injection into MarketReadingAssembler (Chantier 3 Étape 3).

Verifies: (1) with a news pipeline injected, events.news_* are populated;
(2) without one, events stay empty (Chantier 2 compat); (3) the populated
news respect the MarketReading Pydantic schema; (4) a pipeline failure does
not break reading generation.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pytest

from src.intelligence.market_reading_assembler import MarketReadingAssembler
from src.intelligence.market_reading_schema import (
    MarketReading,
    NewsJustPublished,
    NewsUpcoming,
)
from src.intelligence.news_pipeline import NewsPipeline
from src.storage import NewsCacheStore

CLOCK = datetime(2026, 5, 28, 14, 23, 0, tzinfo=timezone.utc)


# --------------------------------------------------------------------------- #
# Minimal mocks (mirroring tests/test_market_reading_assembler.py)
# --------------------------------------------------------------------------- #
class _MockCandle:
    def __init__(self, ts: datetime, close: float):
        self.ts = ts
        self.open = close - 0.5
        self.high = close + 1.0
        self.low = close - 1.0
        self.close = close
        self.volume = 100.0


def _candles(n: int = 30, base: float = 2300.0) -> list[_MockCandle]:
    start = datetime(2026, 5, 28, 0, 0, 0, tzinfo=timezone.utc)
    return [_MockCandle(start + timedelta(minutes=15 * i), base + i * 2.0) for i in range(n)]


class _MockDataProvider:
    def fetch_candles(self, instrument, timeframe, count):
        return _candles()[-count:]


class _MockCandlesStore:
    def upsert_candles(self, instrument, timeframe, candles):
        return len(candles)


class _MockReadingsStore:
    def __init__(self):
        self._latest: Optional[dict] = None

    def get_latest_reading(self, instrument, timeframe):
        return self._latest

    def save_reading(self, instrument, timeframe, candle_close_ts, payload):
        self._latest = payload
        return 1

    def mark_combination_active(self, instrument, timeframe):
        pass


def _stub_smc(candles):
    return ({"ATR": 5.0}, None)


def _clock():
    return CLOCK


def _build_assembler(news_pipeline: Any) -> MarketReadingAssembler:
    return MarketReadingAssembler(
        data_provider=_MockDataProvider(),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc,
        news_pipeline=news_pipeline,
        clock=_clock,
    )


def _ff_event(title, country, dt, impact="high", actual="", forecast=""):
    return {
        "title": title, "country": country, "date": dt.isoformat(),
        "impact": impact, "actual": actual, "forecast": forecast, "previous": "",
    }


# --------------------------------------------------------------------------- #
# Stub news pipeline (records the `now` it receives)
# --------------------------------------------------------------------------- #
class _StubNewsPipeline:
    def __init__(self):
        self.upcoming_now = None
        self.published_now = None

    def get_upcoming(self, currency_filter=None, lookahead_minutes=240, now=None):
        self.upcoming_now = now
        return [
            NewsUpcoming(
                event="US NFP",
                scheduled_at=CLOCK + timedelta(minutes=30),
                time_to_event_min=30,
                impact="high",
                currency="USD",
                potential_effect_description="NFP (USD) — impact élevé, prévu dans 30 min.",
            )
        ]

    def get_just_published(self, currency_filter=None, lookback_minutes=60, now=None):
        self.published_now = now
        return [
            NewsJustPublished(
                event="EU CPI",
                published_at=CLOCK - timedelta(minutes=15),
                actual=2.9,
                forecast=3.2,
                surprise_direction="miss",
                currency="EUR",
                impact="high",
                potential_effect_description="EU CPI (EUR) — impact élevé, publié il y a 15 min.",
            )
        ]


# ===================================================================== #
# Tests
# ===================================================================== #
class TestNewsInjection:
    def test_events_populated_when_pipeline_injected(self):
        stub = _StubNewsPipeline()
        assembler = _build_assembler(stub)
        reading = assembler.get_or_generate("XAUUSD", "M15")
        assert len(reading.events.news_upcoming) == 1
        assert reading.events.news_upcoming[0].event == "US NFP"
        assert len(reading.events.news_just_published) == 1
        assert reading.events.news_just_published[0].event == "EU CPI"

    def test_pipeline_receives_wall_clock_now(self):
        stub = _StubNewsPipeline()
        assembler = _build_assembler(stub)
        assembler.get_or_generate("XAUUSD", "M15")
        assert stub.upcoming_now == CLOCK
        assert stub.published_now == CLOCK

    def test_currency_filter_passed_usd_eur(self):
        captured = {}

        class _CapturingPipeline(_StubNewsPipeline):
            def get_upcoming(self, currency_filter=None, lookahead_minutes=240, now=None):
                captured["upcoming_filter"] = currency_filter
                captured["lookahead"] = lookahead_minutes
                return []

            def get_just_published(self, currency_filter=None, lookback_minutes=60, now=None):
                captured["published_filter"] = currency_filter
                captured["lookback"] = lookback_minutes
                return []

        assembler = _build_assembler(_CapturingPipeline())
        assembler.get_or_generate("XAUUSD", "M15")
        assert captured["upcoming_filter"] == ["USD", "EUR"]
        assert captured["published_filter"] == ["USD", "EUR"]
        assert captured["lookahead"] == 240
        assert captured["lookback"] == 60


class TestChantier2Compat:
    def test_no_pipeline_keeps_events_empty(self):
        assembler = _build_assembler(news_pipeline=None)
        reading = assembler.get_or_generate("XAUUSD", "M15")
        assert reading.events.news_upcoming == []
        assert reading.events.news_just_published == []

    def test_pipeline_failure_falls_back_to_empty(self):
        class _BrokenPipeline:
            def get_upcoming(self, **kwargs):
                raise RuntimeError("boom")

            def get_just_published(self, **kwargs):
                raise RuntimeError("boom")

        assembler = _build_assembler(_BrokenPipeline())
        reading = assembler.get_or_generate("XAUUSD", "M15")
        assert reading.events.news_upcoming == []
        assert reading.events.news_just_published == []


class TestSchemaValidity:
    def test_reading_with_news_roundtrips_through_pydantic(self):
        stub = _StubNewsPipeline()
        assembler = _build_assembler(stub)
        reading = assembler.get_or_generate("XAUUSD", "M15")
        # Full JSON roundtrip proves the populated news block is schema-valid.
        dumped = reading.model_dump(mode="json")
        revalidated = MarketReading.model_validate(dumped)
        assert revalidated.events.news_upcoming[0].impact == "high"

    def test_end_to_end_real_pipeline(self, tmp_path):
        """Real NewsPipeline (injected fetch_fn) → assembler events populated."""
        cache = NewsCacheStore(db_path=str(tmp_path / "news.db"))
        raw = [
            _ff_event("US NFP", "USD", CLOCK + timedelta(minutes=30), impact="high"),
            _ff_event("EU CPI", "EUR", CLOCK - timedelta(minutes=15),
                      impact="high", actual="2.9", forecast="3.2"),
            _ff_event("UK GDP", "GBP", CLOCK + timedelta(minutes=20), impact="high"),
        ]
        pipeline = NewsPipeline(cache_store=cache, fetch_fn=lambda: raw, clock=_clock)
        assembler = _build_assembler(pipeline)
        reading = assembler.get_or_generate("XAUUSD", "M15")

        # Default USD/EUR filter → GBP excluded.
        up_currencies = {u.currency for u in reading.events.news_upcoming}
        assert up_currencies == {"USD"}  # only the +30min USD event is upcoming
        pub_currencies = {p.currency for p in reading.events.news_just_published}
        assert pub_currencies == {"EUR"}
        assert reading.events.news_just_published[0].surprise_direction == "miss"
