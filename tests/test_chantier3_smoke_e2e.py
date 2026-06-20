"""Chantier 3 smoke end-to-end (Étape 6).

Exercises the full wired flow with deterministic mocks (no network, no LLM):

    GET /api/market-reading
      → app_state.market_reading_assembler.get_or_generate
        → data provider + SMC pipeline + REAL NewsPipeline + description engine
      → MarketReading persisted, combination marked active
      → scheduler tick regenerates once a new candle closes
      → clean shutdown stops the scheduler

The assembler is the REAL MarketReadingAssembler so active_combinations,
persistence, and the events/news wiring are genuinely tested; only the data
provider, SMC pipeline and Haiku engine are stubbed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.signal_store import SignalStore
from src.intelligence.market_reading_assembler import (
    MarketReadingAssembler,
    expected_last_candle_close,
)
from src.intelligence.news_pipeline import NewsPipeline
from src.intelligence.scheduler import MarketReadingScheduler
from src.storage import MarketReadingsStore, NewsCacheStore

T0 = datetime(2026, 6, 5, 14, 23, 0, tzinfo=timezone.utc)


class _Clock:
    """Mutable clock so the test can advance time for the scheduler tick."""

    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


class _MockCandle:
    def __init__(self, ts, close):
        self.ts = ts
        self.open = close - 0.5
        self.high = close + 1.0
        self.low = close - 1.0
        self.close = close
        self.volume = 100.0


def _candles(n=40, base=2300.0):
    start = datetime(2026, 6, 4, 0, 0, 0, tzinfo=timezone.utc)
    return [_MockCandle(start + timedelta(hours=i), base + i * 2.0) for i in range(n)]


class _MockDataProvider:
    def fetch_candles(self, instrument, timeframe, count):
        return _candles()[-count:]


class _MockCandlesStore:
    def upsert_candles(self, instrument, timeframe, candles):
        return len(candles)


def _stub_smc(candles):
    # OB_STRENGTH_NORM > 0 → order block; BOS_SIGNAL/EVENT → confirmed BOS.
    return (
        {
            "OB_STRENGTH_NORM": 0.8,
            "BOS_SIGNAL": 1.0,
            "BOS_EVENT": 1.0,
            "FVG_SIGNAL": 1.0,
            "ATR": 5.0,
        },
        None,
    )


class _StubDescriptionEngine:
    def generate(self, tags, regime, structure, price, instrument):
        return "Conditions de marché décrites factuellement pour le test.", "haiku_generated"


def _ff_event(title, country, dt, impact="high", actual="", forecast=""):
    return {
        "title": title, "country": country, "date": dt.isoformat(),
        "impact": impact, "actual": actual, "forecast": forecast, "previous": "",
    }


@pytest.fixture
def wired(tmp_path):
    """Build a fully wired app (real assembler + scheduler) on a shared clock."""
    clock = _Clock(T0)
    readings_store = MarketReadingsStore(db_path=str(tmp_path / "mr.db"))
    news_cache = NewsCacheStore(db_path=str(tmp_path / "news.db"))
    news_pipeline = NewsPipeline(
        cache_store=news_cache,
        fetch_fn=lambda: [
            _ff_event("US Non-Farm Payrolls", "USD", T0 + timedelta(minutes=30), impact="high"),
            _ff_event("EU CPI", "EUR", T0 + timedelta(minutes=120), impact="medium"),
        ],
        clock=clock,
    )
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(),
        readings_store=readings_store,
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc,
        news_pipeline=news_pipeline,
        description_engine=_StubDescriptionEngine(),
        clock=clock,
    )
    scheduler = MarketReadingScheduler(
        assembler=assembler,
        readings_store=assembler.readings_store,
        candles_store=assembler.candles_store,
        tick_interval_seconds=3600,  # no automatic tick during the test
        clock=clock,
    )
    app = create_app(
        signal_store=SignalStore(db_path=str(tmp_path / "signals.db")),
        market_reading_assembler=assembler,
        market_reading_scheduler=scheduler,
    )
    return app, clock, readings_store, scheduler


def test_endpoint_returns_complete_market_reading_with_news_and_structure(wired):
    """Étape 6 #1 — GET /api/market-reading returns a fully-populated MarketReading.

    Exercises the wired flow end-to-end: assembler → SMC stub → REAL NewsPipeline
    (with stub fetch_fn) → description engine stub → persisted payload. Verifies
    every section of the schema is populated, not just status 200.
    """
    app, clock, readings_store, scheduler = wired

    with TestClient(app) as client:
        # Lifespan started the injected scheduler — sanity check before the call.
        assert scheduler.running is True

        resp = client.get("/api/market-reading?instrument=XAUUSD&timeframe=H1")
        assert resp.status_code == 200
        data = resp.json()

        # Header — latest closed H1 candle, instrument echoed.
        first_close = expected_last_candle_close("H1", T0)
        assert data["header"]["instrument"] == "XAUUSD"
        assert data["header"]["timeframe"] == "H1"
        assert data["header"]["candle_close_ts"].startswith(
            first_close.isoformat().replace("+00:00", "")
        )

        # Structure — SMC stub produced an order block.
        assert len(data["structure"]["order_blocks"]) >= 1

        # Regime — derived from the candles.
        assert data["regime"]["trend"] in ("bullish", "bearish", "ranging", "neutral")

        # Events — REAL news pipeline filled news_upcoming (USD/EUR within 4h).
        assert len(data["events"]["news_upcoming"]) >= 1
        titles = {e["event"] for e in data["events"]["news_upcoming"]}
        assert "US Non-Farm Payrolls" in titles

        # Conditions — description engine ran; source = haiku_generated.
        assert data["conditions"]["description"]
        assert data["conditions"]["description_source"] == "haiku_generated"

    # Lifespan exit stops the scheduler cleanly.
    assert scheduler.running is False


def test_active_combinations_updated_after_endpoint_call(wired):
    """Étape 6 #2 — every endpoint call marks (instrument, timeframe) active.

    The hybrid scheduler reads ``active_combinations`` on every tick to decide
    what to regenerate; if the endpoint forgot to mark the combination, the
    auto-stop window would never start counting and the scheduler would do
    nothing on its first wake.
    """
    app, clock, readings_store, scheduler = wired

    # Pre-condition: no active combinations yet.
    pre_active = readings_store.get_active_combinations(since=T0 - timedelta(hours=1))
    assert ("XAUUSD", "H1") not in pre_active

    with TestClient(app) as client:
        resp = client.get("/api/market-reading?instrument=XAUUSD&timeframe=H1")
        assert resp.status_code == 200

        # Combination must now appear in the active set within the 24h window.
        active = readings_store.get_active_combinations(since=T0 - timedelta(hours=1))
        assert ("XAUUSD", "H1") in active

        # A second hit on a different timeframe adds it without removing the first.
        resp2 = client.get("/api/market-reading?instrument=XAUUSD&timeframe=M15")
        assert resp2.status_code == 200
        active2 = readings_store.get_active_combinations(since=T0 - timedelta(hours=1))
        assert ("XAUUSD", "H1") in active2
        assert ("XAUUSD", "M15") in active2


def test_scheduler_regenerates_active_combinations_on_tick(wired):
    """Étape 6 #3 — a manual scheduler tick regenerates active combinations.

    Advances the clock past the next H1 candle close, calls ``tick()`` directly,
    and verifies (a) the assembler was re-invoked (one regeneration counted)
    and (b) the stored reading now points to the new candle close.
    """
    app, clock, readings_store, scheduler = wired

    with TestClient(app) as client:
        # Seed the active set + initial reading via one endpoint call.
        resp = client.get("/api/market-reading?instrument=XAUUSD&timeframe=H1")
        assert resp.status_code == 200
        first_close = expected_last_candle_close("H1", T0)

        # Idempotence — a tick at the same clock = 0 regenerations (no new candle).
        assert scheduler.tick() == 0

        # Advance the clock past the next H1 boundary → tick must regenerate.
        clock.now = T0 + timedelta(minutes=65)
        new_close = expected_last_candle_close("H1", clock.now)
        assert new_close > first_close

        assert scheduler.tick() == 1
        latest = readings_store.get_latest_reading("XAUUSD", "H1")
        assert latest["header"]["candle_close_ts"].startswith(
            new_close.isoformat().replace("+00:00", "")
        )
