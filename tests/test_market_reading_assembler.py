"""Tests for MarketReadingAssembler (Chantier 2 Étape 4)."""

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Sequence

import pytest

from src.intelligence.market_reading_assembler import (
    MarketReadingAssembler,
    READING_LOGIC_VERSION,
    build_cache_mtf_provider,
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
    # DG-1 point 1: candle_close_ts names the LAST CANDLE ANALYSED (07:15 open +
    # 15 min), derived from the data — NOT the wall-clock expected close (14:15).
    # Here the mock feed lags the clock, so the two differ and the header stays
    # honest instead of claiming a candle that was never analysed.
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 7, 30, 0, tzinfo=timezone.utc
    )
    # Decoupling proof: the CACHE key passed to save_reading is still the
    # wall-clock/market-aware expected close (14:15) — only the displayed header
    # follows the data.
    assert readings_store.save_calls[0][2] == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )
    assert reading.header.close_price == candles[-1].close
    assert reading.structure.current_bos is not None  # populated from stub_smc_pipeline
    assert reading.conditions.description_source == "engine_template"

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
    # M15 close at 14:23:00Z → 14:15:00Z. The feed must reach that close (57
    # candles from 00:00 → last open 14:00 → close 14:15) so the honest
    # candle_close_ts (DG-1) equals the expected close and the second access is a
    # genuine cache hit.
    seed_assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(57)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    seed_assembler.get_or_generate("XAUUSD", "M15")

    # Now build a fresh assembler whose store is pre-populated with that payload
    # AS THE STORE ACTUALLY HOLDS IT — i.e. stamped with the current logic version
    # (via _persist_reading). Copying the model dump would drop that stamp and
    # make the cache look stale, so we read it back from the seed store instead.
    payload = seed_assembler.readings_store.get_latest_reading("XAUUSD", "M15")
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
# PERF-2 — a CURRENT-version reading that is a candle behind is SERVED verbatim,
# never rebuilt on the request path (the fix for "5-7 s on every call": under a
# lagging/rate-limited feed the stored close never reaches the wall-clock
# expected_close, so the old code rebuilt — and burned the bounded-provider
# budget — on EVERY request). The scheduler advances it off the request path.
# ---------------------------------------------------------------------------


def _seed_current_version_reading(fixed_clock) -> dict:
    """Build a genuine, schema-valid, current-_logic_version reading (last M15
    close = 14:15 at the fixed clock) and return the payload AS STORED (stamped)."""
    seed = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(57)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    seed.get_or_generate("XAUUSD", "M15")
    payload = seed.readings_store.get_latest_reading("XAUUSD", "M15")
    assert payload["_logic_version"] == READING_LOGIC_VERSION
    return payload


def _later_clock():
    # One bar past the seed: expected M15 close is now 14:30, so the seeded
    # reading (14:15) is a candle BEHIND — the prod "feed lagging" case.
    return datetime(2026, 5, 28, 14, 38, 0, tzinfo=timezone.utc)


def test_current_version_stale_reading_is_served_without_rebuild(fixed_clock):
    payload = _seed_current_version_reading(fixed_clock)
    provider = _MockDataProvider(_build_candles(30))
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=payload)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=_later_clock,
    )

    reading = assembler.get_or_generate("XAUUSD", "M15")

    # Served straight from the store: NO provider call, NO candle upsert, NO save.
    assert provider.call_count == 0
    assert candles_store.upsert_calls == []
    assert readings_store.save_calls == []
    # Still kept warm so the background scheduler advances it.
    assert readings_store.mark_active_calls == [("XAUUSD", "M15")]
    # Detection output is the STORED one, untouched (served verbatim) — the header
    # is the seed's analysed close (14:15), not the wall-clock expected 14:30.
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )


def test_interactive_serve_stored_kill_switch_restores_rebuild(fixed_clock, monkeypatch):
    # The emergency env fall-back forces the old synchronous behaviour: the same
    # stale current-version reading is rebuilt (provider fetched) instead of served.
    monkeypatch.setenv("SENTINEL_INTERACTIVE_SERVE_STORED", "0")
    payload = _seed_current_version_reading(fixed_clock)
    provider = _MockDataProvider(_build_candles(30))
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=payload)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=_later_clock,
    )

    assembler.get_or_generate("XAUUSD", "M15")

    # Kill switch off → rebuild path: provider fetched, reading re-saved.
    assert provider.call_count == 1
    assert len(readings_store.save_calls) == 1


def test_serve_stored_does_not_wait_on_a_slow_provider(fixed_clock):
    # PERF-2 budget guard: serving must be a SQLite read, NOT a provider round-trip.
    # A provider that takes 3 s would blow every time budget if it were on the path;
    # the interactive serve-stored path must return in well under a second and never
    # call it. This fails loudly if a future change re-introduces a synchronous fetch.
    import time as _time

    payload = _seed_current_version_reading(fixed_clock)

    class _SlowProvider:
        def __init__(self):
            self.call_count = 0

        def fetch_candles(self, instrument, timeframe, count):
            self.call_count += 1
            _time.sleep(3.0)
            return _build_candles(30)[-count:]

    provider = _SlowProvider()
    readings_store = _MockReadingsStore(prepopulated=payload)
    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=_later_clock,
    )

    start = _time.perf_counter()
    assembler.get_or_generate("XAUUSD", "M15")
    elapsed = _time.perf_counter() - start

    assert provider.call_count == 0
    assert elapsed < 0.5, (
        f"served in {elapsed:.2f}s — the external provider must never be on the "
        "interactive request path (PERF-2)"
    )


def test_scheduler_path_still_rebuilds_stale_current_version_reading(fixed_clock):
    # The background scheduler (bound_provider=False) is the ONLY provider-touching
    # path now — it must still rebuild a stale reading so candles.db advances.
    payload = _seed_current_version_reading(fixed_clock)
    provider = _MockDataProvider(_build_candles(30))
    candles_store = _MockCandlesStore()
    readings_store = _MockReadingsStore(prepopulated=payload)

    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=candles_store,
        smc_pipeline=_stub_smc_pipeline,
        clock=_later_clock,
    )

    assembler.get_or_generate("XAUUSD", "M15", bound_provider=False)

    assert provider.call_count == 1
    assert len(readings_store.save_calls) == 1


# ---------------------------------------------------------------------------
# Stale LOGIC version — regeneration (LQ-D1: a pure logic fix must reach the
# screen, not stay frozen behind a still-matching cache)
# ---------------------------------------------------------------------------


def _smc_pipeline_with_liquidity(candles):
    feats, sig = _stub_smc_pipeline(candles)
    feats["_liquidity"] = [
        {
            "side": "ssl", "kind": "equal_lows", "level": 1980.0, "touches": 2,
            "is_external": True, "status": "intact",
            "created_at": None, "swept_at": None, "broken_at": None,
        }
    ]
    return feats, sig


def test_stale_logic_version_forces_regeneration(fixed_clock):
    # Seed a reading (persisted WITH the current logic stamp), then strip the
    # stamp to mimic a payload produced by pre-fix logic whose candle_close_ts
    # still matches. get_or_generate must rebuild it, not serve the stale state.
    seed = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(57)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    seed.get_or_generate("XAUUSD", "M15")
    payload = dict(seed.readings_store.get_latest_reading("XAUUSD", "M15"))
    assert payload["_logic_version"] == READING_LOGIC_VERSION  # seed is stamped
    payload.pop("_logic_version")  # ← pre-fix stored reading (no stamp)

    provider = _MockDataProvider(_build_candles(57))
    readings_store = _MockReadingsStore(prepopulated=payload)
    assembler = MarketReadingAssembler(
        data_provider=provider,
        readings_store=readings_store,
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    assembler.get_or_generate("XAUUSD", "M15")

    # Rebuilt (fetch + save), NOT served from the stale-logic cache …
    assert provider.call_count == 1
    assert len(readings_store.save_calls) == 1
    # … and the rebuilt payload carries the current stamp.
    assert readings_store.save_calls[0][3]["_logic_version"] == READING_LOGIC_VERSION


class _FailingDataProvider:
    """Provider that cannot fetch (feed down / no CSV / MT5 offline / quota)."""

    def fetch_candles(self, instrument, timeframe, count):
        raise RuntimeError("provider unavailable")


def test_build_failure_serves_stored_reading_not_blank(fixed_clock):
    # A logic-version bump invalidates the cache and forces a rebuild. If the
    # feed is then down, the app must serve the LAST STORED reading (degraded,
    # flagged behind by the freshness badge) — never a blank screen.
    seed = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(57)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    seed.get_or_generate("XAUUSD", "M15")
    payload = dict(seed.readings_store.get_latest_reading("XAUUSD", "M15"))
    payload.pop("_logic_version")  # simulate an older-version stored reading

    readings_store = _MockReadingsStore(prepopulated=payload)
    assembler = MarketReadingAssembler(
        data_provider=_FailingDataProvider(),  # rebuild WILL fail
        readings_store=readings_store,
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")
    assert reading is not None  # served the stored reading, not a blank/raise
    assert len(readings_store.save_calls) == 0  # nothing new persisted on failure


def test_build_failure_with_nothing_stored_raises(fixed_clock):
    # No stored reading to fall back to → the failure is surfaced honestly.
    assembler = MarketReadingAssembler(
        data_provider=_FailingDataProvider(),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    with pytest.raises(Exception):
        assembler.get_or_generate("XAUUSD", "M15")


def test_liquidity_kill_switch_empties_pools(fixed_clock, monkeypatch):
    # One reversible env value masks liquidity at the serve layer, on every
    # response, without touching the stored data (LQ-D1 kill switch).
    def _make():
        return MarketReadingAssembler(
            data_provider=_MockDataProvider(_build_candles(57)),
            readings_store=_MockReadingsStore(),
            candles_store=_MockCandlesStore(),
            smc_pipeline=_smc_pipeline_with_liquidity,
            clock=fixed_clock,
        )

    monkeypatch.delenv("SENTINEL_LIQUIDITY_DISABLED", raising=False)
    baseline = _make().get_or_generate("XAUUSD", "M15")
    assert baseline.structure.liquidity_pools != []  # the pocket surfaces

    monkeypatch.setenv("SENTINEL_LIQUIDITY_DISABLED", "1")
    masked = _make().get_or_generate("XAUUSD", "M15")
    assert masked.structure.liquidity_pools == []  # same pocket, masked


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
            "tags": ["stale"], "description": "Stale.", "description_source": "engine_template",
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

    # Regenerated — candle_close_ts follows the ANALYSED data (07:30), not the
    # wall-clock expected close (DG-1 point 1); the cache key stays 14:15.
    assert reading.header.candle_close_ts == datetime(
        2026, 5, 28, 7, 30, 0, tzinfo=timezone.utc
    )
    assert readings_store.save_calls[0][2] == datetime(
        2026, 5, 28, 14, 15, 0, tzinfo=timezone.utc
    )
    assert provider.call_count == 1
    assert len(candles_store.upsert_calls) == 1
    assert len(readings_store.save_calls) == 1
    assert readings_store.mark_active_calls == [("XAUUSD", "M15")]


# ---------------------------------------------------------------------------
# DG-1 points 1 + 3 — a stalled feed makes a detection lag VISIBLE, not silent
# ---------------------------------------------------------------------------


def test_stale_feed_surfaces_data_lagged_status(fixed_clock):
    """When the feed lags the clock, the honest candle_close_ts (DG-1 point 1)
    lets market_status report ``data_lagged`` — the lag is no longer silent
    (point 3). Before the fix, candle_close_ts was stamped from the clock, so
    the age was always 0 and the lag could never surface."""
    candles = _build_candles(30)  # feed ends 07:15 while the clock is 14:23
    readings_store = _MockReadingsStore(prepopulated=None)
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(candles),
        readings_store=readings_store,
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )

    assembler.get_or_generate("XAUUSD", "M15")
    status = assembler.market_status("XAUUSD", "M15")

    # 07:30 stored vs 14:15 expected = 6h45 = 27 M15 bars behind (≥ 5 threshold).
    assert status.state == "data_lagged"
    assert status.bars_behind is not None and status.bars_behind >= 5
    assert status.last_close_ts == datetime(2026, 5, 28, 7, 30, 0, tzinfo=timezone.utc)


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
# Narrated reading is composed 100 % by the deterministic engine template
# (mission « narrated-reading template-engine » — no LLM, no injection seam).
# ---------------------------------------------------------------------------


def test_description_is_engine_template_no_llm(fixed_clock):
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(30)),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")
    # Deterministic template is the sole producer; source is a single value.
    assert reading.conditions.description_source == "engine_template"
    assert len(reading.conditions.description) > 0
    # Present-tense socle sentence is always there (never empty, never speculative).
    assert reading.conditions.description.startswith("Tendance")


# ---------------------------------------------------------------------------
# Closed-candles-only contract (audit DETECTION_QUALITY_REVIEW_2026_06_12 §T3)
# ---------------------------------------------------------------------------

from src.intelligence.market_reading_assembler import drop_unclosed_candles


class TestDropUnclosedCandles:
    def test_forming_bar_is_dropped(self):
        """A bar whose close boundary has not elapsed never reaches the pipeline."""
        expected_close = datetime(2026, 5, 28, 14, 15, tzinfo=timezone.utc)
        closed = _MockCandle(datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc), 2380.0)
        forming = _MockCandle(datetime(2026, 5, 28, 14, 15, tzinfo=timezone.utc), 2381.0)
        kept = drop_unclosed_candles([closed, forming], "M15", expected_close)
        assert kept == [closed]

    def test_future_labelled_bar_is_dropped(self):
        """Defence-in-depth vs §T2: exchange-local labels ~+10h ahead of UTC."""
        expected_close = datetime(2026, 6, 12, 14, 15, tzinfo=timezone.utc)
        ok = _MockCandle(datetime(2026, 6, 12, 14, 0, tzinfo=timezone.utc), 4190.0)
        future = _MockCandle(datetime(2026, 6, 13, 0, 0, tzinfo=timezone.utc), 4192.0)
        kept = drop_unclosed_candles([ok, future], "M15", expected_close)
        assert kept == [ok]

    def test_naive_ts_treated_as_utc(self):
        expected_close = datetime(2026, 5, 28, 14, 15, tzinfo=timezone.utc)
        closed = _MockCandle(datetime(2026, 5, 28, 14, 0), 2380.0)  # naive
        forming = _MockCandle(datetime(2026, 5, 28, 14, 15), 2381.0)  # naive
        kept = drop_unclosed_candles([closed, forming], "M15", expected_close)
        assert kept == [closed]

    def test_h1_boundary(self):
        expected_close = datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc)
        closed = _MockCandle(datetime(2026, 5, 28, 13, 0, tzinfo=timezone.utc), 2380.0)
        forming = _MockCandle(datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc), 2381.0)
        kept = drop_unclosed_candles([closed, forming], "H1", expected_close)
        assert kept == [closed]


def test_build_fresh_excludes_forming_bar_from_analysis_and_cache(fixed_clock):
    """The SMC pipeline and the candles cache only ever see closed bars; the
    header close_price is the close of the last CLOSED candle, matching the
    candle_close_ts contract."""
    # fixed_clock = 14:23Z → expected M15 close = 14:15Z. Last closed candle
    # opens at 14:00 (closes exactly 14:15); the forming one opens at 14:15.
    closed_last = _MockCandle(datetime(2026, 5, 28, 14, 0, tzinfo=timezone.utc), 2384.0)
    forming = _MockCandle(datetime(2026, 5, 28, 14, 15, tzinfo=timezone.utc), 2399.0)
    candles = _build_candles(10) + [closed_last, forming]

    seen_by_pipeline: list[list] = []

    def _capturing_pipeline(cs):
        seen_by_pipeline.append(list(cs))
        return _stub_smc_pipeline(cs)

    candles_store = _MockCandlesStore()
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(candles),
        readings_store=_MockReadingsStore(),
        candles_store=candles_store,
        smc_pipeline=_capturing_pipeline,
        clock=fixed_clock,
    )
    reading = assembler.get_or_generate("XAUUSD", "M15")

    assert forming not in seen_by_pipeline[0]
    assert seen_by_pipeline[0][-1] is closed_last
    assert reading.header.close_price == 2384.0  # not the forming bar's 2399.0
    # Cache contract (/api/candles: "stops at the last fully-closed candle")
    assert candles_store.upsert_calls == [("XAUUSD", "M15", len(candles) - 1)]


def test_provider_snapshot_written_before_filtering(fixed_clock, tmp_path, monkeypatch):
    """Observability (§T3): the RAW response — forming bar included — is
    persisted as JSONL so any reading can be replayed bit-for-bit later."""
    monkeypatch.setenv("PROVIDER_SNAPSHOT_ENABLED", "1")
    monkeypatch.setenv("PROVIDER_SNAPSHOT_DIR", str(tmp_path))

    forming = _MockCandle(datetime(2026, 5, 28, 14, 15, tzinfo=timezone.utc), 2399.0)
    candles = _build_candles(5) + [forming]
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(candles),
        readings_store=_MockReadingsStore(),
        candles_store=_MockCandlesStore(),
        smc_pipeline=_stub_smc_pipeline,
        clock=fixed_clock,
    )
    assembler.get_or_generate("XAUUSD", "M15")

    files = list(tmp_path.glob("XAUUSD_M15_*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8").splitlines()[0])
    assert record["instrument"] == "XAUUSD"
    assert len(record["candles"]) == len(candles)  # raw = forming bar included
    assert record["candles"][-1]["close"] == 2399.0


# ---------------------------------------------------------------------------
# build_cache_mtf_provider — cache-only multi-timeframe bias (fix mtf_confluence
# vide en live : aucun provider n'etait cable)
# ---------------------------------------------------------------------------


class _MtfStore:
    """Fake candle store exposing only get_last_n_candles (pure cache read)."""

    def __init__(self, by_tf: dict[str, list[_MockCandle]]):
        self._by_tf = by_tf
        self.calls: list[tuple[str, str, int]] = []

    def get_last_n_candles(self, instrument: str, timeframe: str, n: int):
        self.calls.append((instrument, timeframe, n))
        return self._by_tf.get(timeframe, [])[-n:]


def _rising(n: int, base: float = 100.0) -> list[_MockCandle]:
    start = datetime(2026, 5, 28, tzinfo=timezone.utc)
    return [_MockCandle(start + timedelta(hours=i), base + i) for i in range(n)]


def _zigzag(n: int, drift: float, base: float = 100.0) -> list[_MockCandle]:
    """Rising/falling zigzag with pullbacks so the ENGINE forms swings and fires a
    BOS/CHOCH. TR-1's bias is STRUCTURAL: a monotone ramp has no swings and reads
    ``indeterminate`` by design; ``drift`` > 0 → bullish, < 0 → bearish."""
    start = datetime(2026, 5, 28, tzinfo=timezone.utc)
    amp = 25.0
    out = []
    for i in range(n):
        phase = i % 8
        local = amp if phase in (3, 4) else (-amp if phase in (7, 0) else 0.0)
        close = base + (i // 8) * drift * 8 + local
        out.append(_MockCandle(start + timedelta(hours=i), close))
    return out


def test_mtf_provider_returns_upper_timeframes_only():
    store = _MtfStore({"H1": _rising(30), "H4": _rising(30)})
    provider = build_cache_mtf_provider(store, lookback=20)
    out = provider("XAUUSD", "M15")  # upper TFs of M15 = H1, H4, D1, W1
    assert set(out.keys()) == {"h1", "h4"}  # only the cached ones surface
    # Values are plain OHLC dicts (consumable by candles_to_regime).
    assert set(out["h1"][0].keys()) == {"open", "high", "low", "close"}
    # Pure cache read — only get_last_n_candles was called, with our lookback.
    assert all(call[2] == 20 for call in store.calls)


def test_mtf_provider_skips_unknown_timeframe_and_empty_cache():
    store = _MtfStore({"H4": []})  # H4 cached but empty
    provider = build_cache_mtf_provider(store)
    assert provider("XAUUSD", "ZZ") == {}        # not on the ladder
    assert provider("XAUUSD", "H1") == {}        # H4 empty, D1/W1 absent


def test_mtf_provider_feeds_candles_to_regime_bias():
    """End-to-end: a rising STRUCTURAL upper-TF series yields a bullish bias via
    the engine's detection (TR-1: bias = last BOS/CHOCH sign, no new detection)."""
    from src.intelligence.market_reading_mappers import candles_to_regime

    store = _MtfStore({"H1": _zigzag(64, drift=6.0), "H4": _zigzag(64, drift=6.0)})
    provider = build_cache_mtf_provider(store)
    mtf = provider("XAUUSD", "M15")
    regime = candles_to_regime(
        [{"close": 100 + i, "high": 101 + i, "low": 99 + i} for i in range(40)],
        mtf_candles_above=mtf,
    )
    assert regime.mtf_confluence.get("h1") == "bullish"
    assert regime.mtf_confluence.get("h4") == "bullish"


# ---------------------------------------------------------------------------
# RG-1c — reference_levels wiring: fresh, from cached intraday candles, attached
# to the reading payload (never persisted).
# ---------------------------------------------------------------------------


class _RefLevelsStore(_MockCandlesStore):
    """A candles store that ALSO serves H1 candles for the reference levels."""

    def __init__(self, h1_candles: list[_MockCandle]):
        super().__init__()
        self._h1 = list(h1_candles)

    def get_last_n_candles(self, instrument: str, timeframe: str, n: int):
        return list(self._h1) if timeframe == "H1" else []


def _ref_h1_series() -> list[_MockCandle]:
    # Two full XAU trading days of hourly bars (17:00 NY rollover = 21:00 UTC in
    # July): an anchor, a complete previous day (Jul 24) and the current day.
    def c(h_utc, close):
        return _MockCandle(datetime(2026, 7, 24, h_utc, tzinfo=timezone.utc), close)

    return [
        _MockCandle(datetime(2026, 7, 23, 12, tzinfo=timezone.utc), 4100.0),  # anchor
        c(6, 4200.0),
        c(12, 4160.0),
        c(18, 4170.0),
        _MockCandle(datetime(2026, 7, 25, 6, tzinfo=timezone.utc), 4174.0),  # current day
    ]


def test_reference_levels_method_aggregates_from_cached_h1():
    store = _RefLevelsStore(_ref_h1_series())
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(5)),
        readings_store=_MockReadingsStore(),
        candles_store=store,
        smc_pipeline=_stub_smc_pipeline,
        clock=lambda: datetime(2026, 7, 25, 20, tzinfo=timezone.utc),
    )
    levels = assembler.reference_levels("XAUUSD")
    assert levels is not None
    # Aggregated over ALL of Jul 24 (_MockCandle high = close + 1, low = close − 1).
    assert levels["prev_day_high"] == 4201.0  # from the 4200 close bar
    assert levels["prev_day_low"] == 4159.0  # from the 4160 close bar
    assert levels["day_complete"] is True
    assert levels["day_open"] == 4173.5  # open of the current day's first bar


def test_reference_levels_none_when_no_cached_source():
    assembler = MarketReadingAssembler(
        data_provider=_MockDataProvider(_build_candles(5)),
        readings_store=_MockReadingsStore(),
        candles_store=_RefLevelsStore([]),  # no H1 → nothing to aggregate
        smc_pipeline=_stub_smc_pipeline,
        clock=lambda: datetime(2026, 7, 25, 20, tzinfo=timezone.utc),
    )
    assert assembler.reference_levels("XAUUSD") is None
