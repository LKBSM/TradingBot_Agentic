"""Value states, enrichment, freshness threshold (NW-1c §3).

Deterministic: value fetchers are injected (no network). Covers the four value
states, the three DISTINCT absences, the value enricher (published + revision +
graceful unfetched), and the 24h freshness threshold."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderAttribution,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_providers.values.base_value import (
    MultiValueFetcher,
    ValueFetcher,
    ValuePoint,
    build_value_fetcher,
)
from src.intelligence.calendar_providers.values.ecb_values import _parse_observations
from src.intelligence.calendar_schema import compute_value_state
from src.intelligence.calendar_service import CalendarService
from src.storage.calendar_cache_store import CalendarCacheEvent, CalendarCacheStore

NOW = datetime(2026, 7, 29, 12, 0, tzinfo=timezone.utc)
MAP = {"XAUUSD": ["USD"], "EURUSD": ["USD", "EUR"]}


# --------------------------------------------------------------------------- #
# compute_value_state — the four states, order matters
# --------------------------------------------------------------------------- #
def test_state_published_when_value_present():
    assert compute_value_state("S1", 3.2, NOW - timedelta(days=1), NOW) == "published"


def test_state_pending_when_future():
    assert compute_value_state("S1", None, NOW + timedelta(days=5), NOW) == "pending"
    # future beats no-series: the value cannot exist yet
    assert compute_value_state(None, None, NOW + timedelta(days=5), NOW) == "pending"


def test_state_unfetched_when_past_with_series_no_value():
    assert compute_value_state("S1", None, NOW - timedelta(days=5), NOW) == "unfetched"


def test_state_unavailable_when_past_without_series():
    assert compute_value_state(None, None, NOW - timedelta(days=5), NOW) == "unavailable"


# --------------------------------------------------------------------------- #
# Service — the three absences are distinct end-to-end
# --------------------------------------------------------------------------- #
class _Prov(CalendarProvider):
    def __init__(self, events: List[ProviderEvent]) -> None:
        self._e = events

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        return ProviderFetch(events=self._e)

    def attributions(self) -> List[ProviderAttribution]:
        return [ProviderAttribution("bls", "BLS", "pub", "https://x"),
                ProviderAttribution("federal_reserve", "Federal Reserve Board", "pub", "https://y")]


def _pe(ref, series="S1", when=None, source="bls", actual=None) -> ProviderEvent:
    return ProviderEvent(
        source=source, provider_ref=ref, series_code=series, event=ref, currency="USD",
        scheduled_at=when, organism="BLS", periodicity="monthly", value_unit="indice", actual=actual,
    )


class _FakeFetcher(ValueFetcher):
    def __init__(self, value: Optional[float]) -> None:
        self._v = value

    def fetch(self, series_code: str) -> Optional[ValuePoint]:
        return None if self._v is None else ValuePoint(actual=self._v, previous=self._v - 0.1)


def _svc(events, tmp_path, fetcher=None, clock=NOW):
    return CalendarService(
        provider=_Prov(events),
        store=CalendarCacheStore(db_path=str(tmp_path / "cal.db")),
        market_map=MAP, ttl_seconds=0, clock=lambda: clock,
        value_fetcher=fetcher,
    )


def _states(resp):
    return {e.provider_ref.split(":")[0] if ":" in e.event_id else e.event: e.actual_state
            for e in resp.events}


def test_three_absences_are_distinct(tmp_path):
    events = [
        _pe("future", when=NOW + timedelta(days=5)),                  # pending
        _pe("past_series", when=NOW - timedelta(days=5)),             # unfetched (no fetcher)
        _pe("fomc", series=None, when=NOW - timedelta(days=5), source="federal_reserve"),  # unavailable
    ]
    resp = _svc(events, tmp_path, fetcher=None).get_calendar(
        now=NOW, lookahead_minutes=30 * 1440, lookback_minutes=30 * 1440
    )
    by = {e.event: e for e in resp.events}
    assert by["future"].actual_state == "pending"
    assert by["past_series"].actual_state == "unfetched"
    assert by["past_series"].refreshed_at is not None       # last attempt is recorded
    assert by["fomc"].actual_state == "unavailable"


def test_upcoming_event_gets_previous_value_stays_pending(tmp_path):
    # REC point 2: an UPCOMING release does not have its own value yet (pending),
    # but the previous period's value IS published and must be shown. The latest
    # published observation (fetcher.actual) fills `previous`; `actual` stays None.
    events = [_pe("future_series", when=NOW + timedelta(days=5))]
    resp = _svc(
        events, tmp_path, fetcher=MultiValueFetcher({"bls": _FakeFetcher(2.0)})
    ).get_calendar(now=NOW, lookahead_minutes=30 * 1440, lookback_minutes=1440)
    e = resp.events[0]
    assert e.actual is None            # its own value does not exist yet
    assert e.actual_state == "pending"
    assert e.previous == 2.0           # last published value, shown for context


def test_enricher_publishes_a_past_value(tmp_path):
    events = [_pe("past_series", when=NOW - timedelta(days=5))]
    resp = _svc(events, tmp_path, fetcher=MultiValueFetcher({"bls": _FakeFetcher(3.4)})).get_calendar(
        now=NOW, lookback_minutes=30 * 1440, lookahead_minutes=1440
    )
    e = resp.events[0]
    assert e.actual_state == "published"
    assert e.actual == 3.4
    assert e.previous == 3.3


def test_unfetched_never_confused_with_published(tmp_path):
    # A fetcher that returns None (unreachable) must leave the event unfetched,
    # NEVER a fabricated value — a product state, not a market absence.
    events = [_pe("past_series", when=NOW - timedelta(days=5))]
    resp = _svc(events, tmp_path, fetcher=MultiValueFetcher({"bls": _FakeFetcher(None)})).get_calendar(
        now=NOW, lookback_minutes=30 * 1440, lookahead_minutes=1440
    )
    e = resp.events[0]
    assert e.actual is None
    assert e.actual_state == "unfetched"


def test_enricher_caches_value_fetch_within_ttl(tmp_path):
    """NW-7b — the latest-value enrichment must NOT re-hit the source on every
    120s refresh: that alone exhausted the BLS free 500/day key. A value fetched
    for (source, series_code) is reused within the cache window."""
    calls = {"n": 0}

    class _Counting(ValueFetcher):
        def fetch(self, series_code: str) -> Optional[ValuePoint]:
            calls["n"] += 1
            return ValuePoint(actual=2.0, previous=1.9)

    svc = _svc([_pe("fut", when=NOW + timedelta(days=5))], tmp_path,
               fetcher=MultiValueFetcher({"bls": _Counting()}))
    svc.get_calendar(now=NOW, lookahead_minutes=30 * 1440, lookback_minutes=1440)
    assert calls["n"] == 1                              # one live fetch (cold)
    assert ("bls", "S1") in svc._value_point_cache
    # A second enrichment for the SAME (source, series) within the TTL reuses the
    # cached value — no new call, even for a different upcoming release.
    ce = CalendarCacheEvent(
        event_id="bls:fut2", source="bls", event="fut2", currency="USD",
        scheduled_at=NOW + timedelta(days=6), markets=["XAUUSD"],
        series_code="S1", value_unit="indice",
    )
    svc._enrich_values([ce], NOW)
    assert calls["n"] == 1                              # served from cache


def test_enricher_flags_revision_across_cycles(tmp_path):
    store = CalendarCacheStore(db_path=str(tmp_path / "cal.db"))
    ev = _pe("past_series", when=NOW - timedelta(days=5))
    CalendarService(provider=_Prov([ev]), store=store, market_map=MAP, ttl_seconds=0,
                    clock=lambda: NOW, value_fetcher=MultiValueFetcher({"bls": _FakeFetcher(3.4)})
                    ).get_calendar(now=NOW, lookback_minutes=30 * 1440, lookahead_minutes=1440)
    resp2 = CalendarService(
        provider=_Prov([ev]), store=store, market_map=MAP, ttl_seconds=0,
        clock=lambda: NOW + timedelta(minutes=5),
        value_fetcher=MultiValueFetcher({"bls": _FakeFetcher(3.6)}),
    ).get_calendar(now=NOW + timedelta(minutes=5), lookback_minutes=30 * 1440, lookahead_minutes=1440)
    e = resp2.events[0]
    assert e.actual == 3.6
    assert e.actual_initial == 3.4      # first print preserved
    assert e.revised is True
    assert e.revised_at is not None


def test_default_value_fetcher_is_off(monkeypatch):
    monkeypatch.delenv("CALENDAR_VALUES_LIVE", raising=False)
    assert build_value_fetcher() is None


def test_values_live_flag_builds_fetcher(monkeypatch):
    monkeypatch.setenv("CALENDAR_VALUES_LIVE", "1")
    f = build_value_fetcher()
    assert f is not None
    assert f.fetch_for("ecb", None) is None        # no series → None
    assert f.fetch_for("bea", "NIPA-T10101") is None  # BEA not registered (key-gated) → None


# --------------------------------------------------------------------------- #
# Freshness threshold (24h)
# --------------------------------------------------------------------------- #
def test_source_stale_after_threshold(tmp_path):
    store = CalendarCacheStore(db_path=str(tmp_path / "cal.db"))
    old = NOW - timedelta(hours=30)  # older than STALE_AFTER_HOURS
    store.upsert_events([CalendarCacheEvent(
        event_id="bls:a", source="bls", event="A", currency="USD",
        scheduled_at=NOW + timedelta(days=2), markets=["XAUUSD"])], fetched_at=old)
    svc = CalendarService(provider=_Prov([]), store=store, market_map=MAP,
                          ttl_seconds=10_000, clock=lambda: NOW)
    resp = svc.get_calendar(now=NOW)
    # single source, but its last success is > 24h old → flagged stale (not silent)
    assert resp.coverage.stale_sources == ["bls"]
    assert resp.coverage.last_success["bls"] == old


# --------------------------------------------------------------------------- #
# ECB SDMX-JSON observation parser
# --------------------------------------------------------------------------- #
def test_ecb_parse_observations():
    payload = (
        '{"dataSets":[{"series":{"0:0:0:0:0:0:0":{"observations":'
        '{"0":[2.15],"1":[2.4]}}}}],"structure":{}}'
    )
    assert _parse_observations(payload) == [2.15, 2.4]
    assert _parse_observations("not json") == []
    assert _parse_observations('{"dataSets":[]}') == []
