"""CalendarService + provider factory (NW-1 / NW-1b) — attachment rule,
provider-agnosticism, default source, coverage, attribution, freshness."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

from src.intelligence.calendar_providers import build_calendar_provider
from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderAttribution,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_providers.official_provider import OfficialCalendarProvider
from src.intelligence.calendar_service import CalendarService, load_market_map
from src.intelligence.calendar_schema import CalendarEvent
from src.storage.calendar_cache_store import CalendarCacheStore

NOW = datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc)
MAP = {"XAUUSD": ["USD"], "EURUSD": ["USD", "EUR"]}


class FakeProvider(CalendarProvider):
    """A minimal provider — proves the service needs only the interface."""

    def __init__(self, events: List[ProviderEvent], name: str = "fake",
                 cov: Optional[tuple] = None,
                 attributions: Optional[List[ProviderAttribution]] = None) -> None:
        self._events = events
        self._name = name
        self._cov = cov
        self._atts = attributions or []

    @property
    def source_name(self) -> str:
        return self._name

    def fetch(self) -> ProviderFetch:
        cov = self._cov or (None, None)
        return ProviderFetch(events=self._events, coverage_start=cov[0], coverage_end=cov[1])

    def attributions(self) -> List[ProviderAttribution]:
        return self._atts


def _pe(currency: str, ref: str = "r1", source: str = "fake",
        when: datetime = NOW + timedelta(hours=2), **kw) -> ProviderEvent:
    base = dict(
        source=source,
        provider_ref=ref,
        series_code=None,
        event=f"{currency} event",
        currency=currency,
        scheduled_at=when,
    )
    base.update(kw)
    return ProviderEvent(**base)


def _service(events, tmp_path, name="fake", cov=None, attributions=None,
             ttl=0) -> CalendarService:
    return CalendarService(
        provider=FakeProvider(events, name=name, cov=cov, attributions=attributions),
        store=CalendarCacheStore(db_path=str(tmp_path / "cal.db")),
        market_map=MAP,
        ttl_seconds=ttl,
        clock=lambda: NOW,
    )


# --------------------------------------------------------------------------- #
# Default source
# --------------------------------------------------------------------------- #
def test_default_provider_is_the_official_aggregator(monkeypatch) -> None:
    monkeypatch.delenv("CALENDAR_SOURCE", raising=False)
    provider = build_calendar_provider()
    assert isinstance(provider, OfficialCalendarProvider)
    assert provider.source_name == "official" != "forexfactory"


def test_forexfactory_only_when_explicitly_selected(monkeypatch) -> None:
    monkeypatch.setenv("CALENDAR_SOURCE", "forexfactory")
    assert build_calendar_provider().source_name == "forexfactory"


def test_unknown_source_falls_back_to_official(monkeypatch) -> None:
    monkeypatch.setenv("CALENDAR_SOURCE", "bogus")
    assert build_calendar_provider().source_name == "official"


# --------------------------------------------------------------------------- #
# Market attachment rule
# --------------------------------------------------------------------------- #
def test_usd_attaches_both_markets(tmp_path) -> None:
    resp = _service([_pe("USD")], tmp_path).get_calendar(now=NOW)
    assert resp.events[0].markets == ["XAUUSD", "EURUSD"]


def test_eur_attaches_only_eurusd(tmp_path) -> None:
    resp = _service([_pe("EUR")], tmp_path).get_calendar(now=NOW)
    assert resp.events[0].markets == ["EURUSD"]


def test_event_without_market_is_dropped(tmp_path) -> None:
    svc = _service([_pe("USD", ref="u"), _pe("GBP", ref="g")], tmp_path)
    assert [e.currency for e in svc.get_calendar(now=NOW).events] == ["USD"]


def test_real_config_map_matches_rule() -> None:
    m = load_market_map()
    assert m["XAUUSD"] == ["USD"]
    assert m["EURUSD"] == ["USD", "EUR"]


# --------------------------------------------------------------------------- #
# No impact ranking, no consensus (NW-1b)
# --------------------------------------------------------------------------- #
def test_served_event_has_no_impact_or_forecast_field(tmp_path) -> None:
    e = _service([_pe("USD")], tmp_path).get_calendar(now=NOW).events[0]
    assert isinstance(e, CalendarEvent)
    assert "impact" not in CalendarEvent.model_fields
    assert "forecast" not in CalendarEvent.model_fields
    assert not hasattr(e, "impact")
    assert not hasattr(e, "forecast")


def test_periodicity_and_time_confirmed_flow_through(tmp_path) -> None:
    svc = _service(
        [_pe("USD", periodicity="quarterly", time_confirmed=False)], tmp_path
    )
    e = svc.get_calendar(now=NOW).events[0]
    assert e.periodicity == "quarterly"
    assert e.time_confirmed is False  # marked → NW-2 excludes from measures


# --------------------------------------------------------------------------- #
# Provider-agnosticism + coexistence
# --------------------------------------------------------------------------- #
def test_service_depends_only_on_interface(tmp_path) -> None:
    svc = _service([_pe("USD", source="some-other")], tmp_path, name="some-other")
    resp = svc.get_calendar(now=NOW)
    assert resp.coverage.source == "some-other"
    assert resp.events[0].source == "some-other"


def test_two_sources_coexist_without_id_conflict(tmp_path) -> None:
    ts = NOW + timedelta(hours=2)
    svc = _service(
        [_pe("USD", ref="cpi", source="bls", when=ts),
         _pe("USD", ref="cpi", source="fake", when=ts)],
        tmp_path,
    )
    ids = {e.event_id for e in svc.get_calendar(now=NOW).events}
    assert ids == {"bls:cpi", "fake:cpi"}


def test_null_fields_render_as_none_not_default(tmp_path) -> None:
    svc = _service([_pe("USD", organism=None, series_code=None, value_unit=None)], tmp_path)
    e = svc.get_calendar(now=NOW).events[0]
    assert e.organism is None and e.series_code is None and e.value_unit is None


# --------------------------------------------------------------------------- #
# Attribution (licence condition)
# --------------------------------------------------------------------------- #
def test_no_served_event_without_attribution(tmp_path) -> None:
    atts = [
        ProviderAttribution("bls", "Bureau of Labor Statistics", "Domaine public", "https://bls"),
        ProviderAttribution("ecb", "BCE", "Réutilisation si citée", "https://ecb"),
    ]
    svc = _service(
        [_pe("USD", ref="cpi", source="bls"), _pe("EUR", ref="rate", source="ecb")],
        tmp_path, attributions=atts,
    )
    resp = svc.get_calendar(now=NOW)
    served_sources = {e.source for e in resp.events}
    attr_sources = {a.source for a in resp.attribution}
    # every served source has an attribution entry
    assert served_sources.issubset(attr_sources)
    assert served_sources == {"bls", "ecb"}


def test_attribution_only_lists_sources_actually_served(tmp_path) -> None:
    atts = [
        ProviderAttribution("bls", "BLS", "Domaine public", "https://bls"),
        ProviderAttribution("census", "Census", "Domaine public", "https://census"),
    ]
    # only a BLS event is served → census must NOT appear in attribution
    svc = _service([_pe("USD", source="bls")], tmp_path, attributions=atts)
    resp = svc.get_calendar(now=NOW)
    assert [a.source for a in resp.attribution] == ["bls"]


# --------------------------------------------------------------------------- #
# Resilience: a source that fails keeps its data + shows last success
# --------------------------------------------------------------------------- #
class _StatefulProvider(CalendarProvider):
    """Returns events on the first fetch, then nothing (source unavailable)."""

    def __init__(self, first: List[ProviderEvent]) -> None:
        self._first = first
        self._calls = 0

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        self._calls += 1
        return ProviderFetch(events=self._first if self._calls == 1 else [])


def test_unavailable_source_keeps_stored_data(tmp_path) -> None:
    store = CalendarCacheStore(db_path=str(tmp_path / "cal.db"))
    provider = _StatefulProvider([_pe("USD", ref="cpi", source="bls")])
    svc = CalendarService(provider=provider, store=store, market_map=MAP,
                          ttl_seconds=0, clock=lambda: NOW)
    # cycle 1 — stores the event
    r1 = svc.get_calendar(now=NOW)
    assert len(r1.events) == 1
    # cycle 2 — provider now returns nothing; stored data must remain
    r2 = svc.get_calendar(now=NOW + timedelta(minutes=5))
    assert len(r2.events) == 1
    assert r2.events[0].event_id == "bls:cpi"


def test_stale_source_flagged_with_last_success(tmp_path) -> None:
    store = CalendarCacheStore(db_path=str(tmp_path / "cal.db"))
    t0 = NOW
    t1 = NOW + timedelta(hours=1)
    from src.storage.calendar_cache_store import CalendarCacheEvent
    store.upsert_events([CalendarCacheEvent(
        event_id="bls:a", source="bls", event="A", currency="USD",
        scheduled_at=NOW + timedelta(hours=2), markets=["XAUUSD"])], fetched_at=t0)
    store.upsert_events([CalendarCacheEvent(
        event_id="ecb:b", source="ecb", event="B", currency="EUR",
        scheduled_at=NOW + timedelta(hours=3), markets=["EURUSD"])], fetched_at=t1)
    # provider returns nothing and TTL is huge → no refresh, freshness from store
    svc = CalendarService(provider=FakeProvider([], name="official"), store=store,
                          market_map=MAP, ttl_seconds=10_000,
                          clock=lambda: t1 + timedelta(minutes=1))
    resp = svc.get_calendar(now=t1 + timedelta(minutes=1))
    assert resp.coverage.stale_sources == ["bls"]        # bls not refreshed in last cycle
    assert resp.coverage.last_success["bls"] == t0       # its last success is shown


# --------------------------------------------------------------------------- #
# Coverage honesty
# --------------------------------------------------------------------------- #
def test_partial_coverage_when_window_exceeds_feed(tmp_path) -> None:
    cov_end = NOW + timedelta(days=2)
    svc = _service([_pe("USD")], tmp_path, cov=(NOW - timedelta(days=1), cov_end))
    resp = svc.get_calendar(now=NOW, lookahead_minutes=7 * 24 * 60)
    assert resp.coverage.partial is True
    assert resp.coverage.feed_end == cov_end


def test_not_partial_when_window_within_feed(tmp_path) -> None:
    svc = _service([_pe("USD")], tmp_path,
                   cov=(NOW - timedelta(days=5), NOW + timedelta(days=10)))
    resp = svc.get_calendar(now=NOW, lookahead_minutes=24 * 60, lookback_minutes=24 * 60)
    assert resp.coverage.partial is False
