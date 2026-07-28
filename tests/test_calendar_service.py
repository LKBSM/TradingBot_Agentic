"""CalendarService + provider factory — attachment rule, provider-agnosticism,
default source, coverage (NW-1)."""

from __future__ import annotations

import importlib
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytest

from src.intelligence.calendar_providers import build_calendar_provider
from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_providers.official_provider import OfficialCalendarProvider
from src.intelligence.calendar_service import CalendarService, load_market_map
from src.storage.calendar_cache_store import CalendarCacheStore

NOW = datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc)
MAP = {"XAUUSD": ["USD"], "EURUSD": ["USD", "EUR"]}


class FakeProvider(CalendarProvider):
    """A minimal provider — proves the service needs only the interface, never a
    provider-specific field."""

    def __init__(self, events: List[ProviderEvent], name: str = "fake",
                 cov: Optional[tuple] = None) -> None:
        self._events = events
        self._name = name
        self._cov = cov

    @property
    def source_name(self) -> str:
        return self._name

    def fetch(self) -> ProviderFetch:
        cov = self._cov or (None, None)
        return ProviderFetch(events=self._events, coverage_start=cov[0], coverage_end=cov[1])


def _pe(currency: str, impact: str = "high", ref: str = "r1", source: str = "fake",
        when: datetime = NOW + timedelta(hours=2), **kw) -> ProviderEvent:
    base = dict(
        source=source,
        provider_ref=ref,
        series_code=None,
        event=f"{currency} event",
        currency=currency,
        impact=impact,
        scheduled_at=when,
    )
    base.update(kw)
    return ProviderEvent(**base)


def _service(events, tmp_path, name="fake", cov=None) -> CalendarService:
    return CalendarService(
        provider=FakeProvider(events, name=name, cov=cov),
        store=CalendarCacheStore(db_path=str(tmp_path / "cal.db")),
        market_map=MAP,
        clock=lambda: NOW,
    )


# --------------------------------------------------------------------------- #
# Default source
# --------------------------------------------------------------------------- #
def test_default_provider_is_not_forexfactory(monkeypatch) -> None:
    monkeypatch.delenv("CALENDAR_SOURCE", raising=False)
    provider = build_calendar_provider()
    assert isinstance(provider, OfficialCalendarProvider)
    assert provider.source_name == "official"
    assert provider.source_name != "forexfactory"


def test_forexfactory_only_when_explicitly_selected(monkeypatch) -> None:
    monkeypatch.setenv("CALENDAR_SOURCE", "forexfactory")
    provider = build_calendar_provider()
    assert provider.source_name == "forexfactory"


def test_unknown_source_falls_back_to_official(monkeypatch) -> None:
    monkeypatch.setenv("CALENDAR_SOURCE", "bogus")
    assert build_calendar_provider().source_name == "official"


def test_official_stub_yields_nothing() -> None:
    assert OfficialCalendarProvider().fetch().events == []


# --------------------------------------------------------------------------- #
# Market attachment rule
# --------------------------------------------------------------------------- #
def test_usd_attaches_both_markets(tmp_path) -> None:
    svc = _service([_pe("USD")], tmp_path)
    resp = svc.get_calendar(now=NOW)
    assert len(resp.events) == 1
    assert resp.events[0].markets == ["XAUUSD", "EURUSD"]


def test_eur_attaches_only_eurusd(tmp_path) -> None:
    svc = _service([_pe("EUR")], tmp_path)
    resp = svc.get_calendar(now=NOW)
    assert resp.events[0].markets == ["EURUSD"]


def test_event_without_market_is_dropped(tmp_path) -> None:
    # GBP is not a driver currency of any followed market → never displayed.
    svc = _service([_pe("USD", ref="u"), _pe("GBP", ref="g")], tmp_path)
    resp = svc.get_calendar(now=NOW)
    assert [e.currency for e in resp.events] == ["USD"]


def test_real_config_map_matches_rule() -> None:
    m = load_market_map()
    assert m["XAUUSD"] == ["USD"]
    assert m["EURUSD"] == ["USD", "EUR"]


# --------------------------------------------------------------------------- #
# All impacts + provider-agnosticism + no id conflict
# --------------------------------------------------------------------------- #
def test_all_impacts_kept(tmp_path) -> None:
    svc = _service(
        [_pe("USD", "high", "h"), _pe("USD", "medium", "m"), _pe("USD", "low", "l")],
        tmp_path,
    )
    resp = svc.get_calendar(now=NOW)
    assert {e.impact for e in resp.events} == {"high", "medium", "low"}


def test_service_depends_only_on_interface(tmp_path) -> None:
    # The FakeProvider carries none of ForexFactory's shape — the service still
    # produces a full response, proving no provider-specific coupling.
    svc = _service(
        [_pe("USD", source="some-other-source")], tmp_path, name="some-other-source"
    )
    resp = svc.get_calendar(now=NOW)
    assert resp.coverage.source == "some-other-source"
    assert resp.events[0].source == "some-other-source"


def test_two_sources_coexist_without_id_conflict(tmp_path) -> None:
    ts = NOW + timedelta(hours=2)
    svc = _service(
        [
            _pe("USD", ref="cpi", source="official", when=ts),
            _pe("USD", ref="cpi", source="fake", when=ts),
        ],
        tmp_path,
    )
    resp = svc.get_calendar(now=NOW)
    ids = {e.event_id for e in resp.events}
    assert ids == {"official:cpi", "fake:cpi"}


def test_null_fields_render_as_none_not_default(tmp_path) -> None:
    svc = _service([_pe("USD", organism=None, series_code=None, value_unit=None)], tmp_path)
    e = svc.get_calendar(now=NOW).events[0]
    assert e.organism is None and e.series_code is None and e.value_unit is None


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
    svc = _service(
        [_pe("USD")], tmp_path,
        cov=(NOW - timedelta(days=5), NOW + timedelta(days=10)),
    )
    resp = svc.get_calendar(now=NOW, lookahead_minutes=24 * 60, lookback_minutes=24 * 60)
    assert resp.coverage.partial is False
