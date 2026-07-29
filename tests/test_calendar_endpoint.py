"""GET /api/calendar endpoint (NW-1 / NW-1b). Descriptive-only; injected service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.calendar import router as calendar_router
from src.api.signal_store import SignalStore
from src.intelligence.calendar_providers.base import (
    CalendarProvider,
    ProviderAttribution,
    ProviderEvent,
    ProviderFetch,
)
from src.intelligence.calendar_service import CalendarService
from src.storage.calendar_cache_store import CalendarCacheStore

NOW = datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc)

# No prediction/direction/ranking/consensus fields may ever appear.
FORBIDDEN_KEYS = frozenset(
    {"direction", "bias", "target", "target_1", "score", "confidence",
     "recommendation", "impact", "forecast"}
)


class _FakeProvider(CalendarProvider):
    def __init__(self, events, atts=None):
        self._events = events
        self._atts = atts or []

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        return ProviderFetch(events=self._events)

    def attributions(self):
        return self._atts


def _make_app(tmp_path, events, atts=None):
    app = FastAPI()
    service = CalendarService(
        provider=_FakeProvider(events, atts),
        store=CalendarCacheStore(db_path=str(tmp_path / "cal.db")),
        market_map={"XAUUSD": ["USD"], "EURUSD": ["USD", "EUR"]},
        ttl_seconds=0,
        clock=lambda: NOW,
    )
    app.state.app_state = AppState(
        signal_store=SignalStore(db_path=str(tmp_path / "signals.db")),
        calendar_service=service,
    )
    app.include_router(calendar_router)
    return app


def _pe(currency, ref="r1", source="bls", when=NOW + timedelta(hours=3)):
    return ProviderEvent(
        source=source,
        provider_ref=ref,
        series_code=None,
        event=f"{currency} event",
        currency=currency,
        scheduled_at=when,
        organism="Bureau of Labor Statistics",
        periodicity="monthly",
    )


_BLS_ATT = [ProviderAttribution("bls", "Bureau of Labor Statistics",
                                "Domaine public", "https://www.bls.gov/opub/copyright-information.htm")]


def test_calendar_returns_attached_events(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")], _BLS_ATT))
    body = client.get("/api/calendar").json()
    assert len(body["events"]) == 1
    assert body["events"][0]["markets"] == ["XAUUSD", "EURUSD"]
    assert body["coverage"]["source"] == "official"


def test_calendar_carries_attribution_for_served_sources(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")], _BLS_ATT))
    body = client.get("/api/calendar").json()
    assert len(body["attribution"]) == 1
    a = body["attribution"][0]
    assert a["organism"] == "Bureau of Labor Statistics"
    assert a["policy_url"].startswith("https://")


def test_calendar_drops_unattached_event(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD", "u"), _pe("GBP", "g")], _BLS_ATT))
    body = client.get("/api/calendar").json()
    assert [e["currency"] for e in body["events"]] == ["USD"]


def test_calendar_empty_when_no_source(tmp_path):
    client = TestClient(_make_app(tmp_path, []))
    body = client.get("/api/calendar").json()
    assert body["events"] == []
    assert body["attribution"] == []


def test_no_predictive_ranking_or_consensus_keys_leak(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")], _BLS_ATT))
    body = client.get("/api/calendar").json()
    for ev in body["events"]:
        assert FORBIDDEN_KEYS.isdisjoint(ev.keys())


def test_null_fields_serialize_as_null(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD", source="forexfactory")]))
    ev = client.get("/api/calendar").json()["events"][0]
    # forexfactory event: series_code + unit absent, serialized as null not defaulted
    assert ev["series_code"] is None
    assert ev["value_unit"] is None
