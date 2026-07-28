"""GET /api/calendar endpoint (NW-1). Descriptive-only; injected service."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.calendar import router as calendar_router
from src.api.signal_store import SignalStore
from src.intelligence.calendar_providers.base import ProviderEvent, ProviderFetch
from src.intelligence.calendar_providers.base import CalendarProvider
from src.intelligence.calendar_service import CalendarService
from src.storage.calendar_cache_store import CalendarCacheStore

NOW = datetime(2026, 7, 28, 6, 0, tzinfo=timezone.utc)

# No prediction/direction fields may ever appear in the calendar payload.
FORBIDDEN_KEYS = frozenset(
    {"direction", "bias", "target", "target_1", "score", "confidence", "recommendation"}
)


class _FakeProvider(CalendarProvider):
    def __init__(self, events):
        self._events = events

    @property
    def source_name(self) -> str:
        return "official"

    def fetch(self) -> ProviderFetch:
        return ProviderFetch(events=self._events)


def _make_app(tmp_path, events):
    app = FastAPI()
    service = CalendarService(
        provider=_FakeProvider(events),
        store=CalendarCacheStore(db_path=str(tmp_path / "cal.db")),
        market_map={"XAUUSD": ["USD"], "EURUSD": ["USD", "EUR"]},
        clock=lambda: NOW,
    )
    app.state.app_state = AppState(
        signal_store=SignalStore(db_path=str(tmp_path / "signals.db")),
        calendar_service=service,
    )
    app.include_router(calendar_router)
    return app


def _pe(currency, ref="r1", when=NOW + timedelta(hours=3)):
    return ProviderEvent(
        source="official",
        provider_ref=ref,
        series_code=None,
        event=f"{currency} event",
        currency=currency,
        impact="high",
        scheduled_at=when,
    )


def test_calendar_returns_attached_events(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")]))
    r = client.get("/api/calendar")
    assert r.status_code == 200
    body = r.json()
    assert len(body["events"]) == 1
    assert body["events"][0]["markets"] == ["XAUUSD", "EURUSD"]
    assert body["coverage"]["source"] == "official"


def test_calendar_drops_unattached_event(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD", "u"), _pe("GBP", "g")]))
    body = client.get("/api/calendar").json()
    assert [e["currency"] for e in body["events"]] == ["USD"]


def test_calendar_empty_when_no_source(tmp_path):
    # Official stub semantics: no events → honest empty payload, not an error.
    client = TestClient(_make_app(tmp_path, []))
    body = client.get("/api/calendar").json()
    assert body["events"] == []


def test_no_predictive_keys_leak(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")]))
    body = client.get("/api/calendar").json()
    for ev in body["events"]:
        assert FORBIDDEN_KEYS.isdisjoint(ev.keys())


def test_null_fields_serialize_as_null(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD")]))
    ev = client.get("/api/calendar").json()["events"][0]
    assert ev["organism"] is None
    assert ev["series_code"] is None
    assert ev["value_unit"] is None
