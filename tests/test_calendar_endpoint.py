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


def test_event_by_id_loads_outside_the_list_window(tmp_path):
    """REC point 1: an event far beyond the list window (±30 days) is NOT in the
    list, but the per-event endpoint returns it by id — the detail no longer
    depends on a window."""
    far = _pe("USD", ref="far", when=NOW + timedelta(days=60))
    client = TestClient(_make_app(tmp_path, [far], _BLS_ATT))
    # not in the list (server caps at 30 days)
    assert client.get("/api/calendar?lookahead_days=30").json()["events"] == []
    # …but reachable by id, with its attribution
    body = client.get("/api/calendar/event/bls:far").json()
    assert len(body["events"]) == 1
    assert body["events"][0]["event_id"] == "bls:far"
    assert len(body["attribution"]) == 1


def test_event_by_id_empty_only_for_a_genuinely_unknown_id(tmp_path):
    client = TestClient(_make_app(tmp_path, [_pe("USD", ref="known")], _BLS_ATT))
    assert len(client.get("/api/calendar/event/bls:known").json()["events"]) == 1
    # a genuinely non-existent id → 200 with no event (never a 500)
    r = client.get("/api/calendar/event/bogus:nope")
    assert r.status_code == 200
    assert r.json()["events"] == []


def test_every_listed_event_is_reachable_by_id(tmp_path):
    """Mandatory REC test: every event shown in the calendar leads to a detail
    page that loads — no exception. Covers in-window AND far events together."""
    events = [
        _pe("USD", ref="a", when=NOW + timedelta(hours=2)),
        _pe("EUR", ref="b", when=NOW + timedelta(days=5)),
        _pe("USD", ref="c", when=NOW + timedelta(days=29)),
    ]
    atts = _BLS_ATT + [ProviderAttribution("bls", "Bureau of Labor Statistics", "pub", "https://x")]
    client = TestClient(_make_app(tmp_path, events, atts))
    listed = client.get("/api/calendar?lookahead_days=30&lookback_days=3").json()["events"]
    assert len(listed) >= 3
    for ev in listed:
        body = client.get(f"/api/calendar/event/{ev['event_id']}").json()
        assert len(body["events"]) == 1, f"{ev['event_id']} not reachable by id"
        assert body["events"][0]["event_id"] == ev["event_id"]


def test_real_default_provider_serves_events_over_http(tmp_path):
    """End-to-end production path: the REAL official aggregator (no injection) +
    the shipped schedule serve real, attributed events over the HTTP endpoint —
    the calendar is client-facing, not empty."""
    app = FastAPI()
    service = CalendarService(  # default provider = official aggregator + real config
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
    body = TestClient(app).get("/api/calendar?lookahead_days=30").json()
    assert len(body["events"]) >= 5
    assert len(body["attribution"]) >= 1
    ev = body["events"][0]
    assert ev["organism"] and ev["value_unit"] and ev["periodicity"]
    assert FORBIDDEN_KEYS.isdisjoint(ev.keys())
