"""Tests for the GET /api/candles endpoint (live-data wiring).

The endpoint is STRICTLY DESCRIPTIVE: it serves only OHLC + UTC epoch timestamps
read from candles_cache. These tests pin that contract and assert that no
predictive (InsightSignalV2) field ever leaks into the response.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import AppState
from src.api.routes.candles import router as candles_router
from src.api.signal_store import SignalStore
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage import CandlesCacheStore

# Fields that belong to InsightSignalV2 and must NEVER appear in /api/candles.
FORBIDDEN_PREDICTIVE_KEYS = frozenset(
    {
        "forecast",
        "forecast_atr_pips",
        "confidence_interval",
        "conformal",
        "hmm_posterior",
        "bocpd_changepoint_prob",
        "target_1",
        "target_2",
        "valid_until",
        "valid_until_utc",
        "confluence_signal",
        "confluence_score",
        "score",
        "bias",
        "direction",
    }
)


class _StubAssembler:
    """Mimics the assembler exposing a populated candles_store property."""

    def __init__(self, store):
        self.candles_store = store


def _candle(seconds: int, close: float) -> Candle:
    return Candle(
        ts=datetime(2026, 5, 29, 14, 0, seconds, tzinfo=timezone.utc),
        open=close - 1.0,
        high=close + 2.0,
        low=close - 2.0,
        close=close,
        volume=100.0 + seconds,
    )


def _make_app(*, with_store: bool = True, tmp_path, seed=True) -> FastAPI:
    app = FastAPI()
    signal_store = SignalStore(db_path=str(tmp_path / "signals.db"))
    assembler = None
    if with_store:
        candles_store = CandlesCacheStore(db_path=str(tmp_path / "candles.db"))
        if seed:
            candles_store.upsert_candles(
                "XAUUSD",
                "M15",
                [_candle(i, 2378.0 + i) for i in range(5)],
            )
        assembler = _StubAssembler(candles_store)
    app.state.app_state = AppState(
        signal_store=signal_store,
        market_reading_assembler=assembler,
    )
    app.include_router(candles_router)
    return app


def test_returns_n_well_formed_candles(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["instrument"] == "XAUUSD"
    assert body["timeframe"] == "M15"
    candles = body["candles"]
    assert len(candles) == 5
    # Each candle has exactly the descriptive OHLC keys, nothing more.
    for c in candles:
        assert set(c.keys()) == {"time", "open", "high", "low", "close", "volume"}
        assert isinstance(c["time"], int)
        assert c["high"] >= c["low"]
    # Ascending by time (chart expects oldest-first; never a future projection).
    times = [c["time"] for c in candles]
    assert times == sorted(times)


def test_time_is_utc_epoch_seconds(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    first = resp.json()["candles"][0]
    expected = int(datetime(2026, 5, 29, 14, 0, 0, tzinfo=timezone.utc).timestamp())
    assert first["time"] == expected


def test_limit_caps_window(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 2},
    )
    assert resp.status_code == 200
    candles = resp.json()["candles"]
    assert len(candles) == 2
    # Most recent two, still ascending.
    assert [c["close"] for c in candles] == [2381.0, 2382.0]


def test_no_predictive_field_leaks(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    raw = resp.text.lower()
    for key in FORBIDDEN_PREDICTIVE_KEYS:
        assert key.lower() not in raw, f"predictive field '{key}' leaked into /api/candles"


def test_rejects_unsupported_instrument(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "BTCUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 400
    assert "Unsupported instrument" in resp.json()["detail"]


def test_rejects_unsupported_timeframe(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M30"}
    )
    assert resp.status_code == 400
    assert "Unsupported timeframe" in resp.json()["detail"]


def test_serves_daily_reference_series(tmp_path):
    """D1/W1 are served (read-only reference series for the Régime panel's
    calendar levels) — the 'day'/'week' boundary is the feed's own candle."""
    app = FastAPI()
    signal_store = SignalStore(db_path=str(tmp_path / "signals.db"))
    candles_store = CandlesCacheStore(db_path=str(tmp_path / "candles.db"))
    candles_store.upsert_candles(
        "XAUUSD", "D1", [_candle(i, 2400.0 + i) for i in range(3)]
    )
    app.state.app_state = AppState(
        signal_store=signal_store,
        market_reading_assembler=_StubAssembler(candles_store),
    )
    app.include_router(candles_router)
    client = TestClient(app)
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "D1"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeframe"] == "D1"
    assert len(body["candles"]) == 3


def test_d1_populated_on_demand_when_cache_empty(tmp_path):
    """D1/W1 aren't kept warm; a cache-miss triggers `warm_candles` (feed fetch),
    then the endpoint serves the freshly-cached reference series."""

    class _WarmingAssembler:
        def __init__(self, store):
            self.candles_store = store
            self.warmed: list = []

        def warm_candles(self, instrument, timeframe):
            self.warmed.append((instrument, timeframe))
            rows = [_candle(i, 2400.0 + i) for i in range(3)]
            self.candles_store.upsert_candles(instrument, timeframe, rows)
            return len(rows)

    app = FastAPI()
    signal_store = SignalStore(db_path=str(tmp_path / "signals.db"))
    store = CandlesCacheStore(db_path=str(tmp_path / "candles.db"))
    assembler = _WarmingAssembler(store)
    app.state.app_state = AppState(signal_store=signal_store, market_reading_assembler=assembler)
    app.include_router(candles_router)
    client = TestClient(app)

    resp = client.get("/api/candles", params={"instrument": "XAUUSD", "timeframe": "D1"})
    assert resp.status_code == 200
    assert len(resp.json()["candles"]) == 3
    assert assembler.warmed == [("XAUUSD", "D1")]  # populated on demand


def test_m15_cache_miss_does_not_trigger_warm(tmp_path):
    """A chart timeframe (M15) is scheduler-warmed — a miss is an honest 404, not
    an on-demand feed fetch (only D1/W1 are populated lazily)."""
    app = _make_app(seed=False, tmp_path=tmp_path)  # store present, nothing seeded
    client = TestClient(app)
    resp = client.get("/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"})
    assert resp.status_code == 404


def test_404_when_combo_has_no_cached_candles(tmp_path):
    # Valid combo (EURUSD/H4) but nothing seeded for it.
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "EURUSD", "timeframe": "H4"}
    )
    assert resp.status_code == 404
    assert "No candles cached" in resp.json()["detail"]


def test_503_when_store_not_wired(tmp_path):
    client = TestClient(_make_app(with_store=False, tmp_path=tmp_path))
    resp = client.get(
        "/api/candles", params={"instrument": "XAUUSD", "timeframe": "M15"}
    )
    assert resp.status_code == 503
    assert "not configured" in resp.json()["detail"]


def test_limit_out_of_range_returns_422(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 0},
    )
    assert resp.status_code == 422


def test_limit_cannot_exceed_documented_cap(tmp_path):
    # PERF-2 guard: no request may ask for more than N candles. N is defined and
    # documented as MAX_LIMIT (1000) in the route; asking for more is rejected, so
    # a client can never pull unbounded history on the load path.
    from src.api.routes.candles import MAX_LIMIT

    assert MAX_LIMIT == 1000
    client = TestClient(_make_app(tmp_path=tmp_path))
    resp = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": MAX_LIMIT + 1},
    )
    assert resp.status_code == 422
    # The documented cap itself is accepted (not off-by-one rejected).
    ok = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": MAX_LIMIT},
    )
    assert ok.status_code in (200, 404)  # 404 only if the combo has no cached bars


def test_router_wired_into_app_module():
    from src.api import app as app_module

    assert hasattr(app_module, "candles") or "candles" in dir(app_module)


# ── CHART-1: history pagination (before / has_more) ───────────────────────────
def _epoch(seconds):
    return int(datetime(2026, 5, 29, 14, 0, seconds, tzinfo=timezone.utc).timestamp())


def test_initial_reports_has_more_when_older_candles_exist(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    r = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 2},
    )
    b = r.json()
    assert [c["time"] for c in b["candles"]] == [_epoch(3), _epoch(4)]
    assert b["has_more"] is True  # ts0..ts2 are older than this window


def test_before_paginates_the_previous_page_contiguously(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    # The 2 candles just OLDER than ts3 = ts1, ts2 (ascending); ts0 remains older.
    r = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 2, "before": _epoch(3)},
    )
    assert r.status_code == 200
    b = r.json()
    assert [c["time"] for c in b["candles"]] == [_epoch(1), _epoch(2)]
    assert b["has_more"] is True
    # Next page before ts1 → only ts0, and now we are at the floor.
    r2 = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 2, "before": _epoch(1)},
    )
    b2 = r2.json()
    assert [c["time"] for c in b2["candles"]] == [_epoch(0)]
    assert b2["has_more"] is False


def test_before_at_the_floor_is_200_empty_not_404(tmp_path):
    client = TestClient(_make_app(tmp_path=tmp_path))
    r = client.get(
        "/api/candles",
        params={"instrument": "XAUUSD", "timeframe": "M15", "limit": 5, "before": _epoch(0)},
    )
    # Reaching the start of history is a normal empty 200 (so the chart can say
    # "start of available data"), never an error.
    assert r.status_code == 200
    assert r.json()["candles"] == []
    assert r.json()["has_more"] is False
