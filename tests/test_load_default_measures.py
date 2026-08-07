"""NW-7 — load_default_measures reads the PROD-warm stores, not the gitignored CSVs.

The measures were silently empty in prod because their loader read two gitignored
CSVs absent from the container. This proves the loader now computes from injected
stores (candles.db + calendar_cache.db), so the questions can render on real prod
data without any bundled file.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from src.intelligence.publication_measures import load_default_measures
from src.intelligence.calendar_providers.official_sources.base_official import load_catalog

START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)
BAR = timedelta(minutes=15)


class _FakeCandlesStore:
    """Mimics CandlesCacheStore.get_last_n_candles with synthetic gold M15 bars."""

    def __init__(self, n_days: int = 40):
        n = n_days * 96
        rng = np.random.default_rng(3)
        close = 2000.0 + np.cumsum(rng.normal(0.0, 0.2, n))
        self._rows = [
            SimpleNamespace(
                ts=START + i * BAR,
                open=float(close[i]),
                high=float(close[i]) + 0.6,
                low=float(close[i]) - 0.6,
                close=float(close[i]),
                volume=100.0,
            )
            for i in range(n)
        ]

    def get_last_n_candles(self, instrument, timeframe, n):
        assert timeframe == "M15"
        return self._rows[-n:]


class _FakeCalendarStore:
    """Mimics CalendarCacheStore.get_events_between with past CPI events, matched
    to the catalog series code (never by title)."""

    def __init__(self, series_code: str, moments):
        self._events = [
            SimpleNamespace(series_code=series_code, scheduled_at=t) for t in moments
        ]

    def get_events_between(self, start, end):
        return [e for e in self._events if start <= e.scheduled_at <= end]


def test_load_default_measures_computes_from_injected_stores():
    now = START + timedelta(days=41)
    series = load_catalog()["us_cpi"].series_code  # CUUR0000SA0
    # 8 monthly-ish CPI releases at 13:00 within the candle coverage.
    moments = [START + timedelta(days=3 + 4 * i, hours=13) for i in range(8)]
    measures = load_default_measures(
        event_key="us_cpi",
        market="XAUUSD",
        now=now,
        candles_store=_FakeCandlesStore(),
        calendar_store=_FakeCalendarStore(series, moments),
    )
    assert measures is not None
    assert measures.has_any(), "expected measures to compute from the stores"
    # At least the two baseline-free / baseline measures came through with honest
    # denominators (never fabricated).
    for m in (measures.calm_before, measures.structure_state, measures.return_to_calm):
        if m is not None:
            assert m.provenance.sample_size >= 4
            assert m.provenance.market == "XAUUSD"


def test_pick_gold_m15_prefers_deep_csv_over_shallow_store(monkeypatch):
    """A shallow candles.db (only weeks accumulated since deploy) must NOT be
    preferred over a CSV holding years — else the measures fail on too few
    in-coverage releases (NW-7b)."""
    import pandas as pd

    import src.intelligence.publication_measures as pm

    def _df(days: int, start: datetime):
        n = days * 96
        idx = pd.date_range(start, periods=n, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"open": 2000.0, "high": 2000.6, "low": 1999.4, "close": 2000.0, "volume": 1.0},
            index=idx,
        )

    shallow = _df(20, START)                      # ~20 days in candles.db
    deep = _df(500, START - timedelta(days=520))  # ~500 days in the CSV
    monkeypatch.setattr(pm, "_gold_m15_from_store", lambda *a, **k: shallow)
    monkeypatch.setattr(pm, "_gold_m15_from_csv", lambda *a, **k: deep)

    picked = pm._pick_gold_m15("XAUUSD")
    assert pm._span_days(picked) == pm._span_days(deep)  # the deep CSV won


def test_pick_gold_m15_prefers_store_when_deep_enough(monkeypatch):
    import pandas as pd

    import src.intelligence.publication_measures as pm

    def _df(days: int):
        n = days * 96
        idx = pd.date_range(START, periods=n, freq="15min", tz="UTC")
        return pd.DataFrame(
            {"open": 2000.0, "high": 2000.6, "low": 1999.4, "close": 2000.0, "volume": 1.0},
            index=idx,
        )

    deep_store = _df(400)
    called = {"csv": False}

    def _csv(*a, **k):
        called["csv"] = True
        return None

    monkeypatch.setattr(pm, "_gold_m15_from_store", lambda *a, **k: deep_store)
    monkeypatch.setattr(pm, "_gold_m15_from_csv", _csv)
    picked = pm._pick_gold_m15("XAUUSD")
    assert pm._span_days(picked) == pm._span_days(deep_store)
    assert called["csv"] is False  # deep store used without touching the CSV


def test_releases_from_store_matches_by_event_key_for_null_series():
    """FOMC & co. have no series_code — their past releases must still be found,
    matched by the event's KEY (its event_id middle segment)."""
    from src.intelligence.publication_measures import _releases_from_store

    now = datetime(2026, 8, 1, tzinfo=timezone.utc)

    class _Store:
        def get_events_between(self, a, b):
            return [
                SimpleNamespace(
                    event_id=f"federal_reserve:us_fomc_rate:2026-{m:02d}-01",
                    series_code=None,
                    scheduled_at=datetime(2026, m, 1, 18, 0, tzinfo=timezone.utc),
                )
                for m in range(1, 6)
            ]

    rel = _releases_from_store("us_fomc_rate", now, _Store())
    assert len(rel) == 5  # matched by event_key despite no series code


def test_measured_markets_covers_every_catalog_publication():
    """Every catalog publication is measured (no publication left without the 4
    questions), and every measured key is a real catalog publication."""
    from src.intelligence.calendar_providers.official_sources.base_official import (
        load_catalog,
    )
    from src.intelligence.publication_measures import MEASURED_MARKETS

    cat_keys = set(load_catalog().keys())
    assert set(MEASURED_MARKETS) == cat_keys, (
        f"absentes des mesures: {cat_keys - set(MEASURED_MARKETS)} ; "
        f"clés inconnues: {set(MEASURED_MARKETS) - cat_keys}"
    )
    assert all(m in ("XAUUSD", "EURUSD") for m in MEASURED_MARKETS.values())


def test_load_default_measures_none_when_stores_empty():
    now = START + timedelta(days=41)
    series = load_catalog()["us_cpi"].series_code
    # No candles at all → no measures (honest empty, not a crash).
    class _Empty:
        def get_last_n_candles(self, *a):
            return []

    measures = load_default_measures(
        event_key="us_cpi",
        market="XAUUSD",
        now=now,
        candles_store=_Empty(),
        calendar_store=_FakeCalendarStore(series, [START + timedelta(days=3, hours=13)]),
    )
    # Releases present but no candles (and no bundled CSV in the test env) → None.
    assert measures is None
