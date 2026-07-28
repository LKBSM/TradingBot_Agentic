"""LB-1 — historical backfill: depth-from-config, idempotent, quota, MC-1 aware.

Covers the mission's required backfill guarantees:
  · profondeur par unité de temps effectivement appliquée depuis la config ;
  · relancé deux fois → aucun doublon ;
  · quota respecté (fournisseur mocké, comptage des requêtes) ;
  · reprise : un combo déjà complet est sauté (aucun retéléchargement) ;
  · aucun appel fournisseur pour une période de marché fermé (règle MC-1).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.intelligence import history_backfill as hb
from src.intelligence import lookback_config, market_calendar
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage.candles_cache_store import CandlesCacheStore

UTC = timezone.utc
_TF_MIN = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


@pytest.fixture(autouse=True)
def _reset():
    lookback_config.reset_cache()
    market_calendar.reset_holiday_cache()
    yield
    lookback_config.reset_cache()


@pytest.fixture
def _no_m1(monkeypatch):
    monkeypatch.delenv("LB1_ENABLE_M1", raising=False)
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    lookback_config.reset_cache()


def _candles(n: int, timeframe: str, end_ts: datetime):
    """`n` ascending candles ending at end_ts, stepping back by the tf interval."""
    step = timedelta(minutes=_TF_MIN[timeframe])
    out = []
    for i in range(n):
        ts = end_ts - step * (n - 1 - i)
        out.append(Candle(ts=ts, open=100.0, high=101.0, low=99.0, close=100.5, volume=1.0))
    return out


class FakeProvider:
    """Records every fetch and returns a fixed candle set (never over-returns)."""

    def __init__(self, end_ts: datetime, returns: int = 10):
        self.calls = []
        self._end = end_ts
        self._returns = returns

    def fetch_candles(self, symbol: str, timeframe: str, count: int):
        self.calls.append((symbol, timeframe, count))
        return _candles(min(self._returns, count), timeframe, self._end)


def _store(tmp_path) -> CandlesCacheStore:
    return CandlesCacheStore(db_path=str(tmp_path / "candles.db"))


# --------------------------------------------------------------------------- #
# Depth from config
# --------------------------------------------------------------------------- #
def test_requests_depth_from_config(_no_m1, tmp_path):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)  # a Wednesday
    provider = FakeProvider(end_ts=now)
    store = _store(tmp_path)

    res = hb.backfill_combo(provider, store, "XAUUSD", "H1", now=now)

    assert res.requested == lookback_config.target_bars("XAUUSD", "H1")
    assert provider.calls[0] == ("XAUUSD", "H1", res.requested)
    assert res.skipped is False


# --------------------------------------------------------------------------- #
# Idempotent — twice, no duplicates
# --------------------------------------------------------------------------- #
def test_idempotent_no_duplicates(_no_m1, tmp_path):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    provider = FakeProvider(end_ts=now, returns=40)
    store = _store(tmp_path)

    hb.backfill_combo(provider, store, "XAUUSD", "H1", now=now, target_bars=40, force=True)
    count_after_first = store.count_candles("XAUUSD", "H1")
    hb.backfill_combo(provider, store, "XAUUSD", "H1", now=now, target_bars=40, force=True)
    count_after_second = store.count_candles("XAUUSD", "H1")

    assert count_after_first == 40
    assert count_after_second == 40  # same candles re-upserted → no growth


# --------------------------------------------------------------------------- #
# Quota — one request per non-skipped combo, well under 800/day
# --------------------------------------------------------------------------- #
def test_quota_one_request_per_combo(_no_m1, tmp_path):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    provider = FakeProvider(end_ts=now)
    store = _store(tmp_path)

    results = hb.backfill_all(provider, store, now=now)

    combos = lookback_config.enabled_combos()
    assert len(combos) == 10  # M1 off → 2 × 5
    assert len(provider.calls) == len(combos)
    assert sum(r.calls for r in results) == len(combos)
    assert len(provider.calls) < 800  # nowhere near the daily cap


def test_m1_enabled_adds_two_combos(monkeypatch, tmp_path):
    monkeypatch.setenv("LB1_ENABLE_M1", "1")
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    lookback_config.reset_cache()
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    provider = FakeProvider(end_ts=now)
    hb.backfill_all(provider, _store(tmp_path), now=now)
    assert len(provider.calls) == 12  # 2 × 6


# --------------------------------------------------------------------------- #
# Resumable — an already-complete combo is skipped (no re-download)
# --------------------------------------------------------------------------- #
def test_skips_already_complete_combo(_no_m1, tmp_path):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)  # Wednesday, market open
    store = _store(tmp_path)
    # Pre-fill deep + fresh: newest candle at the expected close.
    expected = market_calendar.market_aware_expected_close("XAUUSD", "H1", now)
    store.upsert_candles("XAUUSD", "H1", _candles(60, "H1", expected))

    provider = FakeProvider(end_ts=now)
    res = hb.backfill_combo(provider, store, "XAUUSD", "H1", now=now, target_bars=50)

    assert res.skipped is True
    assert res.reason == "already_complete"
    assert provider.calls == []  # zero calls: nothing to re-download


def test_refetches_when_stale_even_if_deep(_no_m1, tmp_path):
    now = datetime(2026, 7, 22, 12, 0, tzinfo=UTC)
    store = _store(tmp_path)
    # Deep but stale: newest candle is 10 days behind the expected close.
    stale_end = market_calendar.market_aware_expected_close("XAUUSD", "H1", now) - timedelta(days=10)
    store.upsert_candles("XAUUSD", "H1", _candles(60, "H1", stale_end))

    provider = FakeProvider(end_ts=now)
    res = hb.backfill_combo(provider, store, "XAUUSD", "H1", now=now, target_bars=50)

    assert res.skipped is False
    assert len(provider.calls) == 1


# --------------------------------------------------------------------------- #
# MC-1 — no provider call for a closed market when already complete
# --------------------------------------------------------------------------- #
def test_no_call_on_closed_market_when_complete(_no_m1, tmp_path):
    # Saturday → FX/metals closed. Expected close is frozen at Friday.
    now = datetime(2026, 7, 25, 12, 0, tzinfo=UTC)
    assert market_calendar.calendar_state("XAUUSD", now) != market_calendar.OPEN, "precondition: market closed"

    store = _store(tmp_path)
    friday_close = market_calendar.market_aware_expected_close("XAUUSD", "H4", now)
    store.upsert_candles("XAUUSD", "H4", _candles(60, "H4", friday_close))

    provider = FakeProvider(end_ts=now)
    res = hb.backfill_combo(provider, store, "XAUUSD", "H4", now=now, target_bars=50)

    assert res.skipped is True
    assert res.reason == "market_closed_complete"
    assert provider.calls == []  # zero Twelve Data calls over the weekend
