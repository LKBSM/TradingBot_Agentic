"""NW-7d — deep (paginated) backfill walks history backward and self-terminates.

A single Twelve Data request caps at ~5000 bars (≈52 days of M15), far short of
the months the publication measures need. deep_backfill_combo pages backward via
end_date until the target is met or the provider has no older data. Deterministic:
a fake provider serves a finite window; no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.intelligence.history_backfill import (
    deep_backfill_combo,
    maintain_deep_history,
)

TF = timedelta(minutes=15)
NEWEST = datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc)


class _FakeStore:
    """Segregates by (instrument, timeframe), like CandlesCacheStore."""

    def __init__(self):
        self.by_combo = {}

    def upsert_candles(self, instrument, timeframe, candles):
        self.by_combo.setdefault((instrument, timeframe), []).extend(candles)
        return len(candles)

    def rows_for(self, instrument, timeframe):
        return self.by_combo.setdefault((instrument, timeframe), [])

    def get_coverage(self, instrument, timeframe):
        rows = self.by_combo.get((instrument, timeframe), [])
        if not rows:
            return SimpleNamespace(count=0, oldest_ts=None, newest_ts=None)
        ts = [r.ts for r in rows]
        return SimpleNamespace(count=len(rows), oldest_ts=min(ts), newest_ts=max(ts))


class _FakePagingProvider:
    """Serves M15 bars from [oldest, NEWEST], up to `count` ending at end_date."""

    def __init__(self, oldest: datetime):
        self.oldest = oldest
        self.calls = 0

    def fetch_candles_until(self, symbol, timeframe, count, end_date=None):
        self.calls += 1
        end = NEWEST if not end_date else datetime.strptime(
            end_date, "%Y-%m-%d %H:%M:%S"
        ).replace(tzinfo=timezone.utc)
        bars = []
        t = end
        for _ in range(int(count)):
            if t < self.oldest:
                break
            bars.append(SimpleNamespace(ts=t))
            t -= TF
        return list(reversed(bars))  # chronological (oldest→newest)


def test_deep_backfill_pages_back_to_available_history():
    # ~4 months of M15 available; ask for a big target with a small page → several pages.
    oldest = NEWEST - timedelta(days=120)
    provider = _FakePagingProvider(oldest)
    store = _FakeStore()
    res = deep_backfill_combo(
        provider, store, "XAUUSD", "M15", target_bars=100000, page=2000, max_pages=40
    )
    # It fetched roughly the whole 120-day window (120*96 ≈ 11520 bars).
    assert res["fetched"] >= 11000
    assert provider.calls >= 5  # multiple pages, not one capped request
    # Oldest reached is at (or one bar after) the provider's floor.
    assert res["oldest"] is not None
    assert datetime.fromisoformat(res["oldest"]) <= oldest + TF


def test_deep_backfill_stops_when_depth_exhausted():
    # Only ~10 days available; a large target must NOT loop forever.
    oldest = NEWEST - timedelta(days=10)
    provider = _FakePagingProvider(oldest)
    store = _FakeStore()
    res = deep_backfill_combo(
        provider, store, "XAUUSD", "M15", target_bars=100000, page=5000, max_pages=40
    )
    assert res["fetched"] >= 900          # ~10 days * 96
    assert res["pages"] < 40              # terminated by depth, not the page cap
    assert datetime.fromisoformat(res["oldest"]) <= oldest + TF


def test_deep_backfill_respects_target():
    oldest = NEWEST - timedelta(days=365)
    provider = _FakePagingProvider(oldest)
    store = _FakeStore()
    res = deep_backfill_combo(
        provider, store, "XAUUSD", "M15", target_bars=6000, page=5000, max_pages=40
    )
    # Stops once the target is met (2 pages of 5000 cover 6000), not the full year.
    assert 6000 <= res["fetched"] <= 11000
    assert res["pages"] <= 2


def test_deep_backfill_resumes_from_cache_and_extends_backward():
    # Store already holds the most recent ~3 days; the fill must RESUME from the
    # oldest cached bar and only fetch OLDER history (never re-spend on recent).
    oldest_floor = NEWEST - timedelta(days=200)
    provider = _FakePagingProvider(oldest_floor)
    store = _FakeStore()
    recent = store.rows_for("XAUUSD", "M15")
    t = NEWEST
    while t > NEWEST - timedelta(days=3):
        recent.append(SimpleNamespace(ts=t))
        t -= TF
    have0 = len(recent)
    first_page_end = {"seen": None}
    orig = provider.fetch_candles_until

    def _spy(symbol, tf, count, end_date=None):
        if first_page_end["seen"] is None:
            first_page_end["seen"] = end_date
        return orig(symbol, tf, count, end_date)

    provider.fetch_candles_until = _spy
    res = deep_backfill_combo(
        provider, store, "XAUUSD", "M15", target_bars=100000, page=5000, max_pages=40
    )
    # The very first request ended BEFORE the oldest cached bar (i.e. it resumed).
    assert first_page_end["seen"] is not None
    assert res["have"] == have0
    assert res["fetched"] > 0
    assert res["total"] == have0 + res["fetched"]


def test_maintain_deep_history_covers_every_market():
    provider = _FakePagingProvider(NEWEST - timedelta(days=120))
    store = _FakeStore()
    results = maintain_deep_history(
        provider, store, ["XAUUSD", "EURUSD"],
        timeframe="M15", target_bars_for=lambda m, tf: 4000, max_pages_per_market=3,
    )
    assert [r["instrument"] for r in results] == ["XAUUSD", "EURUSD"]
    assert all(r["fetched"] > 0 for r in results)
