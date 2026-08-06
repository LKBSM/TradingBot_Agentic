"""NW-7d — deep (paginated) backfill walks history backward and self-terminates.

A single Twelve Data request caps at ~5000 bars (≈52 days of M15), far short of
the months the publication measures need. deep_backfill_combo pages backward via
end_date until the target is met or the provider has no older data. Deterministic:
a fake provider serves a finite window; no network.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.intelligence.history_backfill import deep_backfill_combo

TF = timedelta(minutes=15)
NEWEST = datetime(2026, 8, 6, 0, 0, tzinfo=timezone.utc)


class _FakeStore:
    def __init__(self):
        self.rows = []

    def upsert_candles(self, instrument, timeframe, candles):
        self.rows.extend(candles)
        return len(candles)


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
