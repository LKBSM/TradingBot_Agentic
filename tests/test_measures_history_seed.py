"""NW-7 — the bundled deep-history seed loads candles.db at boot (no Twelve Data).

The publication measures need months of intraday history; on a fresh deploy
candles.db is too shallow and the free Twelve Data quota is tight. The seed loads
a bundled gz into candles.db once, generically, idempotently — proven here with a
small synthetic seed (the real XAUUSD seed is exercised end-to-end elsewhere).
"""

from __future__ import annotations

import gzip
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import src.intelligence.measures_history_seed as seed


def _write_seed(dirpath, symbol, tf, *, days=300):
    """A small gz OHLC seed spanning ``days`` (so it counts as 'deep enough')."""
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    step = timedelta(minutes=15)
    n = days * 96
    path = dirpath / f"{symbol}_{tf}.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8", newline="") as fh:
        fh.write("Date,Open,High,Low,Close,Volume\n")
        for i in range(0, n, 96):  # one row/day keeps the file tiny but spans `days`
            ts = start + step * i
            fh.write(f"{ts.strftime('%Y-%m-%d %H:%M:%S')},2000.0,2001.0,1999.0,2000.5,10\n")
    return path


class _FakeStore:
    def __init__(self):
        self.rows = []

    def get_coverage(self, instrument, timeframe):
        if not self.rows:
            return SimpleNamespace(count=0, oldest_ts=None, newest_ts=None)
        ts = [c.ts for c in self.rows]
        return SimpleNamespace(count=len(self.rows), oldest_ts=min(ts), newest_ts=max(ts))

    def upsert_candles(self, instrument, timeframe, candles):
        self.rows.extend(candles)
        return len(candles)


def test_seed_loads_when_shallow_then_is_idempotent(tmp_path, monkeypatch):
    _write_seed(tmp_path, "XAUUSD", "M15", days=300)
    monkeypatch.setattr(seed, "_seed_dir", lambda: tmp_path)

    store = _FakeStore()
    r1 = seed.seed_market(store, "XAUUSD", "M15")
    assert r1["seeded"] > 0 and r1["skipped"] is None
    assert len(store.rows) == r1["seeded"]
    # UTC tz-aware timestamps written into the store.
    assert store.rows[0].ts.tzinfo is not None

    # Second boot: candles.db already spans >200 days → skipped, no reload.
    r2 = seed.seed_market(store, "XAUUSD", "M15")
    assert r2["seeded"] == 0 and r2["skipped"] == "already_deep"


def test_seed_skips_when_no_bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(seed, "_seed_dir", lambda: tmp_path)  # empty dir
    r = seed.seed_market(_FakeStore(), "EURUSD", "M15")
    assert r["seeded"] == 0 and r["skipped"] == "no_seed_file"


def test_seed_measures_history_covers_measured_markets(tmp_path, monkeypatch):
    _write_seed(tmp_path, "XAUUSD", "M15", days=300)
    monkeypatch.setattr(seed, "_seed_dir", lambda: tmp_path)
    monkeypatch.setenv("SENTINEL_SEED_MEASURES", "1")

    store = _FakeStore()
    results = seed.seed_measures_history(store, markets=["XAUUSD"])
    assert results and results[0]["seeded"] > 0


def test_seed_can_be_disabled(tmp_path, monkeypatch):
    _write_seed(tmp_path, "XAUUSD", "M15", days=300)
    monkeypatch.setattr(seed, "_seed_dir", lambda: tmp_path)
    monkeypatch.setenv("SENTINEL_SEED_MEASURES", "0")
    assert seed.seed_measures_history(_FakeStore(), markets=["XAUUSD"]) == []


def test_real_bundled_xauusd_seed_is_present_and_parses():
    """The committed XAUUSD seed must exist and parse (guards against a lost or
    corrupt bundle — the whole feature depends on it shipping in the image)."""
    path = seed._seed_path("XAUUSD", "M15")
    assert path is not None, "bundled seed_data/XAUUSD_M15.csv.gz is missing"
    candles = seed._parse_seed(path)
    assert len(candles) > 50_000  # ~172k bars of gold M15 history
    assert candles[0].ts < candles[-1].ts
