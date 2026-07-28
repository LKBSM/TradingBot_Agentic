"""LB-1 — incremental, persisted detection: NON-REGRESSION vs full recompute.

The mission's hard gate: on a frozen candle set, feeding candles one at a time
into the persisted incremental driver must produce EXACTLY the same surfaced
zones and the same structure events as a single full recompute of that set — on
all six timeframes. Any drift means the ENGINE moved, not the display, and must
stop the mission. Also covers: idempotent replay (no duplicates), the event
journal existing by construction, and refresh work bounded by the window (not the
storage depth).
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from src.intelligence import incremental_detection as inc
from src.intelligence.incremental_detection import IncrementalDetector, _analyze_window
from src.intelligence.data_providers.twelve_data_provider import Candle
from src.storage.candles_cache_store import CandlesCacheStore
from src.storage.structure_store import StructureStore

# Frozen-set size per timeframe: below the detection window (800) so the final
# replay step sees the identical full set — the clean regime where incremental
# and full recompute are provably equal. Kept modest to bound the per-step engine
# runs in this test.
N = 120


def _tf_minutes(tf: str) -> int:
    return {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}[tf]


def _synthetic(tf: str, n: int):
    """Deterministic OHLCV — a seeded pseudo-random walk (no RNG, reproducible).

    Used for M1 (not backfilled while gated off) so the sixth unit is exercised
    through the real M1 store/detection path. Structure-bearing enough to form
    OBs/FVGs and BOS/CHOCH: the walk trends and reverses.
    """
    step = timedelta(minutes=_tf_minutes(tf))
    t0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
    out = []
    price = 2000.0
    for i in range(n):
        # smooth deterministic oscillation + drift → swings the engine can read
        drift = math.sin(i / 7.0) * 6.0 + math.sin(i / 23.0) * 14.0
        price = 2000.0 + drift
        o = price
        c = 2000.0 + math.sin((i + 1) / 7.0) * 6.0 + math.sin((i + 1) / 23.0) * 14.0
        hi = max(o, c) + 1.5 + (i % 3)
        lo = min(o, c) - 1.5 - (i % 2)
        out.append(Candle(ts=t0 + step * i, open=o, high=hi, low=lo, close=c, volume=100.0 + i))
    return out


def _load_candles(tf: str):
    """Real backfilled candles for enabled TFs; synthetic for M1."""
    if tf == "M1":
        return _synthetic("M1", N)
    store = CandlesCacheStore()
    candles = store.get_last_n_candles("XAUUSD", tf, N)
    if len(candles) < N:
        pytest.skip(f"{tf}: only {len(candles)} candles cached (<{N}); run the backfill")
    return candles


def _zone_key(z) -> tuple:
    """Canonical zone identity for comparison (dict from collect_zones OR StoredZone)."""
    if isinstance(z, dict):
        return (
            z["_type"], z["direction"], round(float(z["level_high"]), 5),
            round(float(z["level_low"]), 5),
            z["created_at"].isoformat() if z.get("created_at") else None,
            str(z.get("status", "active")), bool(z.get("tested")),
        )
    return (
        z.zone_type, z.direction, round(z.level_high, 5), round(z.level_low, 5),
        z.created_at.isoformat() if z.created_at else None, z.status, z.tested,
    )


def _full_pass_zone_keys(candles) -> set:
    zones, _events = _analyze_window(candles)
    keys = set()
    for zt, key in (("ob", "order_blocks"), ("fvg", "fair_value_gaps")):
        for z in zones.get(key, []):
            z = {**z, "_type": zt}
            keys.add(_zone_key(z))
    return keys


def _full_pass_event_keys(candles) -> set:
    _zones, events = _analyze_window(candles)
    keys = set()
    for et, key in (("bos", "bos_events"), ("choch", "choch_events")):
        for e in events.get(key, []):
            keys.add((et, e["direction"], round(float(e["level"]), 5), e["broken_at"].isoformat()))
    return keys


def _store_zone_keys(store: StructureStore) -> set:
    return {_zone_key(z) for z in store.get_current_zones("XAUUSD", "TF")}


ALL_TFS = ["M1", "M5", "M15", "H1", "H4", "D1"]


@pytest.mark.parametrize("tf", ALL_TFS)
def test_incremental_equals_full_recompute(tf, tmp_path):
    candles = _load_candles(tf)

    # Full recompute of the frozen set.
    full_zone_keys = _full_pass_zone_keys(candles)
    full_event_keys = _full_pass_event_keys(candles)

    # Incremental: feed one candle at a time into the persisted driver.
    store = StructureStore(db_path=str(tmp_path / f"struct_{tf}.db"))
    det = IncrementalDetector(store)
    det.replay("XAUUSD", tf, candles)

    store_zone_keys = {_zone_key(z) for z in store.get_current_zones("XAUUSD", tf)}
    store_event_keys = {
        (e["event_type"], e["direction"], round(float(e["level"]), 5),
         datetime.fromisoformat(e["event_ts"]).isoformat())
        for e in store.get_event_journal("XAUUSD", tf)
    }

    assert store_zone_keys == full_zone_keys, (
        f"{tf}: zone drift incremental vs full — "
        f"only_store={store_zone_keys - full_zone_keys}, only_full={full_zone_keys - store_zone_keys}"
    )
    assert store_event_keys == full_event_keys, (
        f"{tf}: event drift incremental vs full — "
        f"only_store={store_event_keys - full_event_keys}, only_full={full_event_keys - store_event_keys}"
    )


def test_event_journal_exists_by_construction(tmp_path):
    """A non-trivial series leaves BOS/CHOCH in the persisted journal."""
    candles = _synthetic("M1", N)
    store = StructureStore(db_path=str(tmp_path / "j.db"))
    IncrementalDetector(store).replay("XAUUSD", "M1", candles)
    journal = store.get_event_journal("XAUUSD", "M1")
    assert len(journal) > 0
    # Journal is chronologically ordered and typed.
    ts = [e["event_ts"] for e in journal]
    assert ts == sorted(ts)
    assert all(e["event_type"] in ("bos", "choch") for e in journal)


def test_replay_is_idempotent(tmp_path):
    """Replaying the same frozen set twice duplicates nothing."""
    candles = _synthetic("M1", N)
    store = StructureStore(db_path=str(tmp_path / "idem.db"))
    det = IncrementalDetector(store)
    det.replay("XAUUSD", "M1", candles)
    z1 = store.get_current_zones("XAUUSD", "M1")
    e1 = store.get_event_journal("XAUUSD", "M1")
    det.replay("XAUUSD", "M1", candles)
    z2 = store.get_current_zones("XAUUSD", "M1")
    e2 = store.get_event_journal("XAUUSD", "M1")
    assert len(z1) == len(z2)
    assert len(e1) == len(e2)


def test_refresh_work_is_bounded_by_window_not_depth(tmp_path, monkeypatch):
    """Refresh cost is independent of storage depth: only the trailing window is
    ever analysed, however deep the history handed in."""
    seen_lengths = []
    real = inc._analyze_window

    def _spy(candles):
        seen_lengths.append(len(candles))
        return real(candles)

    monkeypatch.setattr(inc, "_analyze_window", _spy)
    store = StructureStore(db_path=str(tmp_path / "b.db"))
    det = IncrementalDetector(store, window_bars=50)
    deep = _synthetic("M1", 400)  # 400 candles of "storage"

    det.refresh("XAUUSD", "M1", deep)

    assert seen_lengths == [50]  # analysed the last 50 only, not all 400
