"""LB-1 — StructureStore: window-bounded queries + consumed reconciliation.

Fast, engine-free unit tests on hand-built snapshots (the engine-driven
non-regression lives in test_incremental_detection.py).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.storage.structure_store import StructureStore

UTC = timezone.utc
T0 = datetime(2026, 7, 1, tzinfo=UTC)


def _ob(direction, hi, lo, k, status="active", tested=False):
    return {
        "direction": direction, "level_high": hi, "level_low": lo,
        "created_at": T0 + timedelta(hours=k), "status": status, "tested": tested,
        "mitigated_at": None,
    }


def _bos(direction, level, k):
    return {"direction": direction, "level": level, "broken_at": T0 + timedelta(hours=k)}


def _store(tmp_path):
    return StructureStore(db_path=str(tmp_path / "s.db"))


def test_apply_and_read_current(tmp_path):
    s = _store(tmp_path)
    zones = {"order_blocks": [_ob("bullish", 2010, 2000, 1)], "fair_value_gaps": []}
    events = {"bos_events": [_bos("bullish", 2005, 1)], "choch_events": []}
    s.apply_snapshot("XAUUSD", "H1", zones, events, window_start=T0, processed_ts=T0 + timedelta(hours=5))
    cur = s.get_current_zones("XAUUSD", "H1")
    assert len(cur) == 1 and cur[0].direction == "bullish"
    assert len(s.get_event_journal("XAUUSD", "H1")) == 1


def test_consumed_zone_flipped_when_absent_in_window(tmp_path):
    s = _store(tmp_path)
    z1 = {"order_blocks": [_ob("bullish", 2010, 2000, 1)], "fair_value_gaps": []}
    s.apply_snapshot("XAUUSD", "H1", z1, {}, window_start=T0, processed_ts=T0 + timedelta(hours=1))
    assert s.count_zones("XAUUSD", "H1") == 1
    # Next snapshot (same window start) no longer surfaces it → consumed.
    s.apply_snapshot("XAUUSD", "H1", {"order_blocks": [], "fair_value_gaps": []}, {},
                     window_start=T0, processed_ts=T0 + timedelta(hours=2))
    assert s.count_zones("XAUUSD", "H1", surfaced_only=True) == 0
    assert s.count_zones("XAUUSD", "H1", surfaced_only=False) == 1  # kept as history


def test_aged_out_zone_is_frozen_not_consumed(tmp_path):
    s = _store(tmp_path)
    z1 = {"order_blocks": [_ob("bullish", 2010, 2000, 1)], "fair_value_gaps": []}
    s.apply_snapshot("XAUUSD", "H1", z1, {}, window_start=T0, processed_ts=T0 + timedelta(hours=1))
    # Window slid forward past the zone (window_start after its created_at) and it
    # is absent → it aged out, must stay surfaced (frozen), not marked consumed.
    later = T0 + timedelta(hours=10)
    s.apply_snapshot("XAUUSD", "H1", {"order_blocks": [], "fair_value_gaps": []}, {},
                     window_start=later, processed_ts=later)
    assert s.count_zones("XAUUSD", "H1", surfaced_only=True) == 1


def test_zones_in_price_window(tmp_path):
    s = _store(tmp_path)
    zones = {"order_blocks": [
        _ob("bullish", 2010, 2000, 1),
        _ob("bearish", 2110, 2100, 2),
    ], "fair_value_gaps": []}
    s.apply_snapshot("XAUUSD", "H1", zones, {}, window_start=T0, processed_ts=T0 + timedelta(hours=3))
    near = s.get_zones_in_window("XAUUSD", "H1", price_low=1995, price_high=2015)
    assert len(near) == 1 and near[0].level_low == 2000
    both = s.get_zones_in_window("XAUUSD", "H1", price_low=1995, price_high=2200)
    assert len(both) == 2


def test_events_in_time_window(tmp_path):
    s = _store(tmp_path)
    events = {"bos_events": [_bos("bullish", 2005, 1), _bos("bearish", 2105, 8)], "choch_events": []}
    s.apply_snapshot("XAUUSD", "H1", {"order_blocks": [], "fair_value_gaps": []}, events,
                     window_start=T0, processed_ts=T0 + timedelta(hours=9))
    recent = s.get_events_in_window("XAUUSD", "H1", time_from=T0 + timedelta(hours=5))
    assert len(recent) == 1 and recent[0]["direction"] == "bearish"


def test_idempotent_apply(tmp_path):
    s = _store(tmp_path)
    zones = {"order_blocks": [_ob("bullish", 2010, 2000, 1)], "fair_value_gaps": []}
    events = {"bos_events": [_bos("bullish", 2005, 1)], "choch_events": []}
    for _ in range(3):
        s.apply_snapshot("XAUUSD", "H1", zones, events, window_start=T0, processed_ts=T0)
    assert s.count_zones("XAUUSD", "H1") == 1
    assert len(s.get_event_journal("XAUUSD", "H1")) == 1
