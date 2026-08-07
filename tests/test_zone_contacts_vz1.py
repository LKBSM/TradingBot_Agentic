"""VZ-1 — per-contact ledger (edge_touch / entry_exit / traversal / inside),
consumed-zone surfacing and the order-block origin association.

All read-side and additive: these tests also assert the existing touch_count /
status semantics are unchanged (no detection change)."""

import pandas as pd

from src.intelligence.market_reading_mappers import (
    _ob_contacts,
    _fvg_contacts,
    _ob_origin,
    collect_zones,
)


def _frame(rows: list[dict], start: str = "2026-05-28T00:00:00Z") -> pd.DataFrame:
    idx = pd.date_range(start=start, periods=len(rows), freq="15min", tz="UTC")
    cols = [
        "high", "low", "close", "BULLISH_OB_HIGH", "BULLISH_OB_LOW",
        "BEARISH_OB_HIGH", "BEARISH_OB_LOW", "OB_STRENGTH_NORM",
        "FVG_DIR", "FVG_SIZE_NORM", "BOS_EVENT", "CHOCH_SIGNAL", "BOS_BREAK_LEVEL",
    ]
    data = {c: [r.get(c, float("nan")) for r in rows] for c in cols}
    return pd.DataFrame(data, index=idx)


# --------------------------------------------------------------------------- #
# The three distinct outcomes (mission question B)
# --------------------------------------------------------------------------- #
def test_ob_contacts_edge_touch_vs_entry_exit_are_distinct():
    """Zone [97,98] (height 1.0, edge fraction 0.10). A 0.05 kiss of the top edge
    is an edge_touch; a 0.6 penetration that leaves again is an entry_exit. The
    two are NEVER conflated, and each carries the deepest price reached."""
    zhigh, zlow = 98.0, 97.0
    highs = [100, 100, 100, 100, 100, 100, 100, 100]
    #        0    1     2    3    4     5    6    7
    lows = [99, 97.95, 99, 99, 97.4, 99, 99, 99]  # kiss@1, deep entry@4
    closes = [100] * 8
    contacts = _ob_contacts("bullish", zhigh, zlow, highs, lows, closes, created=0, upto=7)
    assert [c["outcome"] for c in contacts] == ["edge_touch", "entry_exit"]
    assert round(contacts[0]["level"], 2) == 97.95  # deepest of the kiss
    assert round(contacts[1]["level"], 2) == 97.40  # deepest of the entry


def test_ob_contact_inside_when_price_currently_in_zone():
    """A run that is still open at the end of the window is `inside` (ongoing),
    never prematurely labelled an exit."""
    zhigh, zlow = 98.0, 97.0
    highs = [100, 100, 100]
    lows = [99, 99, 97.3]  # enters on the last bar and stays
    closes = [100, 100, 97.5]  # closes inside the band (no close-through)
    contacts = _ob_contacts("bullish", zhigh, zlow, highs, lows, closes, 0, 2)
    assert [c["outcome"] for c in contacts] == ["inside"]


def test_ob_traversal_is_a_distinct_outcome_and_ends_the_ledger():
    zhigh, zlow = 98.0, 97.0
    highs = [100, 100, 100, 100]
    lows = [99, 97.4, 96.0, 99]
    closes = [100, 100, 96.0, 100]  # bar 2 closes below zlow → traversal
    contacts = _ob_contacts("bullish", zhigh, zlow, highs, lows, closes, 0, 3)
    assert contacts[-1]["outcome"] == "traversal"
    assert contacts[-1]["level"] == zlow  # crossed to the far edge


def test_ob_contacts_do_not_change_touch_count_semantics():
    """The ledger is additive: contact count matches the distinct-tap count for a
    non-consumed zone (each run = one contact)."""
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(8)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[3].update(low=97.4)          # entry (run 1)
    rows[5].update(low=97.4)          # entry (run 2, after leaving)
    z = collect_zones(_frame(rows), idx=7)
    ob = z["order_blocks"][0]
    assert ob["touch_count"] == 2
    assert len(ob["contacts"]) == 2
    assert {c["outcome"] for c in ob["contacts"]} == {"entry_exit"}


# --------------------------------------------------------------------------- #
# Consumed zones surfaced in a separate, bounded list (« Comblées » group)
# --------------------------------------------------------------------------- #
def test_invalidated_ob_leaves_live_list_but_enters_consumed():
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[4].update(high=98, low=95, close=96.0)  # closes through → invalidated
    z = collect_zones(_frame(rows), idx=5)
    assert z["order_blocks"] == []                      # never in the live list
    assert len(z["consumed_order_blocks"]) == 1
    consumed = z["consumed_order_blocks"][0]
    assert consumed["status"] == "invalidated"
    assert consumed["contacts"][-1]["outcome"] == "traversal"


def test_filled_fvg_enters_consumed_list_with_traversal():
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    # Bullish gap at k=2: band [high[0]=100, low[2]=101] = [100,101]; full fill <=100.
    rows[2].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.5)
    rows[5].update(low=99.5)  # retraces past the far edge (100) → filled
    z = collect_zones(_frame(rows), idx=7)
    assert z["fair_value_gaps"] == []
    assert len(z["consumed_fair_value_gaps"]) == 1
    assert z["consumed_fair_value_gaps"][0]["contacts"][-1]["outcome"] == "traversal"


def test_consumed_list_is_bounded(monkeypatch):
    monkeypatch.setenv("MAX_CONSUMED_ZONES_PER_TYPE", "2")
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(14)]
    # Five bullish OB each invalidated by a later close-through.
    for i, k in enumerate([1, 3, 5, 7, 9]):
        rows[k].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
        rows[k + 1].update(low=95, close=96.0)  # closes through
    z = collect_zones(_frame(rows), idx=13)
    assert len(z["consumed_order_blocks"]) == 2  # capped


# --------------------------------------------------------------------------- #
# Origin association (BOS / CHOCH the OB precedes)
# --------------------------------------------------------------------------- #
def test_ob_origin_links_the_following_same_direction_break():
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[2].update(BOS_EVENT=1.0, BOS_BREAK_LEVEL=100.5)  # bullish break just after
    df = _frame(rows)
    origin = _ob_origin(df, "bullish", created_k=1, upto=5)
    assert origin is not None
    assert origin["kind"] == "bos" and origin["direction"] == "bullish"
    assert origin["level"] == 100.5


def test_ob_origin_marks_choch_when_the_break_bar_is_a_reversal():
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[2].update(BOS_EVENT=1.0, CHOCH_SIGNAL=1.0, BOS_BREAK_LEVEL=100.5)
    origin = _ob_origin(_frame(rows), "bullish", 1, 5)
    assert origin is not None and origin["kind"] == "choch"


def test_ob_origin_absent_without_a_matching_break():
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    # A bearish break (opposite direction) must not be associated.
    rows[2].update(BOS_EVENT=-1.0, BOS_BREAK_LEVEL=96.0)
    assert _ob_origin(_frame(rows), "bullish", 1, 5) is None


def test_fvg_contacts_bearish_entry_and_traversal():
    """Bearish gap fills upward from the bottom edge."""
    zhigh, zlow = 101.0, 100.0  # height 1.0
    highs = [95, 100.5, 95, 101.5]  # entry@1 (pen .5), fill@3 (>=101)
    lows = [94, 94, 94, 94]
    contacts = _fvg_contacts("bearish", zhigh, zlow, highs, lows, created=0, upto=3)
    assert contacts[0]["outcome"] == "entry_exit"
    assert contacts[-1]["outcome"] == "traversal"
