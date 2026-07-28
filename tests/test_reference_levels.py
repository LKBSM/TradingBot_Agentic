"""RG-1c — calendar reference levels aggregated over the MC-1 trading calendar.

These tests pin the exact bug the mission targets: a single candle's high/low is
never a « day » — every level must AGGREGATE all candles of its MC-1 trading
period, and an incompletely-covered period must be dropped, not shown partial.
All data is frozen (no market, no network).

Timezone note: the trading day rolls over at 17:00 New York = 21:00 UTC in July
(EDT = UTC−4). A UTC timestamp in [04:00, 21:00) maps to the same calendar date's
trading day; [21:00, 04:00) rolls into the NEXT trading day. Fixtures place
in-day bars at 06:00–20:00 UTC and use 22:00 UTC only to cross the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from src.intelligence.reference_levels import compute_reference_levels


@dataclass(frozen=True)
class C:
    """A minimal candle stand-in (ts UTC + OHLC)."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def _utc(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=timezone.utc)


NOW = _utc(2026, 7, 25, 20)  # inside the Jul 25 (Sat-labelled? no) trading day


def test_prev_day_high_is_the_max_over_all_candles_of_the_day():
    candles = [
        C(_utc(2026, 7, 23, 12), 4100, 4100, 4100, 4100),  # anchor → day Jul 23
        # previous trading day (Jul 24), three bars — extremes span the whole day
        C(_utc(2026, 7, 24, 6), 4185, 4205, 4182, 4200),  # day high 4205
        C(_utc(2026, 7, 24, 12), 4200, 4203, 4150, 4160),  # day low 4150
        C(_utc(2026, 7, 24, 18), 4160, 4172, 4158, 4170),
        # current trading day (Jul 25)
        C(_utc(2026, 7, 25, 6), 4170, 4176, 4168, 4174),  # first bar → day open
        C(_utc(2026, 7, 25, 12), 4174, 4188, 4171, 4185),
    ]
    r = compute_reference_levels("XAUUSD", candles, NOW)
    assert r.prev_day_high == 4205  # max of the highs of ALL of Jul 24's bars
    assert r.prev_day_low == 4150  # min of the lows
    assert r.prev_day_high - r.prev_day_low > 20  # a real gold day, never ~2.77
    assert r.day_open == 4170  # open of the FIRST candle of the current day
    assert r.day_complete is True


def test_high_and_low_come_from_different_candles():
    candles = [
        C(_utc(2026, 7, 23, 12), 4100, 4100, 4100, 4100),  # anchor
        C(_utc(2026, 7, 24, 6), 4180, 4210, 4179, 4200),  # holds the high
        C(_utc(2026, 7, 24, 12), 4200, 4201, 4150, 4155),  # holds the low
        C(_utc(2026, 7, 25, 6), 4155, 4160, 4150, 4158),  # current day
    ]
    r = compute_reference_levels("XAUUSD", candles, NOW)
    assert r.prev_day_high == 4210  # from bar 1
    assert r.prev_day_low == 4150  # from bar 2 — a different candle


def test_single_bar_day_may_share_high_and_low_source():
    # Legal degenerate: a trading day with exactly one candle — high and low may
    # come from the same bar.
    candles = [
        C(_utc(2026, 7, 23, 12), 4100, 4100, 4100, 4100),  # anchor
        C(_utc(2026, 7, 24, 12), 4180, 4210, 4150, 4200),  # the only bar of Jul 24
        C(_utc(2026, 7, 25, 6), 4200, 4205, 4198, 4203),  # current day
    ]
    r = compute_reference_levels("XAUUSD", candles, NOW)
    assert r.prev_day_high == 4210
    assert r.prev_day_low == 4150


def test_prev_week_high_at_least_prev_day_high_when_day_in_that_week():
    # Coherence check that WOULD have caught the bug: the weekly extreme aggregates
    # a superset of the daily extreme, so it can never be smaller. One bar/day at
    # 12:00 UTC (→ trading day == UTC date). high = 4200 + day-of-month.
    candles = [C(_utc(2026, 7, 10, 12), 4000, 4000, 4000, 4000)]  # anchor (week before)
    for d in (13, 14, 15, 16, 17):  # previous full week (Mon..Fri)
        candles.append(C(_utc(2026, 7, d, 12), 4100 + d, 4200 + d, 4080 + d, 4150 + d))
    candles.append(C(_utc(2026, 7, 20, 12), 4300, 4305, 4295, 4302))  # current week
    r = compute_reference_levels("XAUUSD", candles, _utc(2026, 7, 20, 20))
    assert r.prev_week_high is not None and r.prev_day_high is not None
    assert r.prev_week_high >= r.prev_day_high  # 4217 >= 4217 (Fri Jul 17)
    assert r.week_complete is True


def test_incomplete_period_is_dropped_not_shown_partial():
    # Oldest candle lands MID-way through the previous trading day → that day is
    # truncated by the lookback, so its extremes are dropped, never partial.
    candles = [
        C(_utc(2026, 7, 24, 12), 4200, 4203, 4150, 4160),  # first bar, mid-Jul-24
        C(_utc(2026, 7, 24, 18), 4160, 4172, 4158, 4170),
        C(_utc(2026, 7, 25, 6), 4170, 4176, 4168, 4174),  # current day
    ]
    r = compute_reference_levels("XAUUSD", candles, NOW)
    assert r.prev_day_high is None
    assert r.prev_day_low is None
    assert r.day_complete is False
    assert r.day_open == 4170  # the current day open is still known


def test_empty_series_yields_all_none():
    r = compute_reference_levels("XAUUSD", [], NOW)
    assert r.to_dict() == {
        "day_open": None,
        "week_open": None,
        "prev_day_high": None,
        "prev_day_low": None,
        "prev_week_high": None,
        "prev_week_low": None,
        "day_complete": False,
        "week_complete": False,
    }


def test_eurusd_day_boundary_is_1700_ny_not_utc_midnight():
    # EURUSD rolls at 17:00 NY (21:00 UTC in July). A 20:00 UTC bar is STILL the
    # same trading day; a 22:00 UTC bar is the NEXT one.
    candles = [
        C(_utc(2026, 7, 23, 12), 1.10, 1.10, 1.10, 1.10),  # anchor → Jul 23
        C(_utc(2026, 7, 24, 8), 1.170, 1.175, 1.168, 1.172),  # Jul 24
        C(_utc(2026, 7, 24, 20), 1.172, 1.180, 1.171, 1.179),  # STILL Jul 24 (< 21:00Z)
        C(_utc(2026, 7, 24, 22), 1.179, 1.181, 1.178, 1.180),  # → Jul 25 (rolled over)
    ]
    r = compute_reference_levels("EURUSD", candles, _utc(2026, 7, 25, 12))
    assert r.prev_day_high == 1.180  # max(1.175, 1.180) over BOTH Jul-24 bars
    assert r.prev_day_low == 1.168
    assert r.day_open == 1.179  # first bar after the 21:00Z rollover
