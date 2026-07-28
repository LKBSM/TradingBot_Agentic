"""Calendar reference levels — day/week open + previous day/week extremes.

The six repères of the Régime « Niveaux de référence » tile are ARITHMETIC over
the raw candles, not new detections: each one AGGREGATES every candle of its
period (``prev_day_high`` = max of the highs of ALL candles of the previous
trading day; ``prev_day_low`` = the min of the lows; idem for the week), and the
day/week boundary is the instrument's own MC-1 trading calendar, in the
instrument tz wall-clock (``InstrumentHours.tz`` — America/New_York) with the
weekly rollover at ``close_time`` (17:00 NY for FX, the metal rollover for gold).

This is the ONE definition of « a trading day » in the product: the same
per-instrument ZoneInfo config that drives market status (MC-1). No candle high/
low is ever read in isolation — reading a single bar's high/low is exactly the
bug this module replaces (RG-1c): a single sub-daily candle's range is a couple
of dollars, never a real gold day.

Completeness guard: a level is emitted ONLY when the candles on hand fully cover
its period. If the oldest candle we have starts *after* a period began, that
period is truncated by the lookback window and its max/min would be partial —
so the level is dropped (``None``) and the caller renders « données insuffisantes
pour cette période » rather than a partial value presented as complete.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional, Sequence
from zoneinfo import ZoneInfo

from src.intelligence.market_calendar import hours_for


@dataclass(frozen=True)
class ReferenceLevels:
    """The six calendar levels, each ``None`` when its period is not fully
    covered by the candles on hand. ``day_complete`` / ``week_complete`` say
    whether the previous-day / previous-week window was fully covered, so the
    panel can name « données insuffisantes » for the right period."""

    day_open: Optional[float]
    week_open: Optional[float]
    prev_day_high: Optional[float]
    prev_day_low: Optional[float]
    prev_week_high: Optional[float]
    prev_week_low: Optional[float]
    day_complete: bool
    week_complete: bool

    def to_dict(self) -> dict:
        return {
            "day_open": self.day_open,
            "week_open": self.week_open,
            "prev_day_high": self.prev_day_high,
            "prev_day_low": self.prev_day_low,
            "prev_week_high": self.prev_week_high,
            "prev_week_low": self.prev_week_low,
            "day_complete": self.day_complete,
            "week_complete": self.week_complete,
        }


_EMPTY = ReferenceLevels(None, None, None, None, None, None, False, False)


def _as_utc(ts: datetime) -> datetime:
    """Naive timestamps are UTC (the candles-cache convention)."""
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def compute_reference_levels(
    instrument: str,
    candles: Sequence[Any],
    now: datetime,  # kept for signature symmetry / future "current period" use
) -> ReferenceLevels:
    """Aggregate the six calendar levels from ``candles`` (ascending, oldest
    first — as the candles store returns them). ``candles`` should be an intraday
    series long enough to cover the previous week (H1/H4). Every level is a pure
    max/min (or first-open) over a full MC-1 trading period, or ``None`` when the
    period is not fully covered.
    """
    if not candles:
        return _EMPTY

    hours = hours_for(instrument)
    tz = ZoneInfo(hours.tz)
    close_time = hours.close_time

    def trading_day(ts_utc: datetime) -> date:
        """The trading-day label a UTC timestamp belongs to. The day rolls over
        at ``close_time`` in the instrument tz: a bar at/after 17:00 NY belongs to
        the NEXT trading day (its session opened this evening)."""
        local = ts_utc.astimezone(tz)
        d = local.date()
        if local.time() >= close_time:
            d = d + timedelta(days=1)
        return d

    def day_start_utc(d: date) -> datetime:
        """UTC instant the trading day ``d`` begins — the previous calendar day at
        ``close_time`` in the instrument tz (the weekly/daily rollover boundary)."""
        naive = datetime.combine(d - timedelta(days=1), close_time)
        return naive.replace(tzinfo=tz).astimezone(timezone.utc)

    def week_monday(d: date) -> date:
        """Monday of the trading week containing trading-day ``d`` (labels are
        Mon–Fri, so Monday = d − weekday)."""
        return d - timedelta(days=d.weekday())

    oldest = _as_utc(candles[0].ts)

    # Bucket candles by trading day and by trading week (Monday label).
    days: dict[date, list[Any]] = {}
    weeks: dict[date, list[Any]] = {}
    for c in candles:
        d = trading_day(_as_utc(c.ts))
        days.setdefault(d, []).append(c)
        weeks.setdefault(week_monday(d), []).append(c)

    day_open = _current_open(days)
    week_open = _current_open(weeks)

    prev_day_high, prev_day_low, day_complete = _prev_extremes(days, oldest, day_start_utc)
    prev_week_high, prev_week_low, week_complete = _prev_extremes(
        weeks, oldest, day_start_utc  # week start = its Monday's day_start
    )

    return ReferenceLevels(
        day_open=day_open,
        week_open=week_open,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
        prev_week_high=prev_week_high,
        prev_week_low=prev_week_low,
        day_complete=day_complete,
        week_complete=week_complete,
    )


def _current_open(buckets: dict[date, list[Any]]) -> Optional[float]:
    """Open of the first candle of the most recent (current, forming) period."""
    if not buckets:
        return None
    latest = max(buckets)
    bars = buckets[latest]
    first = min(bars, key=lambda c: _as_utc(c.ts))
    return float(first.open)


def _prev_extremes(
    buckets: dict[date, list[Any]],
    oldest: datetime,
    period_start_utc,
) -> tuple[Optional[float], Optional[float], bool]:
    """Max-high / min-low over ALL candles of the PREVIOUS complete period.

    Returns ``(None, None, False)`` when there is no previous period or it is not
    fully covered (the oldest candle we have starts after the period began, so
    its extremes would be partial)."""
    if len(buckets) < 2:
        return None, None, False
    labels = sorted(buckets)
    prev = labels[-2]
    # Fully covered iff we hold a candle from before this period started.
    if oldest > period_start_utc(prev):
        return None, None, False
    bars = buckets[prev]
    high = max(float(c.high) for c in bars)
    low = min(float(c.low) for c in bars)
    return high, low, True


__all__ = ["ReferenceLevels", "compute_reference_levels"]
