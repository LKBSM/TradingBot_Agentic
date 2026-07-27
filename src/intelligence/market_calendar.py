"""Market calendar — THE single server-side source of truth for whether a
(instrument, timeframe) combination is currently tradeable, and for the
"market-aware" last-candle-close used to gate regeneration.

Mission MC-1. Design principle: **the fact first, the calendar second.**

  * The FACT — always true, calendar-independent — is the age of the last
    fully-closed candle. It alone decides "nothing new has happened".
  * The CALENDAR only *names* the reason (weekend / holiday / daily break) and
    supplies the reopen time. A calendar is never the sole authority: an
    unforeseen holiday must degrade to the age-based fact, never to a false
    "open".
  * If the two contradict — calendar says OPEN but no candle has closed for a
    long time — that is a DATA feed problem and is surfaced honestly as
    ``data_lagged`` rather than hidden.

Timezone handling is DST-safe: every wall-clock rule is expressed in an IANA
zone (``America/New_York``) and resolved by :mod:`zoneinfo`. There is **no**
hard-coded UTC offset — 17:00 New York is 21:00 UTC in summer and 22:00 UTC in
winter, and the library handles the switch.

Trading hours live in :data:`INSTRUMENT_HOURS` **per instrument** (never a
global constant), because spot FX and spot metals differ: EURUSD opens Sunday
17:00 NY with no daily break, XAUUSD opens Sunday 18:00 NY with a 17:00–18:00
daily rollover pause. Adding a market = adding one entry here.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# States
# --------------------------------------------------------------------------- #
OPEN = "open"
CLOSED_WEEKEND = "closed_weekend"
CLOSED_HOLIDAY = "closed_holiday"
DAILY_BREAK = "daily_break"
DATA_LAGGED = "data_lagged"

#: How many timeframe intervals the last closed candle may lag, while the
#: calendar says OPEN, before we call it ``data_lagged``. Aligned with the
#: Conditions Scanner "stale" tier (``>= 5`` bars behind) so the two surfaces
#: agree on what "the feed stopped" means.
DATA_LAGGED_MIN_BARS_BEHIND = 5


# --------------------------------------------------------------------------- #
# Intraday sessions (reading convention, single source — here)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionWindow:
    """A named intraday trading session, in the instrument's ``tz`` wall-clock.

    ``end`` may be earlier than ``start`` when the window WRAPS past midnight
    (the Asian session). Sessions are a **reading convention** — how traders
    segment the global FX/metal day — not a market property; the concept text
    says as much. They live HERE so there is a single source of session hours,
    never a second one on the client.
    """

    name: str            # 'asia' | 'london' | 'new_york'
    start: time
    end: time


# Global FX/metal session convention, America/New_York wall-clock. Anchored on
# the two ranges the design reference pins — New York 08:00–17:00 and the
# London∩NY overlap 08:00–11:30 (⇒ London closes 11:30) — with the standard ICT
# windows consistent with that overlap. Source: common forex session hours in
# Eastern Time (Tokyo 19:00–04:00, London 03:00–11:30, New York 08:00–17:00).
# The London/NY overlap is DERIVED (intersection), never stored twice.
_STANDARD_SESSIONS: tuple[SessionWindow, ...] = (
    SessionWindow("asia", time(19, 0), time(4, 0)),      # wraps past midnight
    SessionWindow("london", time(3, 0), time(11, 30)),
    SessionWindow("new_york", time(8, 0), time(17, 0)),
)


# --------------------------------------------------------------------------- #
# Per-instrument trading hours
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InstrumentHours:
    """Weekly trading window + optional daily break, all in ``tz`` wall-clock.

    ``open_weekday`` / ``close_weekday`` follow Python's convention
    (Mon=0 … Sun=6). The window runs from ``open_time`` on ``open_weekday`` to
    ``close_time`` on ``close_weekday`` (Sunday → Friday for FX/metals).
    ``daily_break`` (start, end) applies Monday–Thursday only; on Friday the
    weekly close pre-empts it and on Sunday the later open pre-empts it.
    ``always_open`` short-circuits everything for 24/7 markets (crypto).
    ``sessions`` = the intraday session windows (empty for a continuous market).
    """

    tz: str = "America/New_York"
    open_weekday: int = 6            # Sunday
    open_time: time = time(17, 0)
    close_weekday: int = 4           # Friday
    close_time: time = time(17, 0)
    daily_break: Optional[tuple[time, time]] = None
    always_open: bool = False
    sessions: tuple[SessionWindow, ...] = _STANDARD_SESSIONS


# Forex convention: Sunday 17:00 NY → Friday 17:00 NY, no daily break.
_FOREX_HOURS = InstrumentHours()
# Spot-metal convention: Sunday 18:00 NY open, Friday 17:00 NY close, daily
# rollover pause 17:00–18:00 NY.
_METAL_HOURS = InstrumentHours(open_time=time(18, 0), daily_break=(time(17, 0), time(18, 0)))
# Crypto: never closes — and has no session découpage (continuous market).
_CRYPTO_HOURS = InstrumentHours(always_open=True, sessions=())

INSTRUMENT_HOURS: dict[str, InstrumentHours] = {
    "XAUUSD": _METAL_HOURS,
    "XAGUSD": _METAL_HOURS,
    "EURUSD": _FOREX_HOURS,
    "GBPUSD": _FOREX_HOURS,
    "USDJPY": _FOREX_HOURS,
    "BTCUSD": _CRYPTO_HOURS,
    "ETHUSD": _CRYPTO_HOURS,
}

# Mirror of the frontend heuristic (webapp/lib/market-reading/session.ts) so a
# symbol we have not explicitly listed but that is obviously crypto is never
# reported as closed.
_CRYPTO_RE = re.compile(r"BTC|ETH|USDT|USDC|crypto", re.IGNORECASE)


def hours_for(instrument: str) -> InstrumentHours:
    """Trading hours for ``instrument`` — explicit entry, else crypto-by-name,
    else the forex default (spot FX week). The forex default is deliberately
    conservative: an unknown symbol is treated as closed on the weekend rather
    than assumed 24/7, and the last-candle-age fact still governs regardless."""
    key = (instrument or "").upper()
    if key in INSTRUMENT_HOURS:
        return INSTRUMENT_HOURS[key]
    if _CRYPTO_RE.search(key):
        return _CRYPTO_HOURS
    return _FOREX_HOURS


def sessions_for(instrument: str) -> tuple[SessionWindow, ...]:
    """Intraday session windows for ``instrument`` (empty for a continuous
    market). Read from the SAME per-instrument config as the trading hours —
    never a second source."""
    return hours_for(instrument).sessions


# --------------------------------------------------------------------------- #
# Holidays (versioned config, age-based fallback beyond coverage)
# --------------------------------------------------------------------------- #
def _default_holidays_path() -> Path:
    env = os.environ.get("SENTINEL_MARKET_HOLIDAYS_PATH")
    if env:
        return Path(env)
    # src/intelligence/market_calendar.py → repo root is two parents up.
    return Path(__file__).resolve().parents[2] / "config" / "market_holidays.json"


class _HolidayCalendar:
    """Loaded once, cached. Beyond ``covered_through_year`` every lookup returns
    False so the caller falls back to the last-candle-age fact — an unknown
    holiday must surface as ``data_lagged``, never as a false ``open``."""

    def __init__(self, path: Path) -> None:
        self._covered_through_year = 0
        self._by_date: dict[date, set[str]] = {}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            self._covered_through_year = int(raw.get("covered_through_year", 0))
            for entry in raw.get("holidays", []):
                d = date.fromisoformat(str(entry["date"]))
                markets = {str(m).upper() for m in entry.get("markets", ["*"])}
                self._by_date.setdefault(d, set()).update(markets)
        except FileNotFoundError:
            logger.warning("market holidays file not found at %s — age-based fallback only", path)
        except Exception:  # pragma: no cover — malformed file must not crash the engine
            logger.exception("failed to load market holidays from %s — age-based fallback only", path)

    def is_holiday(self, instrument: str, day: date) -> bool:
        # Beyond the curated range we do NOT know — defer to the age-based fact.
        if day.year > self._covered_through_year:
            return False
        markets = self._by_date.get(day)
        if not markets:
            return False
        return "*" in markets or (instrument or "").upper() in markets


_holiday_calendar: Optional[_HolidayCalendar] = None


def _holidays() -> _HolidayCalendar:
    global _holiday_calendar
    if _holiday_calendar is None:
        _holiday_calendar = _HolidayCalendar(_default_holidays_path())
    return _holiday_calendar


def reset_holiday_cache() -> None:
    """Force a reload of the holiday file (tests that point at a fixture)."""
    global _holiday_calendar
    _holiday_calendar = None


# --------------------------------------------------------------------------- #
# Calendar state (pure calendar, no data)
# --------------------------------------------------------------------------- #
def _in_weekly_session(hours: InstrumentHours, wd: int, t: time) -> bool:
    """True when the weekly window is open at weekday ``wd`` / local time ``t``
    (Sunday open → Friday close). Ignores holidays and the daily break."""
    if wd == 5:  # Saturday — always closed
        return False
    if wd == hours.open_weekday:      # Sunday — open from open_time
        return t >= hours.open_time
    if wd == hours.close_weekday:     # Friday — open until close_time
        return t < hours.close_time
    return True                       # Monday–Thursday — open all day


def calendar_state(instrument: str, now: datetime) -> str:
    """Pure-calendar state ∈ {OPEN, CLOSED_WEEKEND, CLOSED_HOLIDAY, DAILY_BREAK}.

    Does not consider candle freshness — that is layered on top by
    :func:`compute_market_status` (which can override OPEN with DATA_LAGGED)."""
    hours = hours_for(instrument)
    if hours.always_open:
        return OPEN
    ny = now.astimezone(ZoneInfo(hours.tz))
    if not _in_weekly_session(hours, ny.weekday(), ny.time()):
        return CLOSED_WEEKEND
    if _holidays().is_holiday(instrument, ny.date()):
        return CLOSED_HOLIDAY
    if hours.daily_break and ny.weekday() in (0, 1, 2, 3):
        start, end = hours.daily_break
        if start <= ny.time() < end:
            return DAILY_BREAK
    return OPEN


# --------------------------------------------------------------------------- #
# Candle-boundary helpers (fact side)
# --------------------------------------------------------------------------- #
def _timeframe_minutes(timeframe: str) -> int:
    # Local import avoids a circular import with market_reading_assembler, which
    # imports market_aware_expected_close from this module.
    from src.intelligence.market_reading_assembler import _TIMEFRAME_MINUTES
    tf = timeframe.upper()
    if tf not in _TIMEFRAME_MINUTES:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}")
    return _TIMEFRAME_MINUTES[tf]


def _expected_last_candle_close(timeframe: str, now: datetime) -> datetime:
    from src.intelligence.market_reading_assembler import expected_last_candle_close
    return expected_last_candle_close(timeframe, now)


def market_aware_expected_close(instrument: str, timeframe: str, now: datetime) -> datetime:
    """Close timestamp of the last candle that actually **traded** at or before
    ``now`` — the clock boundary when the market is open, frozen at the last
    pre-closure boundary when it is not.

    This is the value the assembler and scheduler compare against to decide
    whether to regenerate. Because it stops advancing the moment the market
    closes, a stored Friday reading keeps matching all weekend: no rebuild, no
    Twelve Data call, no re-emitted "new" structure. And because the frozen
    value equals the real last traded candle, stamping it as ``candle_close_ts``
    stays honest (never a weekend timestamp on Friday data).
    """
    span = timedelta(minutes=_timeframe_minutes(timeframe))
    boundary = _expected_last_candle_close(timeframe, now)
    hours = hours_for(instrument)
    if hours.always_open:
        return boundary
    # Walk back one candle at a time to the last boundary whose candle traded
    # (market open one second before it closed). Bounded well above any real
    # weekend/holiday gap.
    for _ in range(20000):
        if calendar_state(instrument, boundary - timedelta(seconds=1)) == OPEN:
            return boundary
        boundary -= span
    return boundary  # pragma: no cover — defensive floor


def next_open(instrument: str, now: datetime) -> Optional[datetime]:
    """Next instant (UTC) the market opens, scanning forward up to 8 days.
    ``None`` for a 24/7 market (always open)."""
    hours = hours_for(instrument)
    if hours.always_open:
        return None
    probe = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    step = timedelta(minutes=1)
    for _ in range(8 * 24 * 60):
        probe += step
        if calendar_state(instrument, probe) == OPEN:
            return probe
    return None  # pragma: no cover — defensive


# --------------------------------------------------------------------------- #
# Market status (fact-first, calendar-named)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MarketStatus:
    """What every surface (App badge, Scanner, Zones, agent) reads. The state is
    fact-first; ``reason`` mirrors it for machine use; timestamps are facts (a
    past close, a published reopen time)."""

    state: str
    instrument: str
    timeframe: str
    last_close_ts: Optional[datetime]
    next_open_ts: Optional[datetime]
    bars_behind: Optional[int]

    @property
    def is_tradeable(self) -> bool:
        return self.state == OPEN

    def to_dict(self) -> dict:
        def _iso(dt: Optional[datetime]) -> Optional[str]:
            if dt is None:
                return None
            return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        hours = hours_for(self.instrument)
        return {
            "state": self.state,
            "reason": self.state,
            "instrument": self.instrument,
            "timeframe": self.timeframe,
            "last_close_ts": _iso(self.last_close_ts),
            "next_open_ts": _iso(self.next_open_ts),
            "bars_behind": self.bars_behind,
            # Session découpage (single source: this module's per-instrument
            # config). The client computes the CURRENT session / next transition
            # / overlap state live from these windows + the reader's clock — it
            # never holds its own hours. Empty `sessions` ⇒ continuous market.
            "continuous": hours.always_open,
            "session_tz": hours.tz,
            "sessions": [
                {"name": s.name, "start": s.start.strftime("%H:%M"), "end": s.end.strftime("%H:%M")}
                for s in hours.sessions
            ],
            "weekly_close": {
                "weekday": hours.close_weekday,
                "time": hours.close_time.strftime("%H:%M"),
            },
        }


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def compute_market_status(
    instrument: str,
    timeframe: str,
    last_close_ts: Optional[datetime],
    now: datetime,
) -> MarketStatus:
    """Combine the fact (age of ``last_close_ts``) with the calendar.

    ``last_close_ts`` is the ``candle_close_ts`` of the latest stored reading —
    the real last closed candle. When the calendar says OPEN but that candle is
    ``DATA_LAGGED_MIN_BARS_BEHIND`` intervals or more behind the expected close,
    the feed has stalled and we report ``data_lagged`` (the fact wins). When the
    calendar says closed, we report the calendar reason plus the reopen time.
    """
    last_close_ts = _to_utc(last_close_ts)
    cal = calendar_state(instrument, now)

    if cal == OPEN:
        expected = market_aware_expected_close(instrument, timeframe, now)
        bars_behind = None
        if last_close_ts is not None:
            span_min = _timeframe_minutes(timeframe)
            delta_min = (expected - last_close_ts).total_seconds() / 60.0
            bars_behind = max(0, int(round(delta_min / span_min)))
        lagged = last_close_ts is None or (
            bars_behind is not None and bars_behind >= DATA_LAGGED_MIN_BARS_BEHIND
        )
        if lagged:
            return MarketStatus(
                state=DATA_LAGGED,
                instrument=instrument,
                timeframe=timeframe,
                last_close_ts=last_close_ts,
                next_open_ts=None,
                bars_behind=bars_behind,
            )
        return MarketStatus(
            state=OPEN,
            instrument=instrument,
            timeframe=timeframe,
            last_close_ts=last_close_ts,
            next_open_ts=None,
            bars_behind=bars_behind,
        )

    return MarketStatus(
        state=cal,
        instrument=instrument,
        timeframe=timeframe,
        last_close_ts=last_close_ts,
        next_open_ts=next_open(instrument, now),
        bars_behind=None,
    )


__all__ = [
    "OPEN",
    "CLOSED_WEEKEND",
    "CLOSED_HOLIDAY",
    "DAILY_BREAK",
    "DATA_LAGGED",
    "DATA_LAGGED_MIN_BARS_BEHIND",
    "InstrumentHours",
    "INSTRUMENT_HOURS",
    "MarketStatus",
    "SessionWindow",
    "calendar_state",
    "compute_market_status",
    "hours_for",
    "sessions_for",
    "market_aware_expected_close",
    "next_open",
    "reset_holiday_cache",
]
