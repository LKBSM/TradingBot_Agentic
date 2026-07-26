"""MC-1 — market calendar source of truth (frozen-clock, never the system clock).

Covers the mission's required scenarios: weekend / Friday / Sunday boundaries,
DST summer vs winter switch, holiday closure, the daily break (XAU only),
per-instrument reopen times, the market-aware close freeze, and the fact-first
``data_lagged`` state when the calendar says open but no candle has closed.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from src.intelligence import market_calendar as mc

NY = ZoneInfo("America/New_York")


def _ny(y, mo, d, h, mi=0):
    return datetime(y, mo, d, h, mi, tzinfo=NY)


# --------------------------------------------------------------------------- #
# Weekend / session boundaries (mission scenarios)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "label,instrument,dt,expected",
    [
        ("sat noon", "EURUSD", _ny(2026, 7, 25, 12, 0), mc.CLOSED_WEEKEND),
        ("fri 16:59", "EURUSD", _ny(2026, 7, 24, 16, 59), mc.OPEN),
        ("fri 17:01", "EURUSD", _ny(2026, 7, 24, 17, 1), mc.CLOSED_WEEKEND),
        ("sun 17:59 EUR", "EURUSD", _ny(2026, 7, 26, 16, 59), mc.CLOSED_WEEKEND),
        ("sun 17:01 EUR", "EURUSD", _ny(2026, 7, 26, 17, 1), mc.OPEN),
        # XAU opens Sunday 18:00, not 17:00.
        ("sun 17:30 XAU", "XAUUSD", _ny(2026, 7, 26, 17, 30), mc.CLOSED_WEEKEND),
        ("sun 18:01 XAU", "XAUUSD", _ny(2026, 7, 26, 18, 1), mc.OPEN),
    ],
)
def test_weekly_session_boundaries(label, instrument, dt, expected):
    assert mc.calendar_state(instrument, dt) == expected, label


def test_xau_daily_break_but_not_eurusd():
    # Wednesday 17:30 NY: XAU is in its rollover pause, EURUSD keeps trading.
    assert mc.calendar_state("XAUUSD", _ny(2026, 7, 22, 17, 30)) == mc.DAILY_BREAK
    assert mc.calendar_state("XAUUSD", _ny(2026, 7, 22, 18, 1)) == mc.OPEN
    assert mc.calendar_state("EURUSD", _ny(2026, 7, 22, 17, 30)) == mc.OPEN


def test_crypto_never_closes():
    assert mc.calendar_state("BTCUSD", _ny(2026, 7, 25, 12, 0)) == mc.OPEN
    assert mc.next_open("BTCUSD", _ny(2026, 7, 25, 12, 0)) is None


# --------------------------------------------------------------------------- #
# DST: the 17:00 New-York boundary is 21:00 UTC in summer, 22:00 UTC in winter
# --------------------------------------------------------------------------- #
def test_dst_summer_boundary_is_21_utc():
    # 2026-07-26 is EDT (UTC-4): Sunday 17:00 NY == 21:00 UTC.
    assert mc.calendar_state("EURUSD", datetime(2026, 7, 26, 20, 59, tzinfo=timezone.utc)) == mc.CLOSED_WEEKEND
    assert mc.calendar_state("EURUSD", datetime(2026, 7, 26, 21, 1, tzinfo=timezone.utc)) == mc.OPEN


def test_dst_winter_boundary_is_22_utc():
    # 2026-01-25 is EST (UTC-5): Sunday 17:00 NY == 22:00 UTC.
    assert mc.calendar_state("EURUSD", datetime(2026, 1, 25, 21, 59, tzinfo=timezone.utc)) == mc.CLOSED_WEEKEND
    assert mc.calendar_state("EURUSD", datetime(2026, 1, 25, 22, 1, tzinfo=timezone.utc)) == mc.OPEN


# --------------------------------------------------------------------------- #
# Holidays (config-driven, age-based fallback beyond coverage)
# --------------------------------------------------------------------------- #
def test_configured_holiday_is_closed():
    assert mc.calendar_state("XAUUSD", _ny(2026, 12, 25, 12, 0)) == mc.CLOSED_HOLIDAY
    assert mc.calendar_state("EURUSD", _ny(2026, 1, 1, 12, 0)) == mc.CLOSED_HOLIDAY


def test_unknown_future_holiday_does_not_assume_open():
    # A weekday far beyond the curated calendar must NOT be reported closed_holiday
    # (we don't know) — it stays OPEN so the last-candle-age fact can flag lag.
    far = _ny(2035, 5, 30, 12, 0)  # a Wednesday well past covered_through_year
    assert mc.calendar_state("EURUSD", far) == mc.OPEN


# --------------------------------------------------------------------------- #
# Reopen time is per-instrument
# --------------------------------------------------------------------------- #
def test_next_open_is_per_instrument():
    sat = _ny(2026, 7, 25, 12, 0)
    eur_open = mc.next_open("EURUSD", sat).astimezone(NY)
    xau_open = mc.next_open("XAUUSD", sat).astimezone(NY)
    assert (eur_open.weekday(), eur_open.hour) == (6, 17)  # Sunday 17:00 NY
    assert (xau_open.weekday(), xau_open.hour) == (6, 18)  # Sunday 18:00 NY


# --------------------------------------------------------------------------- #
# Market-aware close freezes during closure (the emission-lock primitive)
# --------------------------------------------------------------------------- #
def test_market_aware_close_freezes_all_weekend():
    fri_close = _ny(2026, 7, 24, 17, 0).astimezone(timezone.utc)
    for now in (_ny(2026, 7, 24, 17, 30), _ny(2026, 7, 25, 12, 0), _ny(2026, 7, 26, 12, 0)):
        assert mc.market_aware_expected_close("EURUSD", "M15", now) == fri_close


def test_market_aware_close_equals_clock_when_open():
    from src.intelligence.market_reading_assembler import expected_last_candle_close

    now = _ny(2026, 7, 22, 10, 7)  # Wednesday, market open
    assert mc.market_aware_expected_close("EURUSD", "M15", now) == expected_last_candle_close("M15", now)


def test_market_aware_close_freezes_during_daily_break():
    xau_break = _ny(2026, 7, 22, 17, 0).astimezone(timezone.utc)
    assert mc.market_aware_expected_close("XAUUSD", "M15", _ny(2026, 7, 22, 17, 45)) == xau_break


# --------------------------------------------------------------------------- #
# compute_market_status — fact-first
# --------------------------------------------------------------------------- #
def test_status_open_when_fresh():
    now = _ny(2026, 7, 22, 10, 0)
    mac = mc.market_aware_expected_close("EURUSD", "M15", now)
    st = mc.compute_market_status("EURUSD", "M15", mac, now)
    assert st.state == mc.OPEN
    assert st.bars_behind == 0
    assert st.next_open_ts is None


def test_status_data_lagged_when_calendar_open_but_feed_stale():
    now = _ny(2026, 7, 22, 10, 0)
    mac = mc.market_aware_expected_close("EURUSD", "M15", now)
    stale = mac - timedelta(hours=3)  # 12 M15 candles behind
    st = mc.compute_market_status("EURUSD", "M15", stale, now)
    assert st.state == mc.DATA_LAGGED
    assert st.bars_behind >= mc.DATA_LAGGED_MIN_BARS_BEHIND


def test_status_data_lagged_when_no_reading_and_open():
    now = _ny(2026, 7, 22, 10, 0)
    st = mc.compute_market_status("EURUSD", "M15", None, now)
    assert st.state == mc.DATA_LAGGED


def test_status_weekend_carries_last_close_and_next_open():
    fri_close = _ny(2026, 7, 24, 17, 0).astimezone(timezone.utc)
    st = mc.compute_market_status("EURUSD", "M15", fri_close, _ny(2026, 7, 25, 12, 0))
    assert st.state == mc.CLOSED_WEEKEND
    assert st.last_close_ts == fri_close
    assert st.next_open_ts is not None
    d = st.to_dict()
    assert d["state"] == "closed_weekend"
    assert d["reason"] == "closed_weekend"
    assert d["last_close_ts"].endswith("Z")
