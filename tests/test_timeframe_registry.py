"""TF-1 — the single timeframe registry.

Pins the registry contract and, crucially, the alignment ladder (decision C) and
per-unit relevance (decision D) across ALL six units — the values every other
surface now derives from.
"""

from __future__ import annotations

import pytest

from src.intelligence import timeframe_registry as tr


@pytest.fixture(autouse=True)
def _reset():
    tr.reset_cache()
    yield
    tr.reset_cache()


def test_perimeter_and_reference():
    assert tr.perimeter_ids() == ("M1", "M5", "M15", "H1", "H4", "D1")
    assert "W1" in tr.reference_ids()
    assert "M30" not in tr.perimeter_ids()  # known unit, not tradeable


@pytest.mark.parametrize(
    "tf,minutes,provider,seconds",
    [
        ("M1", 1, "1min", 60),
        ("M5", 5, "5min", 300),
        ("M15", 15, "15min", 900),
        ("H1", 60, "1h", 3600),
        ("H4", 240, "4h", 14400),
        ("D1", 1440, "1day", 86400),
        ("W1", 10080, "1week", 604800),
    ],
)
def test_core_attributes(tf, minutes, provider, seconds):
    s = tr.spec(tf)
    assert s.minutes == minutes and s.seconds == seconds and s.provider == provider


@pytest.mark.parametrize(
    "tf,expected",
    [
        ("M1", ("M5", "M15", "H1", "H4", "D1")),
        ("M5", ("M15", "H1", "H4", "D1")),
        ("M15", ("H1", "H4", "D1")),
        ("H1", ("H4", "D1")),
        ("H4", ("D1",)),
        ("D1", ("W1",)),        # top of perimeter → reference unit above
        ("W1", ()),             # nothing above → "no higher unit"
    ],
)
def test_alignment_ladder_is_relative(tf, expected):
    assert tr.alignment_timeframes(tf) == expected


@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4"])
def test_intraday_is_session_and_prevlevel_relevant(tf):
    assert tr.is_session_relevant(tf) is True
    assert tr.is_prev_levels_relevant(tf) is True


@pytest.mark.parametrize("tf", ["D1", "W1"])
def test_daily_hides_session_and_prevlevels(tf):
    # A daily candle spans all sessions → Session + "veille" hidden-with-mention.
    assert tr.is_session_relevant(tf) is False
    assert tr.is_prev_levels_relevant(tf) is False


def test_minutes_map_is_the_single_source():
    m = tr.minutes_map()
    assert m["M5"] == 5 and m["D1"] == 1440 and m["W1"] == 10080


def test_unknown_timeframe_raises():
    with pytest.raises(KeyError):
        tr.spec("M7")
    assert tr.has("M7") is False
