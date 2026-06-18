"""Session-gap awareness for FVG detection.

A 3-candle FVG whose window straddles a market closure (week-end, daily gold
maintenance break, holiday) is a *session gap*, not a momentum imbalance, and
must not be surfaced as an FVG. The closure is detected data-driven from the
inter-bar TIME gap (no hardcoded session calendar): a delta markedly larger
than the nominal bar step (> FVG_SESSION_GAP_MULT × step) marks a closure.

These tests pin the behaviour:
  (a) normal-interval window with a real bullish gap  -> FVG emitted
  (b) identical geometry but a time jump in the window -> NO FVG
  (c) bearish symmetric case
  + non-regression (gap OUTSIDE the window keeps the FVG) and the disable toggle.

Covers M15 / H1 / H4.
"""

import numpy as np
import pandas as pd
import pytest

from src.environment.strategy_features import SmartMoneyEngine, SMCConfig


# Pandas freq aliases + their nominal step, one per timeframe under test.
TIMEFRAMES = [
    ("M15", "15min", pd.Timedelta(minutes=15)),
    ("H1", "1h", pd.Timedelta(hours=1)),
    ("H4", "4h", pd.Timedelta(hours=4)),
]

_N = 40  # enough bars to seed a stable ATR(14)


def _seed_rows(n: int) -> list[tuple[float, float, float, float]]:
    """Flat doji baseline at 100.0 so ATR is small and stable."""
    return [(100.0, 100.3, 99.7, 100.0) for _ in range(n)]


def _engineer_bullish(rows: list) -> list:
    """Force a clean bullish FVG at the last bar: low[i] > high[i-2]."""
    rows = list(rows)
    rows[-3] = (100.0, 100.5, 99.6, 100.2)   # high[i-2] = 100.5
    rows[-2] = (100.4, 103.0, 100.3, 102.8)  # strong bull, expands the window
    rows[-1] = (102.9, 103.5, 102.5, 103.2)  # low[i] = 102.5 -> gap = 2.0
    return rows


def _engineer_bearish(rows: list) -> list:
    """Force a clean bearish FVG at the last bar: high[i] < low[i-2]."""
    rows = list(rows)
    rows[-3] = (100.0, 100.4, 99.5, 99.8)    # low[i-2] = 99.5
    rows[-2] = (99.6, 99.7, 97.0, 97.2)      # strong bear
    rows[-1] = (97.1, 97.5, 96.5, 96.8)      # high[i] = 97.5 -> gap = 2.0
    return rows


def _frame(rows: list, timestamps) -> pd.DataFrame:
    o, h, l, c = zip(*rows)
    return pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * len(rows)},
        index=pd.DatetimeIndex(timestamps),
    )


def _continuous_ts(freq: str, n: int = _N):
    # Start on a Monday so a 3-day jump lands cleanly outside any DST edge.
    return list(pd.date_range("2024-01-08 00:00", periods=n, freq=freq))


def _ts_with_gap(freq: str, gap_at: int, n: int = _N):
    """Continuous timestamps, then a closure-sized jump injected at ``gap_at``."""
    ts = _continuous_ts(freq, n)
    big = pd.Timedelta(days=3)  # >> any nominal step (15min / 1h / 4h)
    return ts[:gap_at] + [t + big for t in ts[gap_at:]]


def _last_fvg(df: pd.DataFrame, config: dict | None = None) -> int:
    engine = SmartMoneyEngine(df.copy(), config or {})
    return int(engine.analyze()["FVG_SIGNAL"].iloc[-1])


# --------------------------------------------------------------------------- #
# (a) normal-interval window -> FVG emitted (non-regression)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tf,freq,_step", TIMEFRAMES)
def test_continuous_window_emits_bullish_fvg(tf, freq, _step):
    df = _frame(_engineer_bullish(_seed_rows(_N)), _continuous_ts(freq))
    assert _last_fvg(df) == 1, f"{tf}: continuous window should keep the bullish FVG"


@pytest.mark.parametrize("tf,freq,_step", TIMEFRAMES)
def test_continuous_window_emits_bearish_fvg(tf, freq, _step):
    df = _frame(_engineer_bearish(_seed_rows(_N)), _continuous_ts(freq))
    assert _last_fvg(df) == -1, f"{tf}: continuous window should keep the bearish FVG"


# --------------------------------------------------------------------------- #
# (b) same geometry but a time jump in the window -> NO FVG
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tf,freq,_step", TIMEFRAMES)
@pytest.mark.parametrize("gap_at", [_N - 1, _N - 2])  # i-1->i  and  i-2->i-1
def test_gap_in_window_suppresses_bullish_fvg(tf, freq, _step, gap_at):
    df = _frame(_engineer_bullish(_seed_rows(_N)), _ts_with_gap(freq, gap_at))
    assert _last_fvg(df) == 0, (
        f"{tf}: a closure (jump at {gap_at}) inside the window must suppress the FVG"
    )


# --------------------------------------------------------------------------- #
# (c) bearish symmetric case under a closure
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tf,freq,_step", TIMEFRAMES)
@pytest.mark.parametrize("gap_at", [_N - 1, _N - 2])
def test_gap_in_window_suppresses_bearish_fvg(tf, freq, _step, gap_at):
    df = _frame(_engineer_bearish(_seed_rows(_N)), _ts_with_gap(freq, gap_at))
    assert _last_fvg(df) == 0, f"{tf}: bearish FVG across a closure must be suppressed"


# --------------------------------------------------------------------------- #
# Narrowness: a gap OUTSIDE the 3-candle window leaves the FVG untouched.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("tf,freq,_step", TIMEFRAMES)
def test_gap_outside_window_keeps_fvg(tf, freq, _step):
    # Window for the last bar is [N-3, N-2, N-1]; inject the jump well before it.
    df = _frame(_engineer_bullish(_seed_rows(_N)), _ts_with_gap(freq, _N - 5))
    assert _last_fvg(df) == 1, f"{tf}: a gap outside the window must not touch the FVG"


# --------------------------------------------------------------------------- #
# Disable toggle + all FVG_* columns are zeroed coherently on suppression.
# --------------------------------------------------------------------------- #
def test_disable_toggle_restores_legacy_behaviour():
    df = _frame(_engineer_bullish(_seed_rows(_N)), _ts_with_gap("15min", _N - 1))
    # mult=0 => filter off => legacy geometric FVG survives the closure.
    assert _last_fvg(df, {"FVG_SESSION_GAP_MULT": 0.0}) == 1
    # default => suppressed.
    assert _last_fvg(df) == 0


def test_suppressed_bar_zeroes_full_fvg_tuple():
    df = _frame(_engineer_bullish(_seed_rows(_N)), _ts_with_gap("15min", _N - 1))
    last = SmartMoneyEngine(df.copy(), {}).analyze().iloc[-1]
    for col in ("FVG_SIGNAL", "FVG_DIR", "FVG_SIZE", "FVG_SIZE_NORM"):
        assert last[col] == 0, f"{col} should be zeroed on a suppressed FVG bar"


def test_integer_index_keeps_legacy_behaviour():
    """No DatetimeIndex (e.g. integer-indexed test frames) => filter no-ops."""
    rows = _engineer_bullish(_seed_rows(_N))
    o, h, l, c = zip(*rows)
    df = pd.DataFrame(
        {"open": o, "high": h, "low": l, "close": c, "volume": [1000.0] * _N}
    )  # default RangeIndex
    assert _last_fvg(df) == 1
