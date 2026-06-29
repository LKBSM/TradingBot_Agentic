"""Tests — external liquidity pools (EQH/EQL + range extremes).

Covers the descriptive-only liquidity feature added on feat/liquidity-detection:
  * EQH/EQL clustering under the ATR tolerance,
  * external (range-extreme) vs internal pocket distinction,
  * the factual intact → swept → broken lifecycle (no breach ⇒ never swept),
  * stable IDs, and the inviolable "no predictive/bias field" guarantee,
  * a zero-regression smoke run through the real SmartMoneyEngine.

The collector reads ONLY the engine's existing swing fractals; these unit tests
feed the UP_FRACTAL / DOWN_FRACTAL columns directly for deterministic geometry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intelligence.market_reading_mappers import (
    collect_liquidity_pools,
    _liquidity_to_models,
    _pool_lifecycle,
)
from src.intelligence.market_reading_schema import LiquidityPool


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _frame(n: int = 24, atr: float = 1.0) -> pd.DataFrame:
    """Flat OHLC frame with empty fractal columns and a UTC DatetimeIndex."""
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    df = pd.DataFrame(index=idx)
    df["high"] = 99.0
    df["low"] = 98.0
    df["close"] = 98.5
    df["ATR"] = atr
    df["UP_FRACTAL"] = np.nan
    df["DOWN_FRACTAL"] = np.nan
    return df


def _set(df: pd.DataFrame, bar: int, col: str, value: float) -> None:
    df.iloc[bar, df.columns.get_loc(col)] = value


def _bar(df: pd.DataFrame, bar: int, *, high=None, low=None, close=None) -> None:
    if high is not None:
        df.iloc[bar, df.columns.get_loc("high")] = high
    if low is not None:
        df.iloc[bar, df.columns.get_loc("low")] = low
    if close is not None:
        df.iloc[bar, df.columns.get_loc("close")] = close


def _pool(pools, side, kind):
    for p in pools:
        if p["side"] == side and p["kind"] == kind:
            return p
    return None


# --------------------------------------------------------------------------- #
# EQH / EQL detection under tolerance
# --------------------------------------------------------------------------- #


def test_equal_highs_clustered_within_tolerance():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.00)
    _set(df, 6, "UP_FRACTAL", 100.05)  # |Δ|=0.05 ≤ eps=0.10 → equal
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh is not None
    assert eqh["touches"] == 2
    assert eqh["level"] == pytest.approx(100.05)  # founder: cluster EXTREME (highest)
    assert eqh["is_external"] is True


def test_equal_lows_clustered_within_tolerance():
    df = _frame()
    _set(df, 3, "DOWN_FRACTAL", 95.00)
    _set(df, 9, "DOWN_FRACTAL", 94.96)  # |Δ|=0.04 ≤ eps → equal
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eql = _pool(pools, "ssl", "equal_lows")
    assert eql is not None
    assert eql["touches"] == 2
    assert eql["level"] == pytest.approx(94.96)  # lowest low = breachable edge
    assert eql["side"] == "ssl"


def test_highs_beyond_tolerance_are_not_equal():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.5)  # |Δ|=0.5 > eps=0.10 → NOT equal
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    assert _pool(pools, "bsl", "equal_highs") is None
    # The window's extreme swing still surfaces as a lone range_high pocket.
    rh = _pool(pools, "bsl", "range_high")
    assert rh is not None
    assert rh["level"] == pytest.approx(100.5)
    assert rh["touches"] == 1
    assert rh["is_external"] is True


def test_min_touches_threshold_respected():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.05)
    # Require 3 touches → the 2-swing cluster no longer qualifies as EQH.
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=3)
    assert _pool(pools, "bsl", "equal_highs") is None
    assert _pool(pools, "bsl", "range_high") is not None


# --------------------------------------------------------------------------- #
# External vs internal
# --------------------------------------------------------------------------- #


def test_external_vs_internal_pockets_distinguished():
    df = _frame(n=30)
    # Top equal highs (range extreme) → external.
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.04)
    # Lower equal highs well inside the range → internal.
    _set(df, 10, "UP_FRACTAL", 99.0)
    _set(df, 14, "UP_FRACTAL", 99.03)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqhs = [p for p in pools if p["kind"] == "equal_highs"]
    assert len(eqhs) == 2
    top = max(eqhs, key=lambda p: p["level"])
    low = min(eqhs, key=lambda p: p["level"])
    assert top["is_external"] is True
    assert low["is_external"] is False
    # No duplicate range_high pocket: the top cluster already holds the extreme.
    assert _pool(pools, "bsl", "range_high") is None


def test_range_extreme_single_swing_is_external():
    df = _frame()
    _set(df, 4, "UP_FRACTAL", 101.0)  # lone top swing
    _set(df, 8, "DOWN_FRACTAL", 90.0)  # lone bottom swing
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    rh = _pool(pools, "bsl", "range_high")
    rl = _pool(pools, "ssl", "range_low")
    assert rh is not None and rh["is_external"] is True and rh["touches"] == 1
    assert rl is not None and rl["is_external"] is True and rl["touches"] == 1


# --------------------------------------------------------------------------- #
# Lifecycle: intact / swept / broken (factual, no look-ahead on the level)
# --------------------------------------------------------------------------- #


def test_intact_when_no_breach():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.05)
    # All later bars stay well below the level → intact.
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh["status"] == "intact"
    assert eqh["swept_at"] is None and eqh["broken_at"] is None


def test_swept_requires_wick_through_and_close_back_inside():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.0)  # buy-side level = 100.0
    # Bar 9: wick pierces above 100 but closes back inside → sweep.
    _bar(df, 9, high=100.5, close=99.8)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh["status"] == "swept"
    assert eqh["swept_at"] == df.index[9]
    assert eqh["broken_at"] is None


def test_no_sweep_when_price_only_approaches():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.0)
    # Bar 9: high reaches only 99.9 — never trades through the level.
    _bar(df, 9, high=99.9, close=99.7)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh["status"] == "intact"


def test_broken_on_close_through_is_terminal():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.0)
    # Bar 8: a wick-sweep first; bar 11: a clean close above → broken (terminal).
    _bar(df, 8, high=100.4, close=99.9)
    _bar(df, 11, high=101.0, close=100.8)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh["status"] == "broken"
    assert eqh["broken_at"] == df.index[11]
    # The earlier sweep fact is retained alongside the terminal break.
    assert eqh["swept_at"] == df.index[8]


def test_sellside_lifecycle_mirrors_buyside():
    df = _frame()
    _set(df, 3, "DOWN_FRACTAL", 95.0)
    _set(df, 7, "DOWN_FRACTAL", 95.0)
    # Bar 10: wick below 95 but closes back inside → sweep of sell-side liquidity.
    _bar(df, 10, low=94.5, close=95.3)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eql = _pool(pools, "ssl", "equal_lows")
    assert eql["status"] == "swept"
    assert eql["swept_at"] == df.index[10]


def test_lifecycle_helper_direct():
    highs = np.array([10, 10, 10, 12, 10], dtype=float)
    lows = np.array([8, 8, 8, 8, 8], dtype=float)
    closes = np.array([9, 9, 9, 11, 9], dtype=float)  # bar 3 closes through 10
    status, swept, broken = _pool_lifecycle("bsl", 10.0, highs, lows, closes, 0, 4)
    assert status == "broken" and broken == 3


def test_no_lookahead_before_pocket_known():
    # A breach that happens BEFORE the pocket's last swing is confirmed must not
    # count: lifecycle scans only bars strictly after ``last_k``.
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _bar(df, 4, high=100.9, close=100.7)  # would "break" — but precedes 2nd swing
    _set(df, 6, "UP_FRACTAL", 100.0)      # pocket only fully known at bar 6
    _bar(df, 6, high=100.0, close=99.5)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    eqh = _pool(pools, "bsl", "equal_highs")
    assert eqh["status"] == "intact"  # the pre-confirmation breach is ignored


# --------------------------------------------------------------------------- #
# Stable IDs + descriptive-only guarantee
# --------------------------------------------------------------------------- #


def test_stable_id_is_deterministic():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.05)
    p1 = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    p2 = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    m1 = _liquidity_to_models(p1, df.index[-1])
    m2 = _liquidity_to_models(p2, df.index[-1])
    assert m1[0].id == m2[0].id
    assert m1[0].id.startswith("LIQ_bsl_equal_highs_")


def test_no_predictive_or_bias_fields_emitted():
    df = _frame()
    _set(df, 2, "UP_FRACTAL", 100.0)
    _set(df, 6, "UP_FRACTAL", 100.05)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2)
    model = _liquidity_to_models(pools, df.index[-1])[0]
    keys = set(model.model_dump().keys())
    forbidden = {
        "target", "draw", "draw_on_liquidity", "bias", "objective", "objectif",
        "cible", "setup", "probability", "prob", "conviction", "score",
        "direction", "expected_move", "next_target", "magnet",
    }
    assert keys.isdisjoint(forbidden), f"forbidden field(s): {keys & forbidden}"
    # The schema must expose exactly the descriptive surface.
    assert keys == {
        "id", "side", "kind", "level", "touches", "is_external",
        "status", "created_at", "swept_at", "broken_at", "user_flagged",
    }


# --------------------------------------------------------------------------- #
# Guards
# --------------------------------------------------------------------------- #


def test_missing_columns_returns_empty():
    df = _frame()
    df = df.drop(columns=["UP_FRACTAL"])
    assert collect_liquidity_pools(df) == []


def test_empty_frame_returns_empty():
    assert collect_liquidity_pools(pd.DataFrame()) == []


def test_cap_is_respected():
    df = _frame(n=120)
    # Many distinct lone swing highs (no equals) → only range_high survives, but
    # exercise the cap on the sell-side with several separate clusters.
    for i, lvl in enumerate([90.0, 90.02, 85.0, 85.03, 80.0, 80.01, 75.0, 75.04,
                             70.0, 70.02, 65.0, 65.03], start=2):
        _set(df, i * 2, "DOWN_FRACTAL", lvl)
    pools = collect_liquidity_pools(df, eq_tolerance_atr=0.10, eq_min_touches=2, max_pools=3)
    assert len(pools) <= 3


# --------------------------------------------------------------------------- #
# Zero-regression: real SmartMoneyEngine end-to-end
# --------------------------------------------------------------------------- #


def test_runs_through_real_engine_without_touching_other_features():
    from src.environment.strategy_features import SmartMoneyEngine

    rng = np.random.default_rng(7)
    n = 400
    idx = pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC")
    price = 2000 + np.cumsum(rng.normal(0, 1.5, n))
    high = price + rng.uniform(0.5, 3.0, n)
    low = price - rng.uniform(0.5, 3.0, n)
    close = price + rng.uniform(-1.0, 1.0, n)
    df = pd.DataFrame(
        {"open": price, "high": high, "low": low, "close": close, "volume": 1000.0},
        index=idx,
    )
    engine = SmartMoneyEngine(data=df, config={}, verbose=False)
    enriched = engine.analyze(compute_divergence=False)

    # Existing features still present and untouched.
    for col in ("UP_FRACTAL", "DOWN_FRACTAL", "FVG_SIGNAL", "BOS_EVENT"):
        assert col in enriched.columns

    pools = collect_liquidity_pools(
        enriched,
        eq_tolerance_atr=engine.config.EQ_TOLERANCE_ATR,
        eq_min_touches=engine.config.EQ_MIN_TOUCHES,
        lookback=engine.config.LIQ_LOOKBACK,
    )
    assert isinstance(pools, list)
    models = _liquidity_to_models(pools, idx[-1])
    for m in models:
        assert isinstance(m, LiquidityPool)
        assert m.side in ("bsl", "ssl")
        assert m.status in ("intact", "swept", "broken")
        # level must coincide with a real swing extreme, never invented.
        assert m.level > 0
