"""Tests for institutional backtest metrics."""

from __future__ import annotations

import math

import pytest

from src.backtest.metrics import (
    PerformanceMetrics,
    compute_performance,
    tier_from_score,
)
from src.backtest.state_machine_replay import TradeRecord


def _trade(r: float, score: float = 70.0, bars: int = 5) -> TradeRecord:
    """Build a minimal TradeRecord fixture. Only r_multiple / score / bars
    influence the metrics under test; other fields are populated with
    plausible placeholder values."""
    return TradeRecord(
        signal_id="t",
        direction="LONG",
        entry_bar="2024-01-01 00:00:00",
        exit_bar="2024-01-01 01:00:00",
        entry_price=2000.0,
        exit_price=2000.0 + r * 10.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        confluence_score=score,
        exit_reason="target_reached" if r > 0 else "invalidated",
        bars_held=bars,
        pnl_price=r * 10.0,
        r_multiple=r,
        initial_risk=10.0,
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_trades_yields_zero_metrics():
    m = compute_performance([])
    assert m.total_trades == 0
    assert m.win_rate == 0.0
    assert m.profit_factor == 0.0
    assert m.sharpe_per_trade == 0.0
    assert m.by_tier == []


def test_single_winning_trade():
    m = compute_performance([_trade(2.0)])
    assert m.total_trades == 1
    assert m.wins == 1
    assert m.losses == 0
    assert m.win_rate == 1.0
    assert m.expectancy_r == pytest.approx(2.0)
    # Only one trade — population stdev is 0, so Sharpe is 0 (not inf)
    assert m.sharpe_per_trade == 0.0
    # Profit factor with no losses and a win = inf
    assert math.isinf(m.profit_factor)


def test_single_losing_trade():
    m = compute_performance([_trade(-1.0)])
    assert m.wins == 0
    assert m.losses == 1
    assert m.loss_rate == 1.0
    # No wins → profit factor is 0 (not inf)
    assert m.profit_factor == 0.0


# ---------------------------------------------------------------------------
# Return / distribution stats
# ---------------------------------------------------------------------------

def test_mixed_trades_expectancy_and_stdev():
    # 3 wins (2R, 3R, 1R), 2 losses (-1R, -1R) = +4R total / 5 = 0.8R avg
    trades = [_trade(r) for r in (2.0, 3.0, 1.0, -1.0, -1.0)]
    m = compute_performance(trades)
    assert m.total_trades == 5
    assert m.wins == 3
    assert m.losses == 2
    assert m.win_rate == pytest.approx(0.6)
    assert m.total_r == pytest.approx(4.0)
    assert m.expectancy_r == pytest.approx(0.8)
    assert m.stdev_r > 0
    assert m.best_r == 3.0
    assert m.worst_r == -1.0


def test_profit_factor_and_payoff_ratio():
    # Wins: 2R + 3R = 5R. Losses: -1R + -1R = -2R. PF = 2.5
    # avg_win = 2.5, avg_loss = -1.0, payoff = 2.5
    trades = [_trade(2.0), _trade(3.0), _trade(-1.0), _trade(-1.0)]
    m = compute_performance(trades)
    assert m.profit_factor == pytest.approx(2.5)
    assert m.payoff_ratio == pytest.approx(2.5)


def test_max_drawdown_on_losing_streak():
    # +1, -1, -1, -1, +2. Peak equity after bar 1 = 1R. Trough at bar 4 = -2R.
    # Drawdown = 1 - (-2) = 3R.
    trades = [_trade(r) for r in (1.0, -1.0, -1.0, -1.0, 2.0)]
    m = compute_performance(trades)
    assert m.max_drawdown_r == pytest.approx(3.0)
    assert m.max_consecutive_losses == 3
    assert m.max_consecutive_wins == 1  # trailing +2 is 1 consecutive


def test_breakeven_counted_separately():
    # Exactly-zero R (can happen on straddle-bar exit at entry) is neither
    # a win nor a loss.
    trades = [_trade(2.0), _trade(0.0), _trade(-1.0)]
    m = compute_performance(trades)
    assert m.wins == 1
    assert m.losses == 1
    assert m.breakeven == 1
    assert m.win_rate == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Risk-adjusted
# ---------------------------------------------------------------------------

def test_sharpe_positive_when_mean_positive():
    trades = [_trade(r) for r in (1.5, 1.0, -0.5, 2.0, -0.3)]
    m = compute_performance(trades)
    assert m.sharpe_per_trade > 0
    assert m.sortino_per_trade > 0


def test_sortino_denominator_only_downside():
    # All positive trades → no downside → sortino is 0 by convention
    # (downside_dev = 0 means we can't divide; implementation returns 0).
    trades = [_trade(r) for r in (1.0, 2.0, 3.0)]
    m = compute_performance(trades)
    assert m.sortino_per_trade == 0.0
    assert m.sharpe_per_trade > 0


def test_annualised_sharpe_requires_bars_and_timeframe():
    # Without bars_processed, annualised fields stay None
    trades = [_trade(r) for r in (1.0, -0.5, 1.5, -0.3)]
    m = compute_performance(trades)
    assert m.sharpe_annualised is None
    assert m.trades_per_year is None

    # With bars_processed + valid timeframe, they compute
    m2 = compute_performance(trades, timeframe="M15", bars_processed=252 * 24 * 4)
    assert m2.sharpe_annualised is not None
    assert m2.trades_per_year == pytest.approx(4.0)  # 4 trades in 1 year


def test_annualised_sharpe_rejects_unknown_timeframe():
    trades = [_trade(r) for r in (1.0, -0.5)]
    m = compute_performance(trades, timeframe="UNKNOWN", bars_processed=10000)
    assert m.sharpe_annualised is None


# ---------------------------------------------------------------------------
# Tier breakdown
# ---------------------------------------------------------------------------

def test_tier_from_score_boundaries():
    assert tier_from_score(85.0) == "PREMIUM"
    assert tier_from_score(80.0) == "PREMIUM"
    assert tier_from_score(79.9) == "STANDARD"
    assert tier_from_score(60.0) == "STANDARD"
    assert tier_from_score(59.9) == "WEAK"
    assert tier_from_score(40.0) == "WEAK"
    assert tier_from_score(39.9) == "INVALID"


def test_tier_breakdown_stratifies_trades():
    trades = [
        _trade(2.0, score=85),   # PREMIUM win
        _trade(1.5, score=82),   # PREMIUM win
        _trade(-1.0, score=84),  # PREMIUM loss
        _trade(0.5, score=70),   # STANDARD win
        _trade(-0.8, score=65),  # STANDARD loss
    ]
    m = compute_performance(
        trades, tier_fn=lambda t: tier_from_score(t.confluence_score)
    )
    assert len(m.by_tier) == 2
    premium = next(tb for tb in m.by_tier if tb.tier == "PREMIUM")
    standard = next(tb for tb in m.by_tier if tb.tier == "STANDARD")
    assert premium.trades == 3
    assert premium.win_rate == pytest.approx(2 / 3)
    assert standard.trades == 2
    assert standard.win_rate == pytest.approx(0.5)
    # Prospect-friendly ordering: PREMIUM before STANDARD
    assert m.by_tier[0].tier == "PREMIUM"


def test_tier_breakdown_preserves_unknown_labels():
    # A custom tier label gets its own bucket, not silently dropped
    trades = [_trade(1.0, score=50), _trade(-1.0, score=50)]
    m = compute_performance(trades, tier_fn=lambda t: "CUSTOM")
    assert len(m.by_tier) == 1
    assert m.by_tier[0].tier == "CUSTOM"


# ---------------------------------------------------------------------------
# JSON safety
# ---------------------------------------------------------------------------

def test_infinite_profit_factor_serialises_as_string():
    # All wins → infinite profit factor. to_dict must produce something
    # a JSON encoder can handle.
    trades = [_trade(1.0), _trade(2.0)]
    m = compute_performance(trades)
    payload = m.to_dict()
    assert payload["risk"]["profit_factor"] in ("infinity", "-infinity")


def test_to_dict_round_trips_to_json():
    import json
    trades = [_trade(1.5), _trade(-0.5), _trade(2.0, score=85)]
    m = compute_performance(
        trades, tier_fn=lambda t: tier_from_score(t.confluence_score),
        bars_processed=1000, timeframe="M15",
    )
    payload = m.to_dict()
    # Must not raise
    json.dumps(payload)
