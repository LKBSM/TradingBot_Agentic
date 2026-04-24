"""Tests for the backtest text/JSON report renderer."""

from __future__ import annotations

import json

import pytest

from src.backtest.metrics import compute_performance, tier_from_score
from src.backtest.report import render_json, render_text
from src.backtest.state_machine_replay import ReplayResults, TradeRecord


def _trade(r: float, score: float = 70.0) -> TradeRecord:
    return TradeRecord(
        signal_id="t",
        direction="LONG" if r >= 0 else "SHORT",
        entry_bar="2024-01-01 00:00:00",
        exit_bar="2024-01-01 01:00:00",
        entry_price=2000.0,
        exit_price=2000.0 + r * 10.0,
        stop_loss=1990.0,
        take_profit=2020.0,
        confluence_score=score,
        exit_reason="target_reached" if r > 0 else "invalidated",
        bars_held=5,
        pnl_price=r * 10.0,
        r_multiple=r,
        initial_risk=10.0,
    )


def _results(trades) -> ReplayResults:
    r = ReplayResults(
        symbol="XAUUSD",
        timeframe="M15",
        bars_processed=1000,
        date_range=("2024-01-01", "2024-01-11"),
        state_machine_config={
            "enter_threshold": 75.0,
            "exit_threshold": 55.0,
            "confirm_bars": 2,
            "cooldown_bars": 2,
            "max_signal_age_bars": 12,
        },
        trades=list(trades),
    )
    r.total_trades = len(trades)
    return r


# ---------------------------------------------------------------------------
# render_text
# ---------------------------------------------------------------------------

def test_render_text_handles_empty_results():
    results = _results([])
    text = render_text(results)
    assert "XAUUSD" in text
    assert "no trades completed" in text


def test_render_text_includes_core_metrics():
    trades = [_trade(2.0), _trade(-1.0), _trade(1.5)]
    results = _results(trades)
    text = render_text(results)
    assert "TRADES" in text
    assert "RETURNS" in text
    assert "RISK" in text
    assert "XAUUSD" in text
    assert "M15" in text


def test_render_text_with_metrics_shows_annualised_and_tiers():
    trades = [
        _trade(2.0, score=85),
        _trade(-1.0, score=82),
        _trade(1.0, score=70),
    ]
    results = _results(trades)
    metrics = compute_performance(
        trades, timeframe="M15",
        tier_fn=lambda t: tier_from_score(t.confluence_score),
        bars_processed=results.bars_processed,
    )
    text = render_text(results, metrics)
    assert "Sortino" in text
    assert "PER-TIER" in text
    assert "PREMIUM" in text


def test_render_text_is_80_col_safe():
    trades = [_trade(2.5), _trade(-0.3), _trade(4.2), _trade(-1.5)]
    results = _results(trades)
    metrics = compute_performance(trades, timeframe="M15")
    text = render_text(results, metrics)
    # Most lines should fit in 80 chars. Allow a small overrun tolerance for
    # the rare wide stat line, but no line should exceed 100 chars.
    for line in text.splitlines():
        assert len(line) <= 100, f"line too long ({len(line)}): {line!r}"


# ---------------------------------------------------------------------------
# render_json
# ---------------------------------------------------------------------------

def test_render_json_merges_results_and_metrics():
    trades = [_trade(1.0), _trade(-0.5)]
    results = _results(trades)
    metrics = compute_performance(trades, timeframe="M15", bars_processed=500)
    payload = render_json(results, metrics)
    assert payload["symbol"] == "XAUUSD"
    assert "institutional_metrics" in payload
    # And the payload must be JSON-serialisable
    json.dumps(payload)


def test_render_json_without_metrics_omits_institutional_section():
    results = _results([_trade(1.0)])
    payload = render_json(results)
    assert "institutional_metrics" not in payload
