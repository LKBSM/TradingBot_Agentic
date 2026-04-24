"""Institutional performance metrics for the backtest harness.

The existing :mod:`state_machine_replay` ships per-trade Sharpe, profit factor,
max drawdown and expectancy. This module adds the metrics a prospect will
actually ask for: *annualised* Sharpe, Sortino, Calmar, per-tier breakdowns,
and win/loss distributions with stddev and skew.

All functions are pure — they take a list of :class:`TradeRecord` and a
``timeframe`` string, and return a :class:`PerformanceMetrics` dataclass.
They never mutate input, never touch I/O.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.backtest.state_machine_replay import TradeRecord


# Expected trading bars per year by timeframe. Used to annualise per-trade Sharpe
# under the rough (but industry-standard) assumption that trade cadence is the
# right scaling factor. Crypto uses 365*24 / (minutes/60); FX/equities use ~252
# trading days. Approximate — documented as such.
_BARS_PER_YEAR = {
    "M1":  252 * 24 * 60,
    "M5":  252 * 24 * 12,
    "M15": 252 * 24 * 4,
    "M30": 252 * 24 * 2,
    "H1":  252 * 24,
    "H4":  252 * 6,
    "D1":  252,
    "W1":  52,
}


@dataclass
class TierBreakdown:
    """Metrics stratified by signal tier (PREMIUM / STANDARD / WEAK)."""
    tier: str
    trades: int
    win_rate: float
    expectancy_r: float
    total_r: float
    profit_factor: float
    avg_bars_held: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "expectancy_r": round(self.expectancy_r, 3),
            "total_r": round(self.total_r, 3),
            "profit_factor": round(self.profit_factor, 3),
            "avg_bars_held": round(self.avg_bars_held, 2),
        }


@dataclass
class PerformanceMetrics:
    """Annualised, institutional-grade metrics for a trade series.

    Populate via :func:`compute_performance`. Every field is JSON-serialisable.
    """
    # Trade-count summary
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakeven: int = 0
    win_rate: float = 0.0
    loss_rate: float = 0.0

    # Return metrics (in R-multiples)
    avg_r: float = 0.0
    median_r: float = 0.0
    stdev_r: float = 0.0
    total_r: float = 0.0
    expectancy_r: float = 0.0
    best_r: float = 0.0
    worst_r: float = 0.0
    gross_win_r: float = 0.0
    gross_loss_r: float = 0.0

    # Risk metrics
    profit_factor: float = 0.0
    payoff_ratio: float = 0.0                   # avg_win / |avg_loss|
    max_drawdown_r: float = 0.0
    max_drawdown_pct_of_equity: float = 0.0     # assuming 1R per trade at start
    max_consecutive_losses: int = 0
    max_consecutive_wins: int = 0

    # Risk-adjusted (unannualised = per-trade)
    sharpe_per_trade: float = 0.0
    sortino_per_trade: float = 0.0
    calmar: float = 0.0                         # total_r / max_drawdown_r

    # Risk-adjusted (annualised — requires timeframe + bar cadence)
    sharpe_annualised: Optional[float] = None
    sortino_annualised: Optional[float] = None

    # Lifetime
    avg_bars_held: float = 0.0
    median_bars_held: float = 0.0
    trades_per_year: Optional[float] = None

    # Tier breakdown
    by_tier: List[TierBreakdown] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        def _rnd(x: Any, n: int = 3) -> Any:
            if x is None:
                return None
            if isinstance(x, float) and not math.isfinite(x):
                # JSON can't carry inf — serialise as a string sentinel
                return "infinity" if x > 0 else "-infinity"
            if isinstance(x, float):
                return round(x, n)
            return x

        return {
            "summary": {
                "total_trades": self.total_trades,
                "wins": self.wins,
                "losses": self.losses,
                "breakeven": self.breakeven,
                "win_rate": _rnd(self.win_rate, 4),
                "loss_rate": _rnd(self.loss_rate, 4),
            },
            "returns": {
                "avg_r": _rnd(self.avg_r),
                "median_r": _rnd(self.median_r),
                "stdev_r": _rnd(self.stdev_r),
                "total_r": _rnd(self.total_r),
                "expectancy_r": _rnd(self.expectancy_r),
                "best_r": _rnd(self.best_r),
                "worst_r": _rnd(self.worst_r),
                "gross_win_r": _rnd(self.gross_win_r),
                "gross_loss_r": _rnd(self.gross_loss_r),
            },
            "risk": {
                "profit_factor": _rnd(self.profit_factor),
                "payoff_ratio": _rnd(self.payoff_ratio),
                "max_drawdown_r": _rnd(self.max_drawdown_r),
                "max_consecutive_losses": self.max_consecutive_losses,
                "max_consecutive_wins": self.max_consecutive_wins,
            },
            "risk_adjusted": {
                "sharpe_per_trade": _rnd(self.sharpe_per_trade),
                "sortino_per_trade": _rnd(self.sortino_per_trade),
                "calmar": _rnd(self.calmar),
                "sharpe_annualised": _rnd(self.sharpe_annualised),
                "sortino_annualised": _rnd(self.sortino_annualised),
            },
            "lifetime": {
                "avg_bars_held": _rnd(self.avg_bars_held, 2),
                "median_bars_held": _rnd(self.median_bars_held, 2),
                "trades_per_year": _rnd(self.trades_per_year, 2),
            },
            "by_tier": [tb.to_dict() for tb in self.by_tier],
        }


# =============================================================================
# PUBLIC API
# =============================================================================


def compute_performance(
    trades: Sequence[TradeRecord],
    timeframe: str = "M15",
    tier_fn: Optional[Any] = None,
    bars_processed: Optional[int] = None,
) -> PerformanceMetrics:
    """Compute the full metric set for a list of trades.

    Args:
        trades: Closed trades (one per entry→exit round-trip).
        timeframe: Bar cadence — used to annualise Sharpe/Sortino.
        tier_fn: Optional callable ``TradeRecord -> str`` that maps a trade
            to its tier label. If omitted, tier stratification is skipped.
            (Tier isn't on :class:`TradeRecord` directly — the caller usually
            closes over the ConfluenceSignal or uses the confluence_score.)
        bars_processed: Total bars in the backtest window. Used with
            ``timeframe`` to compute trades-per-year.
    """
    m = PerformanceMetrics()
    m.total_trades = len(trades)
    if m.total_trades == 0:
        return m

    r_series = [t.r_multiple for t in trades]
    bars_held = [t.bars_held for t in trades]

    # Win / loss / breakeven buckets. Breakeven is exactly 0R — rare but possible
    # on straddle-bar exits at entry price. Counting it separately keeps win/loss
    # counts unambiguous (wins are strictly >0, losses are strictly <0).
    wins_r = [r for r in r_series if r > 0]
    losses_r = [r for r in r_series if r < 0]
    m.wins = len(wins_r)
    m.losses = len(losses_r)
    m.breakeven = m.total_trades - m.wins - m.losses
    m.win_rate = m.wins / m.total_trades
    m.loss_rate = m.losses / m.total_trades

    # Return distribution
    m.avg_r = float(np.mean(r_series))
    m.median_r = float(np.median(r_series))
    m.stdev_r = float(statistics.pstdev(r_series)) if len(r_series) > 1 else 0.0
    m.total_r = float(np.sum(r_series))
    m.expectancy_r = m.avg_r
    m.best_r = float(max(r_series))
    m.worst_r = float(min(r_series))
    m.gross_win_r = float(sum(wins_r))
    m.gross_loss_r = float(sum(losses_r))

    # Profit factor and payoff ratio
    denom = abs(m.gross_loss_r)
    if denom > 0:
        m.profit_factor = m.gross_win_r / denom
    else:
        m.profit_factor = float("inf") if m.gross_win_r > 0 else 0.0

    avg_win = (m.gross_win_r / m.wins) if m.wins else 0.0
    avg_loss = (m.gross_loss_r / m.losses) if m.losses else 0.0
    if avg_loss < 0:
        m.payoff_ratio = avg_win / abs(avg_loss)
    else:
        m.payoff_ratio = float("inf") if avg_win > 0 else 0.0

    # Drawdown + consecutive runs
    m.max_drawdown_r = _max_drawdown_r(r_series)
    m.max_consecutive_losses = _max_consecutive_run(r_series, lambda r: r <= 0)
    m.max_consecutive_wins = _max_consecutive_run(r_series, lambda r: r > 0)

    # Risk-adjusted metrics. Per-trade Sharpe/Sortino use population stdev so
    # they coincide with the textbook formula when stdev_r > 0. Sortino's
    # denominator is downside deviation (below-zero returns only).
    if m.stdev_r > 0:
        m.sharpe_per_trade = m.avg_r / m.stdev_r
    downside = [min(0.0, r) for r in r_series]
    downside_dev = (
        math.sqrt(sum(d * d for d in downside) / len(downside))
        if downside and any(d != 0 for d in downside)
        else 0.0
    )
    if downside_dev > 0:
        m.sortino_per_trade = m.avg_r / downside_dev
    if m.max_drawdown_r > 0:
        m.calmar = m.total_r / m.max_drawdown_r
    else:
        m.calmar = float("inf") if m.total_r > 0 else 0.0

    # Annualisation. Requires (1) a known timeframe and (2) bars_processed.
    if bars_processed and bars_processed > 0:
        bars_per_year = _BARS_PER_YEAR.get(timeframe.upper())
        if bars_per_year:
            years = bars_processed / bars_per_year
            if years > 0:
                trades_per_year = m.total_trades / years
                m.trades_per_year = trades_per_year
                if m.sharpe_per_trade:
                    m.sharpe_annualised = m.sharpe_per_trade * math.sqrt(trades_per_year)
                if m.sortino_per_trade:
                    m.sortino_annualised = m.sortino_per_trade * math.sqrt(trades_per_year)

    # Lifetime
    m.avg_bars_held = float(np.mean(bars_held))
    m.median_bars_held = float(np.median(bars_held))

    # Tier breakdown
    if tier_fn is not None:
        m.by_tier = _tier_breakdown(trades, tier_fn)

    return m


def tier_from_score(score: float) -> str:
    """Map a confluence score to its tier label.

    Mirrors :meth:`ConfluenceDetector._classify_tier` — kept in sync manually.
    Sprint 3 makes these thresholds data-driven; until then, 40/60/80.
    """
    if score >= 80.0:
        return "PREMIUM"
    if score >= 60.0:
        return "STANDARD"
    if score >= 40.0:
        return "WEAK"
    return "INVALID"


# =============================================================================
# INTERNALS
# =============================================================================


def _max_drawdown_r(r_series: List[float]) -> float:
    """Peak-to-trough drawdown on the cumulative R-curve."""
    if not r_series:
        return 0.0
    cum = np.cumsum(r_series)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    return float(dd.max())


def _max_consecutive_run(r_series: List[float], predicate: Any) -> int:
    """Longest consecutive run where ``predicate(r)`` is true."""
    best = run = 0
    for r in r_series:
        if predicate(r):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _tier_breakdown(
    trades: Sequence[TradeRecord], tier_fn: Any
) -> List[TierBreakdown]:
    buckets: Dict[str, List[TradeRecord]] = {}
    for t in trades:
        try:
            tier = str(tier_fn(t))
        except Exception:
            tier = "UNKNOWN"
        buckets.setdefault(tier, []).append(t)

    out: List[TierBreakdown] = []
    # Stable, prospect-friendly ordering
    for tier in ("PREMIUM", "STANDARD", "WEAK", "INVALID", "UNKNOWN"):
        if tier not in buckets:
            continue
        tbucket = buckets[tier]
        r_vals = [t.r_multiple for t in tbucket]
        wins = sum(1 for r in r_vals if r > 0)
        gross_win = sum(r for r in r_vals if r > 0)
        gross_loss = sum(r for r in r_vals if r < 0)
        denom = abs(gross_loss)
        pf = (gross_win / denom) if denom > 0 else (
            float("inf") if gross_win > 0 else 0.0
        )
        out.append(
            TierBreakdown(
                tier=tier,
                trades=len(tbucket),
                win_rate=wins / len(tbucket) if tbucket else 0.0,
                expectancy_r=float(np.mean(r_vals)) if r_vals else 0.0,
                total_r=float(sum(r_vals)),
                profit_factor=pf,
                avg_bars_held=float(np.mean([t.bars_held for t in tbucket])),
            )
        )
    # Include any other tier labels the caller emitted (defensive — never drop)
    for tier, tbucket in buckets.items():
        if tier in {tb.tier for tb in out}:
            continue
        r_vals = [t.r_multiple for t in tbucket]
        wins = sum(1 for r in r_vals if r > 0)
        gross_win = sum(r for r in r_vals if r > 0)
        gross_loss = sum(r for r in r_vals if r < 0)
        denom = abs(gross_loss)
        pf = (gross_win / denom) if denom > 0 else (
            float("inf") if gross_win > 0 else 0.0
        )
        out.append(
            TierBreakdown(
                tier=tier,
                trades=len(tbucket),
                win_rate=wins / len(tbucket) if tbucket else 0.0,
                expectancy_r=float(np.mean(r_vals)) if r_vals else 0.0,
                total_r=float(sum(r_vals)),
                profit_factor=pf,
                avg_bars_held=float(np.mean([t.bars_held for t in tbucket])),
            )
        )
    return out
