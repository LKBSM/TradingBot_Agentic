"""Render backtest results as a prospect-ready report.

Two entry points:

- :func:`render_text` — plain-text block (the thing a sales engineer pastes
  into an email). Deterministic, no ANSI colours, 80-column safe.
- :func:`render_json` — the same content as a nested dict, suitable for
  writing to a file and parsing downstream.

Both functions accept a :class:`ReplayResults` plus an optional
:class:`PerformanceMetrics` (from :mod:`src.backtest.metrics`). If the
institutional metrics object is supplied, the annualised + tier-stratified
sections are included; otherwise the report falls back to the per-trade
metrics already on ``ReplayResults``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from src.backtest.metrics import PerformanceMetrics
from src.backtest.state_machine_replay import ReplayResults


# =============================================================================
# PUBLIC API
# =============================================================================


def render_text(
    results: ReplayResults,
    metrics: Optional[PerformanceMetrics] = None,
    *,
    include_diagnostics: bool = True,
) -> str:
    """Return an 80-column text report. Deterministic — no timestamps."""
    lines: List[str] = []
    lines.append("=" * 76)
    lines.append(
        f" Backtest Report  {results.symbol}  {results.timeframe}".ljust(76)
    )
    lines.append("=" * 76)
    lines.append(f" Window   : {results.date_range[0]}  ->  {results.date_range[1]}")
    lines.append(f" Bars     : {results.bars_processed:,}")
    lines.append(
        f" Config   : enter={results.state_machine_config.get('enter_threshold'):.0f}"
        f"  exit={results.state_machine_config.get('exit_threshold'):.0f}"
        f"  confirm={results.state_machine_config.get('confirm_bars')}"
        f"  cooldown={results.state_machine_config.get('cooldown_bars')}"
        f"  max_age={results.state_machine_config.get('max_signal_age_bars')}"
    )
    lines.append("")

    # --- Trade summary --------------------------------------------------- #
    m = metrics
    total = m.total_trades if m else results.total_trades
    wins = m.wins if m else results.wins
    losses = m.losses if m else results.losses
    win_rate = (m.win_rate if m else results.win_rate) * 100.0

    lines.append(" TRADES")
    lines.append("-" * 76)
    lines.append(f" Total                : {total}")
    lines.append(f" Wins                 : {wins}  ({win_rate:.1f}%)")
    lines.append(f" Losses               : {losses}")
    if m is not None and m.breakeven > 0:
        lines.append(f" Breakeven            : {m.breakeven}")
    other = (m.total_trades - m.wins - m.losses - (m.breakeven if m else 0)) \
        if m else results.other_exits
    lines.append(f" Other exits          : {other}")
    lines.append("")

    # --- Returns --------------------------------------------------------- #
    avg_r = m.avg_r if m else results.avg_r
    total_r = m.total_r if m else results.total_r
    best = m.best_r if m else results.best_r
    worst = m.worst_r if m else results.worst_r
    lines.append(" RETURNS  (R-multiples)")
    lines.append("-" * 76)
    lines.append(f" Expectancy / trade   : {avg_r:+.3f} R")
    lines.append(f" Total                : {total_r:+.2f} R")
    if m is not None:
        lines.append(f" Median               : {m.median_r:+.3f} R")
        lines.append(f" Stdev                : {m.stdev_r:.3f} R")
    lines.append(f" Best  /  Worst trade : {best:+.2f} R  /  {worst:+.2f} R")
    lines.append("")

    # --- Risk ------------------------------------------------------------ #
    pf = m.profit_factor if m else results.profit_factor
    max_dd = m.max_drawdown_r if m else results.max_drawdown_r
    mcl = m.max_consecutive_losses if m else results.max_consecutive_losses
    gross_win = m.gross_win_r if m else results.gross_win_r
    gross_loss = m.gross_loss_r if m else results.gross_loss_r
    lines.append(" RISK")
    lines.append("-" * 76)
    lines.append(
        f" Profit factor        : {_fmt_ratio(pf):>8s}   "
        f"(wins {gross_win:+.2f} R  vs losses {gross_loss:+.2f} R)"
    )
    if m is not None:
        lines.append(f" Payoff ratio         : {_fmt_ratio(m.payoff_ratio):>8s}")
    lines.append(f" Max drawdown         : {max_dd:.2f} R")
    lines.append(f" Max consec losses    : {mcl}")
    if m is not None:
        lines.append(f" Max consec wins      : {m.max_consecutive_wins}")
    lines.append("")

    # --- Risk-adjusted --------------------------------------------------- #
    sharpe_pt = m.sharpe_per_trade if m else results.sharpe_per_trade
    lines.append(" RISK-ADJUSTED")
    lines.append("-" * 76)
    lines.append(f" Sharpe (per trade)   : {sharpe_pt:+.3f}")
    if m is not None:
        lines.append(f" Sortino (per trade)  : {m.sortino_per_trade:+.3f}")
        lines.append(f" Calmar               : {_fmt_ratio(m.calmar):>8s}")
        if m.sharpe_annualised is not None:
            lines.append(
                f" Sharpe (annualised)  : {m.sharpe_annualised:+.3f}  "
                f"(~{m.trades_per_year:.0f} trades/year)"
            )
        if m.sortino_annualised is not None:
            lines.append(f" Sortino (annualised) : {m.sortino_annualised:+.3f}")
    lines.append("")

    # --- Lifetime / cadence --------------------------------------------- #
    avg_hold = m.avg_bars_held if m else results.avg_bars_held
    lines.append(" LIFETIME")
    lines.append("-" * 76)
    lines.append(f" Avg bars held        : {avg_hold:.1f}")
    lines.append(f" Signals / day        : {results.signals_per_day:.2f}")
    lines.append(
        f" State-machine lifetime (avg) : "
        f"{results.avg_signal_lifetime_bars_machine:.1f} bars"
    )
    cr = results.confirmation_rate
    if cr is not None:
        lines.append(
            f" Confirmation rate    : {cr * 100:.1f}%  "
            f"({results.arms_confirmed}/{results.arms_started} arms, "
            f"{results.arms_aborted} aborted)"
        )
    lines.append("")

    # --- Per-tier breakdown --------------------------------------------- #
    if m is not None and m.by_tier:
        lines.append(" PER-TIER")
        lines.append("-" * 76)
        lines.append(
            f" {'Tier':<10s}  {'N':>5s}  {'Win%':>6s}  "
            f"{'E[R]':>8s}  {'Total R':>9s}  {'PF':>7s}  {'AvgBars':>8s}"
        )
        for tb in m.by_tier:
            lines.append(
                f" {tb.tier:<10s}  {tb.trades:>5d}  "
                f"{tb.win_rate * 100:>5.1f}%  "
                f"{tb.expectancy_r:>+8.3f}  {tb.total_r:>+9.2f}  "
                f"{_fmt_ratio(tb.profit_factor):>7s}  {tb.avg_bars_held:>8.1f}"
            )
        lines.append("")

    # --- Exit reasons --------------------------------------------------- #
    lines.append(" EXITS BY REASON")
    lines.append("-" * 76)
    if results.exits_by_reason:
        total_exits = sum(results.exits_by_reason.values())
        for reason, n in sorted(
            results.exits_by_reason.items(), key=lambda kv: -kv[1]
        ):
            pct = 100.0 * n / total_exits if total_exits else 0.0
            lines.append(f"  {reason:<18s} {n:>5d}  ({pct:5.1f}%)")
    else:
        lines.append("  (no trades completed)")
    lines.append("")

    # --- Diagnostics ---------------------------------------------------- #
    if include_diagnostics:
        lines.append(" DIAGNOSTICS")
        lines.append("-" * 76)
        lines.append(
            f" BOS events           : {results.bars_with_bos:,}   "
            f"detector signals: {results.signals_produced_by_detector:,}   "
            f"max score: {results.score_max:.1f}"
        )
        if results.score_percentiles:
            pct_str = "  ".join(
                f"{k}={v:.1f}" for k, v in results.score_percentiles.items()
            )
            lines.append(f" Score percentiles    : {pct_str}")
        if results.open_trade_bars > 0:
            lines.append(
                f" Open at end          : {results.open_trade_bars} bars "
                f"(trade not counted)"
            )
        lines.append("")

    lines.append("=" * 76)
    return "\n".join(lines)


def render_json(
    results: ReplayResults,
    metrics: Optional[PerformanceMetrics] = None,
    *,
    include_trades: bool = True,
) -> Dict[str, Any]:
    """Return the report as a nested dict, JSON-safe."""
    payload = results.to_dict(include_trades=include_trades)
    if metrics is not None:
        payload["institutional_metrics"] = metrics.to_dict()
    return payload


# =============================================================================
# INTERNALS
# =============================================================================


def _fmt_ratio(value: float) -> str:
    """Format a ratio as a fixed-width 2-decimal string, with ∞ fallback."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.2f}"
