"""Backtest validation gates — couples `src/research/strategy_gates` to backtest outputs.

Sprint 3 batch 3.4 (audit P0-17): the CPCV / DSR / PBO machinery exists in
``src/research/strategy_gates.py`` and ``src/research/cpcv_harness.py`` but
was never invoked by the main backtest entrypoint
(``scripts/run_backtest.py``). This module is the bridge.

Public API
----------
- :func:`validate_trades_dataframe(trades_df)` — accept a DataFrame with
  per-trade ``pnl_r`` (R-multiple returns) and produce a
  ``GateResult`` (pass/fail per criterion + verdict).
- :func:`validate_backtest_artifact(json_path)` — load a backtest
  ``_summary.json`` output and validate it.

The gates checked:

- ``DSR >= 1.5``               (Bailey & López de Prado 2014)
- ``PBO <= 0.35``              (Bailey-Borwein-LdP-Zhu 2014)
- ``PF lower CI 95% > 1.00``   (bootstrap, n=1000)
- ``DM p-value < 0.05``        (Diebold-Mariano vs constant zero baseline)
- ``n_trades >= 30``           (minimum sample size)

A backtest that fails any gate is **not commercializable**. The expectation
is that during Sprint 3, no strategy passes all gates (that is fine — it
proves the gates are working and that the search must continue).

Reference
---------
- ``src/research/strategy_gates.py:189`` — :func:`evaluate_gates`
- ``audits/2026-Q2/section_3_8_backtest_engine.md`` — P0-17
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src.research.strategy_gates import GateResult, evaluate_gates

logger = logging.getLogger(__name__)


# =============================================================================
# Public API
# =============================================================================


def validate_trades_dataframe(
    trades_df: pd.DataFrame,
    *,
    pnl_column: str = "pnl_r",
    n_trials: int = 1,
    **gate_kwargs,
) -> GateResult:
    """Run admission gates on a per-trade returns DataFrame.

    Parameters
    ----------
    trades_df
        DataFrame with one row per trade. Must contain a column with the
        R-multiple return for each trade (default: ``pnl_r``).
    pnl_column
        Name of the column to use as returns. Falls back to common
        alternatives (``r_multiple``, ``realized_r``, ``pnl``) if absent.
    n_trials
        Number of hyper-parameter trials run during search (passed to DSR
        deflation formula). Use 1 when no search took place; use the actual
        grid size when CPCV sweep was performed.
    gate_kwargs
        Forwarded to :func:`evaluate_gates` (override default thresholds).

    Returns
    -------
    GateResult
        Pass/fail per criterion with metric values and failure reasons.
    """
    if len(trades_df) == 0:
        # No trades at all → cannot evaluate, return failure
        return GateResult(
            n_trades=0,
            sharpe=0.0,
            profit_factor=0.0,
            profit_factor_lo=0.0,
            profit_factor_hi=0.0,
            dsr=0.0,
            pbo=0.5,
            dm_stat=0.0,
            dm_pvalue=1.0,
            trades_pass=False,
            dsr_pass=False,
            pbo_pass=False,
            pf_lo_pass=False,
            dm_pass=False,
            thresholds={},
            failure_reasons=["no_trades_in_dataframe"],
        )
    candidates = [pnl_column, "pnl_r", "r_multiple", "realized_r", "pnl", "r"]
    chosen = next((c for c in candidates if c in trades_df.columns), None)
    if chosen is None:
        raise ValueError(
            f"trades_df has no recognized PnL column. Got {list(trades_df.columns)!r}."
        )

    returns = trades_df[chosen].to_numpy(dtype=float)
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        logger.warning("Empty trades — gates cannot be evaluated meaningfully")
        return GateResult(
            n_trades=0,
            sharpe=0.0,
            profit_factor=0.0,
            profit_factor_lo=0.0,
            profit_factor_hi=0.0,
            dsr=0.0,
            pbo=0.5,
            dm_stat=0.0,
            dm_pvalue=1.0,
            trades_pass=False,
            dsr_pass=False,
            pbo_pass=False,
            pf_lo_pass=False,
            dm_pass=False,
            thresholds={},
            failure_reasons=["no_trades"],
        )

    return evaluate_gates(
        returns=returns,
        n_trials=n_trials,
        baseline_returns=np.zeros_like(returns),
        **gate_kwargs,
    )


def validate_backtest_artifact(
    json_path: Union[str, Path],
    *,
    n_trials: int = 1,
    **gate_kwargs,
) -> GateResult:
    """Validate a backtest output JSON (as produced by ``run_backtest.py``).

    Reads the embedded ``trades`` array, extracts per-trade R-multiples, and
    runs the gates.
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    trades = payload.get("trades") or []
    if not trades:
        logger.warning("Backtest %s has no trades — vacuous run.", path)
    df = pd.DataFrame(trades)
    return validate_trades_dataframe(df, n_trials=n_trials, **gate_kwargs)


def render_gate_report(result: GateResult) -> str:
    """Render a human-readable text report of a GateResult."""
    lines = ["=== STRATEGY ADMISSION GATES ==="]
    lines.append(f"Verdict: {'✅ ALL GATES PASSED' if result.all_passed else '❌ FAILED'}")
    lines.append("")
    lines.append(f"  n_trades           : {result.n_trades}  ({'pass' if result.trades_pass else 'fail'})")
    lines.append(f"  Sharpe             : {result.sharpe:.4f}")
    lines.append(f"  Profit factor      : {result.profit_factor:.4f}")
    lines.append(f"  PF 95% CI          : [{result.profit_factor_lo:.4f}, {result.profit_factor_hi:.4f}]  ({'pass' if result.pf_lo_pass else 'fail'})")
    lines.append(f"  DSR                : {result.dsr:.4f}  ({'pass' if result.dsr_pass else 'fail'})")
    lines.append(f"  PBO                : {result.pbo:.4f}  ({'pass' if result.pbo_pass else 'fail'})")
    lines.append(f"  DM stat            : {result.dm_stat:.4f}")
    lines.append(f"  DM p-value         : {result.dm_pvalue:.4f}  ({'pass' if result.dm_pass else 'fail'})")
    if result.failure_reasons:
        lines.append("")
        lines.append("Failure reasons:")
        for r in result.failure_reasons:
            lines.append(f"  - {r}")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "validate_trades_dataframe",
    "validate_backtest_artifact",
    "render_gate_report",
]
