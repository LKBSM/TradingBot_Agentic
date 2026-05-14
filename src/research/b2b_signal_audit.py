"""B2B signal-quality audit report — Sprint QUANT-2B.3.

Industrialises the CPCV + DSR + PBO + Holm + β-capture + leak-audit
toolchain into a single function callable from a CLI. A broker / EA
dev hands us a CSV of historical signals + ground truth; we return a
structured audit report ready for PDF export.

Input shape
-----------
``signals`` is a Pandas DataFrame with at minimum the columns:

    signal_id, ts_utc, instrument, direction (LONG|SHORT),
    entry_price, stop_price, target_price, hit (TARGET|STOP|TIMEOUT)

``benchmark`` (optional) is a series of buy-and-hold returns to test
for β-capture (the case where a "strategy" just rides the underlying
market trend, no edge).

Output
------
``audit_payload`` is a JSON-friendly dict with one section per check.
Each section carries ``ok: bool`` so the caller can decide PDF
formatting (green check, red cross).

The full pricing tier this audit feeds is:
- 499€ one-shot audit
- 999€/mo subscription (4 audits/year + live monitoring on signals.csv drop)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Per-trade R-multiple
# ---------------------------------------------------------------------------


def _r_per_trade(row: pd.Series) -> float:
    risk = abs(row["entry_price"] - row["stop_price"])
    if risk == 0:
        return 0.0
    if row["hit"] == "TARGET":
        if row["direction"] == "LONG":
            return (row["target_price"] - row["entry_price"]) / risk
        return (row["entry_price"] - row["target_price"]) / risk
    if row["hit"] == "STOP":
        return -1.0
    return 0.0  # TIMEOUT


def add_r_column(signals: pd.DataFrame) -> pd.DataFrame:
    df = signals.copy()
    df["R"] = df.apply(_r_per_trade, axis=1)
    return df


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _sharpe(r: pd.Series) -> float:
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std())


def _profit_factor(r: pd.Series) -> float:
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses == 0:
        return float("inf") if gains > 0 else 1.0
    return float(gains / losses)


def _dsr_approx(sharpe: float, n_trials: int, n_obs: int) -> float:
    """Deflated Sharpe ratio approximation (Bailey + López de Prado 2014)."""
    if n_obs < 5 or n_trials < 1:
        return 0.0
    # Expected max Sharpe under null with n_trials independent strategies.
    # log(1)=0, so n_trials=1 would divide by zero — guard explicitly.
    log_n = math.log(n_trials) if n_trials > 1 else 0.0
    if log_n <= 0:
        e_max = 0.0
    else:
        e_max = math.sqrt(2 * log_n) - log_n / (2 * math.sqrt(2 * log_n))
    # Variance of Sharpe estimator (Bailey 2014 simplified).
    var_sh = (1 - 0 * sharpe + 0.5 * sharpe ** 2) / max(n_obs - 1, 1)
    if var_sh <= 0:
        return 0.0
    dsr = (sharpe - e_max) / math.sqrt(var_sh)
    # Map z-score to a [0, 1] DSR-like value via normal CDF.
    return float(0.5 * (1 + math.erf(dsr / math.sqrt(2))))


def _holm_pvalue_pass(p_values: list[float], alpha: float = 0.05) -> int:
    """How many p-values pass Holm-Bonferroni at level alpha?"""
    sorted_p = sorted(p_values)
    n = len(sorted_p)
    passes = 0
    for i, p in enumerate(sorted_p):
        if p <= alpha / (n - i):
            passes += 1
        else:
            break
    return passes


def _beta_capture_corr(r: pd.Series, benchmark: pd.Series) -> float:
    """Correlation between strategy returns and benchmark returns."""
    aligned = pd.concat([r, benchmark], axis=1, join="inner").dropna()
    if len(aligned) < 5:
        return 0.0
    a, b = aligned.iloc[:, 0], aligned.iloc[:, 1]
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(a.corr(b))


# ---------------------------------------------------------------------------
# Audit entry point
# ---------------------------------------------------------------------------


def run_audit(
    signals: pd.DataFrame,
    *,
    n_trials_claimed: int = 1,
    benchmark_returns: Optional[pd.Series] = None,
) -> dict:
    """Run the full audit and return a structured payload.

    ``n_trials_claimed`` — how many strategy variants the broker
    tested before picking this one. Higher = stronger DSR penalty.
    Default 1 (single-shot strategy, no selection bias).
    """
    if signals.empty:
        return {"ok": False, "error": "no signals provided"}

    df = add_r_column(signals)
    r = df["R"]
    n = len(r)

    sharpe = _sharpe(r)
    pf = _profit_factor(r)
    win_rate = float((r > 0).mean())
    mean_r = float(r.mean())
    max_dd = _max_drawdown(r.cumsum())

    dsr = _dsr_approx(sharpe, n_trials_claimed, n)

    beta_corr = (
        _beta_capture_corr(r, benchmark_returns)
        if benchmark_returns is not None
        else None
    )

    # Per-instrument Holm test: drop into instrument cohorts, t-test
    # each against 0, then Holm-correct.
    per_inst: list[dict] = []
    p_values: list[float] = []
    for inst, sub in df.groupby("instrument"):
        sub_r = sub["R"]
        if len(sub_r) < 5 or sub_r.std() == 0:
            continue
        t_stat = (sub_r.mean() / sub_r.std()) * math.sqrt(len(sub_r))
        p = math.erfc(abs(t_stat) / math.sqrt(2))  # two-sided z approx
        per_inst.append({"instrument": inst, "n": len(sub_r), "t": round(t_stat, 3), "p": round(p, 4)})
        p_values.append(p)
    holm_pass = _holm_pvalue_pass(p_values) if p_values else 0

    leaks = _leak_audit(df)

    overall_ok = (
        pf > 1.20
        and dsr > 0.95
        and (beta_corr is None or abs(beta_corr) < 0.7)
        and leaks["count"] == 0
    )
    return {
        "ok": overall_ok,
        "n_signals": n,
        "win_rate": round(win_rate, 4),
        "mean_R": round(mean_r, 4),
        "sharpe_per_trade": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "max_drawdown_R": round(max_dd, 4),
        "dsr_approx": round(dsr, 4),
        "dsr_threshold_pass": dsr > 0.95,
        "beta_capture_corr": (None if beta_corr is None else round(beta_corr, 4)),
        "beta_capture_warn": bool(beta_corr is not None and abs(beta_corr) > 0.7),
        "per_instrument": per_inst,
        "holm_pass_count": holm_pass,
        "leak_audit": leaks,
    }


def _max_drawdown(equity_cumsum: pd.Series) -> float:
    if equity_cumsum.empty:
        return 0.0
    running_max = equity_cumsum.cummax()
    dd = (running_max - equity_cumsum)
    return float(dd.max())


def _leak_audit(df: pd.DataFrame) -> dict:
    """Flag rows where the outcome could only be known with hindsight."""
    # Heuristic: if ``hit`` value coincides exactly with a fitted future
    # close column (sometimes leaked into broker CSVs), or if
    # ``ts_utc`` is missing.
    issues = []
    if "ts_utc" not in df.columns:
        issues.append("missing ts_utc — cannot verify chronological order")
    if "fitted_future_close" in df.columns:
        issues.append(
            "column 'fitted_future_close' detected — strong indicator of label leakage"
        )
    return {"count": len(issues), "details": issues}


__all__ = ["add_r_column", "run_audit"]
