"""Drawdown comparison — active strategy vs buy-and-hold.

Commercial value of an indicator isn't always to beat passive returns
on a trending asset. The institutional value is often :

1. **Smaller max drawdown** during regime shifts
2. **Lower exposure** during high-vol periods (capital preservation)
3. **Better Sortino** (downside-only risk adjustment)

This script measures these on the best macro-rule cell vs B&H XAU.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_xau() -> pd.DataFrame:
    df = pd.read_csv(ROOT / "data" / "XAU_15MIN_2019_2026.csv", parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns
                       if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    return df


def max_drawdown(equity: pd.Series) -> tuple[float, float]:
    """Return (max_dd, max_dd_pct)."""
    running_max = equity.expanding().max()
    dd = equity - running_max
    return float(dd.min()), float((dd / running_max).min())


def main() -> int:
    from src.intelligence.macro_factors import MacroFactorExtractor

    print("=== Drawdown comparison ===\n")
    ohlcv = load_xau()
    ext = MacroFactorExtractor()
    macro = ext.extract(ohlcv.index)
    df = pd.concat([ohlcv["Close"], macro], axis=1).ffill()
    df["ret"] = np.log(df["Close"] / df["Close"].shift(1))

    # Buy-and-hold (always long)
    bh_eq = (1 + df["ret"]).cumprod()
    bh_dd, bh_dd_pct = max_drawdown(bh_eq)
    bh_sharpe = df["ret"].mean() / df["ret"].std() * np.sqrt(252 * 96)
    bh_sortino_denom = df["ret"][df["ret"] < 0].std()
    bh_sortino = df["ret"].mean() / bh_sortino_denom * np.sqrt(252 * 96) if bh_sortino_denom > 0 else 0

    # Best macro-rules signal (dxy_z<-0.5, rates_z<-1.0 for LONG, opposite for SHORT)
    sig = pd.Series(0, index=df.index, dtype=float)
    sig[(df["dxy_z"] < -0.5) & (df["real_10y_z"] < -1.0)] = 1
    sig[(df["dxy_z"] > 0.5) & (df["real_10y_z"] > 1.0)] = -1

    strat_ret = sig.shift(1) * df["ret"]  # shift to avoid look-ahead
    strat_eq = (1 + strat_ret.fillna(0)).cumprod()
    strat_dd, strat_dd_pct = max_drawdown(strat_eq)
    strat_sharpe = strat_ret.mean() / strat_ret.std() * np.sqrt(252 * 96) if strat_ret.std() > 0 else 0
    strat_sortino_denom = strat_ret[strat_ret < 0].std()
    strat_sortino = strat_ret.mean() / strat_sortino_denom * np.sqrt(252 * 96) if strat_sortino_denom > 0 else 0

    # Half exposure variant: long only when long signal, else flat (no shorts)
    sig_long = (sig == 1).astype(float)
    half_ret = sig_long.shift(1) * df["ret"]
    half_eq = (1 + half_ret.fillna(0)).cumprod()
    half_dd, half_dd_pct = max_drawdown(half_eq)
    half_sharpe = half_ret.mean() / half_ret.std() * np.sqrt(252 * 96) if half_ret.std() > 0 else 0

    # Hedge variant: B&H minus when short signal active (reduce exposure to 50% when SHORT fires)
    hedge_pos = pd.Series(1.0, index=df.index)
    hedge_pos[sig.shift(1) == -1] = 0.5
    hedge_ret = hedge_pos * df["ret"]
    hedge_eq = (1 + hedge_ret.fillna(0)).cumprod()
    hedge_dd, hedge_dd_pct = max_drawdown(hedge_eq)
    hedge_sharpe = hedge_ret.mean() / hedge_ret.std() * np.sqrt(252 * 96) if hedge_ret.std() > 0 else 0

    print(f"{'Strategy':>35} | {'Final eq':>10} | {'Sharpe':>8} | {'Sortino':>8} | {'MaxDD%':>8} | {'Exposed%':>10}")
    print("-" * 110)
    exp_bh = 100.0
    exp_strat = float((sig != 0).mean() * 100)
    exp_half = float((sig_long == 1).mean() * 100)
    exp_hedge = float((hedge_pos < 1).mean() * 100)
    print(f"{'Buy-and-Hold (always long)':>35} | {float(bh_eq.iloc[-1]):>10.3f} | {bh_sharpe:>8.3f} | {bh_sortino:>8.3f} | {bh_dd_pct*100:>7.2f}% | {exp_bh:>10.1f}%")
    print(f"{'Macro long/short (best cell)':>35} | {float(strat_eq.iloc[-1]):>10.3f} | {strat_sharpe:>8.3f} | {strat_sortino:>8.3f} | {strat_dd_pct*100:>7.2f}% | {exp_strat:>10.1f}%")
    print(f"{'Macro long-only (no shorts)':>35} | {float(half_eq.iloc[-1]):>10.3f} | {half_sharpe:>8.3f} | {'n/a':>8} | {half_dd_pct*100:>7.2f}% | {exp_half:>10.1f}%")
    print(f"{'B&H + macro hedge (50% on short)':>35} | {float(hedge_eq.iloc[-1]):>10.3f} | {hedge_sharpe:>8.3f} | {'n/a':>8} | {hedge_dd_pct*100:>7.2f}% | {exp_hedge:>10.1f}%")

    # Annualised returns for context
    n_years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = lambda eq: (float(eq.iloc[-1]) ** (1 / n_years) - 1) * 100
    print(f"\n CAGR :")
    print(f"   Buy-and-Hold      : {cagr(bh_eq):+.2f}% /yr")
    print(f"   Macro L/S         : {cagr(strat_eq):+.2f}% /yr")
    print(f"   Macro long-only   : {cagr(half_eq):+.2f}% /yr")
    print(f"   B&H + hedge       : {cagr(hedge_eq):+.2f}% /yr")

    # Save
    out_dir = ROOT / "reports" / "factor_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    eq_df = pd.DataFrame({
        "bh": bh_eq,
        "macro_LS": strat_eq,
        "macro_long_only": half_eq,
        "bh_hedge": hedge_eq,
    })
    eq_df.to_csv(out_dir / "equity_curves.csv")
    print(f"\n Saved {out_dir / 'equity_curves.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
