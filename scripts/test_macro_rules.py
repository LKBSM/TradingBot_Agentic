"""Test pure macro rules strategy — Bridgewater-style.

No ML, no scoring. Just bank-canonical rules :

- LONG XAU when DXY weak (z < threshold) AND real rates falling (z < threshold)
- SHORT XAU when DXY strong (z > threshold) AND real rates rising (z > threshold)
- HOLD otherwise

This is what every gold trading desk has built since the 1970s.
We test multiple thresholds and compare to buy-and-hold.
"""

from __future__ import annotations

import argparse
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--horizon", type=int, default=96,
                   help="Holding period in M15 bars (default 96 = 1 day)")
    args = p.parse_args()

    from src.intelligence.macro_factors import MacroFactorExtractor

    print("=== Loading XAU + macro ===")
    ohlcv = load_xau()
    print(f"Loaded {len(ohlcv)} bars : {ohlcv.index[0]} → {ohlcv.index[-1]}")

    ext = MacroFactorExtractor()
    macro = ext.extract(ohlcv.index)
    df = pd.concat([ohlcv["Close"], macro], axis=1).ffill()

    # Build forward returns
    df["fwd_ret"] = np.log(df["Close"].shift(-args.horizon) / df["Close"])

    # Z-score thresholds to test
    grid = []
    for dxy_th in [0.5, 1.0, 1.5]:
        for rates_th in [0.0, 0.5, 1.0]:
            grid.append((dxy_th, rates_th))

    print("\n=== Bridgewater-style macro rules grid ===\n")
    print(f"{'dxy_z_th':>10} | {'rates_z_th':>10} | {'n_long':>7} | {'n_short':>7} | "
          f"{'sharpe_strat':>12} | {'sharpe_BH':>10} | {'IR vs BH':>10}")
    print("-" * 95)

    bh_ret = df["fwd_ret"].dropna()
    bh_sharpe = bh_ret.mean() / bh_ret.std() * np.sqrt(252 * 24 * 60 / (args.horizon * 15))

    best = None
    results = []
    for dxy_th, rates_th in grid:
        signal = pd.Series(0, index=df.index, dtype=float)
        # Long when dollar weak + real rates falling
        signal[(df["dxy_z"] < -dxy_th) & (df["real_10y_z"] < -rates_th)] = 1
        # Short when dollar strong + real rates rising
        signal[(df["dxy_z"] > dxy_th) & (df["real_10y_z"] > rates_th)] = -1

        strat_ret = (signal * df["fwd_ret"]).dropna()
        nonzero = strat_ret[signal[strat_ret.index] != 0]
        n_long = int((signal == 1).sum())
        n_short = int((signal == -1).sum())

        if len(nonzero) < 50 or nonzero.std() == 0:
            sharpe = float("nan")
        else:
            sharpe = float(nonzero.mean() / nonzero.std() * np.sqrt(252 * 24 * 60 / (args.horizon * 15)))

        ir = sharpe - bh_sharpe if not np.isnan(sharpe) else float("nan")
        print(f"{dxy_th:>10.2f} | {rates_th:>10.2f} | {n_long:>7} | {n_short:>7} | "
              f"{sharpe:>12.3f} | {bh_sharpe:>10.3f} | {ir:>10.3f}")
        results.append({"dxy_th": dxy_th, "rates_th": rates_th, "sharpe": sharpe, "IR": ir,
                        "n_long": n_long, "n_short": n_short})
        if best is None or (not np.isnan(sharpe) and sharpe > best["sharpe"]):
            best = results[-1]

    print(f"\nBest cell : dxy_z<{best['dxy_th']:.2f}, rates_z<{best['rates_th']:.2f} → "
          f"Sharpe {best['sharpe']:.3f} (IR {best['IR']:+.3f} vs BH {bh_sharpe:.3f})")

    # Save
    out_dir = ROOT / "reports" / "factor_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_dir / "macro_rules_grid.csv", index=False)
    print(f"Saved {out_dir / 'macro_rules_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
