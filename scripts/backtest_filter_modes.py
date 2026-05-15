"""Compare RegimeFilter modes side-by-side on existing replay trades.

Three modes tested:
  α   = ny_mode="all",      vol_pctl_max=0.75   (current default)
  γ   = ny_mode="high_vol", vol_pctl_max=0.75   (surgical NY)
  β   = ny_mode="off",      vol_pctl_max=0.75   (vol-only)
  off = ny_mode="off",      vol_pctl_max=None   (no filter)

Replays the 2363 trades in reports/audit/trades_combined.csv (XAU 7yr)
through each mode and reports n_test, PF_test, expectancy_test, sig/yr,
total_R_test.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.regime_filter import RegimeFilter


def metrics(r: pd.Series) -> dict:
    if len(r) == 0:
        return {"n": 0, "pf": np.nan, "win": np.nan, "exp": np.nan, "tot": 0.0}
    w = r[r > 0].sum(); l = -r[r < 0].sum()
    pf = w / l if l > 0 else float("inf")
    return {"n": len(r), "pf": float(pf) if pf != np.inf else 999.0,
            "win": float((r > 0).mean()), "exp": float(r.mean()),
            "tot": float(r.sum())}


def evaluate_mode(trades: pd.DataFrame, atr_series: pd.Series, label: str, **kw) -> dict:
    rf = RegimeFilter(**kw)
    keep = []
    for _, row in trades.iterrows():
        ts = str(row["entry_bar"])
        atr_hist = atr_series.loc[:row["entry_bar"]]
        keep.append(rf.evaluate(ts, atr_hist).allowed)
    sub = trades[keep]
    test = sub[sub["entry_bar"] >= "2023-01-01"]
    test_years = (sub["entry_bar"].max() - pd.Timestamp("2023-01-01")).days / 365.25
    m = metrics(test["r_multiple"])
    m["sig_yr"] = m["n"] / test_years if test_years > 0 else 0
    m["label"] = label
    m["filter_stats"] = rf.stats()
    return m


def main():
    df = pd.read_csv("data/XAU_15MIN_2019_2026.csv", parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print("enriching...")
    enr = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    atr = enr["ATR"]

    trades = pd.read_csv("reports/audit/trades_combined.csv")
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    print(f"{len(trades)} trades loaded")

    modes = [
        ("off no filter",        dict(ny_mode="off",      vol_pctl_max=None)),
        ("alpha ny=all+vol75",   dict(ny_mode="all",      vol_pctl_max=0.75)),
        ("beta  vol-only 75",    dict(ny_mode="off",      vol_pctl_max=0.75)),
        ("gamma ny=hv vol75",    dict(ny_mode="high_vol", vol_pctl_max=0.75)),
        ("gamma' ny=hv vol85",   dict(ny_mode="high_vol", vol_pctl_max=0.85)),
        ("gamma'' ny=hv volOff", dict(ny_mode="high_vol", vol_pctl_max=0.75)),
    ]
    print(f"\n{'mode':>20s}  {'n':>5s}  {'PF':>6s}  {'win':>5s}  {'exp':>7s}  {'sig/yr':>8s}  {'totR':>7s}  filter_stats")
    print("-" * 110)
    for label, kw in modes:
        r = evaluate_mode(trades, atr, label, **kw)
        print(f"{r['label']:>20s}  {r['n']:>5d}  {r['pf']:>6.3f}  {r['win']:>5.2f}  {r['exp']:>+7.3f}  {r['sig_yr']:>8.0f}  {r['tot']:>+7.2f}  ny={r['filter_stats']['dropped_ny']} vol={r['filter_stats']['dropped_vol']} ok={r['filter_stats']['allowed']}")


if __name__ == "__main__":
    main()
