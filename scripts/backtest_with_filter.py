"""End-to-end backtest verifying RegimeFilter delivers the audited PF lift.

Replays the 2363 trades from baseline_full and applies RegimeFilter to
each trade's (entry_bar, ATR_at_entry) tuple. Reports filter drop counts
and resulting PF.

This is a sanity check that the production filter behaves identically to
the empirical analysis in scripts/audit_subset_edge.py.
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
        return {"n": 0, "winrate": np.nan, "exp": np.nan, "pf": np.nan, "total_r": 0.0}
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    return {"n": len(r), "winrate": float((r > 0).mean()),
            "exp": float(r.mean()), "pf": float(pf) if pf != np.inf else 999.0,
            "total_r": float(r.sum())}


def main():
    df = pd.read_csv("data/XAU_15MIN_2019_2025.csv", parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print(f"loaded {len(df)} bars")
    enr = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    print("enriched")

    trades = pd.read_csv("reports/audit/trades_combined.csv")
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    print(f"loaded {len(trades)} trades")

    rf = RegimeFilter()  # prod defaults
    print(f"filter: skip_ny={rf.skip_ny}, vol_pctl_max={rf.vol_pctl_max}")

    decisions = []
    for _, row in trades.iterrows():
        ts = str(row["entry_bar"])
        # Slice ATR series up to AND INCLUDING entry bar (no lookahead)
        atr_history = enr.loc[:row["entry_bar"], "ATR"]
        d = rf.evaluate(ts, atr_history)
        decisions.append(d.allowed)

    trades["allowed"] = decisions
    kept = trades[trades["allowed"]]
    dropped = trades[~trades["allowed"]]

    train = trades[trades["entry_bar"] < "2023-01-01"]
    test = trades[trades["entry_bar"] >= "2023-01-01"]
    train_kept = kept[kept["entry_bar"] < "2023-01-01"]
    test_kept = kept[kept["entry_bar"] >= "2023-01-01"]

    print(f"\n=== filter stats ===")
    print(f"  total trades:   {len(trades)}")
    print(f"  kept:           {len(kept)} ({len(kept)/len(trades)*100:.1f}%)")
    print(f"  dropped:        {len(dropped)} ({len(dropped)/len(trades)*100:.1f}%)")
    print(f"  filter counters: {rf.stats()}")

    print(f"\n=== UNFILTERED performance ===")
    print(f"  TRAIN: {metrics(train['r_multiple'])}")
    print(f"  TEST:  {metrics(test['r_multiple'])}")

    print(f"\n=== FILTERED performance ===")
    print(f"  TRAIN: {metrics(train_kept['r_multiple'])}")
    print(f"  TEST:  {metrics(test_kept['r_multiple'])}")

    test_pf_before = metrics(test["r_multiple"])["pf"]
    test_pf_after = metrics(test_kept["r_multiple"])["pf"]
    print(f"\n=== VERDICT ===")
    print(f"  PF_test:  {test_pf_before:.3f} → {test_pf_after:.3f} (Δ {test_pf_after-test_pf_before:+.3f})")
    target = 1.30
    print(f"  Acceptance: PF_test ≥ {target} → {'✅ PASS' if test_pf_after >= target else '❌ FAIL'}")


if __name__ == "__main__":
    main()
