"""Combine subset filters and measure resulting strategy PF.

Test 4 candidate filter rules + their combinations:
  R1: skip ATR_Q4_high (vol regime filter)
  R2: skip NY session
  R3: skip Tuesday (dow=1)
  R4: long-only (drop SHORT)

For each combination, compute train/test PF, win%, total_R, n, signals/year.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine

SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]


def session_label(hour):
    for lo, hi, n in SESSION_BINS:
        if lo <= hour < hi:
            return n
    return "OffHours"


def metrics(r):
    if len(r) == 0:
        return {"n": 0, "winrate": np.nan, "exp": np.nan, "pf": np.nan, "total_r": np.nan}
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    return {
        "n": int(len(r)),
        "winrate": float((r > 0).mean()),
        "exp": float(r.mean()),
        "pf": float(pf) if pf != np.inf else 999.0,
        "total_r": float(r.sum()),
    }


def fmt(v, spec=".3f"):
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:{spec}}"
    return str(v)


def main():
    df = pd.read_csv("data/XAU_15MIN_2019_2025.csv", parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enriched["hour"] = enriched.index.hour
    enriched["dow"] = enriched.index.dayofweek
    enriched["session"] = enriched["hour"].map(session_label)
    atr = enriched["ATR"].fillna(0)
    enriched["ATR_PCTL"] = atr.rolling(30 * 96, min_periods=200).rank(pct=True)
    enriched["ATR_Q"] = pd.qcut(enriched["ATR_PCTL"].fillna(0.5), 4,
                                labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    trades = pd.read_csv("reports/audit/trades_combined.csv")
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feats = enriched[["session", "dow", "hour", "ATR_Q"]]
    m = trades.merge(feats, left_on="entry_bar", right_index=True, how="inner")
    m["set"] = np.where(m["entry_bar"] < "2023-01-01", "TRAIN", "TEST")

    # Period spans (years) for signals/year
    train_years = (pd.Timestamp("2023-01-01") - pd.Timestamp("2019-01-01")).days / 365.25
    test_years = (m["entry_bar"].max() - pd.Timestamp("2023-01-01")).days / 365.25

    rules = {
        "R1_skip_Q4_high": lambda d: d["ATR_Q"] != "Q4_high",
        "R2_skip_NY": lambda d: d["session"] != "NY",
        "R3_skip_Tuesday": lambda d: d["dow"] != 1,
        "R4_long_only": lambda d: d["direction"] == "LONG",
    }

    md = ["# Filter Strategy Audit — XAU/USD M15 (Chantier 3 step 2)\n"]
    md.append(f"Train years: {train_years:.2f} | Test years: {test_years:.2f}")
    md.append("Acceptance: PF_test ≥ 1.30 AND n_test ≥ 100 AND PF_test stable vs PF_train (same sign of edge, |drop| < 0.20)\n---\n")

    md.append("| Rules | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | sig/yr_test | ✅ |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")

    rule_names = list(rules.keys())
    all_combos = [()] + [c for k in range(1, len(rule_names) + 1) for c in combinations(rule_names, k)]
    rows = []
    for combo in all_combos:
        mask = pd.Series(True, index=m.index)
        for rn in combo:
            mask &= rules[rn](m)
        sub = m[mask]
        tr = metrics(sub.loc[sub["set"] == "TRAIN", "r_multiple"])
        te = metrics(sub.loc[sub["set"] == "TEST", "r_multiple"])
        sig_per_yr = te["n"] / test_years if test_years > 0 else 0
        stable = (
            te["n"] >= 100
            and not np.isnan(te["pf"])
            and te["pf"] >= 1.30
            and not np.isnan(tr["pf"])
            and (tr["pf"] - 1) * (te["pf"] - 1) > 0
            and abs(tr["pf"] - te["pf"]) < 0.20
        )
        name = " + ".join(combo) if combo else "ALL (no filter)"
        rows.append({
            "name": name,
            "row": f"| {name} | {tr['n']} | {fmt(tr['pf'])} | {te['n']} | {fmt(te['pf'])} | {fmt(te['winrate'])} | {fmt(te['exp'])} | {fmt(te['total_r'])} | {sig_per_yr:.0f} | {'✅' if stable else ''} |",
            "pf_test": te["pf"] if not np.isnan(te["pf"]) else 0,
            "n_test": te["n"],
            "stable": stable,
        })

    rows.sort(key=lambda r: -r["pf_test"])
    for r in rows:
        md.append(r["row"])

    md.append("\n## Top stable filters\n")
    stable = [r for r in rows if r["stable"]]
    if stable:
        for r in stable[:5]:
            md.append(f"- **{r['name']}** — PF_test={r['pf_test']:.3f}, n={r['n_test']}")
    else:
        md.append("**No filter combination passes stability**.")

    out = Path("reports/feature_filter_audit.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
