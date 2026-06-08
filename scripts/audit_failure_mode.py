"""Diagnose WHY NY × Q4_high vol bucket loses −23R.

Hypothesis: SL = 2× ATR is too tight when ATR is itself in the top
quartile (Q4_high). During NY session, intraday range is widest →
stops get whipsawed before targets are reached.

Tests:
  1. Exit reason breakdown (target_reached / invalidated / time_expired)
     for NY × Q4_high vs profitable buckets
  2. R-multiple distribution: do losses cluster at exactly −1.0R (SL hit)?
  3. Bars held median per bucket
  4. Compare ATR at entry across buckets

Output: reports/failure_mode_audit.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine

SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]


def session_label(h):
    for lo, hi, n in SESSION_BINS:
        if lo <= h < hi:
            return n
    return "OffHours"


def fmt(v, spec=".3f"):
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:{spec}}"
    return str(v)


def bucket_metrics(sub):
    if len(sub) == 0:
        return {}
    r = sub["r_multiple"]
    exits = sub["exit_reason"].value_counts(normalize=True).to_dict()
    return {
        "n": len(sub),
        "PF": (r[r > 0].sum() / -r[r < 0].sum()) if (r < 0).any() else float("inf"),
        "win%": (r > 0).mean(),
        "exp_R": r.mean(),
        "median_R": r.median(),
        "median_bars": sub["bars_held"].median(),
        "% SL-loss (R<-0.95)": (r < -0.95).mean(),
        "% near-BE (|R|<0.2)": (r.abs() < 0.2).mean(),
        "% TP-win (R>1.5)": (r > 1.5).mean(),
        "exit_target": exits.get("target_reached", 0),
        "exit_invalid": exits.get("invalidated", 0),
        "exit_time": exits.get("time_expired", 0),
        "exit_score": exits.get("score_decayed", 0),
        "exit_regime": exits.get("regime_shifted", 0),
        "ATR_entry_med": sub["ATR"].median() if "ATR" in sub.columns else np.nan,
    }


def main():
    df = pd.read_csv("data/XAU_15MIN_2019_2026.csv", parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    enr = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enr["hour"] = enr.index.hour
    enr["session"] = enr["hour"].map(session_label)
    atr_pctl = enr["ATR"].fillna(0).rolling(30 * 96, min_periods=200).rank(pct=True)
    enr["ATR_PCTL"] = atr_pctl
    enr["ATR_Q"] = pd.qcut(atr_pctl.fillna(0.5), 4,
                           labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    trades = pd.read_csv("reports/audit/trades_combined.csv")
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feats = enr[["session", "ATR_Q", "ATR", "ATR_PCTL"]]
    m = trades.merge(feats, left_on="entry_bar", right_index=True, how="inner")
    m["set"] = np.where(m["entry_bar"] < "2023-01-01", "TRAIN", "TEST")
    m["bucket"] = m["session"].astype(str) + " × " + m["ATR_Q"].astype(str)

    md = ["# Failure-mode diagnostic: NY × Q4_high\n"]
    md.append("Hypothesis under test: SL = 2× ATR is too tight in high-vol regime → stops whipsawed.\n---\n")

    # 1. Per-bucket exit reason + R distribution
    buckets_of_interest = [
        "NY × Q4_high",       # the saigneur
        "London × Q3",        # winner
        "Asian × Q2",         # winner
        "London × Q4_high",   # control: high vol but not NY
        "NY × Q3",            # control: NY but not Q4_high
        "NY × Q2",            # control: NY low-vol
    ]

    md.append("## Per-bucket comparison (full sample, train+test)\n")
    md.append("| Bucket | n | PF | win% | exp_R | med_R | med_bars | %SL-loss | %near-BE | %TP-win | exit_target | exit_invalid | exit_time | ATR_med |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for b in buckets_of_interest:
        sub = m[m["bucket"] == b]
        x = bucket_metrics(sub)
        if not x:
            continue
        md.append(f"| **{b}** | {x['n']} | {fmt(x['PF'])} | {fmt(x['win%'])} | {fmt(x['exp_R'])} | {fmt(x['median_R'])} | {x['median_bars']:.0f} | {fmt(x['% SL-loss (R<-0.95)'])} | {fmt(x['% near-BE (|R|<0.2)'])} | {fmt(x['% TP-win (R>1.5)'])} | {fmt(x['exit_target'])} | {fmt(x['exit_invalid'])} | {fmt(x['exit_time'])} | {fmt(x['ATR_entry_med'], '.2f')} |")

    # 2. SL-hit rate per ATR_Q
    md.append("\n## SL-hit rate by ATR quartile (all sessions)\n")
    md.append("| ATR_Q | n | %SL-loss (R<-0.95) | %TP-win (R>1.5) | PF | exp_R |")
    md.append("|---|---|---|---|---|---|")
    for q in ["Q1_low", "Q2", "Q3", "Q4_high"]:
        sub = m[m["ATR_Q"] == q]
        x = bucket_metrics(sub)
        md.append(f"| {q} | {x['n']} | {fmt(x['% SL-loss (R<-0.95)'])} | {fmt(x['% TP-win (R>1.5)'])} | {fmt(x['PF'])} | {fmt(x['exp_R'])} |")

    # 3. R distribution histogram (text) for NY × Q4_high
    md.append("\n## R-multiple histogram — NY × Q4_high losers\n")
    losers = m[(m["bucket"] == "NY × Q4_high") & (m["r_multiple"] < 0)]["r_multiple"]
    bins = [-1.5, -1.0, -0.8, -0.5, -0.2, 0]
    hist = pd.cut(losers, bins=bins, include_lowest=True).value_counts().sort_index()
    md.append("| R bucket | count | % of losers |")
    md.append("|---|---|---|")
    for k, v in hist.items():
        md.append(f"| {k} | {v} | {v/len(losers):.1%} |")

    # 4. Average ATR at entry per bucket
    md.append("\n## ATR at entry per bucket (validates Q4_high IS high vol)\n")
    md.append("| Bucket | ATR median | ATR mean | ATR_PCTL median |")
    md.append("|---|---|---|---|")
    for b in buckets_of_interest:
        sub = m[m["bucket"] == b]
        if len(sub) == 0:
            continue
        md.append(f"| {b} | {sub['ATR'].median():.2f} | {sub['ATR'].mean():.2f} | {sub['ATR_PCTL'].median():.2f} |")

    # 5. Verdict
    ny_q4 = m[m["bucket"] == "NY × Q4_high"]
    london_q3 = m[m["bucket"] == "London × Q3"]
    sl_rate_ny = (ny_q4["r_multiple"] < -0.95).mean()
    sl_rate_lon = (london_q3["r_multiple"] < -0.95).mean()
    md.append(f"\n## Verdict\n")
    md.append(f"- NY × Q4_high SL-hit rate: **{sl_rate_ny:.1%}**")
    md.append(f"- London × Q3 SL-hit rate (control winner): **{sl_rate_lon:.1%}**")
    md.append(f"- ATR median NY×Q4_high vs London×Q3: {ny_q4['ATR'].median():.2f} vs {london_q3['ATR'].median():.2f}")
    if sl_rate_ny > sl_rate_lon * 1.3:
        md.append("\n→ **HYPOTHÈSE CONFIRMÉE** — SL hit rate disproportionné en NY×Q4. SL fixe à 2×ATR insuffisant en haute vol.")
    elif sl_rate_ny > sl_rate_lon:
        md.append("\n→ **Hypothèse partiellement confirmée** — SL un peu plus touché en NY×Q4 mais pas dramatiquement. Cherche aussi spread/news.")
    else:
        md.append("\n→ **Hypothèse rejetée** — le SL n'est pas le problème. Le saigneur a une autre cause (regime, direction, news...).")

    out = Path("reports/failure_mode_audit.md")
    out.write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {out}")
    print(f"\nNY×Q4 SL-hit: {sl_rate_ny:.1%}, London×Q3 SL-hit: {sl_rate_lon:.1%}")


if __name__ == "__main__":
    main()
