"""EURUSD subset edge audit — adapted from audit_subset_edge.py.

Differences from XAU version:
  * Uses EURUSD's session boundaries (NY = 12:00–21:00 UTC, not 13:00).
  * Loads `data/EURUSD_15MIN_2019_2025.csv` and `reports/eurusd_trades.csv`.
  * Output → `reports/eurusd_subset_audit.md`.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine


# EURUSD session table — see InstrumentConfig in volatility_forecaster.py:65-71
SESSION_BINS = [
    (0, 7, "Asian"),
    (7, 12, "London"),
    (12, 16, "NY_overlap"),
    (16, 21, "NY_afternoon"),
    (21, 24, "OffHours"),
]


def session_label(h: int) -> str:
    for lo, hi, n in SESSION_BINS:
        if lo <= h < hi:
            return n
    return "OffHours"


# Combine NY_overlap + NY_afternoon into a single "NY" supercategory for
# parity with the XAU report.
def session_super(s: str) -> str:
    if s.startswith("NY"):
        return "NY"
    return s


def metrics(r: pd.Series) -> dict:
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


def fmt(v, spec=".3f") -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:{spec}}"
    return str(v)


def main() -> int:
    csv_path = Path("data/EURUSD_15MIN_2019_2025.csv")
    trades_path = Path("reports/eurusd_trades.csv")
    out_path = Path("reports/eurusd_subset_audit.md")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print(f"[eur] enriching {len(df)} bars...")
    enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enriched["hour"] = enriched.index.hour
    enriched["dow"] = enriched.index.dayofweek
    enriched["session_fine"] = enriched["hour"].map(session_label)
    enriched["session"] = enriched["session_fine"].map(session_super)
    atr = enriched["ATR"].fillna(0)
    enriched["ATR_PCTL"] = atr.rolling(30 * 96, min_periods=200).rank(pct=True)
    enriched["ATR_Q"] = pd.qcut(
        enriched["ATR_PCTL"].fillna(0.5), 4,
        labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )

    trades = pd.read_csv(trades_path)
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feats = enriched[["session", "session_fine", "dow", "hour", "ATR_PCTL", "ATR_Q"]]
    merged = trades.merge(feats, left_on="entry_bar", right_index=True, how="inner")
    merged["set"] = np.where(merged["entry_bar"] < "2023-01-01", "TRAIN", "TEST")
    print(f"[eur] {len(merged)} trades joined "
          f"({(merged['set']=='TRAIN').sum()} train, "
          f"{(merged['set']=='TEST').sum()} test)")

    train_years = (pd.Timestamp("2023-01-01") - pd.Timestamp("2019-01-01")).days / 365.25
    test_years = (merged["entry_bar"].max() - pd.Timestamp("2023-01-01")).days / 365.25

    md = ["# EURUSD M15 — Subset Edge Audit\n"]
    md.append(f"**Trades**: {len(merged)} | Train < 2023-01-01 ({train_years:.2f} yr) | Test ≥ 2023-01-01 ({test_years:.2f} yr)")
    md.append("**Acceptance**: PF_test ≥ 1.30 AND n_test ≥ 100 AND sign(PF_train-1) = sign(PF_test-1)\n")
    md.append("**NB**: EURUSD NY session = 12:00–21:00 UTC (XAU uses 13:00–21:00).\n")
    md.append("---\n")

    overall = {
        "TRAIN": metrics(merged.loc[merged["set"] == "TRAIN", "r_multiple"]),
        "TEST": metrics(merged.loc[merged["set"] == "TEST", "r_multiple"]),
    }
    md.append("## Overall\n")
    for k, v in overall.items():
        md.append(f"- {k}: n={v['n']}, PF={fmt(v['pf'])}, win={fmt(v['winrate'])}, "
                  f"exp={fmt(v['exp'])}, total_R={fmt(v['total_r'])}")
    md.append("")

    def split_table(grouper_cols, title):
        rows = []
        for keys, sub in merged.groupby(grouper_cols, observed=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            tr = metrics(sub.loc[sub["set"] == "TRAIN", "r_multiple"])
            te = metrics(sub.loc[sub["set"] == "TEST", "r_multiple"])
            stable = (
                te["n"] >= 100
                and not np.isnan(te["pf"])
                and te["pf"] >= 1.30
                and not np.isnan(tr["pf"])
                and (tr["pf"] - 1) * (te["pf"] - 1) > 0
            )
            row = {col: k for col, k in zip(grouper_cols, keys)}
            row.update({
                "n_train": tr["n"], "PF_train": fmt(tr["pf"]),
                "n_test": te["n"], "PF_test": fmt(te["pf"]),
                "win_test": fmt(te["winrate"]), "exp_test": fmt(te["exp"]),
                "total_R_test": fmt(te["total_r"]),
                "✅": "✅" if stable else "",
            })
            rows.append(row)
        rows.sort(key=lambda r: -float(r["PF_test"]) if r["PF_test"] != "—" else 0)
        cols = grouper_cols + ["n_train", "PF_train", "n_test", "PF_test",
                               "win_test", "exp_test", "total_R_test", "✅"]
        out = [f"\n## {title}\n"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(out)

    md.append(split_table(["direction"], "By direction"))
    md.append(split_table(["session"], "By session (NY = NY_overlap + NY_afternoon)"))
    md.append(split_table(["session_fine"], "By session (fine: NY split into overlap/afternoon)"))
    md.append(split_table(["ATR_Q"], "By ATR quartile"))
    md.append(split_table(["dow"], "By day of week"))
    md.append(split_table(["session", "ATR_Q"], "By session × ATR_Q (the canonical XAU breakdown)"))
    md.append(split_table(["session", "direction"], "By session × direction"))

    # Filter combination sweep — same rules as XAU
    md.append("\n## Filter combinations (sweep)\n")
    md.append("Acceptance: PF_test ≥ 1.30 AND n_test ≥ 100 AND PF stable train↔test\n")

    rules = {
        "R1_skip_Q4_high":  lambda d: d["ATR_Q"] != "Q4_high",
        "R2_skip_NY":       lambda d: d["session"] != "NY",
        "R3_skip_NY_Q4":    lambda d: ~((d["session"] == "NY") & (d["ATR_Q"] == "Q4_high")),
        "R4_long_only":     lambda d: d["direction"] == "LONG",
        "R5_skip_OffHours": lambda d: d["session"] != "OffHours",
    }

    md.append("| Rules | n_train | PF_train | n_test | PF_test | win_test | exp_test | total_R_test | sig/yr_test | ✅ |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    rule_names = list(rules.keys())
    all_combos = [()] + [c for k in range(1, len(rule_names) + 1) for c in combinations(rule_names, k)]
    rows = []
    for combo in all_combos:
        mask = pd.Series(True, index=merged.index)
        for rn in combo:
            mask &= rules[rn](merged)
        sub = merged[mask]
        tr = metrics(sub.loc[sub["set"] == "TRAIN", "r_multiple"])
        te = metrics(sub.loc[sub["set"] == "TEST", "r_multiple"])
        sig_per_yr = te["n"] / test_years if test_years > 0 else 0
        stable = (
            te["n"] >= 100 and not np.isnan(te["pf"]) and te["pf"] >= 1.30
            and not np.isnan(tr["pf"]) and (tr["pf"] - 1) * (te["pf"] - 1) > 0
        )
        name = " + ".join(combo) if combo else "ALL (no filter)"
        rows.append({
            "name": name, "tr": tr, "te": te, "sig_per_yr": sig_per_yr, "stable": stable,
        })
    # Print sorted by PF_test desc
    rows.sort(key=lambda r: -r["te"]["pf"] if not np.isnan(r["te"]["pf"]) else 0)
    for r in rows:
        md.append(
            f"| {r['name']} | {r['tr']['n']} | {fmt(r['tr']['pf'])} | "
            f"{r['te']['n']} | {fmt(r['te']['pf'])} | {fmt(r['te']['winrate'])} | "
            f"{fmt(r['te']['exp'])} | {fmt(r['te']['total_r'])} | "
            f"{r['sig_per_yr']:.0f} | {'✅' if r['stable'] else ''} |"
        )

    stable_combos = [r for r in rows if r["stable"]]
    md.append("\n## Top 3 stable filter combos\n")
    if stable_combos:
        for r in stable_combos[:3]:
            md.append(f"- **{r['name']}** — PF_test={r['te']['pf']:.3f}, "
                      f"n_test={r['te']['n']}, sig/yr={r['sig_per_yr']:.0f}, "
                      f"total_R_test={r['te']['total_r']:+.2f}")
    else:
        md.append("**No filter combo passes the stability gate.** EURUSD does NOT have a transferable XAU-style edge under this config.")

    # Verdict
    md.append("\n## Verdict\n")
    best = max(rows, key=lambda r: r["te"]["pf"] if not np.isnan(r["te"]["pf"]) else -1)
    best_pf = best["te"]["pf"]
    best_n = best["te"]["n"]
    best_sig = best["sig_per_yr"]
    if stable_combos and best_pf >= 1.40 and best_n >= 200:
        verdict = (f"✅ **ADD to product** — best stable filter `{best['name']}` "
                   f"yields PF_test={best_pf:.3f} on n={best_n} "
                   f"({best_sig:.0f} signals/year).")
    elif stable_combos and best_pf >= 1.30 and best_n >= 200:
        verdict = (f"⚠️ **ADD WITH CAUTION** — best stable filter `{best['name']}` "
                   f"yields PF_test={best_pf:.3f} on n={best_n} "
                   f"({best_sig:.0f} signals/year). Below 1.40 cutoff but above marginal.")
    elif stable_combos and best_n < 200:
        verdict = (f"❌ **REJECT** — only stable filter has n_test={best_n} < 200; "
                   f"too small a sample to ship.")
    else:
        verdict = "❌ **REJECT** — no filter combination passes the stability gate. " \
                  "The XAU regime-filter edge does NOT transfer to EURUSD under this " \
                  "config (enter=40/exit=25, default SL=1.5×ATR, TP=3×ATR)."
    md.append(verdict)

    # NY×Q4 saigneur check
    ny_q4 = merged[(merged["session"] == "NY") & (merged["ATR_Q"] == "Q4_high")]
    ny_q4_test = ny_q4[ny_q4["set"] == "TEST"]
    ny_q4_m = metrics(ny_q4_test["r_multiple"])
    md.append(f"\n## NY × Q4_high specifically\n")
    md.append(f"- n_test={ny_q4_m['n']}, PF_test={fmt(ny_q4_m['pf'])}, "
              f"total_R_test={fmt(ny_q4_m['total_r'])}, "
              f"win={fmt(ny_q4_m['winrate'])}")
    if not np.isnan(ny_q4_m["pf"]) and ny_q4_m["pf"] < 0.9 and ny_q4_m["total_r"] < -5:
        md.append("→ **Saigneur confirmé** (same pattern as XAU).")
    elif not np.isnan(ny_q4_m["pf"]) and ny_q4_m["pf"] < 1.0:
        md.append("→ Mild loser, not a dramatic saigneur like on XAU.")
    else:
        md.append("→ NOT a saigneur on EURUSD — the XAU pattern does not replicate here.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[eur] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
