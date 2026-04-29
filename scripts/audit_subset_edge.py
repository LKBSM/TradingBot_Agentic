"""Chantier 3 (mini) — find regime subsets with PF >= 1.30 OOS.

Splits the 2363 replay trades by:
  * ATR percentile quartile (vol regime)
  * Session (Asian / London / NY / OffHours)
  * Day of week
  * Direction
  * Combinations (ATR_quartile × session)

Reports: per-subset PF, win%, expectancy, n_trades. Train (2019-2022) vs
Test (2023-2025). Highlights subsets with PF_test >= 1.30 AND n_test >= 100.

Usage:
    python scripts/audit_subset_edge.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine


SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]


def session_label(hour: int) -> str:
    for lo, hi, name in SESSION_BINS:
        if lo <= hour < hi:
            return name
    return "OffHours"


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


def fmt(v, fmt_spec=".3f") -> str:
    if isinstance(v, float):
        if np.isnan(v):
            return "—"
        return f"{v:{fmt_spec}}"
    return str(v)


def main() -> int:
    csv_path = Path("data/XAU_15MIN_2019_2025.csv")
    trades_path = Path("reports/audit/trades_combined.csv")
    out_path = Path("reports/feature_subset_audit.md")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print(f"[subset] enriching {len(df)} bars...")
    enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enriched["hour"] = enriched.index.hour
    enriched["dow"] = enriched.index.dayofweek
    enriched["session"] = enriched["hour"].map(session_label)
    atr = enriched["ATR"].fillna(0)
    enriched["ATR_PCTL"] = atr.rolling(30 * 96, min_periods=200).rank(pct=True)
    enriched["ATR_Q"] = pd.qcut(
        enriched["ATR_PCTL"].fillna(0.5), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"],
    )

    trades = pd.read_csv(trades_path)
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feats = enriched[["session", "dow", "hour", "ATR_PCTL", "ATR_Q"]]
    merged = trades.merge(feats, left_on="entry_bar", right_index=True, how="inner")
    merged["set"] = np.where(merged["entry_bar"] < "2023-01-01", "TRAIN", "TEST")
    print(f"[subset] {len(merged)} trades ({(merged['set']=='TRAIN').sum()} train, {(merged['set']=='TEST').sum()} test)")

    md = ["# Subset Edge Audit — XAU/USD M15 (Chantier 3 mini)\n"]
    md.append(f"**Trades**: {len(merged)} | Train < 2023-01-01 | Test ≥ 2023-01-01")
    md.append("**Acceptance**: PF_test ≥ 1.30 AND n_test ≥ 100 AND sign(PF_train-1) = sign(PF_test-1)\n")
    md.append("---\n")

    overall = {
        "TRAIN": metrics(merged.loc[merged["set"] == "TRAIN", "r_multiple"]),
        "TEST": metrics(merged.loc[merged["set"] == "TEST", "r_multiple"]),
    }
    md.append("## Overall\n")
    md.append(f"- TRAIN: n={overall['TRAIN']['n']}, PF={fmt(overall['TRAIN']['pf'])}, win={fmt(overall['TRAIN']['winrate'])}, exp={fmt(overall['TRAIN']['exp'])}, total_R={fmt(overall['TRAIN']['total_r'])}")
    md.append(f"- TEST:  n={overall['TEST']['n']}, PF={fmt(overall['TEST']['pf'])}, win={fmt(overall['TEST']['winrate'])}, exp={fmt(overall['TEST']['exp'])}, total_R={fmt(overall['TEST']['total_r'])}\n")

    def split_table(grouper_cols: list[str], title: str) -> str:
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
                and (tr["pf"] - 1) * (te["pf"] - 1) > 0  # same sign of edge
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
        rows.sort(key=lambda r: -float(r["PF_test"]) if r["PF_test"] not in ("—",) else 0)
        cols = grouper_cols + ["n_train", "PF_train", "n_test", "PF_test", "win_test", "exp_test", "total_R_test", "✅"]
        out = [f"\n## {title}\n"]
        out.append("| " + " | ".join(cols) + " |")
        out.append("|" + "|".join(["---"] * len(cols)) + "|")
        for r in rows:
            out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
        return "\n".join(out)

    md.append(split_table(["direction"], "By direction"))
    md.append(split_table(["session"], "By session"))
    md.append(split_table(["ATR_Q"], "By ATR quartile (vol regime)"))
    md.append(split_table(["dow"], "By day of week"))
    md.append(split_table(["session", "ATR_Q"], "By session × ATR_Q"))
    md.append(split_table(["session", "direction"], "By session × direction"))
    md.append(split_table(["ATR_Q", "direction"], "By ATR_Q × direction"))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[subset] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
