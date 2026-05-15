"""Chantier 1 — feature-by-feature edge audit on the 7-year XAU replay.

For each candidate feature (raw SMC outputs + derived session/vol),
compute its correlation with r_multiple per direction (LONG / SHORT),
and verify it survives an out-of-sample split (train 2019-2022, test
2023-2025). Decision criterion: a feature has actionable edge if
|Pearson_test| >= 0.05 AND sign(Pearson_train) == sign(Pearson_test).

Usage:
    python scripts/audit_feature_edge.py \\
        --csv data/XAU_15MIN_2019_2026.csv \\
        --trades reports/audit/trades_combined.csv \\
        --out reports/feature_edge_audit.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine


SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]


def session_label(hour: int) -> str:
    for lo, hi, name in SESSION_BINS:
        if lo <= hour < hi:
            return name
    return "OffHours"


def load_ohlcv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    return df


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    eng = SmartMoneyEngine(data=df, config={}, verbose=False)
    out = eng.analyze().copy()
    # Derived features
    out["hour"] = out.index.hour
    out["dow"] = out.index.dayofweek
    out["session"] = out["hour"].map(session_label)
    # BB position (-1 below band, 0 mid, +1 above)
    bb_range = (out["BB_H"] - out["BB_L"]).replace(0, np.nan)
    out["BB_POS"] = ((out["close"] - out["BB_M"]) / bb_range).clip(-2, 2)
    # ATR rolling percentile (vol regime proxy, 30-day window)
    atr = out["ATR"].fillna(0)
    out["ATR_PCTL"] = atr.rolling(30 * 96, min_periods=200).rank(pct=True)
    # Body/spread efficiency
    out["BODY_RATIO"] = out["BODY_SIZE"] / out["SPREAD"].replace(0, np.nan)
    return out


def aligned(value: pd.Series, direction: pd.Series) -> pd.Series:
    """Return value*direction so positive means aligned with trade dir."""
    sign = direction.map({"LONG": 1, "SHORT": -1}).astype(float)
    return value * sign


def join_features(trades: pd.DataFrame, enriched: pd.DataFrame) -> pd.DataFrame:
    trades = trades.copy()
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    feats = enriched.loc[
        enriched.index.intersection(trades["entry_bar"].unique())
    ]
    merged = trades.merge(
        feats, left_on="entry_bar", right_index=True, how="inner"
    )
    return merged


def per_feature_stats(
    df: pd.DataFrame, feature: str, target: str = "r_multiple"
) -> dict:
    s = df[[feature, target]].dropna()
    if len(s) < 30 or s[feature].nunique() <= 1:
        return {"n": len(s), "pearson": np.nan, "spearman": np.nan, "mi": np.nan}
    try:
        pr, _ = pearsonr(s[feature], s[target])
    except Exception:
        pr = np.nan
    try:
        sp, _ = spearmanr(s[feature], s[target])
    except Exception:
        sp = np.nan
    try:
        mi = float(
            mutual_info_regression(
                s[feature].values.reshape(-1, 1), s[target].values, random_state=0,
            )[0]
        )
    except Exception:
        mi = np.nan
    return {"n": len(s), "pearson": float(pr), "spearman": float(sp), "mi": mi}


# Features to audit. "aligned" features get sign-flipped for SHORT trades.
RAW_FEATURES_ALIGNED = [
    "BOS_SIGNAL", "BOS_EVENT", "CHOCH_SIGNAL", "BOS_RETEST_ARMED",
    "FVG_SIGNAL", "CHOCH_DIVERGENCE", "MACD_Diff", "BB_POS",
]
RAW_FEATURES_UNSIGNED = [
    "FVG_SIZE_NORM", "OB_STRENGTH_NORM", "RSI", "ATR", "ATR_PCTL",
    "BODY_RATIO", "hour", "dow",
]


def build_feature_frame(merged: pd.DataFrame) -> pd.DataFrame:
    """Return per-trade feature matrix with sign-aligned columns."""
    out = pd.DataFrame(index=merged.index)
    out["entry_bar"] = merged["entry_bar"].values
    out["direction"] = merged["direction"].values
    out["r_multiple"] = merged["r_multiple"].astype(float).values
    out["confluence_score"] = merged["confluence_score"].astype(float).values
    direction = merged["direction"]
    for f in RAW_FEATURES_ALIGNED:
        if f in merged.columns:
            out[f + "_aligned"] = aligned(merged[f].astype(float), direction).values
    for f in RAW_FEATURES_UNSIGNED:
        if f in merged.columns:
            out[f] = merged[f].astype(float).values
    # Special: |FVG_SIZE_NORM| in alignment direction
    if "FVG_SIZE_NORM" in merged.columns and "FVG_SIGNAL" in merged.columns:
        a = aligned(merged["FVG_SIGNAL"].astype(float), direction)
        out["FVG_aligned_size"] = (a > 0).astype(float) * merged["FVG_SIZE_NORM"].abs()
    if "BOS_RETEST_ARMED" in merged.columns:
        a = aligned(merged["BOS_RETEST_ARMED"].astype(float), direction)
        out["RETEST_aligned_bool"] = (a > 0).astype(float)
    return out


def split_train_test(df: pd.DataFrame, cutoff: str = "2023-01-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["entry_bar"] = pd.to_datetime(df["entry_bar"])
    train = df[df["entry_bar"] < cutoff].copy()
    test = df[df["entry_bar"] >= cutoff].copy()
    return train, test


def render_table(
    rows: list[dict], cols: list[str], title: str
) -> str:
    out = [f"\n## {title}\n"]
    out.append("| " + " | ".join(cols) + " |")
    out.append("|" + "|".join(["---"] * len(cols)) + "|")
    for r in rows:
        cells = [str(r.get(c, "")) for c in cols]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def fmt(v) -> str:
    if isinstance(v, float) and not np.isnan(v):
        return f"{v:+.4f}"
    return "—"


def audit(
    csv_path: Path, trades_path: Path, out_path: Path, cutoff: str
) -> int:
    print(f"[audit] loading {csv_path}")
    df = load_ohlcv(csv_path)
    print(f"[audit] enriching {len(df)} bars with SmartMoneyEngine...")
    enriched = enrich(df)
    print(f"[audit] loading trades {trades_path}")
    trades = pd.read_csv(trades_path)
    print(f"[audit] {len(trades)} trades, joining on entry_bar")
    merged = join_features(trades, enriched)
    print(f"[audit] {len(merged)} trades after join")

    feat = build_feature_frame(merged)
    feature_cols = [
        c for c in feat.columns
        if c not in ("entry_bar", "direction", "r_multiple", "confluence_score")
    ]

    train, test = split_train_test(feat, cutoff)
    print(f"[audit] train={len(train)} test={len(test)} (cutoff {cutoff})")

    # Per-feature: train + test correlations, all + per-direction
    rows = []
    for f in feature_cols:
        all_train = per_feature_stats(train, f)
        all_test = per_feature_stats(test, f)
        long_test = per_feature_stats(test[test["direction"] == "LONG"], f)
        short_test = per_feature_stats(test[test["direction"] == "SHORT"], f)
        # Decision: same sign + |test_pearson| >= 0.05 + n_test >= 100
        stable = False
        try:
            if (
                all_train["pearson"] is not np.nan
                and all_test["pearson"] is not np.nan
                and not np.isnan(all_train["pearson"])
                and not np.isnan(all_test["pearson"])
                and np.sign(all_train["pearson"]) == np.sign(all_test["pearson"])
                and abs(all_test["pearson"]) >= 0.05
                and all_test["n"] >= 100
            ):
                stable = True
        except Exception:
            pass

        rows.append({
            "feature": f,
            "n_train": all_train["n"],
            "n_test": all_test["n"],
            "pearson_train": fmt(all_train["pearson"]),
            "pearson_test": fmt(all_test["pearson"]),
            "spearman_test": fmt(all_test["spearman"]),
            "mi_test": fmt(all_test["mi"]),
            "long_pearson_test": fmt(long_test["pearson"]),
            "short_pearson_test": fmt(short_test["pearson"]),
            "stable_edge": "✅" if stable else "—",
        })

    # Sort by absolute test pearson desc (stable first)
    def _sort_key(r):
        v = r["pearson_test"]
        try:
            return -abs(float(v))
        except Exception:
            return 0
    rows.sort(key=_sort_key)

    # Confluence score baseline
    score_train = per_feature_stats(train, "confluence_score")
    score_test = per_feature_stats(test, "confluence_score")

    md = []
    md.append("# Feature Edge Audit — XAU/USD M15 (Chantier 1)\n")
    md.append(f"**Trades**: {len(feat)}  |  **Train**: {len(train)} (< {cutoff})  |  **Test**: {len(test)} (≥ {cutoff})\n")
    md.append(f"**Decision criterion**: stable_edge = ✅ if |Pearson_test| ≥ 0.05 AND sign(Pearson_train) = sign(Pearson_test) AND n_test ≥ 100\n")
    md.append("\n---\n")
    md.append(f"## Baseline — confluence_score (the current product)\n")
    md.append(f"- **Train**: n={score_train['n']}, Pearson={fmt(score_train['pearson'])}, Spearman={fmt(score_train['spearman'])}")
    md.append(f"- **Test**:  n={score_test['n']}, Pearson={fmt(score_test['pearson'])}, Spearman={fmt(score_test['spearman'])}")
    md.append(f"- **Verdict**: edge {'✅' if abs(score_test['pearson'])>=0.05 else '❌'} ({fmt(score_test['pearson'])})\n")

    md.append(render_table(
        rows,
        ["feature", "n_train", "n_test", "pearson_train", "pearson_test",
         "spearman_test", "mi_test", "long_pearson_test", "short_pearson_test", "stable_edge"],
        "Per-feature edge (sorted by |Pearson_test|)",
    ))

    # Top stable features
    stable_feats = [r for r in rows if r["stable_edge"] == "✅"]
    md.append("\n## Stable features (decision: keep for Chantier 2)\n")
    if stable_feats:
        for r in stable_feats:
            md.append(f"- **{r['feature']}**: Pearson_test {r['pearson_test']} (train {r['pearson_train']}), n={r['n_test']}, MI={r['mi_test']}")
    else:
        md.append("**❌ NO feature passes the stability criterion.** The SMC pipeline as-instrumented has no out-of-sample edge on XAU M15. Recommend: pivot to Chantier 3 (subset filter) or accept that the strategy lacks alpha.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[audit] wrote {out_path}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/XAU_15MIN_2019_2026.csv")
    p.add_argument("--trades", default="reports/audit/trades_combined.csv")
    p.add_argument("--out", default="reports/feature_edge_audit.md")
    p.add_argument("--cutoff", default="2023-01-01",
                   help="Train < cutoff, Test >= cutoff (ISO date)")
    a = p.parse_args()
    return audit(Path(a.csv), Path(a.trades), Path(a.out), a.cutoff)


if __name__ == "__main__":
    raise SystemExit(main())
