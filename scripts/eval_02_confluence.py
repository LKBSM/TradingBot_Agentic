"""Eval Prompt 02 — ConfluenceDetector scoring audit.

Reads the 1597-trade 6-year baseline + 924-trade relaxed-30 sweep and produces:
- score distribution (pre/post renormalisation)
- reliability diagram (score bucket -> win rate, R)
- Brier score vs naive baseline
- temporal stability (per-year score stats and expectancy)
- component correlation (requires re-running detector on bars — out of scope here;
  we approximate with a single-instrument grid from the stored components).

Outputs CSVs + a JSON summary under reports/eval_02/.

Usage:
    python -m scripts.eval_02_confluence
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "reports" / "baseline_full_trades.csv"
SWEEP = ROOT / "reports" / "audit" / "trades_combined.csv"
OUT = ROOT / "reports" / "eval_02"
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------- helpers -------------------------------------- #

def _brier(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((probs - outcomes) ** 2))


def _reliability(df: pd.DataFrame, score_col: str, win_col: str, bins: np.ndarray) -> pd.DataFrame:
    df = df.copy()
    df["bucket"] = pd.cut(df[score_col], bins=bins, include_lowest=True)
    rows = []
    for bkt, grp in df.groupby("bucket", observed=True):
        if len(grp) == 0:
            continue
        rows.append(
            {
                "bucket": str(bkt),
                "n": int(len(grp)),
                "win_rate": float(grp[win_col].mean()),
                "expectancy_R": float(grp["r_multiple"].mean()),
                "total_R": float(grp["r_multiple"].sum()),
                "score_mean": float(grp[score_col].mean()),
                "score_min": float(grp[score_col].min()),
                "score_max": float(grp[score_col].max()),
            }
        )
    return pd.DataFrame(rows)


def _yearly(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    df = df.copy()
    df["year"] = pd.to_datetime(df["entry_bar"]).dt.year
    rows = []
    for yr, grp in df.groupby("year"):
        rows.append(
            {
                "year": int(yr),
                "n_trades": int(len(grp)),
                "win_rate": float((grp["r_multiple"] > 0).mean()),
                "expectancy_R": float(grp["r_multiple"].mean()),
                "score_mean": float(grp[score_col].mean()),
                "score_p50": float(grp[score_col].quantile(0.5)),
                "score_p90": float(grp[score_col].quantile(0.9)),
                "score_max": float(grp[score_col].max()),
                "corr_score_R": float(grp[score_col].corr(grp["r_multiple"])),
            }
        )
    return pd.DataFrame(rows)


# ----------------------------- main ----------------------------------------- #

def _load(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["win"] = (df["r_multiple"] > 0).astype(int)
    df["dataset"] = label
    return df


def main() -> None:
    baseline = _load(BASELINE, "baseline_post_renorm")
    sweep = _load(SWEEP, "sweep_relaxed30_pre_renorm")

    # Score distribution summary
    dist = []
    for df, label in [(baseline, "baseline_post_renorm"), (sweep, "sweep_relaxed30_pre_renorm")]:
        s = df["confluence_score"]
        dist.append(
            {
                "dataset": label,
                "n": int(len(df)),
                "mean": float(s.mean()),
                "min": float(s.min()),
                "p10": float(s.quantile(0.10)),
                "p25": float(s.quantile(0.25)),
                "p50": float(s.quantile(0.50)),
                "p75": float(s.quantile(0.75)),
                "p90": float(s.quantile(0.90)),
                "p95": float(s.quantile(0.95)),
                "p99": float(s.quantile(0.99)),
                "max": float(s.max()),
                "share_ge_60": float((s >= 60).mean()),
                "share_ge_70": float((s >= 70).mean()),
                "share_ge_80": float((s >= 80).mean()),
            }
        )
    pd.DataFrame(dist).to_csv(OUT / "score_distribution.csv", index=False)

    # Reliability diagram (baseline = 1597 trades, post-renorm)
    bins_b = np.array([40, 43, 46, 50, 55, 60, 65, 80])
    rel_baseline = _reliability(baseline, "confluence_score", "win", bins_b)
    rel_baseline.to_csv(OUT / "reliability_baseline.csv", index=False)

    bins_s = np.array([0, 30, 35, 40, 45, 50, 60])
    rel_sweep = _reliability(sweep, "confluence_score", "win", bins_s)
    rel_sweep.to_csv(OUT / "reliability_sweep.csv", index=False)

    # Brier — treat score/100 as proba-of-win for a naive calibration check
    base_brier = _brier(baseline["confluence_score"].values / 100.0, baseline["win"].values)
    naive_brier = _brier(np.full(len(baseline), baseline["win"].mean()), baseline["win"].values)
    brier_stats = {
        "n_trades": int(len(baseline)),
        "base_rate_win": float(baseline["win"].mean()),
        "brier_score_as_prob_div100": base_brier,
        "brier_naive_mean": naive_brier,
        "brier_skill_score": float(1.0 - base_brier / naive_brier),
    }
    with open(OUT / "brier.json", "w") as f:
        json.dump(brier_stats, f, indent=2)

    # Spearman rank correlation score -> R (monotonic)
    rank_corr_base = baseline["confluence_score"].rank().corr(baseline["r_multiple"].rank())
    rank_corr_sweep = sweep["confluence_score"].rank().corr(sweep["r_multiple"].rank())
    with open(OUT / "rank_correlation.json", "w") as f:
        json.dump(
            {
                "baseline_spearman_score_vs_R": float(rank_corr_base),
                "sweep_spearman_score_vs_R": float(rank_corr_sweep),
            },
            f,
            indent=2,
        )

    # Temporal stability
    _yearly(baseline, "confluence_score").to_csv(OUT / "yearly_baseline.csv", index=False)
    _yearly(sweep, "confluence_score").to_csv(OUT / "yearly_sweep.csv", index=False)

    # Direction-split calibration on baseline
    for direction in ("LONG", "SHORT"):
        sub = baseline[baseline["direction"] == direction]
        rel_dir = _reliability(sub, "confluence_score", "win", bins_b)
        rel_dir.to_csv(OUT / f"reliability_baseline_{direction}.csv", index=False)

    # Simple monotonicity check: does win-rate rise with score bucket?
    monot = []
    for label, df in [("baseline", rel_baseline), ("sweep", rel_sweep)]:
        wr = df["win_rate"].to_numpy()
        er = df["expectancy_R"].to_numpy()
        monot.append(
            {
                "dataset": label,
                "buckets": len(df),
                "is_winrate_monotone_up": bool(np.all(np.diff(wr) >= 0)),
                "is_expectancy_monotone_up": bool(np.all(np.diff(er) >= 0)),
                "winrate_range": [float(wr.min()), float(wr.max())],
                "expectancy_range": [float(er.min()), float(er.max())],
            }
        )
    with open(OUT / "monotonicity.json", "w") as f:
        json.dump(monot, f, indent=2)

    summary = {
        "score_distribution": dist,
        "reliability_baseline": rel_baseline.to_dict(orient="records"),
        "reliability_sweep": rel_sweep.to_dict(orient="records"),
        "brier": brier_stats,
        "rank_correlation": {
            "baseline_spearman_score_vs_R": float(rank_corr_base),
            "sweep_spearman_score_vs_R": float(rank_corr_sweep),
        },
        "yearly_baseline": _yearly(baseline, "confluence_score").to_dict(orient="records"),
        "monotonicity": monot,
    }
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Wrote {OUT}/ — {len(list(OUT.glob('*')))} artefacts")


if __name__ == "__main__":
    main()
