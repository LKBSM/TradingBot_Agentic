"""Evaluate BOCPD (Bayesian Online Changepoint Detection) on OOS data.

Q1 (Justesse factuelle):
  - Distribution sanity: cp_prob should occupy a non-trivial portion of (0, 1).
  - Variance-shift agreement: a bar flagged with high cp_prob (>= 0.5) must
    coincide (within ±8 bars) with a statistically detectable variance shift
    in 32-bar pre/post windows (Welch t-test on log|r|² with p<0.01).

Q2 (Stabilité):
  - Average inter-changepoint distance vs the prior hazard 1/240.
  - Expected run-length trajectory (should reset on high cp_prob bars).

Q3 (Calibration):
  - Bin cp_prob into [0-0.05, 0.05-0.2, 0.2-0.5, 0.5-0.8, 0.8-1.0]; for each
    bin, report the fraction of bars whose ±8 neighbourhood contains a
    statistically significant variance shift. ECE on cp_prob vs hit-rate.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from scripts.audit.descriptive_quality._harness import (  # noqa: E402
    InstrumentData,
    bootstrap_ci,
    load_instrument,
    verdict_ece,
)
from src.intelligence.bocpd import bocpd_run  # noqa: E402


PRE_POST_WIN = 32
NEIGHBOR_WIN = 8
ALPHA = 0.01  # Welch test significance


def _variance_shift_flags(abs_r2: np.ndarray) -> np.ndarray:
    """Return a bool mask: True at bar i if a Welch t-test on
    log(|r|²) before vs after at window PRE_POST_WIN rejects equality at ALPHA.
    """
    from scipy.stats import ttest_ind

    n = len(abs_r2)
    flags = np.zeros(n, dtype=bool)
    log_r2 = np.log(np.maximum(abs_r2, 1e-30))
    for i in range(PRE_POST_WIN, n - PRE_POST_WIN):
        pre = log_r2[i - PRE_POST_WIN: i]
        post = log_r2[i: i + PRE_POST_WIN]
        _, p = ttest_ind(pre, post, equal_var=False)
        if p < ALPHA:
            flags[i] = True
    return flags


def _neighborhood_match(cp_idx: np.ndarray, shift_flags: np.ndarray, window: int) -> np.ndarray:
    """Return bool array of len(cp_idx): True if any shift in ±window."""
    n = len(shift_flags)
    matched = np.zeros(len(cp_idx), dtype=bool)
    for k, i in enumerate(cp_idx):
        lo = max(0, i - window)
        hi = min(n, i + window + 1)
        if shift_flags[lo:hi].any():
            matched[k] = True
    return matched


def eval_bocpd_for(inst: InstrumentData) -> dict:
    oos_close = inst.oos["close"].to_numpy()
    oos_ret = np.diff(np.log(oos_close))
    oos_ret = oos_ret[np.isfinite(oos_ret)]
    n = len(oos_ret)

    cp_prob = bocpd_run(oos_ret, hazard_inv=240.0, max_run_length=300)

    # Distribution sanity
    cp_quantiles = {
        "q01": float(np.quantile(cp_prob, 0.01)),
        "q50": float(np.quantile(cp_prob, 0.50)),
        "q95": float(np.quantile(cp_prob, 0.95)),
        "q99": float(np.quantile(cp_prob, 0.99)),
        "mean": float(cp_prob.mean()),
        "min": float(cp_prob.min()),
        "max": float(cp_prob.max()),
    }

    # Variance shift flags (proxy ground truth)
    abs_r2 = oos_ret ** 2
    shift_flags = _variance_shift_flags(abs_r2)
    n_shifts = int(shift_flags.sum())

    # Inter-changepoint distance (cp_prob>0.5)
    high_cp_idx = np.where(cp_prob >= 0.5)[0]
    diffs = np.diff(high_cp_idx) if len(high_cp_idx) > 1 else np.array([])
    mean_dist = float(diffs.mean()) if len(diffs) else float("nan")

    # Q1 — high cp matches variance shift
    if len(high_cp_idx):
        matched_high = _neighborhood_match(high_cp_idx, shift_flags, NEIGHBOR_WIN)
        match_rate_high = float(matched_high.mean())
    else:
        match_rate_high = float("nan")

    # Q3 — calibration: bin cp_prob and compute fraction with variance shift in neighborhood
    bins_def = [(0.0, 0.05), (0.05, 0.20), (0.20, 0.50), (0.50, 0.80), (0.80, 1.001)]
    bin_results = []
    ece = 0.0
    total = 0
    for lo, hi in bins_def:
        in_bin = (cp_prob >= lo) & (cp_prob < hi)
        nb = int(in_bin.sum())
        if nb == 0:
            bin_results.append({"lo": lo, "hi": hi, "n": 0, "hit_rate": None, "mean_cp_prob": None})
            continue
        idx_in_bin = np.where(in_bin)[0]
        matched = _neighborhood_match(idx_in_bin, shift_flags, NEIGHBOR_WIN)
        hit_rate = float(matched.mean())
        mean_cp = float(cp_prob[in_bin].mean())
        bin_results.append({
            "lo": lo, "hi": hi, "n": nb,
            "hit_rate": hit_rate, "mean_cp_prob": mean_cp,
        })
        total += nb
    for r in bin_results:
        if r["n"] == 0:
            continue
        ece += abs(r["hit_rate"] - r["mean_cp_prob"]) * (r["n"] / max(1, total))
    ece = float(ece)

    return {
        "symbol": inst.symbol,
        "n_oos_returns": n,
        "n_high_cp_05": int(len(high_cp_idx)),
        "n_variance_shifts": int(n_shifts),
        "q1_distribution": cp_quantiles,
        "q1_high_cp_matches_shift": {
            "rate": match_rate_high,
            "n_high_cp": int(len(high_cp_idx)),
            "neighbor_window_bars": NEIGHBOR_WIN,
            "definition": f"high cp_prob (>=0.5) flagged within ±{NEIGHBOR_WIN} bars of a Welch-test variance shift (p<{ALPHA})",
        },
        "q2_persistence": {
            "mean_inter_cp_distance_bars": mean_dist,
            "prior_hazard_bars": 240,
            "definition": "mean distance between consecutive bars with cp_prob>=0.5",
        },
        "q3_calibration": {
            "ece": ece,
            "verdict": verdict_ece(ece),
            "bins": bin_results,
            "n_total": total,
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_bocpd_for(inst)
        results[sym] = out
        q = out["q1_distribution"]
        print(
            f"  n={out['n_oos_returns']:,d}  cp>=0.5: {out['n_high_cp_05']:,d}  "
            f"variance shifts: {out['n_variance_shifts']:,d}\n"
            f"  Q1 cp_prob distribution: q01={q['q01']:.4f} q50={q['q50']:.4f} q95={q['q95']:.4f} q99={q['q99']:.4f} mean={q['mean']:.4f}\n"
            f"  Q1 high-cp matches shift: {out['q1_high_cp_matches_shift']['rate']:.3f}\n"
            f"  Q2 mean inter-cp dist:    {out['q2_persistence']['mean_inter_cp_distance_bars']:.1f} bars (prior 240)\n"
            f"  Q3 ECE = {out['q3_calibration']['ece']:.4f} {out['q3_calibration']['verdict']}"
        )
        for b in out["q3_calibration"]["bins"]:
            if b["n"] == 0:
                continue
            print(f"     [{b['lo']:.2f}-{b['hi']:.2f}]: n={b['n']:>6d}  hit_rate={b['hit_rate']:.3f}  mean_cp={b['mean_cp_prob']:.4f}")

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "bocpd_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
