"""Evaluate the 3-state HMM regime classifier on OOS data.

Pipeline: fit on TRAIN returns (2019-2023), predict on OOS (2024+).

Q1 (Justesse factuelle):
  - Label sanity: bars labeled "high_vol_stress" must have higher
    realized variance of returns than "low_vol_*" bars.
  - Label separation: between low_vol_trending and low_vol_ranging, the
    realized |mean return| over rolling 32-bar windows must be larger for
    trending vs ranging.

Q2 (Stabilité temporelle):
  - Mean dwell time per state (median run-length).
  - Flicker rate: fraction of bars where state changes from previous bar.
  - Transition matrix from observed sequence.

Q3 (Calibration de l'incertitude):
  - HMM exposes a posterior probability for the chosen state. Calibration
    test: bin posterior into [0.4-0.5, 0.5-0.6, ..., 0.9-1.0]; for each
    bin, compute the fraction of bars where the next 8 bars confirm the
    label (high_vol_stress → next-8 realized vol > train-median high-vol
    threshold; low_vol_* → below threshold).
  - ECE (Expected Calibration Error) on this binned check.
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
from src.intelligence.regime_classifier import (  # noqa: E402
    LABEL_HIGH_VOL_STRESS,
    LABEL_LOW_VOL_RANGING,
    LABEL_LOW_VOL_TRENDING,
    RegimeClassifier,
)


CONFIRM_HORIZON = 8     # bars
N_BINS = 6              # posterior bins 0.4-0.5, 0.5-0.6, ..., 0.9-1.0


def eval_hmm_for(inst: InstrumentData) -> dict:
    train_close = inst.train["close"].to_numpy()
    oos_close = inst.oos["close"].to_numpy()

    train_ret = np.diff(np.log(train_close))
    oos_ret = np.diff(np.log(oos_close))
    train_ret = train_ret[np.isfinite(train_ret)]
    oos_ret = oos_ret[np.isfinite(oos_ret)]

    # Fit on train, predict on OOS
    clf = RegimeClassifier(random_state=42).fit(train_ret)
    preds = clf.predict_with_confidence(oos_ret)
    labels = np.array([p.label for p in preds])
    states = np.array([p.state for p in preds])
    confs = np.array([p.confidence for p in preds])
    n = len(preds)

    # Variance proxy: rolling |r|² over 32 bars (forward-looking confirmation)
    abs2 = oos_ret ** 2
    win = CONFIRM_HORIZON
    fwd_realised_var = np.full(n, np.nan)
    for i in range(n - win):
        fwd_realised_var[i] = float(np.mean(abs2[i + 1: i + 1 + win]))

    # Train-based threshold: above which we call it "stress" objectively
    train_abs2 = train_ret ** 2
    stress_thresh = float(np.quantile(train_abs2, 2 / 3))  # top 1/3 of train returns

    # === Q1.a — high_vol_stress bars must have higher realized variance ===
    stress_mask = labels == LABEL_HIGH_VOL_STRESS
    non_stress_mask = ~stress_mask & ~np.isnan(fwd_realised_var)
    stress_mask_obs = stress_mask & ~np.isnan(fwd_realised_var)
    med_var_stress = float(np.median(fwd_realised_var[stress_mask_obs])) if stress_mask_obs.any() else float("nan")
    med_var_other = float(np.median(fwd_realised_var[non_stress_mask])) if non_stress_mask.any() else float("nan")
    ratio_stress = med_var_stress / med_var_other if med_var_other > 0 else float("nan")

    # === Q1.b — trending vs ranging |drift| ===
    # Rolling 32-bar mean return as drift proxy
    drift = pd.Series(oos_ret).rolling(32, min_periods=10).mean().to_numpy()
    trending_mask = labels == LABEL_LOW_VOL_TRENDING
    ranging_mask = labels == LABEL_LOW_VOL_RANGING
    drift_trending = float(np.median(np.abs(drift[trending_mask][~np.isnan(drift[trending_mask])])))
    drift_ranging = float(np.median(np.abs(drift[ranging_mask][~np.isnan(drift[ranging_mask])])))
    drift_ratio = drift_trending / drift_ranging if drift_ranging > 0 else float("nan")

    # === Q2 — persistence ===
    transitions = (states[1:] != states[:-1]).sum()
    flicker_rate = transitions / max(1, n - 1)
    # Dwell time per state: median run length
    dwell = {}
    for s in [0, 1, 2]:
        runs = []
        cur = 0
        for i in range(n):
            if states[i] == s:
                cur += 1
            else:
                if cur > 0:
                    runs.append(cur)
                cur = 0
        if cur > 0:
            runs.append(cur)
        dwell[clf.state_labels()[s]] = {
            "median_bars": float(np.median(runs)) if runs else 0.0,
            "mean_bars": float(np.mean(runs)) if runs else 0.0,
            "n_runs": len(runs),
        }

    # Transition matrix
    trans = np.zeros((3, 3), dtype=int)
    for i in range(1, n):
        trans[states[i - 1], states[i]] += 1
    row_sums = trans.sum(axis=1, keepdims=True)
    trans_prob = (trans / np.maximum(row_sums, 1)).tolist()

    # === Q3 — calibration: per-bin label-match rate ===
    # Define ground-truth label per bar from fwd_realised_var:
    #   stress = fwd_var >= stress_thresh
    #   non-stress = below
    true_stress = (fwd_realised_var >= stress_thresh).astype(int)
    pred_stress = stress_mask.astype(int)
    # For stress, "correct" = pred_stress == true_stress
    correct = (pred_stress == true_stress) & ~np.isnan(fwd_realised_var)
    valid = ~np.isnan(fwd_realised_var)

    bin_edges = np.linspace(0.4, 1.0, N_BINS + 1)
    bins = []
    ece = 0.0
    n_valid = int(valid.sum())
    for j in range(N_BINS):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        in_bin = (confs >= lo) & (confs < hi) & valid
        nb = int(in_bin.sum())
        if nb == 0:
            bins.append({"lo": float(lo), "hi": float(hi), "n": 0, "accuracy": None, "mean_conf": None})
            continue
        acc = float(correct[in_bin].mean())
        mc = float(confs[in_bin].mean())
        ece += abs(acc - mc) * (nb / max(1, n_valid))
        bins.append({"lo": float(lo), "hi": float(hi), "n": nb,
                     "accuracy": acc, "mean_conf": mc})
    ece = float(ece)

    return {
        "symbol": inst.symbol,
        "n_train_returns": len(train_ret),
        "n_oos_returns": len(oos_ret),
        "state_labels": clf.state_labels(),
        "label_counts": {
            LABEL_LOW_VOL_TRENDING: int((labels == LABEL_LOW_VOL_TRENDING).sum()),
            LABEL_LOW_VOL_RANGING: int((labels == LABEL_LOW_VOL_RANGING).sum()),
            LABEL_HIGH_VOL_STRESS: int((labels == LABEL_HIGH_VOL_STRESS).sum()),
        },
        "q1_label_sanity_stress": {
            "median_fwd_var_stress": med_var_stress,
            "median_fwd_var_non_stress": med_var_other,
            "ratio": float(ratio_stress),
            "n_stress": int(stress_mask_obs.sum()),
            "n_non_stress": int(non_stress_mask.sum()),
            "definition": f"forward {CONFIRM_HORIZON}-bar median squared return on stress vs non-stress bars; ratio > 1 = label is descriptive",
        },
        "q1_label_sanity_trending": {
            "median_drift_trending": drift_trending,
            "median_drift_ranging": drift_ranging,
            "ratio": float(drift_ratio),
            "n_trending": int(trending_mask.sum()),
            "n_ranging": int(ranging_mask.sum()),
            "definition": "median |32-bar rolling mean return| on trending vs ranging bars; ratio > 1 = label is descriptive",
        },
        "q2_persistence": {
            "flicker_rate_per_bar": float(flicker_rate),
            "dwell_times": dwell,
            "transition_matrix": trans_prob,
            "n": n,
        },
        "q3_posterior_calibration": {
            "ece": ece,
            "verdict": verdict_ece(ece),
            "bins": bins,
            "n_valid": n_valid,
            "definition": f"binned posterior vs accuracy of stress/non-stress prediction confirmed by next-{CONFIRM_HORIZON}-bar realized var vs train-derived top-1/3 threshold",
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_hmm_for(inst)
        results[sym] = out
        print(f"  train returns={out['n_train_returns']:,d} oos returns={out['n_oos_returns']:,d}")
        print(f"  Label counts: " + ", ".join(f"{k}={v:,d}" for k, v in out["label_counts"].items()))
        q1s = out["q1_label_sanity_stress"]
        q1t = out["q1_label_sanity_trending"]
        q2 = out["q2_persistence"]
        q3 = out["q3_posterior_calibration"]
        print(
            f"  Q1 stress label sanity:    var_stress/var_non-stress = {q1s['ratio']:.2f}  "
            f"(n_stress={q1s['n_stress']}, n_other={q1s['n_non_stress']})\n"
            f"  Q1 trending label sanity:  drift_trending/drift_ranging = {q1t['ratio']:.2f}  "
            f"(n_t={q1t['n_trending']}, n_r={q1t['n_ranging']})\n"
            f"  Q2 flicker rate:           {q2['flicker_rate_per_bar']:.4f} per bar"
        )
        for label, d in q2["dwell_times"].items():
            print(f"     {label:23s}: median {d['median_bars']:5.1f} bars, mean {d['mean_bars']:5.1f}, runs={d['n_runs']}")
        print(f"  Q3 ECE = {q3['ece']:.4f} {q3['verdict']}")
        for b in q3["bins"]:
            if b["n"] == 0:
                continue
            print(f"     [{b['lo']:.2f}-{b['hi']:.2f}]: n={b['n']:>5d}  acc={b['accuracy']:.3f}  mean_conf={b['mean_conf']:.3f}")

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "hmm_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
