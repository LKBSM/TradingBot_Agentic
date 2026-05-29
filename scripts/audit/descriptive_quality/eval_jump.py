"""Evaluate jump_ratio (bipower variation jump share) on OOS data.

Q1 (Justesse factuelle):
  - Range sanity: jump_ratio in [0, 1] always (clamped by construction).
  - Distribution: median + extreme quantiles, % bars with NaN (window warmup).
  - Extreme-return alignment: when jump_ratio is high (top 5% of OOS),
    the 96-bar window must contain at least one |return| above the 99th
    percentile of the local return distribution.

Q2 (Stabilité):
  - Autocorrelation of jump_ratio at lag-1, lag-10 (96-bar rolling stat is
    naturally smooth — should be high). Mean run-length above a threshold.

Q3 (Calibration): N/A — jump_ratio is a continuous decomposition share, not
a probability.
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
    load_instrument,
)
from src.intelligence.regime_gate import jump_ratio  # noqa: E402


WIN = 96


def eval_jump_for(inst: InstrumentData) -> dict:
    oos_close = inst.oos["close"].to_numpy()
    oos_ret = np.diff(np.log(oos_close))
    oos_ret = oos_ret[np.isfinite(oos_ret)]
    n = len(oos_ret)

    jr = jump_ratio(oos_ret, window=WIN)
    valid_mask = ~np.isnan(jr)
    valid_count = int(valid_mask.sum())

    # Q1 — range sanity
    in_unit_interval = bool(((jr[valid_mask] >= 0) & (jr[valid_mask] <= 1)).all())
    quantiles = {
        "q01": float(np.quantile(jr[valid_mask], 0.01)),
        "q25": float(np.quantile(jr[valid_mask], 0.25)),
        "q50": float(np.quantile(jr[valid_mask], 0.50)),
        "q75": float(np.quantile(jr[valid_mask], 0.75)),
        "q95": float(np.quantile(jr[valid_mask], 0.95)),
        "q99": float(np.quantile(jr[valid_mask], 0.99)),
        "mean": float(jr[valid_mask].mean()),
    }

    # Q1 — extreme-return alignment
    high_mask = valid_mask & (jr >= quantiles["q95"])
    high_idx = np.where(high_mask)[0]
    extreme_thresh = float(np.quantile(np.abs(oos_ret), 0.99))
    alignment_hits = 0
    for i in high_idx:
        win_returns = oos_ret[max(0, i - WIN + 1): i + 1]
        if (np.abs(win_returns) >= extreme_thresh).any():
            alignment_hits += 1
    alignment_rate = alignment_hits / len(high_idx) if len(high_idx) else float("nan")

    # Q2 — autocorrelation
    jr_valid = jr[valid_mask]
    if len(jr_valid) > 10:
        autocorr1 = float(np.corrcoef(jr_valid[:-1], jr_valid[1:])[0, 1])
        autocorr10 = float(np.corrcoef(jr_valid[:-10], jr_valid[10:])[0, 1])
    else:
        autocorr1 = autocorr10 = float("nan")

    # Q2 — run-length above q75
    above = (jr >= quantiles["q75"]) & valid_mask
    runs = []
    cur = 0
    for v in above:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    median_run = float(np.median(runs)) if runs else 0.0

    return {
        "symbol": inst.symbol,
        "n_oos_returns": n,
        "n_valid": valid_count,
        "window": WIN,
        "q1_in_unit_interval": bool(in_unit_interval),
        "q1_distribution": quantiles,
        "q1_extreme_alignment": {
            "rate": float(alignment_rate),
            "n_high_jump_bars": int(len(high_idx)),
            "definition": "fraction of top-5% jump_ratio bars whose 96-bar window contains a |return| >= 99th percentile of OOS returns",
            "extreme_threshold_logret": extreme_thresh,
        },
        "q2_autocorrelation": {
            "lag1": autocorr1,
            "lag10": autocorr10,
            "note": "96-bar rolling stats should have high lag-1 autocorr by construction",
        },
        "q2_run_length_above_q75": {
            "median_run_bars": median_run,
            "n_runs": len(runs),
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_jump_for(inst)
        results[sym] = out
        q = out["q1_distribution"]
        ea = out["q1_extreme_alignment"]
        ac = out["q2_autocorrelation"]
        print(
            f"  n={out['n_oos_returns']:,d}  valid={out['n_valid']:,d}\n"
            f"  Q1 unit-interval: {out['q1_in_unit_interval']}\n"
            f"  Q1 distribution: median={q['q50']:.4f} q95={q['q95']:.4f} q99={q['q99']:.4f} mean={q['mean']:.4f}\n"
            f"  Q1 extreme alignment (top-5% jump_ratio): {ea['rate']:.3f}  "
            f"(n_high={ea['n_high_jump_bars']}, |r|_99={ea['extreme_threshold_logret']*1e4:.1f}bps)\n"
            f"  Q2 autocorr: lag1={ac['lag1']:.3f}  lag10={ac['lag10']:.3f}\n"
            f"  Q2 median run above q75: {out['q2_run_length_above_q75']['median_run_bars']} bars"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "jump_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
