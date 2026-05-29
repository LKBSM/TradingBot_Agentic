"""Evaluate BOS / bos_level / bos_event_age_bars on OOS data.

Q1 (Justesse factuelle):
  - Sanity: does each prod BOS_EVENT actually have close > break_level (bull) / < (bear)?
  - Level reality (exact): does bos_break_level appear as an OHLC extreme in the
    past 500 bars (matching within 0.05x ATR)?
  - Cross-method F1 vs an independent 3-bar swing detector at multiple match windows.

Q2 (Stabilité temporelle, fenêtre 16 bars = 4h M15):
  - hold_rate_mean: fraction of next-16-bar closes that respect the BOS direction
    (>= L for bull, <= L for bear). Mean-style stability.
  - no_opposite_bos_rate: fraction of events with NO opposite BOS within 16 bars
    (structural invalidation by counter-event).
  - median_time_to_recross: median bars until first close past the broken level.

Q3 (Calibration): N/A — BOS carries no probabilistic claim.
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
    detect_bos_independent,
    f1_from_counts,
    load_instrument,
    verdict_f1,
)


LEVEL_LOOKBACK = 500          # bars to look back for level reality
LEVEL_TOL_RATIO = 0.05        # bos_level must match a past OHLC within 0.05x ATR
LIFETIME_BARS = 16            # 4h on M15 — the doc's valid_until claim
MATCH_WINDOWS = [2, 5, 10]    # bars tolerance for cross-method matching


def _cross_method_f1(prod_ev: np.ndarray, ref_ev: np.ndarray, window: int) -> dict:
    n = len(prod_ev)
    matched_prod = np.zeros(n, dtype=bool)
    matched_ref = np.zeros(n, dtype=bool)
    prod_idx = np.where(prod_ev != 0)[0]
    for pi in prod_idx:
        lo = max(0, pi - window)
        hi = min(n - 1, pi + window)
        for ri in range(lo, hi + 1):
            if ref_ev[ri] == prod_ev[pi] and not matched_ref[ri]:
                matched_prod[pi] = True
                matched_ref[ri] = True
                break
    tp = int(matched_prod[prod_ev != 0].sum())
    fp = int(((prod_ev != 0) & (~matched_prod)).sum())
    fn = int(((ref_ev != 0) & (~matched_ref)).sum())
    p, r, f1 = f1_from_counts(tp, fp, fn)
    return {
        "window": window,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": p,
        "recall": r,
        "f1": f1,
    }


def eval_bos_for(inst: InstrumentData, label: str) -> dict:
    df = inst.oos.copy()
    df["bos_event"] = df["BOS_EVENT"].astype(int)
    df["bos_level"] = df["BOS_BREAK_LEVEL"].astype(float)
    df["atr"] = df["ATR"].astype(float)

    ev_mask = df["bos_event"] != 0
    n_events = int(ev_mask.sum())
    if n_events == 0:
        return {"symbol": inst.symbol, "n_events": 0, "skipped": True}

    ev = df.loc[ev_mask, ["close", "bos_event", "bos_level"]].dropna()
    bull = ev[ev["bos_event"] == 1]
    bear = ev[ev["bos_event"] == -1]

    # === Q1.a — sanity: close vs broken level on event bar ===
    sanity_bull = (bull["close"] > bull["bos_level"]).sum() / max(1, len(bull))
    sanity_bear = (bear["close"] < bear["bos_level"]).sum() / max(1, len(bear))
    sanity = (
        (bull["close"] > bull["bos_level"]).sum()
        + (bear["close"] < bear["bos_level"]).sum()
    ) / len(ev)

    # === Q1.b — level reality (exact OHLC match within 0.05xATR over 500 bars) ===
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    levels = df["bos_level"].to_numpy()
    atrs = df["atr"].to_numpy()
    events = df["bos_event"].to_numpy()
    idx = np.where(events != 0)[0]
    real_count = 0
    real_envelope = 0  # inside [past_min, past_max] bound
    for i in idx:
        s = max(0, i - LEVEL_LOOKBACK)
        lvl = levels[i]
        atr = atrs[i]
        if np.isnan(lvl) or np.isnan(atr) or atr <= 0 or i <= s:
            continue
        tol = atr * LEVEL_TOL_RATIO
        if events[i] == 1:
            # Match against past highs (level should be a past resistance)
            if np.any(np.abs(highs[s:i] - lvl) <= tol):
                real_count += 1
            if lows[s:i].min() <= lvl <= highs[s:i].max():
                real_envelope += 1
        else:
            # Match against past lows (level should be a past support)
            if np.any(np.abs(lows[s:i] - lvl) <= tol):
                real_count += 1
            if lows[s:i].min() <= lvl <= highs[s:i].max():
                real_envelope += 1
    level_reality = real_count / n_events
    level_envelope = real_envelope / n_events

    # === Q1.c — cross-method F1 vs 3-bar swing detector, multiple match windows ===
    ref = detect_bos_independent(
        df[["open", "high", "low", "close", "ATR"]].rename(columns={"ATR": "atr"})
    )
    ref_ev = ref["ref_bos_event"].to_numpy()
    n_ref = int((ref_ev != 0).sum())

    cross_windows = {f"W{w}": _cross_method_f1(events, ref_ev, w) for w in MATCH_WINDOWS}

    # Bootstrap CI on F1 at W=5 (our headline window)
    rng = np.random.default_rng(42)
    n = len(df)
    f1_boot = []
    for _ in range(500):
        sample_idx = rng.integers(0, n, n)
        prod_sub = events[sample_idx]
        ref_sub = ref_ev[sample_idx]
        m_prod = np.zeros(n, dtype=bool)
        m_ref = np.zeros(n, dtype=bool)
        for pi in np.where(prod_sub != 0)[0]:
            lo = max(0, pi - 5)
            hi = min(n - 1, pi + 5)
            for ri in range(lo, hi + 1):
                if ref_sub[ri] == prod_sub[pi] and not m_ref[ri]:
                    m_prod[pi] = True
                    m_ref[ri] = True
                    break
        tp_b = int(m_prod[prod_sub != 0].sum())
        fp_b = int(((prod_sub != 0) & (~m_prod)).sum())
        fn_b = int(((ref_sub != 0) & (~m_ref)).sum())
        _, _, f1_b = f1_from_counts(tp_b, fp_b, fn_b)
        f1_boot.append(f1_b)
    f1_w5_lo = float(np.quantile(f1_boot, 0.025))
    f1_w5_hi = float(np.quantile(f1_boot, 0.975))

    # === Q2 — stability over 16-bar lifetime ===
    closes = df["close"].to_numpy()
    hold_fracs = []      # for each event, fraction of next-16 closes respecting direction
    time_to_recross = []  # bars until first close re-crossing level (np.inf if never)
    no_opposite = 0       # fraction of events with no opposite BOS in next 16 bars
    n_with_window = 0
    for i in idx:
        lvl = levels[i]
        d = events[i]
        if np.isnan(lvl):
            continue
        j_end = min(n, i + 1 + LIFETIME_BARS)
        future_close = closes[i + 1: j_end]
        future_evs = events[i + 1: j_end]
        if len(future_close) == 0:
            continue
        n_with_window += 1
        if d == 1:
            respects = future_close >= lvl
        else:
            respects = future_close <= lvl
        hold_fracs.append(respects.mean())
        # First recross
        breach = ~respects
        if breach.any():
            time_to_recross.append(int(breach.argmax()) + 1)
        else:
            time_to_recross.append(LIFETIME_BARS + 1)  # censor sentinel
        # No opposite BOS in next 16 bars
        opp_mask = future_evs == -d
        if not opp_mask.any():
            no_opposite += 1

    hold_arr = np.array(hold_fracs)
    hold_mean = float(hold_arr.mean()) if len(hold_arr) else float("nan")
    _, hold_lo, hold_hi = bootstrap_ci(hold_arr, np.mean, n_boot=1000) if len(hold_arr) else (
        float("nan"), float("nan"), float("nan")
    )
    times = np.array(time_to_recross)
    finite_times = times[times <= LIFETIME_BARS]
    median_time = float(np.median(finite_times)) if len(finite_times) else float("nan")
    no_opp_rate = no_opposite / n_with_window if n_with_window else float("nan")

    return {
        "symbol": inst.symbol,
        "label": label,
        "n_events": int(n_events),
        "n_bull": int(len(bull)),
        "n_bear": int(len(bear)),
        "q1_sanity_close_vs_level": {
            "value": float(sanity),
            "bull": float(sanity_bull),
            "bear": float(sanity_bear),
            "n": int(len(ev)),
        },
        "q1_level_reality": {
            "exact_match_rate": float(level_reality),
            "definition_exact": f"bos_break_level matches a past OHLC extreme within {LEVEL_TOL_RATIO}xATR over past {LEVEL_LOOKBACK} bars",
            "envelope_rate": float(level_envelope),
            "definition_envelope": f"bos_break_level inside [past_min, past_max] over past {LEVEL_LOOKBACK} bars",
            "n": int(n_events),
        },
        "q1_cross_method_f1": {
            "headline_window_bars": 5,
            "f1_w5": cross_windows["W5"]["f1"],
            "f1_w5_ci95_lo": f1_w5_lo,
            "f1_w5_ci95_hi": f1_w5_hi,
            "verdict_w5": verdict_f1(cross_windows["W5"]["f1"]),
            "windows": cross_windows,
            "n_prod_events": int(n_events),
            "n_ref_events": int(n_ref),
            "ref_method": "3-bar swing causal",
            "note": "Prod uses 2-bar Williams fractal (more conservative); ref uses 3-bar swing (more permissive). Low F1 reflects definition divergence between two valid SMC implementations.",
        },
        "q2_stability_16bars": {
            "hold_rate_mean": hold_mean,
            "hold_rate_ci95_lo": float(hold_lo),
            "hold_rate_ci95_hi": float(hold_hi),
            "median_time_to_recross_bars": median_time,
            "no_opposite_bos_rate": float(no_opp_rate),
            "n": int(n_with_window),
            "lifetime_bars": LIFETIME_BARS,
            "definition": "hold_rate_mean = mean over events of (fraction of next-16-bar closes respecting BOS direction); no_opposite_bos_rate = fraction of events with no opposite-sign BOS_EVENT in next 16 bars",
        },
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_bos_for(inst, f"{sym} M15 OOS 2024+")
        results[sym] = out
        if out.get("skipped"):
            print(f"  skipped (no events)")
            continue
        q1lr = out["q1_level_reality"]
        q1f1 = out["q1_cross_method_f1"]
        q2 = out["q2_stability_16bars"]
        print(
            f"  events={out['n_events']:,d} bull={out['n_bull']:,d} bear={out['n_bear']:,d}\n"
            f"  Q1 sanity close-vs-level: {out['q1_sanity_close_vs_level']['value']:.4f}\n"
            f"  Q1 level reality (exact): {q1lr['exact_match_rate']:.4f}  "
            f"(envelope: {q1lr['envelope_rate']:.4f})\n"
            f"  Q1 cross-method F1 W=5:   {q1f1['f1_w5']:.3f} "
            f"[{q1f1['f1_w5_ci95_lo']:.3f}, {q1f1['f1_w5_ci95_hi']:.3f}]  "
            f"{q1f1['verdict_w5']}\n"
            f"    W=2: F1={q1f1['windows']['W2']['f1']:.3f}  "
            f"W=10: F1={q1f1['windows']['W10']['f1']:.3f}  "
            f"(prod={q1f1['n_prod_events']}, ref={q1f1['n_ref_events']})\n"
            f"  Q2 hold-rate @16bars:     {q2['hold_rate_mean']:.4f}  "
            f"[{q2['hold_rate_ci95_lo']:.3f}, {q2['hold_rate_ci95_hi']:.3f}]\n"
            f"    no-opposite-BOS rate:   {q2['no_opposite_bos_rate']:.4f}  "
            f"median time-to-recross: {q2['median_time_to_recross_bars']}"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "bos_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
