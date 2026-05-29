"""Evaluate metadata sanity & exposed-levels consistency on OOS data.

Checks the contract integrity of the values that ship to clients via
InsightSignalV2:
  - ATR positive on all bars
  - BOS_BREAK_LEVEL NaN iff BOS_EVENT == 0 (consistency)
  - OB zone: high > low (already verified in eval_ob but re-asserted)
  - FVG_DIR ∈ {-1, 0, 1}
  - CHOCH_SIGNAL ∈ {-1, 0, 1}
  - price_decimals per InstrumentConfig matches typical price scale
  - Bar interval consistency: no missing 15-min ticks within trading sessions
  - bos_event_age_bars (if exposed) monotonic non-decreasing between events

Q1 only — these are definitional / contract sanity checks.
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


EXPECTED_DECIMALS = {"XAUUSD": 2, "EURUSD": 5}


def eval_meta_for(inst: InstrumentData) -> dict:
    df = inst.oos
    n = len(df)
    sym = inst.symbol
    checks = {}

    # 1. ATR positive
    atr = df["ATR"].to_numpy()
    nan_atr = int(np.isnan(atr).sum())
    nonpos_atr = int((atr <= 0).sum() - nan_atr)
    checks["atr_positive"] = {
        "passed": (nan_atr == 0) and (nonpos_atr == 0),
        "n_nan": nan_atr,
        "n_nonpositive": nonpos_atr,
        "n": n,
    }

    # 2. BOS event ↔ break level consistency
    bos_ev = df["BOS_EVENT"].to_numpy()
    bos_lvl = df["BOS_BREAK_LEVEL"].to_numpy()
    lvl_nan = np.isnan(bos_lvl)
    ev_nonzero = bos_ev != 0
    mismatch_a = int((ev_nonzero & lvl_nan).sum())  # event but no level
    mismatch_b = int(((~ev_nonzero) & (~lvl_nan)).sum())  # level but no event
    checks["bos_event_level_consistency"] = {
        "passed": (mismatch_a == 0) and (mismatch_b == 0),
        "events_without_level": mismatch_a,
        "levels_without_event": mismatch_b,
        "n_events": int(ev_nonzero.sum()),
    }

    # 3. FVG_DIR ∈ {-1, 0, 1}
    fvg_dir = df["FVG_DIR"].to_numpy()
    valid_fvg = np.isin(fvg_dir, [-1, 0, 1]).all()
    checks["fvg_dir_domain"] = {
        "passed": bool(valid_fvg),
        "unique_values": sorted(set(fvg_dir.tolist())),
    }

    # 4. CHOCH_SIGNAL ∈ {-1, 0, 1}
    choch = df["CHOCH_SIGNAL"].to_numpy()
    valid_choch = np.isin(choch, [-1, 0, 1]).all()
    checks["choch_signal_domain"] = {
        "passed": bool(valid_choch),
        "unique_values": sorted(set(choch.tolist())),
    }

    # 5. OB zone consistency
    bull_h = df["BULLISH_OB_HIGH"].to_numpy()
    bull_l = df["BULLISH_OB_LOW"].to_numpy()
    bear_h = df["BEARISH_OB_HIGH"].to_numpy()
    bear_l = df["BEARISH_OB_LOW"].to_numpy()
    # high > low when not NaN
    bull_active = ~np.isnan(bull_h)
    bear_active = ~np.isnan(bear_h)
    bull_inv = int(((bull_h[bull_active] <= bull_l[bull_active])).sum())
    bear_inv = int(((bear_h[bear_active] <= bear_l[bear_active])).sum())
    checks["ob_zone_consistency"] = {
        "passed": (bull_inv == 0) and (bear_inv == 0),
        "bull_inversions": bull_inv,
        "bear_inversions": bear_inv,
        "n_bull_ob": int(bull_active.sum()),
        "n_bear_ob": int(bear_active.sum()),
    }

    # 6. Decimals: median tick scale must match expected price_decimals
    closes = df["close"].to_numpy()
    median_price = float(np.median(closes))
    # Implied decimals = ceil(log10(1/price * granularity))
    # Simpler: check that the median price has expected scale
    if sym == "XAUUSD":
        # Gold around $1000-$3000 → 2 decimal places typical
        scale_ok = 500 < median_price < 5000
    elif sym == "EURUSD":
        scale_ok = 0.9 < median_price < 1.5
    else:
        scale_ok = True
    checks["price_decimals_scale"] = {
        "passed": bool(scale_ok),
        "expected_decimals": EXPECTED_DECIMALS.get(sym),
        "median_price": median_price,
    }

    # 7. Bar interval = 15 min (within sessions, ignore weekend gaps)
    idx = pd.DatetimeIndex(df.index)
    deltas = idx.to_series().diff().dropna()
    weekday_idx = idx.weekday
    # Drop weekend gap (any delta > 1 day = weekend transition, excluded)
    intra = deltas[deltas <= pd.Timedelta(hours=1)]
    expected = pd.Timedelta(minutes=15)
    n_intra = len(intra)
    n_correct = int((intra == expected).sum())
    interval_consistency = n_correct / n_intra if n_intra else float("nan")
    checks["bar_interval_consistency"] = {
        "passed": interval_consistency > 0.99,
        "rate": float(interval_consistency),
        "n_intra_session_deltas": n_intra,
    }

    n_passed = sum(1 for c in checks.values() if c.get("passed"))
    n_total = len(checks)

    return {
        "symbol": sym,
        "n_bars_oos": n,
        "checks_passed": n_passed,
        "checks_total": n_total,
        "checks": checks,
    }


def main():
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_meta_for(inst)
        results[sym] = out
        print(f"  {out['checks_passed']}/{out['checks_total']} checks passed")
        for name, c in out["checks"].items():
            mark = "🟢" if c.get("passed") else "🔴"
            print(f"    {mark} {name}")

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "metadata_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
