"""MT-D1 — compute all quantitative results. Writes results.json. Engine READ-ONLY."""
from __future__ import annotations

import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from scripts.audit.mt_d1 import harness as H
from scripts.audit.mt_d1 import variant_detect as V

OUT = Path("docs/audits/ECHANTILLON-DETECTION-2026-07-29")


def arrays(enr):
    return (
        enr["close"].values.astype(float),
        enr["high"].values.astype(float),
        enr["low"].values.astype(float),
        enr["UP_FRACTAL"].values.astype(float),
        enr["DOWN_FRACTAL"].values.astype(float),
        enr["ATR"].values.astype(float),
    )


def parity_check(enr):
    c, h, l, uf, df_, atr = arrays(enr)
    res = V.detect(c, h, l, uf, df_, atr, cross_mode="close", margin_atr=0.0)
    eng_bos = enr["BOS_EVENT"].values.astype(int)
    eng_ch = enr["CHOCH_SIGNAL"].values.astype(int)
    ok_bos = np.array_equal(res["bos_event"], eng_bos)
    ok_ch = np.array_equal(res["choch_signal"], eng_ch)
    return bool(ok_bos and ok_ch), int((eng_bos != 0).sum()), int((eng_ch != 0).sum())


def in_window(enr, days):
    now = enr.index[-1]
    cut = now - timedelta(days=days)
    return [i for i, ts in enumerate(enr.index) if ts >= cut]


def main():
    results = {}
    for tf in ("H1", "M15"):
        candles = H.load_candles(tf)
        enr = H.enrich(candles)  # default engine
        c, h, l, uf, df_, atr = arrays(enr)

        parity_ok, n_bos_eng, n_ch_eng = parity_check(enr)

        # ---- Section 6: deep vs surface ----
        ev_all = H.extract_events(enr)
        sec6 = {}
        for days in (30, 60):
            idxs = set(in_window(enr, days))
            b = sum(1 for e in ev_all if e["k"] in idxs and e["type"] == "BOS")
            ch = sum(1 for e in ev_all if e["k"] in idxs and e["type"] == "CHOCH")
            sec6[f"deep_{days}d"] = {"bos": b, "choch": ch, "total": b + ch}
        surf500 = H.surface_events(enr, window_bars=500, cap=8)
        sec6["surface_live"] = {
            "window_bars": surf500["window_bars"],
            "bos_in_window": surf500["bos_in_window"],
            "choch_in_window": surf500["choch_in_window"],
            "bos_surfaced": surf500["bos_surfaced"],
            "choch_surfaced": surf500["choch_surfaced"],
        }
        # window span in calendar/trading terms
        win = enr.iloc[-500:] if len(enr) > 500 else enr
        span_days = (win.index[-1] - win.index[0]).total_seconds() / 86400.0
        sec6["surface_live"]["window_calendar_days"] = round(span_days, 1)

        # ---- Section 7: sensitivity, over last 60 days ----
        win60 = set(in_window(enr, 60))
        sens = {}
        # (a) FRACTAL_WINDOW via REAL engine
        fw = {}
        for N in (1, 2, 3, 4):
            e2 = H.enrich(candles, fractal_window=N)
            ev2 = H.extract_events(e2)
            idx60 = set(in_window(e2, 60))
            b = sum(1 for e in ev2 if e["k"] in idx60 and e["type"] == "BOS")
            ch = sum(1 for e in ev2 if e["k"] in idx60 and e["type"] == "CHOCH")
            fw[N] = {"bos": b, "choch": ch, "total": b + ch}
        sens["fractal_window"] = fw
        ref_total = fw[2]["total"]

        # (b) cross_mode wick vs close (variant, default fractals N=2)
        cm = {}
        for mode in ("close", "wick"):
            r = V.detect(c, h, l, uf, df_, atr, cross_mode=mode, margin_atr=0.0)
            b = sum(1 for k in range(len(enr)) if k in win60 and r["bos_event"][k] != 0)
            ch = sum(1 for k in range(len(enr)) if k in win60 and r["choch_signal"][k] != 0)
            cm[mode] = {"bos": b, "choch": ch, "total": b + ch}
        sens["cross_mode"] = cm

        # (c) margin_atr (variant): 0 (engine) vs stricter margins
        mg = {}
        for k_atr in (0.0, 0.1, 0.25, 0.5):
            r = V.detect(c, h, l, uf, df_, atr, cross_mode="close", margin_atr=k_atr)
            b = sum(1 for k in range(len(enr)) if k in win60 and r["bos_event"][k] != 0)
            ch = sum(1 for k in range(len(enr)) if k in win60 and r["choch_signal"][k] != 0)
            mg[f"{k_atr}"] = {"bos": b, "choch": ch, "total": b + ch}
        sens["margin_atr"] = mg
        sens["reference_total_60d"] = ref_total

        # ---- blocked crossings (non-events) full list, default engine ----
        r_def = V.detect(c, h, l, uf, df_, atr)
        results[tf] = {
            "parity_ok": parity_ok,
            "n_bars": len(enr),
            "first_ts": str(enr.index[0]), "last_ts": str(enr.index[-1]),
            "n_bos_engine": n_bos_eng, "n_choch_engine": n_ch_eng,
            "section6": sec6,
            "section7": sens,
            "n_blocked": len(r_def["blocked"]),
            "blocked_by_reason": {
                reason: sum(1 for x in r_def["blocked"] if x["reason"] == reason)
                for reason in ("wick_only_close_inside", "level_already_broken")
            },
        }
        print(f"{tf}: parity={parity_ok} bars={len(enr)} "
              f"BOS={n_bos_eng} CHOCH={n_ch_eng} blocked={len(r_def['blocked'])}")

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print("wrote results.json")
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
