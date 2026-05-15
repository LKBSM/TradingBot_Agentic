"""Empirical audit for section 3.2 (Smart Money detection).

Reads-only. Computes firing-rate statistics, retest behaviour, sensitivity to
configurable parameters, cross-asset comparison and timing on the current
production CSVs.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment.strategy_features import (  # noqa: E402
    SmartMoneyEngine,
    calculate_bos_choch_fast,
    _calculate_bos_choch_python,
    calculate_bos_retest_fast,
    _calculate_bos_retest_python,
    SMCConfig,
    NUMBA_AVAILABLE,
)


def _load_csv(path: Path, nrows: int | None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def run_engine(df: pd.DataFrame, config: dict | None = None) -> tuple[pd.DataFrame, dict]:
    cfg = config or {}
    eng = SmartMoneyEngine(data=df.copy(), config=cfg, verbose=False)
    out = eng.analyze()
    return out, eng.get_timing_report()


def stats_from_output(out: pd.DataFrame) -> dict:
    n = len(out)
    s = {
        "n_bars": n,
        "bos_event_up": int((out["BOS_EVENT"] == 1).sum()),
        "bos_event_down": int((out["BOS_EVENT"] == -1).sum()),
        "bos_event_total": int((out["BOS_EVENT"] != 0).sum()),
        "bos_signal_up_bars": int((out["BOS_SIGNAL"] == 1).sum()),
        "bos_signal_down_bars": int((out["BOS_SIGNAL"] == -1).sum()),
        "choch_up": int((out["CHOCH_SIGNAL"] == 1).sum()),
        "choch_down": int((out["CHOCH_SIGNAL"] == -1).sum()),
        "fvg_up": int((out["FVG_SIGNAL"] == 1).sum()),
        "fvg_down": int((out["FVG_SIGNAL"] == -1).sum()),
        "fvg_dir_up_raw": int((out["FVG_DIR"] == 1).sum()),
        "fvg_dir_down_raw": int((out["FVG_DIR"] == -1).sum()),
        "ob_bullish": int(out["BULLISH_OB_HIGH"].notna().sum()),
        "ob_bearish": int(out["BEARISH_OB_HIGH"].notna().sum()),
        "retest_armed_up_bars": int((out["BOS_RETEST_ARMED"] == 1).sum()),
        "retest_armed_down_bars": int((out["BOS_RETEST_ARMED"] == -1).sum()),
        "retest_state_awaiting_up": int((out["BOS_RETEST_STATE"] == 1).sum()),
        "retest_state_awaiting_down": int((out["BOS_RETEST_STATE"] == -1).sum()),
        "retest_state_armed_up": int((out["BOS_RETEST_STATE"] == 2).sum()),
        "retest_state_armed_down": int((out["BOS_RETEST_STATE"] == -2).sum()),
        "up_fractals": int(out["UP_FRACTAL"].notna().sum()),
        "down_fractals": int(out["DOWN_FRACTAL"].notna().sum()),
        "divergence_bull": int((out["CHOCH_DIVERGENCE"] == 1).sum()),
        "divergence_bear": int((out["CHOCH_DIVERGENCE"] == -1).sum()),
    }
    s["bos_event_pct"] = round(100.0 * s["bos_event_total"] / n, 3)
    s["fvg_signal_pct"] = round(100.0 * (s["fvg_up"] + s["fvg_down"]) / n, 3)
    s["fvg_dir_pct_unfiltered"] = round(100.0 * (s["fvg_dir_up_raw"] + s["fvg_dir_down_raw"]) / n, 3)
    s["ob_pct"] = round(100.0 * (s["ob_bullish"] + s["ob_bearish"]) / n, 3)
    s["retest_armed_pct"] = round(100.0 * (s["retest_armed_up_bars"] + s["retest_armed_down_bars"]) / n, 3)
    return s


def compute_fractal_to_bos_latency(out: pd.DataFrame) -> dict:
    """Median bars since last fractal to BOS event (sanity check on latency)."""
    res = {"bos_up": None, "bos_down": None}
    closes_idx = np.arange(len(out))
    up_fr_idx = np.where(out["UP_FRACTAL"].notna().values)[0]
    down_fr_idx = np.where(out["DOWN_FRACTAL"].notna().values)[0]
    bos_up_idx = np.where(out["BOS_EVENT"].values == 1)[0]
    bos_down_idx = np.where(out["BOS_EVENT"].values == -1)[0]

    def _latencies(events_idx, fr_idx):
        if len(events_idx) == 0 or len(fr_idx) == 0:
            return None
        lat = []
        j = 0
        for e in events_idx:
            # advance j to latest fractal at <= e
            while j + 1 < len(fr_idx) and fr_idx[j + 1] <= e:
                j += 1
            if fr_idx[j] <= e:
                lat.append(e - fr_idx[j])
        if not lat:
            return None
        arr = np.array(lat)
        return {
            "n": int(arr.size),
            "mean": round(float(arr.mean()), 2),
            "median": float(np.median(arr)),
            "p25": float(np.percentile(arr, 25)),
            "p75": float(np.percentile(arr, 75)),
            "max": int(arr.max()),
        }

    res["bos_up"] = _latencies(bos_up_idx, up_fr_idx)
    res["bos_down"] = _latencies(bos_down_idx, down_fr_idx)
    return res


def reproducibility_check(df: pd.DataFrame) -> dict:
    """Run engine twice, check that outputs are byte-identical."""
    out1, _ = run_engine(df.copy())
    out2, _ = run_engine(df.copy())
    cols = ["BOS_EVENT", "BOS_SIGNAL", "BOS_BREAK_LEVEL", "CHOCH_SIGNAL",
            "FVG_SIGNAL", "FVG_SIZE", "FVG_SIZE_NORM",
            "BULLISH_OB_HIGH", "BEARISH_OB_HIGH", "OB_STRENGTH_NORM",
            "BOS_RETEST_STATE", "BOS_RETEST_ARMED", "CHOCH_DIVERGENCE",
            "RSI", "ATR"]
    rep = {}
    for c in cols:
        a = out1[c].values
        b = out2[c].values
        if a.dtype.kind == "f":
            same = np.allclose(np.nan_to_num(a, nan=-9.999), np.nan_to_num(b, nan=-9.999))
        else:
            same = np.array_equal(a, b)
        rep[c] = bool(same)
    rep["all_match"] = bool(all(rep.values()))
    return rep


def numba_python_parity(df: pd.DataFrame) -> dict:
    """Compare BOS Numba vs Python fallback on a 5k bar sample."""
    smc = SmartMoneyEngine(data=df.copy(), config={}, verbose=False)
    # Need to compute ATR + fractals first
    smc._add_ta_indicators()
    smc._add_smc_base_features()

    closes = smc.df["close"].values.astype(np.float64)
    highs = smc.df["high"].values.astype(np.float64)
    lows = smc.df["low"].values.astype(np.float64)
    up_fr = smc.df["UP_FRACTAL"].values.astype(np.float64)
    dn_fr = smc.df["DOWN_FRACTAL"].values.astype(np.float64)

    bs_n, ch_n, ev_n, brk_n = calculate_bos_choch_fast(closes, highs, lows, up_fr, dn_fr)
    bs_p, ch_p, ev_p, brk_p = _calculate_bos_choch_python(closes, highs, lows, up_fr, dn_fr)

    parity = {
        "bos_signal": bool(np.array_equal(bs_n, bs_p)),
        "choch_signal": bool(np.array_equal(ch_n, ch_p)),
        "bos_event": bool(np.array_equal(ev_n, ev_p)),
        "bos_break_level_close": bool(
            np.allclose(np.nan_to_num(brk_n, nan=-9.999), np.nan_to_num(brk_p, nan=-9.999))
        ),
    }
    parity["all_parity"] = all(parity.values())

    # Retest parity
    atr = smc.df["ATR"].values.astype(np.float64)
    rs_n, ra_n = calculate_bos_retest_fast(closes, highs, lows, ev_n, brk_n, atr)
    rs_p, ra_p = _calculate_bos_retest_python(
        closes, highs, lows, ev_n, brk_n, atr,
        retest_tol_atr=0.5, invalid_tol_atr=1.0,
        awaiting_timeout=20, armed_window=30,
    )
    parity["retest_state_match"] = bool(np.array_equal(rs_n, rs_p))
    parity["retest_armed_match"] = bool(np.array_equal(ra_n, ra_p))
    return parity


def sensitivity_sweep(df: pd.DataFrame) -> dict:
    """Sweep FVG_THRESHOLD and RETEST_TOL_ATR, report counts."""
    base_n = len(df)
    fvg_sweep = {}
    for thr in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        out, _ = run_engine(df, {"FVG_THRESHOLD": thr})
        n_fvg = int((out["FVG_SIGNAL"] != 0).sum())
        fvg_sweep[str(thr)] = {
            "n_fvg": n_fvg,
            "fvg_per_1k_bars": round(1000.0 * n_fvg / len(out), 2),
        }

    retest_sweep = {}
    for tol in [0.1, 0.25, 0.5, 0.75, 1.0]:
        out, _ = run_engine(df, {"RETEST_TOL_ATR": tol})
        n_armed_up = int((out["BOS_RETEST_ARMED"] == 1).sum())
        n_armed_dn = int((out["BOS_RETEST_ARMED"] == -1).sum())
        retest_sweep[str(tol)] = {
            "armed_up_bars": n_armed_up,
            "armed_down_bars": n_armed_dn,
            "armed_per_1k_bars": round(1000.0 * (n_armed_up + n_armed_dn) / len(out), 2),
        }

    fractal_sweep = {}
    for N in [2, 3, 4, 5]:
        out, _ = run_engine(df, {"FRACTAL_WINDOW": N})
        fractal_sweep[str(N)] = {
            "up_fractals": int(out["UP_FRACTAL"].notna().sum()),
            "down_fractals": int(out["DOWN_FRACTAL"].notna().sum()),
            "bos_events": int((out["BOS_EVENT"] != 0).sum()),
        }

    ob_req_fvg = {}
    for flag in [False, True]:
        out, _ = run_engine(df, {"OB_REQUIRE_FVG": flag})
        ob_req_fvg[str(flag)] = {
            "ob_bullish": int(out["BULLISH_OB_HIGH"].notna().sum()),
            "ob_bearish": int(out["BEARISH_OB_HIGH"].notna().sum()),
        }

    return {
        "fvg_threshold_sweep": fvg_sweep,
        "retest_tol_sweep": retest_sweep,
        "fractal_window_sweep": fractal_sweep,
        "ob_require_fvg_sweep": ob_req_fvg,
    }


def perf_bench(df: pd.DataFrame, n_runs: int = 3) -> dict:
    """Time the analyze() pipeline on the given DataFrame."""
    timings = []
    breakdowns = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        _, br = run_engine(df)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
        breakdowns.append(br)
    return {
        "n_bars": len(df),
        "wall_clock_runs_s": [round(t, 3) for t in timings],
        "median_wall_clock_s": round(float(np.median(timings)), 3),
        "median_breakdown_s": {
            k: round(float(np.median([b.get(k, 0.0) for b in breakdowns])), 4)
            for k in breakdowns[0].keys()
        },
        "numba_available": bool(NUMBA_AVAILABLE),
    }


def ob_anchor_check(df_out: pd.DataFrame) -> dict:
    """For each detected OB, how many are within +/- 20 bars of a BOS event?
    A canonical ICT OB sits immediately before the impulse that creates the BOS.
    """
    bos_idx = np.where(df_out["BOS_EVENT"].values != 0)[0]
    bull_ob_idx = np.where(df_out["BULLISH_OB_HIGH"].notna().values)[0]
    bear_ob_idx = np.where(df_out["BEARISH_OB_HIGH"].notna().values)[0]

    def _near(target_idx, ref_idx, win=20):
        if len(ref_idx) == 0:
            return None
        # for each target, find nearest ref distance
        ref_sorted = np.sort(ref_idx)
        near_count = 0
        for t in target_idx:
            j = np.searchsorted(ref_sorted, t)
            best = float("inf")
            if j < len(ref_sorted):
                best = min(best, abs(ref_sorted[j] - t))
            if j > 0:
                best = min(best, abs(t - ref_sorted[j - 1]))
            if best <= win:
                near_count += 1
        return {
            "ob_count": int(len(target_idx)),
            "near_bos_count": int(near_count),
            "near_bos_pct": round(100.0 * near_count / max(1, len(target_idx)), 1),
        }

    return {
        "bullish_within_20_bars_of_bos": _near(bull_ob_idx, bos_idx),
        "bearish_within_20_bars_of_bos": _near(bear_ob_idx, bos_idx),
        "n_bos_events": int(len(bos_idx)),
    }


def main():
    out_dir = ROOT / "audits" / "2026-Q2"
    out_dir.mkdir(parents=True, exist_ok=True)

    xau_path = ROOT / "data" / "XAU_15MIN_2019_2026.csv"
    eur_path = ROOT / "data" / "EURUSD_15MIN_2019_2025.csv"

    results: dict = {"meta": {
        "csv_xau": str(xau_path.name),
        "csv_eur": str(eur_path.name),
        "numba_available": bool(NUMBA_AVAILABLE),
        "generated_at": pd.Timestamp.utcnow().isoformat(),
    }}

    print("[1/8] Loading XAU 20 000 bars...")
    df_xau = _load_csv(xau_path, nrows=20000)
    results["xau_20k"] = {"n_bars": len(df_xau)}

    out_xau, timing_xau = run_engine(df_xau)
    results["xau_20k"]["stats"] = stats_from_output(out_xau)
    results["xau_20k"]["timing"] = {k: round(v, 4) for k, v in timing_xau.items()}
    results["xau_20k"]["latency"] = compute_fractal_to_bos_latency(out_xau)
    results["xau_20k"]["ob_anchor"] = ob_anchor_check(out_xau)

    print("[2/8] Reproducibility (XAU)...")
    results["xau_20k"]["reproducibility"] = reproducibility_check(df_xau)

    print("[3/8] Numba vs Python parity (5k bars)...")
    results["xau_5k"] = {"parity": numba_python_parity(df_xau.head(5000))}

    print("[4/8] Sensitivity sweep (XAU 10k bars)...")
    results["xau_sensitivity"] = sensitivity_sweep(df_xau.head(10000))

    print("[5/8] Loading XAU FULL (172k bars)...")
    df_full = _load_csv(xau_path, nrows=None)
    results["xau_full"] = {"n_bars": len(df_full)}
    out_full, timing_full = run_engine(df_full)
    results["xau_full"]["stats"] = stats_from_output(out_full)
    results["xau_full"]["timing"] = {k: round(v, 4) for k, v in timing_full.items()}
    results["xau_full"]["latency"] = compute_fractal_to_bos_latency(out_full)
    results["xau_full"]["ob_anchor"] = ob_anchor_check(out_full)

    print("[6/8] Loading EURUSD 20 000 bars...")
    df_eur = _load_csv(eur_path, nrows=20000)
    results["eur_20k"] = {"n_bars": len(df_eur)}
    out_eur, timing_eur = run_engine(df_eur)
    results["eur_20k"]["stats"] = stats_from_output(out_eur)
    results["eur_20k"]["timing"] = {k: round(v, 4) for k, v in timing_eur.items()}
    results["eur_20k"]["ob_anchor"] = ob_anchor_check(out_eur)

    print("[7/8] Performance bench on 20k bars...")
    results["perf_20k"] = perf_bench(df_xau, n_runs=3)

    print("[8/8] Per-year stats (XAU full)...")
    per_year = {}
    for year in sorted(set(out_full.index.year.tolist())):
        mask = out_full.index.year == year
        sub = out_full[mask]
        if len(sub) < 100:
            continue
        per_year[str(year)] = {
            "bars": int(len(sub)),
            "bos_up": int((sub["BOS_EVENT"] == 1).sum()),
            "bos_down": int((sub["BOS_EVENT"] == -1).sum()),
            "choch_up": int((sub["CHOCH_SIGNAL"] == 1).sum()),
            "choch_down": int((sub["CHOCH_SIGNAL"] == -1).sum()),
            "fvg_up": int((sub["FVG_SIGNAL"] == 1).sum()),
            "fvg_down": int((sub["FVG_SIGNAL"] == -1).sum()),
            "ob_bullish": int(sub["BULLISH_OB_HIGH"].notna().sum()),
            "ob_bearish": int(sub["BEARISH_OB_HIGH"].notna().sum()),
        }
    results["xau_full"]["per_year"] = per_year

    out_json = out_dir / "section_3_2_smart_money_stats.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nWrote: {out_json}")
    print(json.dumps({k: results[k].get("n_bars", "n/a") if isinstance(results[k], dict) else "n/a" for k in results}, indent=2))


if __name__ == "__main__":
    main()
