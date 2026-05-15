"""SL/TP multiplier sweep on the *filtered* XAU M15 strategy.

Approach: simulation. We re-use the existing trade entry list
(`reports/audit/trades_combined.csv`), apply the validated filters
(skip NY session, skip ATR_PCTL > 0.75), then for each (sl_mult, tp_mult)
combo we walk forward from entry_bar over the next 12 bars and check which
of {SL, TP, time-out} is hit first. r_multiple is recomputed against the
new SL distance.

Output: reports/sweep_sl_tp.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.environment.strategy_features import SmartMoneyEngine

SESSION_BINS = [(0, 7, "Asian"), (7, 13, "London"), (13, 21, "NY"), (21, 24, "OffHours")]
MAX_BARS = 12
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_GRID = [1.5, 2.0, 3.0, 4.0, 5.0]
ATR_PCTL_FILTER = 0.75


def session_label(h):
    for lo, hi, n in SESSION_BINS:
        if lo <= h < hi:
            return n
    return "OffHours"


def metrics(r):
    if len(r) == 0:
        return {"n": 0, "win": np.nan, "exp": np.nan, "pf": np.nan, "tot": np.nan}
    wins = r[r > 0].sum()
    losses = -r[r < 0].sum()
    pf = wins / losses if losses > 0 else (999.0 if wins > 0 else np.nan)
    return {
        "n": int(len(r)),
        "win": float((r > 0).mean()),
        "exp": float(r.mean()),
        "pf": float(pf),
        "tot": float(r.sum()),
    }


def fmt(v, spec=".3f"):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:{spec}}" if isinstance(v, float) else str(v)


def simulate(trades, ohlc_idx, highs, lows, closes, sl_mult, tp_mult):
    """Vectorised-ish simulation. For each trade row, walk forward MAX_BARS."""
    out_r = np.zeros(len(trades))
    for i, row in enumerate(trades.itertuples()):
        atr = row.atr_entry
        if not np.isfinite(atr) or atr <= 0:
            out_r[i] = np.nan
            continue
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        entry_price = row.entry_price
        is_long = row.direction == "LONG"
        if is_long:
            sl_lvl, tp_lvl = entry_price - sl_dist, entry_price + tp_dist
        else:
            sl_lvl, tp_lvl = entry_price + sl_dist, entry_price - tp_dist

        start = row.entry_idx + 1  # next bar after entry
        end = min(start + MAX_BARS, len(ohlc_idx))
        if start >= end:
            out_r[i] = 0.0
            continue

        seg_high = highs[start:end]
        seg_low = lows[start:end]

        if is_long:
            sl_hit = seg_low <= sl_lvl
            tp_hit = seg_high >= tp_lvl
        else:
            sl_hit = seg_high >= sl_lvl
            tp_hit = seg_low <= tp_lvl

        sl_idx = np.argmax(sl_hit) if sl_hit.any() else -1
        tp_idx = np.argmax(tp_hit) if tp_hit.any() else -1

        if sl_idx == -1 and tp_idx == -1:
            # time-out: exit at close of last bar
            exit_price = closes[end - 1]
            r = (exit_price - entry_price) / sl_dist if is_long else (entry_price - exit_price) / sl_dist
        elif tp_idx == -1:
            r = -1.0  # SL hit cleanly
        elif sl_idx == -1:
            r = tp_dist / sl_dist  # TP hit cleanly
        else:
            # Both hit — ambiguity. Conservative: assume SL first if same bar.
            if sl_idx <= tp_idx:
                r = -1.0
            else:
                r = tp_dist / sl_dist
        out_r[i] = r
    return out_r


def main():
    csv_path = Path("data/XAU_15MIN_2019_2026.csv")
    trades_path = Path("reports/audit/trades_combined.csv")
    out_path = Path("reports/sweep_sl_tp.md")

    df = pd.read_csv(csv_path, parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print(f"[sweep] enriching {len(df)} bars...")
    enr = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    enr["hour"] = enr.index.hour
    enr["session"] = enr["hour"].map(session_label)
    enr["ATR_PCTL"] = enr["ATR"].fillna(0).rolling(30 * 96, min_periods=200).rank(pct=True)

    trades = pd.read_csv(trades_path)
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    trades = trades.merge(
        enr[["session", "ATR_PCTL", "ATR"]].rename(columns={"ATR": "atr_entry"}),
        left_on="entry_bar", right_index=True, how="inner",
    )
    print(f"[sweep] {len(trades)} trades after merge")

    # Apply filters: skip NY + skip ATR_PCTL>0.75
    n_pre = len(trades)
    trades = trades[(trades["session"] != "NY") & (trades["ATR_PCTL"] <= ATR_PCTL_FILTER)].copy()
    print(f"[sweep] filtered: {len(trades)} (dropped {n_pre - len(trades)})")

    # Map entry_bar -> integer index in ohlc
    idx_map = {ts: i for i, ts in enumerate(enr.index)}
    trades["entry_idx"] = trades["entry_bar"].map(idx_map)
    trades = trades.dropna(subset=["entry_idx", "atr_entry"])
    trades["entry_idx"] = trades["entry_idx"].astype(int)
    trades["set"] = np.where(trades["entry_bar"] < "2023-01-01", "TRAIN", "TEST")

    highs = enr["high"].to_numpy()
    lows = enr["low"].to_numpy()
    closes = enr["close"].to_numpy()
    ohlc_idx = enr.index

    print(f"[sweep] running {len(SL_GRID)} × {len(TP_GRID)} combos on {len(trades)} trades...")
    results = []  # list of dict
    for sl in SL_GRID:
        for tp in TP_GRID:
            if tp <= sl:
                continue
            r_arr = simulate(trades, ohlc_idx, highs, lows, closes, sl, tp)
            trades["_r_sim"] = r_arr
            tr = metrics(trades.loc[trades["set"] == "TRAIN", "_r_sim"].dropna())
            te = metrics(trades.loc[trades["set"] == "TEST", "_r_sim"].dropna())
            stable = (
                te["n"] >= 100
                and te["pf"] >= 1.30
                and (tr["pf"] - 1) * (te["pf"] - 1) > 0
            )
            results.append({
                "sl": sl, "tp": tp, "rr": tp / sl,
                "n_train": tr["n"], "pf_train": tr["pf"], "exp_train": tr["exp"],
                "n_test": te["n"], "pf_test": te["pf"], "win_test": te["win"],
                "exp_test": te["exp"], "tot_test": te["tot"],
                "stable": stable,
            })

    # Build report
    md = ["# SL/TP Sweep — Filtered XAU M15 (skip NY + skip ATR_PCTL>0.75)\n"]
    md.append(f"**Trades after filter**: {len(trades)} ({(trades['set']=='TRAIN').sum()} train / {(trades['set']=='TEST').sum()} test)")
    md.append(f"**Simulation**: walk-forward {MAX_BARS} bars, SL/TP intrabar (high/low), tie → SL")
    md.append(f"**Baseline current**: SL=2.0× ATR, TP=4.0× ATR (R:R 1:2)\n---\n")

    # Heatmap of PF_test (rows=SL, cols=TP)
    md.append("## PF_test heatmap (rows=SL, cols=TP)\n")
    header = "| SL \\ TP | " + " | ".join(f"{tp:.1f}" for tp in TP_GRID) + " |"
    md.append(header)
    md.append("|" + "|".join(["---"] * (len(TP_GRID) + 1)) + "|")
    grid = {(r["sl"], r["tp"]): r for r in results}
    for sl in SL_GRID:
        cells = [f"{sl:.1f}"]
        for tp in TP_GRID:
            if tp <= sl:
                cells.append("—")
            else:
                r = grid.get((sl, tp))
                if r is None:
                    cells.append("—")
                else:
                    mark = "✅" if r["stable"] else ""
                    cells.append(f"{r['pf_test']:.2f}{mark}")
        md.append("| " + " | ".join(cells) + " |")

    # Sort by pf_test desc and show top-3 + baseline
    md.append("\n## Top-3 by PF_test (n≥100)\n")
    md.append("| rank | SL | TP | R:R | n_train | PF_train | exp_train | n_test | PF_test | win_test | exp_test | total_R_test | stable |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    sorted_r = sorted(
        [r for r in results if r["n_test"] >= 100],
        key=lambda x: -x["pf_test"] if not np.isnan(x["pf_test"]) else 0,
    )
    for i, r in enumerate(sorted_r[:3], 1):
        md.append(
            f"| {i} | {r['sl']:.1f} | {r['tp']:.1f} | {r['rr']:.2f} | {r['n_train']} | {fmt(r['pf_train'])} | {fmt(r['exp_train'])} | {r['n_test']} | {fmt(r['pf_test'])} | {fmt(r['win_test'])} | {fmt(r['exp_test'])} | {fmt(r['tot_test'])} | {'✅' if r['stable'] else ''} |"
        )

    # Baseline row for comparison (SL=2, TP=4)
    base = grid.get((2.0, 4.0))
    if base:
        md.append("\n## Baseline (SL=2.0, TP=4.0)\n")
        md.append(f"- TRAIN: n={base['n_train']}, PF={fmt(base['pf_train'])}, exp_R={fmt(base['exp_train'])}")
        md.append(f"- TEST:  n={base['n_test']}, PF={fmt(base['pf_test'])}, win={fmt(base['win_test'])}, exp_R={fmt(base['exp_test'])}, total_R={fmt(base['tot_test'])}")

    # Decision
    md.append("\n## Decision\n")
    if sorted_r:
        best = sorted_r[0]
        if best["pf_test"] >= 1.50 and best["n_test"] >= 200 and best["stable"]:
            md.append(f"**ADOPT new config**: SL={best['sl']:.1f}× ATR, TP={best['tp']:.1f}× ATR (R:R {best['rr']:.2f}). PF_test={best['pf_test']:.3f}, n={best['n_test']}, vs baseline {fmt(base['pf_test'] if base else None)}.")
        elif best["pf_test"] >= 1.30 and best["stable"]:
            base_pf = base["pf_test"] if base else 0
            uplift = best["pf_test"] - base_pf
            md.append(f"**MARGINAL**: best is SL={best['sl']:.1f}/TP={best['tp']:.1f} (PF_test {best['pf_test']:.3f}, uplift +{uplift:.3f} vs baseline {fmt(base_pf)}). Below 1.50 threshold — keep current config or adopt only if reproducible on a forward sample.")
        else:
            md.append(f"**NO IMPROVEMENT**: best PF_test = {best['pf_test']:.3f} (SL={best['sl']:.1f}/TP={best['tp']:.1f}), below 1.30 threshold.")
    else:
        md.append("No combo has n≥100 in test set — sample too small to decide.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[sweep] wrote {out_path}")


if __name__ == "__main__":
    main()
