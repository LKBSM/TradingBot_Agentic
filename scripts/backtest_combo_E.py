"""Combo test: production filter (B) + alternate TP multiplier (D).

Re-runs the full XAU M15 replay with TP_ATR_MULT swept across {3, 4, 5, 6}
keeping SL_ATR_MULT=2.0, then applies the production RegimeFilter to the
resulting trades. Reports PF before/after filter for each TP.

Question answered: does TP=5 (the marginal D winner) actually compound with
B's regime filter, or is the +0.08 PF a simulation artefact that washes out
under the real state-machine exit logic?
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence import confluence_detector as cd_mod
from src.intelligence.confluence_detector import ConfluenceDetector
from src.intelligence.regime_filter import RegimeFilter
from src.intelligence.signal_state_machine import StateMachineConfig
from src.backtest.state_machine_replay import SignalReplay


def metrics(r: pd.Series) -> dict:
    if len(r) == 0:
        return {"n": 0, "pf": np.nan, "win": np.nan, "exp": np.nan, "tot": 0.0}
    w = r[r > 0].sum()
    l = -r[r < 0].sum()
    pf = w / l if l > 0 else float("inf")
    return {"n": len(r), "pf": float(pf) if pf != np.inf else 999.0,
            "win": float((r > 0).mean()), "exp": float(r.mean()),
            "tot": float(r.sum())}


def run_one(enriched: pd.DataFrame, sl_mult: float, tp_mult: float) -> pd.DataFrame:
    """Patch detector constants, run replay, return trade DataFrame."""
    cd_mod.SL_ATR_MULT = sl_mult
    cd_mod.TP_ATR_MULT = tp_mult
    cfg = StateMachineConfig(
        symbol="XAUUSD", enter_threshold=40.0, exit_threshold=25.0,
        confirm_bars=2, cooldown_bars=2, max_signal_age_bars=12,
    )
    detector = ConfluenceDetector(symbol="XAUUSD", min_score=25.0, require_retest=True)
    detector._sl_atr_mult = sl_mult
    detector._tp_atr_mult = tp_mult
    replay = SignalReplay(
        symbol="XAUUSD", timeframe="M15", state_machine_config=cfg,
        confluence_detector=detector, use_regime=True, use_vol_regime=True,
        warmup_bars=100,
    )
    res = replay.run(enriched)
    rows = [t.to_dict() for t in res.trades]
    df = pd.DataFrame(rows)
    return df


def apply_filter(trades: pd.DataFrame, atr_series: pd.Series) -> pd.DataFrame:
    rf = RegimeFilter()
    keep = []
    for _, row in trades.iterrows():
        ts = pd.to_datetime(row["entry_bar"])
        atr_hist = atr_series.loc[:ts]
        keep.append(rf.evaluate(str(ts), atr_hist).allowed)
    out = trades.copy()
    out["kept"] = keep
    return out


def report(name: str, trades: pd.DataFrame) -> dict:
    trades["entry_bar"] = pd.to_datetime(trades["entry_bar"])
    test = trades[trades["entry_bar"] >= "2023-01-01"]
    test_kept = test[test["kept"]] if "kept" in test.columns else test
    raw = metrics(test["r_multiple"])
    filt = metrics(test_kept["r_multiple"])
    print(f"\n{name}:")
    print(f"  RAW    : n={raw['n']:>4d}  PF={raw['pf']:.3f}  win={raw['win']:.3f}  exp={raw['exp']:+.3f}  tot={raw['tot']:+.2f}")
    print(f"  FILTER : n={filt['n']:>4d}  PF={filt['pf']:.3f}  win={filt['win']:.3f}  exp={filt['exp']:+.3f}  tot={filt['tot']:+.2f}")
    return {"name": name, "raw": raw, "filter": filt}


def main():
    print("loading XAU OHLCV...")
    df = pd.read_csv("data/XAU_15MIN_2019_2025.csv", parse_dates=["Date"]).set_index("Date")
    df.columns = [c.lower() for c in df.columns]
    print(f"enriching {len(df)} bars (one-shot, reused for all TP runs)...")
    enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze()
    atr_series = enriched["ATR"]

    results = []
    for tp in [3.0, 4.0, 5.0, 6.0]:
        print(f"\n=== running SL=2.0, TP={tp} ===")
        trades = run_one(enriched.copy(), sl_mult=2.0, tp_mult=tp)
        if len(trades) == 0:
            print("  no trades, skipping")
            continue
        trades = apply_filter(trades, atr_series)
        results.append(report(f"TP={tp}", trades))

    # Final table
    print("\n\n=== SUMMARY (test set 2023-2025, filter applied) ===")
    print(f"{'TP':>4s}  {'n':>5s}  {'PF':>6s}  {'win':>5s}  {'exp':>7s}  {'tot':>7s}")
    for r in results:
        f = r["filter"]
        print(f"{r['name'][3:]:>4s}  {f['n']:>5d}  {f['pf']:>6.3f}  {f['win']:>5.2f}  {f['exp']:>+7.3f}  {f['tot']:>+7.2f}")


if __name__ == "__main__":
    main()
