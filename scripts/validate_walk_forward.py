"""Formal admission gates on walk-forward predictions.

Runs the institutional validation chain (DSR / PBO / PF lo / DM p-value)
on the walk-forward output of `walk_forward_factor_model.py`. This is the
gate that decides if the AI is **commercially validated**.

Usage::

    python scripts/validate_walk_forward.py --asset EURUSD
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


ASSET_CSV = {
    "XAU": "data/XAU_15MIN_2019_2026.csv",
    "EURUSD": "data/EURUSD_15MIN_2019_2025.csv",
}


def load_ohlcv(asset: str) -> pd.DataFrame:
    df = pd.read_csv(ROOT / ASSET_CSV[asset], parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns
                       if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0
    return df


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from src.intelligence.macro_factors import MacroFactorExtractor
    from src.intelligence.microstructure import MicrostructureExtractor
    macro = MacroFactorExtractor().extract(ohlcv.index).drop(
        columns=["vix_regime"], errors="ignore")
    micro = MicrostructureExtractor().extract(ohlcv)
    return pd.concat([macro, micro], axis=1).ffill().fillna(0)


def walk_forward_predict(feats, target, train_bars, refit_bars, horizon):
    import lightgbm as lgb
    preds = pd.Series(np.nan, index=feats.index)
    n = len(feats)
    start = train_bars
    while start < n:
        end = min(start + refit_bars, n)
        X_tr = feats.iloc[max(0, start - train_bars):start - horizon]
        y_tr = target.iloc[max(0, start - train_bars):start - horizon]
        mask = y_tr.notna() & X_tr.notna().all(axis=1)
        X_tr_arr = X_tr.loc[mask].to_numpy(float)
        y_tr_arr = y_tr.loc[mask].to_numpy(float)
        if len(X_tr_arr) < 500:
            start = end
            continue
        model = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.03, n_estimators=300,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=0.0,
            random_state=42, verbose=-1, deterministic=True, force_row_wise=True,
        )
        model.fit(X_tr_arr, y_tr_arr)
        X_te = feats.iloc[start:end].to_numpy(float)
        preds.iloc[start:end] = model.predict(X_te)
        start = end
    return preds


def build_trades(preds, ret_1bar, q=0.6) -> pd.DataFrame:
    """Build pseudo-trades from walk-forward predictions for gates.

    A 'trade' = a contiguous period where signal != 0. The trade's
    `r_multiple` proxy = cumulative log return over the period.
    """
    valid = preds.dropna()
    high_th = valid.quantile(q)
    low_th = valid.quantile(1 - q)
    signal = pd.Series(0, index=preds.index, dtype=float)
    signal[preds > high_th] = 1
    signal[preds < low_th] = -1

    # Per-bar PnL (already-shifted signal to avoid lookahead)
    per_bar_pnl = signal.shift(1).fillna(0) * ret_1bar
    # Group contiguous same-signal periods as 1 trade
    signal_shifted = signal.shift(1).fillna(0)
    trade_id = (signal_shifted != signal_shifted.shift(1)).cumsum()
    trades = []
    for tid, group in per_bar_pnl.groupby(trade_id):
        if group.empty:
            continue
        sig_val = signal_shifted.loc[group.index].iloc[0]
        if sig_val == 0:
            continue
        # Trade R-multiple = sum of log returns (over signal duration), scaled
        # by realized vol of the asset to put on R-scale
        cumret = float(group.sum())
        # Estimate per-trade std (proxy for risk)
        bar_std = max(float(group.std(ddof=0)), 1e-6)
        n_bars = len(group)
        # R = cumret / (bar_std * sqrt(n_bars))  ≈ trade Sharpe contribution
        r_mult = cumret / (bar_std * np.sqrt(n_bars))
        trades.append({
            "signal_id": f"trade_{tid}",
            "direction": "LONG" if sig_val > 0 else "SHORT",
            "n_bars": n_bars,
            "pnl_log": cumret,
            "r_multiple": r_mult,
            "pnl_r": r_mult,
            "entry_bar": str(group.index[0]),
            "exit_bar": str(group.index[-1]),
        })
    return pd.DataFrame(trades)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="EURUSD", choices=list(ASSET_CSV.keys()))
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--refit-days", type=int, default=30)
    p.add_argument("--threshold-quantile", type=float, default=0.6)
    args = p.parse_args()

    from src.backtest.validation import validate_trades_dataframe, render_gate_report

    print(f"=== Loading {args.asset} ===")
    ohlcv = load_ohlcv(args.asset)
    feats = build_features(ohlcv)
    target = np.log(ohlcv["Close"].shift(-args.horizon) / ohlcv["Close"])
    ret_1bar = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))

    print("=== Walk-forward predictions ===")
    bars_per_day = 96
    preds = walk_forward_predict(
        feats, target,
        train_bars=args.train_days * bars_per_day,
        refit_bars=args.refit_days * bars_per_day,
        horizon=args.horizon,
    )
    print(f"Predictions : {preds.notna().sum()} non-null on {len(preds)} bars")

    print("=== Building synthetic trades from predictions ===")
    trades_df = build_trades(preds, ret_1bar, q=args.threshold_quantile)
    print(f"Built {len(trades_df)} pseudo-trades")
    print(f"  mean R = {trades_df['r_multiple'].mean():+.4f}")
    print(f"  median R = {trades_df['r_multiple'].median():+.4f}")
    print(f"  wins = {(trades_df['r_multiple'] > 0).sum()}, losses = {(trades_df['r_multiple'] < 0).sum()}")

    print("\n=== Running ADMISSION GATES (DSR/PBO/PF_lo/DM) ===")
    result = validate_trades_dataframe(
        trades_df,
        pnl_column="r_multiple",
        n_trials=int((365 * 7) / args.refit_days),  # number of walk-forward refits
        n_bootstraps=1000,
    )
    print()
    print(render_gate_report(result))

    # Save
    out_dir = ROOT / "reports" / "factor_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / f"walk_forward_{args.asset}_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    print(f"Saved trades: {trades_path}")

    report = out_dir / f"walk_forward_{args.asset}_gates.md"
    lines = [
        f"# Admission gates — {args.asset} walk-forward",
        "",
        f"**Date** : {pd.Timestamp.now().isoformat()}",
        f"**n_trades** : {len(trades_df)}",
        f"**Refit horizon** : {args.refit_days} days",
        "",
        "## Gate verdict",
        "",
        f"**ALL GATES PASSED** : {'✅ YES' if result.all_passed else '❌ NO'}",
        "",
        "| Gate | Threshold | Value | Pass |",
        "| --- | --- | --- | --- |",
        f"| trades >= 30 | 30 | {result.n_trades} | {'✅' if result.trades_pass else '❌'} |",
        f"| DSR >= 1.5 | 1.5 | {result.dsr:.3f} | {'✅' if result.dsr_pass else '❌'} |",
        f"| PBO <= 0.35 | 0.35 | {result.pbo:.3f} | {'✅' if result.pbo_pass else '❌'} |",
        f"| PF lo > 1.0 | 1.0 | {result.profit_factor_lo:.3f} | {'✅' if result.pf_lo_pass else '❌'} |",
        f"| DM p < 0.05 | 0.05 | {result.dm_pvalue:.4f} | {'✅' if result.dm_pass else '❌'} |",
        "",
        f"**Sharpe** : {result.sharpe:.4f}",
        f"**Profit factor** : {result.profit_factor:.4f}",
        f"**PF 95% CI** : [{result.profit_factor_lo:.4f}, {result.profit_factor_hi:.4f}]",
        f"**DM statistic** : {result.dm_stat:.4f}",
    ]
    if result.failure_reasons:
        lines.append("")
        lines.append("**Failure reasons** :")
        for r in result.failure_reasons:
            lines.append(f"- {r}")
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {report}")

    return 0 if result.all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
