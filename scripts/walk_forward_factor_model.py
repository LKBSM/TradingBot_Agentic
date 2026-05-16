"""Walk-forward factor model with cross-asset + period segmentation.

3 institutional-grade tests in one script :

1. **Walk-forward refit** : refit every N bars (60 days default) instead of
   single-pass train/test. This is how banks actually deploy models —
   monthly retraining to adapt to regime changes.

2. **Cross-asset** : same pipeline on XAU vs EURUSD. If macro factors
   carry signal, they should work on FX too (DXY-driven).

3. **Period segmentation** : split into sub-periods (2019-2021 / 2022 /
   2023-2026) to see where active beats passive.

Usage::

    python scripts/walk_forward_factor_model.py --asset XAU
    python scripts/walk_forward_factor_model.py --asset EURUSD
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


def build_features(asset: str, ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Macro + microstructure features (PIT-safe)."""
    from src.intelligence.macro_factors import MacroFactorExtractor
    from src.intelligence.microstructure import MicrostructureExtractor

    macro = MacroFactorExtractor().extract(ohlcv.index).drop(
        columns=["vix_regime"], errors="ignore")
    micro = MicrostructureExtractor().extract(ohlcv)
    feats = pd.concat([macro, micro], axis=1).ffill().fillna(0)
    return feats


def walk_forward_predict(
    feats: pd.DataFrame,
    target: pd.Series,
    train_bars: int,
    refit_bars: int,
    horizon: int,
) -> pd.Series:
    """Walk-forward predictions.

    At each refit step :
    - Train on the last ``train_bars`` ending at refit boundary
    - Predict the next ``refit_bars``
    """
    import lightgbm as lgb

    preds = pd.Series(np.nan, index=feats.index)
    n = len(feats)
    start = train_bars
    i_refit = 0
    while start < n:
        end = min(start + refit_bars, n)
        train_slice = slice(max(0, start - train_bars), start - horizon)  # exclude target window
        test_slice = slice(start, end)

        X_tr = feats.iloc[train_slice]
        y_tr = target.iloc[train_slice]
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

        X_te = feats.iloc[test_slice].to_numpy(float)
        pred_arr = model.predict(X_te)
        preds.iloc[test_slice] = pred_arr

        i_refit += 1
        if i_refit % 5 == 0:
            print(f"   refit {i_refit} | train={len(X_tr_arr)} | test={end-start} | done={end}/{n}")
        start = end
    return preds


def sharpe(ret: pd.Series, bars_per_year: float) -> float:
    ret = ret.dropna()
    if len(ret) < 10 or ret.std() == 0:
        return 0.0
    return float(ret.mean() / ret.std() * np.sqrt(bars_per_year))


def max_dd_pct(equity: pd.Series) -> float:
    running_max = equity.expanding().max()
    return float(((equity - running_max) / running_max).min())


def evaluate(name: str, signal: pd.Series, ret_1bar: pd.Series, bars_per_year: float) -> dict:
    """Evaluate strategy with signal applied to next-bar return.

    Signal at time t determines exposure for the next bar return.
    """
    strat_ret = signal.shift(1).fillna(0) * ret_1bar
    strat_eq = (1 + strat_ret.fillna(0)).cumprod()
    bh_eq = (1 + ret_1bar.fillna(0)).cumprod()
    n = len(ret_1bar.dropna())
    return {
        "name": name,
        "n_bars": n,
        "exposed_pct": float((signal != 0).mean() * 100),
        "sharpe": sharpe(strat_ret, bars_per_year),
        "bh_sharpe": sharpe(ret_1bar, bars_per_year),
        "ir_vs_bh": sharpe(strat_ret, bars_per_year) - sharpe(ret_1bar, bars_per_year),
        "max_dd_strat": max_dd_pct(strat_eq) * 100,
        "max_dd_bh": max_dd_pct(bh_eq) * 100,
        "final_eq_strat": float(strat_eq.iloc[-1]),
        "final_eq_bh": float(bh_eq.iloc[-1]),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="XAU", choices=list(ASSET_CSV.keys()))
    p.add_argument("--horizon", type=int, default=96,
                   help="Forecast horizon in M15 bars (default 96 = 1 day)")
    p.add_argument("--train-days", type=int, default=365,
                   help="Training window in days (default 365)")
    p.add_argument("--refit-days", type=int, default=30,
                   help="Refit period in days (default 30 = monthly)")
    p.add_argument("--threshold-quantile", type=float, default=0.6,
                   help="Predict quantile threshold for entry (0.6 = top/bottom 40%)")
    args = p.parse_args()

    print(f"=== Walk-forward {args.asset} ===")
    ohlcv = load_ohlcv(args.asset)
    print(f"Loaded {len(ohlcv)} bars : {ohlcv.index[0]} → {ohlcv.index[-1]}")

    print("=== Building features (macro + microstructure) ===")
    feats = build_features(args.asset, ohlcv)
    print(f"Features : {feats.shape[1]}")

    print("=== Target : log return H+{} ===".format(args.horizon))
    target = np.log(ohlcv["Close"].shift(-args.horizon) / ohlcv["Close"])

    bars_per_year = 252 * 96 / args.horizon  # M15 bars over the horizon

    print(f"=== Walk-forward refit (train={args.train_days}d, refit={args.refit_days}d) ===")
    bars_per_day = 96  # M15
    train_bars = args.train_days * bars_per_day
    refit_bars = args.refit_days * bars_per_day
    preds = walk_forward_predict(feats, target, train_bars, refit_bars, args.horizon)

    # Build signal: long if pred above some quantile, short if below
    valid = preds.dropna()
    if len(valid) < 100:
        print(f"❌ Too few valid predictions: {len(valid)}")
        return 1
    high_th = valid.quantile(args.threshold_quantile)
    low_th = valid.quantile(1 - args.threshold_quantile)
    print(f"Threshold high (q={args.threshold_quantile:.2f}) : {high_th:+.6f}")
    print(f"Threshold low  (q={1-args.threshold_quantile:.2f}) : {low_th:+.6f}")

    signal = pd.Series(0, index=preds.index, dtype=float)
    signal[preds > high_th] = 1
    signal[preds < low_th] = -1

    # 1-bar return for evaluation
    ret_1bar = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))

    # ---- Overall evaluation ----
    print("\n=== OVERALL ===")
    res = evaluate("WALK_FORWARD", signal, ret_1bar, bars_per_day * 252)
    for k, v in res.items():
        print(f"  {k}: {v}")

    # ---- Period segmentation ----
    print("\n=== PERIOD SEGMENTATION ===")
    periods = [
        ("2019-2021 (cyclical)", "2019-01-01", "2022-01-01"),
        ("2022 (range-bound)", "2022-01-01", "2023-01-01"),
        ("2023-2026 (bull)", "2023-01-01", "2027-01-01"),
    ]
    period_results = []
    for label, start, end in periods:
        mask = (ret_1bar.index >= start) & (ret_1bar.index < end)
        sig_p = signal[mask]
        ret_p = ret_1bar[mask]
        if ret_p.dropna().empty:
            continue
        res_p = evaluate(label, sig_p, ret_p, bars_per_day * 252)
        period_results.append(res_p)
        print(f"\n[{label}]")
        for k in ("n_bars", "exposed_pct", "sharpe", "bh_sharpe", "ir_vs_bh",
                  "max_dd_strat", "max_dd_bh", "final_eq_strat", "final_eq_bh"):
            print(f"  {k}: {res_p[k]}")

    # Save report
    out_dir = ROOT / "reports" / "factor_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / f"walk_forward_{args.asset}.md"
    lines = [
        f"# Walk-forward factor model — {args.asset}",
        "",
        f"**Date** : {pd.Timestamp.now().isoformat()}",
        f"**Asset** : {args.asset}",
        f"**Horizon** : {args.horizon} M15 bars ({args.horizon * 15 / 60:.1f} h)",
        f"**Train window** : {args.train_days} days, **refit every** {args.refit_days} days",
        f"**Threshold quantile** : {args.threshold_quantile:.2f}",
        f"**Total bars** : {len(ohlcv)}",
        "",
        "## Overall walk-forward",
        "",
        f"- Sharpe strategy : {res['sharpe']:+.3f}",
        f"- Sharpe B&H      : {res['bh_sharpe']:+.3f}",
        f"- IR vs B&H       : **{res['ir_vs_bh']:+.3f}**",
        f"- Max DD strat    : {res['max_dd_strat']:.2f}%",
        f"- Max DD B&H      : {res['max_dd_bh']:.2f}%",
        f"- Final equity    : {res['final_eq_strat']:.3f} vs B&H {res['final_eq_bh']:.3f}",
        f"- Exposed         : {res['exposed_pct']:.1f}% of bars",
        "",
        "## Per-period",
        "",
        "| Period | Sharpe strat | Sharpe B&H | IR vs B&H | MaxDD strat | MaxDD B&H | Final eq strat | Final eq B&H |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in period_results:
        lines.append(
            f"| {r['name']} | {r['sharpe']:+.3f} | {r['bh_sharpe']:+.3f} | "
            f"**{r['ir_vs_bh']:+.3f}** | {r['max_dd_strat']:.2f}% | "
            f"{r['max_dd_bh']:.2f}% | {r['final_eq_strat']:.3f} | {r['final_eq_bh']:.3f} |"
        )
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    if res["ir_vs_bh"] > 0:
        lines.append("✅ **Walk-forward bat B&H** sur l'overall. Edge institutionnel confirmé.")
    elif any(r["ir_vs_bh"] > 0 for r in period_results):
        winning_periods = [r["name"] for r in period_results if r["ir_vs_bh"] > 0]
        lines.append(f"🟡 **Edge conditionnel** : bat B&H sur {winning_periods}.")
    else:
        lines.append("❌ **Pas d'edge** : strat sous-performe B&H sur tous les sous-périodes.")

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
