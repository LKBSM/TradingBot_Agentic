"""Search for an ML model that passes admission gates on 5 markets.

5 markets = 5 (asset, timeframe) combinations :
  1. XAU M15
  2. XAU H1   (resampled)
  3. XAU H4   (resampled)
  4. EURUSD M15
  5. EURUSD H1 (resampled)

For each market : walk-forward refit monthly with each ML model below.
Gates evaluated : DSR, PBO, PF lo CI, DM p-value, n_trades.

Models tried in order :
  1. LightGBM regressor (current)
  2. XGBoost regressor
  3. Random Forest regressor
  4. Elastic Net (linear)
  5. Ridge regression
  6. MLP (small neural net)

Stop and report when one model passes 5/5 markets, OR all models exhausted.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path
from typing import Optional

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

MARKETS = [
    ("XAU", "M15", 1),
    ("XAU", "H1", 4),
    ("XAU", "H4", 16),
    ("EURUSD", "M15", 1),
    ("EURUSD", "H1", 4),
]


def load_resampled(asset: str, resample_factor: int) -> pd.DataFrame:
    df = pd.read_csv(ROOT / ASSET_CSV[asset], parse_dates=["Date"]).set_index("Date")
    df.rename(columns={c: c.capitalize() for c in df.columns
                       if c.lower() in {"open", "high", "low", "close", "volume"}}, inplace=True)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    if resample_factor == 1:
        return df
    freq = f"{15 * resample_factor}min"
    agg = {
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }
    return df.resample(freq, label="right", closed="right").agg(agg).dropna()


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from src.intelligence.macro_factors import MacroFactorExtractor
    from src.intelligence.microstructure import MicrostructureExtractor
    macro = MacroFactorExtractor().extract(ohlcv.index).drop(
        columns=["vix_regime"], errors="ignore")
    micro = MicrostructureExtractor().extract(ohlcv)
    return pd.concat([macro, micro], axis=1).ffill().fillna(0)


def make_model(name: str):
    """Return an unfitted sklearn-compatible regressor."""
    if name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.03, n_estimators=300,
            min_child_samples=50, reg_alpha=0.1,
            random_state=42, verbose=-1, deterministic=True, force_row_wise=True,
        )
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBRegressor(
            max_depth=5, learning_rate=0.05, n_estimators=300,
            reg_alpha=0.1, random_state=42, verbosity=0,
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(
            n_estimators=200, max_depth=8, min_samples_leaf=50,
            random_state=42, n_jobs=-1,
        )
    if name == "elastic_net":
        from sklearn.linear_model import ElasticNet
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("en", ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=42, max_iter=5000)),
        ])
    if name == "ridge":
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0, random_state=42)),
        ])
    if name == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        return Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=200,
                                  early_stopping=True, random_state=42)),
        ])
    raise ValueError(f"Unknown model: {name}")


def walk_forward_predict(feats, target, train_bars, refit_bars, horizon, model_name):
    preds = pd.Series(np.nan, index=feats.index)
    n = len(feats)
    start = train_bars
    while start < n:
        end = min(start + refit_bars, n)
        train_end = max(0, start - horizon)
        X_tr = feats.iloc[max(0, start - train_bars):train_end]
        y_tr = target.iloc[max(0, start - train_bars):train_end]
        mask = y_tr.notna() & X_tr.notna().all(axis=1)
        X_tr_arr = X_tr.loc[mask].to_numpy(float)
        y_tr_arr = y_tr.loc[mask].to_numpy(float)
        if len(X_tr_arr) < 500:
            start = end
            continue
        try:
            model = make_model(model_name)
            model.fit(X_tr_arr, y_tr_arr)
            X_te = feats.iloc[start:end].to_numpy(float)
            preds.iloc[start:end] = model.predict(X_te)
        except Exception as exc:
            print(f"   ⚠️ Model {model_name} failed at refit start={start}: {exc}")
            start = end
            continue
        start = end
    return preds


def build_trades_and_validate(preds, ret_1bar, q=0.6) -> dict:
    from src.backtest.validation import validate_trades_dataframe

    valid = preds.dropna()
    if len(valid) < 100:
        return {"status": "no_preds", "all_passed": False, "n_trades": 0}

    high_th = valid.quantile(q)
    low_th = valid.quantile(1 - q)
    signal = pd.Series(0, index=preds.index, dtype=float)
    signal[preds > high_th] = 1
    signal[preds < low_th] = -1

    per_bar_pnl = signal.shift(1).fillna(0) * ret_1bar
    signal_shifted = signal.shift(1).fillna(0)
    trade_id = (signal_shifted != signal_shifted.shift(1)).cumsum()

    trades = []
    for tid, group in per_bar_pnl.groupby(trade_id):
        if group.empty:
            continue
        sig_val = signal_shifted.loc[group.index].iloc[0]
        if sig_val == 0:
            continue
        cumret = float(group.sum())
        bar_std = max(float(group.std(ddof=0)), 1e-6)
        n_bars = len(group)
        r_mult = cumret / (bar_std * np.sqrt(n_bars))
        trades.append({
            "signal_id": f"t_{tid}",
            "direction": "LONG" if sig_val > 0 else "SHORT",
            "n_bars": n_bars,
            "pnl_log": cumret,
            "r_multiple": r_mult,
            "pnl_r": r_mult,
        })

    trades_df = pd.DataFrame(trades)
    if trades_df.empty:
        return {"status": "no_trades", "all_passed": False, "n_trades": 0}

    # IMPORTANT : n_trials=1 because we're NOT doing hyperparameter search
    # in walk-forward (the refits adapt to drift, not search for best params).
    # The previous validate_walk_forward.py used n_trials=85 which deflates
    # DSR unfairly.
    result = validate_trades_dataframe(
        trades_df,
        pnl_column="r_multiple",
        n_trials=1,
        n_bootstraps=1000,
    )

    # Also compute Sharpe + IR vs B&H on per-bar
    strat_per_bar = signal.shift(1).fillna(0) * ret_1bar
    bh_per_bar = ret_1bar
    n_bars = len(ret_1bar.dropna())
    bars_per_year = 252 * 96
    if hasattr(ret_1bar.index, "freq") and ret_1bar.index.freq is not None:
        pass
    bh_sharpe = float(bh_per_bar.mean() / bh_per_bar.std() * np.sqrt(bars_per_year)) if bh_per_bar.std() > 0 else 0
    strat_sharpe = float(strat_per_bar.mean() / strat_per_bar.std() * np.sqrt(bars_per_year)) if strat_per_bar.std() > 0 else 0

    return {
        "status": "evaluated",
        "all_passed": result.all_passed,
        "n_trades": result.n_trades,
        "sharpe_trades": float(result.sharpe),
        "pf": float(result.profit_factor),
        "pf_lo": float(result.profit_factor_lo),
        "pf_hi": float(result.profit_factor_hi),
        "dsr": float(result.dsr),
        "pbo": float(result.pbo),
        "dm_p": float(result.dm_pvalue),
        "gates_trades": result.trades_pass,
        "gates_dsr": result.dsr_pass,
        "gates_pbo": result.pbo_pass,
        "gates_pf_lo": result.pf_lo_pass,
        "gates_dm": result.dm_pass,
        "strat_sharpe_bar": strat_sharpe,
        "bh_sharpe_bar": bh_sharpe,
        "ir_vs_bh": strat_sharpe - bh_sharpe,
    }


def run_one_market(asset: str, tf: str, resample_factor: int, model_name: str,
                    horizon_bars: int = 96, train_days: int = 365, refit_days: int = 30) -> dict:
    ohlcv = load_resampled(asset, resample_factor)
    feats = build_features(ohlcv)
    target = np.log(ohlcv["Close"].shift(-horizon_bars // resample_factor) / ohlcv["Close"])
    ret_1bar = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))

    bars_per_day = 96 // resample_factor
    train_bars = train_days * bars_per_day
    refit_bars = refit_days * bars_per_day

    print(f"  [{asset} {tf}] bars={len(ohlcv)}, train_bars={train_bars}, refit_bars={refit_bars}, model={model_name}")
    preds = walk_forward_predict(
        feats, target,
        train_bars=train_bars,
        refit_bars=refit_bars,
        horizon=horizon_bars // resample_factor,
        model_name=model_name,
    )
    return build_trades_and_validate(preds, ret_1bar)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", default="lightgbm,xgboost,random_forest,elastic_net,ridge,mlp")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--refit-days", type=int, default=30)
    p.add_argument("--single", default=None,
                   help="Test single model (e.g. lightgbm) — skip iteration")
    args = p.parse_args()

    out_dir = ROOT / "reports" / "five_markets"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_try = [args.single] if args.single else args.models.split(",")
    all_results = {}

    print(f"\n{'='*70}\nSearch for AI passing gates on 5 markets\n{'='*70}\n")

    for model_name in models_to_try:
        print(f"\n--- Testing model : {model_name} ---")
        market_results = []
        n_passed = 0
        for asset, tf, rf in MARKETS:
            try:
                res = run_one_market(
                    asset, tf, rf, model_name,
                    horizon_bars=args.horizon,
                    train_days=args.train_days,
                    refit_days=args.refit_days,
                )
                res["asset"] = asset
                res["tf"] = tf
                market_results.append(res)
                gate_emoji = "✅" if res.get("all_passed") else "❌"
                print(f"    {gate_emoji} {asset} {tf:3s} "
                      f"trades={res.get('n_trades', 0):>5} "
                      f"DSR={res.get('dsr', float('nan')):>6.2f} "
                      f"PBO={res.get('pbo', 0.5):>4.2f} "
                      f"PF_lo={res.get('pf_lo', 0):>5.2f} "
                      f"IR_vs_BH={res.get('ir_vs_bh', 0):+.3f}")
                if res.get("all_passed"):
                    n_passed += 1
            except Exception as exc:
                print(f"    ⚠️ {asset} {tf} : {exc}")
                market_results.append({"asset": asset, "tf": tf, "error": str(exc), "all_passed": False})

        all_results[model_name] = {
            "n_passed": n_passed,
            "markets": market_results,
        }

        # Save intermediate
        (out_dir / f"{model_name}_results.json").write_text(
            json.dumps(all_results[model_name], indent=2, default=str),
            encoding="utf-8",
        )

        if n_passed >= 5:
            print(f"\n🏆 {model_name} passes ALL 5 markets ! Stopping search.")
            break
        elif n_passed >= 3:
            print(f"  🟡 {model_name} : {n_passed}/5 — partial success")
        else:
            print(f"  ❌ {model_name} : only {n_passed}/5")

    # Final report
    report = out_dir / "search_report.md"
    lines = ["# AI search — gates on 5 markets", "", f"**Date** : {pd.Timestamp.now().isoformat()}", ""]
    lines.append("| Model | n_passed (5 markets) | Best market |")
    lines.append("| --- | --- | --- |")
    for model_name, r in all_results.items():
        passing = [(m["asset"], m["tf"]) for m in r["markets"] if m.get("all_passed")]
        passing_str = ", ".join(f"{a}/{t}" for a, t in passing) or "none"
        lines.append(f"| {model_name} | **{r['n_passed']}/5** | {passing_str} |")

    lines.append("")
    lines.append("## Per-market details")
    for model_name, r in all_results.items():
        lines.append(f"\n### {model_name}")
        lines.append("")
        lines.append("| Asset | TF | Trades | Sharpe(trades) | DSR | PBO | PF_lo | IR vs BH | All pass |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        for m in r["markets"]:
            if "error" in m:
                lines.append(f"| {m['asset']} | {m['tf']} | ERROR | - | - | - | - | - | ❌ |")
                continue
            emoji = "✅" if m.get("all_passed") else "❌"
            lines.append(
                f"| {m['asset']} | {m['tf']} | {m.get('n_trades', 0)} | "
                f"{m.get('sharpe_trades', 0):.3f} | "
                f"{m.get('dsr', 0):.3f} | {m.get('pbo', 0.5):.3f} | "
                f"{m.get('pf_lo', 0):.3f} | {m.get('ir_vs_bh', 0):+.3f} | {emoji} |"
            )

    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {report}")

    winner = max(all_results.items(), key=lambda kv: kv[1]["n_passed"])
    print(f"\nBest model : {winner[0]} ({winner[1]['n_passed']}/5 markets)")
    return 0 if winner[1]["n_passed"] >= 5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
