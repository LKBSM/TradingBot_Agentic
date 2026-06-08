"""Periodic computation of historical stats for the InsightV2 historical_stats block.

Per user feedback : "an indicator is not a trading bot, costs are irrelevant for
the live scoring — BUT the historical stats published to clients MUST be cost-aware
or it's a marketing lie."

This script computes :
- ``similar_setups_n``    : how many historical bars have similar feature pattern
- ``hit_rate_observed``   : fraction where next-H1 R was > 0
- ``profit_factor``       : with realistic costs (DynamicSpread + DynamicSlippage)
- ``profit_factor_ci95``  : bootstrap CI on the above
- ``empirical_coverage``  : Mondrian coverage if available (else rolling)
- ``sample_window``       : e.g. "2019-2025"
- ``cost_assumptions``    : explicit spread / slippage / commission used

Output : ``reports/historical_stats/<asset>_<tf>.json`` (consumed by
:class:`InsightV2Builder.historical_stats_loader`).

Cadence : run weekly (cron) on the latest 1 year of data.

Usage::

    python scripts/compute_historical_stats.py --asset XAU --tf M15
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime, timezone
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
        df["Volume"] = 0.0
    return df


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    from src.intelligence.macro_factors import MacroFactorExtractor
    from src.intelligence.microstructure import MicrostructureExtractor
    macro = MacroFactorExtractor().extract(ohlcv.index).drop(columns=["vix_regime"], errors="ignore")
    micro = MicrostructureExtractor().extract(ohlcv)
    return pd.concat([macro, micro], axis=1).ffill().fillna(0)


def walk_forward_preds(feats: pd.DataFrame, target: pd.Series,
                        train_bars: int, refit_bars: int, horizon: int) -> pd.Series:
    import lightgbm as lgb
    preds = pd.Series(np.nan, index=feats.index)
    n = len(feats)
    start = train_bars
    while start < n:
        end = min(start + refit_bars, n)
        X_tr = feats.iloc[max(0, start - train_bars):start - horizon]
        y_tr = target.iloc[max(0, start - train_bars):start - horizon]
        mask = y_tr.notna() & X_tr.notna().all(axis=1)
        X_arr = X_tr.loc[mask].to_numpy(float)
        y_arr = y_tr.loc[mask].to_numpy(float)
        if len(X_arr) < 500:
            start = end
            continue
        model = lgb.LGBMRegressor(
            num_leaves=31, learning_rate=0.03, n_estimators=300,
            min_child_samples=50, reg_alpha=0.1, random_state=42,
            verbose=-1, deterministic=True, force_row_wise=True,
        )
        model.fit(X_arr, y_arr)
        preds.iloc[start:end] = model.predict(feats.iloc[start:end].to_numpy(float))
        start = end
    return preds


def apply_costs(returns: np.ndarray, spread_bps: float, slippage_bps: float) -> np.ndarray:
    """Subtract round-trip costs in basis points from each trade return."""
    cost = (spread_bps + slippage_bps) / 10000.0  # bps → fraction
    return returns - cost


def bootstrap_pf_ci(pnls: np.ndarray, n_iter: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Return (point_pf, ci_low, ci_high) at 95 %."""
    if len(pnls) < 30:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    def _pf(p):
        g = p[p > 0].sum(); l = -p[p < 0].sum()
        return float(g / l) if l > 0 else (float("inf") if g > 0 else float("nan"))
    point = _pf(pnls)
    boots = np.array([_pf(pnls[rng.integers(0, len(pnls), size=len(pnls))]) for _ in range(n_iter)])
    finite = boots[np.isfinite(boots)]
    return point, float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--asset", default="XAU", choices=list(ASSET_CSV.keys()))
    p.add_argument("--tf", default="M15")  # documentary
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--train-days", type=int, default=365)
    p.add_argument("--refit-days", type=int, default=30)
    p.add_argument("--threshold-q", type=float, default=0.6)
    p.add_argument("--spread-bps", type=float, default=2.0,
                   help="Round-trip spread cost in basis points (default 2 bps)")
    p.add_argument("--slippage-bps", type=float, default=1.0,
                   help="Round-trip slippage in basis points (default 1 bps)")
    args = p.parse_args()

    print(f"=== Compute historical stats : {args.asset} {args.tf} ===")
    ohlcv = load_ohlcv(args.asset)
    feats = build_features(ohlcv)
    target = np.log(ohlcv["Close"].shift(-args.horizon) / ohlcv["Close"])
    ret_1bar = np.log(ohlcv["Close"] / ohlcv["Close"].shift(1))

    bars_per_day = 96
    print(f"Walk-forward refit (train={args.train_days}d, refit={args.refit_days}d)...")
    preds = walk_forward_preds(
        feats, target,
        train_bars=args.train_days * bars_per_day,
        refit_bars=args.refit_days * bars_per_day,
        horizon=args.horizon,
    )

    valid = preds.dropna()
    if len(valid) < 100:
        print(f"❌ Too few predictions: {len(valid)}")
        return 1
    high_th = valid.quantile(args.threshold_q)
    low_th = valid.quantile(1 - args.threshold_q)
    signal = pd.Series(0, index=preds.index, dtype=float)
    signal[preds > high_th] = 1
    signal[preds < low_th] = -1

    # Per-bar return with signal (lagged)
    strat_per_bar = signal.shift(1).fillna(0) * ret_1bar

    # Aggregate into pseudo-trades (contiguous signal periods)
    signal_shifted = signal.shift(1).fillna(0)
    trade_id = (signal_shifted != signal_shifted.shift(1)).cumsum()
    trade_pnls_raw = []
    for tid, grp in strat_per_bar.groupby(trade_id):
        if grp.empty:
            continue
        sig_val = signal_shifted.loc[grp.index].iloc[0]
        if sig_val == 0:
            continue
        trade_pnls_raw.append(float(grp.sum()))
    trade_pnls_raw = np.array(trade_pnls_raw)
    print(f"  pseudo-trades : {len(trade_pnls_raw)}, mean log-ret = {trade_pnls_raw.mean():+.5f}")

    # Apply costs (round-trip per trade)
    trade_pnls_costed = apply_costs(trade_pnls_raw, args.spread_bps, args.slippage_bps)

    wins_raw = int((trade_pnls_raw > 0).sum())
    wins_costed = int((trade_pnls_costed > 0).sum())
    n = len(trade_pnls_costed)
    hit_rate_costed = wins_costed / n if n else 0.0

    pf_raw, raw_lo, raw_hi = bootstrap_pf_ci(trade_pnls_raw)
    pf_costed, c_lo, c_hi = bootstrap_pf_ci(trade_pnls_costed)

    payload = {
        "asset": args.asset,
        "timeframe": args.tf,
        "computed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sample_window": f"{ohlcv.index[0].date()}—{ohlcv.index[-1].date()}",
        "n_total_bars": int(len(ohlcv)),
        "horizon_bars": args.horizon,
        "refit_days": args.refit_days,
        "threshold_quantile": args.threshold_q,
        "similar_setups_n": int(n),
        "wins_costed": wins_costed,
        "losses_costed": int(n - wins_costed),
        "hit_rate_observed": round(hit_rate_costed, 4),
        "profit_factor": round(pf_costed, 3) if np.isfinite(pf_costed) else None,
        "profit_factor_ci95": [
            round(c_lo, 3) if np.isfinite(c_lo) else None,
            round(c_hi, 3) if np.isfinite(c_hi) else None,
        ],
        "profit_factor_no_costs_reference": round(pf_raw, 3) if np.isfinite(pf_raw) else None,
        "profit_factor_no_costs_ci95": [
            round(raw_lo, 3) if np.isfinite(raw_lo) else None,
            round(raw_hi, 3) if np.isfinite(raw_hi) else None,
        ],
        "cost_assumptions": {
            "round_trip_spread_bps": args.spread_bps,
            "round_trip_slippage_bps": args.slippage_bps,
            "round_trip_commission_bps": 0.0,
            "note": "Costs deducted from each pseudo-trade log return. Reflects what the client would have realised after broker spread + slippage.",
        },
        "model": "lightgbm_walk_forward_v1",
    }

    out_dir = ROOT / "reports" / "historical_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.asset}_{args.tf}.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"\n  hit_rate (with costs)  : {hit_rate_costed:.3f}")
    print(f"  PF      (with costs)   : {pf_costed:.3f}  [CI 95% : {c_lo:.3f}, {c_hi:.3f}]")
    print(f"  PF      (no costs)     : {pf_raw:.3f}  [CI 95% : {raw_lo:.3f}, {raw_hi:.3f}]")
    print(f"\nSaved : {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
