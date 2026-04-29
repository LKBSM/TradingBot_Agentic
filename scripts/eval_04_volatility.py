"""Walk-forward benchmark for Prompt 04 — volatility forecasting.

Compares naive ATR14 / HAR-RV / LightGBM / Hybrid on out-of-sample
windows, with Diebold-Mariano tests and per-year regime slicing.
Also records forecast latency (P50/P95/P99).

Usage
-----
    python scripts/eval_04_volatility.py \
        --data data/XAU_15MIN_2019_2024.csv \
        --calendar data/economic_calendar_HIGH_IMPACT_2019_2025.csv \
        --out reports/eval_04 \
        --sample-step 192
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intelligence.volatility_forecaster import (  # noqa: E402
    HybridForecaster,
    InstrumentConfig,
    VolatilityForecaster,
)


# ---------------------------------------------------------------- IO

def load_ohlcv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else "date"
    df["timestamp"] = pd.to_datetime(df[ts_col])
    return df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp").reset_index(drop=True)


def load_calendar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df.columns and "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ---------------------------------------------------------------- feature build

def add_tr_atr(df: pd.DataFrame) -> pd.DataFrame:
    prev_close = df["close"].shift(1).fillna(df["close"].iloc[0])
    tr = np.maximum.reduce([
        (df["high"] - df["low"]).values,
        (df["high"] - prev_close).abs().values,
        (df["low"] - prev_close).abs().values,
    ])
    df = df.copy()
    df["tr"] = tr
    df["atr_14"] = pd.Series(tr, index=df.index).rolling(14).mean()
    return df


def forward_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Mean TR over next `horizon` bars — same target HAR regresses on."""
    return df["tr"].rolling(horizon).mean().shift(-horizon)


# ---------------------------------------------------------------- statistics

def metrics(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    mask = np.isfinite(pred) & np.isfinite(truth)
    if mask.sum() < 2:
        return {"rmse": float("nan"), "mae": float("nan"), "mape": float("nan"), "bias": float("nan"), "n": int(mask.sum())}
    err = pred[mask] - truth[mask]
    t = truth[mask]
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "mape": float(np.mean(np.abs(err) / np.maximum(np.abs(t), 1e-9)) * 100.0),
        "bias": float(np.mean(err)),
        "n": int(mask.sum()),
    }


def dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 5) -> Tuple[float, float]:
    """Diebold-Mariano test on squared-error loss.

    H0: E[e1^2 - e2^2] = 0. Positive stat => model 2 (e2) lower loss.
    Newey-West HAC variance with bandwidth h-1.
    """
    d = (e1 ** 2) - (e2 ** 2)
    d = d[np.isfinite(d)]
    T = len(d)
    if T < 30:
        return (float("nan"), float("nan"))
    d_mean = d.mean()
    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        if T - lag <= 0:
            break
        gamma = np.mean((d[:-lag] - d_mean) * (d[lag:] - d_mean))
        var_d += 2.0 * (1.0 - lag / h) * gamma
    var_d = max(var_d, 1e-12) / T
    stat = d_mean / np.sqrt(var_d)
    # two-sided p-value (normal approx)
    from math import erf, sqrt as _sqrt
    p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(stat) / _sqrt(2.0))))
    return (float(stat), float(p))


# ---------------------------------------------------------------- walk-forward

def walk_forward(
    df: pd.DataFrame,
    cal: pd.DataFrame,
    cuts: List[str],
    model_factory: Callable,
    label: str,
    sample_step: int,
    window_bars: int,
    min_train: int = 5000,
) -> pd.DataFrame:
    """Expanding-window walk-forward for a single model."""
    rows = []
    df = df.copy()
    for split_i in range(len(cuts) - 1):
        cut_start = pd.Timestamp(cuts[split_i])
        cut_end = pd.Timestamp(cuts[split_i + 1])

        train_mask = df["timestamp"] < cut_start
        train_df = df.loc[train_mask]
        if len(train_df) < min_train:
            print(f"  [{label}] split {split_i}: train too small ({len(train_df)})", flush=True)
            continue

        cal_slice = cal[cal["timestamp"] < cut_start] if cal is not None and len(cal) else None

        t0 = time.perf_counter()
        model = model_factory()
        try:
            model.calibrate(train_df, cal_slice)
        except Exception as exc:
            print(f"  [{label}] split {split_i}: calibrate FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue
        fit_s = time.perf_counter() - t0

        test_mask = (df["timestamp"] >= cut_start) & (df["timestamp"] < cut_end)
        test_idx = list(df.index[test_mask])[::sample_step]
        print(
            f"  [{label}] split {split_i}: train<{cut_start.date()} "
            f"(n={len(train_df)}, fit={fit_s:.1f}s) -> test<{cut_end.date()} "
            f"({len(test_idx)} preds)",
            flush=True,
        )

        for i in test_idx:
            lo = max(0, i - window_bars)
            window = df.iloc[lo:i + 1]
            t_fc = time.perf_counter_ns()
            try:
                fc = model.forecast(window)
            except Exception:
                continue
            lat_us = (time.perf_counter_ns() - t_fc) / 1000.0
            rows.append({
                "model": label,
                "split": split_i,
                "timestamp": df["timestamp"].iloc[i],
                "year": int(df["timestamp"].iloc[i].year),
                "forecast": float(fc.forecast_atr),
                "naive": float(fc.naive_atr),
                "regime": fc.regime_state,
                "is_fallback": bool(fc.is_fallback),
                "latency_us": lat_us,
                "fit_seconds": fit_s,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/XAU_15MIN_2019_2024.csv")
    ap.add_argument("--calendar", default="data/economic_calendar_HIGH_IMPACT_2019_2025.csv")
    ap.add_argument("--out", default="reports/eval_04")
    ap.add_argument("--sample-step", type=int, default=192, help="stride between forecasts (M15 bars)")
    ap.add_argument("--window-bars", type=int, default=3000, help="rolling window passed to forecast()")
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--cuts", nargs="+", default=[
        "2022-01-01", "2022-07-01", "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01", "2025-01-01",
    ])
    ap.add_argument("--models", nargs="+", default=["har", "lgbm", "hybrid"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.data}", flush=True)
    df = load_ohlcv(args.data)
    df = add_tr_atr(df)
    truth = forward_target(df, args.horizon)
    df_truth = df[["timestamp"]].copy()
    df_truth["truth"] = truth

    cal = load_calendar(args.calendar) if Path(args.calendar).exists() else None
    print(
        f"  bars={len(df)}  range=[{df['timestamp'].min()}, {df['timestamp'].max()}]  "
        f"calendar={0 if cal is None else len(cal)}",
        flush=True,
    )

    factories = {
        "har": lambda: VolatilityForecaster(),
        "lgbm": lambda: HybridForecaster(mode="lgbm"),
        "hybrid": lambda: HybridForecaster(mode="hybrid"),
    }

    all_rows = []
    for m in args.models:
        if m not in factories:
            print(f"Unknown model: {m}", flush=True)
            continue
        print(f"\n=== Walk-forward: {m} ===", flush=True)
        r = walk_forward(
            df, cal, args.cuts, factories[m], m,
            sample_step=args.sample_step, window_bars=args.window_bars,
        )
        all_rows.append(r)

    if not all_rows:
        print("No results.", flush=True)
        return

    res = pd.concat(all_rows, ignore_index=True)
    res = res.merge(df_truth, on="timestamp", how="left")
    res.to_csv(out_dir / "walkforward_raw.csv", index=False)

    # --- aggregate metrics per model, all + per-year ---
    summary: Dict[str, Dict] = {}
    for m in res["model"].unique():
        sub = res[res["model"] == m]
        sub = sub.dropna(subset=["truth"])
        summary[m] = {
            "forecast_vs_truth": metrics(sub["forecast"].values, sub["truth"].values),
            "naive_vs_truth": metrics(sub["naive"].values, sub["truth"].values),
            "fallback_rate": float(sub["is_fallback"].mean()) if len(sub) else float("nan"),
            "by_year": {},
            "by_regime": {},
        }
        for y in sorted(sub["year"].unique()):
            yr = sub[sub["year"] == y]
            summary[m]["by_year"][int(y)] = {
                "forecast": metrics(yr["forecast"].values, yr["truth"].values),
                "naive": metrics(yr["naive"].values, yr["truth"].values),
            }
        for reg in sorted(sub["regime"].dropna().unique()):
            rr = sub[sub["regime"] == reg]
            summary[m]["by_regime"][reg] = metrics(rr["forecast"].values, rr["truth"].values)

    # --- DM tests ---
    dm: Dict[str, Dict] = {}
    for m in res["model"].unique():
        sub = res[res["model"] == m].dropna(subset=["truth"])
        e_model = (sub["forecast"].values - sub["truth"].values)
        e_naive = (sub["naive"].values - sub["truth"].values)
        stat, p = dm_test(e_model, e_naive, h=args.horizon)
        # positive stat => model has LOWER loss than naive (beats it)
        dm[f"{m}_vs_naive"] = {"stat": stat, "p": p, "lower_loss": "model" if stat < 0 else "naive"}

    if "har" in res["model"].unique():
        har_df = res[res["model"] == "har"].set_index("timestamp")
        for m in ("lgbm", "hybrid"):
            if m not in res["model"].unique():
                continue
            oth = res[res["model"] == m].set_index("timestamp")
            j = har_df[["forecast", "truth"]].rename(columns={"forecast": "f_har"}).join(
                oth[["forecast"]].rename(columns={"forecast": "f_oth"}), how="inner"
            ).dropna()
            if len(j) < 30:
                continue
            e_har = (j["f_har"].values - j["truth"].values)
            e_oth = (j["f_oth"].values - j["truth"].values)
            stat, p = dm_test(e_har, e_oth, h=args.horizon)
            dm[f"{m}_vs_har"] = {
                "stat": stat, "p": p,
                "lower_loss": m if stat > 0 else "har",
                "n_paired": int(len(j)),
            }

    # --- latency ---
    latency = {}
    for m in res["model"].unique():
        sub = res[res["model"] == m]
        lat = sub["latency_us"].dropna()
        if len(lat) == 0:
            continue
        latency[m] = {
            "p50_us": float(lat.quantile(0.5)),
            "p95_us": float(lat.quantile(0.95)),
            "p99_us": float(lat.quantile(0.99)),
            "mean_us": float(lat.mean()),
            "max_us": float(lat.max()),
            "n": int(len(lat)),
            "fit_seconds_mean": float(sub["fit_seconds"].mean()),
        }

    out = {
        "config": {
            "data": args.data,
            "calendar": args.calendar,
            "sample_step": args.sample_step,
            "window_bars": args.window_bars,
            "horizon": args.horizon,
            "cuts": args.cuts,
            "n_bars_total": int(len(df)),
            "models": args.models,
        },
        "summary": summary,
        "dm_tests": dm,
        "latency": latency,
    }
    summary_path = out_dir / "walkforward_summary.json"
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print("\n========== RESULTS ==========")
    print(json.dumps(out, indent=2, default=str))
    print(f"\nRaw: {out_dir / 'walkforward_raw.csv'}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
