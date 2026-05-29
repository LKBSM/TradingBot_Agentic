"""Evaluate HAR-RV volatility forecast + conformal CI on OOS data.

Pipeline: calibrate on TRAIN (2019-2023), forecast at sampled OOS bars (≥800
bars). Each forecast is a 5-bar-ahead future_atr prediction with a TCP
confidence interval (target α=0.05 ⇒ 95% nominal coverage).

Q1 (Justesse factuelle):
  - RMSE / MAPE of forecast_atr vs realized future_atr (mean TR over next 5 bars).
  - R² of forecast vs realized.
  - Compare to naive ATR baseline (last bar's ATR_14).

Q2 (Stabilité): N/A — point forecast, but we report RMSE consistency on
random subsamples.

Q3 (Calibration de l'incertitude):
  - PICP (Prediction Interval Coverage Probability) at nominal 95%.
  - MPIW (Mean Prediction Interval Width).
  - Verdict via |PICP - 0.95| ≤ 0.02 / ≤ 0.05 thresholds.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))

from scripts.audit.descriptive_quality._harness import (  # noqa: E402
    InstrumentData,
    bootstrap_ci,
    load_calendar,
    load_instrument,
    verdict_picp,
)
from src.intelligence.volatility_forecaster import (  # noqa: E402
    InstrumentConfig,
    VolatilityForecaster,
    get_instrument_registry,
)


SAMPLE_SIZE = 800       # number of OOS bars to evaluate (compute budget)
PRED_HORIZON = 5        # bars ahead (matches default config)
NOMINAL_COVERAGE = 0.95  # tcp_alpha=0.05 → 95% nominal


def eval_volatility_for(inst: InstrumentData, calendar_df: pd.DataFrame, label: str) -> dict:
    sym = inst.symbol
    cfg = get_instrument_registry().get(sym) or InstrumentConfig(symbol=sym)
    forecaster = VolatilityForecaster(config=cfg)

    # Prepare train df: take raw OHLCV, reset index to a column named "date"
    train_raw = inst.raw[inst.raw.index <= pd.Timestamp("2023-12-31 23:59:59")].copy()
    train_raw = train_raw.reset_index().rename(columns={"date": "timestamp"})
    print(f"  calibrating on {len(train_raw):,d} train bars ...", flush=True)
    t0 = time.perf_counter()
    stats = forecaster.calibrate(train_raw, calendar_df=calendar_df)
    t_cal = time.perf_counter() - t0
    print(f"  calibration in {t_cal:.1f}s  har_fitted={stats.get('har_fitted')}, "
          f"blend_w={stats.get('blend_weight', 'N/A')}", flush=True)

    # Prepare OOS df with TR + ATR
    oos_raw = inst.raw[inst.raw.index >= pd.Timestamp("2024-01-01")].copy().reset_index().rename(
        columns={"date": "timestamp"}
    )
    # Compute realized future_atr per bar (mean TR over next 5 bars)
    prev_close = oos_raw["close"].shift(1)
    tr = pd.concat(
        [
            oos_raw["high"] - oos_raw["low"],
            (oos_raw["high"] - prev_close).abs(),
            (oos_raw["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    oos_raw["tr"] = tr
    oos_raw["atr_14"] = tr.rolling(14).mean()
    oos_raw["future_atr"] = (
        tr.rolling(PRED_HORIZON).mean().shift(-PRED_HORIZON)
    )

    # Need enough history: skip first cfg.har_train_min bars + add 14 for ATR
    min_start = max(cfg.har_train_min, 14) + 5
    n_oos = len(oos_raw)
    valid = oos_raw.index[(oos_raw.index >= min_start)
                          & (oos_raw["future_atr"].notna())
                          & (oos_raw["atr_14"].notna())].to_numpy()
    if len(valid) < SAMPLE_SIZE:
        sampled = valid
    else:
        rng = np.random.default_rng(42)
        sampled = np.sort(rng.choice(valid, size=SAMPLE_SIZE, replace=False))

    # Hold full history up to each sampled bar
    forecasts = []
    naives = []
    actuals = []
    lowers = []
    uppers = []
    print(f"  forecasting {len(sampled):,d} OOS bars ...", flush=True)
    t0 = time.perf_counter()
    for idx in sampled:
        slice_df = oos_raw.iloc[: idx + 1]
        ts = pd.Timestamp(slice_df["timestamp"].iloc[-1])
        f = forecaster.forecast(slice_df, ts)
        forecasts.append(f.forecast_atr)
        naives.append(f.naive_atr)
        actuals.append(float(oos_raw["future_atr"].iloc[idx]))
        lowers.append(f.confidence_lower)
        uppers.append(f.confidence_upper)
    t_fc = time.perf_counter() - t0
    print(f"  forecasting in {t_fc:.1f}s  ({t_fc / max(1, len(sampled)) * 1000:.0f} ms/bar)", flush=True)

    forecasts = np.array(forecasts)
    naives = np.array(naives)
    actuals = np.array(actuals)
    lowers = np.array(lowers)
    uppers = np.array(uppers)

    # Q1 — RMSE, MAPE, R²
    err = forecasts - actuals
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err / np.maximum(actuals, 1e-12))))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((actuals - actuals.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Naive baseline (ATR_14)
    err_n = naives - actuals
    rmse_n = float(np.sqrt(np.mean(err_n ** 2)))
    mape_n = float(np.mean(np.abs(err_n / np.maximum(actuals, 1e-12))))
    ss_res_n = float(np.sum(err_n ** 2))
    r2_n = 1 - ss_res_n / ss_tot if ss_tot > 0 else float("nan")

    # Diebold-Mariano (simplified: paired sign test)
    diff_sq = err ** 2 - err_n ** 2
    n = len(diff_sq)
    mean_d = float(diff_sq.mean())
    sd_d = float(diff_sq.std(ddof=1))
    dm_stat = mean_d / (sd_d / np.sqrt(n)) if sd_d > 0 else float("nan")
    # one-sided p (negative dm_stat => forecast is better)
    from scipy.stats import norm
    dm_p_better = float(norm.cdf(dm_stat))

    # Q3 — PICP / MPIW
    covered = (actuals >= lowers) & (actuals <= uppers)
    picp = float(covered.mean())
    mpiw = float((uppers - lowers).mean())
    # MPIW as fraction of mean actual ATR
    mpiw_rel = mpiw / max(actuals.mean(), 1e-12)
    picp_lo, picp_hi = float("nan"), float("nan")
    _, picp_lo, picp_hi = bootstrap_ci(covered.astype(float), np.mean, n_boot=1000)

    return {
        "symbol": sym,
        "label": label,
        "n_sampled": int(len(sampled)),
        "min_history_bars": int(min_start),
        "q1_point_forecast": {
            "rmse_har_blended": rmse,
            "rmse_naive": rmse_n,
            "rmse_improvement_pct": float((rmse_n - rmse) / rmse_n * 100) if rmse_n > 0 else float("nan"),
            "mape_har_blended": mape,
            "mape_naive": mape_n,
            "r2_har_blended": float(r2),
            "r2_naive": float(r2_n),
            "diebold_mariano_stat": float(dm_stat),
            "dm_p_value_har_better": dm_p_better,
            "n": int(len(sampled)),
            "definition": "future_atr = mean True Range over next 5 bars (M15); naive = ATR_14 at forecast time",
        },
        "q3_calibration_conformal": {
            "nominal_coverage": NOMINAL_COVERAGE,
            "empirical_coverage_picp": picp,
            "picp_ci95_lo": picp_lo,
            "picp_ci95_hi": picp_hi,
            "mpiw": mpiw,
            "mpiw_relative_to_mean_atr": float(mpiw_rel),
            "verdict": verdict_picp(picp, NOMINAL_COVERAGE),
            "n": int(len(sampled)),
            "tcp_alpha": cfg.tcp_alpha,
            "definition": "TCP confidence interval coverage on sampled OOS bars",
        },
    }


def main():
    cal = load_calendar()
    cal_for_forecaster = cal.rename(columns={"date": "timestamp"})
    results = {}
    for sym in ["XAUUSD", "EURUSD"]:
        print(f"--- {sym} ---")
        inst = load_instrument(sym)
        out = eval_volatility_for(inst, cal_for_forecaster, f"{sym} M15 OOS 2024+")
        results[sym] = out
        q1 = out["q1_point_forecast"]
        q3 = out["q3_calibration_conformal"]
        print(
            f"  Q1 RMSE: HAR={q1['rmse_har_blended']:.4f}  naive={q1['rmse_naive']:.4f}  "
            f"({q1['rmse_improvement_pct']:+.1f}% improvement)\n"
            f"  Q1 R²:   HAR={q1['r2_har_blended']:.4f}  naive={q1['r2_naive']:.4f}\n"
            f"  Q1 DM stat={q1['diebold_mariano_stat']:+.3f}  p(HAR better)={q1['dm_p_value_har_better']:.4f}\n"
            f"  Q3 PICP: {q3['empirical_coverage_picp']:.4f}  "
            f"[{q3['picp_ci95_lo']:.3f}, {q3['picp_ci95_hi']:.3f}]  nominal={q3['nominal_coverage']}  "
            f"{q3['verdict']}\n"
            f"  Q3 MPIW: {q3['mpiw']:.4f}  (relative to mean ATR: {q3['mpiw_relative_to_mean_atr']:.2f})"
        )

    out_path = HERE.parent.parent.parent / "docs" / "audits" / "data" / "volatility_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=float), encoding="utf-8")
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
