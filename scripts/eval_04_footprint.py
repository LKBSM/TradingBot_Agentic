"""Mesure le footprint mémoire et disque des 3 modes de VolatilityForecaster.

Sortie : reports/eval_04/footprint.json
"""
from __future__ import annotations

import gc
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.intelligence.volatility_forecaster import (  # noqa: E402
    HybridForecaster,
    VolatilityForecaster,
)


def load_data(path: str, n_bars: int = 30000) -> pd.DataFrame:
    df = pd.read_csv(path).tail(n_bars).copy()
    df.columns = [c.lower() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["date"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]].reset_index(drop=True)


def load_calendar(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" in df.columns and "timestamp" not in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def measure_rss_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return float("nan")


def main():
    data_path = "data/XAU_15MIN_2019_2024.csv"
    cal_path = "data/economic_calendar_HIGH_IMPACT_2019_2025.csv"

    print("Loading data (last 30k bars)...")
    df = load_data(data_path, 30000)
    cal = load_calendar(cal_path)
    print(f"  bars={len(df)}  events={len(cal)}")

    out = {"data_bars": len(df), "modes": {}}

    for mode in ("har", "lgbm", "hybrid"):
        print(f"\n=== {mode} ===")
        gc.collect()
        rss_before = measure_rss_mb()

        if mode == "har":
            model = VolatilityForecaster()
        else:
            model = HybridForecaster(mode=mode)

        t0 = time.perf_counter()
        try:
            model.calibrate(df, cal)
        except Exception as e:
            print(f"  calibrate failed: {e}")
            out["modes"][mode] = {"error": str(e)}
            continue
        fit_s = time.perf_counter() - t0

        rss_after = measure_rss_mb()

        # Save state to measure on-disk size
        with tempfile.TemporaryDirectory() as td:
            state_path = os.path.join(td, "state.pkl")
            try:
                model.save_state(state_path)
            except Exception as e:
                print(f"  save_state failed: {e}")
                out["modes"][mode] = {"error_save": str(e), "fit_seconds": fit_s}
                continue

            files = []
            for f in Path(td).glob("*"):
                files.append({"name": f.name, "bytes": f.stat().st_size})
            total_bytes = sum(f["bytes"] for f in files)

        # Forecast latency micro-bench (200 calls on tail)
        latencies = []
        for i in range(len(df) - 200, len(df)):
            t = time.perf_counter_ns()
            try:
                model.forecast(df.iloc[max(0, i - 3000):i + 1])
            except Exception:
                continue
            latencies.append((time.perf_counter_ns() - t) / 1000.0)
        lat = np.array(latencies)

        out["modes"][mode] = {
            "fit_seconds": fit_s,
            "rss_mb_before": rss_before,
            "rss_mb_after": rss_after,
            "rss_mb_delta": rss_after - rss_before,
            "state_total_bytes": total_bytes,
            "state_files": files,
            "forecast_latency_us": {
                "p50": float(np.quantile(lat, 0.5)),
                "p95": float(np.quantile(lat, 0.95)),
                "p99": float(np.quantile(lat, 0.99)),
                "mean": float(lat.mean()),
                "n": int(len(lat)),
            },
        }
        print(f"  fit={fit_s:.1f}s  rss+={rss_after - rss_before:.1f}MB  "
              f"state={total_bytes/1024:.1f}KB  P95_lat={out['modes'][mode]['forecast_latency_us']['p95']:.0f}µs")

        del model
        gc.collect()

    out_path = Path("reports/eval_04/footprint.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
