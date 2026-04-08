"""
=============================================================================
POC: Kronos Volatility Forecasting for Smart Sentinel AI
=============================================================================
Goal: Determine if Kronos predicts Gold M15 volatility (ATR) better than
      the naive baseline (ATR_current = ATR_future).

The ONLY question this POC answers:
  "Should we integrate Kronos into the Smart Sentinel pipeline for
   adaptive SL/TP sizing based on predicted volatility?"

NOTE: The data CSV (XAU_15MIN_2019_2025.csv) is NOT in the GitHub repo
      (gitignored due to size). Place it in data/ before running.

Usage:
  Local (CPU, slower):   python scripts/poc_kronos_volatility.py --device cpu --model small
  Colab (GPU, faster):   python scripts/poc_kronos_volatility.py --device cuda:0 --model base

Author: Smart Sentinel AI
Date: 2026-04-08
=============================================================================
"""

import os
import sys
import time
import argparse
import subprocess
import importlib
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 0. SETUP: Clone Kronos if not present, add to path
# ---------------------------------------------------------------------------

KRONOS_DIR = Path(__file__).resolve().parent.parent / "kronos_repo"
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "XAU_15MIN_2019_2025.csv"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "poc_results"


def setup_kronos():
    """Clone Kronos repo and install dependencies if needed."""
    if not KRONOS_DIR.exists():
        print("[SETUP] Cloning Kronos repository...")
        subprocess.run(
            ["git", "clone", "https://github.com/shiyu-coder/Kronos.git", str(KRONOS_DIR)],
            check=True,
        )
        print("[SETUP] Clone complete.")
    else:
        print(f"[SETUP] Kronos repo found at {KRONOS_DIR}")

    # Install torch if not present
    try:
        import torch
        print(f"[SETUP] PyTorch {torch.__version__} already installed")
    except ImportError:
        print("[SETUP] Installing PyTorch...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "torch", "torchvision", "--quiet"],
            check=True,
        )

    # Install transformers/huggingface_hub if not present
    for pkg in ["transformers", "huggingface_hub", "safetensors"]:
        try:
            importlib.import_module(pkg)
        except ImportError:
            print(f"[SETUP] Installing {pkg}...")
            subprocess.run(
                [sys.executable, "-m", "pip", "install", pkg, "--quiet"],
                check=True,
            )

    # Add Kronos to Python path
    if str(KRONOS_DIR) not in sys.path:
        sys.path.insert(0, str(KRONOS_DIR))

    print("[SETUP] Ready.\n")


# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPARATION
# ---------------------------------------------------------------------------

def load_gold_data(path: Path, test_start: str = "2024-07-01") -> tuple:
    """
    Load XAU/USD M15 data and split into context/test periods.

    Returns:
        full_df: Complete DataFrame with OHLCV
        test_start_idx: Index where test period begins
    """
    print(f"[DATA] Loading {path}...")
    df = pd.read_csv(path, parse_dates=["Date"])
    df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Basic stats
    print(f"[DATA] Total bars: {len(df):,}")
    print(f"[DATA] Date range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

    # Compute ATR (7-period, matching Smart Sentinel config)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1)),
        ),
    )
    df["atr_7"] = df["tr"].rolling(7).mean()

    # Split
    test_mask = df["timestamp"] >= test_start
    test_start_idx = df[test_mask].index[0]
    print(f"[DATA] Test period starts at index {test_start_idx} ({test_start})")
    print(f"[DATA] Test bars: {test_mask.sum():,}")
    print()

    return df, test_start_idx


# ---------------------------------------------------------------------------
# 2. KRONOS PREDICTOR WRAPPER
# ---------------------------------------------------------------------------

class KronosVolatilityPredictor:
    """Wraps Kronos model for volatility (ATR) prediction."""

    def __init__(self, model_name: str = "small", device: str = "cpu", max_context: int = 400):
        from model import Kronos, KronosTokenizer, KronosPredictor

        model_map = {
            "mini": ("NeoQuasar/Kronos-mini", "NeoQuasar/Kronos-Tokenizer-mini"),
            "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
            "base": ("NeoQuasar/Kronos-base", "NeoQuasar/Kronos-Tokenizer-base"),
        }

        model_id, tokenizer_id = model_map[model_name]
        print(f"[MODEL] Loading Kronos-{model_name} ({model_id})...")
        print(f"[MODEL] Device: {device}")

        self.tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        self.model = Kronos.from_pretrained(model_id)
        self.predictor = KronosPredictor(
            self.model, self.tokenizer, device=device, max_context=max_context
        )
        self.max_context = max_context
        self.device = device
        print(f"[MODEL] Loaded successfully.\n")

    def predict_bars(
        self, df_context: pd.DataFrame, timestamps_context: pd.Series,
        timestamps_future: pd.Series, pred_len: int,
        n_samples: int = 20, temperature: float = 1.0
    ) -> dict:
        """
        Predict future OHLC bars and derive volatility metrics.

        Args:
            df_context: DataFrame with ['open','high','low','close'] columns
            timestamps_context: Timestamps for context bars
            timestamps_future: Timestamps for prediction horizon
            pred_len: Number of bars to predict
            n_samples: Number of sample paths for uncertainty estimation
            temperature: Sampling temperature (1.0 = calibrated)

        Returns:
            dict with predicted ATR, price ranges, and uncertainty bands
        """
        x_df = df_context[["open", "high", "low", "close"]].reset_index(drop=True)
        x_ts = timestamps_context.reset_index(drop=True)
        y_ts = timestamps_future.reset_index(drop=True)

        # Generate multiple sample paths for uncertainty
        all_preds = []
        for _ in range(n_samples):
            pred_df = self.predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=temperature,
                top_p=0.9,
                sample_count=1,
            )
            all_preds.append(pred_df)

        # Stack predictions: shape (n_samples, pred_len, 4)
        pred_arrays = np.array([p[["open", "high", "low", "close"]].values for p in all_preds])

        # Median prediction
        median_pred = np.median(pred_arrays, axis=0)

        # Predicted True Range per bar (from median)
        pred_highs = median_pred[:, 1]  # high column
        pred_lows = median_pred[:, 2]   # low column
        pred_closes = median_pred[:, 3]  # close column

        pred_tr = pred_highs - pred_lows  # simplified TR (no prev close)
        pred_atr = np.mean(pred_tr)       # average predicted ATR over horizon

        # Uncertainty: width of 80% prediction interval at each step
        q10 = np.percentile(pred_arrays[:, :, 3], 10, axis=0)  # close q10
        q90 = np.percentile(pred_arrays[:, :, 3], 90, axis=0)  # close q90
        uncertainty = np.mean(q90 - q10)

        return {
            "pred_atr": pred_atr,
            "pred_close_median": median_pred[-1, 3],  # last bar close
            "pred_range_q10": q10,
            "pred_range_q90": q90,
            "uncertainty": uncertainty,
            "n_samples": n_samples,
        }


# ---------------------------------------------------------------------------
# 3. NAIVE BASELINE
# ---------------------------------------------------------------------------

def naive_atr_baseline(current_atr: float) -> float:
    """Baseline: ATR stays the same (random walk of volatility)."""
    return current_atr


# ---------------------------------------------------------------------------
# 4. WALK-FORWARD EVALUATION
# ---------------------------------------------------------------------------

def run_evaluation(
    df: pd.DataFrame,
    test_start_idx: int,
    predictor: KronosVolatilityPredictor,
    context_len: int = 400,
    pred_horizon: int = 5,
    n_samples: int = 20,
    step_size: int = 20,     # evaluate every N bars (speed vs granularity)
    max_evals: int = 500,    # max number of evaluation points
):
    """
    Walk-forward evaluation: at each test point, predict ATR at +pred_horizon
    bars and compare against actual ATR and naive baseline.
    """
    results = []
    eval_indices = list(range(test_start_idx, len(df) - pred_horizon, step_size))

    if len(eval_indices) > max_evals:
        eval_indices = eval_indices[:max_evals]

    total = len(eval_indices)
    print(f"[EVAL] Running {total} evaluation points")
    print(f"[EVAL] Context: {context_len} bars, Horizon: {pred_horizon} bars ({pred_horizon * 15} min)")
    print(f"[EVAL] Step size: {step_size} bars, Samples per prediction: {n_samples}")
    print()

    t_start = time.time()

    for i, idx in enumerate(eval_indices):
        # Context window
        ctx_start = max(0, idx - context_len)
        ctx_df = df.iloc[ctx_start:idx]
        ctx_timestamps = df["timestamp"].iloc[ctx_start:idx]

        # Future window
        fut_timestamps = df["timestamp"].iloc[idx : idx + pred_horizon]

        # Actual ATR at horizon (what we're trying to predict)
        actual_atr_now = df["atr_7"].iloc[idx]
        actual_atr_future = df["atr_7"].iloc[idx + pred_horizon]

        if np.isnan(actual_atr_now) or np.isnan(actual_atr_future):
            continue

        # Actual volatility change direction
        actual_vol_direction = 1 if actual_atr_future > actual_atr_now else -1

        # --- Kronos prediction ---
        try:
            kronos_result = predictor.predict_bars(
                df_context=ctx_df,
                timestamps_context=ctx_timestamps,
                timestamps_future=fut_timestamps,
                pred_len=pred_horizon,
                n_samples=n_samples,
            )
            kronos_atr = kronos_result["pred_atr"]
            kronos_uncertainty = kronos_result["uncertainty"]
        except Exception as e:
            print(f"  [WARN] Kronos failed at idx {idx}: {e}")
            continue

        # --- Naive baseline ---
        naive_atr = naive_atr_baseline(actual_atr_now)

        # --- Errors ---
        kronos_error = abs(kronos_atr - actual_atr_future)
        naive_error = abs(naive_atr - actual_atr_future)

        # --- Directional accuracy ---
        kronos_vol_direction = 1 if kronos_atr > actual_atr_now else -1
        kronos_dir_correct = kronos_vol_direction == actual_vol_direction

        results.append({
            "timestamp": df["timestamp"].iloc[idx],
            "actual_atr_now": actual_atr_now,
            "actual_atr_future": actual_atr_future,
            "kronos_atr": kronos_atr,
            "naive_atr": naive_atr,
            "kronos_error": kronos_error,
            "naive_error": naive_error,
            "kronos_better": kronos_error < naive_error,
            "kronos_dir_correct": kronos_dir_correct,
            "kronos_uncertainty": kronos_uncertainty,
            "actual_vol_change_pct": (actual_atr_future - actual_atr_now) / actual_atr_now * 100,
        })

        # Progress
        elapsed = time.time() - t_start
        avg_per_eval = elapsed / (i + 1)
        eta = avg_per_eval * (total - i - 1)

        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i+1}/{total}] "
                f"ATR: {actual_atr_now:.2f}→{actual_atr_future:.2f} | "
                f"Kronos: {kronos_atr:.2f} (err={kronos_error:.3f}) | "
                f"Naive: {naive_atr:.2f} (err={naive_error:.3f}) | "
                f"Kronos wins: {kronos_error < naive_error} | "
                f"ETA: {eta/60:.1f}min"
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 5. ANALYSIS & REPORTING
# ---------------------------------------------------------------------------

def analyze_results(results_df: pd.DataFrame) -> dict:
    """Compute summary statistics from walk-forward evaluation."""

    n = len(results_df)
    if n == 0:
        print("[ERROR] No results to analyze!")
        return {}

    # --- Core Metrics ---
    kronos_mae = results_df["kronos_error"].mean()
    naive_mae = results_df["naive_error"].mean()
    improvement_pct = (naive_mae - kronos_mae) / naive_mae * 100

    kronos_wins = results_df["kronos_better"].sum()
    kronos_win_rate = kronos_wins / n * 100

    kronos_dir_accuracy = results_df["kronos_dir_correct"].mean() * 100

    # --- Correlation ---
    kronos_corr = np.corrcoef(
        results_df["actual_atr_future"].values,
        results_df["kronos_atr"].values,
    )[0, 1]

    naive_corr = np.corrcoef(
        results_df["actual_atr_future"].values,
        results_df["naive_atr"].values,
    )[0, 1]

    # --- Performance in volatile periods (ATR change > 20%) ---
    volatile_mask = abs(results_df["actual_vol_change_pct"]) > 20
    if volatile_mask.sum() > 0:
        volatile_df = results_df[volatile_mask]
        vol_kronos_mae = volatile_df["kronos_error"].mean()
        vol_naive_mae = volatile_df["naive_error"].mean()
        vol_kronos_wins = (volatile_df["kronos_error"] < volatile_df["naive_error"]).mean() * 100
        vol_n = len(volatile_df)
    else:
        vol_kronos_mae = vol_naive_mae = vol_kronos_wins = vol_n = 0

    # --- Uncertainty calibration ---
    # Does high uncertainty correlate with larger actual ATR changes?
    uncertainty_corr = np.corrcoef(
        results_df["kronos_uncertainty"].values,
        abs(results_df["actual_vol_change_pct"]).values,
    )[0, 1]

    metrics = {
        "n_evaluations": n,
        "kronos_mae": kronos_mae,
        "naive_mae": naive_mae,
        "improvement_pct": improvement_pct,
        "kronos_win_rate": kronos_win_rate,
        "kronos_dir_accuracy": kronos_dir_accuracy,
        "kronos_correlation": kronos_corr,
        "naive_correlation": naive_corr,
        "volatile_periods_n": vol_n,
        "volatile_kronos_mae": vol_kronos_mae,
        "volatile_naive_mae": vol_naive_mae,
        "volatile_kronos_win_rate": vol_kronos_wins,
        "uncertainty_vs_actual_change_corr": uncertainty_corr,
    }

    return metrics


def print_report(metrics: dict):
    """Print a clear, actionable report."""

    print("\n" + "=" * 70)
    print("  KRONOS VOLATILITY POC — RESULTS")
    print("=" * 70)

    print(f"\n  Evaluation points: {metrics['n_evaluations']}")

    print(f"\n  --- ATR Prediction Accuracy (MAE) ---")
    print(f"  Kronos MAE:        {metrics['kronos_mae']:.4f}")
    print(f"  Naive MAE:         {metrics['naive_mae']:.4f}")
    imp = metrics["improvement_pct"]
    symbol = "+" if imp > 0 else ""
    print(f"  Improvement:       {symbol}{imp:.1f}%")

    print(f"\n  --- Head-to-Head ---")
    print(f"  Kronos wins:       {metrics['kronos_win_rate']:.1f}% of evaluations")

    print(f"\n  --- Directional Accuracy ---")
    print(f"  Kronos predicts vol direction correctly: {metrics['kronos_dir_accuracy']:.1f}%")
    print(f"  (>50% = better than coin flip, >60% = useful, >70% = strong)")

    print(f"\n  --- Correlation with Actual Future ATR ---")
    print(f"  Kronos:  {metrics['kronos_correlation']:.4f}")
    print(f"  Naive:   {metrics['naive_correlation']:.4f}")

    if metrics["volatile_periods_n"] > 0:
        print(f"\n  --- Performance in Volatile Periods (ATR change >20%) ---")
        print(f"  N periods:         {metrics['volatile_periods_n']}")
        print(f"  Kronos MAE:        {metrics['volatile_kronos_mae']:.4f}")
        print(f"  Naive MAE:         {metrics['volatile_naive_mae']:.4f}")
        print(f"  Kronos win rate:   {metrics['volatile_kronos_win_rate']:.1f}%")

    print(f"\n  --- Uncertainty Calibration ---")
    uc = metrics["uncertainty_vs_actual_change_corr"]
    print(f"  Uncertainty ↔ Actual vol change: {uc:.4f}")
    print(f"  (>0.3 = useful for confidence scoring, >0.5 = strong)")

    # --- VERDICT ---
    print("\n" + "=" * 70)
    print("  VERDICT")
    print("=" * 70)

    go = True
    reasons = []

    if imp < 5:
        go = False
        reasons.append(f"MAE improvement only {imp:.1f}% (need >5%)")
    else:
        reasons.append(f"MAE improvement {imp:.1f}% (threshold: 5%)")

    if metrics["kronos_dir_accuracy"] < 55:
        go = False
        reasons.append(f"Directional accuracy {metrics['kronos_dir_accuracy']:.1f}% (need >55%)")
    else:
        reasons.append(f"Directional accuracy {metrics['kronos_dir_accuracy']:.1f}% (threshold: 55%)")

    if metrics["kronos_win_rate"] < 52:
        go = False
        reasons.append(f"Win rate {metrics['kronos_win_rate']:.1f}% (need >52%)")
    else:
        reasons.append(f"Win rate {metrics['kronos_win_rate']:.1f}% (threshold: 52%)")

    for r in reasons:
        status = "PASS" if "threshold" not in r or float(r.split("(")[0].strip().split()[-1].rstrip('%')) >= float(r.split(">")[1].rstrip('%)')) else "FAIL"
        print(f"  {'PASS' if 'threshold' in r else 'FAIL'}: {r}")

    if go:
        print("\n  >>> RECOMMENDATION: INTEGRATE Kronos for adaptive SL/TP <<<")
        print("  Next step: Add KronosVolPredictor to sentinel_scanner.py")
    else:
        print("\n  >>> RECOMMENDATION: DO NOT INTEGRATE Kronos <<<")
        print("  Kronos does not beat the naive baseline on Gold M15 volatility.")
        print("  Keep fixed ATR multipliers (2.0/4.0) in ConfluenceDetector.")

    print("=" * 70 + "\n")

    return go


# ---------------------------------------------------------------------------
# 6. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Kronos Volatility POC for Smart Sentinel AI")
    parser.add_argument("--device", default="cpu", help="cpu or cuda:0")
    parser.add_argument("--model", default="small", choices=["mini", "small", "base"],
                        help="Kronos model size")
    parser.add_argument("--context", type=int, default=400, help="Context window (bars)")
    parser.add_argument("--horizon", type=int, default=5, help="Prediction horizon (bars)")
    parser.add_argument("--samples", type=int, default=20, help="Sample paths per prediction")
    parser.add_argument("--step", type=int, default=20, help="Step between eval points (bars)")
    parser.add_argument("--max-evals", type=int, default=500, help="Max evaluation points")
    parser.add_argument("--test-start", default="2024-07-01", help="Test period start date")
    args = parser.parse_args()

    print("=" * 70)
    print("  KRONOS VOLATILITY POC — Smart Sentinel AI")
    print(f"  Model: Kronos-{args.model} | Device: {args.device}")
    print(f"  Context: {args.context} bars | Horizon: {args.horizon} bars ({args.horizon * 15} min)")
    print(f"  Samples: {args.samples} | Step: {args.step} | Max evals: {args.max_evals}")
    print("=" * 70 + "\n")

    # Setup
    setup_kronos()

    # Load data
    df, test_start_idx = load_gold_data(DATA_PATH, test_start=args.test_start)

    # Load model
    predictor = KronosVolatilityPredictor(
        model_name=args.model,
        device=args.device,
        max_context=args.context,
    )

    # Run evaluation
    print("[EVAL] Starting walk-forward evaluation...\n")
    results_df = run_evaluation(
        df=df,
        test_start_idx=test_start_idx,
        predictor=predictor,
        context_len=args.context,
        pred_horizon=args.horizon,
        n_samples=args.samples,
        step_size=args.step,
        max_evals=args.max_evals,
    )

    # Save raw results
    RESULTS_DIR.mkdir(exist_ok=True)
    results_path = RESULTS_DIR / f"kronos_poc_{args.model}_{args.horizon}bar.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\n[SAVE] Raw results saved to {results_path}")

    # Analyze
    metrics = analyze_results(results_df)

    # Save metrics
    metrics_path = RESULTS_DIR / f"kronos_poc_metrics_{args.model}_{args.horizon}bar.json"
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVE] Metrics saved to {metrics_path}")

    # Print report
    should_integrate = print_report(metrics)

    return 0 if should_integrate else 1


if __name__ == "__main__":
    sys.exit(main())
