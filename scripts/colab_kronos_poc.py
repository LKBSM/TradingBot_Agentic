"""
=============================================================================
COLAB VERSION: Kronos Volatility POC for Smart Sentinel AI
=============================================================================
Copy-paste each section into a separate Colab cell and run in order.

DATA SETUP (do ONE of these before running):
  Option A: Upload XAU_15MIN_2019_2025.csv via Colab file browser (left panel)
  Option B: Mount Google Drive and set DATA_PATH to your Drive path:
            from google.colab import drive
            drive.mount('/content/drive')
            DATA_PATH = "/content/drive/MyDrive/YOUR_FOLDER/XAU_15MIN_2019_2025.csv"

NOTE: The data CSV is NOT in the GitHub repo (gitignored due to size).
      You must provide it separately via upload or Google Drive.
=============================================================================
"""

# ============================================================================
# CELL 1: Setup & Install
# ============================================================================

import subprocess
import sys
import os

print("=" * 60)
print("  KRONOS VOLATILITY POC — COLAB SETUP")
print("=" * 60)

# Clone Kronos
if not os.path.exists("Kronos"):
    subprocess.run(["git", "clone", "https://github.com/shiyu-coder/Kronos.git"], check=True)
    print("[OK] Kronos cloned")
else:
    print("[OK] Kronos already present")

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "huggingface_hub", "safetensors", "pandas", "numpy"])
print("[OK] Dependencies installed")

sys.path.insert(0, "Kronos")

import torch
print(f"[OK] PyTorch {torch.__version__}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# CELL 2: Configuration
# ============================================================================

# --- EDIT THESE ---
DATA_PATH = "XAU_15MIN_2019_2025.csv"   # Or: "/content/drive/MyDrive/data/XAU_15MIN_2019_2025.csv"
MODEL_SIZE = "small"                      # "mini" (4M), "small" (25M), "base" (102M)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# --- POC Parameters ---
CONTEXT_LEN = 400       # Bars of history to feed Kronos (max 512)
PRED_HORIZON = 5        # Predict 5 bars ahead = 1h15
N_SAMPLES = 20          # Sample paths for uncertainty (more = slower but better)
STEP_SIZE = 20          # Evaluate every 20 bars (5 hours)
MAX_EVALS = 300         # Max evaluation points (300 × ~2s = ~10 min on GPU)
TEST_START = "2024-07-01"  # Out-of-sample test period

print(f"\nConfig: Kronos-{MODEL_SIZE} | {DEVICE} | {CONTEXT_LEN} context | {PRED_HORIZON} horizon")
print(f"Evals: max {MAX_EVALS} points, step {STEP_SIZE} bars\n")

# ============================================================================
# CELL 3: Load Data
# ============================================================================

import numpy as np
import pandas as pd
import time

print("Loading Gold M15 data...")
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df = df.sort_values("timestamp").reset_index(drop=True)

# Compute ATR (7-period)
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1)),
    ),
)
df["atr_7"] = df["tr"].rolling(7).mean()

test_mask = df["timestamp"] >= TEST_START
test_start_idx = df[test_mask].index[0]

print(f"Total bars: {len(df):,}")
print(f"Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
print(f"Test period: {TEST_START} (idx {test_start_idx}, {test_mask.sum():,} bars)\n")

# ============================================================================
# CELL 4: Load Kronos Model
# ============================================================================

from model import Kronos, KronosTokenizer, KronosPredictor

model_map = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-mini"),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base"),
}

model_id, tok_id = model_map[MODEL_SIZE]
print(f"Loading {model_id}...")

tokenizer = KronosTokenizer.from_pretrained(tok_id)
model = Kronos.from_pretrained(model_id)
predictor = KronosPredictor(model, tokenizer, device=DEVICE, max_context=CONTEXT_LEN)

print(f"Kronos-{MODEL_SIZE} loaded on {DEVICE}\n")

# ============================================================================
# CELL 5: Run Walk-Forward Evaluation
# ============================================================================

def predict_volatility(idx, pred_horizon, n_samples):
    """Predict ATR at idx + pred_horizon using Kronos."""
    ctx_start = max(0, idx - CONTEXT_LEN)
    x_df = df.iloc[ctx_start:idx][["open", "high", "low", "close"]].reset_index(drop=True)
    x_ts = df["timestamp"].iloc[ctx_start:idx].reset_index(drop=True)
    y_ts = df["timestamp"].iloc[idx:idx + pred_horizon].reset_index(drop=True)

    all_preds = []
    for _ in range(n_samples):
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=pred_horizon, T=1.0, top_p=0.9, sample_count=1,
        )
        all_preds.append(pred_df[["open", "high", "low", "close"]].values)

    pred_arr = np.array(all_preds)  # (n_samples, pred_horizon, 4)
    median = np.median(pred_arr, axis=0)

    # Predicted ATR = mean(High - Low) over horizon
    pred_atr = np.mean(median[:, 1] - median[:, 2])

    # Uncertainty = mean width of 80% interval on close
    q10 = np.percentile(pred_arr[:, :, 3], 10, axis=0)
    q90 = np.percentile(pred_arr[:, :, 3], 90, axis=0)
    uncertainty = np.mean(q90 - q10)

    return pred_atr, uncertainty


print("Starting walk-forward evaluation...")
print(f"{'='*70}")

eval_indices = list(range(test_start_idx, len(df) - PRED_HORIZON, STEP_SIZE))[:MAX_EVALS]
results = []
t0 = time.time()

for i, idx in enumerate(eval_indices):
    actual_atr_now = df["atr_7"].iloc[idx]
    actual_atr_future = df["atr_7"].iloc[idx + PRED_HORIZON]

    if np.isnan(actual_atr_now) or np.isnan(actual_atr_future):
        continue

    try:
        kronos_atr, kronos_unc = predict_volatility(idx, PRED_HORIZON, N_SAMPLES)
    except Exception as e:
        print(f"  [SKIP] idx {idx}: {e}")
        continue

    naive_atr = actual_atr_now  # Baseline: ATR stays the same

    k_err = abs(kronos_atr - actual_atr_future)
    n_err = abs(naive_atr - actual_atr_future)
    vol_change = (actual_atr_future - actual_atr_now) / actual_atr_now * 100
    k_dir = 1 if kronos_atr > actual_atr_now else -1
    a_dir = 1 if actual_atr_future > actual_atr_now else -1

    results.append({
        "ts": df["timestamp"].iloc[idx],
        "atr_now": actual_atr_now,
        "atr_future": actual_atr_future,
        "kronos_atr": kronos_atr,
        "naive_atr": naive_atr,
        "k_err": k_err,
        "n_err": n_err,
        "k_wins": k_err < n_err,
        "k_dir_ok": k_dir == a_dir,
        "k_unc": kronos_unc,
        "vol_chg_pct": vol_change,
    })

    elapsed = time.time() - t0
    eta = elapsed / (i + 1) * (len(eval_indices) - i - 1)

    if (i + 1) % 10 == 0:
        print(
            f"  [{i+1}/{len(eval_indices)}] "
            f"ATR {actual_atr_now:.2f}->{actual_atr_future:.2f} | "
            f"Kronos:{kronos_atr:.2f} Naive:{naive_atr:.2f} | "
            f"Winner:{'K' if k_err < n_err else 'N'} | "
            f"ETA:{eta/60:.1f}m"
        )

rdf = pd.DataFrame(results)
total_time = time.time() - t0
print(f"\nDone! {len(results)} evaluations in {total_time/60:.1f} min\n")

# ============================================================================
# CELL 6: Results Analysis
# ============================================================================

n = len(rdf)
if n == 0:
    print("ERROR: No results!")
else:
    k_mae = rdf["k_err"].mean()
    n_mae = rdf["n_err"].mean()
    imp = (n_mae - k_mae) / n_mae * 100
    k_wr = rdf["k_wins"].mean() * 100
    k_dir = rdf["k_dir_ok"].mean() * 100
    k_corr = np.corrcoef(rdf["atr_future"], rdf["kronos_atr"])[0, 1]
    n_corr = np.corrcoef(rdf["atr_future"], rdf["naive_atr"])[0, 1]
    unc_corr = np.corrcoef(rdf["k_unc"], abs(rdf["vol_chg_pct"]))[0, 1]

    # Volatile periods (ATR change > 20%)
    vol_mask = abs(rdf["vol_chg_pct"]) > 20
    vol_n = vol_mask.sum()

    print("=" * 70)
    print("  KRONOS VOLATILITY POC — FINAL RESULTS")
    print("=" * 70)
    print(f"\n  Evaluations: {n} | Time: {total_time/60:.1f} min")
    print(f"\n  {'METRIC':<40} {'KRONOS':>10} {'NAIVE':>10} {'WINNER':>10}")
    print(f"  {'-'*70}")
    print(f"  {'MAE (lower=better)':<40} {k_mae:>10.4f} {n_mae:>10.4f} {'KRONOS' if k_mae < n_mae else 'NAIVE':>10}")
    print(f"  {'Correlation with actual':<40} {k_corr:>10.4f} {n_corr:>10.4f} {'KRONOS' if k_corr > n_corr else 'NAIVE':>10}")
    print(f"  {'Head-to-head win rate':<40} {k_wr:>9.1f}% {100-k_wr:>9.1f}%  {'KRONOS' if k_wr > 50 else 'NAIVE':>10}")
    print(f"  {'Vol direction accuracy':<40} {k_dir:>9.1f}% {'50.0':>9}%  {'KRONOS' if k_dir > 50 else 'NAIVE':>10}")
    print(f"  {'Uncertainty calibration corr':<40} {unc_corr:>10.4f} {'N/A':>10}")

    if vol_n > 0:
        vk = rdf.loc[vol_mask, "k_err"].mean()
        vn = rdf.loc[vol_mask, "n_err"].mean()
        vwr = (rdf.loc[vol_mask, "k_err"] < rdf.loc[vol_mask, "n_err"]).mean() * 100
        print(f"\n  --- High Volatility Periods (ATR change >20%, n={vol_n}) ---")
        print(f"  {'MAE':<40} {vk:>10.4f} {vn:>10.4f} {'KRONOS' if vk < vn else 'NAIVE':>10}")
        print(f"  {'Win rate':<40} {vwr:>9.1f}% {100-vwr:>9.1f}%  {'KRONOS' if vwr > 50 else 'NAIVE':>10}")

    # --- VERDICT ---
    print(f"\n  {'='*70}")

    passes = 0
    checks = [
        ("MAE improvement > 5%", imp > 5),
        ("Win rate > 52%", k_wr > 52),
        ("Direction accuracy > 55%", k_dir > 55),
        ("Correlation > naive", k_corr > n_corr),
    ]
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        passes += passed
        print(f"  [{status}] {label}")

    print(f"\n  Score: {passes}/4")

    if passes >= 3:
        print("\n  >>> VERDICT: INTEGRATE Kronos for adaptive SL/TP <<<")
        print("  Kronos beats naive baseline on Gold M15 volatility.")
        print("  Next: Add KronosVolPredictor to sentinel_scanner.py pipeline.")
    elif passes >= 2:
        print("\n  >>> VERDICT: MARGINAL — test with Kronos-base before deciding <<<")
        print("  Mixed results. Try larger model or longer horizon.")
    else:
        print("\n  >>> VERDICT: DO NOT INTEGRATE <<<")
        print("  Kronos doesn't beat naive baseline. Keep fixed ATR multipliers.")

    print("=" * 70)

    # Save results
    rdf.to_csv("kronos_poc_results.csv", index=False)
    print(f"\nResults saved to kronos_poc_results.csv")
