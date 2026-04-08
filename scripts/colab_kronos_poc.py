"""
Kronos Volatility POC for Smart Sentinel AI
=============================================
Run in Google Colab (Runtime > Change runtime type > T4 GPU)

Single-cell execution:
  !git clone https://github.com/LKBSM/TradingBot_Agentic.git
  %run TradingBot_Agentic/scripts/colab_kronos_poc.py

Data auto-downloads from GitHub release v1.0-data.
"""

import subprocess, sys, os, time, urllib.request
import numpy as np
import pandas as pd

# ==========================================================================
# CONFIG — edit these if needed
# ==========================================================================
MODEL_SIZE   = "small"       # "mini" (4M), "small" (25M), "base" (102M)
CONTEXT_LEN  = 400           # bars of history fed to Kronos (max 512)
PRED_HORIZON = 5             # predict 5 bars ahead = 1h15
N_SAMPLES    = 20            # sample paths for uncertainty
STEP_SIZE    = 20            # evaluate every 20 bars (5 hours)
MAX_EVALS    = 300           # max evaluation points (~10-15 min on GPU)
TEST_START   = "2024-07-01"  # out-of-sample test period start

DATA_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/XAU_15MIN_2019_2025.csv"
DATA_FILE = "XAU_15MIN_2019_2025.csv"

# ==========================================================================
# 1. SETUP
# ==========================================================================
print("=" * 70)
print("  KRONOS VOLATILITY POC — Smart Sentinel AI")
print("=" * 70)

# Clone Kronos repo
if not os.path.exists("Kronos"):
    print("\n[1/4] Cloning Kronos...")
    subprocess.run(["git", "clone", "-q", "https://github.com/shiyu-coder/Kronos.git"], check=True)
print("[OK] Kronos repo ready")

# Install deps
print("[1/4] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "torch", "transformers", "huggingface_hub", "safetensors"],
               check=True)
sys.path.insert(0, "Kronos")

import torch
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[OK] PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")

# ==========================================================================
# 2. DOWNLOAD DATA
# ==========================================================================
if not os.path.exists(DATA_FILE):
    print(f"\n[2/4] Downloading data from GitHub release v1.0-data...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    print(f"[OK] {DATA_FILE} ({os.path.getsize(DATA_FILE) / 1024 / 1024:.1f} MB)")
else:
    print(f"\n[2/4] {DATA_FILE} already present")

# Load and prepare
df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df = df.sort_values("timestamp").reset_index(drop=True)

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

print(f"[OK] {len(df):,} bars | {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
print(f"[OK] Test: {TEST_START} onwards ({test_mask.sum():,} bars)")

# ==========================================================================
# 3. LOAD KRONOS MODEL
# ==========================================================================
print(f"\n[3/4] Loading Kronos-{MODEL_SIZE}...")
from model import Kronos, KronosTokenizer, KronosPredictor

model_map = {
    "mini":  ("NeoQuasar/Kronos-mini",  "NeoQuasar/Kronos-Tokenizer-mini"),
    "small": ("NeoQuasar/Kronos-small", "NeoQuasar/Kronos-Tokenizer-base"),
    "base":  ("NeoQuasar/Kronos-base",  "NeoQuasar/Kronos-Tokenizer-base"),
}
model_id, tok_id = model_map[MODEL_SIZE]

tokenizer = KronosTokenizer.from_pretrained(tok_id)
kronos_model = Kronos.from_pretrained(model_id)
predictor = KronosPredictor(kronos_model, tokenizer, device=DEVICE, max_context=CONTEXT_LEN)
print(f"[OK] Kronos-{MODEL_SIZE} on {DEVICE}")

# ==========================================================================
# 4. WALK-FORWARD EVALUATION
# ==========================================================================
def predict_volatility(idx):
    ctx_start = max(0, idx - CONTEXT_LEN)
    x_df = df.iloc[ctx_start:idx][["open", "high", "low", "close"]].reset_index(drop=True)
    x_ts = df["timestamp"].iloc[ctx_start:idx].reset_index(drop=True)
    y_ts = df["timestamp"].iloc[idx:idx + PRED_HORIZON].reset_index(drop=True)

    all_preds = []
    for _ in range(N_SAMPLES):
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_ts, y_timestamp=y_ts,
            pred_len=PRED_HORIZON, T=1.0, top_p=0.9, sample_count=1,
        )
        all_preds.append(pred_df[["open", "high", "low", "close"]].values)

    pred_arr = np.array(all_preds)
    median = np.median(pred_arr, axis=0)
    pred_atr = np.mean(median[:, 1] - median[:, 2])

    q10 = np.percentile(pred_arr[:, :, 3], 10, axis=0)
    q90 = np.percentile(pred_arr[:, :, 3], 90, axis=0)
    uncertainty = np.mean(q90 - q10)
    return pred_atr, uncertainty


print(f"\n[4/4] Running {MAX_EVALS} evaluations (horizon={PRED_HORIZON} bars, step={STEP_SIZE})...")
print("=" * 70)

eval_indices = list(range(test_start_idx, len(df) - PRED_HORIZON, STEP_SIZE))[:MAX_EVALS]
results = []
t0 = time.time()

for i, idx in enumerate(eval_indices):
    actual_atr_now = df["atr_7"].iloc[idx]
    actual_atr_future = df["atr_7"].iloc[idx + PRED_HORIZON]

    if np.isnan(actual_atr_now) or np.isnan(actual_atr_future):
        continue

    try:
        kronos_atr, kronos_unc = predict_volatility(idx)
    except Exception as e:
        print(f"  [SKIP] idx {idx}: {e}")
        continue

    naive_atr = actual_atr_now
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
            f"{'K wins' if k_err < n_err else 'N wins'} | "
            f"ETA:{eta/60:.1f}m"
        )

rdf = pd.DataFrame(results)
total_time = time.time() - t0
print(f"\nDone! {len(results)} evaluations in {total_time/60:.1f} min")

# ==========================================================================
# 5. RESULTS & VERDICT
# ==========================================================================
n = len(rdf)
if n == 0:
    print("\nERROR: No results produced!")
else:
    k_mae = rdf["k_err"].mean()
    n_mae = rdf["n_err"].mean()
    imp = (n_mae - k_mae) / n_mae * 100
    k_wr = rdf["k_wins"].mean() * 100
    k_dir_acc = rdf["k_dir_ok"].mean() * 100
    k_corr = np.corrcoef(rdf["atr_future"], rdf["kronos_atr"])[0, 1]
    n_corr = np.corrcoef(rdf["atr_future"], rdf["naive_atr"])[0, 1]
    unc_corr = np.corrcoef(rdf["k_unc"], abs(rdf["vol_chg_pct"]))[0, 1]

    vol_mask = abs(rdf["vol_chg_pct"]) > 20
    vol_n = vol_mask.sum()

    print("\n" + "=" * 70)
    print("  KRONOS VOLATILITY POC - FINAL RESULTS")
    print("=" * 70)
    print(f"\n  Evaluations: {n} | Time: {total_time/60:.1f} min")
    print(f"  Model: Kronos-{MODEL_SIZE} | Device: {DEVICE}")
    print(f"  Context: {CONTEXT_LEN} bars | Horizon: {PRED_HORIZON} bars ({PRED_HORIZON*15}min)")
    print(f"\n  {'METRIC':<40} {'KRONOS':>10} {'NAIVE':>10} {'WINNER':>10}")
    print(f"  {'-'*70}")
    print(f"  {'MAE (lower=better)':<40} {k_mae:>10.4f} {n_mae:>10.4f} {'KRONOS' if k_mae < n_mae else 'NAIVE':>10}")
    print(f"  {'Correlation with actual':<40} {k_corr:>10.4f} {n_corr:>10.4f} {'KRONOS' if k_corr > n_corr else 'NAIVE':>10}")
    print(f"  {'Head-to-head win rate':<40} {k_wr:>9.1f}% {100-k_wr:>9.1f}%  {'KRONOS' if k_wr > 50 else 'NAIVE':>10}")
    print(f"  {'Vol direction accuracy':<40} {k_dir_acc:>9.1f}% {'50.0':>9}%  {'KRONOS' if k_dir_acc > 50 else 'COIN':>10}")
    print(f"  {'Uncertainty calibration corr':<40} {unc_corr:>10.4f} {'N/A':>10}")

    if vol_n > 0:
        vk = rdf.loc[vol_mask, "k_err"].mean()
        vn = rdf.loc[vol_mask, "n_err"].mean()
        vwr = (rdf.loc[vol_mask, "k_err"] < rdf.loc[vol_mask, "n_err"]).mean() * 100
        print(f"\n  --- High Volatility Periods (ATR change >20%, n={vol_n}) ---")
        print(f"  {'MAE':<40} {vk:>10.4f} {vn:>10.4f} {'KRONOS' if vk < vn else 'NAIVE':>10}")
        print(f"  {'Win rate':<40} {vwr:>9.1f}% {100-vwr:>9.1f}%  {'KRONOS' if vwr > 50 else 'NAIVE':>10}")

    print(f"\n  {'='*70}")

    passes = 0
    checks = [
        ("MAE improvement > 5%", imp > 5),
        ("Win rate > 52%", k_wr > 52),
        ("Direction accuracy > 55%", k_dir_acc > 55),
        ("Correlation > naive", k_corr > n_corr),
    ]
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        passes += int(passed)
        print(f"  [{status}] {label}")

    print(f"\n  Score: {passes}/4")

    if passes >= 3:
        print("\n  >>> VERDICT: INTEGRER Kronos pour adaptive SL/TP <<<")
    elif passes >= 2:
        print("\n  >>> VERDICT: MARGINAL - tester avec Kronos-base <<<")
    else:
        print("\n  >>> VERDICT: NE PAS INTEGRER <<<")

    print("=" * 70)

    rdf.to_csv("kronos_poc_results.csv", index=False)
    print(f"\nResultats sauvegardes: kronos_poc_results.csv")
