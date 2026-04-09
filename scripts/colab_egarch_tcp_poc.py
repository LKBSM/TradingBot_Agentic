"""
EGARCH + TCP Volatility POC for Smart Sentinel AI
===================================================
Run in Google Colab (no GPU needed — CPU is sufficient)

Single-cell execution:
  !git clone https://github.com/LKBSM/TradingBot_Agentic.git
  %run TradingBot_Agentic/scripts/colab_egarch_tcp_poc.py

Data auto-downloads from GitHub release v1.0-data.

Replaces Kronos (scored 1/4). Uses:
  - EGARCH(1,1) with Student-t distribution for asymmetric volatility
  - Temporal Conformal Prediction (TCP) for calibrated intervals
"""

import subprocess, sys, os, time, urllib.request, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================================================
# CONFIG
# ==========================================================================
PRED_HORIZON   = 5             # predict 5 bars ahead = 1h15
STEP_SIZE      = 20            # evaluate every 20 bars
MAX_EVALS      = 300           # max evaluation points
TEST_START     = "2024-07-01"  # out-of-sample start
FIT_WINDOW     = 2000          # rolling window for EGARCH fit (bars)
TCP_ALPHA      = 0.05          # target miscoverage (95% intervals)
TCP_GAMMA      = 0.01          # TCP learning rate (Robbins-Monro)

DATA_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/XAU_15MIN_2019_2025.csv"
DATA_FILE = "XAU_15MIN_2019_2025.csv"

# ==========================================================================
# 1. SETUP
# ==========================================================================
print("=" * 70)
print("  EGARCH + TCP VOLATILITY POC — Smart Sentinel AI")
print("=" * 70)

print("\n[1/4] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "arch", "scipy"], check=True)
print("[OK] Dependencies installed")

from arch import arch_model
from scipy import stats

# ==========================================================================
# 2. DATA
# ==========================================================================
print("\n[2/4] Loading data...")

if not os.path.exists(DATA_FILE):
    print(f"  Downloading from GitHub release...")
    urllib.request.urlretrieve(DATA_URL, DATA_FILE)

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Compute returns and realized volatility
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
df["returns_pct"] = df["log_return"] * 100  # EGARCH needs percentage returns

# Compute ATR (True Range)
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    )
)
df["ATR_14"] = df["tr"].rolling(14).mean()

# Future realized volatility (what we're predicting)
df["future_atr"] = df["tr"].shift(-1).rolling(PRED_HORIZON).mean().shift(-(PRED_HORIZON - 1))

df.dropna(subset=["ATR_14", "future_atr", "returns_pct"], inplace=True)

test_mask = df["timestamp"] >= TEST_START
test_indices = df.index[test_mask].tolist()

print(f"  Total bars: {len(df):,}")
print(f"  Test bars:  {len(test_indices):,} (from {TEST_START})")
print(f"  ATR mean:   {df['ATR_14'].mean():.4f}")

# ==========================================================================
# 3. WALK-FORWARD EVALUATION
# ==========================================================================
print(f"\n[3/4] Walk-forward evaluation ({MAX_EVALS} points)...")

eval_points = test_indices[::STEP_SIZE][:MAX_EVALS]
n_evals = len(eval_points)

results = []
tcp_width = 1.0  # Initial TCP interval width multiplier (adapts online)

start_time = time.time()
fit_count = 0
last_model = None
last_fit_idx = -FIT_WINDOW  # Force first fit

for eval_num, idx in enumerate(eval_points):
    # Get position in dataframe
    pos = df.index.get_loc(idx) if isinstance(idx, int) and idx in df.index else idx

    # Rolling window of returns
    window_start = max(0, pos - FIT_WINDOW)
    returns_window = df["returns_pct"].iloc[window_start:pos].values

    if len(returns_window) < 200:
        continue

    # ---------- EGARCH FIT (refit every 100 steps) ----------
    if pos - last_fit_idx >= 100 or last_model is None:
        try:
            am = arch_model(
                returns_window,
                vol="EGARCH",
                p=1, o=1, q=1,
                dist="studentst",
                mean="Constant",
            )
            res = am.fit(disp="off", show_warning=False)
            last_model = res
            last_fit_idx = pos
            fit_count += 1
        except Exception:
            if last_model is None:
                continue
            # Use last successful model

    # ---------- FORECAST ----------
    try:
        forecast = last_model.forecast(
            horizon=PRED_HORIZON,
            reindex=False
        )
        # Conditional variance → conditional volatility (sigma per bar)
        predicted_var = forecast.variance.values[-1]  # (horizon,) array
        predicted_vol = np.sqrt(predicted_var)  # annualized-ish pct vol per bar

        # Convert percentage vol to price-space ATR estimate
        # sigma_pct * close / 100 ≈ expected bar range
        current_close = df["close"].iloc[pos]
        egarch_atr = np.mean(predicted_vol) * current_close / 100.0

    except Exception:
        continue

    # ---------- NAIVE BASELINE ----------
    naive_atr = df["ATR_14"].iloc[pos]

    # ---------- ACTUAL ----------
    actual_atr = df["future_atr"].iloc[pos]

    # ---------- TCP CONFORMAL UPDATE ----------
    # Prediction interval: [egarch_atr - tcp_width*egarch_atr, egarch_atr + tcp_width*egarch_atr]
    interval_lower = egarch_atr * (1 - tcp_width)
    interval_upper = egarch_atr * (1 + tcp_width)
    covered = interval_lower <= actual_atr <= interval_upper

    # Robbins-Monro update: shrink if covered, expand if not
    tcp_width = tcp_width + TCP_GAMMA * (TCP_ALPHA - (0.0 if covered else 1.0))
    tcp_width = max(0.05, min(2.0, tcp_width))  # Clamp

    # ---------- METRICS ----------
    e_err = abs(egarch_atr - actual_atr)
    n_err = abs(naive_atr - actual_atr)

    vol_change = (actual_atr - naive_atr) / naive_atr * 100 if naive_atr > 0 else 0
    e_dir = np.sign(egarch_atr - naive_atr)
    a_dir = np.sign(actual_atr - naive_atr)
    dir_correct = (e_dir == a_dir) if a_dir != 0 else False

    results.append({
        "timestamp": df["timestamp"].iloc[pos],
        "actual_atr": actual_atr,
        "egarch_atr": egarch_atr,
        "naive_atr": naive_atr,
        "e_err": e_err,
        "n_err": n_err,
        "e_wins": e_err < n_err,
        "dir_ok": dir_correct,
        "vol_chg_pct": vol_change,
        "tcp_width": tcp_width,
        "covered": covered,
        "interval_lower": interval_lower,
        "interval_upper": interval_upper,
    })

    if (eval_num + 1) % 50 == 0:
        elapsed = time.time() - start_time
        print(f"  [{eval_num+1}/{n_evals}] {elapsed:.0f}s | "
              f"EGARCH MAE={np.mean([r['e_err'] for r in results]):.4f} | "
              f"Naive MAE={np.mean([r['n_err'] for r in results]):.4f} | "
              f"TCP width={tcp_width:.3f}")

total_time = time.time() - start_time

# ==========================================================================
# 4. RESULTS
# ==========================================================================
rdf = pd.DataFrame(results)
n = len(rdf)

if n < 10:
    print(f"\nERROR: Only {n} evaluation points. Check data range.")
    sys.exit(1)

# Core metrics
e_mae = rdf["e_err"].mean()
n_mae = rdf["n_err"].mean()
imp = (1 - e_mae / n_mae) * 100 if n_mae > 0 else 0

e_wr = rdf["e_wins"].mean() * 100
e_dir_acc = rdf["dir_ok"].mean() * 100
e_corr = np.corrcoef(rdf["actual_atr"], rdf["egarch_atr"])[0, 1]
n_corr = np.corrcoef(rdf["actual_atr"], rdf["naive_atr"])[0, 1]

# TCP calibration metrics
tcp_coverage = rdf["covered"].mean() * 100
tcp_avg_width = rdf["tcp_width"].mean()
tcp_final_width = rdf["tcp_width"].iloc[-1]

# High-volatility regime performance
vol_mask = abs(rdf["vol_chg_pct"]) > 20
vol_n = vol_mask.sum()

print("\n" + "=" * 70)
print("  EGARCH + TCP VOLATILITY POC - FINAL RESULTS")
print("=" * 70)
print(f"\n  Evaluations: {n} | Time: {total_time/60:.1f} min | EGARCH refits: {fit_count}")
print(f"  Model: EGARCH(1,1) Student-t | Fit window: {FIT_WINDOW} bars")
print(f"  Horizon: {PRED_HORIZON} bars ({PRED_HORIZON*15}min) | TCP alpha: {TCP_ALPHA}")

print(f"\n  {'METRIC':<40} {'EGARCH':>10} {'NAIVE':>10} {'WINNER':>10}")
print(f"  {'-'*70}")
print(f"  {'MAE (lower=better)':<40} {e_mae:>10.4f} {n_mae:>10.4f} {'EGARCH' if e_mae < n_mae else 'NAIVE':>10}")
print(f"  {'Correlation with actual':<40} {e_corr:>10.4f} {n_corr:>10.4f} {'EGARCH' if e_corr > n_corr else 'NAIVE':>10}")
print(f"  {'Head-to-head win rate':<40} {e_wr:>9.1f}% {100-e_wr:>9.1f}%  {'EGARCH' if e_wr > 50 else 'NAIVE':>10}")
print(f"  {'Vol direction accuracy':<40} {e_dir_acc:>9.1f}% {'50.0':>9}%  {'EGARCH' if e_dir_acc > 50 else 'COIN':>10}")

print(f"\n  --- TCP Conformal Prediction (target: {(1-TCP_ALPHA)*100:.0f}% coverage) ---")
print(f"  {'Empirical coverage':<40} {tcp_coverage:>9.1f}%")
print(f"  {'Average interval width':<40} {tcp_avg_width:>10.3f}")
print(f"  {'Final interval width':<40} {tcp_final_width:>10.3f}")

calibration_ok = abs(tcp_coverage - (1 - TCP_ALPHA) * 100) < 5
print(f"  {'Calibration quality':<40} {'GOOD' if calibration_ok else 'NEEDS TUNING':>10}")

if vol_n > 0:
    vk = rdf.loc[vol_mask, "e_err"].mean()
    vn = rdf.loc[vol_mask, "n_err"].mean()
    vwr = (rdf.loc[vol_mask, "e_err"] < rdf.loc[vol_mask, "n_err"]).mean() * 100
    vcov = rdf.loc[vol_mask, "covered"].mean() * 100
    print(f"\n  --- High Volatility Periods (ATR change >20%, n={vol_n}) ---")
    print(f"  {'MAE':<40} {vk:>10.4f} {vn:>10.4f} {'EGARCH' if vk < vn else 'NAIVE':>10}")
    print(f"  {'Win rate':<40} {vwr:>9.1f}% {100-vwr:>9.1f}%  {'EGARCH' if vwr > 50 else 'NAIVE':>10}")
    print(f"  {'TCP coverage during vol spikes':<40} {vcov:>9.1f}%")

print(f"\n  {'='*70}")

# ==========================================================================
# GO/NO-GO CRITERIA
# ==========================================================================
passes = 0
checks = [
    ("MAE improvement > 5%",          imp > 5),
    ("Win rate > 52%",                 e_wr > 52),
    ("Direction accuracy > 55%",       e_dir_acc > 55),
    ("Correlation > naive",            e_corr > n_corr),
    ("TCP coverage within 5% target",  calibration_ok),
]
for label, passed in checks:
    status = "PASS" if passed else "FAIL"
    passes += int(passed)
    print(f"  [{status}] {label}")

print(f"\n  Score: {passes}/5")

if passes >= 4:
    print("\n  >>> VERDICT: INTEGRATE EGARCH+TCP for adaptive SL/TP <<<")
elif passes >= 3:
    print("\n  >>> VERDICT: PROMISING — fine-tune parameters <<<")
elif passes >= 2:
    print("\n  >>> VERDICT: MARGINAL — consider HAR-RV ensemble <<<")
else:
    print("\n  >>> VERDICT: DO NOT INTEGRATE <<<")

print("=" * 70)

# Save results
rdf.to_csv("egarch_tcp_poc_results.csv", index=False)
print(f"\nResults saved: egarch_tcp_poc_results.csv")

# ==========================================================================
# COMPARISON WITH KRONOS
# ==========================================================================
print(f"\n{'='*70}")
print("  EGARCH vs KRONOS COMPARISON")
print(f"{'='*70}")
print(f"  Kronos score:    1/4  (FAILED)")
print(f"  EGARCH+TCP score: {passes}/5")
print(f"  EGARCH fit time:  {total_time/60:.1f} min (CPU only, no GPU needed)")
print(f"  Inference:        ~{total_time/n*1000:.0f}ms per forecast")
print(f"  Dependencies:     arch, scipy (vs torch, transformers for Kronos)")
print(f"{'='*70}")
