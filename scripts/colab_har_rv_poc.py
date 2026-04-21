"""
HAR-RV + Diurnal + Calendar + HMM  Volatility POC v1
=====================================================
Run in Google Colab (no GPU needed — CPU is sufficient)

Single-cell execution:
  !git clone https://github.com/LKBSM/TradingBot_Agentic.git
  %run TradingBot_Agentic/scripts/colab_har_rv_poc.py

Replaces EGARCH POC (scored 1/5). Uses four orthogonal signal sources:
  1. HAR-RV (Heterogeneous Autoregressive Realized Volatility) — Corsi 2009
     Multi-scale persistence: daily + weekly + monthly lookback
  2. Yang-Zhang realized volatility estimator — 14x more efficient than close-to-close
  3. Diurnal profile — intraday seasonality multiplier per hour (ATR's biggest blind spot)
  4. Calendar event multipliers — NFP/CPI/FOMC known vol spikes
  5. HMM regime detection — 3-state (low/normal/high vol) conditioning
  6. TCP conformal prediction — calibrated prediction intervals

Architecture:  vol_forecast = HAR_RV_base * diurnal(hour) * calendar(event) * regime(HMM)
Each component addresses a DIFFERENT failure mode of naive ATR.
"""

import subprocess, sys, os, time, urllib.request, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ==========================================================================
# CONFIG
# ==========================================================================
PRED_HORIZON     = 5            # predict 5 bars ahead (75min)
STEP_SIZE        = 20           # evaluate every 20 bars
MAX_EVALS        = 500          # max evaluation points
TEST_START       = "2024-07-01" # out-of-sample start
HAR_TRAIN_MIN    = 2200         # min bars to train HAR (need 2112 for monthly)
HMM_N_STATES     = 3            # low / normal / high vol regimes
DIURNAL_STRENGTH = 0.5          # dampen diurnal: 0=ignore, 1=full raw adjustment
BLEND_GRID       = True         # calibrate HAR/naive blend weight on training data
TCP_ALPHA        = 0.05         # target miscoverage (95% intervals)
TCP_GAMMA        = 0.05         # TCP learning rate (Robbins-Monro)

# HAR lookback windows (in M15 bars)
HAR_DAILY   = 96    # 1 day  = 96 M15 bars
HAR_WEEKLY  = 480   # 5 days = 480 M15 bars
HAR_MONTHLY = 2112  # 22 days = 2112 M15 bars

# Calendar event detection
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls", "Federal Funds Rate", "CPI m/m", "Core CPI m/m",
    "FOMC Statement", "FOMC Press Conference", "GDP q/q",
    "Core PCE Price Index m/m", "Retail Sales m/m",
]
EVENT_WINDOW_HOURS = 4  # hours before/after event to apply multiplier

DATA_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/XAU_15MIN_2019_2025.csv"
DATA_FILE = "XAU_15MIN_2019_2025.csv"
CAL_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/economic_calendar_2019_2025.csv"
CAL_FILE = "economic_calendar_2019_2025.csv"

# ==========================================================================
# 1. SETUP
# ==========================================================================
print("=" * 70)
print("  HAR-RV + DIURNAL + CALENDAR + HMM  VOL POC — Smart Sentinel AI")
print("=" * 70)

print("\n[1/6] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "hmmlearn", "scikit-learn", "scipy"], check=True)
print("[OK] Dependencies installed")

from sklearn.linear_model import LinearRegression
from hmmlearn.hmm import GaussianHMM

# ==========================================================================
# 2. DATA
# ==========================================================================
print("\n[2/6] Loading data...")

for url, fname in [(DATA_URL, DATA_FILE), (CAL_URL, CAL_FILE)]:
    if not os.path.exists(fname):
        print(f"  Downloading {fname}...")
        urllib.request.urlretrieve(url, fname)

df = pd.read_csv(DATA_FILE, parse_dates=["Date"])
df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)

# Load economic calendar
cal = pd.read_csv(CAL_FILE, parse_dates=["Date"])
cal.columns = ["timestamp", "currency", "event", "impact", "actual", "forecast", "previous"]
# Filter to high-impact USD events that move Gold
cal_high = cal[
    (cal["impact"] == "HIGH") &
    (cal["currency"] == "USD") &
    (cal["event"].isin(HIGH_IMPACT_EVENTS))
].copy()
cal_high = cal_high.sort_values("timestamp").reset_index(drop=True)
event_times = cal_high["timestamp"].values  # numpy datetime64 array
print(f"  High-impact events loaded: {len(cal_high)}")

# ==========================================================================
# 3. FEATURE ENGINEERING
# ==========================================================================
print("\n[3/6] Computing features...")

# --- Log returns (percentage) ---
df["returns_pct"] = np.log(df["close"] / df["close"].shift(1)) * 100

# --- True Range and ATR_14 (naive baseline) ---
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    )
)
df["ATR_14"] = df["tr"].rolling(14).mean()

# --- Yang-Zhang Realized Volatility (per bar) ---
# YZ is 14x more efficient than close-to-close for OHLC data
log_ho = np.log(df["high"] / df["open"])
log_lo = np.log(df["low"] / df["open"])
log_co = np.log(df["close"] / df["open"])
log_oc = np.log(df["open"] / df["close"].shift(1))

# Rogers-Satchell component (per bar)
rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)

# Yang-Zhang rolling RV (14-bar window)
yz_window = 14
k = 0.34 / (1.0 + (yz_window + 1) / (yz_window - 1))
sigma_overnight = log_oc.rolling(yz_window).var()
sigma_oc = log_co.rolling(yz_window).var()
sigma_rs = rs_var.rolling(yz_window).mean()
df["yz_rv"] = np.sqrt(np.abs(sigma_overnight + k * sigma_oc + (1 - k) * sigma_rs))

# --- HAR-RV components (multi-scale) ---
# Use Yang-Zhang per-bar squared vol as the base RV measure
df["rv_bar"] = rs_var.clip(lower=0)  # per-bar RV (Rogers-Satchell, non-negative)
df["rv_daily"]   = df["rv_bar"].rolling(HAR_DAILY).mean()
df["rv_weekly"]  = df["rv_bar"].rolling(HAR_WEEKLY).mean()
df["rv_monthly"] = df["rv_bar"].rolling(HAR_MONTHLY).mean()

# --- Future realized ATR: mean TR over next PRED_HORIZON bars ---
df["future_atr"] = df["tr"].rolling(PRED_HORIZON).mean().shift(-PRED_HORIZON)

# --- Hour of day (for diurnal profile) ---
df["hour"] = df["timestamp"].dt.hour

# --- Calendar event proximity ---
# For each bar, compute hours to nearest high-impact event
def compute_event_proximity(timestamps, event_times, window_hours=4):
    """Compute event multiplier: 1.0 = no event, >1.0 = near event."""
    result = np.ones(len(timestamps))
    window_ns = np.timedelta64(window_hours, 'h')

    ts_values = timestamps.values
    for evt in event_times:
        # Find bars within window of this event
        delta = np.abs(ts_values - evt)
        mask = delta <= window_ns
        if not mask.any():
            continue
        # Multiplier: strongest at event time, decays linearly
        hours_away = delta[mask].astype('timedelta64[m]').astype(float) / 60.0
        multiplier = 1.0 + 1.5 * np.maximum(0, 1 - hours_away / window_hours)
        # Take max if overlapping events
        result[mask] = np.maximum(result[mask], multiplier)
    return result

print("  Computing event proximity (this may take a moment)...")
df["event_mult"] = compute_event_proximity(df["timestamp"], event_times, EVENT_WINDOW_HOURS)
event_bars = (df["event_mult"] > 1.0).sum()
print(f"  Bars near high-impact events: {event_bars:,} ({event_bars/len(df)*100:.1f}%)")

# --- Drop NaN rows, reset index ---
df.dropna(subset=["ATR_14", "future_atr", "rv_daily", "rv_weekly", "rv_monthly",
                   "yz_rv", "returns_pct"], inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Split ---
test_start_idx = df.index[df["timestamp"] >= TEST_START].min()
test_end_idx = df.index.max()

print(f"  Total bars after cleanup: {len(df):,}")
print(f"  Test range: idx {test_start_idx} to {test_end_idx} ({test_end_idx - test_start_idx:,} bars)")
print(f"  ATR mean: {df['ATR_14'].mean():.4f}")
print(f"  YZ RV mean: {df['yz_rv'].mean():.6f}")

# ==========================================================================
# 4. DIURNAL PROFILE (from training data)
# ==========================================================================
print("\n[4/6] Computing diurnal profile from training data...")

train_df = df.loc[:test_start_idx - 1]

# Average TR by hour of day, normalized so overall mean = 1.0
hourly_tr = train_df.groupby("hour")["tr"].mean()
diurnal_profile = (hourly_tr / hourly_tr.mean()).to_dict()

print("  Diurnal profile (hour: multiplier):")
for h in sorted(diurnal_profile.keys()):
    bar = "#" * int(diurnal_profile[h] * 20)
    label = ""
    if h == 0: label = " (Asian start)"
    elif h == 7: label = " (London open)"
    elif h == 13: label = " (NY open)"
    elif h == 17: label = " (NY afternoon)"
    elif h == 21: label = " (After-hours)"
    print(f"    {h:02d}:00 = {diurnal_profile[h]:.3f} {bar}{label}")

# ==========================================================================
# 4b. CALIBRATE BLEND WEIGHT (on training data)
# ==========================================================================
print("\n[4b/6] Calibrating HAR/naive blend weight...")

# Fit a HAR model on first 80% of training, validate on last 20%
calib_split = int(len(train_df) * 0.8)
calib_train = train_df.iloc[:calib_split]
calib_val = train_df.iloc[calib_split:]

# Fit HAR on calibration training set
calib_valid_mask = calib_train[["rv_daily", "rv_weekly", "rv_monthly", "ATR_14", "future_atr"]].notna().all(axis=1)
calib_train_valid = calib_train[calib_valid_mask]
calib_val_valid_mask = calib_val[["rv_daily", "rv_weekly", "rv_monthly", "ATR_14", "future_atr"]].notna().all(axis=1)
calib_val_valid = calib_val[calib_val_valid_mask]

blend_w = 0.5  # default
if len(calib_train_valid) > 500 and len(calib_val_valid) > 100:
    X_ct = calib_train_valid[["rv_daily", "rv_weekly", "rv_monthly", "ATR_14"]].values
    y_ct = calib_train_valid["future_atr"].values
    calib_har = LinearRegression().fit(X_ct, y_ct)

    # Predict on validation set
    X_cv = calib_val_valid[["rv_daily", "rv_weekly", "rv_monthly", "ATR_14"]].values
    y_cv = calib_val_valid["future_atr"].values
    har_pred_cv = calib_har.predict(X_cv).clip(min=0.01)
    naive_cv = calib_val_valid["ATR_14"].values

    # Apply dampened diurnal + calendar multipliers on validation set
    hours_cv = calib_val_valid["hour"].values
    cal_cv = calib_val_valid["event_mult"].values
    diurnal_cv = np.array([1.0 + DIURNAL_STRENGTH * (diurnal_profile.get(int(h), 1.0) - 1.0)
                           for h in hours_cv])
    har_adj_cv = har_pred_cv * diurnal_cv * cal_cv

    # Grid search for best blend weight
    best_w, best_mae = 0.5, float('inf')
    for w in np.arange(0.05, 0.96, 0.05):
        blended = w * har_adj_cv + (1 - w) * naive_cv
        mae = np.mean(np.abs(blended - y_cv))
        if mae < best_mae:
            best_mae = mae
            best_w = w

    blend_w = best_w
    naive_mae_cv = np.mean(np.abs(naive_cv - y_cv))
    print(f"  Best blend weight: {blend_w:.2f} (HAR) / {1-blend_w:.2f} (naive)")
    print(f"  Validation MAE: blended={best_mae:.4f}, naive={naive_mae_cv:.4f}, "
          f"improvement={((1 - best_mae/naive_mae_cv) * 100):.1f}%")
else:
    print(f"  Insufficient calibration data, using default blend_w={blend_w}")

# ==========================================================================
# 5. WALK-FORWARD EVALUATION
# ==========================================================================
print(f"\n[5/6] Walk-forward evaluation...")

eval_indices = list(range(test_start_idx, test_end_idx, STEP_SIZE))[:MAX_EVALS]
n_evals = len(eval_indices)
print(f"  Eval points: {n_evals}")

results = []
tcp_width = 0.5
start_time = time.time()
har_fit_count = 0
hmm_fit_count = 0
skip_count = 0

# Refit intervals
HAR_REFIT_EVERY  = 50   # refit HAR every 50 eval steps (~1000 bars)
HMM_REFIT_EVERY  = 100  # refit HMM every 100 eval steps (~2000 bars)

har_model = None
har_last_fit = -HAR_REFIT_EVERY
hmm_model = None
hmm_last_fit = -HMM_REFIT_EVERY
regime_multipliers = {0: 0.7, 1: 1.0, 2: 1.5}  # initial guess, will be calibrated

for eval_num, idx in enumerate(eval_indices):

    # Check we have enough history for HAR monthly component
    if idx < HAR_TRAIN_MIN:
        skip_count += 1
        continue

    # ---------- HAR-RV FIT (periodic) ----------
    if eval_num - har_last_fit >= HAR_REFIT_EVERY or har_model is None:
        try:
            # Training data: everything up to current eval point
            train_slice = df.loc[:idx - 1]
            # Need valid HAR features
            valid_mask = train_slice[["rv_daily", "rv_weekly", "rv_monthly", "future_atr"]].notna().all(axis=1)
            train_valid = train_slice[valid_mask]

            if len(train_valid) > 500:
                # Include ATR_14 as 4th feature — lets HAR learn optimal naive blend
                X_train = train_valid[["rv_daily", "rv_weekly", "rv_monthly", "ATR_14"]].values
                # Target: future realized volatility (use future_atr as proxy)
                y_train = train_valid["future_atr"].values

                har_model = LinearRegression()
                har_model.fit(X_train, y_train)
                har_last_fit = eval_num
                har_fit_count += 1

                if har_fit_count == 1:
                    print(f"  [DEBUG] HAR coefficients: "
                          f"daily={har_model.coef_[0]:.4f}, "
                          f"weekly={har_model.coef_[1]:.4f}, "
                          f"monthly={har_model.coef_[2]:.4f}, "
                          f"ATR_14={har_model.coef_[3]:.4f}, "
                          f"intercept={har_model.intercept_:.4f}")
        except Exception as e:
            if har_fit_count == 0:
                print(f"  [DEBUG] First HAR fit failed: {e}")
            if har_model is None:
                skip_count += 1
                continue

    # ---------- HMM FIT (periodic) ----------
    if eval_num - hmm_last_fit >= HMM_REFIT_EVERY or hmm_model is None:
        try:
            # Use daily returns and daily RV for regime detection
            hmm_slice = df.loc[max(0, idx - 5000):idx - 1]
            hmm_returns = hmm_slice["returns_pct"].values
            hmm_rv = hmm_slice["rv_daily"].values

            # Remove NaN/inf
            valid = np.isfinite(hmm_returns) & np.isfinite(hmm_rv)
            hmm_X = np.column_stack([hmm_returns[valid], hmm_rv[valid]])

            if len(hmm_X) > 500:
                hmm_model = GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                hmm_model.fit(hmm_X)
                hmm_last_fit = eval_num
                hmm_fit_count += 1

                # Calibrate regime multipliers from training data
                # Decode states for training period
                states = hmm_model.predict(hmm_X)
                state_vol = {}
                for s in range(HMM_N_STATES):
                    s_mask = states == s
                    if s_mask.sum() > 10:
                        state_vol[s] = np.mean(np.abs(hmm_returns[valid][s_mask]))
                    else:
                        state_vol[s] = np.mean(np.abs(hmm_returns[valid]))

                # Sort states by volatility level
                sorted_states = sorted(state_vol.keys(), key=lambda s: state_vol[s])
                overall_vol = np.mean(np.abs(hmm_returns[valid]))

                for rank, s in enumerate(sorted_states):
                    regime_multipliers[s] = state_vol[s] / overall_vol if overall_vol > 0 else 1.0
                    # Clamp to reasonable range
                    regime_multipliers[s] = np.clip(regime_multipliers[s], 0.5, 2.5)

                if hmm_fit_count == 1:
                    print(f"  [DEBUG] HMM regime multipliers: {regime_multipliers}")

        except Exception as e:
            if hmm_fit_count == 0:
                print(f"  [DEBUG] First HMM fit failed: {e}")
            # HMM failure is non-fatal — fall back to multiplier=1.0

    if har_model is None:
        skip_count += 1
        continue

    # ---------- HAR-RV FORECAST ----------
    naive_atr = df["ATR_14"].values[idx]
    try:
        har_features = np.array([[
            df["rv_daily"].values[idx],
            df["rv_weekly"].values[idx],
            df["rv_monthly"].values[idx],
            naive_atr,
        ]])
        if not np.all(np.isfinite(har_features)):
            skip_count += 1
            continue
        har_base = har_model.predict(har_features)[0]
        har_base = max(0.01, har_base)  # floor at 0.01 to avoid negative
    except Exception:
        skip_count += 1
        continue

    # ---------- DIURNAL MULTIPLIER (dampened) ----------
    hour = int(df["hour"].values[idx])
    raw_diurnal = diurnal_profile.get(hour, 1.0)
    # Shrink toward 1.0: avoid over-adjusting in calm sessions
    diurnal_mult = 1.0 + DIURNAL_STRENGTH * (raw_diurnal - 1.0)

    # ---------- CALENDAR EVENT MULTIPLIER ----------
    cal_mult = df["event_mult"].values[idx]

    # ---------- HMM REGIME MULTIPLIER ----------
    regime_mult = 1.0
    if hmm_model is not None:
        try:
            curr_ret = df["returns_pct"].values[idx]
            curr_rv = df["rv_daily"].values[idx]
            if np.isfinite(curr_ret) and np.isfinite(curr_rv):
                obs = np.array([[curr_ret, curr_rv]])
                state = hmm_model.predict(obs)[0]
                regime_mult = regime_multipliers.get(state, 1.0)
        except Exception:
            regime_mult = 1.0

    # ---------- COMBINED FORECAST ----------
    # HAR-adjusted: base * dampened multipliers
    har_adjusted = har_base * diurnal_mult * cal_mult * regime_mult

    # Blend with naive (Bates & Granger 1969 — forecast combination)
    # blend_w calibrated from training data; preserves naive's calm-period
    # accuracy while capturing HAR's event/regime advantage
    forecast_atr = blend_w * har_adjusted + (1 - blend_w) * naive_atr

    # Sanity clamp: forecast should be between 0.2x and 5x naive ATR
    forecast_atr = np.clip(forecast_atr, 0.2 * naive_atr, 5.0 * naive_atr)

    if not np.isfinite(forecast_atr) or forecast_atr <= 0:
        skip_count += 1
        continue

    # ---------- ACTUAL ----------
    actual_atr = df["future_atr"].values[idx]
    if not (np.isfinite(naive_atr) and np.isfinite(actual_atr)):
        skip_count += 1
        continue

    # ---------- TCP CONFORMAL UPDATE ----------
    interval_lower = forecast_atr * (1 - tcp_width)
    interval_upper = forecast_atr * (1 + tcp_width)
    covered = interval_lower <= actual_atr <= interval_upper

    # Robbins-Monro: BIG expansion on miss, small shrink on hit
    if covered:
        tcp_width -= TCP_GAMMA * TCP_ALPHA           # small shrink
    else:
        tcp_width += TCP_GAMMA * (1 - TCP_ALPHA)     # big expansion
    tcp_width = max(0.05, min(3.0, tcp_width))

    # ---------- METRICS ----------
    e_err = abs(forecast_atr - actual_atr)
    n_err = abs(naive_atr - actual_atr)

    vol_change = (actual_atr - naive_atr) / naive_atr * 100 if naive_atr > 0 else 0
    e_dir = np.sign(forecast_atr - naive_atr)
    a_dir = np.sign(actual_atr - naive_atr)
    dir_correct = bool(e_dir == a_dir) if a_dir != 0 else False

    results.append({
        "timestamp": str(df["timestamp"].values[idx]),
        "actual_atr": actual_atr,
        "forecast_atr": forecast_atr,
        "naive_atr": naive_atr,
        "har_base": har_base,
        "diurnal_mult": diurnal_mult,
        "cal_mult": cal_mult,
        "regime_mult": regime_mult,
        "e_err": e_err,
        "n_err": n_err,
        "e_wins": e_err < n_err,
        "dir_ok": dir_correct,
        "vol_chg_pct": vol_change,
        "tcp_width": tcp_width,
        "covered": covered,
    })

    # Debug first eval
    if len(results) == 1:
        r = results[0]
        print(f"  [DEBUG] First forecast: HAR_base={r['har_base']:.4f}, "
              f"diurnal={r['diurnal_mult']:.3f}, cal={r['cal_mult']:.3f}, "
              f"regime={r['regime_mult']:.3f}")
        print(f"  [DEBUG]   forecast={r['forecast_atr']:.4f}, "
              f"actual={r['actual_atr']:.4f}, naive={r['naive_atr']:.4f}")

    if (eval_num + 1) % 100 == 0:
        elapsed = time.time() - start_time
        cur_mae = np.mean([r["e_err"] for r in results])
        n_mae_cur = np.mean([r["n_err"] for r in results])
        cur_wr = np.mean([r["e_wins"] for r in results]) * 100
        cur_cov = np.mean([r["covered"] for r in results]) * 100
        print(f"  [{eval_num+1}/{n_evals}] {elapsed:.0f}s | "
              f"MAE: HAR={cur_mae:.4f} Naive={n_mae_cur:.4f} | "
              f"WR={cur_wr:.1f}% | Cov={cur_cov:.1f}% | "
              f"HAR fits={har_fit_count}, HMM fits={hmm_fit_count}")

total_time = time.time() - start_time

# ==========================================================================
# 6. RESULTS
# ==========================================================================
rdf = pd.DataFrame(results)
n = len(rdf)

print(f"\n  Completed: {n} eval points | Skipped: {skip_count}")
print(f"  HAR refits: {har_fit_count} | HMM refits: {hmm_fit_count}")

if n < 10:
    print(f"\n  ERROR: Only {n} evaluation points succeeded.")
    print(f"  Check: test_start_idx={test_start_idx}, HAR_TRAIN_MIN={HAR_TRAIN_MIN}")
    sys.exit(1)

# Core metrics
e_mae = rdf["e_err"].mean()
n_mae = rdf["n_err"].mean()
imp = (1 - e_mae / n_mae) * 100 if n_mae > 0 else 0

e_wr = rdf["e_wins"].mean() * 100
e_dir_acc = rdf["dir_ok"].mean() * 100
e_corr = np.corrcoef(rdf["actual_atr"], rdf["forecast_atr"])[0, 1]
n_corr = np.corrcoef(rdf["actual_atr"], rdf["naive_atr"])[0, 1]

# Component contribution analysis
diurnal_range = rdf["diurnal_mult"].max() - rdf["diurnal_mult"].min()
cal_active = (rdf["cal_mult"] > 1.0).mean() * 100
regime_std = rdf["regime_mult"].std()

# TCP calibration
tcp_coverage = rdf["covered"].mean() * 100
tcp_avg_width = rdf["tcp_width"].mean()
tcp_final_width = rdf["tcp_width"].iloc[-1]

# High-vol regime
vol_mask = abs(rdf["vol_chg_pct"]) > 20
vol_n = vol_mask.sum()

# Event day performance
event_mask = rdf["cal_mult"] > 1.0
event_n = event_mask.sum()

print("\n" + "=" * 70)
print("  HAR-RV + DIURNAL + CALENDAR + HMM — FINAL RESULTS")
print("=" * 70)
print(f"\n  Evaluations: {n} | Time: {total_time:.1f}s")
print(f"  Model: HAR-RV (YZ estimator) + diurnal + calendar + HMM({HMM_N_STATES})")
print(f"  Horizon: {PRED_HORIZON} bars ({PRED_HORIZON*15}min) | TCP alpha: {TCP_ALPHA}")

print(f"\n  {'METRIC':<40} {'HAR-RV':>10} {'NAIVE':>10} {'WINNER':>10}")
print(f"  {'-'*70}")
print(f"  {'MAE (lower=better)':<40} {e_mae:>10.4f} {n_mae:>10.4f} {'HAR-RV' if e_mae < n_mae else 'NAIVE':>10}")
print(f"  {'Correlation with actual':<40} {e_corr:>10.4f} {n_corr:>10.4f} {'HAR-RV' if e_corr > n_corr else 'NAIVE':>10}")
print(f"  {'Head-to-head win rate':<40} {e_wr:>9.1f}% {100-e_wr:>9.1f}%  {'HAR-RV' if e_wr > 50 else 'NAIVE':>10}")
print(f"  {'Vol direction accuracy':<40} {e_dir_acc:>9.1f}% {'50.0':>9}%  {'HAR-RV' if e_dir_acc > 50 else 'COIN':>10}")
print(f"  {'MAE improvement vs naive':<40} {imp:>9.1f}%")

print(f"\n  --- Component Activity ---")
print(f"  {'Blend weight (HAR/naive)':<40} {blend_w:.2f} / {1-blend_w:.2f}")
print(f"  {'Diurnal strength':<40} {DIURNAL_STRENGTH:.2f} (range: {diurnal_range:.3f})")
print(f"  {'Bars near events':<40} {cal_active:>9.1f}%")
print(f"  {'Regime multiplier std':<40} {regime_std:>10.3f}")
print(f"  {'HAR coefficients':<40} d={har_model.coef_[0]:.1f} w={har_model.coef_[1]:.1f} m={har_model.coef_[2]:.1f} atr={har_model.coef_[3]:.3f}")

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
    print(f"  {'MAE':<40} {vk:>10.4f} {vn:>10.4f} {'HAR-RV' if vk < vn else 'NAIVE':>10}")
    print(f"  {'Win rate':<40} {vwr:>9.1f}% {100-vwr:>9.1f}%  {'HAR-RV' if vwr > 50 else 'NAIVE':>10}")
    print(f"  {'TCP coverage during vol spikes':<40} {vcov:>9.1f}%")

if event_n > 0:
    ek = rdf.loc[event_mask, "e_err"].mean()
    en = rdf.loc[event_mask, "n_err"].mean()
    ewr = (rdf.loc[event_mask, "e_err"] < rdf.loc[event_mask, "n_err"]).mean() * 100
    print(f"\n  --- Event Day Performance (n={event_n}) ---")
    print(f"  {'MAE':<40} {ek:>10.4f} {en:>10.4f} {'HAR-RV' if ek < en else 'NAIVE':>10}")
    print(f"  {'Win rate':<40} {ewr:>9.1f}% {100-ewr:>9.1f}%  {'HAR-RV' if ewr > 50 else 'NAIVE':>10}")

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
    print("\n  >>> VERDICT: INTEGRATE HAR-RV + multipliers for adaptive SL/TP <<<")
elif passes >= 3:
    print("\n  >>> VERDICT: PROMISING — upgrade to LightGBM meta-learner (Rank 2) <<<")
elif passes >= 2:
    print("\n  >>> VERDICT: MARGINAL — try LightGBM with full feature stack <<<")
else:
    print("\n  >>> VERDICT: NEEDS REWORK <<<")

print("=" * 70)

rdf.to_csv("har_rv_poc_results.csv", index=False)
print(f"\nResults saved: har_rv_poc_results.csv")

print(f"\n{'='*70}")
print("  COMPARISON: ALL POC ATTEMPTS")
print(f"{'='*70}")
print(f"  Kronos (TSFM):          1/4  FAIL — domain shift, GPU required")
print(f"  EGARCH v1 (simulation): 1/5  FAIL — scale explosion")
print(f"  EGARCH v2 (adjusted):   1/5  FAIL — GARCH is a filter, not forecaster")
print(f"  HAR-RV + multipliers:   {passes}/5  {'PASS' if passes >= 3 else 'FAIL'}")
print(f"  Time: {total_time:.1f}s (CPU only, no GPU needed)")
print(f"  Dependencies: sklearn, hmmlearn (lightweight)")
print(f"{'='*70}")
