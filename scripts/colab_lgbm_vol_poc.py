"""
LightGBM Meta-Learner Volatility POC v1 (Rank 2)
==================================================
Run in Google Colab (no GPU needed — CPU is sufficient)

Single-cell execution:
  !git clone https://github.com/LKBSM/TradingBot_Agentic.git
  %run TradingBot_Agentic/scripts/colab_lgbm_vol_poc.py

Rank 2 volatility forecaster: LightGBM meta-learner with 21 features.
Builds on the HAR-RV POC (Rank 1, scored 4/5) by adding:
  1. HAR-RV features as inputs (rv_daily, rv_weekly, rv_monthly) — Corsi 2009
  2. ATR variants (ATR_14, ATR_7, rate of change at 5 and 20 bars)
  3. Returns features (abs_return_1, abs_return_5, rolling_std_20)
  4. Session dummies (Asian/London/NY overlap/NY afternoon/after-hours)
  5. Calendar event proximity (hours to nearest high-impact USD event)
  6. HMM regime state + multiplier (3-state: low/normal/high vol)
  7. Technical indicators (RSI_14, Bollinger %B, MACD histogram sign)
  8. TCP conformal prediction — calibrated prediction intervals

Expected improvement: 20-35% MAE reduction vs naive ATR (vs 10-15% for HAR-RV alone).
LightGBM captures nonlinear interactions that linear HAR cannot.

Architecture: LightGBM(21 features) -> future_atr prediction
  - HAR-RV fitted first as sub-component (provides rv features + HMM regime)
  - Walk-forward: train on expanding window, evaluate on out-of-sample
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
TCP_ALPHA        = 0.05         # target miscoverage (95% intervals)
TCP_GAMMA        = 0.05         # TCP learning rate (Robbins-Monro)

# HAR lookback windows (in M15 bars)
HAR_DAILY   = 96    # 1 day  = 96 M15 bars
HAR_WEEKLY  = 480   # 5 days = 480 M15 bars
HAR_MONTHLY = 2112  # 22 days = 2112 M15 bars

# LightGBM hyperparameters
LGBM_N_ESTIMATORS    = 500
LGBM_MAX_DEPTH       = 6
LGBM_LEARNING_RATE   = 0.05
LGBM_MIN_CHILD       = 50
LGBM_SUBSAMPLE       = 0.8
LGBM_COLSAMPLE       = 0.8
LGBM_EARLY_STOPPING  = 50

# LightGBM refit frequency (in eval steps)
LGBM_REFIT_EVERY = 100  # refit every 100 eval steps (~2000 bars)
HMM_REFIT_EVERY  = 100  # refit HMM every 100 eval steps

# Calendar event detection
HIGH_IMPACT_EVENTS = [
    "Non-Farm Payrolls", "Federal Funds Rate", "CPI m/m", "Core CPI m/m",
    "FOMC Statement", "FOMC Press Conference", "GDP q/q",
    "Core PCE Price Index m/m", "Retail Sales m/m",
]
EVENT_WINDOW_HOURS = 4  # hours before/after event to apply multiplier

# Session hour ranges (UTC) — for XAUUSD / Gold
SESSION_HOURS = {
    "asian":        (0, 8),
    "london":       (8, 13),
    "ny_overlap":   (13, 17),
    "ny_afternoon": (17, 21),
    "after_hours":  (21, 24),
}

# Feature names (must match LGBMVolForecaster.FEATURE_NAMES exactly)
FEATURE_NAMES = [
    # HAR-RV
    "rv_daily", "rv_weekly", "rv_monthly",
    # ATR features
    "atr_14", "atr_7", "atr_change_5", "atr_change_20",
    # Returns
    "abs_return_1", "abs_return_5", "rolling_std_20",
    # Session dummies (one-hot)
    "session_asian", "session_london", "session_ny_overlap",
    "session_ny_afternoon", "session_after_hours",
    # Calendar
    "event_proximity_hours",
    # HMM regime
    "regime_state_ord", "regime_multiplier",
    # Technical
    "rsi_14", "bb_pct", "macd_hist_sign",
]

DATA_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/XAU_15MIN_2019_2025.csv"
DATA_FILE = "XAU_15MIN_2019_2025.csv"
CAL_URL = "https://github.com/LKBSM/TradingBot_Agentic/releases/download/v1.0-data/economic_calendar_2019_2025.csv"
CAL_FILE = "economic_calendar_2019_2025.csv"

# ==========================================================================
# 1. SETUP
# ==========================================================================
print("=" * 70)
print("  LIGHTGBM META-LEARNER VOL POC (RANK 2) — Smart Sentinel AI")
print("=" * 70)

print("\n[1/7] Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "lightgbm", "hmmlearn", "scikit-learn"], check=True)
print("[OK] Dependencies installed")

import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from hmmlearn.hmm import GaussianHMM

# ==========================================================================
# 2. DATA
# ==========================================================================
print("\n[2/7] Loading data...")

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
# 3. FEATURE ENGINEERING (21 features)
# ==========================================================================
print("\n[3/7] Computing 21 features...")

# --- Log returns (percentage) ---
df["returns_pct"] = np.log(df["close"] / df["close"].shift(1)) * 100

# --- True Range and ATR variants ---
df["tr"] = np.maximum(
    df["high"] - df["low"],
    np.maximum(
        abs(df["high"] - df["close"].shift(1)),
        abs(df["low"] - df["close"].shift(1))
    )
)
df["atr_14"] = df["tr"].rolling(14).mean()
df["atr_7"]  = df["tr"].rolling(7).mean()
df["atr_change_5"]  = df["atr_14"].pct_change(5)
df["atr_change_20"] = df["atr_14"].pct_change(20)

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
# Use Rogers-Satchell per-bar squared vol as the base RV measure
df["rv_bar"]     = rs_var.clip(lower=0)  # per-bar RV (non-negative)
df["rv_daily"]   = df["rv_bar"].rolling(HAR_DAILY).mean()
df["rv_weekly"]  = df["rv_bar"].rolling(HAR_WEEKLY).mean()
df["rv_monthly"] = df["rv_bar"].rolling(HAR_MONTHLY).mean()

# --- Returns features ---
df["abs_return_1"]   = df["returns_pct"].abs()
df["abs_return_5"]   = df["returns_pct"].abs().rolling(5).mean()
df["rolling_std_20"] = df["returns_pct"].rolling(20).std()

# --- Session dummies ---
df["hour"] = df["timestamp"].dt.hour
for session_name, (start_h, end_h) in SESSION_HOURS.items():
    col = f"session_{session_name}"
    if end_h > start_h:
        df[col] = ((df["hour"] >= start_h) & (df["hour"] < end_h)).astype(float)
    else:
        # Wraps around midnight
        df[col] = ((df["hour"] >= start_h) | (df["hour"] < end_h)).astype(float)

# --- Calendar event proximity (hours to nearest event) ---
def compute_event_proximity_hours(timestamps, event_times, window_hours=4):
    """Compute hours to nearest high-impact event (capped at window_hours)."""
    result = np.full(len(timestamps), float(window_hours))
    window_ns = np.timedelta64(window_hours, 'h')

    ts_values = timestamps.values
    for evt in event_times:
        delta = np.abs(ts_values - evt)
        mask = delta <= window_ns
        if not mask.any():
            continue
        hours_away = delta[mask].astype('timedelta64[m]').astype(float) / 60.0
        result[mask] = np.minimum(result[mask], hours_away)
    return result

print("  Computing event proximity (this may take a moment)...")
df["event_proximity_hours"] = compute_event_proximity_hours(
    df["timestamp"], event_times, EVENT_WINDOW_HOURS
)
event_bars = (df["event_proximity_hours"] < EVENT_WINDOW_HOURS).sum()
print(f"  Bars near high-impact events: {event_bars:,} ({event_bars/len(df)*100:.1f}%)")

# --- Technical indicators ---
# RSI 14
delta = df["close"].diff()
gain = delta.where(delta > 0, 0.0).rolling(14).mean()
loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

# Bollinger %B (20-period)
sma_20 = df["close"].rolling(20).mean()
std_20 = df["close"].rolling(20).std()
bb_upper = sma_20 + 2 * std_20
bb_lower = sma_20 - 2 * std_20
bb_width = bb_upper - bb_lower
df["bb_pct"] = ((df["close"] - bb_lower) / bb_width.replace(0, np.nan)).fillna(0.5)

# MACD histogram sign
ema12 = df["close"].ewm(span=12, adjust=False).mean()
ema26 = df["close"].ewm(span=26, adjust=False).mean()
macd_line = ema12 - ema26
macd_signal = macd_line.ewm(span=9, adjust=False).mean()
macd_hist = macd_line - macd_signal
df["macd_hist_sign"] = np.sign(macd_hist)

# --- Future realized ATR: mean TR over next PRED_HORIZON bars ---
df["future_atr"] = df["tr"].rolling(PRED_HORIZON).mean().shift(-PRED_HORIZON)

# HMM regime columns will be added during walk-forward (they need fitting)
df["regime_state_ord"] = 1.0   # placeholder — normal
df["regime_multiplier"] = 1.0  # placeholder — neutral

# --- Drop NaN rows, reset index ---
required_cols = [
    "atr_14", "atr_7", "future_atr", "rv_daily", "rv_weekly", "rv_monthly",
    "yz_rv", "returns_pct", "atr_change_5", "atr_change_20",
    "abs_return_1", "abs_return_5", "rolling_std_20",
    "rsi_14", "bb_pct",
]
df.dropna(subset=required_cols, inplace=True)
df.reset_index(drop=True, inplace=True)

# --- Split ---
test_start_idx = df.index[df["timestamp"] >= TEST_START].min()
test_end_idx = df.index.max()

print(f"  Total bars after cleanup: {len(df):,}")
print(f"  Test range: idx {test_start_idx} to {test_end_idx} ({test_end_idx - test_start_idx:,} bars)")
print(f"  Features: {len(FEATURE_NAMES)}")
for i, f in enumerate(FEATURE_NAMES):
    sample_val = df[f].iloc[test_start_idx] if f in df.columns else "N/A"
    if isinstance(sample_val, float):
        print(f"    {i+1:2d}. {f:<25s} sample={sample_val:.4f}")
    else:
        print(f"    {i+1:2d}. {f:<25s} sample={sample_val}")

# ==========================================================================
# 4. FIT INITIAL HAR-RV (sub-component for HMM regime features)
# ==========================================================================
print("\n[4/7] Fitting initial HAR-RV + HMM on training data...")

train_df = df.loc[:test_start_idx - 1].copy()

# --- Fit HAR-RV on training data ---
har_features = ["rv_daily", "rv_weekly", "rv_monthly", "atr_14"]
har_valid = train_df[har_features + ["future_atr"]].notna().all(axis=1)
har_train = train_df[har_valid]

har_model = None
if len(har_train) > 500:
    X_har = har_train[har_features].values
    y_har = har_train["future_atr"].values
    har_model = LinearRegression()
    har_model.fit(X_har, y_har)
    print(f"  HAR-RV fitted on {len(har_train):,} bars")
    print(f"  HAR coefficients: daily={har_model.coef_[0]:.4f}, "
          f"weekly={har_model.coef_[1]:.4f}, monthly={har_model.coef_[2]:.4f}, "
          f"ATR_14={har_model.coef_[3]:.4f}, intercept={har_model.intercept_:.4f}")
else:
    print(f"  WARNING: Insufficient HAR training data ({len(har_train)} bars)")

# --- Fit HMM on training data for regime detection ---
hmm_model = None
regime_multipliers = {0: 0.7, 1: 1.0, 2: 1.5}
regime_labels = {0: "low", 1: "normal", 2: "high"}

hmm_valid = (
    train_df["returns_pct"].notna() &
    train_df["rv_daily"].notna() &
    np.isfinite(train_df["returns_pct"]) &
    np.isfinite(train_df["rv_daily"])
)
hmm_data = train_df[hmm_valid]

if len(hmm_data) > 500:
    hmm_X = np.column_stack([
        hmm_data["returns_pct"].values,
        hmm_data["rv_daily"].values,
    ])
    hmm_model = GaussianHMM(
        n_components=HMM_N_STATES,
        covariance_type="diag",
        n_iter=100,
        random_state=42,
    )
    hmm_model.fit(hmm_X)

    # Calibrate regime multipliers
    states = hmm_model.predict(hmm_X)
    state_vol = {}
    for s in range(HMM_N_STATES):
        s_mask = states == s
        if s_mask.sum() > 10:
            state_vol[s] = np.mean(np.abs(hmm_data["returns_pct"].values[s_mask]))
        else:
            state_vol[s] = np.mean(np.abs(hmm_data["returns_pct"].values))

    sorted_states = sorted(state_vol.keys(), key=lambda s: state_vol[s])
    overall_vol = np.mean(np.abs(hmm_data["returns_pct"].values))

    for rank, s in enumerate(sorted_states):
        regime_multipliers[s] = np.clip(state_vol[s] / overall_vol if overall_vol > 0 else 1.0, 0.5, 2.5)
        regime_labels[s] = ["low", "normal", "high"][rank]

    print(f"  HMM fitted: {HMM_N_STATES} states")
    print(f"  Regime multipliers: {regime_multipliers}")
    print(f"  Regime labels: {regime_labels}")

    # Assign regime features to full dataframe using HMM
    def assign_regime_features(df_chunk, hmm_m, reg_mult, reg_labels):
        """Assign regime_state_ord and regime_multiplier to dataframe rows."""
        for i in range(len(df_chunk)):
            ret = df_chunk["returns_pct"].iloc[i]
            rv  = df_chunk["rv_daily"].iloc[i]
            if np.isfinite(ret) and np.isfinite(rv):
                try:
                    obs = np.array([[ret, rv]])
                    state = int(hmm_m.predict(obs)[0])
                    df_chunk.iloc[i, df_chunk.columns.get_loc("regime_multiplier")] = reg_mult.get(state, 1.0)
                    ordinal = {"low": 0.0, "normal": 1.0, "high": 2.0}.get(
                        reg_labels.get(state, "normal"), 1.0
                    )
                    df_chunk.iloc[i, df_chunk.columns.get_loc("regime_state_ord")] = ordinal
                except Exception:
                    pass
        return df_chunk

    # Assign to full df (vectorized where possible, fallback to loop for HMM)
    print("  Assigning HMM regime features to all bars...")
    valid_mask = (
        df["returns_pct"].notna() &
        df["rv_daily"].notna() &
        np.isfinite(df["returns_pct"]) &
        np.isfinite(df["rv_daily"])
    )
    valid_indices = df.index[valid_mask]

    if len(valid_indices) > 0:
        hmm_obs = np.column_stack([
            df.loc[valid_indices, "returns_pct"].values,
            df.loc[valid_indices, "rv_daily"].values,
        ])
        predicted_states = hmm_model.predict(hmm_obs)

        for i, idx in enumerate(valid_indices):
            state = int(predicted_states[i])
            df.at[idx, "regime_multiplier"] = regime_multipliers.get(state, 1.0)
            ordinal = {"low": 0.0, "normal": 1.0, "high": 2.0}.get(
                regime_labels.get(state, "normal"), 1.0
            )
            df.at[idx, "regime_state_ord"] = ordinal

    print(f"  Regime features assigned to {len(valid_indices):,} bars")
else:
    print(f"  WARNING: Insufficient HMM data ({len(hmm_data)} bars), using defaults")

# ==========================================================================
# 5. WALK-FORWARD EVALUATION
# ==========================================================================
print(f"\n[5/7] Walk-forward evaluation with LightGBM...")

eval_indices = list(range(test_start_idx, test_end_idx, STEP_SIZE))[:MAX_EVALS]
n_evals = len(eval_indices)
print(f"  Eval points: {n_evals}")

results = []
tcp_width = 0.5
start_time = time.time()
lgbm_fit_count = 0
hmm_refit_count = 0
skip_count = 0

lgbm_model = None
lgbm_last_fit = -LGBM_REFIT_EVERY
hmm_last_fit = -HMM_REFIT_EVERY

# Feature importance accumulator
importance_sums = {f: 0.0 for f in FEATURE_NAMES}
importance_count = 0

for eval_num, idx in enumerate(eval_indices):

    # Check we have enough history for HAR monthly component
    if idx < HAR_TRAIN_MIN:
        skip_count += 1
        continue

    # ---------- HMM REFIT (periodic) ----------
    if eval_num - hmm_last_fit >= HMM_REFIT_EVERY and hmm_model is not None:
        try:
            hmm_slice = df.loc[max(0, idx - 5000):idx - 1]
            hmm_returns = hmm_slice["returns_pct"].values
            hmm_rv = hmm_slice["rv_daily"].values
            valid = np.isfinite(hmm_returns) & np.isfinite(hmm_rv)
            hmm_X_refit = np.column_stack([hmm_returns[valid], hmm_rv[valid]])

            if len(hmm_X_refit) > 500:
                hmm_model = GaussianHMM(
                    n_components=HMM_N_STATES,
                    covariance_type="diag",
                    n_iter=100,
                    random_state=42,
                )
                hmm_model.fit(hmm_X_refit)

                # Re-calibrate regime multipliers
                states = hmm_model.predict(hmm_X_refit)
                state_vol = {}
                for s in range(HMM_N_STATES):
                    s_mask = states == s
                    if s_mask.sum() > 10:
                        state_vol[s] = np.mean(np.abs(hmm_returns[valid][s_mask]))
                    else:
                        state_vol[s] = np.mean(np.abs(hmm_returns[valid]))

                sorted_states = sorted(state_vol.keys(), key=lambda s: state_vol[s])
                overall_vol = np.mean(np.abs(hmm_returns[valid]))
                for rank, s in enumerate(sorted_states):
                    regime_multipliers[s] = np.clip(
                        state_vol[s] / overall_vol if overall_vol > 0 else 1.0, 0.5, 2.5
                    )
                    regime_labels[s] = ["low", "normal", "high"][rank]

                hmm_last_fit = eval_num
                hmm_refit_count += 1

                # Update regime features for recent bars (idx-500:idx)
                update_start = max(0, idx - 500)
                update_slice = df.loc[update_start:idx]
                update_valid = (
                    update_slice["returns_pct"].notna() &
                    update_slice["rv_daily"].notna() &
                    np.isfinite(update_slice["returns_pct"]) &
                    np.isfinite(update_slice["rv_daily"])
                )
                update_indices = update_slice.index[update_valid]

                if len(update_indices) > 0:
                    hmm_obs_up = np.column_stack([
                        df.loc[update_indices, "returns_pct"].values,
                        df.loc[update_indices, "rv_daily"].values,
                    ])
                    pred_states = hmm_model.predict(hmm_obs_up)
                    for i, ui in enumerate(update_indices):
                        state = int(pred_states[i])
                        df.at[ui, "regime_multiplier"] = regime_multipliers.get(state, 1.0)
                        ordinal = {"low": 0.0, "normal": 1.0, "high": 2.0}.get(
                            regime_labels.get(state, "normal"), 1.0
                        )
                        df.at[ui, "regime_state_ord"] = ordinal
        except Exception as e:
            if hmm_refit_count == 0:
                print(f"  [DEBUG] HMM refit failed: {e}")

    # ---------- LIGHTGBM FIT (periodic, expanding window) ----------
    if eval_num - lgbm_last_fit >= LGBM_REFIT_EVERY or lgbm_model is None:
        try:
            # Training data: everything up to current eval point
            train_slice = df.loc[:idx - 1]

            # Prepare feature matrix
            feature_valid_mask = (
                train_slice[FEATURE_NAMES + ["future_atr"]].notna().all(axis=1) &
                train_slice[FEATURE_NAMES + ["future_atr"]].apply(
                    lambda row: np.all(np.isfinite(row)), axis=1
                )
            )
            train_valid = train_slice[feature_valid_mask]

            if len(train_valid) > 500:
                X_train_all = train_valid[FEATURE_NAMES].values
                y_train_all = train_valid["future_atr"].values

                # 85/15 train/val split for early stopping
                split_idx = int(len(X_train_all) * 0.85)
                X_tr, X_vl = X_train_all[:split_idx], X_train_all[split_idx:]
                y_tr, y_vl = y_train_all[:split_idx], y_train_all[split_idx:]

                train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES)
                val_data = lgb.Dataset(X_vl, label=y_vl, feature_name=FEATURE_NAMES,
                                       reference=train_data)

                params = {
                    "objective": "regression",
                    "metric": "mae",
                    "max_depth": LGBM_MAX_DEPTH,
                    "learning_rate": LGBM_LEARNING_RATE,
                    "min_child_samples": LGBM_MIN_CHILD,
                    "subsample": LGBM_SUBSAMPLE,
                    "colsample_bytree": LGBM_COLSAMPLE,
                    "verbosity": -1,
                    "seed": 42,
                    "num_threads": -1,
                }

                callbacks = [lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False)]

                lgbm_model = lgb.train(
                    params,
                    train_data,
                    num_boost_round=LGBM_N_ESTIMATORS,
                    valid_sets=[val_data],
                    callbacks=callbacks,
                )

                lgbm_last_fit = eval_num
                lgbm_fit_count += 1

                # Track feature importance
                importance = lgbm_model.feature_importance(importance_type="gain")
                total_imp = importance.sum()
                if total_imp > 0:
                    for fi, fname in enumerate(FEATURE_NAMES):
                        importance_sums[fname] += importance[fi] / total_imp
                    importance_count += 1

                if lgbm_fit_count == 1:
                    # Print initial model info
                    best_iter = lgbm_model.best_iteration
                    val_pred = lgbm_model.predict(X_vl)
                    val_mae = np.mean(np.abs(val_pred - y_vl))
                    naive_mae_v = np.mean(np.abs(X_vl[:, FEATURE_NAMES.index("atr_14")] - y_vl))
                    imp_pct = (1 - val_mae / naive_mae_v) * 100 if naive_mae_v > 0 else 0
                    print(f"  [DEBUG] First LightGBM fit: {len(train_valid):,} bars, "
                          f"best_iter={best_iter}, "
                          f"val_MAE={val_mae:.4f}, naive_MAE={naive_mae_v:.4f}, "
                          f"improvement={imp_pct:.1f}%")

                    # Top-5 features
                    feat_imp = sorted(zip(FEATURE_NAMES, importance / total_imp),
                                      key=lambda x: -x[1])[:5]
                    print(f"  [DEBUG] Top-5 features: "
                          + ", ".join(f"{n}={v:.3f}" for n, v in feat_imp))

        except Exception as e:
            if lgbm_fit_count == 0:
                print(f"  [DEBUG] First LightGBM fit failed: {e}")
                import traceback
                traceback.print_exc()
            if lgbm_model is None:
                skip_count += 1
                continue

    if lgbm_model is None:
        skip_count += 1
        continue

    # ---------- LIGHTGBM FORECAST ----------
    naive_atr = df["atr_14"].values[idx]
    try:
        feature_vec = np.array([[df[f].values[idx] for f in FEATURE_NAMES]])
        if not np.all(np.isfinite(feature_vec)):
            skip_count += 1
            continue
        forecast_atr = float(lgbm_model.predict(feature_vec)[0])
        forecast_atr = max(0.01, forecast_atr)
    except Exception:
        skip_count += 1
        continue

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

    # ---------- REGIME INFO ----------
    regime_mult = df["regime_multiplier"].values[idx]
    regime_ord  = df["regime_state_ord"].values[idx]
    regime_label_map = {0.0: "low", 1.0: "normal", 2.0: "high"}
    regime_label = regime_label_map.get(regime_ord, "unknown")

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
        "regime_label": regime_label,
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
        print(f"  [DEBUG] First forecast: LGBM={r['forecast_atr']:.4f}, "
              f"actual={r['actual_atr']:.4f}, naive={r['naive_atr']:.4f}, "
              f"regime={r['regime_label']}")

    if (eval_num + 1) % 100 == 0:
        elapsed = time.time() - start_time
        cur_mae = np.mean([r["e_err"] for r in results])
        n_mae_cur = np.mean([r["n_err"] for r in results])
        cur_wr = np.mean([r["e_wins"] for r in results]) * 100
        cur_cov = np.mean([r["covered"] for r in results]) * 100
        print(f"  [{eval_num+1}/{n_evals}] {elapsed:.0f}s | "
              f"MAE: LGBM={cur_mae:.4f} Naive={n_mae_cur:.4f} | "
              f"WR={cur_wr:.1f}% | Cov={cur_cov:.1f}% | "
              f"LGBM fits={lgbm_fit_count}, HMM refits={hmm_refit_count}")

total_time = time.time() - start_time

# ==========================================================================
# 6. RESULTS
# ==========================================================================
rdf = pd.DataFrame(results)
n = len(rdf)

print(f"\n  Completed: {n} eval points | Skipped: {skip_count}")
print(f"  LightGBM refits: {lgbm_fit_count} | HMM refits: {hmm_refit_count}")

if n < 10:
    print(f"\n  ERROR: Only {n} evaluation points succeeded.")
    print(f"  Check: test_start_idx={test_start_idx}, HAR_TRAIN_MIN={HAR_TRAIN_MIN}")
    sys.exit(1)

# Core metrics
e_mae = rdf["e_err"].mean()
n_mae = rdf["n_err"].mean()
e_rmse = np.sqrt((rdf["e_err"] ** 2).mean())
n_rmse = np.sqrt((rdf["n_err"] ** 2).mean())
mae_imp = (1 - e_mae / n_mae) * 100 if n_mae > 0 else 0
rmse_imp = (1 - e_rmse / n_rmse) * 100 if n_rmse > 0 else 0

e_wr = rdf["e_wins"].mean() * 100
e_dir_acc = rdf["dir_ok"].mean() * 100
e_corr = np.corrcoef(rdf["actual_atr"], rdf["forecast_atr"])[0, 1]
n_corr = np.corrcoef(rdf["actual_atr"], rdf["naive_atr"])[0, 1]

# TCP calibration
tcp_coverage = rdf["covered"].mean() * 100
tcp_avg_width = rdf["tcp_width"].mean()
tcp_final_width = rdf["tcp_width"].iloc[-1]

# High-vol regime
vol_mask = abs(rdf["vol_chg_pct"]) > 20
vol_n = vol_mask.sum()

# Event proximity performance (bars near events)
event_mask = rdf.apply(
    lambda r: any(
        abs(pd.Timestamp(r["timestamp"]) - pd.Timestamp(evt)) <= pd.Timedelta(hours=EVENT_WINDOW_HOURS)
        for evt in event_times[-200:]  # check last 200 events for speed
    ), axis=1
) if len(event_times) > 0 else pd.Series(False, index=rdf.index)
event_n = event_mask.sum()

# Per-regime breakdown
regime_breakdown = {}
for regime_name in ["low", "normal", "high"]:
    regime_mask = rdf["regime_label"] == regime_name
    regime_n = regime_mask.sum()
    if regime_n > 5:
        r_mae = rdf.loc[regime_mask, "e_err"].mean()
        r_n_mae = rdf.loc[regime_mask, "n_err"].mean()
        r_wr = (rdf.loc[regime_mask, "e_err"] < rdf.loc[regime_mask, "n_err"]).mean() * 100
        r_imp = (1 - r_mae / r_n_mae) * 100 if r_n_mae > 0 else 0
        regime_breakdown[regime_name] = {
            "n": regime_n, "mae": r_mae, "n_mae": r_n_mae,
            "wr": r_wr, "imp": r_imp,
        }

print("\n" + "=" * 70)
print("  LIGHTGBM META-LEARNER (RANK 2) — FINAL RESULTS")
print("=" * 70)
print(f"\n  Evaluations: {n} | Time: {total_time:.1f}s")
print(f"  Model: LightGBM (21 features, depth={LGBM_MAX_DEPTH}, lr={LGBM_LEARNING_RATE})")
print(f"  Horizon: {PRED_HORIZON} bars ({PRED_HORIZON*15}min) | TCP alpha: {TCP_ALPHA}")

print(f"\n  {'METRIC':<40} {'LGBM':>10} {'NAIVE':>10} {'WINNER':>10}")
print(f"  {'-'*70}")
print(f"  {'MAE (lower=better)':<40} {e_mae:>10.4f} {n_mae:>10.4f} {'LGBM' if e_mae < n_mae else 'NAIVE':>10}")
print(f"  {'RMSE (lower=better)':<40} {e_rmse:>10.4f} {n_rmse:>10.4f} {'LGBM' if e_rmse < n_rmse else 'NAIVE':>10}")
print(f"  {'Correlation with actual':<40} {e_corr:>10.4f} {n_corr:>10.4f} {'LGBM' if e_corr > n_corr else 'NAIVE':>10}")
print(f"  {'Head-to-head win rate':<40} {e_wr:>9.1f}% {100-e_wr:>9.1f}%  {'LGBM' if e_wr > 50 else 'NAIVE':>10}")
print(f"  {'Vol direction accuracy':<40} {e_dir_acc:>9.1f}% {'50.0':>9}%  {'LGBM' if e_dir_acc > 50 else 'COIN':>10}")
print(f"  {'MAE improvement vs naive':<40} {mae_imp:>9.1f}%")
print(f"  {'RMSE improvement vs naive':<40} {rmse_imp:>9.1f}%")

# --- Feature importance (top 10) ---
print(f"\n  --- Feature Importance (top 10, averaged across {importance_count} fits) ---")
if importance_count > 0:
    avg_importance = {f: importance_sums[f] / importance_count for f in FEATURE_NAMES}
    sorted_importance = sorted(avg_importance.items(), key=lambda x: -x[1])
    for rank, (fname, imp_val) in enumerate(sorted_importance[:10], 1):
        bar = "#" * int(imp_val * 50)
        print(f"    {rank:2d}. {fname:<25s} {imp_val:.4f} {bar}")

# --- TCP conformal prediction ---
calibration_ok = abs(tcp_coverage - (1 - TCP_ALPHA) * 100) < 5

print(f"\n  --- TCP Conformal Prediction (target: {(1-TCP_ALPHA)*100:.0f}% coverage) ---")
print(f"  {'Empirical coverage':<40} {tcp_coverage:>9.1f}%")
print(f"  {'Average interval width':<40} {tcp_avg_width:>10.3f}")
print(f"  {'Final interval width':<40} {tcp_final_width:>10.3f}")
print(f"  {'Calibration quality':<40} {'GOOD' if calibration_ok else 'NEEDS TUNING':>10}")

# --- Per-regime breakdown ---
if regime_breakdown:
    print(f"\n  --- Per-Regime Performance (HMM {HMM_N_STATES}-state) ---")
    print(f"  {'REGIME':<15} {'N':>6} {'MAE':>10} {'NAIVE':>10} {'WR':>8} {'IMP':>8}")
    print(f"  {'-'*57}")
    for regime_name in ["low", "normal", "high"]:
        if regime_name in regime_breakdown:
            rb = regime_breakdown[regime_name]
            print(f"  {regime_name:<15} {rb['n']:>6} {rb['mae']:>10.4f} {rb['n_mae']:>10.4f} "
                  f"{rb['wr']:>7.1f}% {rb['imp']:>7.1f}%")

# --- High volatility periods ---
if vol_n > 0:
    vk = rdf.loc[vol_mask, "e_err"].mean()
    vn = rdf.loc[vol_mask, "n_err"].mean()
    vwr = (rdf.loc[vol_mask, "e_err"] < rdf.loc[vol_mask, "n_err"]).mean() * 100
    vcov = rdf.loc[vol_mask, "covered"].mean() * 100
    vimp = (1 - vk / vn) * 100 if vn > 0 else 0
    print(f"\n  --- High Volatility Periods (ATR change >20%, n={vol_n}) ---")
    print(f"  {'MAE':<40} {vk:>10.4f} {vn:>10.4f} {'LGBM' if vk < vn else 'NAIVE':>10}")
    print(f"  {'Win rate':<40} {vwr:>9.1f}% {100-vwr:>9.1f}%  {'LGBM' if vwr > 50 else 'NAIVE':>10}")
    print(f"  {'TCP coverage during vol spikes':<40} {vcov:>9.1f}%")
    print(f"  {'MAE improvement in high-vol':<40} {vimp:>9.1f}%")

# --- Event day performance ---
if event_n > 0:
    ek = rdf.loc[event_mask, "e_err"].mean()
    en = rdf.loc[event_mask, "n_err"].mean()
    ewr = (rdf.loc[event_mask, "e_err"] < rdf.loc[event_mask, "n_err"]).mean() * 100
    print(f"\n  --- Event Day Performance (n={event_n}) ---")
    print(f"  {'MAE':<40} {ek:>10.4f} {en:>10.4f} {'LGBM' if ek < en else 'NAIVE':>10}")
    print(f"  {'Win rate':<40} {ewr:>9.1f}% {100-ewr:>9.1f}%  {'LGBM' if ewr > 50 else 'NAIVE':>10}")

print(f"\n  {'='*70}")

# ==========================================================================
# GO/NO-GO CRITERIA
# ==========================================================================
passes = 0
checks = [
    ("MAE improvement > 15%",          mae_imp > 15),
    ("RMSE improvement > 10%",         rmse_imp > 10),
    ("Win rate > 55%",                  e_wr > 55),
    ("Direction accuracy > 55%",        e_dir_acc > 55),
    ("Correlation > naive",             e_corr > n_corr),
    ("TCP coverage within 5% target",   calibration_ok),
]
for label, passed in checks:
    status = "PASS" if passed else "FAIL"
    passes += int(passed)
    print(f"  [{status}] {label}")

print(f"\n  Score: {passes}/6")

if passes >= 5:
    print("\n  >>> VERDICT: INTEGRATE LightGBM as primary vol forecaster <<<")
elif passes >= 4:
    print("\n  >>> VERDICT: STRONG — upgrade to Hybrid (HAR + LGBM residual, Rank 3) <<<")
elif passes >= 3:
    print("\n  >>> VERDICT: PROMISING — tune hyperparameters, add more features <<<")
else:
    print("\n  >>> VERDICT: NEEDS REWORK — check feature engineering <<<")

print("=" * 70)

# ==========================================================================
# 7. SAVE ARTIFACTS
# ==========================================================================
print(f"\n[7/7] Saving artifacts...")

# Save results CSV
results_file = "lgbm_vol_poc_results.csv"
rdf.to_csv(results_file, index=False)
print(f"  Results: {results_file}")

# Save LightGBM model
model_file = "lgbm_vol_model.txt"
if lgbm_model is not None:
    lgbm_model.save_model(model_file)
    print(f"  Model: {model_file}")

    # Also save feature importance
    import json
    imp_file = "lgbm_vol_feature_importance.json"
    if importance_count > 0:
        avg_importance = {f: importance_sums[f] / importance_count for f in FEATURE_NAMES}
        with open(imp_file, "w") as fp:
            json.dump({
                "feature_importance": avg_importance,
                "n_fits": importance_count,
                "feature_names": FEATURE_NAMES,
                "params": {
                    "n_estimators": LGBM_N_ESTIMATORS,
                    "max_depth": LGBM_MAX_DEPTH,
                    "learning_rate": LGBM_LEARNING_RATE,
                    "min_child_samples": LGBM_MIN_CHILD,
                    "subsample": LGBM_SUBSAMPLE,
                    "colsample_bytree": LGBM_COLSAMPLE,
                },
                "metrics": {
                    "mae": float(e_mae),
                    "rmse": float(e_rmse),
                    "naive_mae": float(n_mae),
                    "naive_rmse": float(n_rmse),
                    "mae_improvement_pct": float(mae_imp),
                    "rmse_improvement_pct": float(rmse_imp),
                    "win_rate": float(e_wr),
                    "direction_accuracy": float(e_dir_acc),
                    "tcp_coverage": float(tcp_coverage),
                },
            }, fp, indent=2)
        print(f"  Feature importance: {imp_file}")

# For Colab: copy to /content/ for easy download
try:
    import shutil
    for f in [results_file, model_file, imp_file]:
        if os.path.exists(f):
            dst = os.path.join("/content", f)
            shutil.copy2(f, dst)
            print(f"  Copied to {dst}")
except Exception:
    pass  # not on Colab

# ==========================================================================
# SCORECARD: LightGBM vs HAR-RV vs Naive ATR
# ==========================================================================
print(f"\n{'='*70}")
print("  SCORECARD: LGBM vs HAR-RV vs NAIVE ATR")
print(f"{'='*70}")
print(f"  {'MODEL':<30} {'MAE':>10} {'MAE IMP':>10} {'WR':>8} {'DIR':>8}")
print(f"  {'-'*66}")
print(f"  {'Naive ATR (baseline)':<30} {n_mae:>10.4f} {'---':>10} {'---':>8} {'50.0%':>8}")
print(f"  {'HAR-RV (Rank 1, prev POC)':<30} {'~':>10} {'~10-15%':>10} {'~55%':>8} {'~58%':>8}")
print(f"  {'LightGBM (Rank 2, this POC)':<30} {e_mae:>10.4f} {mae_imp:>9.1f}% {e_wr:>7.1f}% {e_dir_acc:>7.1f}%")
print(f"  {'Target range':<30} {'':>10} {'20-35%':>10} {'>55%':>8} {'>55%':>8}")

print(f"\n{'='*70}")
print("  COMPARISON: ALL POC ATTEMPTS")
print(f"{'='*70}")
print(f"  Kronos (TSFM):           1/4  FAIL  — domain shift, GPU required")
print(f"  EGARCH v1 (simulation):  1/5  FAIL  — scale explosion")
print(f"  EGARCH v2 (adjusted):    1/5  FAIL  — GARCH is a filter, not forecaster")
print(f"  HAR-RV + multipliers:    4/5  PASS  — integrated as Rank 1")
print(f"  LightGBM (21 features):  {passes}/6  {'PASS' if passes >= 4 else 'FAIL'}")
print(f"  Time: {total_time:.1f}s (CPU only, no GPU needed)")
print(f"  Dependencies: lightgbm, hmmlearn, sklearn (lightweight)")
print(f"{'='*70}")
