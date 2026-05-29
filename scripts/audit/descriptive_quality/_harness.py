"""Shared harness for the descriptive-quality audit (2026-05-27).

Loads OHLCV + calendar once, runs the prod SmartMoneyEngine, exposes train/OOS
splits to the per-block evaluation scripts. Does not modify prod code.

Conventions:
  TRAIN = 2019-01-01 → 2023-12-31  (calibration window for any model we fit)
  OOS   = 2024-01-01 → end-of-file (XAU: ~2026-05; EUR: ~2025-12)
"""

from __future__ import annotations

import os
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Quiet down library noise that pollutes audit output
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("HMM_VERBOSE", "0")

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from src.environment.strategy_features import SmartMoneyEngine, SMCConfig  # noqa: E402

# Force INFO-only logging from the engine so its summary doesn't get lost
import logging

logging.getLogger("src.environment.strategy_features").setLevel(logging.WARNING)


TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
OOS_START = pd.Timestamp("2024-01-01 00:00:00")


@dataclass
class InstrumentData:
    symbol: str
    raw: pd.DataFrame          # full OHLCV
    enriched: pd.DataFrame     # full OHLCV + SMC features (causal over full history)
    oos_mask: pd.Series        # True where index >= OOS_START
    train_mask: pd.Series      # True where index <= TRAIN_END

    @property
    def oos(self) -> pd.DataFrame:
        return self.enriched[self.oos_mask]

    @property
    def train(self) -> pd.DataFrame:
        return self.enriched[self.train_mask]


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalize column names + parse timestamps
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Strip rows with NaN OHLC (rare; e.g., daylight savings)
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def load_instrument(symbol: str) -> InstrumentData:
    """Load one instrument, run SmartMoneyEngine over the FULL history (causal)."""
    csv_path = {
        "XAUUSD": REPO / "data" / "XAU_15MIN_2019_2026.csv",
        "EURUSD": REPO / "data" / "EURUSD_15MIN_2019_2025.csv",
    }[symbol]

    raw = _load_csv(csv_path)
    config = {
        "RSI_WINDOW": 14,
        "MACD_FAST": 12,
        "MACD_SLOW": 26,
        "MACD_SIGNAL": 9,
        "BB_WINDOW": 20,
        "ATR_WINDOW": 14,
        "FRACTAL_WINDOW": 2,
        "FVG_THRESHOLD": 0.1,
        "OB_REQUIRE_FVG": False,
        "OB_FVG_BONUS": 0.2,
        "RETEST_TOL_ATR": 0.5,
        "RETEST_INVALID_TOL_ATR": 1.0,
        "RETEST_AWAITING_TIMEOUT": 20,
        "RETEST_ARMED_WINDOW": 30,
    }
    engine = SmartMoneyEngine(raw, config, verbose=False)
    enriched = engine.analyze()
    # Engine drops some initial rows for indicator warmup. Re-align on its own index.
    enriched.index = raw.index[-len(enriched):] if len(enriched) < len(raw) else raw.index

    train_mask = pd.Series(enriched.index <= TRAIN_END, index=enriched.index)
    oos_mask = pd.Series(enriched.index >= OOS_START, index=enriched.index)

    return InstrumentData(
        symbol=symbol,
        raw=raw,
        enriched=enriched,
        oos_mask=oos_mask,
        train_mask=train_mask,
    )


def load_calendar(start: Optional[pd.Timestamp] = None,
                  end: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Load the HIGH-impact calendar CSV used by prod."""
    path = REPO / "data" / "economic_calendar_HIGH_IMPACT_2019_2025.csv"
    cal = pd.read_csv(path)
    cal.columns = [c.lower() for c in cal.columns]
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date").reset_index(drop=True)
    if start is not None:
        cal = cal[cal["date"] >= start]
    if end is not None:
        cal = cal[cal["date"] <= end]
    return cal.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Independent causal SMC reference (used to cross-check the prod engine).
#
# The prod engine uses a 2-bar Williams fractal. To give a meaningful Q1 score
# (not 1.00 by construction) we re-detect structures using a textbook
# alternative: a 3-bar swing definition (Tom DeMark style) over a windowed
# search. Agreement between *two* reasonable SMC implementations is the
# descriptive truth signal we report.
# ----------------------------------------------------------------------------

def detect_swings_3bar(highs: np.ndarray, lows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (swing_high_at_i, swing_low_at_i) with a 3-bar centered window,
    shifted by +1 to make it causal (a swing at bar i is only known at bar i+1)."""
    n = len(highs)
    sh = np.full(n, np.nan)
    sl = np.full(n, np.nan)
    for i in range(1, n - 1):
        if highs[i] >= highs[i - 1] and highs[i] >= highs[i + 1]:
            sh[i] = highs[i]
        if lows[i] <= lows[i - 1] and lows[i] <= lows[i + 1]:
            sl[i] = lows[i]
    # Causal shift: only known after the right neighbor closes
    sh_causal = np.full(n, np.nan)
    sl_causal = np.full(n, np.nan)
    sh_causal[2:] = sh[1:-1]
    sl_causal[2:] = sl[1:-1]
    return sh_causal, sl_causal


def detect_bos_independent(df: pd.DataFrame) -> pd.DataFrame:
    """Independent BOS/CHOCH detection using 3-bar swings instead of 2-bar fractals.

    Returns a frame indexed like df, with columns:
      ref_bos_event   : 1 / -1 / 0 (event bar)
      ref_bos_level   : level broken on event bar (NaN otherwise)
      ref_swing_high  : last known swing-high level at bar i
      ref_swing_low   : last known swing-low level at bar i
    """
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    sh, sl = detect_swings_3bar(highs, lows)

    n = len(df)
    # Running last-known swing levels
    last_h = np.full(n, np.nan)
    last_l = np.full(n, np.nan)
    h = np.nan
    l = np.nan
    for i in range(n):
        if not np.isnan(sh[i]):
            h = sh[i]
        if not np.isnan(sl[i]):
            l = sl[i]
        last_h[i] = h
        last_l[i] = l

    bos_event = np.zeros(n, dtype=np.int32)
    bos_level = np.full(n, np.nan)
    trend = 0  # +1 up, -1 down
    prev_h = np.nan  # the high we're tracking as resistance
    prev_l = np.nan
    for i in range(1, n):
        h_ref = prev_h if not np.isnan(prev_h) else last_h[i - 1]
        l_ref = prev_l if not np.isnan(prev_l) else last_l[i - 1]
        if not np.isnan(h_ref) and closes[i] > h_ref and (trend != 1 or last_h[i - 1] > h_ref):
            bos_event[i] = 1
            bos_level[i] = h_ref
            trend = 1
            prev_h = last_h[i]
            prev_l = last_l[i]
        elif not np.isnan(l_ref) and closes[i] < l_ref and (trend != -1 or last_l[i - 1] < l_ref):
            bos_event[i] = -1
            bos_level[i] = l_ref
            trend = -1
            prev_h = last_h[i]
            prev_l = last_l[i]
        else:
            # Carry forward
            prev_h = last_h[i] if not np.isnan(last_h[i]) else prev_h
            prev_l = last_l[i] if not np.isnan(last_l[i]) else prev_l

    return pd.DataFrame(
        {
            "ref_bos_event": bos_event,
            "ref_bos_level": bos_level,
            "ref_swing_high": last_h,
            "ref_swing_low": last_l,
        },
        index=df.index,
    )


def detect_fvg_independent(df: pd.DataFrame) -> pd.DataFrame:
    """Independent FVG detection: textbook formal definition with no ATR threshold.

    Bullish FVG at bar i ⇔ low[i] > high[i-2] (gap between i-2 and i, mid-bar at i-1).
    Causal: known at bar i (which is bar i in prod's vectorized terms).
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    n = len(df)
    fvg_dir = np.zeros(n, dtype=np.int32)
    fvg_low = np.full(n, np.nan)
    fvg_high = np.full(n, np.nan)
    for i in range(2, n):
        if l[i] > h[i - 2]:
            fvg_dir[i] = 1
            fvg_low[i] = h[i - 2]
            fvg_high[i] = l[i]
        elif h[i] < l[i - 2]:
            fvg_dir[i] = -1
            fvg_low[i] = h[i]
            fvg_high[i] = l[i - 2]
    size = np.where(fvg_dir != 0, fvg_high - fvg_low, 0.0)
    return pd.DataFrame(
        {
            "ref_fvg_dir": fvg_dir,
            "ref_fvg_low": fvg_low,
            "ref_fvg_high": fvg_high,
            "ref_fvg_size": size,
        },
        index=df.index,
    )


def detect_ob_independent(df: pd.DataFrame) -> pd.DataFrame:
    """Independent OB: dernier opposite candle before a >1xATR impulse move."""
    o = df["open"].to_numpy()
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    atr = df["atr"].to_numpy() if "atr" in df.columns else df["ATR"].to_numpy()
    n = len(df)
    bull_ob = np.zeros(n, dtype=bool)
    bear_ob = np.zeros(n, dtype=bool)
    for i in range(2, n):
        # Bullish OB: bar i-1 was bearish (close<open) and bar i is bullish with high>high[i-1]
        # and the body of bar i exceeds 1×ATR (impulse).
        if (
            c[i - 1] < o[i - 1]
            and c[i] > o[i]
            and h[i] > h[i - 1]
            and (c[i] - o[i]) > atr[i] * 0.8
        ):
            bull_ob[i] = True
        if (
            c[i - 1] > o[i - 1]
            and c[i] < o[i]
            and l[i] < l[i - 1]
            and (o[i] - c[i]) > atr[i] * 0.8
        ):
            bear_ob[i] = True

    bull_high = np.where(bull_ob, np.roll(h, 1), np.nan)
    bull_low = np.where(bull_ob, np.roll(l, 1), np.nan)
    bear_high = np.where(bear_ob, np.roll(h, 1), np.nan)
    bear_low = np.where(bear_ob, np.roll(l, 1), np.nan)
    return pd.DataFrame(
        {
            "ref_bull_ob": bull_ob.astype(np.int32),
            "ref_bear_ob": bear_ob.astype(np.int32),
            "ref_bull_ob_high": bull_high,
            "ref_bull_ob_low": bull_low,
            "ref_bear_ob_high": bear_high,
            "ref_bear_ob_low": bear_low,
        },
        index=df.index,
    )


# ----------------------------------------------------------------------------
# Stat helpers
# ----------------------------------------------------------------------------

def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Return (point, lo, hi) for a statistic via percentile bootstrap.

    `statistic` is a callable taking a 1-D array and returning a scalar.
    """
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return (float("nan"), float("nan"), float("nan"))
    point = float(statistic(values))
    boots = np.empty(n_boot)
    n = len(values)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = statistic(values[idx])
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return point, lo, hi


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def verdict_f1(f1: float) -> str:
    if f1 >= 0.85:
        return "🟢"
    if f1 >= 0.65:
        return "🟡"
    return "🔴"


def verdict_ece(ece: float) -> str:
    if ece <= 0.05:
        return "🟢"
    if ece <= 0.10:
        return "🟡"
    return "🔴"


def verdict_picp(empirical: float, nominal: float) -> str:
    diff = abs(empirical - nominal)
    if diff <= 0.02:
        return "🟢"
    if diff <= 0.05:
        return "🟡"
    return "🔴"


if __name__ == "__main__":
    print("Loading XAUUSD M15 ...")
    xau = load_instrument("XAUUSD")
    print(f"  raw bars:      {len(xau.raw):>8,d}")
    print(f"  enriched bars: {len(xau.enriched):>8,d}")
    print(f"  train bars:    {xau.train_mask.sum():>8,d}  ({xau.enriched.index[xau.train_mask].min()} -> {TRAIN_END})")
    print(f"  OOS bars:      {xau.oos_mask.sum():>8,d}  ({OOS_START} -> {xau.enriched.index[xau.oos_mask].max()})")

    print("\nLoading EURUSD M15 ...")
    eur = load_instrument("EURUSD")
    print(f"  raw bars:      {len(eur.raw):>8,d}")
    print(f"  enriched bars: {len(eur.enriched):>8,d}")
    print(f"  train bars:    {eur.train_mask.sum():>8,d}")
    print(f"  OOS bars:      {eur.oos_mask.sum():>8,d}")

    print("\nLoading calendar ...")
    cal = load_calendar()
    print(f"  events: {len(cal):,d}")
    print(f"  first:  {cal['date'].iloc[0]}")
    print(f"  last:   {cal['date'].iloc[-1]}")
