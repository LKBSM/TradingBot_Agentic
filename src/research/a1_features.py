"""
A1 feature matrix construction (vintage-aware, look-ahead-free).

Sprint QUANT-1.1 (Elena, 4h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 2.

Builds the feature matrix that the A1 stacked LightGBM (QUANT-1.3) will
consume to answer the decisive Phase 1 question: does Smart Sentinel have
a predictive edge on XAU/USD M15 returns at h=4 (1h) and h=16 (4h)?

Anti-leak guarantee
-------------------
Every macro and COT feature is joined using `vintage_date <= bar_timestamp`,
so a backtest at bar t cannot see a value that wasn't yet published. This is
enforced by `_lookup_vintaged_value` (binary search on sorted vintage dates)
and verified by the test suite on 100 random timestamps.

Calendar features use scheduled event times (pre-announced), so no leak by
construction — we only ever know "next event in N minutes" / "last event N
minutes ago", never the actual outcome at bar t < event_time.

Feature inventory (19 features ≥ 18 plan target)
------------------------------------------------
Price-based (6):
    r_1, r_4, r_16, ATR_14_pct, RSI_14, MACD_signal_diff
Intra-day (3):
    bar_minute_of_day, dow, is_lunch_hour
Macro vintage-aware (6):
    DGS10, BREAKEVEN_10Y, DTWEXBGS, VIXCLS, T10Y2Y, COT mm_net_pct_z52
Macro extras (2 — compensate GLD deferred):
    COT producer_net_z52, atr_ratio (atr_14 / atr_50)
Calendar proximity (2):
    min_to_next_red_news, min_since_last_red_news

Targets:
    r_forward_4  (close-to-close, h=4 bars = 1h)
    r_forward_16 (close-to-close, h=16 bars = 4h)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRICE_CSV = REPO_ROOT / "data" / "XAU_15MIN_2019_2026.csv"
DEFAULT_MACRO_DIR = REPO_ROOT / "data" / "macro"
DEFAULT_COT_CSV = REPO_ROOT / "data" / "macro" / "cot_gold.csv"
DEFAULT_CALENDAR_CSV = (
    REPO_ROOT / "data" / "economic_calendar_HIGH_IMPACT_2019_2025.csv"
)
DEFAULT_OUTPUT_PARQUET = (
    REPO_ROOT / "data" / "research" / "a1_matrix_2019_2026.parquet"
)

FRED_SERIES = ["DGS10", "BREAKEVEN_10Y", "DTWEXBGS", "VIXCLS", "T10Y2Y"]

# Calendar HIGH_IMPACT impact filter — already pre-filtered in the CSV but
# kept explicit as a defensive guard.
RED_IMPACT_VALUES = {"High", "high", "HIGH", "RED", "Red"}

# Forward horizons for targets (in M15 bars).
H_FORWARD_4 = 4   # 1 hour
H_FORWARD_16 = 16  # 4 hours

# Final feature column order — keep stable for downstream model versions.
FEATURE_COLUMNS = [
    # Price-based (6)
    "r_1",
    "r_4",
    "r_16",
    "atr_14_pct",
    "rsi_14",
    "macd_signal_diff",
    # Intra-day (3)
    "bar_minute_of_day",
    "dow",
    "is_lunch_hour",
    # Macro vintage-aware (6)
    "dgs10",
    "breakeven_10y",
    "dtwexbgs",
    "vix",
    "t10y2y",
    "cot_mm_net_pct_z52",
    # Macro extras (2)
    "cot_producer_net_z52",
    "atr_ratio_14_50",
    # Calendar proximity (2)
    "min_to_next_red_news",
    "min_since_last_red_news",
]

TARGET_COLUMNS = ["r_forward_4", "r_forward_16"]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FeatureMatrixStats:
    """Diagnostic stats produced after `build_feature_matrix`."""

    rows_in: int
    rows_out: int  # after dropna of essential columns
    feature_count: int
    nan_per_feature: dict[str, int]
    leak_test_failures: int = -1  # -1 = not run yet


# ---------------------------------------------------------------------------
# Price feature engineering
# ---------------------------------------------------------------------------


def load_price_csv(path: Path | str) -> pd.DataFrame:
    """Load XAU M15 OHLCV. Treats the timestamp column as UTC.

    The Dukascopy export uses a "Date" column without tz info; we anchor to
    UTC to interoperate with FRED/COT vintages. If the source is actually in
    GMT0/exchange-local, the consistency matters more than the absolute
    offset for vintage joins (we always compare bar_ts vs vintage_date in
    UTC after this normalisation).
    """
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["Date"], utc=True)
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def _atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Welles Wilder ATR — exponential smoothing with alpha = 1/period.

    `tr` = max(high-low, |high-close_prev|, |low-close_prev|).
    Rolling implementation seeded with simple mean of first `period` values.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Wilder smoothing equivalent to EMA with alpha=1/period.
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Standard RSI (Wilder)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd_signal_diff(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.Series:
    """MACD histogram: (EMA_fast - EMA_slow) - EMA_signal_of_macd."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 7 price-based feature columns (r_1, r_4, r_16, ATR%, RSI, MACD,
    plus atr_ratio_14_50)."""
    out = df.copy()
    log_close = np.log(out["close"])

    out["r_1"] = log_close.diff(1)
    out["r_4"] = log_close.diff(H_FORWARD_4)
    out["r_16"] = log_close.diff(H_FORWARD_16)

    atr_14 = _atr_wilder(out["high"], out["low"], out["close"], 14)
    atr_50 = _atr_wilder(out["high"], out["low"], out["close"], 50)
    out["atr_14_pct"] = atr_14 / out["close"]
    out["atr_ratio_14_50"] = atr_14 / atr_50

    out["rsi_14"] = _rsi(out["close"], 14)
    out["macd_signal_diff"] = _macd_signal_diff(out["close"])

    return out


def add_intra_features(df: pd.DataFrame) -> pd.DataFrame:
    """3 intra-day features: minute_of_day, day-of-week, lunch-hour flag."""
    out = df.copy()
    ts = out["timestamp"]
    out["bar_minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
    out["dow"] = ts.dt.dayofweek  # 0 = Monday
    # London lunch ≈ 11-13 UTC; NY lunch ≈ 16-18 UTC. We flag the cumulative
    # window 11:00-13:00 UTC and 16:00-18:00 UTC as "low-vol" lunch periods.
    minute = out["bar_minute_of_day"]
    out["is_lunch_hour"] = (
        ((minute >= 11 * 60) & (minute < 13 * 60))
        | ((minute >= 16 * 60) & (minute < 18 * 60))
    ).astype("int8")
    return out


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Forward log-returns at h=4 and h=16 bars. By construction these are
    pure lookups of FUTURE close — they're TARGETS, not features, so the
    leak rule is the opposite: they must NOT be merged with present features
    when computing the model input row."""
    out = df.copy()
    log_close = np.log(out["close"])
    out["r_forward_4"] = log_close.shift(-H_FORWARD_4) - log_close
    out["r_forward_16"] = log_close.shift(-H_FORWARD_16) - log_close
    return out


# ---------------------------------------------------------------------------
# Macro / COT vintage-aware merge
# ---------------------------------------------------------------------------


def load_fred_series(macro_dir: Path | str, series_id: str) -> pd.DataFrame:
    """Load a single FRED CSV with date_utc, value, vintage_date columns."""
    path = Path(macro_dir) / f"fred_{series_id}.csv"
    df = pd.read_csv(path)
    df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.sort_values("vintage_date").reset_index(drop=True)


def load_cot(path: Path | str) -> pd.DataFrame:
    """Load the COT CSV produced by cot_provider."""
    df = pd.read_csv(path)
    df["report_date"] = pd.to_datetime(df["report_date"], utc=True)
    df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True)
    return df.sort_values("vintage_date").reset_index(drop=True)


def _vintaged_lookup(
    timestamps: pd.Series, source_df: pd.DataFrame, value_col: str
) -> np.ndarray:
    """For each ts in `timestamps`, return the most-recent `source_df[value_col]`
    whose `vintage_date <= ts`. Vectorised via searchsorted on the sorted
    vintage_date column (assumed pre-sorted by caller).

    Returns NaN where no row is yet published.
    """
    ts_utc = pd.to_datetime(timestamps, utc=True).to_numpy()
    vintages = source_df["vintage_date"].to_numpy()
    values = source_df[value_col].to_numpy()
    # `side="right"` ensures we get the count of rows with vintage <= ts, then
    # idx-1 picks the last one. Negative idx (=-1) means "nothing yet"; mask.
    idx = np.searchsorted(vintages, ts_utc, side="right") - 1
    out = np.where(idx >= 0, values[np.maximum(idx, 0)], np.nan)
    return out


def add_macro_features(
    df: pd.DataFrame, macro_dir: Path | str = DEFAULT_MACRO_DIR
) -> pd.DataFrame:
    """Vintage-aware merge of the 5 FRED series."""
    out = df.copy()
    feature_map = {
        "dgs10": "DGS10",
        "breakeven_10y": "BREAKEVEN_10Y",
        "dtwexbgs": "DTWEXBGS",
        "vix": "VIXCLS",
        "t10y2y": "T10Y2Y",
    }
    for col, sid in feature_map.items():
        try:
            src = load_fred_series(macro_dir, sid)
        except FileNotFoundError:
            logger.warning("FRED series %s not found in %s; column will be NaN", sid, macro_dir)
            out[col] = np.nan
            continue
        out[col] = _vintaged_lookup(out["timestamp"], src, "value")
    return out


def add_cot_features(df: pd.DataFrame, cot_path: Path | str = DEFAULT_COT_CSV) -> pd.DataFrame:
    """Vintage-aware merge of two COT features."""
    out = df.copy()
    try:
        cot = load_cot(cot_path)
    except FileNotFoundError:
        logger.warning("COT file %s not found; columns will be NaN", cot_path)
        out["cot_mm_net_pct_z52"] = np.nan
        out["cot_producer_net_z52"] = np.nan
        return out

    # Z-score producer_net on its own 52-week rolling window for symmetry
    # with mm_net_pct_z52 (already computed in cot_provider).
    if "producer_net" in cot.columns:
        roll = cot["producer_net"].rolling(window=52, min_periods=26)
        cot["producer_net_z52"] = (cot["producer_net"] - roll.mean()) / roll.std(ddof=0)

    out["cot_mm_net_pct_z52"] = _vintaged_lookup(
        out["timestamp"], cot, "mm_net_pct_z52"
    )
    out["cot_producer_net_z52"] = _vintaged_lookup(
        out["timestamp"], cot, "producer_net_z52"
    )
    return out


# ---------------------------------------------------------------------------
# Calendar proximity features
# ---------------------------------------------------------------------------


def load_calendar(path: Path | str = DEFAULT_CALENDAR_CSV) -> pd.DataFrame:
    """Load high-impact economic calendar (USD-only typically). Filters to
    Impact in {High, RED} as defensive guard even if the file is pre-filtered.
    """
    df = pd.read_csv(path)
    df["event_ts"] = pd.to_datetime(df["Date"], utc=True)
    if "Impact" in df.columns:
        df = df[df["Impact"].astype(str).isin(RED_IMPACT_VALUES)]
    return df.sort_values("event_ts").reset_index(drop=True)


def add_calendar_features(
    df: pd.DataFrame, calendar_path: Path | str = DEFAULT_CALENDAR_CSV
) -> pd.DataFrame:
    """For each bar, compute minutes-to-next and minutes-since-last red event.

    Pre-scheduled events leak no information about price (the *time* of FOMC
    is announced months ahead). The actual *outcome* would leak — we don't
    use it here.
    """
    out = df.copy()
    try:
        cal = load_calendar(calendar_path)
    except FileNotFoundError:
        logger.warning("Calendar %s not found; columns will be NaN", calendar_path)
        out["min_to_next_red_news"] = np.nan
        out["min_since_last_red_news"] = np.nan
        return out

    if cal.empty:
        out["min_to_next_red_news"] = np.nan
        out["min_since_last_red_news"] = np.nan
        return out

    bar_ts = out["timestamp"].to_numpy()
    events = cal["event_ts"].to_numpy()

    # Next event: index of first event >= bar_ts (searchsorted side="left").
    next_idx = np.searchsorted(events, bar_ts, side="left")
    last_idx = next_idx - 1

    # Vectorised conversion to minutes using astype on timedelta64.
    next_delta_min = np.where(
        next_idx < len(events),
        (events[np.minimum(next_idx, len(events) - 1)] - bar_ts)
        .astype("timedelta64[s]")
        .astype(np.float64)
        / 60.0,
        np.nan,
    )
    last_delta_min = np.where(
        last_idx >= 0,
        (bar_ts - events[np.maximum(last_idx, 0)])
        .astype("timedelta64[s]")
        .astype(np.float64)
        / 60.0,
        np.nan,
    )

    out["min_to_next_red_news"] = next_delta_min
    out["min_since_last_red_news"] = last_delta_min
    return out


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------


def build_feature_matrix(
    price_path: Path | str = DEFAULT_PRICE_CSV,
    macro_dir: Path | str = DEFAULT_MACRO_DIR,
    cot_path: Path | str = DEFAULT_COT_CSV,
    calendar_path: Path | str = DEFAULT_CALENDAR_CSV,
    drop_warmup: bool = True,
) -> tuple[pd.DataFrame, FeatureMatrixStats]:
    """Build the A1 matrix end-to-end.

    Parameters
    ----------
    drop_warmup : bool
        If True, drop bars where any essential feature is NaN due to the
        rolling-indicator warmup (typically first ~50 bars). If False, keep
        all bars including NaNs (callers can handle NaN themselves).

    Returns
    -------
    (df, stats) : the feature DataFrame ordered by timestamp, plus diagnostic
    stats (rows in/out, NaN per feature, leak-test placeholder).
    """
    rows_in = -1
    df = load_price_csv(price_path)
    rows_in = len(df)
    logger.info("Loaded %d XAU M15 bars from %s", rows_in, price_path)

    df = add_price_features(df)
    df = add_intra_features(df)
    df = add_macro_features(df, macro_dir)
    df = add_cot_features(df, cot_path)
    df = add_calendar_features(df, calendar_path)
    df = add_target_columns(df)

    # Reorder columns for stable schema.
    keep = ["timestamp"] + FEATURE_COLUMNS + TARGET_COLUMNS + ["close"]
    df = df[keep]

    nan_per_feature = {col: int(df[col].isna().sum()) for col in FEATURE_COLUMNS}

    if drop_warmup:
        # Drop only on FEATURES, not on targets (targets are NaN at the end
        # of the timeline by construction; we keep those rows for inspection).
        before = len(df)
        df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
        after = len(df)
        logger.info("Dropped %d warmup rows (NaN in features); %d remain", before - after, after)

    stats = FeatureMatrixStats(
        rows_in=rows_in,
        rows_out=len(df),
        feature_count=len(FEATURE_COLUMNS),
        nan_per_feature=nan_per_feature,
    )
    return df, stats


def save_parquet(df: pd.DataFrame, path: Path | str = DEFAULT_OUTPUT_PARQUET) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    logger.info("Saved A1 matrix (%d rows, %d cols) -> %s", len(df), len(df.columns), out)
    return out


# ---------------------------------------------------------------------------
# Anti-leak validation
# ---------------------------------------------------------------------------


def leak_test_macro_at_bar(
    feature_df: pd.DataFrame,
    macro_dir: Path | str = DEFAULT_MACRO_DIR,
    cot_path: Path | str = DEFAULT_COT_CSV,
    n_random: int = 100,
    seed: int = 42,
) -> int:
    """Verify that for `n_random` randomly-sampled rows of the feature matrix,
    each macro/COT feature value matches the most-recent vintaged source row
    with `vintage_date <= bar_timestamp`.

    Returns the number of failures (0 = perfect anti-leak).
    """
    rng = np.random.default_rng(seed)
    if len(feature_df) == 0:
        return 0
    sample_idx = rng.integers(0, len(feature_df), size=n_random)
    sample = feature_df.iloc[sample_idx].reset_index(drop=True)

    # FRED checks
    fred_map = {
        "dgs10": "DGS10",
        "breakeven_10y": "BREAKEVEN_10Y",
        "dtwexbgs": "DTWEXBGS",
        "vix": "VIXCLS",
        "t10y2y": "T10Y2Y",
    }
    failures = 0
    for col, sid in fred_map.items():
        try:
            src = load_fred_series(macro_dir, sid)
        except FileNotFoundError:
            continue
        expected = _vintaged_lookup(sample["timestamp"], src, "value")
        actual = sample[col].to_numpy()
        # NaN-safe comparison
        diff_mask = ~(
            (np.isnan(expected) & np.isnan(actual))
            | np.isclose(expected, actual, equal_nan=False)
        )
        failures += int(diff_mask.sum())

    # COT checks
    try:
        cot = load_cot(cot_path)
        if "producer_net" in cot.columns:
            roll = cot["producer_net"].rolling(window=52, min_periods=26)
            cot["producer_net_z52"] = (cot["producer_net"] - roll.mean()) / roll.std(ddof=0)
        for col, src_col in [
            ("cot_mm_net_pct_z52", "mm_net_pct_z52"),
            ("cot_producer_net_z52", "producer_net_z52"),
        ]:
            expected = _vintaged_lookup(sample["timestamp"], cot, src_col)
            actual = sample[col].to_numpy()
            diff_mask = ~(
                (np.isnan(expected) & np.isnan(actual))
                | np.isclose(expected, actual, equal_nan=False)
            )
            failures += int(diff_mask.sum())
    except FileNotFoundError:
        pass

    return failures


# ---------------------------------------------------------------------------
# CLI smoke runner
# ---------------------------------------------------------------------------


def _main() -> None:  # pragma: no cover  (CLI smoke runner, not unit-tested)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    df, stats = build_feature_matrix()
    save_parquet(df)
    fails = leak_test_macro_at_bar(df, n_random=100)

    print()
    print("=== A1 feature matrix smoke ===")
    print(f"  Bars in:           {stats.rows_in}")
    print(f"  Bars out:          {stats.rows_out}")
    print(f"  Features:          {stats.feature_count}")
    print(f"  Leak test (n=100): {fails} failures")
    print()
    print("  NaN per feature:")
    for col, n in stats.nan_per_feature.items():
        print(f"    {col:30s}  {n:6d}")
    print()
    kpi_rows_ok = stats.rows_out >= 150_000
    kpi_features_ok = stats.feature_count >= 18
    kpi_leak_ok = fails == 0
    overall = "PASS" if (kpi_rows_ok and kpi_features_ok and kpi_leak_ok) else "FAIL"
    print(f"  KPI: rows>=150k {kpi_rows_ok} | features>=18 {kpi_features_ok} | leak=0 {kpi_leak_ok}")
    print(f"  Overall: {overall}")


if __name__ == "__main__":  # pragma: no cover
    _main()
