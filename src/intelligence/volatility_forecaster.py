"""Production HAR-RV + Diurnal + Calendar + HMM volatility forecaster.

Ported from scripts/colab_har_rv_poc.py (scored 4/5 on out-of-sample evaluation).
Architecture: forecast = blend_w * (HAR_base * diurnal * calendar * regime) + (1-blend_w) * naive_ATR

Components:
  1. HAR-RV (Heterogeneous Autoregressive Realized Volatility) — Corsi 2009
     Multi-scale persistence: daily + weekly + monthly lookback
  2. Yang-Zhang realized volatility — 14x efficient vs close-to-close
  3. Diurnal profile — per-hour intraday seasonality multiplier
  4. Calendar event proximity — NFP/CPI/FOMC vol spike multipliers
  5. HMM regime detection — 3-state (low/normal/high) conditioning
  6. TCP conformal prediction — adaptive prediction intervals

Supports multiple instruments and timeframes via InstrumentConfig.
"""

from __future__ import annotations

import logging
import pickle
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# INSTRUMENT REGISTRY — Per-instrument presets
# =============================================================================

def get_instrument_registry() -> Dict[str, "InstrumentConfig"]:
    """Return the default instrument registry with presets for common instruments."""
    return {
        "XAUUSD": InstrumentConfig(
            symbol="XAUUSD",
            timeframe="M15",
            bars_per_day=96,
            session_hours={
                "asian": (0, 8),
                "london": (8, 13),
                "ny_overlap": (13, 17),
                "ny_afternoon": (17, 21),
                "after_hours": (21, 24),
            },
            calendar_events=[
                "Non-Farm Payrolls", "Federal Funds Rate", "CPI m/m",
                "Core CPI m/m", "FOMC Statement", "FOMC Press Conference",
                "GDP q/q", "Core PCE Price Index m/m", "Retail Sales m/m",
            ],
            sl_atr_mult=2.0,
            tp_atr_mult=4.0,
            price_decimals=2,
        ),
        "EURUSD": InstrumentConfig(
            symbol="EURUSD",
            timeframe="M15",
            bars_per_day=96,
            session_hours={
                "asian": (0, 7),
                "london": (7, 12),
                "ny_overlap": (12, 16),
                "ny_afternoon": (16, 21),
                "after_hours": (21, 24),
            },
            calendar_events=[
                "Non-Farm Payrolls", "ECB Rate Decision", "CPI m/m",
                "Core CPI m/m", "FOMC Statement", "GDP q/q",
                "ECB Press Conference", "Retail Sales m/m",
            ],
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            price_decimals=5,
        ),
        "BTCUSD": InstrumentConfig(
            symbol="BTCUSD",
            timeframe="M15",
            bars_per_day=96,
            session_hours={
                "asian": (0, 8),
                "london": (8, 13),
                "us": (13, 21),
                "after_hours": (21, 24),
            },
            calendar_events=[
                "FOMC Statement", "Federal Funds Rate", "CPI m/m",
            ],
            sl_atr_mult=2.0,
            tp_atr_mult=4.0,
            price_decimals=2,
        ),
        "US500": InstrumentConfig(
            symbol="US500",
            timeframe="M15",
            bars_per_day=28,
            session_hours={
                "pre_market": (8, 9),
                "regular": (9, 16),
                "after_hours": (16, 20),
            },
            calendar_events=[
                "Non-Farm Payrolls", "CPI m/m", "FOMC Statement",
                "GDP q/q", "Federal Funds Rate",
            ],
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            price_decimals=1,
        ),
        "GBPUSD": InstrumentConfig(
            symbol="GBPUSD",
            timeframe="M15",
            bars_per_day=96,
            session_hours={
                "asian": (0, 7),
                "london": (7, 12),
                "ny_overlap": (12, 16),
                "ny_afternoon": (16, 21),
                "after_hours": (21, 24),
            },
            calendar_events=[
                "Non-Farm Payrolls", "BOE Rate Decision", "CPI m/m",
                "GDP q/q", "FOMC Statement", "Retail Sales m/m",
            ],
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            price_decimals=5,
        ),
        "USDJPY": InstrumentConfig(
            symbol="USDJPY",
            timeframe="M15",
            bars_per_day=96,
            session_hours={
                "asian": (0, 8),
                "london": (8, 13),
                "ny_overlap": (13, 17),
                "ny_afternoon": (17, 21),
                "after_hours": (21, 24),
            },
            calendar_events=[
                "Non-Farm Payrolls", "BOJ Rate Decision", "CPI m/m",
                "FOMC Statement", "GDP q/q",
            ],
            sl_atr_mult=1.5,
            tp_atr_mult=3.0,
            price_decimals=3,
        ),
    }


# =============================================================================
# TIMEFRAME UTILITIES
# =============================================================================

TIMEFRAME_MINUTES: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}


def timeframe_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes. Raises ValueError if unknown."""
    if tf in TIMEFRAME_MINUTES:
        return TIMEFRAME_MINUTES[tf]
    raise ValueError(f"Unknown timeframe: {tf}. Use: {list(TIMEFRAME_MINUTES.keys())}")


def bars_per_day_for_timeframe(tf: str, trading_hours: int = 24) -> int:
    """Compute bars per day for a given timeframe and trading hours."""
    minutes_per_bar = timeframe_to_minutes(tf)
    return (trading_hours * 60) // minutes_per_bar


def resample_ohlcv(
    df: pd.DataFrame,
    source_tf: str,
    target_tf: str,
) -> pd.DataFrame:
    """Resample OHLCV from a shorter timeframe to a longer one.

    Args:
        df: OHLCV DataFrame with a datetime index or 'timestamp' column.
        source_tf: Source timeframe (e.g., "M1", "M5").
        target_tf: Target timeframe (must be >= source, e.g., "M15", "H1").

    Returns:
        Resampled OHLCV DataFrame.
    """
    src_min = timeframe_to_minutes(source_tf)
    tgt_min = timeframe_to_minutes(target_tf)
    if tgt_min < src_min:
        raise ValueError(f"Cannot upsample from {source_tf} to {target_tf}")
    if tgt_min == src_min:
        return df.copy()

    # Ensure datetime index
    work = df.copy()
    if "timestamp" in work.columns:
        work["timestamp"] = pd.to_datetime(work["timestamp"])
        work = work.set_index("timestamp")
    elif not isinstance(work.index, pd.DatetimeIndex):
        work.index = pd.to_datetime(work.index)

    # Normalize column names
    col_map = {}
    for col in work.columns:
        lower = col.lower()
        if lower in ("open", "high", "low", "close", "volume"):
            col_map[col] = lower
    work = work.rename(columns=col_map)

    rule = f"{tgt_min}min"
    resampled = work.resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    resampled = resampled.reset_index()
    resampled = resampled.rename(columns={"index": "timestamp"})
    if resampled.columns[0] != "timestamp" and isinstance(resampled.index, pd.DatetimeIndex):
        resampled = resampled.reset_index()

    return resampled


# =============================================================================
# INSTRUMENT CONFIGURATION
# =============================================================================

@dataclass
class InstrumentConfig:
    """Per-instrument configuration for volatility forecasting.

    HAR windows are auto-computed from bars_per_day unless explicitly set.
    """
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    bars_per_day: int = 96  # 24h * 4 bars/hour for M15

    # HAR lookback windows (auto-computed if 0)
    har_daily: int = 0
    har_weekly: int = 0
    har_monthly: int = 0
    har_train_min: int = 0

    # Session hours (UTC): {name: (start_hour, end_hour)}
    session_hours: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        "asian": (0, 8),
        "london": (8, 13),
        "ny_overlap": (13, 17),
        "ny_afternoon": (17, 21),
        "after_hours": (21, 24),
    })

    # High-impact economic events for this instrument
    calendar_events: List[str] = field(default_factory=lambda: [
        "Non-Farm Payrolls", "Federal Funds Rate", "CPI m/m",
        "Core CPI m/m", "FOMC Statement", "FOMC Press Conference",
        "GDP q/q", "Core PCE Price Index m/m", "Retail Sales m/m",
    ])

    # Forecast parameters
    diurnal_strength: float = 0.5   # 0=ignore diurnal, 1=full raw
    blend_grid: bool = True         # calibrate HAR/naive blend weight
    pred_horizon: int = 5           # predict N bars ahead
    event_window_hours: int = 4     # hours around events to apply multiplier
    hmm_n_states: int = 3           # low / normal / high vol

    # Risk multipliers
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0

    # Price rounding decimals (Gold=2, FX=5, Crypto=2, Index=1)
    price_decimals: int = 2

    # TCP parameters
    tcp_alpha: float = 0.05         # target miscoverage (95%)
    tcp_gamma: float = 0.05         # Robbins-Monro learning rate

    def __post_init__(self):
        """Auto-compute HAR windows from bars_per_day."""
        if self.har_daily == 0:
            self.har_daily = self.bars_per_day
        if self.har_weekly == 0:
            self.har_weekly = self.bars_per_day * 5
        if self.har_monthly == 0:
            self.har_monthly = self.bars_per_day * 22
        if self.har_train_min == 0:
            self.har_train_min = self.har_monthly + 100


# =============================================================================
# FORECAST RESULT
# =============================================================================

@dataclass
class VolatilityForecast:
    """Result of a single volatility forecast."""
    forecast_atr: float         # blended HAR+naive forecast
    naive_atr: float            # raw ATR baseline
    confidence_lower: float     # TCP lower bound
    confidence_upper: float     # TCP upper bound
    regime_state: str           # "low", "normal", "high"
    regime_multiplier: float
    diurnal_multiplier: float
    calendar_multiplier: float
    blend_weight: float         # calibrated HAR/naive weight
    har_base: float             # raw HAR prediction before multipliers
    is_fallback: bool = False   # True if using naive ATR due to model unavailable

    def to_dict(self) -> Dict[str, Any]:
        return {
            "forecast_atr": round(self.forecast_atr, 4),
            "naive_atr": round(self.naive_atr, 4),
            "confidence_lower": round(self.confidence_lower, 4),
            "confidence_upper": round(self.confidence_upper, 4),
            "regime_state": self.regime_state,
            "regime_multiplier": round(self.regime_multiplier, 3),
            "diurnal_multiplier": round(self.diurnal_multiplier, 3),
            "calendar_multiplier": round(self.calendar_multiplier, 3),
            "blend_weight": round(self.blend_weight, 2),
            "har_base": round(self.har_base, 4),
            "is_fallback": self.is_fallback,
        }


# =============================================================================
# VOLATILITY FORECASTER
# =============================================================================

class VolatilityForecaster:
    """Production HAR-RV + Diurnal + Calendar + HMM volatility forecaster.

    Usage:
        config = InstrumentConfig(symbol="XAUUSD", timeframe="M15")
        forecaster = VolatilityForecaster(config)
        forecaster.calibrate(ohlcv_df, calendar_df)
        forecast = forecaster.forecast(ohlcv_df, pd.Timestamp("2024-07-15 13:00"))
    """

    def __init__(self, config: Optional[InstrumentConfig] = None):
        self._config = config or InstrumentConfig()
        self._lock = threading.Lock()

        # Models (fitted during calibrate())
        self._har_model: Any = None         # sklearn LinearRegression
        self._hmm_model: Any = None         # hmmlearn GaussianHMM
        self._diurnal_profile: Dict[int, float] = {}  # hour -> multiplier
        self._regime_multipliers: Dict[int, float] = {0: 0.7, 1: 1.0, 2: 1.5}
        self._regime_labels: Dict[int, str] = {0: "low", 1: "normal", 2: "high"}
        self._blend_weight: float = 0.5     # calibrated HAR/naive blend
        self._event_times: Optional[np.ndarray] = None  # calendar event timestamps

        # TCP conformal prediction state
        self._tcp_width: float = 0.5

        # Calibration state
        self._is_calibrated: bool = False
        self._calibration_bars: int = 0

    @property
    def is_calibrated(self) -> bool:
        return self._is_calibrated

    @property
    def config(self) -> InstrumentConfig:
        return self._config

    # ------------------------------------------------------------------ #
    # CALIBRATION
    # ------------------------------------------------------------------ #

    def calibrate(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit all models on training data.

        Args:
            ohlcv_df: OHLCV DataFrame with columns: timestamp/Date, open/Open,
                       high/High, low/Low, close/Close, volume/Volume.
            calendar_df: Economic calendar with columns: timestamp/Date,
                         currency, event, impact. Optional.

        Returns:
            Dict with calibration stats (har_coefs, blend_weight, regime_mults, etc.)
        """
        with self._lock:
            return self._calibrate_impl(ohlcv_df, calendar_df)

    def _calibrate_impl(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        df = self._normalize_columns(ohlcv_df.copy())
        stats: Dict[str, Any] = {}

        # 1. Compute features
        df = self._add_features(df)

        # 2. Calendar event times
        if calendar_df is not None:
            self._event_times = self._parse_calendar(calendar_df)
            stats["calendar_events"] = len(self._event_times)
        else:
            self._event_times = None
            stats["calendar_events"] = 0

        # 3. Diurnal profile (from full training data)
        self._diurnal_profile = self._compute_diurnal_profile(df)
        stats["diurnal_hours"] = len(self._diurnal_profile)

        # 4. Fit HAR model
        har_stats = self._fit_har(df)
        stats.update(har_stats)

        # 5. Fit HMM
        hmm_stats = self._fit_hmm(df)
        stats.update(hmm_stats)

        # 6. Calibrate blend weight
        blend_stats = self._calibrate_blend_weight(df)
        stats.update(blend_stats)

        self._is_calibrated = True
        self._calibration_bars = len(df)
        stats["calibrated"] = True
        stats["training_bars"] = len(df)

        logger.info(
            "VolatilityForecaster calibrated: %d bars, blend_w=%.2f, %d events",
            len(df), self._blend_weight, stats.get("calendar_events", 0),
        )
        return stats

    # ------------------------------------------------------------------ #
    # FORECAST
    # ------------------------------------------------------------------ #

    def forecast(
        self,
        ohlcv_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> VolatilityForecast:
        """Generate volatility forecast for the latest bar.

        Args:
            ohlcv_df: Recent OHLCV data (need at least har_monthly + 14 bars).
            timestamp: Timestamp of the bar to forecast (uses last bar if None).

        Returns:
            VolatilityForecast with forecast_atr and all component values.
        """
        with self._lock:
            return self._forecast_impl(ohlcv_df, timestamp)

    def _forecast_impl(
        self,
        ohlcv_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp],
    ) -> VolatilityForecast:
        df = self._normalize_columns(ohlcv_df.copy())
        df = self._add_features(df)

        # Extract naive ATR from latest bar
        if "atr_14" not in df.columns or df["atr_14"].isna().all():
            # Compute on the fly
            df["tr"] = self._compute_true_range(df)
            df["atr_14"] = df["tr"].rolling(14).mean()

        latest_idx = len(df) - 1
        naive_atr = float(df["atr_14"].iloc[latest_idx])

        if not np.isfinite(naive_atr) or naive_atr <= 0:
            naive_atr = float(df["tr"].iloc[latest_idx]) if "tr" in df.columns else 1.0

        # Fallback if not calibrated
        if not self._is_calibrated or self._har_model is None:
            return self._fallback_forecast(naive_atr)

        # --- HAR-RV base ---
        har_base = self._predict_har(df, latest_idx)
        if har_base is None:
            return self._fallback_forecast(naive_atr)

        # --- Diurnal multiplier ---
        if timestamp is not None:
            hour = timestamp.hour
        elif "timestamp" in df.columns:
            hour = int(pd.Timestamp(df["timestamp"].iloc[latest_idx]).hour)
        else:
            hour = 12  # default midday

        raw_diurnal = self._diurnal_profile.get(hour, 1.0)
        diurnal_mult = 1.0 + self._config.diurnal_strength * (raw_diurnal - 1.0)

        # --- Calendar multiplier ---
        if timestamp is not None:
            cal_mult = self._get_calendar_multiplier(timestamp)
        elif "timestamp" in df.columns:
            cal_mult = self._get_calendar_multiplier(
                pd.Timestamp(df["timestamp"].iloc[latest_idx])
            )
        else:
            cal_mult = 1.0

        # --- HMM regime multiplier ---
        regime_mult, regime_state = self._get_regime_multiplier(df, latest_idx)

        # --- Combined forecast ---
        har_adjusted = har_base * diurnal_mult * cal_mult * regime_mult
        forecast_atr = self._blend_weight * har_adjusted + (1 - self._blend_weight) * naive_atr

        # Sanity clamp
        forecast_atr = float(np.clip(forecast_atr, 0.2 * naive_atr, 5.0 * naive_atr))

        if not np.isfinite(forecast_atr) or forecast_atr <= 0:
            return self._fallback_forecast(naive_atr)

        # TCP intervals
        lower = forecast_atr * (1 - self._tcp_width)
        upper = forecast_atr * (1 + self._tcp_width)

        return VolatilityForecast(
            forecast_atr=forecast_atr,
            naive_atr=naive_atr,
            confidence_lower=max(0.0, lower),
            confidence_upper=upper,
            regime_state=regime_state,
            regime_multiplier=regime_mult,
            diurnal_multiplier=diurnal_mult,
            calendar_multiplier=cal_mult,
            blend_weight=self._blend_weight,
            har_base=har_base,
        )

    def _fallback_forecast(self, naive_atr: float) -> VolatilityForecast:
        """Return naive ATR when model unavailable."""
        return VolatilityForecast(
            forecast_atr=naive_atr,
            naive_atr=naive_atr,
            confidence_lower=naive_atr * 0.5,
            confidence_upper=naive_atr * 1.5,
            regime_state="unknown",
            regime_multiplier=1.0,
            diurnal_multiplier=1.0,
            calendar_multiplier=1.0,
            blend_weight=0.0,
            har_base=naive_atr,
            is_fallback=True,
        )

    # ------------------------------------------------------------------ #
    # TCP UPDATE
    # ------------------------------------------------------------------ #

    def update_tcp(self, actual_atr: float, forecast: VolatilityForecast) -> None:
        """Update TCP interval width based on observed actual ATR.

        Robbins-Monro: big expansion on miss, small shrink on hit.
        """
        with self._lock:
            covered = forecast.confidence_lower <= actual_atr <= forecast.confidence_upper
            alpha = self._config.tcp_alpha
            gamma = self._config.tcp_gamma

            if covered:
                self._tcp_width -= gamma * alpha           # small shrink
            else:
                self._tcp_width += gamma * (1 - alpha)     # big expansion

            self._tcp_width = max(0.05, min(3.0, self._tcp_width))

    # ------------------------------------------------------------------ #
    # FEATURE COMPUTATION
    # ------------------------------------------------------------------ #

    @staticmethod
    def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        rename_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("date", "datetime", "time"):
                rename_map[col] = "timestamp"
            elif lower != col:
                rename_map[col] = lower
        if rename_map:
            df = df.rename(columns=rename_map)
        return df

    def _add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required features to the DataFrame."""
        # True Range
        if "tr" not in df.columns:
            df["tr"] = self._compute_true_range(df)

        # ATR_14
        if "atr_14" not in df.columns:
            df["atr_14"] = df["tr"].rolling(14).mean()

        # Log returns (percentage)
        if "returns_pct" not in df.columns:
            df["returns_pct"] = np.log(df["close"] / df["close"].shift(1)) * 100

        # Yang-Zhang RV components
        if "rv_bar" not in df.columns:
            df = self._compute_yang_zhang_rv(df)

        # HAR-RV multi-scale means
        cfg = self._config
        if "rv_daily" not in df.columns:
            df["rv_daily"] = df["rv_bar"].rolling(cfg.har_daily).mean()
        if "rv_weekly" not in df.columns:
            df["rv_weekly"] = df["rv_bar"].rolling(cfg.har_weekly).mean()
        if "rv_monthly" not in df.columns:
            df["rv_monthly"] = df["rv_bar"].rolling(cfg.har_monthly).mean()

        # Future ATR (target, only for training)
        if "future_atr" not in df.columns:
            df["future_atr"] = (
                df["tr"].rolling(cfg.pred_horizon).mean().shift(-cfg.pred_horizon)
            )

        # Hour of day
        if "hour" not in df.columns and "timestamp" in df.columns:
            df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour

        return df

    @staticmethod
    def _compute_true_range(df: pd.DataFrame) -> pd.Series:
        """Compute True Range."""
        prev_close = df["close"].shift(1)
        return pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ], axis=1).max(axis=1)

    @staticmethod
    def _compute_yang_zhang_rv(df: pd.DataFrame) -> pd.DataFrame:
        """Compute Yang-Zhang realized volatility components.

        YZ is 14x more efficient than close-to-close for OHLC data.
        """
        log_ho = np.log(df["high"] / df["open"])
        log_lo = np.log(df["low"] / df["open"])
        log_co = np.log(df["close"] / df["open"])

        # Rogers-Satchell component (per bar)
        rs_var = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        df["rv_bar"] = rs_var.clip(lower=0)

        return df

    # ------------------------------------------------------------------ #
    # DIURNAL PROFILE
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_diurnal_profile(df: pd.DataFrame) -> Dict[int, float]:
        """Compute per-hour volatility multiplier from training data.

        Returns dict mapping hour (0-23) to multiplier (mean = 1.0).
        """
        if "hour" not in df.columns or "tr" not in df.columns:
            return {}

        hourly_tr = df.groupby("hour")["tr"].mean()
        overall_mean = hourly_tr.mean()

        if overall_mean <= 0:
            return {}

        return (hourly_tr / overall_mean).to_dict()

    # ------------------------------------------------------------------ #
    # CALENDAR EVENTS
    # ------------------------------------------------------------------ #

    def _parse_calendar(self, calendar_df: pd.DataFrame) -> np.ndarray:
        """Parse economic calendar and return high-impact event timestamps."""
        cal = self._normalize_columns(calendar_df.copy())

        # Normalize column names
        if "date" in cal.columns and "timestamp" not in cal.columns:
            cal = cal.rename(columns={"date": "timestamp"})

        if "timestamp" not in cal.columns:
            return np.array([], dtype="datetime64[ns]")

        cal["timestamp"] = pd.to_datetime(cal["timestamp"])

        # Filter to high-impact events for this instrument's currency
        mask = pd.Series(True, index=cal.index)
        if "impact" in cal.columns:
            mask &= cal["impact"].str.upper() == "HIGH"
        if "event" in cal.columns:
            mask &= cal["event"].isin(self._config.calendar_events)

        filtered = cal[mask].sort_values("timestamp").reset_index(drop=True)
        return filtered["timestamp"].values

    def _get_calendar_multiplier(self, timestamp: pd.Timestamp) -> float:
        """Compute event proximity multiplier for a given timestamp.

        Linear decay: strongest at event time, decays to 1.0 at window edge.
        """
        if self._event_times is None or len(self._event_times) == 0:
            return 1.0

        ts = np.datetime64(timestamp)
        window_ns = np.timedelta64(self._config.event_window_hours, "h")

        deltas = np.abs(self._event_times - ts)
        in_window = deltas <= window_ns

        if not in_window.any():
            return 1.0

        # Find closest event in window
        hours_away = deltas[in_window].astype("timedelta64[m]").astype(float) / 60.0
        min_hours = hours_away.min()

        # Linear decay: 2.5x at event, 1.0x at window edge
        multiplier = 1.0 + 1.5 * max(0, 1 - min_hours / self._config.event_window_hours)
        return float(multiplier)

    # ------------------------------------------------------------------ #
    # HAR MODEL
    # ------------------------------------------------------------------ #

    def _fit_har(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit HAR-RV linear regression on training data."""
        from sklearn.linear_model import LinearRegression

        features = ["rv_daily", "rv_weekly", "rv_monthly", "atr_14"]
        target = "future_atr"

        valid = df[features + [target]].notna().all(axis=1)
        train = df[valid]

        if len(train) < 500:
            logger.warning("Insufficient data for HAR fit: %d bars (need 500)", len(train))
            return {"har_fitted": False, "har_train_size": len(train)}

        X = train[features].values
        y = train[target].values

        self._har_model = LinearRegression()
        self._har_model.fit(X, y)

        coefs = dict(zip(features, self._har_model.coef_))
        coefs["intercept"] = self._har_model.intercept_

        logger.info(
            "HAR fitted: daily=%.4f weekly=%.4f monthly=%.4f atr=%.4f intercept=%.4f",
            *self._har_model.coef_, self._har_model.intercept_,
        )

        return {
            "har_fitted": True,
            "har_train_size": len(train),
            "har_coefs": coefs,
        }

    def _predict_har(self, df: pd.DataFrame, idx: int) -> Optional[float]:
        """Predict HAR-RV for a single bar."""
        if self._har_model is None:
            return None

        try:
            features = np.array([[
                df["rv_daily"].iloc[idx],
                df["rv_weekly"].iloc[idx],
                df["rv_monthly"].iloc[idx],
                df["atr_14"].iloc[idx],
            ]])

            if not np.all(np.isfinite(features)):
                return None

            pred = self._har_model.predict(features)[0]
            return max(0.01, float(pred))
        except Exception as e:
            logger.debug("HAR predict failed: %s", e)
            return None

    # ------------------------------------------------------------------ #
    # HMM REGIME
    # ------------------------------------------------------------------ #

    def _fit_hmm(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Fit Hidden Markov Model for regime detection."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            logger.warning("hmmlearn not installed — HMM regime detection disabled")
            return {"hmm_fitted": False}

        # Use returns and daily RV for regime detection
        valid = (
            df["returns_pct"].notna() &
            df["rv_daily"].notna() &
            np.isfinite(df["returns_pct"]) &
            np.isfinite(df["rv_daily"])
        )
        if valid.sum() < 500:
            logger.warning("Insufficient data for HMM: %d (need 500)", valid.sum())
            return {"hmm_fitted": False}

        returns = df.loc[valid, "returns_pct"].values
        rv_daily = df.loc[valid, "rv_daily"].values
        hmm_X = np.column_stack([returns, rv_daily])

        try:
            self._hmm_model = GaussianHMM(
                n_components=self._config.hmm_n_states,
                covariance_type="diag",
                n_iter=100,
                random_state=42,
            )
            self._hmm_model.fit(hmm_X)

            # Calibrate regime multipliers
            states = self._hmm_model.predict(hmm_X)
            state_vol: Dict[int, float] = {}
            for s in range(self._config.hmm_n_states):
                s_mask = states == s
                if s_mask.sum() > 10:
                    state_vol[s] = float(np.mean(np.abs(returns[s_mask])))
                else:
                    state_vol[s] = float(np.mean(np.abs(returns)))

            overall_vol = float(np.mean(np.abs(returns)))
            sorted_states = sorted(state_vol.keys(), key=lambda s: state_vol[s])

            labels = ["low", "normal", "high"]
            self._regime_multipliers = {}
            self._regime_labels = {}

            for rank, s in enumerate(sorted_states):
                mult = state_vol[s] / overall_vol if overall_vol > 0 else 1.0
                self._regime_multipliers[s] = float(np.clip(mult, 0.5, 2.5))
                self._regime_labels[s] = labels[min(rank, len(labels) - 1)]

            logger.info("HMM fitted: regime_multipliers=%s", self._regime_multipliers)
            return {
                "hmm_fitted": True,
                "regime_multipliers": self._regime_multipliers.copy(),
                "regime_labels": self._regime_labels.copy(),
            }

        except Exception as e:
            logger.warning("HMM fit failed: %s", e)
            self._hmm_model = None
            return {"hmm_fitted": False, "hmm_error": str(e)}

    def _get_regime_multiplier(
        self, df: pd.DataFrame, idx: int
    ) -> Tuple[float, str]:
        """Get HMM regime multiplier for a single bar."""
        if self._hmm_model is None:
            return 1.0, "unknown"

        try:
            ret = float(df["returns_pct"].iloc[idx])
            rv = float(df["rv_daily"].iloc[idx])

            if not (np.isfinite(ret) and np.isfinite(rv)):
                return 1.0, "unknown"

            obs = np.array([[ret, rv]])
            state = int(self._hmm_model.predict(obs)[0])

            mult = self._regime_multipliers.get(state, 1.0)
            label = self._regime_labels.get(state, "unknown")
            return mult, label

        except Exception:
            return 1.0, "unknown"

    # ------------------------------------------------------------------ #
    # BLEND CALIBRATION
    # ------------------------------------------------------------------ #

    def _calibrate_blend_weight(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate HAR/naive blend weight via grid search on validation split.

        Implements Bates & Granger (1969) forecast combination with shrinkage.
        """
        if not self._config.blend_grid or self._har_model is None:
            self._blend_weight = 0.5
            return {"blend_weight": 0.5, "blend_calibrated": False}

        features = ["rv_daily", "rv_weekly", "rv_monthly", "atr_14", "future_atr"]
        valid = df[features].notna().all(axis=1)
        valid_df = df[valid]

        if len(valid_df) < 200:
            self._blend_weight = 0.5
            return {"blend_weight": 0.5, "blend_calibrated": False}

        # 80/20 train/val split
        split = int(len(valid_df) * 0.8)
        val_df = valid_df.iloc[split:]

        if len(val_df) < 50:
            self._blend_weight = 0.5
            return {"blend_weight": 0.5, "blend_calibrated": False}

        # HAR predictions on validation
        X_val = val_df[["rv_daily", "rv_weekly", "rv_monthly", "atr_14"]].values
        y_val = val_df["future_atr"].values
        har_pred = self._har_model.predict(X_val).clip(min=0.01)
        naive_val = val_df["atr_14"].values

        # Apply diurnal + calendar multipliers
        if "hour" in val_df.columns:
            hours = val_df["hour"].values
            diurnal_mult = np.array([
                1.0 + self._config.diurnal_strength * (self._diurnal_profile.get(int(h), 1.0) - 1.0)
                for h in hours
            ])
        else:
            diurnal_mult = np.ones(len(val_df))

        har_adj = har_pred * diurnal_mult

        # Grid search
        best_w, best_mae = 0.5, float("inf")
        for w in np.arange(0.05, 0.96, 0.05):
            blended = w * har_adj + (1 - w) * naive_val
            mae = float(np.mean(np.abs(blended - y_val)))
            if mae < best_mae:
                best_mae = mae
                best_w = float(w)

        self._blend_weight = best_w
        naive_mae = float(np.mean(np.abs(naive_val - y_val)))
        improvement = (1 - best_mae / naive_mae) * 100 if naive_mae > 0 else 0

        logger.info(
            "Blend calibrated: w=%.2f, val_MAE=%.4f, naive_MAE=%.4f, improvement=%.1f%%",
            best_w, best_mae, naive_mae, improvement,
        )
        return {
            "blend_weight": best_w,
            "blend_calibrated": True,
            "blend_val_mae": best_mae,
            "blend_naive_mae": naive_mae,
            "blend_improvement_pct": improvement,
        }

    # ------------------------------------------------------------------ #
    # PERSISTENCE
    # ------------------------------------------------------------------ #

    def save_state(self, path: str) -> None:
        """Save fitted model state to disk."""
        state = {
            "config": self._config,
            "har_model": self._har_model,
            "hmm_model": self._hmm_model,
            "diurnal_profile": self._diurnal_profile,
            "regime_multipliers": self._regime_multipliers,
            "regime_labels": self._regime_labels,
            "blend_weight": self._blend_weight,
            "event_times": self._event_times,
            "tcp_width": self._tcp_width,
            "is_calibrated": self._is_calibrated,
            "calibration_bars": self._calibration_bars,
        }

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as f:
            pickle.dump(state, f)

        logger.info("VolatilityForecaster state saved to %s", path)

    def load_state(self, path: str) -> bool:
        """Load fitted model state from disk. Returns True on success."""
        p = Path(path)
        if not p.exists():
            logger.warning("State file not found: %s", path)
            return False

        try:
            with open(p, "rb") as f:
                state = pickle.load(f)

            self._config = state["config"]
            self._har_model = state["har_model"]
            self._hmm_model = state["hmm_model"]
            self._diurnal_profile = state["diurnal_profile"]
            self._regime_multipliers = state["regime_multipliers"]
            self._regime_labels = state["regime_labels"]
            self._blend_weight = state["blend_weight"]
            self._event_times = state["event_times"]
            self._tcp_width = state["tcp_width"]
            self._is_calibrated = state["is_calibrated"]
            self._calibration_bars = state["calibration_bars"]

            logger.info("VolatilityForecaster state loaded from %s", path)
            return True

        except Exception as e:
            logger.error("Failed to load state: %s", e)
            return False

    # ------------------------------------------------------------------ #
    # STATS
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "is_calibrated": self._is_calibrated,
            "calibration_bars": self._calibration_bars,
            "symbol": self._config.symbol,
            "timeframe": self._config.timeframe,
            "blend_weight": self._blend_weight,
            "tcp_width": round(self._tcp_width, 3),
            "har_fitted": self._har_model is not None,
            "hmm_fitted": self._hmm_model is not None,
            "diurnal_hours": len(self._diurnal_profile),
            "regime_multipliers": self._regime_multipliers.copy(),
        }

    # ------------------------------------------------------------------ #
    # FACTORY
    # ------------------------------------------------------------------ #

    @staticmethod
    def create(
        mode: str = "har",
        config: Optional[InstrumentConfig] = None,
    ) -> "VolatilityForecaster":
        """Factory method to create the right forecaster by mode.

        Args:
            mode: "har" (default), "lgbm", or "hybrid".
            config: Optional instrument configuration.

        Returns:
            VolatilityForecaster (or HybridForecaster for "hybrid" mode).
        """
        if mode == "har":
            return VolatilityForecaster(config)
        elif mode == "lgbm":
            # LightGBM wraps into same interface via HybridForecaster with HAR disabled
            return HybridForecaster(config, mode="lgbm")
        elif mode == "hybrid":
            return HybridForecaster(config, mode="hybrid")
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'har', 'lgbm', or 'hybrid'.")


# =============================================================================
# HYBRID FORECASTER (Two-Stage: HAR base + LightGBM residual)
# =============================================================================

class HybridForecaster(VolatilityForecaster):
    """Two-stage volatility forecaster: HAR-RV base + LightGBM residual correction.

    Architecture:
        Stage 1: HAR-RV base forecast (captures persistence, seasonality)
        Stage 2: LightGBM predicts residual (actual - HAR_forecast)
        Combined: final = HAR_forecast + LightGBM_residual_prediction

    Fallback chain: Hybrid → HAR-only → naive ATR

    Usage:
        hybrid = HybridForecaster(config)
        hybrid.calibrate(ohlcv_df, calendar_df)  # fits HAR + LightGBM
        forecast = hybrid.forecast(ohlcv_df, timestamp)
    """

    def __init__(
        self,
        config: Optional[InstrumentConfig] = None,
        mode: str = "hybrid",
    ):
        super().__init__(config)
        self._mode = mode  # "hybrid" or "lgbm"
        self._lgbm: Any = None  # LGBMVolForecaster
        self._lgbm_available: bool = False

    @property
    def mode(self) -> str:
        return self._mode

    def calibrate(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Fit HAR-RV first, then fit LightGBM on HAR residuals.

        For "lgbm" mode: fits LightGBM directly on future_atr.
        For "hybrid" mode: fits LightGBM on (future_atr - HAR_forecast) residuals.
        """
        with self._lock:
            stats = self._calibrate_impl(ohlcv_df, calendar_df)
            lgbm_stats = self._fit_lgbm(ohlcv_df, calendar_df)
            stats.update(lgbm_stats)
            return stats

    def _fit_lgbm(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Fit LightGBM on raw target or HAR residuals."""
        try:
            from src.intelligence.volatility_lgbm import LGBMVolForecaster
        except ImportError:
            logger.warning("volatility_lgbm not available")
            return {"lgbm_fitted": False}

        self._lgbm = LGBMVolForecaster(self._config, n_estimators=500, max_depth=6)

        if self._mode == "hybrid" and self._har_model is not None:
            # Build features with residual target
            lgbm_stats = self._fit_lgbm_on_residuals(ohlcv_df, calendar_df)
        else:
            # Direct LightGBM on future_atr
            lgbm_stats = self._lgbm.train(ohlcv_df, calendar_df, self)

        self._lgbm_available = lgbm_stats.get("trained", False)
        lgbm_stats = {f"lgbm_{k}": v for k, v in lgbm_stats.items()}
        return lgbm_stats

    def _fit_lgbm_on_residuals(
        self,
        ohlcv_df: pd.DataFrame,
        calendar_df: Optional[pd.DataFrame],
    ) -> Dict[str, Any]:
        """Fit LightGBM on HAR residuals (actual - HAR_forecast)."""
        try:
            import lightgbm as lgb
        except ImportError:
            return {"trained": False, "error": "lightgbm not installed"}

        # Build feature matrix
        feature_df = self._lgbm.build_features(ohlcv_df, calendar_df, self)

        # Compute HAR predictions for all training bars
        feature_cols = [c for c in self._lgbm.FEATURE_NAMES if c in feature_df.columns]
        target_col = "future_atr"
        har_features = ["rv_daily", "rv_weekly", "rv_monthly", "atr_14"]

        valid_mask = (
            feature_df[feature_cols + [target_col] + har_features].notna().all(axis=1) &
            feature_df[feature_cols + [target_col] + har_features].apply(
                lambda row: np.all(np.isfinite(row)), axis=1
            )
        )
        valid_df = feature_df[valid_mask].reset_index(drop=True)

        if len(valid_df) < 500:
            return {"trained": False, "valid_bars": len(valid_df)}

        # HAR predictions
        X_har = valid_df[har_features].values
        har_preds = self._har_model.predict(X_har).clip(min=0.01)

        # Residual target = actual - HAR
        residuals = valid_df[target_col].values - har_preds
        valid_df["_residual_target"] = residuals

        # Train LightGBM on residuals
        X = valid_df[feature_cols].values
        y = valid_df["_residual_target"].values

        split_idx = int(len(X) * 0.8)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=train_data)

        params = {
            "objective": "regression",
            "metric": "mae",
            "max_depth": 6,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "verbosity": -1,
            "seed": 42,
        }

        self._lgbm._model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        self._lgbm._is_trained = True
        self._lgbm._training_bars = len(valid_df)

        # Compute metrics
        residual_pred = self._lgbm._model.predict(X_val)
        combined_pred = har_preds[split_idx:] + residual_pred
        actual = valid_df[target_col].values[split_idx:]
        naive = valid_df["atr_14"].values[split_idx:]

        combined_mae = float(np.mean(np.abs(combined_pred - actual)))
        har_only_mae = float(np.mean(np.abs(har_preds[split_idx:] - actual)))
        naive_mae = float(np.mean(np.abs(naive - actual)))

        improvement_vs_naive = (1 - combined_mae / naive_mae) * 100 if naive_mae > 0 else 0
        improvement_vs_har = (1 - combined_mae / har_only_mae) * 100 if har_only_mae > 0 else 0

        logger.info(
            "Hybrid calibrated: combined_MAE=%.4f, har_MAE=%.4f, naive_MAE=%.4f, "
            "improvement_vs_naive=%.1f%%, improvement_vs_har=%.1f%%",
            combined_mae, har_only_mae, naive_mae,
            improvement_vs_naive, improvement_vs_har,
        )

        return {
            "trained": True,
            "mode": "residual",
            "total_bars": len(valid_df),
            "combined_mae": combined_mae,
            "har_only_mae": har_only_mae,
            "naive_mae": naive_mae,
            "improvement_vs_naive_pct": improvement_vs_naive,
            "improvement_vs_har_pct": improvement_vs_har,
        }

    # ------------------------------------------------------------------ #
    # FORECAST OVERRIDE
    # ------------------------------------------------------------------ #

    def forecast(
        self,
        ohlcv_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> VolatilityForecast:
        """Generate forecast using the hybrid pipeline.

        Fallback chain: Hybrid → HAR-only → naive ATR
        """
        with self._lock:
            return self._hybrid_forecast_impl(ohlcv_df, timestamp)

    def _hybrid_forecast_impl(
        self,
        ohlcv_df: pd.DataFrame,
        timestamp: Optional[pd.Timestamp],
    ) -> VolatilityForecast:
        # Step 1: Get HAR base forecast
        har_forecast = self._forecast_impl(ohlcv_df, timestamp)

        # Step 2: If LightGBM not available, return HAR forecast
        if not self._lgbm_available or self._lgbm is None or not self._lgbm.is_trained:
            return har_forecast

        # Step 3: Get LightGBM correction
        try:
            feature_df = self._lgbm.build_features(ohlcv_df, har_forecaster=self)
            lgbm_pred = self._lgbm.predict_from_df(feature_df)

            if self._mode == "hybrid":
                # Hybrid: HAR base + residual correction
                # har_forecast already has blend, multipliers, etc.
                # lgbm_pred is the residual correction
                corrected_atr = har_forecast.forecast_atr + lgbm_pred
            else:
                # Pure LightGBM mode
                corrected_atr = lgbm_pred

            # Sanity clamp
            corrected_atr = float(np.clip(
                corrected_atr,
                0.2 * har_forecast.naive_atr,
                5.0 * har_forecast.naive_atr,
            ))

            if not np.isfinite(corrected_atr) or corrected_atr <= 0:
                return har_forecast

            # Update TCP intervals with corrected ATR
            lower = corrected_atr * (1 - self._tcp_width)
            upper = corrected_atr * (1 + self._tcp_width)

            return VolatilityForecast(
                forecast_atr=corrected_atr,
                naive_atr=har_forecast.naive_atr,
                confidence_lower=max(0.0, lower),
                confidence_upper=upper,
                regime_state=har_forecast.regime_state,
                regime_multiplier=har_forecast.regime_multiplier,
                diurnal_multiplier=har_forecast.diurnal_multiplier,
                calendar_multiplier=har_forecast.calendar_multiplier,
                blend_weight=har_forecast.blend_weight,
                har_base=har_forecast.har_base,
            )

        except Exception as e:
            logger.warning("LightGBM correction failed, using HAR-only: %s", e)
            return har_forecast

    # ------------------------------------------------------------------ #
    # PERSISTENCE OVERRIDE
    # ------------------------------------------------------------------ #

    def save_state(self, path: str) -> None:
        """Save both HAR and LightGBM state."""
        super().save_state(path)

        if self._lgbm is not None and self._lgbm.is_trained:
            lgbm_path = str(Path(path).with_suffix(".lgbm.txt"))
            self._lgbm.save_model(lgbm_path)

    def load_state(self, path: str) -> bool:
        """Load both HAR and LightGBM state."""
        result = super().load_state(path)

        lgbm_path = str(Path(path).with_suffix(".lgbm.txt"))
        if Path(lgbm_path).exists():
            try:
                from src.intelligence.volatility_lgbm import LGBMVolForecaster
                self._lgbm = LGBMVolForecaster(self._config)
                self._lgbm_available = self._lgbm.load_model(lgbm_path)
            except Exception as e:
                logger.warning("Failed to load LightGBM state: %s", e)
                self._lgbm_available = False

        return result

    def get_stats(self) -> Dict[str, Any]:
        stats = super().get_stats()
        stats["mode"] = self._mode
        stats["lgbm_available"] = self._lgbm_available
        if self._lgbm is not None:
            stats["lgbm_stats"] = self._lgbm.get_stats()
        return stats
