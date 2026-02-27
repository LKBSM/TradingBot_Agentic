"""
Sprint 7: Incremental Feature Engine — O(1) per-bar TA indicator updates.

Replaces batch DataFrame reprocessing with streaming Wilder/EMA smoothing.
Each indicator maintains its own running state and updates with a single new
bar of data, matching batch computation to within floating-point tolerance.

Supported indicators:
  - RSI (Wilder smoothing)
  - MACD (EMA fast/slow + signal)
  - Bollinger Bands (running mean/variance)
  - ATR (Wilder smoothing on True Range)
"""

import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# INDIVIDUAL INCREMENTAL INDICATORS
# =============================================================================

class IncrementalRSI:
    """Wilder-smoothed RSI with O(1) per-bar update."""

    __slots__ = ('_period', '_avg_gain', '_avg_loss', '_prev_close',
                 '_warmup_count', '_value')

    def __init__(self, period: int = 10):
        self._period = period
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self._prev_close: Optional[float] = None
        self._warmup_count = 0
        self._value = 50.0  # Neutral default

    def seed(self, closes: np.ndarray) -> float:
        """Initialize from historical close prices (requires >= period+1 values)."""
        if len(closes) < self._period + 1:
            raise ValueError(
                f"RSI seed requires >= {self._period + 1} closes, got {len(closes)}"
            )
        changes = np.diff(closes)
        gains = np.maximum(changes, 0.0)
        losses = np.maximum(-changes, 0.0)

        # First average: simple mean over period
        self._avg_gain = float(np.mean(gains[:self._period]))
        self._avg_loss = float(np.mean(losses[:self._period]))
        self._warmup_count = self._period

        # Wilder smoothing for remaining bars
        for i in range(self._period, len(changes)):
            self._avg_gain = (self._avg_gain * (self._period - 1) + gains[i]) / self._period
            self._avg_loss = (self._avg_loss * (self._period - 1) + losses[i]) / self._period
            self._warmup_count += 1

        self._prev_close = float(closes[-1])
        self._value = self._compute_rsi()
        return self._value

    def update(self, new_close: float) -> float:
        """Update RSI with a single new close price. O(1)."""
        if self._prev_close is None:
            self._prev_close = new_close
            return self._value

        change = new_close - self._prev_close
        gain = max(0.0, change)
        loss = max(0.0, -change)

        self._avg_gain = (self._avg_gain * (self._period - 1) + gain) / self._period
        self._avg_loss = (self._avg_loss * (self._period - 1) + loss) / self._period
        self._warmup_count += 1
        self._prev_close = new_close

        self._value = self._compute_rsi()
        return self._value

    def _compute_rsi(self) -> float:
        if self._avg_loss < 1e-10:
            return 100.0 if self._avg_gain > 1e-10 else 50.0
        rs = self._avg_gain / self._avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @property
    def value(self) -> float:
        return self._value

    @property
    def is_warm(self) -> bool:
        return self._warmup_count >= self._period


class IncrementalMACD:
    """EMA-based MACD with O(1) per-bar update."""

    __slots__ = ('_fast_period', '_slow_period', '_signal_period',
                 '_ema_fast', '_ema_slow', '_ema_signal',
                 '_alpha_fast', '_alpha_slow', '_alpha_signal',
                 '_warmup_count', '_macd_line', '_macd_signal', '_macd_diff')

    def __init__(self, fast: int = 8, slow: int = 17, signal: int = 9):
        self._fast_period = fast
        self._slow_period = slow
        self._signal_period = signal

        self._alpha_fast = 2.0 / (fast + 1)
        self._alpha_slow = 2.0 / (slow + 1)
        self._alpha_signal = 2.0 / (signal + 1)

        self._ema_fast: Optional[float] = None
        self._ema_slow: Optional[float] = None
        self._ema_signal: Optional[float] = None
        self._warmup_count = 0
        self._macd_line = 0.0
        self._macd_signal = 0.0
        self._macd_diff = 0.0

    def seed(self, closes: np.ndarray) -> Dict[str, float]:
        """Initialize from historical close prices."""
        min_required = self._slow_period + self._signal_period
        if len(closes) < min_required:
            raise ValueError(
                f"MACD seed requires >= {min_required} closes, got {len(closes)}"
            )

        # Initialize EMAs with first value
        self._ema_fast = float(closes[0])
        self._ema_slow = float(closes[0])

        # Build up MACD line history for signal EMA
        macd_history = []
        for close in closes:
            self._ema_fast = self._alpha_fast * close + (1 - self._alpha_fast) * self._ema_fast
            self._ema_slow = self._alpha_slow * close + (1 - self._alpha_slow) * self._ema_slow
            macd_history.append(self._ema_fast - self._ema_slow)

        # Initialize signal EMA
        self._ema_signal = float(macd_history[0])
        for macd_val in macd_history:
            self._ema_signal = self._alpha_signal * macd_val + (1 - self._alpha_signal) * self._ema_signal

        self._macd_line = macd_history[-1]
        self._macd_signal = self._ema_signal
        self._macd_diff = self._macd_line - self._macd_signal
        self._warmup_count = len(closes)

        return self.values

    def update(self, new_close: float) -> Dict[str, float]:
        """Update MACD with a single new close price. O(1)."""
        if self._ema_fast is None:
            self._ema_fast = new_close
            self._ema_slow = new_close
            self._ema_signal = 0.0
            self._warmup_count = 1
            return self.values

        self._ema_fast = self._alpha_fast * new_close + (1 - self._alpha_fast) * self._ema_fast
        self._ema_slow = self._alpha_slow * new_close + (1 - self._alpha_slow) * self._ema_slow

        self._macd_line = self._ema_fast - self._ema_slow
        self._ema_signal = self._alpha_signal * self._macd_line + (1 - self._alpha_signal) * self._ema_signal
        self._macd_signal = self._ema_signal
        self._macd_diff = self._macd_line - self._macd_signal
        self._warmup_count += 1

        return self.values

    @property
    def values(self) -> Dict[str, float]:
        return {
            'macd_line': self._macd_line,
            'macd_signal': self._macd_signal,
            'macd_diff': self._macd_diff,
        }

    @property
    def is_warm(self) -> bool:
        return self._warmup_count >= self._slow_period + self._signal_period


class IncrementalBollingerBands:
    """Running mean/variance Bollinger Bands with O(1) update."""

    __slots__ = ('_period', '_num_std', '_buffer', '_buf_idx', '_buf_full',
                 '_running_sum', '_running_sq_sum', '_count',
                 '_upper', '_middle', '_lower')

    def __init__(self, period: int = 20, num_std: float = 2.0):
        self._period = period
        self._num_std = num_std
        self._buffer = np.zeros(period, dtype=np.float64)
        self._buf_idx = 0
        self._buf_full = False
        self._running_sum = 0.0
        self._running_sq_sum = 0.0
        self._count = 0
        self._upper = 0.0
        self._middle = 0.0
        self._lower = 0.0

    def seed(self, closes: np.ndarray) -> Dict[str, float]:
        """Initialize from historical close prices."""
        if len(closes) < self._period:
            raise ValueError(
                f"BB seed requires >= {self._period} closes, got {len(closes)}"
            )
        # Fill the buffer with the last `period` values
        seed_data = closes[-self._period:]
        self._buffer[:] = seed_data
        self._buf_idx = 0
        self._buf_full = True
        self._running_sum = float(np.sum(seed_data))
        self._running_sq_sum = float(np.sum(seed_data ** 2))
        self._count = self._period
        self._compute()

        # Process any remaining values after the initial period
        # (the seed already placed the last `period` values in the buffer)
        return self.values

    def update(self, new_close: float) -> Dict[str, float]:
        """Update Bollinger Bands with a single new close. O(1)."""
        if self._buf_full:
            # Remove oldest value from running sums
            old_val = self._buffer[self._buf_idx]
            self._running_sum -= old_val
            self._running_sq_sum -= old_val * old_val
        else:
            self._count += 1

        # Add new value
        self._buffer[self._buf_idx] = new_close
        self._running_sum += new_close
        self._running_sq_sum += new_close * new_close

        self._buf_idx = (self._buf_idx + 1) % self._period
        if self._buf_idx == 0:
            self._buf_full = True

        self._compute()
        return self.values

    def _compute(self) -> None:
        n = self._period if self._buf_full else self._count
        if n < 2:
            return
        mean = self._running_sum / n
        variance = (self._running_sq_sum / n) - (mean * mean)
        # Clamp to avoid negative variance from floating-point drift
        std = np.sqrt(max(0.0, variance))
        self._middle = mean
        self._upper = mean + self._num_std * std
        self._lower = mean - self._num_std * std

    @property
    def values(self) -> Dict[str, float]:
        return {
            'bb_upper': self._upper,
            'bb_middle': self._middle,
            'bb_lower': self._lower,
        }

    @property
    def is_warm(self) -> bool:
        return self._buf_full


class IncrementalATR:
    """Wilder-smoothed ATR with O(1) per-bar update."""

    __slots__ = ('_period', '_atr', '_prev_close', '_warmup_count')

    def __init__(self, period: int = 7):
        self._period = period
        self._atr: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._warmup_count = 0

    def seed(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
        """Initialize from historical OHLC data."""
        if len(highs) < self._period + 1:
            raise ValueError(
                f"ATR seed requires >= {self._period + 1} bars, got {len(highs)}"
            )
        # Compute True Range series
        tr = np.maximum(
            highs[1:] - lows[1:],
            np.maximum(
                np.abs(highs[1:] - closes[:-1]),
                np.abs(lows[1:] - closes[:-1])
            )
        )

        # First ATR: simple mean of first `period` TRs
        self._atr = float(np.mean(tr[:self._period]))
        self._warmup_count = self._period

        # Wilder smoothing for remaining
        for i in range(self._period, len(tr)):
            self._atr = (self._atr * (self._period - 1) + tr[i]) / self._period
            self._warmup_count += 1

        self._prev_close = float(closes[-1])
        return self._atr

    def update(self, high: float, low: float, close: float) -> float:
        """Update ATR with a single new bar. O(1)."""
        if self._prev_close is None:
            self._prev_close = close
            self._atr = high - low
            self._warmup_count = 1
            return self._atr

        # True Range
        tr = max(
            high - low,
            abs(high - self._prev_close),
            abs(low - self._prev_close)
        )

        if self._atr is None:
            self._atr = tr
        else:
            # Wilder smoothing
            self._atr = (self._atr * (self._period - 1) + tr) / self._period

        self._prev_close = close
        self._warmup_count += 1
        return self._atr

    @property
    def value(self) -> float:
        return self._atr if self._atr is not None else 0.0

    @property
    def is_warm(self) -> bool:
        return self._warmup_count >= self._period


# =============================================================================
# COMPOSITE ENGINE
# =============================================================================

@dataclass
class IncrementalFeatureConfig:
    """Configuration for IncrementalFeatureEngine."""
    rsi_period: int = 10
    macd_fast: int = 8
    macd_slow: int = 17
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 7


class IncrementalFeatureEngine:
    """
    Composite incremental TA feature engine.

    Maintains RSI, MACD, Bollinger Bands, and ATR indicators that update
    in O(1) per new bar instead of reprocessing the entire DataFrame.

    Usage:
        engine = IncrementalFeatureEngine(config)
        engine.seed(highs, lows, closes)  # Warm up from history

        # Per bar:
        features = engine.update(high, low, close)
    """

    def __init__(self, config: Optional[IncrementalFeatureConfig] = None):
        if config is None:
            config = IncrementalFeatureConfig()
        self._config = config

        self.rsi = IncrementalRSI(period=config.rsi_period)
        self.macd = IncrementalMACD(
            fast=config.macd_fast, slow=config.macd_slow, signal=config.macd_signal
        )
        self.bb = IncrementalBollingerBands(
            period=config.bb_period, num_std=config.bb_std
        )
        self.atr = IncrementalATR(period=config.atr_period)
        self._seeded = False

    def seed(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> Dict[str, float]:
        """
        Warm up all indicators from historical OHLC data.

        Args:
            highs: Array of high prices
            lows: Array of low prices
            closes: Array of close prices

        Returns:
            Dict of all current indicator values
        """
        self.rsi.seed(closes)
        self.macd.seed(closes)
        self.bb.seed(closes)
        self.atr.seed(highs, lows, closes)
        self._seeded = True
        return self.current_values

    def update(self, high: float, low: float, close: float) -> Dict[str, float]:
        """
        Update all indicators with a single new bar. O(1) total.

        Returns:
            Dict of all current indicator values
        """
        self.rsi.update(close)
        self.macd.update(close)
        self.bb.update(close)
        self.atr.update(high, low, close)
        return self.current_values

    @property
    def current_values(self) -> Dict[str, float]:
        """Get all current indicator values as a flat dict."""
        vals = {
            'RSI': self.rsi.value,
            'ATR': self.atr.value,
        }
        vals.update(self.macd.values)
        vals.update(self.bb.values)
        return vals

    @property
    def is_warm(self) -> bool:
        """True when all indicators have enough data for reliable output."""
        return (self.rsi.is_warm and self.macd.is_warm
                and self.bb.is_warm and self.atr.is_warm)

    @property
    def feature_names(self) -> list:
        """Ordered list of feature names produced by this engine."""
        return ['RSI', 'ATR', 'macd_line', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_middle', 'bb_lower']

    def to_array(self) -> np.ndarray:
        """Get current values as a numpy array (same order as feature_names)."""
        vals = self.current_values
        return np.array([vals[k] for k in self.feature_names], dtype=np.float64)
