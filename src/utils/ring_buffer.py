# =============================================================================
# RING BUFFER - Fixed-Size Circular Buffer for Time Series Data
# =============================================================================
# Memory-efficient storage for price history and other time series data.
#
# Features:
#   - O(1) append and access
#   - Fixed memory footprint
#   - NumPy-backed for fast vectorized operations
#   - Thread-safe option
#   - Multiple data type support
#
# Usage:
#   buffer = RingBuffer(max_size=10000)
#   buffer.append(price)
#   recent_prices = buffer.get_last(100)
#   all_prices = buffer.to_array()
#
# =============================================================================

import numpy as np
import threading
from typing import Optional, Union, List, Tuple, Any
from dataclasses import dataclass


# =============================================================================
# BASIC RING BUFFER
# =============================================================================

class RingBuffer:
    """
    Fixed-size circular buffer backed by NumPy array.

    Memory stays constant regardless of how many values are appended.
    Old values are automatically overwritten when buffer is full.

    Example:
        buffer = RingBuffer(max_size=1000)

        for price in price_stream:
            buffer.append(price)

        # Get last 100 prices
        recent = buffer.get_last(100)

        # Get all prices in order
        all_prices = buffer.to_array()

        # Vectorized operations
        mean = buffer.mean()
        std = buffer.std()
    """

    def __init__(
        self,
        max_size: int,
        dtype: np.dtype = np.float64,
        fill_value: float = np.nan
    ):
        """
        Initialize ring buffer.

        Args:
            max_size: Maximum number of elements
            dtype: NumPy data type
            fill_value: Initial fill value
        """
        self.max_size = max_size
        self.dtype = dtype
        self._buffer = np.full(max_size, fill_value, dtype=dtype)
        self._index = 0  # Next write position
        self._count = 0  # Number of values written (up to max_size)

    def append(self, value: float) -> None:
        """Append a value to the buffer. O(1) operation."""
        self._buffer[self._index] = value
        self._index = (self._index + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)

    def extend(self, values: Union[List[float], np.ndarray]) -> None:
        """Append multiple values efficiently."""
        values = np.asarray(values, dtype=self.dtype)
        n = len(values)

        if n >= self.max_size:
            # Just take the last max_size values
            self._buffer[:] = values[-self.max_size:]
            self._index = 0
            self._count = self.max_size
        else:
            # Calculate positions
            end_idx = self._index + n
            if end_idx <= self.max_size:
                self._buffer[self._index:end_idx] = values
            else:
                # Wrap around
                first_part = self.max_size - self._index
                self._buffer[self._index:] = values[:first_part]
                self._buffer[:n - first_part] = values[first_part:]

            self._index = end_idx % self.max_size
            self._count = min(self._count + n, self.max_size)

    def get_last(self, n: int) -> np.ndarray:
        """
        Get the last n values in chronological order.

        Args:
            n: Number of values to retrieve

        Returns:
            NumPy array of last n values
        """
        n = min(n, self._count)
        if n == 0:
            return np.array([], dtype=self.dtype)

        # Calculate start position
        start = (self._index - n) % self.max_size

        if start < self._index:
            return self._buffer[start:self._index].copy()
        else:
            # Wrap around
            return np.concatenate([
                self._buffer[start:],
                self._buffer[:self._index]
            ])

    def to_array(self) -> np.ndarray:
        """Get all values in chronological order."""
        return self.get_last(self._count)

    def __getitem__(self, idx: int) -> float:
        """Get item by index (0 = oldest, -1 = newest)."""
        if self._count == 0:
            raise IndexError("Buffer is empty")

        if idx < 0:
            idx = self._count + idx

        if idx < 0 or idx >= self._count:
            raise IndexError(f"Index {idx} out of range [0, {self._count})")

        # Calculate actual position
        actual_idx = (self._index - self._count + idx) % self.max_size
        return self._buffer[actual_idx]

    def __len__(self) -> int:
        """Return current number of elements."""
        return self._count

    @property
    def is_full(self) -> bool:
        """Check if buffer is at maximum capacity."""
        return self._count == self.max_size

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self._count == 0

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.fill(np.nan)
        self._index = 0
        self._count = 0

    # =========================================================================
    # VECTORIZED OPERATIONS
    # =========================================================================

    def mean(self) -> float:
        """Calculate mean of buffer values."""
        if self._count == 0:
            return np.nan
        return np.nanmean(self.to_array())

    def std(self) -> float:
        """Calculate standard deviation."""
        if self._count < 2:
            return np.nan
        return np.nanstd(self.to_array())

    def min(self) -> float:
        """Get minimum value."""
        if self._count == 0:
            return np.nan
        return np.nanmin(self.to_array())

    def max(self) -> float:
        """Get maximum value."""
        if self._count == 0:
            return np.nan
        return np.nanmax(self.to_array())

    def sum(self) -> float:
        """Calculate sum."""
        if self._count == 0:
            return 0.0
        return np.nansum(self.to_array())

    def percentile(self, q: float) -> float:
        """Calculate percentile (0-100)."""
        if self._count == 0:
            return np.nan
        return np.nanpercentile(self.to_array(), q)

    def returns(self) -> np.ndarray:
        """Calculate simple returns."""
        arr = self.to_array()
        if len(arr) < 2:
            return np.array([])
        return np.diff(arr) / arr[:-1]

    def log_returns(self) -> np.ndarray:
        """Calculate log returns."""
        arr = self.to_array()
        if len(arr) < 2:
            return np.array([])
        return np.diff(np.log(arr))

    def rolling_mean(self, window: int) -> np.ndarray:
        """Calculate rolling mean."""
        arr = self.to_array()
        if len(arr) < window:
            return np.array([])

        # Efficient convolution-based rolling mean
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='valid')

    def rolling_std(self, window: int) -> np.ndarray:
        """Calculate rolling standard deviation."""
        arr = self.to_array()
        if len(arr) < window:
            return np.array([])

        # Use stride tricks for efficiency
        shape = (len(arr) - window + 1, window)
        strides = (arr.strides[0], arr.strides[0])
        windows = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)
        return np.std(windows, axis=1)


# =============================================================================
# THREAD-SAFE RING BUFFER
# =============================================================================

class ThreadSafeRingBuffer(RingBuffer):
    """Thread-safe version of RingBuffer."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def append(self, value: float) -> None:
        with self._lock:
            super().append(value)

    def extend(self, values: Union[List[float], np.ndarray]) -> None:
        with self._lock:
            super().extend(values)

    def get_last(self, n: int) -> np.ndarray:
        with self._lock:
            return super().get_last(n)

    def to_array(self) -> np.ndarray:
        with self._lock:
            return super().to_array()

    def __getitem__(self, idx: int) -> float:
        with self._lock:
            return super().__getitem__(idx)

    def clear(self) -> None:
        with self._lock:
            super().clear()


# =============================================================================
# TYPED RING BUFFER (Multi-Column)
# =============================================================================

@dataclass
class OHLCVBar:
    """OHLCV bar data."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float


class TypedRingBuffer:
    """
    Multi-column ring buffer for OHLCV or similar structured data.

    Stores multiple related time series with same alignment.

    Example:
        buffer = TypedRingBuffer(
            max_size=10000,
            columns=['open', 'high', 'low', 'close', 'volume']
        )

        buffer.append({
            'open': 1.0850,
            'high': 1.0855,
            'low': 1.0845,
            'close': 1.0852,
            'volume': 1000
        })

        # Get specific column
        closes = buffer.get_column('close')

        # Get last n bars as dict
        recent = buffer.get_last_dict(100)
    """

    def __init__(
        self,
        max_size: int,
        columns: List[str],
        dtype: np.dtype = np.float64
    ):
        """
        Initialize typed ring buffer.

        Args:
            max_size: Maximum number of rows
            columns: Column names
            dtype: NumPy data type
        """
        self.max_size = max_size
        self.columns = columns
        self.dtype = dtype
        self._n_columns = len(columns)
        self._column_index = {name: i for i, name in enumerate(columns)}

        # 2D buffer: rows x columns
        self._buffer = np.full((max_size, self._n_columns), np.nan, dtype=dtype)
        self._index = 0
        self._count = 0
        self._lock = threading.RLock()

    def append(self, data: dict) -> None:
        """Append a row of data."""
        with self._lock:
            for col_name, col_idx in self._column_index.items():
                if col_name in data:
                    self._buffer[self._index, col_idx] = data[col_name]

            self._index = (self._index + 1) % self.max_size
            self._count = min(self._count + 1, self.max_size)

    def append_ohlcv(self, bar: OHLCVBar) -> None:
        """Append OHLCV bar."""
        self.append({
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        })

    def get_column(self, name: str, n: Optional[int] = None) -> np.ndarray:
        """Get a single column's values."""
        with self._lock:
            col_idx = self._column_index.get(name)
            if col_idx is None:
                raise KeyError(f"Unknown column: {name}")

            n = n or self._count
            n = min(n, self._count)

            if n == 0:
                return np.array([], dtype=self.dtype)

            start = (self._index - n) % self.max_size

            if start < self._index:
                return self._buffer[start:self._index, col_idx].copy()
            else:
                return np.concatenate([
                    self._buffer[start:, col_idx],
                    self._buffer[:self._index, col_idx]
                ])

    def get_last(self, n: int) -> np.ndarray:
        """Get last n rows as 2D array."""
        with self._lock:
            n = min(n, self._count)
            if n == 0:
                return np.array([]).reshape(0, self._n_columns)

            start = (self._index - n) % self.max_size

            if start < self._index:
                return self._buffer[start:self._index].copy()
            else:
                return np.concatenate([
                    self._buffer[start:],
                    self._buffer[:self._index]
                ])

    def get_last_dict(self, n: int) -> dict:
        """Get last n rows as dict of arrays."""
        data = self.get_last(n)
        return {
            col: data[:, idx]
            for col, idx in self._column_index.items()
        }

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame(
            self.get_last(self._count),
            columns=self.columns
        )

    def __len__(self) -> int:
        return self._count

    @property
    def is_full(self) -> bool:
        return self._count == self.max_size

    def clear(self) -> None:
        with self._lock:
            self._buffer.fill(np.nan)
            self._index = 0
            self._count = 0


# =============================================================================
# SPECIALIZED BUFFERS
# =============================================================================

class PriceBuffer(RingBuffer):
    """Specialized buffer for price data with trading helpers."""

    def __init__(self, max_size: int = 10000):
        super().__init__(max_size, dtype=np.float64)

    def volatility(self, window: int = 20) -> float:
        """Calculate annualized volatility."""
        returns = self.log_returns()
        if len(returns) < window:
            return np.nan
        recent_returns = returns[-window:]
        return np.std(recent_returns) * np.sqrt(252)  # Annualized

    def sma(self, period: int) -> float:
        """Simple moving average."""
        if self._count < period:
            return np.nan
        return np.mean(self.get_last(period))

    def ema(self, period: int) -> float:
        """Exponential moving average."""
        arr = self.to_array()
        if len(arr) < period:
            return np.nan

        alpha = 2 / (period + 1)
        ema = arr[0]
        for price in arr[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """Calculate Bollinger Bands (upper, middle, lower)."""
        if self._count < period:
            return np.nan, np.nan, np.nan

        recent = self.get_last(period)
        middle = np.mean(recent)
        std = np.std(recent)

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        return upper, middle, lower

    def atr(self, high_buffer: 'PriceBuffer', low_buffer: 'PriceBuffer', period: int = 14) -> float:
        """Average True Range (requires high/low buffers)."""
        closes = self.get_last(period + 1)
        highs = high_buffer.get_last(period)
        lows = low_buffer.get_last(period)

        if len(closes) < period + 1:
            return np.nan

        prev_closes = closes[:-1]

        tr = np.maximum(
            highs - lows,
            np.maximum(
                np.abs(highs - prev_closes),
                np.abs(lows - prev_closes)
            )
        )

        return np.mean(tr)
