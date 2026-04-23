"""Data provider abstraction for Smart Sentinel AI.

Supports multiple data sources via a common interface:
  - CSVDataProvider: Local CSV files (testing/backtesting)
  - MT5DataProvider: MetaTrader 5 (live trading)

Each provider implements get_ohlcv(symbol, timeframe, lookback) → DataFrame.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataProvider(abc.ABC):
    """Abstract base class for OHLCV data providers."""

    @abc.abstractmethod
    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
    ) -> pd.DataFrame:
        """Fetch OHLCV data.

        Args:
            symbol: Instrument symbol (e.g., "XAUUSD").
            timeframe: Timeframe string (e.g., "M15", "H1").
            lookback: Number of bars to return.

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume
            and a DatetimeIndex.
        """
        ...

    @abc.abstractmethod
    def available_symbols(self) -> list[str]:
        """Return list of available symbols."""
        ...


class CSVDataProvider(DataProvider):
    """Load OHLCV data from local CSV files.

    Expected file naming: {data_dir}/{symbol}_{timeframe}.csv
    Expected columns: timestamp/Date, Open, High, Low, Close, Volume

    Usage:
        provider = CSVDataProvider("./data")
        df = provider.get_ohlcv("XAUUSD", "M15", 200)
    """

    def __init__(self, data_dir: str, file_pattern: str = "{symbol}_{timeframe}.csv"):
        self._data_dir = Path(data_dir)
        self._file_pattern = file_pattern
        self._cache: Dict[str, pd.DataFrame] = {}
        self._cache_mtime: Dict[str, float] = {}

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
    ) -> pd.DataFrame:
        cache_key = f"{symbol}_{timeframe}"
        filepath = self._data_dir / self._file_pattern.format(
            symbol=symbol, timeframe=timeframe
        )

        # Invalidate cache if the underlying file has been modified. Without
        # this, a CSV being streamed to (e.g. live-simulation, append-only)
        # would never surface new bars.
        current_mtime = filepath.stat().st_mtime if filepath.exists() else 0.0
        cached_mtime = self._cache_mtime.get(cache_key)

        if cache_key not in self._cache or cached_mtime != current_mtime:
            self._cache[cache_key] = self._load_csv(symbol, timeframe)
            self._cache_mtime[cache_key] = current_mtime

        df = self._cache[cache_key]
        return df.tail(lookback).copy()

    def _load_csv(self, symbol: str, timeframe: str) -> pd.DataFrame:
        filename = self._file_pattern.format(symbol=symbol, timeframe=timeframe)
        filepath = self._data_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")

        df = pd.read_csv(filepath)

        # Normalize columns
        rename = {}
        for col in df.columns:
            lower = col.lower()
            if lower in ("date", "datetime", "time", "timestamp"):
                rename[col] = "timestamp"

        if rename:
            df = df.rename(columns=rename)

        # Normalize OHLCV column capitalization
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if lower == "open":
                col_map[col] = "Open"
            elif lower == "high":
                col_map[col] = "High"
            elif lower == "low":
                col_map[col] = "Low"
            elif lower == "close":
                col_map[col] = "Close"
            elif lower == "volume":
                col_map[col] = "Volume"
        df = df.rename(columns=col_map)

        # Set DatetimeIndex
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")

        logger.info("Loaded %s_%s: %d bars from %s", symbol, timeframe, len(df), filepath)
        return df

    def available_symbols(self) -> list[str]:
        symbols = set()
        for f in self._data_dir.glob("*.csv"):
            parts = f.stem.split("_")
            if len(parts) >= 2:
                symbols.add(parts[0])
        return sorted(symbols)

    def clear_cache(self) -> None:
        self._cache.clear()


class MT5DataProvider(DataProvider):
    """MetaTrader 5 live data provider.

    Requires MetaTrader5 Python package: pip install MetaTrader5
    Only works on Windows with MT5 terminal installed.

    Features:
      - Auto-reconnection on connection drop
      - Symbol validation before data fetch
      - Symbol auto-selection (MarketWatch)

    Usage:
        provider = MT5DataProvider()
        provider.connect(login=12345, password="pwd", server="Broker-Demo")
        df = provider.get_ohlcv("XAUUSD", "M15", 200)
    """

    TIMEFRAME_MAP = {
        "M1": "TIMEFRAME_M1",
        "M5": "TIMEFRAME_M5",
        "M15": "TIMEFRAME_M15",
        "M30": "TIMEFRAME_M30",
        "H1": "TIMEFRAME_H1",
        "H4": "TIMEFRAME_H4",
        "D1": "TIMEFRAME_D1",
        "W1": "TIMEFRAME_W1",
    }

    MAX_RECONNECT_ATTEMPTS = 3
    RECONNECT_DELAY_S = 2.0

    def __init__(self):
        self._mt5: Any = None
        self._connected = False
        # Store credentials for reconnection
        self._login: Optional[int] = None
        self._password: Optional[str] = None
        self._server: Optional[str] = None
        self._path: Optional[str] = None
        # Cache validated symbols to avoid repeated lookups
        self._validated_symbols: set[str] = set()
        # Track volume source per symbol (real_volume vs tick_volume)
        self._volume_source: Dict[str, str] = {}

    def connect(
        self,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        path: Optional[str] = None,
    ) -> bool:
        """Initialize MT5 connection.

        Credentials are stored internally for automatic reconnection.
        Returns True on success.
        """
        try:
            import MetaTrader5 as mt5
            self._mt5 = mt5
        except ImportError:
            logger.error(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
            return False

        # Store for reconnection
        if login is not None:
            self._login = login
        if password is not None:
            self._password = password
        if server is not None:
            self._server = server
        if path is not None:
            self._path = path

        kwargs: Dict[str, Any] = {}
        if self._login is not None:
            kwargs["login"] = self._login
        if self._password is not None:
            kwargs["password"] = self._password
        if self._server is not None:
            kwargs["server"] = self._server
        if self._path is not None:
            kwargs["path"] = self._path

        if not self._mt5.initialize(**kwargs):
            error = self._mt5.last_error()
            logger.error("MT5 initialization failed: %s", error)
            return False

        self._connected = True
        self._validated_symbols.clear()
        terminal_info = self._mt5.terminal_info()
        if terminal_info:
            logger.info("MT5 connected: %s", terminal_info.company)
        return True

    def disconnect(self) -> None:
        if self._mt5 is not None and self._connected:
            self._mt5.shutdown()
            self._connected = False
            self._validated_symbols.clear()

    def reconnect(self) -> bool:
        """Disconnect and reconnect using stored credentials.

        Returns True on success.
        """
        logger.info("Attempting MT5 reconnection...")
        try:
            if self._mt5 is not None and self._connected:
                self._mt5.shutdown()
                self._connected = False
        except Exception:
            pass  # Best-effort shutdown

        import time
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            time.sleep(self.RECONNECT_DELAY_S)
            if self.connect():
                logger.info("MT5 reconnected on attempt %d", attempt)
                return True
            logger.warning("MT5 reconnect attempt %d/%d failed",
                           attempt, self.MAX_RECONNECT_ATTEMPTS)
        logger.error("MT5 reconnection failed after %d attempts",
                      self.MAX_RECONNECT_ATTEMPTS)
        return False

    def ensure_connected(self) -> bool:
        """Check connection health and reconnect if needed.

        Returns True if connected (or successfully reconnected).
        """
        if not self._connected or self._mt5 is None:
            return self.reconnect()

        # Quick health check — account_info() is the cheapest MT5 call
        try:
            info = self._mt5.account_info()
            if info is not None:
                return True
        except Exception:
            pass

        logger.warning("MT5 connection appears stale — reconnecting")
        return self.reconnect()

    def _validate_and_select_symbol(self, symbol: str) -> None:
        """Ensure symbol exists on broker and is visible in MarketWatch.

        Raises ValueError if symbol is not available.
        """
        if symbol in self._validated_symbols:
            return  # Already validated this session

        symbol_info = self._mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(
                f"Symbol '{symbol}' not found on this broker. "
                f"Check spelling or use available_symbols() to see what's offered."
            )

        if not symbol_info.visible:
            if not self._mt5.symbol_select(symbol, True):
                raise RuntimeError(
                    f"Symbol '{symbol}' exists but could not be added to MarketWatch."
                )

        self._validated_symbols.add(symbol)

    def get_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        lookback: int,
    ) -> pd.DataFrame:
        if not self.ensure_connected():
            raise RuntimeError("MT5 not connected and reconnection failed.")

        tf_attr = self.TIMEFRAME_MAP.get(timeframe)
        if tf_attr is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")

        self._validate_and_select_symbol(symbol)

        mt5_tf = getattr(self._mt5, tf_attr)
        rates = self._mt5.copy_rates_from_pos(symbol, mt5_tf, 0, lookback)

        if rates is None or len(rates) == 0:
            error = self._mt5.last_error()
            raise RuntimeError(f"Failed to get rates for {symbol}: {error}")

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.set_index("time")

        # Prefer real_volume when the broker exposes it and it's non-zero.
        # Fall back to tick_volume (quote-update count) otherwise. Real volume
        # is essential for SMC volume confirmation — tick volume correlates
        # with activity but not institutional flow.
        volume_col = "tick_volume"
        if "real_volume" in df.columns:
            recent = df["real_volume"].tail(50)
            if (recent > 0).any():
                volume_col = "real_volume"

        if self._volume_source.get(symbol) != volume_col:
            logger.info(
                "MT5 %s: using %s for Volume",
                symbol, volume_col,
            )
            self._volume_source[symbol] = volume_col

        df = df.rename(columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            volume_col: "Volume",
        })

        return df[["Open", "High", "Low", "Close", "Volume"]]

    def available_symbols(self) -> list[str]:
        if not self._connected or self._mt5 is None:
            return []
        symbols = self._mt5.symbols_get()
        return [s.name for s in symbols] if symbols else []
