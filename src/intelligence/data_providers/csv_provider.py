"""CSV-backed OHLCV data provider (testing/backtesting)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from src.intelligence.data_providers.base import DataProvider

logger = logging.getLogger(__name__)


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
