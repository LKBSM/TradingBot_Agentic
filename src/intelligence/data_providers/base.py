"""Abstract base class for OHLCV data providers."""

from __future__ import annotations

import abc

import pandas as pd


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
