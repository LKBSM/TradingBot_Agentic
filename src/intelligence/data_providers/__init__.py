"""Data provider abstraction for Smart Sentinel AI.

Supports multiple data sources via a common interface:
  - CSVDataProvider: Local CSV files (testing/backtesting)
  - MT5DataProvider: MetaTrader 5 (live trading)

Each provider implements get_ohlcv(symbol, timeframe, lookback) → DataFrame.
"""

from src.intelligence.data_providers.base import DataProvider
from src.intelligence.data_providers.csv_provider import CSVDataProvider
from src.intelligence.data_providers.mt5_provider import MT5DataProvider

__all__ = ["DataProvider", "CSVDataProvider", "MT5DataProvider"]
