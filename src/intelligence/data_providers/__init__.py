"""Data provider abstraction for Smart Sentinel AI.

Supports multiple data sources via a common interface:
  - CSVDataProvider: Local CSV files (testing/backtesting)
  - MT5DataProvider: MetaTrader 5 (live trading)
  - TwelveDataProvider: Twelve Data REST API (MIA Markets V1 backend)

Each provider implements get_ohlcv(symbol, timeframe, lookback) → DataFrame.
"""

from src.intelligence.data_providers.base import DataProvider
from src.intelligence.data_providers.csv_provider import CSVDataProvider
from src.intelligence.data_providers.mt5_provider import MT5DataProvider
from src.intelligence.data_providers.twelve_data_provider import TwelveDataProvider

__all__ = ["DataProvider", "CSVDataProvider", "MT5DataProvider", "TwelveDataProvider"]
