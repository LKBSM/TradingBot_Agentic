"""Tests for Sprint 8: Data Providers & Operational Entry Point.

Tests cover:
  - CSVDataProvider loads CSV files correctly
  - CSVDataProvider lookback slicing
  - CSVDataProvider caching
  - CSVDataProvider handles missing files
  - CSVDataProvider available_symbols
  - DataProvider ABC interface
  - build_system() wiring (without actual API startup)
  - _NullAgent stub
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.data_providers import CSVDataProvider, DataProvider, MT5DataProvider
from src.intelligence.main import _NullAgent, build_system


# =============================================================================
# CSV DATA PROVIDER
# =============================================================================

def _write_csv(tmpdir: Path, filename: str, n_bars: int = 500) -> Path:
    """Write a synthetic OHLCV CSV file."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n_bars, freq="15min")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 0.5)
    df = pd.DataFrame({
        "timestamp": ts,
        "Open": close + rng.randn(n_bars) * 0.3,
        "High": close + rng.uniform(0.5, 3.0, n_bars),
        "Low": close - rng.uniform(0.5, 3.0, n_bars),
        "Close": close,
        "Volume": rng.uniform(100, 10000, n_bars),
    })
    path = tmpdir / filename
    df.to_csv(path, index=False)
    return path


class TestCSVDataProvider:
    def test_load_csv(self):
        """Should load OHLCV data from CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 500)
            provider = CSVDataProvider(tmpdir)
            df = provider.get_ohlcv("XAUUSD", "M15", 200)

            assert len(df) == 200
            assert "Open" in df.columns
            assert "High" in df.columns
            assert "Low" in df.columns
            assert "Close" in df.columns
            assert "Volume" in df.columns

    def test_lookback_slicing(self):
        """Should return only the last N bars."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 500)
            provider = CSVDataProvider(tmpdir)

            df_100 = provider.get_ohlcv("XAUUSD", "M15", 100)
            df_50 = provider.get_ohlcv("XAUUSD", "M15", 50)

            assert len(df_100) == 100
            assert len(df_50) == 50
            # Last bar should be the same
            assert df_100["Close"].iloc[-1] == df_50["Close"].iloc[-1]

    def test_caching(self):
        """Second load should come from cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 500)
            provider = CSVDataProvider(tmpdir)

            df1 = provider.get_ohlcv("XAUUSD", "M15", 200)
            df2 = provider.get_ohlcv("XAUUSD", "M15", 200)

            # Should be different DataFrame objects (copy)
            assert df1 is not df2
            # But same data
            pd.testing.assert_frame_equal(df1, df2)

    def test_missing_file_raises(self):
        """Should raise FileNotFoundError for missing CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = CSVDataProvider(tmpdir)
            with pytest.raises(FileNotFoundError):
                provider.get_ohlcv("NONEXISTENT", "M15", 100)

    def test_available_symbols(self):
        """Should detect available symbols from CSV files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv")
            _write_csv(Path(tmpdir), "EURUSD_M15.csv")
            _write_csv(Path(tmpdir), "BTCUSD_H1.csv")

            provider = CSVDataProvider(tmpdir)
            symbols = provider.available_symbols()

            assert "BTCUSD" in symbols
            assert "EURUSD" in symbols
            assert "XAUUSD" in symbols

    def test_datetime_index(self):
        """DataFrame should have DatetimeIndex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 100)
            provider = CSVDataProvider(tmpdir)
            df = provider.get_ohlcv("XAUUSD", "M15", 50)
            assert isinstance(df.index, pd.DatetimeIndex)

    def test_clear_cache(self):
        """clear_cache should empty the internal cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 100)
            provider = CSVDataProvider(tmpdir)
            provider.get_ohlcv("XAUUSD", "M15", 50)
            assert len(provider._cache) == 1
            provider.clear_cache()
            assert len(provider._cache) == 0

    def test_lowercase_columns(self):
        """Should handle lowercase column names in CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            rng = np.random.RandomState(42)
            n = 100
            ts = pd.date_range("2024-01-01", periods=n, freq="15min")
            df = pd.DataFrame({
                "date": ts,
                "open": rng.randn(n) + 2000,
                "high": rng.randn(n) + 2002,
                "low": rng.randn(n) + 1998,
                "close": rng.randn(n) + 2000,
                "volume": rng.uniform(100, 1000, n),
            })
            df.to_csv(Path(tmpdir) / "TEST_M15.csv", index=False)

            provider = CSVDataProvider(tmpdir)
            result = provider.get_ohlcv("TEST", "M15", 50)
            assert "Open" in result.columns
            assert "Close" in result.columns


# =============================================================================
# DATA PROVIDER ABSTRACT CLASS
# =============================================================================

class TestDataProviderABC:
    def test_cannot_instantiate(self):
        """DataProvider ABC should not be directly instantiable."""
        with pytest.raises(TypeError):
            DataProvider()

    def test_csv_is_subclass(self):
        assert issubclass(CSVDataProvider, DataProvider)

    def test_mt5_is_subclass(self):
        assert issubclass(MT5DataProvider, DataProvider)


# =============================================================================
# NULL AGENT
# =============================================================================

class TestNullAgent:
    def test_analyze_returns_none(self):
        agent = _NullAgent()
        assert agent.analyze(1, 2, 3) is None

    def test_evaluate_news_returns_none(self):
        agent = _NullAgent()
        assert agent.evaluate_news_impact("test") is None


# =============================================================================
# BUILD SYSTEM (unit test — no real API)
# =============================================================================

class TestBuildSystem:
    def test_build_system_creates_scanner(self):
        """build_system should create a working scanner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write dummy data
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 3000)

            # Also need signal DB dir
            db_path = str(Path(tmpdir) / "signals.db")

            with patch("src.intelligence.main._create_regime_agent", return_value=_NullAgent()), \
                 patch("src.intelligence.main._create_news_agent", return_value=_NullAgent()):

                system = build_system(
                    symbols=["XAUUSD"],
                    data_dir=tmpdir,
                    signal_db=db_path,
                    vol_mode="har",
                )

            assert "scanner" in system
            assert "api_app" in system
            assert "signal_store" in system
            assert system["scanner"] is not None

    def test_build_system_multi_symbol(self):
        """build_system with multiple symbols should create MultiSymbolScanner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_csv(Path(tmpdir), "XAUUSD_M15.csv", 3000)
            _write_csv(Path(tmpdir), "EURUSD_M15.csv", 3000)
            db_path = str(Path(tmpdir) / "signals.db")

            with patch("src.intelligence.main._create_regime_agent", return_value=_NullAgent()), \
                 patch("src.intelligence.main._create_news_agent", return_value=_NullAgent()):

                system = build_system(
                    symbols=["XAUUSD", "EURUSD"],
                    data_dir=tmpdir,
                    signal_db=db_path,
                    vol_mode="har",
                )

            from src.intelligence.sentinel_scanner import MultiSymbolScanner
            assert isinstance(system["scanner"], MultiSymbolScanner)
