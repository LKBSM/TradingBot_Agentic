"""MT5 integration tests for Smart Sentinel AI.

Tests the MT5DataProvider and full pipeline startup with MT5 data source.
MT5 tests are automatically skipped when the terminal is not running.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch, PropertyMock
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.intelligence.data_providers import MT5DataProvider, CSVDataProvider, DataProvider


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def mock_mt5():
    """Create a mock MetaTrader5 module with symbol validation support."""
    mt5 = MagicMock()
    mt5.__version__ = "5.0.5735"
    mt5.TIMEFRAME_M1 = 1
    mt5.TIMEFRAME_M5 = 5
    mt5.TIMEFRAME_M15 = 15
    mt5.TIMEFRAME_M30 = 30
    mt5.TIMEFRAME_H1 = 16385
    mt5.TIMEFRAME_H4 = 16388
    mt5.TIMEFRAME_D1 = 16408
    mt5.TIMEFRAME_W1 = 32769

    # Default: symbol_info returns a valid symbol, account_info returns valid info
    symbol_info = MagicMock()
    symbol_info.visible = True
    mt5.symbol_info.return_value = symbol_info
    mt5.symbol_select.return_value = True

    account_info = MagicMock()
    account_info.login = 12345
    mt5.account_info.return_value = account_info

    return mt5


@pytest.fixture()
def sample_rates():
    """Generate sample OHLCV data as MT5 returns it."""
    now = datetime.utcnow()
    bars = 200
    times = [int((now - timedelta(minutes=15 * (bars - i))).timestamp()) for i in range(bars)]
    np.random.seed(42)
    close = 2350.0 + np.cumsum(np.random.randn(bars) * 2)

    # Create structured array like MT5 returns
    dtype = np.dtype([
        ('time', 'i8'), ('open', 'f8'), ('high', 'f8'),
        ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
        ('spread', 'i4'), ('real_volume', 'i8'),
    ])
    data = np.zeros(bars, dtype=dtype)
    data['time'] = times
    data['close'] = close
    data['open'] = close - np.random.rand(bars) * 1.5
    data['high'] = np.maximum(data['open'], data['close']) + np.random.rand(bars) * 2
    data['low'] = np.minimum(data['open'], data['close']) - np.random.rand(bars) * 2
    data['tick_volume'] = np.random.randint(100, 10000, bars)
    return data


# =============================================================================
# MT5DataProvider UNIT TESTS
# =============================================================================

class TestMT5DataProvider:
    """Test MT5DataProvider with mocked MT5 module."""

    def test_connect_success(self, mock_mt5):
        """Successful MT5 connection."""
        mock_mt5.initialize.return_value = True
        terminal_info = MagicMock()
        terminal_info.company = "Test Broker"
        mock_mt5.terminal_info.return_value = terminal_info

        provider = MT5DataProvider()
        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            result = provider.connect(login=12345, password="test", server="TestServer")

        assert result is True
        assert provider._connected is True

    def test_connect_stores_credentials(self, mock_mt5):
        """connect() stores credentials for reconnection."""
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(company="Test")

        provider = MT5DataProvider()
        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            provider.connect(login=12345, password="secret", server="TestSrv")

        assert provider._login == 12345
        assert provider._password == "secret"
        assert provider._server == "TestSrv"

    def test_connect_failure(self, mock_mt5):
        """Failed MT5 connection returns False."""
        mock_mt5.initialize.return_value = False
        mock_mt5.last_error.return_value = (-6, "Auth failed")

        provider = MT5DataProvider()
        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            result = provider.connect()

        assert result is False
        assert provider._connected is False

    def test_connect_no_package(self):
        """Missing MetaTrader5 package returns False."""
        provider = MT5DataProvider()
        with patch.dict("sys.modules", {"MetaTrader5": None}):
            with patch("builtins.__import__", side_effect=ImportError("No module")):
                result = provider.connect()

        assert result is False

    def test_get_ohlcv_success(self, mock_mt5, sample_rates):
        """Fetch OHLCV data successfully."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        df = provider.get_ohlcv("XAUUSD", "M15", 200)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 200
        assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_get_ohlcv_not_connected(self):
        """get_ohlcv raises when not connected and reconnect fails."""
        provider = MT5DataProvider()

        with patch.object(provider, "ensure_connected", return_value=False):
            with pytest.raises(RuntimeError, match="not connected"):
                provider.get_ohlcv("XAUUSD", "M15", 200)

    def test_get_ohlcv_bad_timeframe(self, mock_mt5):
        """Unsupported timeframe raises ValueError."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        with patch.object(provider, "ensure_connected", return_value=True):
            with pytest.raises(ValueError, match="Unsupported timeframe"):
                provider.get_ohlcv("XAUUSD", "M2", 200)

    def test_get_ohlcv_no_data(self, mock_mt5):
        """No data returned raises RuntimeError."""
        mock_mt5.copy_rates_from_pos.return_value = None
        mock_mt5.last_error.return_value = (-1, "No data")

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True
        provider._validated_symbols.add("XAUUSD")  # Skip symbol validation

        with pytest.raises(RuntimeError, match="Failed to get rates"):
            provider.get_ohlcv("XAUUSD", "M15", 200)

    def test_available_symbols(self, mock_mt5):
        """available_symbols returns list of symbol names."""
        sym1 = MagicMock()
        sym1.name = "XAUUSD"
        sym2 = MagicMock()
        sym2.name = "EURUSD"
        mock_mt5.symbols_get.return_value = [sym1, sym2]

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        symbols = provider.available_symbols()
        assert symbols == ["XAUUSD", "EURUSD"]

    def test_available_symbols_not_connected(self):
        """available_symbols returns empty list when not connected."""
        provider = MT5DataProvider()
        assert provider.available_symbols() == []

    def test_disconnect(self, mock_mt5):
        """disconnect calls shutdown."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        provider.disconnect()

        mock_mt5.shutdown.assert_called_once()
        assert provider._connected is False

    def test_disconnect_when_not_connected(self, mock_mt5):
        """disconnect is safe when not connected."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = False

        provider.disconnect()  # Should not raise
        mock_mt5.shutdown.assert_not_called()

    def test_all_timeframes_mapped(self, mock_mt5, sample_rates):
        """All supported timeframes map to MT5 constants."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates[:50]

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        for tf in ["M1", "M5", "M15", "M30", "H1", "H4", "D1", "W1"]:
            df = provider.get_ohlcv("XAUUSD", tf, 50)
            assert len(df) == 50

    def test_ohlcv_column_types(self, mock_mt5, sample_rates):
        """OHLCV columns are float64."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        df = provider.get_ohlcv("XAUUSD", "M15", 200)

        for col in ["Open", "High", "Low", "Close"]:
            assert df[col].dtype == np.float64


# =============================================================================
# SYMBOL VALIDATION TESTS
# =============================================================================

class TestSymbolValidation:
    """Test symbol validation and auto-selection."""

    def test_invalid_symbol_raises(self, mock_mt5):
        """Unknown symbol raises ValueError."""
        mock_mt5.symbol_info.return_value = None

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        with pytest.raises(ValueError, match="not found"):
            provider.get_ohlcv("INVALID", "M15", 200)

    def test_symbol_auto_selected(self, mock_mt5, sample_rates):
        """Hidden symbol is auto-selected into MarketWatch."""
        symbol_info = MagicMock()
        symbol_info.visible = False  # Not in MarketWatch
        mock_mt5.symbol_info.return_value = symbol_info
        mock_mt5.symbol_select.return_value = True
        mock_mt5.copy_rates_from_pos.return_value = sample_rates

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        df = provider.get_ohlcv("EURUSD", "M15", 200)
        assert len(df) == 200
        mock_mt5.symbol_select.assert_called_with("EURUSD", True)

    def test_symbol_select_failure(self, mock_mt5):
        """Symbol that can't be selected raises RuntimeError."""
        symbol_info = MagicMock()
        symbol_info.visible = False
        mock_mt5.symbol_info.return_value = symbol_info
        mock_mt5.symbol_select.return_value = False

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        with pytest.raises(RuntimeError, match="could not be added"):
            provider.get_ohlcv("ODDPAIR", "M15", 200)

    def test_validated_symbol_cached(self, mock_mt5, sample_rates):
        """Once validated, symbol is not re-checked."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates[:50]

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        provider.get_ohlcv("XAUUSD", "M15", 50)
        provider.get_ohlcv("XAUUSD", "M15", 50)

        # symbol_info should be called only once (cached after first)
        assert mock_mt5.symbol_info.call_count == 1

    def test_validated_cache_cleared_on_disconnect(self, mock_mt5, sample_rates):
        """Disconnect clears the validated symbols cache."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates[:50]

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        provider.get_ohlcv("XAUUSD", "M15", 50)
        assert "XAUUSD" in provider._validated_symbols

        provider.disconnect()
        assert len(provider._validated_symbols) == 0


# =============================================================================
# RECONNECTION TESTS
# =============================================================================

class TestReconnection:
    """Test auto-reconnection logic."""

    def test_ensure_connected_when_healthy(self, mock_mt5):
        """ensure_connected returns True when connection is healthy."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        assert provider.ensure_connected() is True

    def test_ensure_connected_reconnects_on_stale(self, mock_mt5):
        """ensure_connected reconnects when account_info fails."""
        mock_mt5.account_info.return_value = None  # Stale
        mock_mt5.initialize.return_value = True
        mock_mt5.terminal_info.return_value = MagicMock(company="Test")

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True

        with patch.object(provider, "reconnect", return_value=True) as mock_reconnect:
            result = provider.ensure_connected()

        assert result is True
        mock_reconnect.assert_called_once()

    def test_ensure_connected_when_disconnected(self, mock_mt5):
        """ensure_connected triggers reconnect when not connected."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = False

        with patch.object(provider, "reconnect", return_value=True) as mock_reconnect:
            result = provider.ensure_connected()

        assert result is True
        mock_reconnect.assert_called_once()

    def test_reconnect_success(self, mock_mt5):
        """reconnect() disconnects and reconnects."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True
        provider._login = 12345

        # Patch connect() to simulate success without real MT5 import
        with patch.object(provider, "connect", return_value=True) as mock_connect:
            with patch("time.sleep"):
                result = provider.reconnect()

        assert result is True
        mock_connect.assert_called_once()

    def test_reconnect_failure_retries(self, mock_mt5):
        """reconnect() retries MAX_RECONNECT_ATTEMPTS times."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = False

        with patch.object(provider, "connect", return_value=False) as mock_connect:
            with patch("time.sleep"):
                result = provider.reconnect()

        assert result is False
        assert mock_connect.call_count == MT5DataProvider.MAX_RECONNECT_ATTEMPTS

    def test_get_ohlcv_auto_reconnects(self, mock_mt5, sample_rates):
        """get_ohlcv auto-reconnects if connection drops."""
        mock_mt5.copy_rates_from_pos.return_value = sample_rates[:50]

        # account_info returns None first (stale), then valid after reconnect
        call_count = [0]
        def account_side_effect():
            call_count[0] += 1
            if call_count[0] <= 1:
                return None  # First call: stale
            return MagicMock(login=12345)  # After reconnect: healthy
        mock_mt5.account_info.side_effect = account_side_effect

        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True
        provider._login = 12345

        # Patch reconnect to just set _connected = True (skip real MT5 import)
        def fake_reconnect():
            provider._connected = True
            return True
        with patch.object(provider, "reconnect", side_effect=fake_reconnect):
            df = provider.get_ohlcv("XAUUSD", "M15", 50)

        assert len(df) == 50


# =============================================================================
# GRACEFUL SHUTDOWN TESTS
# =============================================================================

class TestGracefulShutdown:
    """Test that main.py disconnects data provider on shutdown."""

    def test_mt5_provider_has_disconnect(self):
        """MT5DataProvider exposes disconnect() for graceful shutdown."""
        provider = MT5DataProvider()
        assert hasattr(provider, "disconnect")
        assert callable(provider.disconnect)

    def test_csv_provider_shutdown_safe(self, tmp_path):
        """CSV provider has no disconnect — main.py handles this via hasattr."""
        from src.intelligence.main import build_system

        dates = pd.date_range("2024-01-01", periods=500, freq="15min")
        np.random.seed(42)
        close = 2350.0 + np.cumsum(np.random.randn(500) * 2)
        df = pd.DataFrame({
            "timestamp": dates,
            "Open": close - 1, "High": close + 2,
            "Low": close - 2, "Close": close,
            "Volume": np.random.randint(100, 5000, 500),
        })
        df.to_csv(tmp_path / "XAUUSD_M15.csv", index=False)

        system = build_system(
            symbols=["XAUUSD"],
            data_dir=str(tmp_path),
            signal_db=str(tmp_path / "signals.db"),
            vol_mode="har",
            data_source="csv",
        )

        data_provider = system["data_provider"]
        # CSVDataProvider has clear_cache but no disconnect — that's fine
        # main.py uses hasattr guard so no crash on shutdown
        assert isinstance(data_provider, CSVDataProvider)

    def test_mt5_disconnect_called_on_shutdown(self, mock_mt5):
        """Simulates the main.py shutdown path for MT5 provider."""
        provider = MT5DataProvider()
        provider._mt5 = mock_mt5
        provider._connected = True
        provider._validated_symbols.add("XAUUSD")

        # Simulate the shutdown code from main.py
        if hasattr(provider, "disconnect"):
            provider.disconnect()

        mock_mt5.shutdown.assert_called_once()
        assert provider._connected is False
        assert len(provider._validated_symbols) == 0


# =============================================================================
# DATA PROVIDER INTERFACE TESTS
# =============================================================================

class TestDataProviderInterface:
    """Verify both providers implement the same interface."""

    def test_mt5_is_data_provider(self):
        assert issubclass(MT5DataProvider, DataProvider)

    def test_csv_is_data_provider(self):
        assert issubclass(CSVDataProvider, DataProvider)

    def test_csv_provider_missing_file(self, tmp_path):
        provider = CSVDataProvider(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            provider.get_ohlcv("XAUUSD", "M15", 200)

    def test_csv_provider_loads_data(self, tmp_path):
        """CSVDataProvider loads and normalizes CSV data."""
        dates = pd.date_range("2024-01-01", periods=100, freq="15min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.rand(100) * 100 + 2300,
            "high": np.random.rand(100) * 100 + 2310,
            "low": np.random.rand(100) * 100 + 2290,
            "close": np.random.rand(100) * 100 + 2300,
            "volume": np.random.randint(100, 5000, 100),
        })
        df.to_csv(tmp_path / "XAUUSD_M15.csv", index=False)

        provider = CSVDataProvider(str(tmp_path))
        result = provider.get_ohlcv("XAUUSD", "M15", 50)

        assert len(result) == 50
        assert "Open" in result.columns
        assert "Close" in result.columns


# =============================================================================
# BUILD_SYSTEM MT5 INTEGRATION TESTS
# =============================================================================

class TestBuildSystemMT5:
    """Test that build_system() can wire MT5 into the pipeline."""

    def test_build_system_mt5_connection_failure(self, tmp_path):
        """build_system raises RuntimeError when MT5 connection fails."""
        from src.intelligence.main import build_system

        with patch("src.intelligence.data_providers.MT5DataProvider.connect", return_value=False):
            with pytest.raises(RuntimeError, match="MT5 connection failed"):
                build_system(
                    symbols=["XAUUSD"],
                    data_dir=str(tmp_path),
                    signal_db=str(tmp_path / "signals.db"),
                    vol_mode="har",
                    data_source="mt5",
                )

    def test_build_system_csv_fallback(self, tmp_path):
        """build_system works with CSV when no MT5."""
        from src.intelligence.main import build_system

        dates = pd.date_range("2024-01-01", periods=500, freq="15min")
        np.random.seed(42)
        close = 2350.0 + np.cumsum(np.random.randn(500) * 2)
        df = pd.DataFrame({
            "timestamp": dates,
            "Open": close - 1, "High": close + 2,
            "Low": close - 2, "Close": close,
            "Volume": np.random.randint(100, 5000, 500),
        })
        df.to_csv(tmp_path / "XAUUSD_M15.csv", index=False)

        system = build_system(
            symbols=["XAUUSD"],
            data_dir=str(tmp_path),
            signal_db=str(tmp_path / "signals.db"),
            vol_mode="har",
            data_source="csv",
        )

        assert "scanner" in system
        assert "api_app" in system
        assert "data_provider" in system

    def test_build_system_mt5_success(self, tmp_path, mock_mt5, sample_rates):
        """build_system succeeds with mocked MT5 connection."""
        from src.intelligence.main import build_system

        mock_mt5.initialize.return_value = True
        terminal_info = MagicMock()
        terminal_info.company = "Test Broker"
        mock_mt5.terminal_info.return_value = terminal_info
        mock_mt5.copy_rates_from_pos.return_value = sample_rates

        with patch.dict("sys.modules", {"MetaTrader5": mock_mt5}):
            with patch("src.intelligence.data_providers.MT5DataProvider.__init__", return_value=None):
                with patch("src.intelligence.data_providers.MT5DataProvider.connect", return_value=True):
                    with patch("src.intelligence.data_providers.MT5DataProvider.get_ohlcv") as mock_ohlcv:
                        rates_df = pd.DataFrame(sample_rates)
                        rates_df["time"] = pd.to_datetime(rates_df["time"], unit="s")
                        rates_df = rates_df.set_index("time")
                        rates_df = rates_df.rename(columns={
                            "open": "Open", "high": "High",
                            "low": "Low", "close": "Close",
                            "tick_volume": "Volume",
                        })
                        mock_ohlcv.return_value = rates_df[["Open", "High", "Low", "Close", "Volume"]]

                        system = build_system(
                            symbols=["XAUUSD"],
                            data_dir=str(tmp_path),
                            signal_db=str(tmp_path / "signals.db"),
                            vol_mode="har",
                            data_source="mt5",
                        )

                        assert "scanner" in system
                        assert "api_app" in system


# =============================================================================
# MT5 SETUP SCRIPT TESTS
# =============================================================================

class TestMT5SetupScript:
    """Test the mt5_setup.py helper functions."""

    def test_find_mt5_terminal(self):
        """find_mt5_terminal returns a path or None."""
        from scripts.mt5_setup import find_mt5_terminal
        result = find_mt5_terminal()
        assert result is None or os.path.exists(result)

    def test_is_mt5_running(self):
        """is_mt5_running returns bool."""
        from scripts.mt5_setup import is_mt5_running
        result = is_mt5_running()
        assert isinstance(result, bool)
