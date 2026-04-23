"""Tests for Sprint 7: Multi-Timeframe Support.

Tests cover:
  - timeframe_to_minutes() conversion
  - bars_per_day_for_timeframe() computation
  - resample_ohlcv() M1→M15, M5→H1, same-tf passthrough
  - InstrumentConfig with different timeframes
  - HAR windows adapt to bars_per_day
  - MultiSymbolScanner with mixed timeframes
"""

import numpy as np
import pandas as pd
import pytest

from src.intelligence.volatility_forecaster import (
    InstrumentConfig,
    VolatilityForecaster,
    bars_per_day_for_timeframe,
    get_instrument_registry,
    resample_ohlcv,
    timeframe_to_minutes,
)
from src.intelligence.sentinel_scanner import MultiSymbolScanner
from unittest.mock import MagicMock


# =============================================================================
# TIMEFRAME CONVERSION TESTS
# =============================================================================

class TestTimeframeToMinutes:
    def test_m1(self):
        assert timeframe_to_minutes("M1") == 1

    def test_m5(self):
        assert timeframe_to_minutes("M5") == 5

    def test_m15(self):
        assert timeframe_to_minutes("M15") == 15

    def test_m30(self):
        assert timeframe_to_minutes("M30") == 30

    def test_h1(self):
        assert timeframe_to_minutes("H1") == 60

    def test_h4(self):
        assert timeframe_to_minutes("H4") == 240

    def test_d1(self):
        assert timeframe_to_minutes("D1") == 1440

    def test_w1(self):
        assert timeframe_to_minutes("W1") == 10080

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown timeframe"):
            timeframe_to_minutes("M7")


class TestBarsPerDay:
    def test_m15_24h(self):
        assert bars_per_day_for_timeframe("M15", 24) == 96

    def test_h1_24h(self):
        assert bars_per_day_for_timeframe("H1", 24) == 24

    def test_m15_equity_7h(self):
        """US equities ~7 trading hours."""
        assert bars_per_day_for_timeframe("M15", 7) == 28

    def test_h4_24h(self):
        assert bars_per_day_for_timeframe("H4", 24) == 6

    def test_d1(self):
        assert bars_per_day_for_timeframe("D1", 24) == 1


# =============================================================================
# OHLCV RESAMPLING TESTS
# =============================================================================

def _make_m1_data(n_bars: int = 960) -> pd.DataFrame:
    """Generate M1 OHLCV data."""
    rng = np.random.RandomState(42)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 0.1)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close + rng.randn(n_bars) * 0.05,
        "high": close + rng.uniform(0.1, 0.5, n_bars),
        "low": close - rng.uniform(0.1, 0.5, n_bars),
        "close": close,
        "volume": rng.uniform(10, 100, n_bars),
    })


class TestResampleOHLCV:
    def test_m1_to_m15(self):
        """M1 → M15 should reduce bar count by 15x."""
        m1 = _make_m1_data(960)
        m15 = resample_ohlcv(m1, "M1", "M15")
        assert len(m15) == 64  # 960 / 15

    def test_m1_to_h1(self):
        """M1 → H1 should reduce bar count by 60x."""
        m1 = _make_m1_data(960)
        h1 = resample_ohlcv(m1, "M1", "H1")
        assert len(h1) == 16  # 960 / 60

    def test_same_timeframe_passthrough(self):
        """Same source and target should return copy."""
        m1 = _make_m1_data(100)
        result = resample_ohlcv(m1, "M1", "M1")
        assert len(result) == 100

    def test_upsample_raises(self):
        """Upsampling (H1 → M1) should raise ValueError."""
        m1 = _make_m1_data(100)
        with pytest.raises(ValueError, match="Cannot upsample"):
            resample_ohlcv(m1, "H1", "M1")

    def test_ohlcv_aggregation_correct(self):
        """Resampled OHLCV should use correct aggregation."""
        m1 = _make_m1_data(15)  # Exactly 1 M15 bar
        m15 = resample_ohlcv(m1, "M1", "M15")
        assert len(m15) == 1

        # Open should be first bar's open
        assert m15["open"].iloc[0] == pytest.approx(m1["open"].iloc[0])
        # High should be max of all bars
        assert m15["high"].iloc[0] == pytest.approx(m1["high"].max())
        # Low should be min of all bars
        assert m15["low"].iloc[0] == pytest.approx(m1["low"].min())
        # Close should be last bar's close
        assert m15["close"].iloc[0] == pytest.approx(m1["close"].iloc[-1])
        # Volume should be sum
        assert m15["volume"].iloc[0] == pytest.approx(m1["volume"].sum())

    def test_handles_capitalized_columns(self):
        """Should work with capitalized column names (Open, High, etc.)."""
        m1 = _make_m1_data(60)
        m1 = m1.rename(columns={
            "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume",
        })
        h1 = resample_ohlcv(m1, "M1", "H1")
        assert len(h1) == 1

    def test_handles_datetime_index(self):
        """Should work with datetime index (no timestamp column)."""
        m1 = _make_m1_data(60)
        m1 = m1.set_index("timestamp")
        h1 = resample_ohlcv(m1, "M1", "H1")
        assert len(h1) == 1


# =============================================================================
# INSTRUMENT CONFIG WITH DIFFERENT TIMEFRAMES
# =============================================================================

class TestInstrumentConfigTimeframes:
    def test_m15_default(self):
        config = InstrumentConfig(timeframe="M15", bars_per_day=96)
        assert config.har_daily == 96
        assert config.har_weekly == 480

    def test_h1_config(self):
        config = InstrumentConfig(timeframe="H1", bars_per_day=24)
        assert config.har_daily == 24
        assert config.har_weekly == 120
        assert config.har_monthly == 528

    def test_h4_config(self):
        config = InstrumentConfig(timeframe="H4", bars_per_day=6)
        assert config.har_daily == 6
        assert config.har_weekly == 30
        assert config.har_monthly == 132

    def test_d1_config(self):
        config = InstrumentConfig(timeframe="D1", bars_per_day=1)
        assert config.har_daily == 1
        assert config.har_weekly == 5
        assert config.har_monthly == 22

    def test_custom_bars_per_day(self):
        """Custom bars_per_day should override auto-compute."""
        config = InstrumentConfig(timeframe="M15", bars_per_day=28)  # Equity
        assert config.har_daily == 28


# =============================================================================
# MULTI-TIMEFRAME SCANNER
# =============================================================================

class TestMultiTimeframeScanner:
    def test_mixed_timeframes(self):
        """Scanner should support mixed timeframes via InstrumentConfig."""
        registry = {
            "XAUUSD_M15": InstrumentConfig(symbol="XAUUSD", timeframe="M15", bars_per_day=96),
            "XAUUSD_H1": InstrumentConfig(symbol="XAUUSD", timeframe="H1", bars_per_day=24),
            "EURUSD_M15": InstrumentConfig(symbol="EURUSD", timeframe="M15", bars_per_day=96),
        }

        multi = MultiSymbolScanner(
            symbols=["XAUUSD_M15", "XAUUSD_H1", "EURUSD_M15"],
            instrument_registry=registry,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )

        assert len(multi.scanners) == 3
        assert multi.scanners["XAUUSD_M15"]._timeframe == "M15"
        assert multi.scanners["XAUUSD_H1"]._timeframe == "H1"

    def test_per_timeframe_vol_forecaster(self):
        """Each timeframe should get its own vol forecaster with correct windows."""
        registry = {
            "XAUUSD_M15": InstrumentConfig(symbol="XAUUSD", timeframe="M15", bars_per_day=96),
            "XAUUSD_H1": InstrumentConfig(symbol="XAUUSD", timeframe="H1", bars_per_day=24),
        }

        created_configs = []
        def vol_factory(config):
            created_configs.append(config)
            return MagicMock()

        multi = MultiSymbolScanner(
            symbols=["XAUUSD_M15", "XAUUSD_H1"],
            instrument_registry=registry,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            vol_forecaster_factory=vol_factory,
        )

        assert len(created_configs) == 2
        # M15 config: har_daily=96
        m15_config = [c for c in created_configs if c.timeframe == "M15"][0]
        assert m15_config.har_daily == 96
        # H1 config: har_daily=24
        h1_config = [c for c in created_configs if c.timeframe == "H1"][0]
        assert h1_config.har_daily == 24
