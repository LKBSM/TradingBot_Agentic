"""Tests for Sprint 6: Multi-Instrument Support.

Tests cover:
  - InstrumentRegistry provides presets for 6 instruments
  - Per-instrument SL/TP multipliers in ConfluenceDetector
  - ConfluenceDetector uses instrument_config when provided
  - MultiSymbolScanner creation and per-symbol state
  - MultiSymbolScanner scan_all_once / scan_symbol
  - MultiSymbolScanner calibrate_forecasters
  - Backward compatibility: default ConfluenceDetector works without config
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    SL_ATR_MULT,
    SignalTier,
    SignalType,
    TP_ATR_MULT,
)
from src.intelligence.volatility_forecaster import (
    InstrumentConfig,
    VolatilityForecaster,
    get_instrument_registry,
)
from src.intelligence.sentinel_scanner import MultiSymbolScanner, SentinelScanner


# =============================================================================
# INSTRUMENT REGISTRY TESTS
# =============================================================================

class TestInstrumentRegistry:
    def test_registry_contains_xauusd(self):
        reg = get_instrument_registry()
        assert "XAUUSD" in reg
        assert reg["XAUUSD"].symbol == "XAUUSD"

    def test_registry_contains_eurusd(self):
        reg = get_instrument_registry()
        assert "EURUSD" in reg
        assert reg["EURUSD"].sl_atr_mult == 1.5
        assert reg["EURUSD"].tp_atr_mult == 3.0

    def test_registry_contains_btcusd(self):
        reg = get_instrument_registry()
        assert "BTCUSD" in reg

    def test_registry_contains_us500(self):
        reg = get_instrument_registry()
        assert "US500" in reg
        assert reg["US500"].bars_per_day == 28  # Equity market hours

    def test_registry_contains_gbpusd(self):
        reg = get_instrument_registry()
        assert "GBPUSD" in reg

    def test_registry_contains_usdjpy(self):
        reg = get_instrument_registry()
        assert "USDJPY" in reg

    def test_registry_has_6_instruments(self):
        reg = get_instrument_registry()
        assert len(reg) == 6

    def test_each_instrument_has_session_hours(self):
        reg = get_instrument_registry()
        for symbol, config in reg.items():
            assert len(config.session_hours) >= 2, f"{symbol} missing session hours"

    def test_each_instrument_has_calendar_events(self):
        reg = get_instrument_registry()
        for symbol, config in reg.items():
            assert len(config.calendar_events) >= 2, f"{symbol} missing calendar events"

    def test_har_windows_auto_computed(self):
        reg = get_instrument_registry()
        xau = reg["XAUUSD"]
        assert xau.har_daily == 96  # 96 bars/day for M15
        assert xau.har_weekly == 480  # 96 * 5
        assert xau.har_monthly == 2112  # 96 * 22

    def test_us500_har_windows(self):
        reg = get_instrument_registry()
        us500 = reg["US500"]
        assert us500.har_daily == 28
        assert us500.har_weekly == 140


# =============================================================================
# CONFLUENCE DETECTOR — PER-INSTRUMENT SL/TP
# =============================================================================

class TestConfluenceDetectorInstrumentConfig:
    def test_default_sl_tp_mults(self):
        """Default detector uses module-level SL/TP constants."""
        detector = ConfluenceDetector()
        assert detector._sl_atr_mult == SL_ATR_MULT
        assert detector._tp_atr_mult == TP_ATR_MULT

    def test_instrument_config_overrides_sl_tp(self):
        """InstrumentConfig should override SL/TP multipliers."""
        config = InstrumentConfig(symbol="EURUSD", sl_atr_mult=1.5, tp_atr_mult=3.0)
        detector = ConfluenceDetector(instrument_config=config)

        assert detector._sl_atr_mult == 1.5
        assert detector._tp_atr_mult == 3.0

    def test_instrument_config_overrides_symbol(self):
        """InstrumentConfig should set the symbol."""
        config = InstrumentConfig(symbol="BTCUSD")
        detector = ConfluenceDetector(symbol="IGNORED", instrument_config=config)
        assert detector.symbol == "BTCUSD"

    def test_eurusd_tighter_sl_tp(self):
        """EURUSD should produce tighter SL/TP than XAUUSD for same ATR."""
        reg = get_instrument_registry()

        xau_detector = ConfluenceDetector(instrument_config=reg["XAUUSD"])
        eur_detector = ConfluenceDetector(instrument_config=reg["EURUSD"])

        # XAUUSD: SL = 2.0 * ATR, TP = 4.0 * ATR
        # EURUSD: SL = 1.5 * ATR, TP = 3.0 * ATR
        assert xau_detector._sl_atr_mult > eur_detector._sl_atr_mult
        assert xau_detector._tp_atr_mult > eur_detector._tp_atr_mult

    def test_analyze_uses_instance_mults(self):
        """analyze() should use per-instrument SL/TP multipliers."""
        config = InstrumentConfig(symbol="EURUSD", sl_atr_mult=1.5, tp_atr_mult=3.0)
        detector = ConfluenceDetector(
            instrument_config=config,
            min_score=0.0,  # Accept any signal for testing
        )

        smc_features = {
            "BOS_SIGNAL": 1.0,  # Bullish
            "FVG_SIGNAL": 1.0,
            "OB_STRENGTH_NORM": 0.8,
            "RSI": 60.0,
            "MACD_Diff": 0.001,
        }

        # Use price/ATR that round cleanly to 2 decimals
        signal = detector.analyze(
            smc_features=smc_features,
            regime=None,
            news=None,
            price=1.10,
            atr=0.02,
        )

        assert signal is not None
        # SL should be 1.5 * 0.02 = 0.03 from entry → 1.10 - 0.03 = 1.07
        assert signal.stop_loss == pytest.approx(1.07, abs=0.01)
        # TP should be 3.0 * 0.02 = 0.06 from entry → 1.10 + 0.06 = 1.16
        assert signal.take_profit == pytest.approx(1.16, abs=0.01)
        # R:R should be 3.0/1.5 = 2.0
        assert signal.rr_ratio == pytest.approx(2.0, abs=0.01)

    def test_backward_compat_no_config(self):
        """ConfluenceDetector should work without instrument_config."""
        detector = ConfluenceDetector(min_score=0.0)
        smc_features = {
            "BOS_SIGNAL": 1.0,
            "FVG_SIGNAL": 1.0,
            "OB_STRENGTH_NORM": 0.5,
            "RSI": 55.0,
            "MACD_Diff": 0.001,
        }
        signal = detector.analyze(
            smc_features=smc_features,
            regime=None,
            news=None,
            price=2000.0,
            atr=3.0,
        )
        assert signal is not None
        assert signal.symbol == "XAUUSD"  # Default


# =============================================================================
# MULTI-SYMBOL SCANNER TESTS
# =============================================================================

def _make_mock_data_provider():
    """Create a mock data provider that returns synthetic OHLCV."""
    provider = MagicMock()

    def get_ohlcv(symbol, timeframe, lookback):
        rng = np.random.RandomState(42)
        n = lookback
        ts = pd.date_range("2024-01-01", periods=n, freq="15min")
        close = 2000.0 + np.cumsum(rng.randn(n) * 0.5)
        df = pd.DataFrame({
            "Open": close + rng.randn(n) * 0.3,
            "High": close + rng.uniform(0.5, 3.0, n),
            "Low": close - rng.uniform(0.5, 3.0, n),
            "Close": close,
            "Volume": rng.uniform(100, 10000, n),
        }, index=ts)
        return df

    provider.get_ohlcv = MagicMock(side_effect=get_ohlcv)
    return provider


class TestMultiSymbolScanner:
    def test_creation_with_3_symbols(self):
        """Should create a scanner per symbol."""
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD", "BTCUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )

        assert len(multi.scanners) == 3
        assert "XAUUSD" in multi.scanners
        assert "EURUSD" in multi.scanners
        assert "BTCUSD" in multi.scanners

    def test_each_scanner_has_correct_symbol(self):
        """Each internal scanner should have the correct symbol."""
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )

        assert multi.scanners["XAUUSD"]._symbol == "XAUUSD"
        assert multi.scanners["EURUSD"]._symbol == "EURUSD"

    def test_each_scanner_has_instrument_config_sl_tp(self):
        """Each scanner's ConfluenceDetector should use per-instrument SL/TP."""
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )

        xau_det = multi.scanners["XAUUSD"]._confluence
        eur_det = multi.scanners["EURUSD"]._confluence

        assert xau_det._sl_atr_mult == 2.0
        assert eur_det._sl_atr_mult == 1.5

    def test_symbols_property(self):
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "BTCUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )
        assert set(multi.symbols) == {"XAUUSD", "BTCUSD"}

    def test_scan_symbol_unknown_raises(self):
        """scan_symbol with unknown symbol should raise KeyError."""
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )
        with pytest.raises(KeyError, match="UNKNOWN"):
            multi.scan_symbol("UNKNOWN")

    def test_vol_forecaster_factory(self):
        """vol_forecaster_factory should be called per symbol."""
        reg = get_instrument_registry()
        factory = MagicMock(return_value=MagicMock())

        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            vol_forecaster_factory=factory,
        )

        # Factory called once per symbol
        assert factory.call_count == 2
        # Each scanner should have a vol forecaster
        for scanner in multi.scanners.values():
            assert scanner._vol_forecaster is not None

    def test_get_stats_structure(self):
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD", "EURUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
        )
        multi._start_time = time.time()
        stats = multi.get_stats()

        assert "running" in stats
        assert "symbols" in stats
        assert "per_symbol" in stats
        assert "XAUUSD" in stats["per_symbol"]

    def test_calibrate_forecasters(self):
        """calibrate_forecasters should call each forecaster's calibrate()."""
        reg = get_instrument_registry()
        mock_forecaster = MagicMock()
        mock_forecaster.calibrate.return_value = {"calibrated": True}

        multi = MultiSymbolScanner(
            symbols=["XAUUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            vol_forecaster_factory=lambda config: mock_forecaster,
        )

        df = pd.DataFrame({
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="15min"),
            "open": np.ones(100), "high": np.ones(100),
            "low": np.ones(100), "close": np.ones(100),
            "volume": np.ones(100),
        })

        results = multi.calibrate_forecasters({"XAUUSD": df})
        assert results["XAUUSD"]["calibrated"] is True
        mock_forecaster.calibrate.assert_called_once()

    def test_calibrate_missing_data(self):
        """calibrate_forecasters with missing data returns reason."""
        reg = get_instrument_registry()
        multi = MultiSymbolScanner(
            symbols=["XAUUSD"],
            instrument_registry=reg,
            data_provider=MagicMock(),
            smc_factory=MagicMock(),
            regime_agent=MagicMock(),
            news_agent=MagicMock(),
            llm_engine=MagicMock(),
            cache=MagicMock(),
            signal_store=MagicMock(),
            vol_forecaster_factory=lambda c: MagicMock(),
        )

        results = multi.calibrate_forecasters({})  # No data
        assert results["XAUUSD"]["reason"] == "no data"


import time
