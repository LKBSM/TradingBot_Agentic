"""
Sprint 6: Standardized Indicator Periods Tests
Verifies RSI=14, MACD 12/26/9, ATR=14 are the new defaults.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.strategy_features import SmartMoneyEngine, SMCConfig


def _make_ohlcv_data(n=100, base_price=1900.0, seed=42):
    """Generate synthetic OHLCV data."""
    np.random.seed(seed)
    dates = pd.date_range('2024-01-01', periods=n, freq='15min')

    closes = np.zeros(n)
    closes[0] = base_price
    for i in range(1, n):
        closes[i] = closes[i - 1] + np.random.randn() * 3.0

    opens = closes + np.random.randn(n) * 1.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n)) * 2.0
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n)) * 2.0
    volumes = np.random.uniform(500, 5000, n)

    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes
    }, index=dates)


class TestDefaultPeriods:
    """Verify new standard defaults."""

    def test_rsi_default_14(self):
        config = SMCConfig()
        assert config.RSI_WINDOW == 14

    def test_macd_fast_default_12(self):
        config = SMCConfig()
        assert config.MACD_FAST == 12

    def test_macd_slow_default_26(self):
        config = SMCConfig()
        assert config.MACD_SLOW == 26

    def test_macd_signal_default_9(self):
        config = SMCConfig()
        assert config.MACD_SIGNAL == 9

    def test_atr_default_14(self):
        config = SMCConfig()
        assert config.ATR_WINDOW == 14

    def test_bb_default_20(self):
        config = SMCConfig()
        assert config.BB_WINDOW == 20

    def test_fractal_window_default_2(self):
        config = SMCConfig()
        assert config.FRACTAL_WINDOW == 2


class TestIndicatorCalculation:
    """Verify indicators calculate correctly with new periods."""

    def test_rsi_14_produces_valid_range(self):
        """RSI with period 14 should produce values in [0, 100]."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        rsi = result['RSI'].dropna()
        assert len(rsi) > 0
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_macd_12_26_9_produces_signal(self):
        """MACD 12/26/9 should produce valid diff values."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        macd_diff = result['MACD_Diff'].dropna()
        assert len(macd_diff) > 0
        assert not macd_diff.isnull().all()

    def test_atr_14_produces_positive_values(self):
        """ATR with period 14 should produce positive values."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        atr = result['ATR'].dropna()
        assert len(atr) > 0
        assert (atr > 0).all()

    def test_engine_handles_minimum_data(self):
        """Engine should handle data with just enough bars for ATR/MACD."""
        # MACD slow=26 needs at least 27 bars + signal line
        df = _make_ohlcv_data(40, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        assert 'RSI' in result.columns
        assert 'MACD_Diff' in result.columns
        assert 'ATR' in result.columns

    def test_custom_periods_still_work(self):
        """Custom periods should override defaults."""
        df = _make_ohlcv_data(100, seed=42)
        config = {
            'RSI_WINDOW': 10,
            'MACD_FAST': 8,
            'MACD_SLOW': 17,
            'ATR_WINDOW': 7,
        }
        engine = SmartMoneyEngine(df.copy(), config)
        assert engine.config.RSI_WINDOW == 10
        assert engine.config.MACD_FAST == 8
        assert engine.config.MACD_SLOW == 17
        assert engine.config.ATR_WINDOW == 7

        result = engine.analyze()
        assert 'RSI' in result.columns


class TestPeriodConsistency:
    """Verify period changes don't break pipeline."""

    def test_full_pipeline_with_new_defaults(self):
        """Full analyze() pipeline should work with new standard periods."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        expected_cols = [
            'RSI', 'MACD_Diff', 'ATR', 'BB_L', 'BB_M', 'BB_H',
            'BOS_SIGNAL', 'CHOCH_SIGNAL', 'FVG_SIGNAL',
            'BULLISH_OB_HIGH', 'BEARISH_OB_HIGH', 'OB_STRENGTH_NORM'
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_explosion(self):
        """Longer periods shouldn't cause excessive NaN rows."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        # With ATR=14 and MACD_SLOW=26, first ~30 rows may have NaN
        # But the rest should be complete
        rsi_valid = result['RSI'].dropna()
        assert len(rsi_valid) >= 60, f"Too many RSI NaN: {100 - len(rsi_valid)}"

        atr_valid = result['ATR'].dropna()
        assert len(atr_valid) >= 70, f"Too many ATR NaN: {100 - len(atr_valid)}"
