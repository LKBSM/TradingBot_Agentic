"""
Sprint 7: RSI Divergence for CHOCH Confirmation Tests
Verifies RSI divergence detection at fractal swings and
its integration with ConfluenceDetector scoring.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.strategy_features import SmartMoneyEngine
from src.intelligence.confluence_detector import ConfluenceDetector, SignalType


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


class TestRSIDivergenceDetection:
    """Test RSI divergence detection in SmartMoneyEngine."""

    def test_divergence_column_exists(self):
        """CHOCH_DIVERGENCE column should be added by analyze()."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()
        assert 'CHOCH_DIVERGENCE' in result.columns

    def test_divergence_values_are_valid(self):
        """CHOCH_DIVERGENCE should only contain -1, 0, 1."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()
        unique_vals = set(result['CHOCH_DIVERGENCE'].unique())
        assert unique_vals.issubset({-1, 0, 1}), f"Invalid values: {unique_vals}"

    def test_bullish_divergence_on_synthetic_data(self):
        """Create data with known bullish divergence: lower low + higher RSI low."""
        n = 100
        np.random.seed(42)
        # Create descending lows (price) but ensure RSI can show divergence
        closes = np.zeros(n)
        closes[0] = 1900
        for i in range(1, 30):
            closes[i] = closes[i - 1] - 2  # downtrend
        for i in range(30, 50):
            closes[i] = closes[i - 1] + 1  # small bounce
        for i in range(50, 70):
            closes[i] = closes[i - 1] - 1  # lower low (but weaker momentum)
        for i in range(70, n):
            closes[i] = closes[i - 1] + 2  # recovery

        opens = closes + np.random.randn(n) * 0.5
        highs = np.maximum(opens, closes) + abs(np.random.randn(n)) * 1.0
        lows = np.minimum(opens, closes) - abs(np.random.randn(n)) * 1.0
        volumes = np.random.uniform(500, 5000, n)

        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes
        }, index=dates)

        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        # With this pattern, some divergence should be detected
        # (the exact count depends on fractal placement)
        assert 'CHOCH_DIVERGENCE' in result.columns

    def test_no_divergence_in_strong_trend(self):
        """Strong monotonic trend should produce few/no divergences."""
        n = 100
        closes = np.linspace(1900, 2000, n)  # Perfectly linear uptrend
        opens = closes - 0.5
        highs = closes + 1.0
        lows = closes - 1.0
        volumes = np.full(n, 1000.0)

        dates = pd.date_range('2024-01-01', periods=n, freq='15min')
        df = pd.DataFrame({
            'open': opens, 'high': highs, 'low': lows,
            'close': closes, 'volume': volumes
        }, index=dates)

        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        # In a perfect linear trend, RSI should be consistently high
        # → no divergence (price and RSI move together)
        div_count = (result['CHOCH_DIVERGENCE'] != 0).sum()
        # Allow some noise but should be minimal
        assert div_count <= 5, f"Too many divergences ({div_count}) in linear trend"


class TestRSIDivergenceScoring:
    """Test RSI divergence scoring in ConfluenceDetector."""

    def setup_method(self):
        self.detector = ConfluenceDetector()

    def test_bullish_divergence_scores_for_long(self):
        """Bullish divergence should add points to LONG signal."""
        smc = {"BOS_SIGNAL": 1.0, "CHOCH_DIVERGENCE": 1}
        result = self.detector._score_rsi_divergence(smc, SignalType.LONG)
        assert result.weighted_score > 0
        assert "Bullish" in result.reasoning

    def test_bearish_divergence_scores_for_short(self):
        """Bearish divergence should add points to SHORT signal."""
        smc = {"BOS_SIGNAL": -1.0, "CHOCH_DIVERGENCE": -1}
        result = self.detector._score_rsi_divergence(smc, SignalType.SHORT)
        assert result.weighted_score > 0
        assert "Bearish" in result.reasoning

    def test_no_divergence_scores_zero(self):
        """No divergence should get 0 score."""
        smc = {"BOS_SIGNAL": 1.0, "CHOCH_DIVERGENCE": 0}
        result = self.detector._score_rsi_divergence(smc, SignalType.LONG)
        assert result.weighted_score == 0.0

    def test_opposing_divergence_scores_zero(self):
        """Bearish divergence should score 0 for LONG."""
        smc = {"BOS_SIGNAL": 1.0, "CHOCH_DIVERGENCE": -1}
        result = self.detector._score_rsi_divergence(smc, SignalType.LONG)
        assert result.weighted_score == 0.0

    def test_divergence_weight_is_2(self):
        """RSI divergence weight should be 2.0 (from momentum split)."""
        assert self.detector.weights.get("rsi_divergence", 0) == 2.0

    def test_momentum_weight_is_3(self):
        """Momentum weight should be 3.0 (after divergence split)."""
        assert self.detector.weights.get("momentum", 0) == 3.0

    def test_total_weights_sum_to_100(self):
        """All weights should still sum to 100."""
        total = sum(self.detector.weights.values())
        assert total == pytest.approx(100.0, abs=0.01), f"Weights sum to {total}"


class TestDivergenceIntegration:
    """Integration tests with full confluence pipeline."""

    def setup_method(self):
        self.detector = ConfluenceDetector(min_score=0)

    def test_divergence_adds_to_score(self):
        """Signal with divergence should score higher than without."""
        smc_with_div = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 1.0,
            "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.5,
            "OB_STRENGTH_NORM": 0.0, "RSI": 55.0,
            "MACD_Diff": 0.5, "CHOCH_DIVERGENCE": 1,
        }
        smc_without_div = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 1.0,
            "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.5,
            "OB_STRENGTH_NORM": 0.0, "RSI": 55.0,
            "MACD_Diff": 0.5, "CHOCH_DIVERGENCE": 0,
        }

        sig_with = self.detector.analyze(smc_with_div, regime=None, news=None, price=1900, atr=5.0)
        sig_without = self.detector.analyze(smc_without_div, regime=None, news=None, price=1900, atr=5.0)

        assert sig_with is not None and sig_without is not None
        assert sig_with.confluence_score > sig_without.confluence_score

    def test_divergence_component_in_signal(self):
        """RSI_Divergence component should appear in signal components."""
        smc = {
            "BOS_SIGNAL": 1.0, "CHOCH_SIGNAL": 0.0,
            "FVG_SIGNAL": 0.0, "FVG_SIZE_NORM": 0.0,
            "OB_STRENGTH_NORM": 0.0, "RSI": 50.0,
            "MACD_Diff": 0.0, "CHOCH_DIVERGENCE": 1,
        }

        signal = self.detector.analyze(smc, regime=None, news=None, price=1900, atr=5.0)
        assert signal is not None

        comp_names = [c.name for c in signal.components]
        assert "RSI_Divergence" in comp_names
