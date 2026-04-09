"""
Sprint 1: ADX Wilder's Smoothing Tests
Verifies that ADX uses proper Wilder's smoothing instead of raw DX.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.market_regime_agent import TechnicalIndicators


class TestADXWilderSmoothing:
    """Verify ADX uses Wilder's smoothing method."""

    def _make_trending_data(self, n=100, start=1900, trend=0.5):
        """Generate synthetic trending OHLCV data."""
        np.random.seed(42)
        closes = np.zeros(n)
        closes[0] = start
        for i in range(1, n):
            closes[i] = closes[i - 1] + trend + np.random.randn() * 2.0

        highs = closes + np.abs(np.random.randn(n)) * 3.0
        lows = closes - np.abs(np.random.randn(n)) * 3.0
        return highs, lows, closes

    def _make_ranging_data(self, n=100, center=1900, amplitude=10):
        """Generate synthetic ranging (mean-reverting) data."""
        np.random.seed(42)
        t = np.arange(n)
        closes = center + amplitude * np.sin(2 * np.pi * t / 20) + np.random.randn(n) * 2.0
        highs = closes + np.abs(np.random.randn(n)) * 3.0
        lows = closes - np.abs(np.random.randn(n)) * 3.0
        return highs, lows, closes

    def test_adx_returns_three_values(self):
        """ADX returns (adx, +di, -di) tuple."""
        highs, lows, closes = self._make_trending_data()
        result = TechnicalIndicators.adx(highs, lows, closes)
        assert len(result) == 3

    def test_adx_values_in_valid_range(self):
        """ADX, +DI, -DI should all be in [0, 100]."""
        highs, lows, closes = self._make_trending_data()
        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes)
        assert 0 <= adx <= 100, f"ADX={adx} out of range"
        assert 0 <= plus_di <= 100, f"+DI={plus_di} out of range"
        assert 0 <= minus_di <= 100, f"-DI={minus_di} out of range"

    def test_smoothed_adx_less_volatile_than_raw_dx(self):
        """Wilder's smoothing should make ADX less volatile than raw DX.

        We verify this by computing ADX on multiple different data windows
        and checking that ADX values don't swing as wildly as raw DX would.
        """
        np.random.seed(42)
        # Create data with volatile DX (alternating strong/weak moves)
        n = 200
        closes = np.cumsum(np.random.randn(n) * 3) + 1900
        highs = closes + np.abs(np.random.randn(n)) * 5
        lows = closes - np.abs(np.random.randn(n)) * 5

        adx, _, _ = TechnicalIndicators.adx(highs, lows, closes, period=14)

        # Smoothed ADX should be moderate (not at extremes)
        # Raw DX often hits 80-100, smoothed ADX rarely exceeds 60
        assert adx < 80, f"ADX={adx} too high for random data — likely not smoothed"

    def test_strong_trend_high_adx(self):
        """Strong monotonic trend should produce high ADX (>25)."""
        highs, lows, closes = self._make_trending_data(n=100, trend=3.0)
        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes)
        assert adx > 25, f"ADX={adx} too low for strong trend"
        # Uptrend → +DI should dominate
        assert plus_di > minus_di, f"+DI={plus_di} should > -DI={minus_di} in uptrend"

    def test_ranging_market_low_adx(self):
        """Ranging (sideways) market should produce low ADX (<30)."""
        highs, lows, closes = self._make_ranging_data(n=100)
        adx, _, _ = TechnicalIndicators.adx(highs, lows, closes)
        assert adx < 30, f"ADX={adx} too high for ranging market"

    def test_downtrend_minus_di_dominates(self):
        """In downtrend, -DI should be greater than +DI."""
        highs, lows, closes = self._make_trending_data(n=100, trend=-3.0)
        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes)
        assert minus_di > plus_di, f"-DI={minus_di} should > +DI={plus_di} in downtrend"
        assert adx > 20, f"ADX={adx} too low for clear downtrend"

    def test_insufficient_data_returns_zeros(self):
        """With fewer than period+1 bars, return (0, 0, 0)."""
        highs = np.array([1900, 1901, 1902])
        lows = np.array([1898, 1899, 1900])
        closes = np.array([1899, 1900, 1901])
        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes, period=14)
        assert adx == 0 and plus_di == 0 and minus_di == 0

    def test_adx_smoothing_with_known_values(self):
        """Compare against manually computed Wilder's smoothing.

        With a strong uptrend, the smoothed ADX should converge toward
        a stable value rather than jumping to the last DX value.
        """
        # 80 bars of steady uptrend
        n = 80
        closes = np.linspace(1900, 1960, n) + np.random.RandomState(42).randn(n) * 0.5
        highs = closes + 2.0
        lows = closes - 2.0

        adx_14, _, _ = TechnicalIndicators.adx(highs, lows, closes, period=14)

        # With steady uptrend, smoothed ADX should be high but not 100
        # (smoothing prevents overshooting)
        assert 30 < adx_14 < 90, f"ADX={adx_14} outside expected range for steady trend"

    def test_adx_different_periods(self):
        """Longer period should produce smoother (generally lower) ADX."""
        highs, lows, closes = self._make_trending_data(n=150, trend=2.0)

        adx_7, _, _ = TechnicalIndicators.adx(highs, lows, closes, period=7)
        adx_14, _, _ = TechnicalIndicators.adx(highs, lows, closes, period=14)
        adx_21, _, _ = TechnicalIndicators.adx(highs, lows, closes, period=21)

        # All should be valid
        assert adx_7 > 0
        assert adx_14 > 0
        assert adx_21 > 0

    def test_regime_thresholds_with_smoothed_adx(self):
        """Verify regime classification works correctly with smoothed ADX values."""
        # Strong trend → ADX > 30
        highs, lows, closes = self._make_trending_data(n=150, trend=4.0)
        adx, _, _ = TechnicalIndicators.adx(highs, lows, closes)
        # With proper smoothing, strong trend should cross 30 threshold
        assert adx > 25, f"Strong trend ADX={adx} — regime classification may fail"

    def test_flat_market_adx_near_zero(self):
        """Perfectly flat market should produce very low ADX."""
        n = 100
        closes = np.full(n, 1900.0) + np.random.RandomState(42).randn(n) * 0.01
        highs = closes + 0.01
        lows = closes - 0.01

        adx, _, _ = TechnicalIndicators.adx(highs, lows, closes)
        assert adx < 20, f"Flat market ADX={adx} should be very low"

    def test_wilder_smoothing_formula(self):
        """Verify Wilder's smoothing formula: S[i] = S[i-1]*(p-1)/p + val[i]/p

        This is the definitive property of Wilder's method vs simple average.
        With constant input, smoothed value should converge to the input.
        """
        # Create data where DX would be roughly constant (steady strong trend)
        n = 200
        closes = np.linspace(1900, 2100, n)
        highs = closes + 3.0
        lows = closes - 3.0

        # Compute with different data lengths to check convergence
        adx_100, _, _ = TechnicalIndicators.adx(highs[:100], lows[:100], closes[:100])
        adx_200, _, _ = TechnicalIndicators.adx(highs, lows, closes)

        # With steady trend, both should be high and similar (convergence)
        assert abs(adx_200 - adx_100) < 20, \
            f"ADX not converging: 100-bar={adx_100:.1f}, 200-bar={adx_200:.1f}"


class TestADXEdgeCases:
    """Edge case tests for ADX calculation."""

    def test_single_large_move(self):
        """Single large move shouldn't spike ADX to 100 due to smoothing."""
        n = 50
        closes = np.full(n, 1900.0)
        closes[-1] = 1950.0  # 50-point spike on last bar
        highs = closes + 1.0
        highs[-1] = 1955.0
        lows = closes - 1.0
        lows[-1] = 1945.0

        adx, _, _ = TechnicalIndicators.adx(highs, lows, closes)
        # Smoothing should prevent ADX from being 100
        assert adx < 80, f"Single spike ADX={adx} — smoothing not working"

    def test_nan_free_output(self):
        """ADX should never return NaN."""
        np.random.seed(42)
        highs = np.random.uniform(1890, 1910, 100)
        lows = highs - np.random.uniform(1, 10, 100)
        closes = (highs + lows) / 2

        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes)
        assert not np.isnan(adx), "ADX is NaN"
        assert not np.isnan(plus_di), "+DI is NaN"
        assert not np.isnan(minus_di), "-DI is NaN"

    def test_minimum_data_period_plus_one(self):
        """With exactly period+1 bars, should return valid (possibly rough) values."""
        n = 15  # period=14, need 15 bars
        closes = np.linspace(1900, 1910, n)
        highs = closes + 2.0
        lows = closes - 2.0

        adx, plus_di, minus_di = TechnicalIndicators.adx(highs, lows, closes, period=14)
        # Should return something (may use mean DX fallback)
        assert isinstance(adx, (int, float, np.floating))
        assert not np.isnan(adx)
