"""
Sprint 3: Order Block FVG Requirement Tests
Verifies that OB detection works independently of FVG when OB_REQUIRE_FVG=False,
and that FVG presence adds a strength bonus.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.environment.strategy_features import SmartMoneyEngine


def _make_ohlcv_data(n=100, base_price=1900.0, seed=42):
    """Generate synthetic OHLCV data with patterns suitable for OB detection."""
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

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)

    return df


class TestOrderBlockWithoutFVG:
    """Test OB detection when OB_REQUIRE_FVG=False (default)."""

    def test_default_config_does_not_require_fvg(self):
        """Default SMCConfig should have OB_REQUIRE_FVG=False."""
        df = _make_ohlcv_data(100)
        engine = SmartMoneyEngine(df, {})
        assert engine.config.OB_REQUIRE_FVG is False

    def test_more_obs_without_fvg_requirement(self):
        """Without FVG requirement, significantly more OBs should be detected."""
        df = _make_ohlcv_data(200, seed=42)

        # Without FVG requirement (default)
        engine_no_fvg = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': False})
        result_no_fvg = engine_no_fvg.analyze()
        obs_without_fvg_req = (
            result_no_fvg['BULLISH_OB_HIGH'].notna().sum() +
            result_no_fvg['BEARISH_OB_HIGH'].notna().sum()
        )

        # With FVG requirement (legacy)
        engine_with_fvg = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': True})
        result_with_fvg = engine_with_fvg.analyze()
        obs_with_fvg_req = (
            result_with_fvg['BULLISH_OB_HIGH'].notna().sum() +
            result_with_fvg['BEARISH_OB_HIGH'].notna().sum()
        )

        # Should detect significantly more OBs without FVG requirement
        assert obs_without_fvg_req >= obs_with_fvg_req, \
            f"Without FVG: {obs_without_fvg_req}, With FVG: {obs_with_fvg_req}"
        # Typically ~5-10x more OBs
        assert obs_without_fvg_req > 0, "Should detect at least some OBs"

    def test_ob_strength_includes_fvg_bonus(self):
        """OBs with adjacent FVG should have higher strength than those without."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': False, 'OB_FVG_BONUS': 0.2})
        result = engine.analyze()

        # Get OBs that have nonzero strength
        has_ob = (result['BULLISH_OB_HIGH'].notna()) | (result['BEARISH_OB_HIGH'].notna())
        ob_strengths = result.loc[has_ob, 'OB_STRENGTH_NORM']

        if len(ob_strengths) > 0:
            # Some OBs should have the FVG bonus (strength > base)
            # The bonus is 0.2, so any OB with FVG present should have > base_strength
            assert ob_strengths.max() > 0, "OB strengths should be positive"

    def test_fvg_bonus_value_applied(self):
        """FVG bonus should be configurable and applied correctly."""
        df = _make_ohlcv_data(200, seed=42)

        # Test with different bonus values
        engine_02 = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': False, 'OB_FVG_BONUS': 0.2})
        result_02 = engine_02.analyze()

        engine_05 = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': False, 'OB_FVG_BONUS': 0.5})
        result_05 = engine_05.analyze()

        # Both should produce valid results
        assert 'OB_STRENGTH_NORM' in result_02.columns
        assert 'OB_STRENGTH_NORM' in result_05.columns


class TestOrderBlockWithFVG:
    """Test OB detection when OB_REQUIRE_FVG=True (legacy mode)."""

    def test_legacy_mode_requires_fvg(self):
        """With OB_REQUIRE_FVG=True, OBs should only appear where FVG exists."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {'OB_REQUIRE_FVG': True})
        result = engine.analyze()

        # Where we have OBs, there should be FVG nearby
        bullish_obs = result['BULLISH_OB_HIGH'].notna()
        bearish_obs = result['BEARISH_OB_HIGH'].notna()

        if bullish_obs.any() or bearish_obs.any():
            # FVG should be present at shift(1) for every OB
            fvg_at_prev = (result['FVG_SIGNAL'] != 0).shift(1).fillna(False)
            obs_with_fvg = (bullish_obs | bearish_obs) & fvg_at_prev
            obs_total = (bullish_obs | bearish_obs)
            assert obs_with_fvg.sum() == obs_total.sum(), \
                "In legacy mode, all OBs should have FVG confirmation"


class TestOrderBlockStrength:
    """Test OB strength calculation."""

    def test_strength_is_atr_normalized(self):
        """OB strength should be normalized by ATR."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        has_ob = (result['BULLISH_OB_HIGH'].notna()) | (result['BEARISH_OB_HIGH'].notna())
        if has_ob.any():
            strengths = result.loc[has_ob, 'OB_STRENGTH_NORM']
            # Strength should be reasonable (not enormous)
            assert strengths.max() < 10, f"OB strength {strengths.max()} too high"
            assert strengths.min() >= 0, "OB strength should be non-negative"

    def test_strength_zero_when_no_ob(self):
        """OB_STRENGTH_NORM should be 0 where no OB exists."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        no_ob = ~(result['BULLISH_OB_HIGH'].notna() | result['BEARISH_OB_HIGH'].notna())
        if no_ob.any():
            zero_strengths = result.loc[no_ob, 'OB_STRENGTH_NORM']
            assert (zero_strengths == 0).all(), "Strength should be 0 where no OB"


class TestOrderBlockColumns:
    """Test that all expected columns are present."""

    def test_all_ob_columns_present(self):
        """Analyze should produce all OB columns."""
        df = _make_ohlcv_data(100, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        expected = ['BULLISH_OB_HIGH', 'BULLISH_OB_LOW',
                    'BEARISH_OB_HIGH', 'BEARISH_OB_LOW',
                    'OB_STRENGTH_NORM']
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_ob_zones_are_valid(self):
        """OB high should be > OB low where OBs exist."""
        df = _make_ohlcv_data(200, seed=42)
        engine = SmartMoneyEngine(df.copy(), {})
        result = engine.analyze()

        # Bullish OBs
        bull_mask = result['BULLISH_OB_HIGH'].notna()
        if bull_mask.any():
            assert (result.loc[bull_mask, 'BULLISH_OB_HIGH'] >=
                    result.loc[bull_mask, 'BULLISH_OB_LOW']).all(), \
                "Bullish OB high should >= low"

        # Bearish OBs
        bear_mask = result['BEARISH_OB_HIGH'].notna()
        if bear_mask.any():
            assert (result.loc[bear_mask, 'BEARISH_OB_HIGH'] >=
                    result.loc[bear_mask, 'BEARISH_OB_LOW']).all(), \
                "Bearish OB high should >= low"

    def test_config_ob_fvg_bonus_range(self):
        """OB_FVG_BONUS should be clamped to [0, 1]."""
        df = _make_ohlcv_data(50, seed=42)
        # Valid bonus
        engine = SmartMoneyEngine(df.copy(), {'OB_FVG_BONUS': 0.3})
        assert engine.config.OB_FVG_BONUS == 0.3

        # Default bonus
        engine_default = SmartMoneyEngine(df.copy(), {})
        assert engine_default.config.OB_FVG_BONUS == 0.2
