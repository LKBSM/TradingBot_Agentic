# =============================================================================
# Sprint 10 Validation: Volatility-Adjusted Slippage Model (LIVE-2)
# =============================================================================
# Verifies that:
# 1. DynamicSlippageModel scales slippage with ATR
# 2. Environment wires the model correctly
# 3. Slippage increases during high-volatility bars
# 4. Slippage stays at base during normal conditions
# 5. Backward compatible when disabled
#
# Run with: python -m pytest tests/test_sprint10_dynamic_slippage.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import USE_DYNAMIC_SLIPPAGE, SLIPPAGE_ATR_SCALE, SLIPPAGE_PERCENTAGE
from src.environment.execution_model import DynamicSlippageModel


def _make_data(n_rows=800, base_price=2000.0, atr_values=None):
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 0.5, n_rows))
    if atr_values is None:
        atr_values = np.full(n_rows, 10.0)
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * 0.999,
        'High': prices * 1.003,
        'Low': prices * 0.997,
        'Close': prices,
        'Volume': np.full(n_rows, 500),
        'ATR': atr_values,
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(**kwargs):
    df_kwargs = {k: v for k, v in kwargs.items() if k in ('n_rows', 'base_price', 'atr_values')}
    env_kwargs = {k: v for k, v in kwargs.items() if k not in ('n_rows', 'base_price', 'atr_values')}
    df = _make_data(**df_kwargs)
    from src.environment.environment import TradingEnv
    return TradingEnv(df, strict_scaler_mode=False, **env_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config constants exist
# ─────────────────────────────────────────────────────────────────────────────
def test_config_constants():
    """USE_DYNAMIC_SLIPPAGE and SLIPPAGE_ATR_SCALE should exist."""
    assert USE_DYNAMIC_SLIPPAGE is True
    assert SLIPPAGE_ATR_SCALE == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: ATR == median → slippage == base
# ─────────────────────────────────────────────────────────────────────────────
def test_atr_equals_median_gives_base():
    """When ATR equals median, slippage should equal base."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)
    slippage = model.get_slippage(current_atr=10.0, median_atr=10.0)
    assert slippage == 0.0001, f"Expected 0.0001, got {slippage}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: ATR == 2x median → slippage == 2x base
# ─────────────────────────────────────────────────────────────────────────────
def test_atr_double_gives_double_slippage():
    """When ATR is 2x median, slippage should be 2x base."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)
    slippage = model.get_slippage(current_atr=20.0, median_atr=10.0)
    assert abs(slippage - 0.0002) < 1e-10, f"Expected 0.0002, got {slippage}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: ATR == 5x median → slippage == 5x base
# ─────────────────────────────────────────────────────────────────────────────
def test_atr_5x_gives_5x_slippage():
    """When ATR is 5x median (extreme vol), slippage should be 5x base."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)
    slippage = model.get_slippage(current_atr=50.0, median_atr=10.0)
    assert abs(slippage - 0.0005) < 1e-10, f"Expected 0.0005, got {slippage}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: ATR below median → slippage stays at base (floor)
# ─────────────────────────────────────────────────────────────────────────────
def test_atr_below_median_gives_base():
    """When ATR < median (calm market), slippage should NOT go below base."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)
    slippage = model.get_slippage(current_atr=5.0, median_atr=10.0)
    assert slippage == 0.0001, f"Expected base 0.0001, got {slippage}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: No median → fallback to base
# ─────────────────────────────────────────────────────────────────────────────
def test_no_median_gives_base():
    """When median_atr is None or zero, should return base slippage."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=1.0)

    assert model.get_slippage(10.0, None) == 0.0001
    assert model.get_slippage(10.0, 0.0) == 0.0001


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Environment initializes slippage model
# ─────────────────────────────────────────────────────────────────────────────
def test_env_has_slippage_model():
    """TradingEnv should have _slippage_model and _median_atr when enabled."""
    env = _make_env()
    env.reset()

    assert hasattr(env, '_slippage_model')
    assert hasattr(env, '_median_atr')
    assert env._use_dynamic_slippage is True
    assert env._slippage_model is not None
    assert env._median_atr is not None and env._median_atr > 0
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Dynamic slippage disabled falls back to static
# ─────────────────────────────────────────────────────────────────────────────
def test_disabled_gives_static_slippage():
    """When use_dynamic_slippage=False, slippage should be static."""
    env = _make_env(use_dynamic_slippage=False)
    env.reset()

    assert env._use_dynamic_slippage is False
    slippage = env._get_current_slippage()
    assert slippage == env.slippage_percentage
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: sqrt scale factor gives conservative scaling
# ─────────────────────────────────────────────────────────────────────────────
def test_sqrt_scale_factor():
    """With atr_scale=0.5, ATR=4x → slippage=2x (sqrt scaling)."""
    model = DynamicSlippageModel(base_slippage=0.0001, atr_scale_factor=0.5)
    slippage = model.get_slippage(current_atr=40.0, median_atr=10.0)
    # 4^0.5 = 2.0
    assert abs(slippage - 0.0002) < 1e-10, f"Expected 0.0002, got {slippage}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
