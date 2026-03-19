# =============================================================================
# Sprint 11 Validation: Session-Dependent Spread Model (LIVE-3)
# =============================================================================
# Verifies that:
# 1. DynamicSpreadModel returns correct session-based spreads
# 2. News multiplier widens spreads correctly
# 3. Environment wires the model correctly
# 4. Backward compatible when disabled
#
# Run with: python -m pytest tests/test_sprint11_session_spread.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import USE_DYNAMIC_SPREAD, SPREAD_NEWS_MULTIPLIER
from src.environment.execution_model import DynamicSpreadModel


def _make_data(n_rows=800, base_price=2000.0):
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 0.5, n_rows))
    # Create timestamps spanning different trading sessions
    dates = pd.date_range('2023-01-02 00:00', periods=n_rows, freq='15min')
    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * 0.999,
        'High': prices * 1.003,
        'Low': prices * 0.997,
        'Close': prices,
        'Volume': np.full(n_rows, 500),
        'ATR': np.full(n_rows, 10.0),
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(**kwargs):
    df_kwargs = {k: v for k, v in kwargs.items() if k in ('n_rows', 'base_price')}
    env_kwargs = {k: v for k, v in kwargs.items() if k not in ('n_rows', 'base_price')}
    df = _make_data(**df_kwargs)
    from src.environment.environment import TradingEnv
    return TradingEnv(df, strict_scaler_mode=False, **env_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config constants exist
# ─────────────────────────────────────────────────────────────────────────────
def test_config_constants():
    """USE_DYNAMIC_SPREAD and SPREAD_NEWS_MULTIPLIER should exist."""
    assert USE_DYNAMIC_SPREAD is True
    assert SPREAD_NEWS_MULTIPLIER == 6.0  # v4: increased from 3.0 (real Gold: 5-10x during NFP/FOMC)


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: London session (10:00 UTC) → 0.0003 (3 bps)
# ─────────────────────────────────────────────────────────────────────────────
def test_london_session_spread():
    """Trade at 10:00 UTC (London) should have tight spread."""
    model = DynamicSpreadModel()
    spread = model.get_spread(hour_utc=10)
    assert spread == 0.0003, f"Expected 0.0003, got {spread}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Asian session (03:00 UTC) → 0.0008 (8 bps)
# ─────────────────────────────────────────────────────────────────────────────
def test_asian_session_spread():
    """Trade at 03:00 UTC (Asian) should have wider spread."""
    model = DynamicSpreadModel()
    spread = model.get_spread(hour_utc=3)
    assert spread == 0.0008, f"Expected 0.0008, got {spread}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: London/NY overlap (14:00 UTC) → 0.0003
# ─────────────────────────────────────────────────────────────────────────────
def test_london_ny_overlap_spread():
    """Trade at 14:00 UTC (London/NY overlap) should have tight spread."""
    model = DynamicSpreadModel()
    spread = model.get_spread(hour_utc=14)
    assert spread == 0.0003, f"Expected 0.0003, got {spread}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: NY afternoon (18:00 UTC) → 0.0005
# ─────────────────────────────────────────────────────────────────────────────
def test_ny_afternoon_spread():
    """Trade at 18:00 UTC (NY afternoon) should have moderate spread."""
    model = DynamicSpreadModel()
    spread = model.get_spread(hour_utc=18)
    assert spread == 0.0005, f"Expected 0.0005, got {spread}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: After-hours (22:00 UTC) → 0.0008
# ─────────────────────────────────────────────────────────────────────────────
def test_after_hours_spread():
    """Trade at 22:00 UTC (after-hours) should have wide spread."""
    model = DynamicSpreadModel()
    spread = model.get_spread(hour_utc=22)
    assert spread == 0.0008, f"Expected 0.0008, got {spread}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: News window → 3x multiplier
# ─────────────────────────────────────────────────────────────────────────────
def test_news_window_multiplier():
    """During news window, spread should be 3x wider."""
    model = DynamicSpreadModel(news_multiplier=3.0)
    # London session + news
    spread = model.get_spread(hour_utc=10, is_news_window=True)
    assert abs(spread - 0.0009) < 1e-10, f"Expected 0.0009, got {spread}"
    # Asian session + news
    spread_asian = model.get_spread(hour_utc=3, is_news_window=True)
    assert abs(spread_asian - 0.0024) < 1e-10, f"Expected 0.0024, got {spread_asian}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Environment initializes spread model
# ─────────────────────────────────────────────────────────────────────────────
def test_env_has_spread_model():
    """TradingEnv should have _spread_model when enabled."""
    env = _make_env()
    env.reset()

    assert hasattr(env, '_spread_model')
    assert hasattr(env, '_use_dynamic_spread')
    assert env._use_dynamic_spread is True
    assert env._spread_model is not None
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Dynamic spread disabled falls back to static
# ─────────────────────────────────────────────────────────────────────────────
def test_disabled_gives_static_spread():
    """When use_dynamic_spread=False, spread should be static."""
    env = _make_env(use_dynamic_spread=False)
    env.reset()

    assert env._use_dynamic_spread is False
    spread = env._get_current_spread()
    assert spread == env.transaction_fee_percentage
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 10: Hour wraps around for hour > 23
# ─────────────────────────────────────────────────────────────────────────────
def test_hour_wraps():
    """Hour values >= 24 should wrap correctly."""
    model = DynamicSpreadModel()
    # hour 25 should wrap to 1 (Asian)
    assert model.get_spread(25) == model.get_spread(1)
    # hour 36 should wrap to 12 (London)
    assert model.get_spread(36) == model.get_spread(12)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
