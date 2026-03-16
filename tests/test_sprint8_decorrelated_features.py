# =============================================================================
# Sprint 8 Validation: Wire Decorrelated Features Into Observation Space (OBS-1)
# =============================================================================
# Verifies that:
# 1. compute_decorrelated_ohlcv() produces valid decorrelated features
# 2. _process_data() calls it when USE_DECORRELATED_FEATURES=True
# 3. FEATURES list uses decorrelated features (no raw OHLC in obs)
# 4. Close/High/Low still available in DataFrame for trading logic
# 5. Observation space shape changes correctly
#
# Run with: python -m pytest tests/test_sprint8_decorrelated_features.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from config import (
    USE_DECORRELATED_FEATURES, FEATURES,
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    POSITION_FLAT, POSITION_LONG,
)
from src.environment.feature_reducer import compute_decorrelated_ohlcv


def _make_data(n_rows=800, base_price=2000.0):
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 1, n_rows))
    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * 0.999,
        'High': prices * 1.003,
        'Low': prices * 0.997,
        'Close': prices,
        'Volume': np.full(n_rows, 500),
        'ATR': np.full(n_rows, 10.0),
        'RSI': np.full(n_rows, 50.0),
        'BOS_SIGNAL': np.zeros(n_rows),
        'OB_SIGNAL': np.zeros(n_rows),
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(**kwargs):
    df = _make_data(**{k: v for k, v in kwargs.items() if k in ('n_rows', 'base_price')})
    env_kwargs = {k: v for k, v in kwargs.items() if k not in ('n_rows', 'base_price')}
    from src.environment.environment import TradingEnv
    return TradingEnv(df, strict_scaler_mode=False, **env_kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config flag and feature list
# ─────────────────────────────────────────────────────────────────────────────
def test_config_flag_and_features():
    """USE_DECORRELATED_FEATURES should be True and FEATURES should reflect that."""
    assert USE_DECORRELATED_FEATURES is True
    if USE_DECORRELATED_FEATURES:
        assert 'log_return' in FEATURES
        assert 'hl_range' in FEATURES
        assert 'close_position' in FEATURES
        assert 'Open' not in FEATURES, "Raw Open should not be in FEATURES"
        assert 'High' not in FEATURES, "Raw High should not be in FEATURES"
        assert 'Low' not in FEATURES, "Raw Low should not be in FEATURES"
        assert 'Close' not in FEATURES, "Raw Close should not be in FEATURES"


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: compute_decorrelated_ohlcv produces correct columns
# ─────────────────────────────────────────────────────────────────────────────
def test_decorrelated_output_columns():
    """compute_decorrelated_ohlcv should add log_return, hl_range, close_position."""
    df = _make_data()
    result = compute_decorrelated_ohlcv(df)

    assert 'log_return' in result.columns
    assert 'hl_range' in result.columns
    assert 'close_position' in result.columns
    # OHLC should still be in the DataFrame (not dropped — Sprint 8 fix)
    assert 'Close' in result.columns, "Close must remain for trading logic"
    assert 'High' in result.columns, "High must remain for SL/TP checks"
    assert 'Low' in result.columns, "Low must remain for SL/TP checks"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: No NaN/Inf in decorrelated features
# ─────────────────────────────────────────────────────────────────────────────
def test_decorrelated_no_nan_inf():
    """Decorrelated features should be clean (no NaN except possibly first row)."""
    df = _make_data()
    result = compute_decorrelated_ohlcv(df)

    for col in ['log_return', 'hl_range', 'close_position']:
        vals = result[col].values
        # First value of log_return is 0 by construction
        assert not np.any(np.isnan(vals)), f"NaN found in {col}"
        assert not np.any(np.isinf(vals)), f"Inf found in {col}"


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: log_return is bounded and centered near 0
# ─────────────────────────────────────────────────────────────────────────────
def test_log_return_properties():
    """log_return should be small (centered around 0 for M15 bars)."""
    df = _make_data()
    result = compute_decorrelated_ohlcv(df)

    log_ret = result['log_return'].values
    assert log_ret[0] == 0.0, "First log_return should be 0"
    assert abs(np.mean(log_ret)) < 0.01, "Mean log_return should be near 0"
    assert np.max(np.abs(log_ret)) < 0.1, "Max |log_return| should be reasonable"


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: hl_range is positive
# ─────────────────────────────────────────────────────────────────────────────
def test_hl_range_positive():
    """hl_range = (High - Low) / ATR should always be positive."""
    df = _make_data()
    result = compute_decorrelated_ohlcv(df)

    hl = result['hl_range'].values
    assert np.all(hl >= 0), "hl_range should be non-negative"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: close_position is bounded [-1, 1]
# ─────────────────────────────────────────────────────────────────────────────
def test_close_position_bounded():
    """close_position = (Close - Open) / (High - Low) should be in [-1, 1]."""
    df = _make_data()
    result = compute_decorrelated_ohlcv(df)

    cp = result['close_position'].values
    assert np.all(cp >= -1.1), f"close_position min {cp.min()} is too low"
    assert np.all(cp <= 1.1), f"close_position max {cp.max()} is too high"


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Environment creates observations with decorrelated features
# ─────────────────────────────────────────────────────────────────────────────
def test_env_uses_decorrelated_features():
    """TradingEnv should have decorrelated features in its feature list."""
    env = _make_env()
    env.reset()

    if USE_DECORRELATED_FEATURES:
        assert 'log_return' in env.features, "log_return should be in env.features"
        assert 'hl_range' in env.features, "hl_range should be in env.features"
        assert 'close_position' in env.features, "close_position should be in env.features"

    # Close should still be accessible in the DataFrame for trading
    assert 'Close' in env.df.columns, "Close must be in env.df for trade execution"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Trading still works with decorrelated features
# ─────────────────────────────────────────────────────────────────────────────
def test_trading_works_with_decorrelated():
    """Full trade lifecycle should work with decorrelated obs space."""
    env = _make_env()
    env.reset()

    # Open a position
    for _ in range(10):
        env.step(ACTION_HOLD)
    obs, _, done, _, _ = env.step(ACTION_OPEN_LONG)
    if not done:
        assert env.position_type == POSITION_LONG

        # Hold a few bars
        for _ in range(5):
            env.step(ACTION_HOLD)

        # Close
        env.step(ACTION_CLOSE_LONG)
        assert env.position_type == POSITION_FLAT

    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: Observation shape is correct
# ─────────────────────────────────────────────────────────────────────────────
def test_obs_shape():
    """Observation shape should match features × lookback + state dims."""
    env = _make_env()
    obs, _ = env.reset()

    n_features = len(env.features)
    lookback = env.lookback_window_size
    state_dims = 3  # balance_ratio, position_type, hold_duration
    expected_dim = lookback * n_features + state_dims

    assert obs.shape == (expected_dim,), (
        f"Expected obs shape ({expected_dim},), got {obs.shape}. "
        f"n_features={n_features}, lookback={lookback}"
    )
    env.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
