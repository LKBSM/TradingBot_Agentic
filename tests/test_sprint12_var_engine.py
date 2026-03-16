# =============================================================================
# Sprint 12 Validation: Wire VaR Engine Into Training/Eval Loop (VAR-1)
# =============================================================================
# Verifies that:
# 1. VaREngine is instantiated in TradingEnv
# 2. Portfolio returns are fed each step
# 3. VaR is exposed in info dict after enough observations
# 4. VaR resets between episodes
# 5. Backward compatible when disabled
#
# Run with: python -m pytest tests/test_sprint12_var_engine.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.risk.var_engine import VaREngine, VaRResult


def _make_data(n_rows=800, base_price=2000.0):
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 0.5, n_rows))
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
    })
    df.set_index('Date', inplace=True)
    return df


def _make_env(**kwargs):
    df = _make_data()
    from src.environment.environment import TradingEnv
    return TradingEnv(df, strict_scaler_mode=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: VaREngine standalone — compute after enough data
# ─────────────────────────────────────────────────────────────────────────────
def test_var_engine_computes_after_min_observations():
    """VaR should be non-zero after 30+ observations."""
    engine = VaREngine(confidence=0.95, window=252, method='cornish_fisher')
    assert not engine.is_ready

    # Feed 50 returns
    np.random.seed(42)
    for _ in range(50):
        engine.update(np.random.normal(0, 0.01))

    assert engine.is_ready
    result = engine.compute()
    assert isinstance(result, VaRResult)
    assert result.var_95 != 0.0
    assert result.var_99 != 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: VaR engine resets correctly
# ─────────────────────────────────────────────────────────────────────────────
def test_var_engine_resets():
    """After reset, VaR should not be ready."""
    engine = VaREngine(confidence=0.95, window=252, method='cornish_fisher')
    for _ in range(50):
        engine.update(0.001)

    assert engine.is_ready
    engine.reset()
    assert not engine.is_ready
    assert engine.buffer_size == 0


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Environment has VaR engine
# ─────────────────────────────────────────────────────────────────────────────
def test_env_has_var_engine():
    """TradingEnv should have _var_engine when enabled."""
    env = _make_env()
    env.reset()

    assert hasattr(env, '_var_engine')
    assert env._var_engine is not None
    assert env._use_var_engine is True
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: VaR not in info before enough steps
# ─────────────────────────────────────────────────────────────────────────────
def test_var_not_in_info_early():
    """VaR should not be in info before 30 steps."""
    env = _make_env()
    env.reset()

    # Take a few steps
    for _ in range(5):
        _, _, done, _, info = env.step(0)  # HOLD
        if done:
            break

    assert 'var_95' not in info, "VaR should not appear before enough observations"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: VaR appears in info after enough steps
# ─────────────────────────────────────────────────────────────────────────────
def test_var_appears_after_min_steps():
    """VaR should appear in info after 30+ steps."""
    env = _make_env()
    env.reset()

    info = {}
    for _ in range(35):
        _, _, done, _, info = env.step(0)  # HOLD
        if done:
            break

    assert 'var_95' in info, f"VaR should appear after 35 steps. Keys: {list(info.keys())}"
    assert isinstance(info['var_95'], float)
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: VaR resets between episodes
# ─────────────────────────────────────────────────────────────────────────────
def test_var_resets_between_episodes():
    """VaR engine should reset when environment resets."""
    env = _make_env()
    env.reset()

    # Run 35 steps to build up VaR data
    for _ in range(35):
        _, _, done, _, _ = env.step(0)
        if done:
            break

    # Verify VaR has data
    assert env._var_engine.buffer_size >= 30

    # Reset environment
    env.reset()

    # VaR should be cleared
    assert env._var_engine.buffer_size == 0
    assert not env._var_engine.is_ready
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Disabled VaR engine
# ─────────────────────────────────────────────────────────────────────────────
def test_disabled_var_engine():
    """When use_var_engine=False, no VaR engine should exist."""
    env = _make_env(use_var_engine=False)
    env.reset()

    assert env._var_engine is None
    assert env._use_var_engine is False

    # Step should still work without VaR
    for _ in range(5):
        _, _, done, _, info = env.step(0)
        if done:
            break
    assert 'var_95' not in info
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: VaR is a reasonable number
# ─────────────────────────────────────────────────────────────────────────────
def test_var_is_reasonable():
    """VaR should be a small positive number (loss threshold) for typical returns."""
    engine = VaREngine(confidence=0.95, window=100, method='cornish_fisher')
    np.random.seed(42)
    for _ in range(100):
        engine.update(np.random.normal(0, 0.005))  # 0.5% daily vol

    result = engine.compute()
    # VaR is reported as positive loss threshold
    # For 0.5% vol, 95% VaR should be in a reasonable range
    assert result.var_95 != 0.0, "VaR should be non-zero"
    assert result.var_95 < 0.1, f"VaR seems too extreme: {result.var_95}"
    assert result.var_99 > result.var_95, "99% VaR should exceed 95% VaR"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
