# =============================================================================
# Sprint 6 Validation: Rolling Win Rate for Kelly Position Sizing (PS-2)
# =============================================================================
# Verifies that hardcoded win_prob=0.5 has been replaced with an empirical
# rolling win rate that adapts Kelly position sizing based on recent trade
# performance.
#
# Run with: python -m pytest tests/test_sprint6_rolling_win_rate.py -v
# =============================================================================

import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import (
    ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
    ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
    ROLLING_WIN_RATE_WINDOW, ROLLING_WIN_RATE_MIN_TRADES,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0, trend="flat"):
    np.random.seed(42)
    if trend == "up":
        prices = base_price + np.linspace(0, 80, n_rows)
    elif trend == "down":
        prices = base_price + np.linspace(0, -80, n_rows)
    else:
        prices = np.full(n_rows, base_price)
    noise = np.random.normal(0, 0.3, n_rows)
    prices = prices + noise

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
    df = _make_data(**{k: v for k, v in kwargs.items() if k in ('n_rows', 'base_price', 'trend')})
    env_kwargs = {k: v for k, v in kwargs.items() if k not in ('n_rows', 'base_price', 'trend')}
    return TradingEnv(df, strict_scaler_mode=False, **env_kwargs)


def _step_n(env, action, n):
    result = None
    for _ in range(n):
        result = env.step(action)
        if result[2]:
            break
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Config constants exist
# ─────────────────────────────────────────────────────────────────────────────
def test_config_constants():
    """ROLLING_WIN_RATE_WINDOW and ROLLING_WIN_RATE_MIN_TRADES exist."""
    assert ROLLING_WIN_RATE_WINDOW == 50
    assert ROLLING_WIN_RATE_MIN_TRADES == 10


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: TradingEnv initializes rolling win rate state
# ─────────────────────────────────────────────────────────────────────────────
def test_env_has_rolling_win_rate():
    """TradingEnv should have _rolling_win_rate and _win_rate_window."""
    env = _make_env()
    env.reset()
    assert hasattr(env, '_rolling_win_rate')
    assert hasattr(env, '_win_rate_window')
    assert env._rolling_win_rate == 0.5, "Prior should start at 0.5"
    assert len(env._win_rate_window) == 0, "Window should be empty at start"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: reset() clears win rate window and restores prior
# ─────────────────────────────────────────────────────────────────────────────
def test_reset_clears_win_rate():
    """After reset(), rolling win rate should return to 0.5 prior."""
    env = _make_env(trend="up")
    env.reset()

    # Do a round-trip trade to populate the window
    _step_n(env, ACTION_HOLD, 10)
    env.step(ACTION_OPEN_LONG)
    _step_n(env, ACTION_HOLD, 20)
    env.step(ACTION_CLOSE_LONG)

    assert len(env._win_rate_window) > 0, "Window should have data after a trade"

    # Reset
    env.reset()
    assert env._rolling_win_rate == 0.5, "Prior should be reset to 0.5"
    assert len(env._win_rate_window) == 0, "Window should be cleared"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Winning close_long appends 1.0 to window
# ─────────────────────────────────────────────────────────────────────────────
def test_winning_close_long_appends_win():
    """A profitable close_long should append 1.0 to the win rate window."""
    env = _make_env(trend="up")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    # Artificially set entry price below current for guaranteed win
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 0.98  # 2% below

    _step_n(env, ACTION_HOLD, 5)
    env.step(ACTION_CLOSE_LONG)

    assert len(env._win_rate_window) == 1
    assert env._win_rate_window[-1] == 1.0, "Winning trade should append 1.0"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Losing close_long appends 0.0 to window
# ─────────────────────────────────────────────────────────────────────────────
def test_losing_close_long_appends_loss():
    """A losing close_long should append 0.0 to the win rate window."""
    env = _make_env(trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_LONG)
    # Artificially set entry price above current for guaranteed loss
    current_price = env.df.iloc[env.current_step]['Close']
    env.entry_price = current_price * 1.02  # 2% above

    _step_n(env, ACTION_HOLD, 5)
    env.step(ACTION_CLOSE_LONG)

    assert len(env._win_rate_window) == 1
    assert env._win_rate_window[-1] == 0.0, "Losing trade should append 0.0"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: close_short also updates the win rate window
# ─────────────────────────────────────────────────────────────────────────────
def test_close_short_updates_win_rate():
    """close_short should also track wins/losses in the rolling window."""
    env = _make_env(trend="down")
    env.reset()
    _step_n(env, ACTION_HOLD, 10)

    env.step(ACTION_OPEN_SHORT)
    assert env.position_type == POSITION_SHORT

    _step_n(env, ACTION_HOLD, 20)
    env.step(ACTION_CLOSE_SHORT)

    assert len(env._win_rate_window) >= 1, "close_short should update the window"
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 7: Win rate doesn't update until MIN_TRADES reached
# ─────────────────────────────────────────────────────────────────────────────
def test_win_rate_uses_prior_before_min_trades():
    """Before ROLLING_WIN_RATE_MIN_TRADES, _rolling_win_rate stays at 0.5."""
    env = _make_env(trend="up")
    env.reset()

    # Do fewer trades than the minimum threshold
    for i in range(min(ROLLING_WIN_RATE_MIN_TRADES - 1, 5)):
        _step_n(env, ACTION_HOLD, 5)
        env.step(ACTION_OPEN_LONG)
        if env.position_type != POSITION_LONG:
            break
        current_price = env.df.iloc[env.current_step]['Close']
        env.entry_price = current_price * 0.98  # Force win
        _step_n(env, ACTION_HOLD, 3)
        env.step(ACTION_CLOSE_LONG)

    # Should still be at prior because < MIN_TRADES completed
    assert env._rolling_win_rate == 0.5, (
        f"Win rate should remain at 0.5 prior with only "
        f"{len(env._win_rate_window)} trades (need {ROLLING_WIN_RATE_MIN_TRADES})"
    )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 8: Win rate updates after MIN_TRADES reached
# ─────────────────────────────────────────────────────────────────────────────
def test_win_rate_updates_after_min_trades():
    """After ROLLING_WIN_RATE_MIN_TRADES, _rolling_win_rate reflects actual results."""
    env = _make_env(trend="up", n_rows=2000)
    env.reset()

    # Force exactly MIN_TRADES winning trades by manipulating entry_price
    trades_done = 0
    for i in range(ROLLING_WIN_RATE_MIN_TRADES + 5):  # Extra attempts in case some fail
        if trades_done >= ROLLING_WIN_RATE_MIN_TRADES:
            break
        _step_n(env, ACTION_HOLD, 3)
        obs, _, done, _, _ = env.step(ACTION_OPEN_LONG)
        if done:
            break
        if env.position_type != POSITION_LONG:
            continue

        # Force a winning trade
        current_price = env.df.iloc[env.current_step]['Close']
        env.entry_price = current_price * 0.98  # 2% below = guaranteed win

        _step_n(env, ACTION_HOLD, 2)
        obs, _, done, _, _ = env.step(ACTION_CLOSE_LONG)
        if done:
            break
        trades_done += 1

    if trades_done >= ROLLING_WIN_RATE_MIN_TRADES:
        # All forced wins → win rate should be 1.0
        assert env._rolling_win_rate == 1.0, (
            f"Expected win rate 1.0 after {trades_done} forced wins, "
            f"got {env._rolling_win_rate}"
        )
    env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Test 9: No remaining hardcoded win_prob=0.5 in open methods
# ─────────────────────────────────────────────────────────────────────────────
def test_no_hardcoded_win_prob():
    """Verify win_prob=0.5 is not hardcoded in environment.py open methods."""
    import inspect
    source_open_long = inspect.getsource(TradingEnv._execute_open_long)
    source_open_short = inspect.getsource(TradingEnv._execute_open_short)

    assert 'win_prob=0.5' not in source_open_long, (
        "Hardcoded win_prob=0.5 still present in _execute_open_long"
    )
    assert 'win_prob=0.5' not in source_open_short, (
        "Hardcoded win_prob=0.5 still present in _execute_open_short"
    )
    assert '_rolling_win_rate' in source_open_long, (
        "_rolling_win_rate not used in _execute_open_long"
    )
    assert '_rolling_win_rate' in source_open_short, (
        "_rolling_win_rate not used in _execute_open_short"
    )


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
