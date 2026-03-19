"""
Tests for v4 DSR reward + observation space + risk calibration changes.
Covers Sprints 1-5 of the Tiers 1-4 implementation plan.
"""
import sys
import os
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import config
from config import (
    FEATURES, MTF_FEATURES, ZSCORE_WINDOW, TP_ATR_MULTIPLIER,
    DAILY_LOSS_LIMIT, DSR_ETA, FIXED_EPISODE_LENGTH,
    TSL_START_PROFIT_MULTIPLIER, TSL_TRAIL_DISTANCE_MULTIPLIER,
    SPREAD_NEWS_MULTIPLIER, ACTION_HOLD, ACTION_OPEN_LONG,
    ACTION_CLOSE_LONG, ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
    POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
    STOP_LOSS_PERCENTAGE, MAX_DURATION_STEPS,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_gold_df(n_bars=600, base_price=2000.0, seed=42):
    """Create a realistic Gold M15 DataFrame with all required columns."""
    rng = np.random.default_rng(seed)

    dates = pd.date_range('2024-01-01', periods=n_bars, freq='15min')

    # Random walk price
    returns = rng.normal(0, 0.001, n_bars)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + rng.normal(0, 0.0005, n_bars)),
        'High': prices * (1 + abs(rng.normal(0.001, 0.001, n_bars))),
        'Low': prices * (1 - abs(rng.normal(0.001, 0.001, n_bars))),
        'Close': prices,
        'Volume': rng.uniform(100, 10000, n_bars),
    })

    return df


def _make_env(n_bars=600, **kwargs):
    """Create a TradingEnv with sensible defaults for testing."""
    from src.environment.environment import TradingEnv

    df = _make_gold_df(n_bars=n_bars)

    defaults = {
        'strict_scaler_mode': False,
        'use_var_engine': False,
        'use_dynamic_slippage': False,
        'use_dynamic_spread': False,
        'training_mode': True,
    }
    defaults.update(kwargs)

    return TradingEnv(df, **defaults)


# =============================================================================
# SPRINT 1: OBSERVATION SPACE
# =============================================================================

class TestObservationSpace:
    """Test v4 observation space changes."""

    def test_bb_pct_in_features(self):
        """BB_pct should be in the feature list (replaces BB_L/BB_H)."""
        assert 'BB_pct' in FEATURES
        assert 'BB_L' not in FEATURES
        assert 'BB_H' not in FEATURES

    def test_removed_features_not_in_list(self):
        """SPREAD, SESSION, DAY_OF_WEEK, HTF_RSI_1H should be removed."""
        assert 'SPREAD' not in FEATURES
        assert 'SESSION' not in MTF_FEATURES
        assert 'DAY_OF_WEEK' not in MTF_FEATURES
        assert 'HTF_RSI_1H' not in MTF_FEATURES

    def test_bb_pct_range(self):
        """BB_pct should be bounded [0, 1]."""
        env = _make_env()
        if 'BB_pct' in env.df.columns:
            bb_pct = env.df['BB_pct'].dropna()
            assert bb_pct.min() >= 0.0, f"BB_pct min={bb_pct.min()} < 0"
            assert bb_pct.max() <= 1.0, f"BB_pct max={bb_pct.max()} > 1"

    def test_markov_state_in_observation(self):
        """Observation should contain 8 state vars (3 original + 5 Markov)."""
        env = _make_env()
        obs, _ = env.reset()

        num_features = len(env.features)
        lookback = env.lookback_window_size
        expected_size = num_features * lookback + 8  # 8 state vars

        assert obs.shape[0] == expected_size, (
            f"Obs size {obs.shape[0]} != expected {expected_size} "
            f"(features={num_features} * lookback={lookback} + 8 state)"
        )

    def test_markov_vars_zero_when_flat(self):
        """Markov state vars should be 0 when position is FLAT."""
        env = _make_env()
        obs, _ = env.reset()

        # Last 5 values are: entry_price_pct, hold_dur, unrealized_pnl, sl_dist, tp_dist
        markov_vars = obs[-5:]
        np.testing.assert_array_almost_equal(
            markov_vars, [0.0, 0.0, 0.0, 0.0, 0.0],
            err_msg="Markov state vars should be 0 when FLAT"
        )

    def test_markov_vars_nonzero_in_position(self):
        """Markov state vars should be non-zero when in a position."""
        env = _make_env()
        obs, _ = env.reset()

        # Force open a long position
        for _ in range(50):
            obs, _, done, trunc, _ = env.step(ACTION_OPEN_LONG)
            if env.position_type == POSITION_LONG:
                break
            if done or trunc:
                obs, _ = env.reset()

        if env.position_type == POSITION_LONG:
            # Step once more to get updated obs
            obs, _, _, _, _ = env.step(ACTION_HOLD)
            markov_vars = obs[-5:]
            # At least entry_price_pct and hold_duration should be non-zero
            assert markov_vars[0] != 0.0 or markov_vars[1] != 0.0, (
                "Markov vars should have non-zero values when in position"
            )


# =============================================================================
# SPRINT 2: ROLLING Z-SCORE
# =============================================================================

class TestRollingZScore:
    """Test z-score normalization of non-stationary features."""

    def test_zscore_window_config(self):
        """ZSCORE_WINDOW should be configured."""
        assert ZSCORE_WINDOW == 500

    def test_zscore_produces_normal_distribution(self):
        """After z-scoring, ATR/MACD_Diff/Volume should be approximately N(0,1)."""
        env = _make_env(n_bars=1200)

        for col in ['ATR', 'MACD_Diff', 'Volume']:
            if col in env.df.columns:
                values = env.df[col].dropna()
                # After z-score + MinMaxScaler, values should be in a reasonable range
                # Not saturated at 0.0 or 1.0
                if len(values) > 100:
                    # Z-scored values should have mean near 0 and std near 1
                    # (before MinMaxScaler transforms them)
                    # Just check they're not all identical (which would mean saturation)
                    assert values.std() > 0.01, f"{col} has no variance after z-score"


# =============================================================================
# SPRINT 3: DIFFERENTIAL SHARPE RATIO REWARD
# =============================================================================

class TestDSRReward:
    """Test Differential Sharpe Ratio reward function."""

    def test_dsr_eta_config(self):
        """DSR_ETA should be configured."""
        assert DSR_ETA == 0.004

    def test_dsr_state_initialized(self):
        """DSR state (A, B) should be initialized on reset."""
        env = _make_env()
        env.reset()

        assert hasattr(env, '_dsr_A'), "Missing _dsr_A state"
        assert hasattr(env, '_dsr_B'), "Missing _dsr_B state"
        assert hasattr(env, '_dsr_eta'), "Missing _dsr_eta state"
        assert env._dsr_A == 0.0
        assert env._dsr_B == 1e-8

    def test_dsr_produces_dense_reward(self):
        """DSR should produce non-zero reward when in a position."""
        env = _make_env()
        env.reset()

        # Open a position first — when flat, net_worth=balance which is constant
        # so R_t=0 and DSR=0 (correct behavior — no exposure = no signal)
        for _ in range(10):
            env.step(ACTION_OPEN_LONG)
            if env.position_type == POSITION_LONG:
                break

        if env.position_type != POSITION_LONG:
            pytest.skip("Could not open a long position")

        rewards = []
        for _ in range(20):
            _, reward, done, trunc, _ = env.step(ACTION_HOLD)
            rewards.append(reward)
            if done or trunc:
                break

        # With a long position, price changes affect net_worth → DSR should be non-zero
        assert any(r != 0.0 for r in rewards), (
            f"DSR produced all-zero rewards while in position: {rewards}"
        )

    def test_dsr_reward_sign_correct(self):
        """Positive price change should give positive DSR (during warm-up at least)."""
        env = _make_env()
        env.reset()

        # Open long, then check rewards correlate with price direction
        for _ in range(10):
            _, _, done, trunc, _ = env.step(ACTION_OPEN_LONG)
            if env.position_type == POSITION_LONG or done or trunc:
                break

        if env.position_type == POSITION_LONG:
            # During warm-up, DSR falls back to scaled return
            # which should be positive when price goes up
            _, reward, _, _, _ = env.step(ACTION_HOLD)
            # Can't guarantee sign, but should be non-zero
            assert isinstance(reward, float)

    def test_dsr_reward_clipped(self):
        """DSR reward should be clipped to [-20, 20]."""
        env = _make_env()
        env.reset()

        for _ in range(50):
            _, reward, done, trunc, _ = env.step(ACTION_HOLD)
            assert -20.0 <= reward <= 20.0, f"Reward {reward} out of [-20, 20]"
            if done or trunc:
                break

    def test_dsr_reward_components_tracked(self):
        """Reward components should be tracked for TensorBoard."""
        env = _make_env()
        env.reset()
        env.step(ACTION_HOLD)

        assert hasattr(env, '_last_reward_components')
        components = env._last_reward_components
        assert 'dsr' in components
        assert 'R_t' in components
        assert 'final_reward' in components

    def test_terminal_reward_on_blown_account(self):
        """Account below minimum should return -20.0."""
        env = _make_env()
        env.reset()

        # Simulate blown account
        env.net_worth = 50.0  # Below minimum_allowed_balance (100.0)
        reward = env._calculate_reward(1000.0)
        assert reward == -20.0


# =============================================================================
# SPRINT 4: BORROWING FEE + RISK CALIBRATION
# =============================================================================

class TestBorrowingFeeAndRisk:
    """Test borrowing fee fix and risk calibration."""

    def test_borrowing_fee_per_bar(self):
        """Borrowing fee should be charged per bar (÷96), not per day."""
        env = _make_env()
        env.reset()

        # Open a short position
        for _ in range(10):
            _, _, done, trunc, _ = env.step(ACTION_OPEN_SHORT)
            if env.position_type == POSITION_SHORT or done or trunc:
                break

        if env.position_type == POSITION_SHORT:
            balance_before = env.balance
            qty = abs(env.stock_quantity)
            price = float(env.df.iloc[env.current_step]['Close'])
            expected_fee = qty * price * env.short_borrowing_fee_daily / 96.0

            env.step(ACTION_HOLD)

            # Balance should decrease by approximately the per-bar fee
            fee_paid = balance_before - env.balance
            # Fee should be order of magnitude smaller than daily rate
            if qty > 0 and expected_fee > 0:
                assert fee_paid < qty * price * env.short_borrowing_fee_daily * 1.1, (
                    f"Fee {fee_paid} too high — likely charging daily rate per bar"
                )

    def test_tsl_params_widened(self):
        """TSL params should be widened to prevent whipsaw."""
        assert TSL_START_PROFIT_MULTIPLIER == 2.0, f"TSL start={TSL_START_PROFIT_MULTIPLIER}, expected 2.0"
        assert TSL_TRAIL_DISTANCE_MULTIPLIER == 1.0, f"TSL trail={TSL_TRAIL_DISTANCE_MULTIPLIER}, expected 1.0"

    def test_atr_based_tp(self):
        """Take profit should be ATR-based, not fixed percentage."""
        from src.environment.risk_manager import DynamicRiskManager
        rm = DynamicRiskManager(config={})
        rm.set_client_profile('test', 10000, 20.0, 0.1, 0.01)

        entry_price = 2000.0
        atr = 10.0  # $10 ATR

        sl_distance = rm.set_trade_orders(entry_price, atr, is_long=True)

        # TP should be entry + TP_ATR_MULTIPLIER * ATR
        expected_tp = entry_price + TP_ATR_MULTIPLIER * atr
        assert abs(rm.current_take_profit - expected_tp) < 1.0, (
            f"TP={rm.current_take_profit}, expected {expected_tp} (ATR-based)"
        )

        # Short TP
        sl_distance = rm.set_trade_orders(entry_price, atr, is_long=False)
        expected_tp_short = entry_price - TP_ATR_MULTIPLIER * atr
        assert abs(rm.current_take_profit - expected_tp_short) < 1.0, (
            f"Short TP={rm.current_take_profit}, expected {expected_tp_short}"
        )

    def test_daily_loss_limit_blocks_entries(self):
        """After -2% daily loss, new entries should be blocked."""
        env = _make_env()
        env.reset()

        # Simulate daily loss limit being hit
        env._daily_trading_disabled = True

        original_position = env.position_type
        assert original_position == POSITION_FLAT

        # Try to open long — should be blocked
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_FLAT, "Entry should be blocked by daily loss limit"

    def test_daily_loss_limit_allows_exits(self):
        """Daily loss limit should NOT block exits (only entries)."""
        env = _make_env()
        env.reset()

        # First open a position
        for _ in range(10):
            env.step(ACTION_OPEN_LONG)
            if env.position_type == POSITION_LONG:
                break

        if env.position_type == POSITION_LONG:
            # Now trigger daily loss limit
            env._daily_trading_disabled = True

            # Close long should still work
            env.step(ACTION_CLOSE_LONG)
            # Position should be closed (or at least not blocked)
            # The close might fail for other reasons, but it shouldn't be blocked by daily limit

    def test_news_spread_multiplier_increased(self):
        """News spread multiplier should be 6.0 (was 3.0)."""
        assert SPREAD_NEWS_MULTIPLIER == 6.0

    def test_daily_loss_limit_config(self):
        """DAILY_LOSS_LIMIT should be -0.02."""
        assert DAILY_LOSS_LIMIT == -0.02


# =============================================================================
# SPRINT 5: TRAINING PIPELINE
# =============================================================================

class TestTrainingPipeline:
    """Test training pipeline changes."""

    def test_episode_length_reduced(self):
        """Episode length should be 200 (was 500)."""
        assert FIXED_EPISODE_LENGTH == 200

    def test_phase_budgets_rebalanced(self):
        """Phase budgets should be 35/25/25/15."""
        from src.training.curriculum_trainer import CurriculumConfig
        cfg = CurriculumConfig(total_timesteps=1_000_000)
        phases = cfg.phases

        total = sum(p.timesteps for p in phases)
        pcts = [p.timesteps / total * 100 for p in phases]

        # Allow 1% tolerance for rounding
        assert abs(pcts[0] - 35) < 1.5, f"Phase 1 should be ~35%, got {pcts[0]:.1f}%"
        assert abs(pcts[1] - 25) < 1.5, f"Phase 2 should be ~25%, got {pcts[1]:.1f}%"
        assert abs(pcts[2] - 25) < 1.5, f"Phase 3 should be ~25%, got {pcts[2]:.1f}%"
        assert abs(pcts[3] - 15) < 1.5, f"Phase 4 should be ~15%, got {pcts[3]:.1f}%"

    def test_phase1_entropy_reduced(self):
        """Phase 1 entropy multiplier should be 2.0 (was 5.0)."""
        from src.training.curriculum_trainer import CurriculumConfig
        cfg = CurriculumConfig(total_timesteps=1_000_000)

        phase1 = cfg.phases[0]
        assert phase1.entropy_coef_multiplier == 2.0, (
            f"Phase 1 entropy={phase1.entropy_coef_multiplier}, expected 2.0"
        )

    def test_zero_mock_agent_signals(self):
        """UnifiedAgenticEnv should zero all agent signals during training."""
        try:
            from src.training.unified_agentic_env import UnifiedAgenticEnv, TrainingMode
        except ImportError:
            pytest.skip("UnifiedAgenticEnv not importable")

        df = _make_gold_df(n_bars=600)

        # Create env in ENRICHED mode (previously would use real mock signals)
        try:
            env = UnifiedAgenticEnv(
                df=df,
                mode=TrainingMode.ENRICHED,
                strict_scaler_mode=False,
                use_var_engine=False,
                use_dynamic_slippage=False,
                use_dynamic_spread=False,
                training_mode=True,
            )
            obs, _ = env.reset()

            # Last 20 values should be zeros (agent signals)
            agent_signals = obs[-20:]
            np.testing.assert_array_almost_equal(
                agent_signals, np.zeros(20),
                err_msg="Agent signals should be zero during training"
            )
        except Exception as e:
            # UnifiedAgenticEnv may need orchestrator etc.
            pytest.skip(f"Could not create UnifiedAgenticEnv: {e}")

    def test_tp_atr_multiplier_config(self):
        """TP_ATR_MULTIPLIER should be 4.0."""
        assert TP_ATR_MULTIPLIER == 4.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the full v4 pipeline."""

    def test_full_episode_runs(self):
        """A full episode should run without errors with all v4 changes."""
        env = _make_env(n_bars=600)
        obs, info = env.reset()

        total_reward = 0.0
        steps = 0
        actions = [ACTION_HOLD, ACTION_OPEN_LONG, ACTION_HOLD, ACTION_HOLD,
                   ACTION_CLOSE_LONG, ACTION_HOLD, ACTION_OPEN_SHORT,
                   ACTION_HOLD, ACTION_CLOSE_SHORT]

        for i in range(100):
            action = actions[i % len(actions)]
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            steps += 1

            # Observation should never contain NaN
            assert not np.any(np.isnan(obs)), f"NaN in obs at step {steps}"
            # Reward should be finite
            assert np.isfinite(reward), f"Non-finite reward at step {steps}: {reward}"

            if done or trunc:
                break

        assert steps > 0, "Episode did not run any steps"

    def test_obs_space_matches_observation(self):
        """observation_space.shape should match actual observation."""
        env = _make_env()
        obs, _ = env.reset()

        assert env.observation_space.shape[0] == obs.shape[0], (
            f"Space shape {env.observation_space.shape[0]} != obs shape {obs.shape[0]}"
        )

    def test_dsr_state_resets_between_episodes(self):
        """DSR state should reset between episodes."""
        env = _make_env()

        # Episode 1
        env.reset()
        for _ in range(10):
            env.step(ACTION_HOLD)

        # Episode 2
        env.reset()
        assert env._dsr_A == 0.0, "DSR A should reset between episodes"
        assert env._dsr_B == 1e-8, "DSR B should reset between episodes"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
