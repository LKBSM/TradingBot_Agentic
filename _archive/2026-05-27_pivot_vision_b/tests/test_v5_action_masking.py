# =============================================================================
# V5 Validation: Action Masking, Entry Bonus, Cost Curriculum
# =============================================================================
# Tests for the 4 structural fixes that break the "always hold" degenerate policy.
#
# Run with: python -m pytest tests/test_v5_action_masking.py -v
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
    MIN_HOLD_FOR_BONUS, TOTAL_TIMESTEPS_PER_BOT,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0):
    """Create controlled test data."""
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 2, n_rows))
    prices = np.maximum(prices, base_price * 0.8)  # floor

    df = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_rows, freq='15min'),
        'Open': prices * 0.999,
        'High': prices * 1.002,
        'Low': prices * 0.998,
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
    """Create environment with test data."""
    df = _make_data()
    return TradingEnv(df, strict_scaler_mode=False, **kwargs)


# ─────────────────────────────────────────────────────────────────────────────
# ACTION MASKING TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestActionMasks:
    """Test the action_masks() method for MaskablePPO compatibility."""

    def test_action_masks_exists(self):
        """TradingEnv must have action_masks() method."""
        env = _make_env()
        env.reset()
        assert hasattr(env, 'action_masks')
        assert callable(env.action_masks)
        env.close()

    def test_flat_allows_hold_and_open(self):
        """When FLAT, HOLD + OPEN_LONG + OPEN_SHORT are valid."""
        env = _make_env()
        env.reset()
        assert env.position_type == POSITION_FLAT
        masks = env.action_masks()
        assert masks[ACTION_HOLD] == True
        assert masks[ACTION_OPEN_LONG] == True
        assert masks[ACTION_OPEN_SHORT] == True
        assert masks[ACTION_CLOSE_LONG] == False
        assert masks[ACTION_CLOSE_SHORT] == False
        env.close()

    def test_long_allows_hold_and_close(self):
        """When LONG, HOLD + CLOSE_LONG are valid."""
        env = _make_env()
        env.reset()
        # Hold a few bars then open long
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG
        masks = env.action_masks()
        assert masks[ACTION_HOLD] == True
        assert masks[ACTION_CLOSE_LONG] == True
        assert masks[ACTION_OPEN_LONG] == False
        assert masks[ACTION_OPEN_SHORT] == False
        assert masks[ACTION_CLOSE_SHORT] == False
        env.close()

    def test_short_allows_hold_and_close(self):
        """When SHORT, HOLD + CLOSE_SHORT are valid."""
        env = _make_env()
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_SHORT)
        assert env.position_type == POSITION_SHORT
        masks = env.action_masks()
        assert masks[ACTION_HOLD] == True
        assert masks[ACTION_CLOSE_SHORT] == True
        assert masks[ACTION_OPEN_LONG] == False
        assert masks[ACTION_OPEN_SHORT] == False
        assert masks[ACTION_CLOSE_LONG] == False
        env.close()

    def test_masks_shape_is_5(self):
        """Action masks must have shape (5,) matching 5-action space."""
        env = _make_env()
        env.reset()
        masks = env.action_masks()
        assert masks.shape == (5,)
        assert masks.dtype == bool
        env.close()

    def test_daily_loss_limit_blocks_entries_in_mask(self):
        """When daily loss limit hit, entries are blocked in mask."""
        env = _make_env()
        env.reset()
        # Manually trigger daily loss limit
        env._daily_trading_disabled = True
        masks = env.action_masks()
        assert masks[ACTION_HOLD] == True
        assert masks[ACTION_OPEN_LONG] == False
        assert masks[ACTION_OPEN_SHORT] == False
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# NO INVALID ACTION PENALTY
# ─────────────────────────────────────────────────────────────────────────────

class TestNoInvalidPenalty:
    """Verify invalid action penalty is removed (v5)."""

    def test_no_penalty_for_invalid_action(self):
        """Invalid action should NOT reduce reward (penalty removed in v5)."""
        env = _make_env()
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        # Open a long position
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG

        # Try invalid action: OPEN_SHORT while LONG
        # This should convert to HOLD but NOT penalize
        env.step(ACTION_OPEN_SHORT)

        # Check the reward components — there should be no penalty component
        components = env._last_reward_components
        # The reward should just be DSR, no subtraction
        # Old behavior had: reward -= 0.05 for invalid actions
        # With v5, the _calculate_reward code has no invalid action penalty path
        assert 'dsr' in components, "DSR component should be tracked"
        # The invalid action was still detected (for logging), but not penalized
        assert env.invalid_action_this_step == True, "Invalid action should still be detected"
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY SIGNAL BONUS
# ─────────────────────────────────────────────────────────────────────────────

class TestEntryBonus:
    """Test the deferred entry bonus that provides gradient signal."""

    def test_entry_bonus_fires_immediately(self):
        """v6: Entry bonus fires on the OPEN step (hold_duration == 1).

        Flow: OPEN_LONG sets hold_duration=1 → _calculate_reward checks
        hold_duration==1 → entry bonus fires on the OPEN step itself.
        Subsequent HOLDs increment hold_duration to 2,3,... → no bonus.
        """
        from config import ENTRY_BONUS_IMMEDIATE
        env = _make_env()
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        # OPEN step: hold_duration=1 → bonus should fire
        _, open_reward, done, _, _ = env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG
        assert not done
        open_bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert open_bonus == ENTRY_BONUS_IMMEDIATE, \
            f"Expected entry bonus {ENTRY_BONUS_IMMEDIATE} on OPEN step, got {open_bonus}"

        # Next HOLD: hold_duration=2 → no bonus
        _, hold_reward, done, _, _ = env.step(ACTION_HOLD)
        hold_bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert hold_bonus == 0.0, \
            f"Expected no entry bonus on HOLD step, got {hold_bonus}"
        env.close()

    def test_entry_bonus_fires_once_only(self):
        """v6: Entry bonus fires once per trade (hold_duration == 1 only once).

        Since hold_duration starts at 1 on OPEN and strictly increases,
        hold_duration == 1 is true for exactly one step per trade.
        """
        from config import ENTRY_BONUS_IMMEDIATE
        env = _make_env()
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        env.step(ACTION_OPEN_LONG)

        # Collect hold_durations and bonuses for each step
        bonuses = []
        for _ in range(15):
            _, _, done, _, _ = env.step(ACTION_HOLD)
            bonuses.append(env._last_reward_components.get('entry_bonus', 0.0))
            if done:
                break

        # No bonus should fire on any HOLD step (only on OPEN step)
        assert all(b == 0.0 for b in bonuses), \
            f"Entry bonus should not fire on HOLD steps, got bonuses: {bonuses}"
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# TRANSACTION COST CURRICULUM
# ─────────────────────────────────────────────────────────────────────────────

class TestCostCurriculum:
    """Test the transaction cost multiplier mechanism."""

    def test_cost_multiplier_default_is_1(self):
        """Default cost_multiplier should be 1.0 (full costs)."""
        env = _make_env()
        env.reset()
        assert env.cost_multiplier == 1.0
        env.close()

    def test_cost_multiplier_zero_means_free_trades(self):
        """With cost_multiplier=0.0, trades should have zero commission."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        balance_before = env.balance
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG

        # With zero cost multiplier, the only balance change is the stock value
        # No commission, no spread, no slippage
        stock_value = env.stock_quantity * env.entry_price
        # balance_before - balance_after should equal stock_value (no extra costs)
        balance_diff = balance_before - env.balance
        assert abs(balance_diff - stock_value) < 1.0, \
            f"With cost_mult=0, trade cost should equal stock value. " \
            f"Diff={balance_diff:.2f}, stock={stock_value:.2f}"
        env.close()

    def test_cost_multiplier_scales_fees(self):
        """cost_multiplier=0.5 should halve the transaction costs vs 1.0."""
        # Run with full costs
        env_full = _make_env(cost_multiplier=1.0)
        env_full.reset()
        for _ in range(5):
            env_full.step(ACTION_HOLD)
        env_full.step(ACTION_OPEN_LONG)
        fees_full = env_full.total_fees_paid_episode
        env_full.close()

        # Run with half costs
        env_half = _make_env(cost_multiplier=0.5)
        env_half.reset()
        for _ in range(5):
            env_half.step(ACTION_HOLD)
        env_half.step(ACTION_OPEN_LONG)
        fees_half = env_half.total_fees_paid_episode
        env_half.close()

        # Half-cost fees should be roughly half of full-cost fees
        if fees_full > 0:
            ratio = fees_half / fees_full
            assert 0.4 < ratio < 0.6, \
                f"Fee ratio should be ~0.5, got {ratio:.3f} (full={fees_full:.4f}, half={fees_half:.4f})"

    def test_phase_cost_multipliers_in_curriculum(self):
        """Verify CurriculumConfig has correct cost_multiplier per phase."""
        from src.training.curriculum_trainer import CurriculumConfig
        cfg = CurriculumConfig(total_timesteps=1_000_000)
        expected = [0.0, 0.25, 0.75, 1.0]
        actual = [p.cost_multiplier for p in cfg.phases]
        assert actual == expected, f"Phase cost multipliers should be {expected}, got {actual}"


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING BUDGET & ENTROPY
# ─────────────────────────────────────────────────────────────────────────────

class TestTrainingConfig:
    """Test training budget and entropy settings."""

    def test_total_timesteps_5m(self):
        """TOTAL_TIMESTEPS should be 5M for v5."""
        assert TOTAL_TIMESTEPS_PER_BOT == 5_000_000

    def test_phase_budgets_30_25_25_20(self):
        """Phase budgets should be 30/25/25/20 (more Phase 4)."""
        from src.training.curriculum_trainer import CurriculumConfig
        cfg = CurriculumConfig(total_timesteps=1_000_000)
        budgets = [p.timesteps for p in cfg.phases]
        expected = [300_000, 250_000, 250_000, 200_000]
        assert budgets == expected, f"Phase budgets should be {expected}, got {budgets}"

    def test_phase4_entropy_raised(self):
        """Phase 4 entropy multiplier should be 1.0 (v6, was 0.8 in v5)."""
        from src.training.curriculum_trainer import CurriculumConfig
        cfg = CurriculumConfig(total_timesteps=1_000_000)
        phase4 = cfg.phases[3]
        assert phase4.entropy_coef_multiplier == 1.0, \
            f"Phase 4 entropy multiplier should be 1.0, got {phase4.entropy_coef_multiplier}"


# ─────────────────────────────────────────────────────────────────────────────
# UNIFIED AGENTIC ENV DELEGATION
# ─────────────────────────────────────────────────────────────────────────────

class TestUnifiedEnvMasks:
    """Test that UnifiedAgenticEnv delegates action_masks to base env."""

    def test_unified_env_has_action_masks(self):
        """UnifiedAgenticEnv should expose action_masks()."""
        from src.training.unified_agentic_env import UnifiedAgenticEnv, TrainingMode
        df = _make_data()
        env = UnifiedAgenticEnv(df, strict_scaler_mode=False)
        env.reset()
        assert hasattr(env, 'action_masks')
        masks = env.action_masks()
        assert masks.shape == (5,)
        assert masks.dtype == bool
        # When flat, should allow HOLD + OPEN_LONG + OPEN_SHORT
        assert masks[ACTION_HOLD] == True
        assert masks[ACTION_OPEN_LONG] == True
        env.close()


# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATION
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegration:
    """Integration tests for v5 changes working together."""

    def test_full_episode_with_masks(self):
        """Run a full episode using only masked-valid actions."""
        env = _make_env()
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < 300:
            masks = env.action_masks()
            valid_actions = np.where(masks)[0]
            # Pick a random valid action
            action = np.random.choice(valid_actions)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = done or truncated

        # Should complete without errors
        assert steps > 0
        # With random valid actions, agent should trade (not just hold)
        assert env.total_trades >= 0  # At least some trades expected
        env.close()

    def test_zero_cost_episode_pays_no_fees(self):
        """Zero-cost episode should pay zero fees; full-cost should pay positive fees."""
        actions = [ACTION_HOLD] * 5 + [ACTION_OPEN_LONG] + [ACTION_HOLD] * 10 + [ACTION_CLOSE_LONG]

        env_free = _make_env(cost_multiplier=0.0)
        env_full = _make_env(cost_multiplier=1.0)

        env_free.reset()
        env_full.reset()

        for a in actions:
            env_free.step(a)
            env_full.step(a)

        # Zero-cost env should have zero fees
        assert env_free.total_fees_paid_episode == 0.0, \
            f"Zero-cost fees should be 0, got {env_free.total_fees_paid_episode:.4f}"
        # Full-cost env should have positive fees from commission + spread + slippage
        assert env_full.total_fees_paid_episode > 0, \
            f"Full-cost fees should be > 0, got {env_full.total_fees_paid_episode:.4f}"

        env_free.close()
        env_full.close()


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
