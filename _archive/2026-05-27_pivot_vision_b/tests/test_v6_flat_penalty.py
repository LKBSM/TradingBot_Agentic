# =============================================================================
# V6 Validation: Flat Inactivity Penalty, Immediate Entry Bonus, MaskableEval
# =============================================================================
# Tests for the 3-sprint fix that breaks the "always hold" zero-trade equilibrium.
#
# Run with: python -m pytest tests/test_v6_flat_penalty.py -v
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
    FLAT_PENALTY_WARMUP, FLAT_PENALTY_PER_BAR, FLAT_PENALTY_CAP,
    ENTRY_BONUS_IMMEDIATE, ENTRY_BONUS_MIN_PREV_HOLD,
)
from src.environment.environment import TradingEnv


def _make_data(n_rows=800, base_price=2000.0):
    """Create controlled test data with flat prices for predictable rewards."""
    np.random.seed(42)
    prices = base_price + np.cumsum(np.random.normal(0, 2, n_rows))
    prices = np.maximum(prices, base_price * 0.8)

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
# FLAT INACTIVITY PENALTY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestFlatPenalty:
    """Test the flat inactivity penalty that breaks the 'always hold' equilibrium."""

    def test_no_penalty_during_warmup(self):
        """No penalty should be applied during the first FLAT_PENALTY_WARMUP bars."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        penalties = []
        for _ in range(FLAT_PENALTY_WARMUP):
            _, r, done, _, _ = env.step(ACTION_HOLD)
            penalty = env._last_reward_components.get('flat_penalty', 0.0)
            penalties.append(penalty)
            if done:
                break

        assert all(p == 0.0 for p in penalties), \
            f"Expected no penalty during warmup, got: {penalties}"

    def test_penalty_starts_after_warmup(self):
        """Penalty should start after FLAT_PENALTY_WARMUP bars."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        # Hold for warmup period
        for _ in range(FLAT_PENALTY_WARMUP):
            env.step(ACTION_HOLD)

        # Next bar should have penalty
        _, r, done, _, _ = env.step(ACTION_HOLD)
        assert not done
        penalty = env._last_reward_components.get('flat_penalty', 0.0)
        expected = 1 * FLAT_PENALTY_PER_BAR  # 1 bar past warmup
        assert abs(penalty - expected) < 1e-6, \
            f"Expected penalty {expected} at bar {FLAT_PENALTY_WARMUP + 1}, got {penalty}"

    def test_penalty_linear_ramp(self):
        """Penalty should ramp linearly: bars_over_warmup * FLAT_PENALTY_PER_BAR."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        # Hold through warmup
        for _ in range(FLAT_PENALTY_WARMUP):
            env.step(ACTION_HOLD)

        # Check linear ramp for next 5 bars
        penalties = []
        for i in range(1, 6):
            _, r, done, _, _ = env.step(ACTION_HOLD)
            penalty = env._last_reward_components.get('flat_penalty', 0.0)
            expected = i * FLAT_PENALTY_PER_BAR
            penalties.append(penalty)
            assert abs(penalty - expected) < 1e-6, \
                f"At bar {FLAT_PENALTY_WARMUP + i}: expected penalty {expected}, got {penalty}"
            if done:
                break

    def test_penalty_cap(self):
        """Penalty should cap at FLAT_PENALTY_CAP."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        # Hold well past the cap point
        bars_to_cap = FLAT_PENALTY_WARMUP + int(FLAT_PENALTY_CAP / FLAT_PENALTY_PER_BAR) + 5
        for _ in range(bars_to_cap):
            _, r, done, _, _ = env.step(ACTION_HOLD)
            if done:
                break

        penalty = env._last_reward_components.get('flat_penalty', 0.0)
        assert abs(penalty - FLAT_PENALTY_CAP) < 1e-6, \
            f"Expected capped penalty {FLAT_PENALTY_CAP}, got {penalty}"

    def test_penalty_resets_on_open(self):
        """Flat penalty counter resets when a trade is opened."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        # Accumulate penalty
        for _ in range(FLAT_PENALTY_WARMUP + 5):
            env.step(ACTION_HOLD)

        # Verify penalty is active
        penalty_before = env._last_reward_components.get('flat_penalty', 0.0)
        assert penalty_before > 0, "Penalty should be active"

        # Open a long trade
        env.step(ACTION_OPEN_LONG)
        assert env._consecutive_flat_bars == 0, \
            "Flat bar counter should reset on open"

    def test_penalty_independent_of_cost_multiplier(self):
        """Flat penalty applies regardless of cost_multiplier (even at zero cost)."""
        env_zero = _make_env(cost_multiplier=0.0)
        env_full = _make_env(cost_multiplier=1.0)
        env_zero.reset()
        env_full.reset()

        # Both should accumulate flat penalty
        for _ in range(FLAT_PENALTY_WARMUP + 3):
            env_zero.step(ACTION_HOLD)
            env_full.step(ACTION_HOLD)

        p_zero = env_zero._last_reward_components.get('flat_penalty', 0.0)
        p_full = env_full._last_reward_components.get('flat_penalty', 0.0)

        assert p_zero > 0, "Penalty should apply at cost_multiplier=0"
        assert p_full > 0, "Penalty should apply at cost_multiplier=1"
        assert abs(p_zero - p_full) < 1e-6, \
            f"Penalty should be same regardless of cost_multiplier: {p_zero} vs {p_full}"

    def test_consecutive_flat_bars_counter(self):
        """_consecutive_flat_bars increments on flat and resets on open."""
        env = _make_env()
        env.reset()

        for i in range(5):
            env.step(ACTION_HOLD)
            assert env._consecutive_flat_bars == i + 1, \
                f"Expected {i + 1} flat bars, got {env._consecutive_flat_bars}"

        # Open trade — counter resets
        env.step(ACTION_OPEN_LONG)
        assert env._consecutive_flat_bars == 0

    def test_flat_counter_resets_in_env_reset(self):
        """_consecutive_flat_bars resets to 0 on env.reset()."""
        env = _make_env()
        env.reset()

        for _ in range(20):
            env.step(ACTION_HOLD)
        assert env._consecutive_flat_bars > 0

        env.reset()
        assert env._consecutive_flat_bars == 0


# ─────────────────────────────────────────────────────────────────────────────
# IMMEDIATE ENTRY BONUS TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestImmediateEntryBonus:
    """Test the v6 immediate entry bonus that replaces the v5 deferred bonus."""

    def test_bonus_on_first_trade(self):
        """First trade always gets entry bonus (bootstrap)."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        env.step(ACTION_OPEN_LONG)
        bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert bonus == ENTRY_BONUS_IMMEDIATE, \
            f"First trade should get bonus {ENTRY_BONUS_IMMEDIATE}, got {bonus}"

    def test_bonus_is_immediate_on_open_step(self):
        """Bonus fires on the OPEN step itself (hold_duration == 1)."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        # OPEN step
        _, open_reward, _, _, _ = env.step(ACTION_OPEN_LONG)
        assert env.current_hold_duration == 1
        open_bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert open_bonus == ENTRY_BONUS_IMMEDIATE

        # Next HOLD — hold_duration == 2, no bonus
        _, hold_reward, _, _, _ = env.step(ACTION_HOLD)
        assert env.current_hold_duration == 2
        hold_bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert hold_bonus == 0.0

    def test_bonus_on_short_trade(self):
        """Short trades also get entry bonus."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        env.step(ACTION_OPEN_SHORT)
        assert env.position_type == POSITION_SHORT
        bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert bonus == ENTRY_BONUS_IMMEDIATE, \
            f"Short trade should get bonus {ENTRY_BONUS_IMMEDIATE}, got {bonus}"

    def test_anti_churning_blocks_bonus_after_quick_close(self):
        """No bonus if previous trade held < ENTRY_BONUS_MIN_PREV_HOLD bars.

        Uses trade_cooldown_steps=0 to bypass cooldown and allow truly quick closes.
        Default cooldown (2 steps) ensures min duration == MIN_PREV_HOLD (3),
        so we disable it to test the anti-churning gate in isolation.
        """
        env = _make_env(cost_multiplier=0.0, trade_cooldown_steps=0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        # First trade — gets bonus (bootstrap, total_trades <= 1)
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG
        assert env._last_reward_components.get('entry_bonus', 0.0) == ENTRY_BONUS_IMMEDIATE

        # Close immediately (hold_duration = 2 on close step: 1 from open + 1 increment)
        env.step(ACTION_CLOSE_LONG)
        assert env.position_type == POSITION_FLAT, "Close should succeed"
        assert env.total_trades == 1, f"Expected 1 trade, got {env.total_trades}"
        assert env._last_trade_hold_duration < ENTRY_BONUS_MIN_PREV_HOLD, \
            f"Expected short hold duration, got {env._last_trade_hold_duration}"

        # Second trade — total_trades == 1, so total_trades <= 1 → bonus (grace period)
        for _ in range(3):
            env.step(ACTION_HOLD)
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG
        second_bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert second_bonus == ENTRY_BONUS_IMMEDIATE, \
            "Second trade gets bonus (total_trades <= 1 grace period)"

        # Close immediately again
        env.step(ACTION_CLOSE_LONG)
        assert env.position_type == POSITION_FLAT, "Second close should succeed"
        assert env.total_trades == 2, f"Expected 2 trades, got {env.total_trades}"
        last_dur = env._last_trade_hold_duration
        assert last_dur < ENTRY_BONUS_MIN_PREV_HOLD, \
            f"Expected short hold dur, got {last_dur}"

        # Third trade — total_trades == 2 (> 1) AND last_dur < MIN_PREV_HOLD → NO bonus
        for _ in range(3):
            env.step(ACTION_HOLD)
        assert env.position_type == POSITION_FLAT
        env.step(ACTION_OPEN_LONG)
        assert env.position_type == POSITION_LONG, "Third open should succeed"
        bonus = env._last_reward_components.get('entry_bonus', 0.0)
        assert bonus == 0.0, \
            f"Anti-churning should block bonus (total_trades={env.total_trades}, " \
            f"last_dur={env._last_trade_hold_duration}), got {bonus}"

    def test_bonus_after_adequate_hold(self):
        """Bonus given if previous trade held >= ENTRY_BONUS_MIN_PREV_HOLD bars."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        for _ in range(5):
            env.step(ACTION_HOLD)

        # First trade
        env.step(ACTION_OPEN_LONG)
        # Hold for adequate duration
        for _ in range(ENTRY_BONUS_MIN_PREV_HOLD + 2):
            _, _, done, _, _ = env.step(ACTION_HOLD)
            if done:
                break

        # Close trade
        if env.position_type == POSITION_LONG:
            env.step(ACTION_CLOSE_LONG)
            assert env._last_trade_hold_duration >= ENTRY_BONUS_MIN_PREV_HOLD

            # Second trade
            for _ in range(3):
                env.step(ACTION_HOLD)
            env.step(ACTION_OPEN_LONG)
            # Hold adequately again
            for _ in range(ENTRY_BONUS_MIN_PREV_HOLD + 2):
                _, _, done, _, _ = env.step(ACTION_HOLD)
                if done:
                    break

            # Close second trade
            if env.position_type == POSITION_LONG:
                env.step(ACTION_CLOSE_LONG)

                # Third trade — total_trades > 1 AND adequate prev hold → bonus
                for _ in range(3):
                    env.step(ACTION_HOLD)
                if env.position_type == POSITION_FLAT:
                    env.step(ACTION_OPEN_LONG)
                    bonus = env._last_reward_components.get('entry_bonus', 0.0)
                    assert bonus == ENTRY_BONUS_IMMEDIATE, \
                        f"Expected bonus after adequate hold, got {bonus}"

    def test_last_trade_hold_duration_recorded(self):
        """_last_trade_hold_duration is set on close and reset on env.reset()."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()
        assert env._last_trade_hold_duration == 0

        for _ in range(5):
            env.step(ACTION_HOLD)

        env.step(ACTION_OPEN_LONG)
        # Track how long we actually held (SL/TP may close us early)
        held_steps = 0
        for _ in range(7):
            _, _, done, _, _ = env.step(ACTION_HOLD)
            held_steps += 1
            if done or env.position_type != POSITION_LONG:
                break

        if env.position_type == POSITION_LONG:
            # Manual close: hold_duration increments +1 in the close step
            # (line 1697 runs before close action), so _last = current + 1
            expected_dur = env.current_hold_duration + 1
            env.step(ACTION_CLOSE_LONG)
            assert env._last_trade_hold_duration == expected_dur, \
                f"Expected last hold duration {expected_dur}, got {env._last_trade_hold_duration}"
        else:
            # SL/TP closed it: _last_trade_hold_duration should be set and > 0
            assert env._last_trade_hold_duration > 0, \
                "_last_trade_hold_duration should be set after SL/TP close"

        env.reset()
        assert env._last_trade_hold_duration == 0


# ─────────────────────────────────────────────────────────────────────────────
# REWARD INEQUALITY TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestRewardInequality:
    """Verify that always-hold is worse than trading (the key equilibrium break)."""

    def test_always_hold_produces_negative_reward(self):
        """200 steps of always-hold should produce deeply negative total reward."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        total_reward = 0.0
        for _ in range(200):
            _, r, done, _, _ = env.step(ACTION_HOLD)
            total_reward += r
            if done:
                break

        # Expected: penalty starts at bar 11, caps at 0.5/bar by bar 35
        # Total penalty should be substantially negative
        assert total_reward < -10.0, \
            f"Always-hold total reward should be << 0, got {total_reward:.2f}"

    def test_single_trade_positive_reward_zero_cost(self):
        """A single trade with zero costs should produce positive reward (bonus)."""
        env = _make_env(cost_multiplier=0.0)
        env.reset()

        total_reward = 0.0
        # Hold a few bars, then open
        for _ in range(5):
            _, r, done, _, _ = env.step(ACTION_HOLD)
            total_reward += r

        # Open long — gets +1.0 bonus
        _, r, done, _, _ = env.step(ACTION_OPEN_LONG)
        total_reward += r
        open_bonus = env._last_reward_components.get('entry_bonus', 0.0)

        # The entry bonus alone should be ENTRY_BONUS_IMMEDIATE = 1.0
        assert open_bonus == ENTRY_BONUS_IMMEDIATE
        # Even with a short flat period penalty, the trade entry should
        # provide positive signal. At 5 bars flat, no penalty yet (warmup=10).
        assert total_reward > 0, \
            f"Expected positive reward with single trade entry, got {total_reward:.2f}"


# ─────────────────────────────────────────────────────────────────────────────
# ENTROPY CONFIG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestEntropyConfig:
    """Test v6 entropy multiplier changes."""

    def test_phase1_entropy_multiplier(self):
        """Phase 1 entropy should be 3.0x (v6, was 2.0x in v5)."""
        from src.training.curriculum_trainer import CurriculumConfig
        config = CurriculumConfig(total_timesteps=1_000_000)
        phase1 = config.phases[0]
        assert phase1.entropy_coef_multiplier == 3.0, \
            f"Phase 1 entropy multiplier should be 3.0, got {phase1.entropy_coef_multiplier}"

    def test_phase4_entropy_multiplier(self):
        """Phase 4 entropy should be 1.0x (v6, was 0.8x in v5)."""
        from src.training.curriculum_trainer import CurriculumConfig
        config = CurriculumConfig(total_timesteps=1_000_000)
        phase4 = config.phases[3]
        assert phase4.entropy_coef_multiplier == 1.0, \
            f"Phase 4 entropy multiplier should be 1.0, got {phase4.entropy_coef_multiplier}"

    def test_middle_phases_unchanged(self):
        """Phase 2 and 3 entropy should be unchanged (1.5x, 1.0x)."""
        from src.training.curriculum_trainer import CurriculumConfig
        config = CurriculumConfig(total_timesteps=1_000_000)
        assert config.phases[1].entropy_coef_multiplier == 1.5
        assert config.phases[2].entropy_coef_multiplier == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# MASKABLE EVAL CALLBACK TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestMaskableEvalCallback:
    """Test that the MaskableEvalCallback passes action_masks."""

    def test_maskable_eval_callback_exists(self):
        """MaskableEvalCallback should be importable from curriculum_trainer."""
        from src.training.curriculum_trainer import MaskableEvalCallback
        assert MaskableEvalCallback is not None

    def test_maskable_eval_callback_has_on_step(self):
        """MaskableEvalCallback must implement _on_step."""
        from src.training.curriculum_trainer import MaskableEvalCallback
        assert hasattr(MaskableEvalCallback, '_on_step')

    def test_curriculum_predict_uses_masks(self):
        """_evaluate_final should pass action_masks when available."""
        # Verify the code passes masks by checking source
        import inspect
        from src.training.curriculum_trainer import CurriculumTrainer
        source = inspect.getsource(CurriculumTrainer._evaluate_final)
        assert 'action_masks' in source, \
            "_evaluate_final should reference action_masks for MaskablePPO"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestV6Config:
    """Test v6 config constants exist with correct values."""

    def test_flat_penalty_warmup(self):
        assert FLAT_PENALTY_WARMUP == 10

    def test_flat_penalty_per_bar(self):
        assert FLAT_PENALTY_PER_BAR == 0.02

    def test_flat_penalty_cap(self):
        assert FLAT_PENALTY_CAP == 0.5

    def test_entry_bonus_immediate(self):
        assert ENTRY_BONUS_IMMEDIATE == 1.0

    def test_entry_bonus_min_prev_hold(self):
        assert ENTRY_BONUS_MIN_PREV_HOLD == 3

    def test_training_version(self):
        """TRAINING_VERSION in colab script should be v6."""
        # Read the colab script and check
        colab_path = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'colab_training_full.py')
        with open(colab_path, 'r') as f:
            content = f.read()
        assert 'v6_flat_penalty_entry_bonus' in content, \
            "colab_training_full.py should have TRAINING_VERSION = v6_flat_penalty_entry_bonus"
