# =============================================================================
# Tests for Sprint 2: Reward Function Restructuring
# =============================================================================
# 20 regression test scenarios + structural tests to verify the reward function
# aligns with Sprint 2 goals: no hold penalty, profitable-hold bonus,
# convex loss penalty, RR-based trade bonus, anti-churning.
# =============================================================================

import sys
import os
import math
import ast
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import HOLD_PENALTY_FACTOR, W_TURNOVER, W_DURATION, W_DRAWDOWN


# ===========================================================================
# Standalone reward function (extracted from environment.py for testability)
# ===========================================================================

POSITION_FLAT = 0
POSITION_LONG = 1
POSITION_SHORT = -1


def _calculate_reward(state: dict, previous_net_worth: float) -> tuple:
    """
    Standalone version of the reward function for testing.
    Takes a state dict instead of self, returns (reward, components).
    Must stay in sync with environment.py _calculate_reward.
    """
    import logging
    _logger = logging.getLogger("test_reward")

    net_worth = state['net_worth']
    initial_balance = state['initial_balance']
    peak_nav = state['peak_nav']
    previous_drawdown_level = state['previous_drawdown_level']
    transaction_cost = state['transaction_cost_incurred_step']
    traded_value = state['traded_value_step']
    current_leverage = state['current_leverage']
    max_leverage_limit = state['max_leverage_limit']
    position_type = state['position_type']
    entry_price = state['entry_price']
    stock_quantity = state['stock_quantity']
    balance = state['balance']
    trade_details = state['trade_details']
    invalid_action = state['invalid_action_this_step']
    minimum_balance = state['minimum_allowed_balance']
    reward_tanh_scale = state['reward_tanh_scale']
    reward_output_scale = state['reward_output_scale']
    stop_loss_pct = state['stop_loss_percentage']

    # STEP 1: VALIDATION
    if previous_net_worth <= 1e-9 or net_worth <= 1e-9:
        return -20.0, {}
    if np.isnan(net_worth) or np.isinf(net_worth):
        return -20.0, {}
    if np.isnan(previous_net_worth) or np.isinf(previous_net_worth):
        return 0.0, {}

    # STEP 2: CORE PROFITABILITY
    log_return = np.log(net_worth / previous_net_worth)
    profitability_reward = log_return * 100.0

    # STEP 3: PENALTIES
    total_penalty = 0.0
    dd_penalty = 0.0
    friction_penalty = 0.0
    leverage_penalty = 0.0
    turnover_penalty = 0.0

    current_drawdown = peak_nav - net_worth
    drawdown_increase = max(0.0, current_drawdown - previous_drawdown_level)
    if drawdown_increase > 0:
        dd_penalty = (drawdown_increase / initial_balance) * 10.0
        total_penalty += dd_penalty

    if transaction_cost > 0:
        friction_penalty = (transaction_cost / initial_balance) * 3.0
        total_penalty += friction_penalty

    leverage_excess = max(0.0, current_leverage - max_leverage_limit)
    if leverage_excess > 0:
        leverage_penalty = (leverage_excess ** 2) * 10.0
        total_penalty += leverage_penalty

    if traded_value > 0:
        turnover_ratio = traded_value / max(net_worth, 1.0)
        if turnover_ratio > 0.3:
            turnover_penalty = (turnover_ratio - 0.3) * 3.0
            total_penalty += turnover_penalty

    if invalid_action:
        total_penalty += 0.5

    # STEP 4: HOLD BONUS/PENALTY
    hold_reward = 0.0
    if position_type != POSITION_FLAT and not np.isnan(entry_price):
        if position_type == POSITION_LONG:
            current_price = net_worth - balance
            if stock_quantity > 0:
                current_price = current_price / stock_quantity
            else:
                current_price = entry_price
            unrealized_pnl = (current_price - entry_price) * stock_quantity
        else:  # SHORT
            unrealized_pnl = net_worth - balance

        unrealized_pnl_pct = unrealized_pnl / initial_balance
        if unrealized_pnl > 0:
            hold_reward = min(0.5, unrealized_pnl_pct * 100)
        else:
            loss_pct = abs(unrealized_pnl_pct)
            hold_reward = max(-5.0, -(loss_pct * 100) ** 1.5)

    # STEP 5: TRADE CLOSE BONUS
    trade_bonus = 0.0
    closed_trade_types = ['sell', 'close_long', 'close_short']
    if trade_details.get('trade_type') in closed_trade_types and trade_details.get('trade_success'):
        trade_pnl_abs = trade_details.get('trade_pnl_abs', 0.0)
        trade_pnl_pct = trade_details.get('trade_pnl_pct', 0.0)
        if trade_pnl_abs > 0:
            actual_rr = abs(trade_pnl_pct) / max(stop_loss_pct * 100, 0.1)
            trade_bonus = min(3.0, actual_rr)
        else:
            trade_bonus = -0.5

    # STEP 6: COMPOSITE + NORMALIZATION
    raw_reward = profitability_reward - total_penalty + hold_reward + trade_bonus
    normalized_reward = np.tanh(raw_reward * reward_tanh_scale)
    scaled_reward = normalized_reward * reward_output_scale

    # STEP 7: SPECIAL CASES
    if net_worth <= minimum_balance:
        return -20.0, {}

    if current_drawdown > 0:
        dd_ratio = current_drawdown / peak_nav
        if dd_ratio > 0.15:
            scaled_reward -= 5.0

    # STEP 8: CLIP
    final_reward = float(np.clip(scaled_reward, -20.0, 20.0))

    components = {
        'profitability': profitability_reward,
        'dd_penalty': dd_penalty,
        'friction_penalty': friction_penalty,
        'leverage_penalty': leverage_penalty,
        'turnover_penalty': turnover_penalty,
        'hold_reward': hold_reward,
        'trade_bonus': trade_bonus,
        'raw_reward': raw_reward,
        'final_reward': final_reward,
    }

    if np.isnan(final_reward) or np.isinf(final_reward):
        return 0.0, components

    return final_reward, components


def make_state(**overrides) -> dict:
    """Create a default state dict with optional overrides."""
    defaults = {
        'net_worth': 100_000.0,
        'initial_balance': 100_000.0,
        'balance': 100_000.0,
        'peak_nav': 100_000.0,
        'previous_drawdown_level': 0.0,
        'transaction_cost_incurred_step': 0.0,
        'traded_value_step': 0.0,
        'current_leverage': 0.0,
        'max_leverage_limit': 1.0,
        'position_type': POSITION_FLAT,
        'entry_price': float('nan'),
        'stock_quantity': 0.0,
        'trade_details': {
            'trade_pnl_abs': 0.0, 'trade_pnl_pct': 0.0,
            'trade_type': 'hold', 'trade_success': False
        },
        'invalid_action_this_step': False,
        'minimum_allowed_balance': 50_000.0,
        'reward_tanh_scale': 0.3,
        'reward_output_scale': 5.0,
        'stop_loss_percentage': 0.01,
    }
    defaults.update(overrides)
    return defaults


# ===========================================================================
# 1. CONFIG-LEVEL TESTS
# ===========================================================================

class TestConfigValues:

    def test_hold_penalty_is_zero(self):
        assert HOLD_PENALTY_FACTOR == 0.0

    def test_turnover_weight_positive(self):
        assert W_TURNOVER > 0

    def test_duration_weight_zero(self):
        assert W_DURATION == 0.0

    def test_drawdown_weight_increased(self):
        assert W_DRAWDOWN >= 1.0


# ===========================================================================
# 2. REGRESSION TESTS (20 Known Scenarios)
# ===========================================================================

class TestRewardRegression:

    def test_01_flat_hold_no_trade(self):
        """Holding flat with no trade should produce ~0 reward."""
        r, _ = _calculate_reward(make_state(), 100_000.0)
        assert abs(r) < 0.1, f"Flat hold should be ~0, got {r}"

    def test_02_flat_unchanged(self):
        """No change in net worth = zero profitability reward."""
        r, _ = _calculate_reward(make_state(), 100_000.0)
        assert abs(r) < 0.1, f"Unchanged should be ~0, got {r}"

    def test_03_small_profit_step(self):
        """0.1% gain should produce small positive reward."""
        state = make_state(
            net_worth=100_100.0,
            position_type=POSITION_LONG,
            entry_price=2000.0,
            stock_quantity=0.05,
            balance=100_000.0,
        )
        r, _ = _calculate_reward(state, 100_000.0)
        assert r > 0, f"Small profit should be positive, got {r}"

    def test_04_small_loss_step(self):
        """0.1% loss should produce negative reward."""
        state = make_state(net_worth=99_900.0, peak_nav=100_000.0)
        r, _ = _calculate_reward(state, 100_000.0)
        assert r < 0, f"Small loss should be negative, got {r}"

    def test_05_holding_profitable(self):
        """Holding a profitable position should produce positive reward."""
        state = make_state(
            position_type=POSITION_LONG, entry_price=2000.0,
            stock_quantity=0.05, net_worth=100_500.0, balance=100_000.0,
        )
        r, _ = _calculate_reward(state, 100_400.0)
        assert r > 0, f"Profitable hold should be positive, got {r}"

    def test_06_holding_losing(self):
        """Holding a losing position should get penalty."""
        state = make_state(
            position_type=POSITION_LONG, entry_price=2000.0,
            stock_quantity=0.05, net_worth=99_000.0, balance=99_000.0,
            peak_nav=100_000.0,
        )
        r, _ = _calculate_reward(state, 99_100.0)
        assert r < 0, f"Losing hold should be negative, got {r}"

    def test_07_profitable_beats_losing(self):
        """Reward for profitable hold must exceed reward for losing hold."""
        s1 = make_state(
            position_type=POSITION_LONG, entry_price=2000.0,
            stock_quantity=0.05, net_worth=100_500.0, balance=100_000.0,
        )
        r_profit, _ = _calculate_reward(s1, 100_400.0)

        s2 = make_state(
            position_type=POSITION_LONG, entry_price=2000.0,
            stock_quantity=0.05, net_worth=99_500.0, balance=99_500.0,
            peak_nav=100_000.0,
        )
        r_loss, _ = _calculate_reward(s2, 99_600.0)

        assert r_profit > r_loss

    def test_08_close_winning_rr2(self):
        """Closing a winning trade with RR=2 should give large positive."""
        state = make_state(
            net_worth=100_200.0,
            trade_details={
                'trade_type': 'close_long', 'trade_success': True,
                'trade_pnl_abs': 200.0, 'trade_pnl_pct': 2.0,
            },
        )
        r, _ = _calculate_reward(state, 100_000.0)
        assert r > 0.5, f"Winning trade RR=2 should give large positive, got {r}"

    def test_09_close_winning_rr05(self):
        """RR=2 should get more reward than RR=0.5."""
        s1 = make_state(
            net_worth=100_050.0,
            trade_details={
                'trade_type': 'close_long', 'trade_success': True,
                'trade_pnl_abs': 50.0, 'trade_pnl_pct': 0.5,
            },
        )
        r_small, _ = _calculate_reward(s1, 100_000.0)

        s2 = make_state(
            net_worth=100_200.0,
            trade_details={
                'trade_type': 'close_long', 'trade_success': True,
                'trade_pnl_abs': 200.0, 'trade_pnl_pct': 2.0,
            },
        )
        r_big, _ = _calculate_reward(s2, 100_000.0)

        assert r_big > r_small

    def test_10_close_losing_trade(self):
        """Closing a losing trade should produce negative reward."""
        state = make_state(
            net_worth=99_900.0, peak_nav=100_000.0,
            trade_details={
                'trade_type': 'close_long', 'trade_success': True,
                'trade_pnl_abs': -100.0, 'trade_pnl_pct': -1.0,
            },
        )
        r, _ = _calculate_reward(state, 100_000.0)
        assert r < 0, f"Losing trade should be negative, got {r}"

    def test_11_severe_drawdown(self):
        """5% drawdown should produce significantly negative reward."""
        state = make_state(
            net_worth=95_000.0, peak_nav=100_000.0,
            previous_drawdown_level=0.0,
        )
        r, _ = _calculate_reward(state, 96_000.0)
        assert r < -1.0, f"5% drawdown should give large negative, got {r}"

    def test_12_friction_penalty(self):
        """Trading with high fees should reduce reward."""
        s1 = make_state(transaction_cost_incurred_step=100.0)
        r_fees, _ = _calculate_reward(s1, 100_000.0)

        s2 = make_state(transaction_cost_incurred_step=0.0)
        r_no_fees, _ = _calculate_reward(s2, 100_000.0)

        assert r_no_fees > r_fees

    def test_13_turnover_penalty(self):
        """High turnover should be penalized."""
        s1 = make_state(traded_value_step=50_000.0)
        r_high, _ = _calculate_reward(s1, 100_000.0)

        s2 = make_state(traded_value_step=0.0)
        r_low, _ = _calculate_reward(s2, 100_000.0)

        assert r_low > r_high

    def test_14_invalid_action_penalty(self):
        """Invalid actions should get penalized."""
        s1 = make_state(invalid_action_this_step=True)
        r_invalid, _ = _calculate_reward(s1, 100_000.0)

        s2 = make_state(invalid_action_this_step=False)
        r_valid, _ = _calculate_reward(s2, 100_000.0)

        assert r_valid > r_invalid

    def test_15_account_depleted(self):
        """Account at minimum should return -20."""
        state = make_state(net_worth=49_000.0)
        r, _ = _calculate_reward(state, 50_000.0)
        assert r == -20.0

    def test_16_leverage_violation(self):
        """Leverage > limit should be penalized."""
        s1 = make_state(current_leverage=1.5, max_leverage_limit=1.0)
        r_leverage, _ = _calculate_reward(s1, 100_000.0)

        s2 = make_state(current_leverage=0.5)
        r_normal, _ = _calculate_reward(s2, 100_000.0)

        assert r_normal > r_leverage

    def test_17_extreme_drawdown_extra_penalty(self):
        """15%+ total drawdown should trigger additional penalty."""
        state = make_state(
            net_worth=84_000.0, peak_nav=100_000.0,
            previous_drawdown_level=15_000.0,
        )
        r, _ = _calculate_reward(state, 84_500.0)
        assert r < -2.0, f"16% DD should be very negative, got {r}"

    def test_18_nan_net_worth(self):
        """NaN net worth should return -20 (fail safe)."""
        state = make_state(net_worth=float('nan'))
        r, _ = _calculate_reward(state, 100_000.0)
        assert r == -20.0

    def test_19_close_short_winning(self):
        """Closing a profitable short should produce positive reward."""
        state = make_state(
            net_worth=100_150.0,
            trade_details={
                'trade_type': 'close_short', 'trade_success': True,
                'trade_pnl_abs': 150.0, 'trade_pnl_pct': 1.5,
            },
        )
        r, _ = _calculate_reward(state, 100_000.0)
        assert r > 0, f"Winning short close should be positive, got {r}"

    def test_20_reward_components_tracked(self):
        """Components dict should be populated after every call."""
        state = make_state(net_worth=100_100.0)
        _, components = _calculate_reward(state, 100_000.0)
        assert 'profitability' in components
        assert 'dd_penalty' in components
        assert 'hold_reward' in components
        assert 'trade_bonus' in components
        assert 'final_reward' in components


# ===========================================================================
# 3. REWARD RANGE TEST
# ===========================================================================

class TestRewardRange:

    def test_reward_bounded_across_random_scenarios(self):
        """Rewards must stay in [-20, +20] across 10,000 random scenarios."""
        np.random.seed(42)
        out_of_range = 0

        for _ in range(10_000):
            nw = np.random.uniform(60_000, 140_000)
            prev = np.random.uniform(60_000, 140_000)
            pnav = max(nw, prev, 100_000)
            pos_type = np.random.choice([0, 1, -1])
            ep = np.random.uniform(1800, 2200) if pos_type != 0 else float('nan')
            sq = np.random.uniform(0.01, 0.1) if pos_type != 0 else 0.0
            bal = nw - sq * ep if pos_type != 0 and not np.isnan(ep) else nw

            state = make_state(
                net_worth=nw, balance=bal, peak_nav=pnav,
                previous_drawdown_level=np.random.uniform(0, 5000),
                transaction_cost_incurred_step=np.random.uniform(0, 200),
                traded_value_step=np.random.uniform(0, 80_000),
                current_leverage=np.random.uniform(0, 2.0),
                position_type=pos_type,
                entry_price=ep,
                stock_quantity=sq,
            )

            r, _ = _calculate_reward(state, prev)
            if r < -20.0 or r > 20.0:
                out_of_range += 1

        assert out_of_range == 0, f"{out_of_range}/10000 rewards out of [-20, +20] range"


# ===========================================================================
# 4. SOURCE CODE VERIFICATION
# ===========================================================================

class TestSourceCodeVerification:

    def test_no_hold_penalty_in_reward_function(self):
        """The reward function should NOT contain hold_penalty = 0.01."""
        env_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'environment', 'environment.py'
        )
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Search for the old hold penalty pattern
        assert 'hold_penalty = 0.01' not in content, (
            "Old hold_penalty = 0.01 found in environment.py"
        )

    def test_reward_components_stored(self):
        """_calculate_reward should store _last_reward_components."""
        env_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'environment', 'environment.py'
        )
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert '_last_reward_components' in content, (
            "_last_reward_components not found in environment.py"
        )

    def test_no_print_in_reward_function(self):
        """The reward function should use logger, not print()."""
        env_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'environment', 'environment.py'
        )
        with open(env_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract just the _calculate_reward function
        start = content.find('def _calculate_reward(')
        end = content.find('\n    def ', start + 1)
        if end == -1:
            end = len(content)
        reward_fn_code = content[start:end]

        assert 'print(' not in reward_fn_code, (
            "print() found in _calculate_reward — use logger instead"
        )

    def test_ent_coef_is_001(self):
        """Sprint 2: ent_coef should be 0.01."""
        trainer_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'training', 'sophisticated_trainer.py'
        )
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()

        assert "'ent_coef': 0.01" in content, (
            "ent_coef should be 0.01 in sophisticated_trainer.py"
        )
