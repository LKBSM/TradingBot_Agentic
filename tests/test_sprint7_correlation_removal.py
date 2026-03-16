# =============================================================================
# Sprint 7 Validation: Dead Correlation Multiplier Removal (PS-3)
# =============================================================================
# Verifies that:
# 1. correlation_multiplier code has been removed from position sizing
# 2. Position sizing is identical regardless of market_state['correlation_multiplier']
# 3. No regression on existing position sizing behavior
#
# Run with: python -m pytest tests/test_sprint7_correlation_removal.py -v
# =============================================================================

import sys
import os
import inspect

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.environment.risk_manager import DynamicRiskManager


def _make_risk_manager():
    """Create a DynamicRiskManager with standard config."""
    config = {
        'RISK_PERCENTAGE_PER_TRADE': 0.01,
        'STOP_LOSS_PERCENTAGE': 0.01,
        'TAKE_PROFIT_PERCENTAGE': 0.02,
        'MIN_TRADE_QUANTITY': 0.001,
    }
    rm = DynamicRiskManager(config)
    rm.set_client_profile("test", 100000, 15.0, 0.25, 0.02)
    return rm


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: correlation_multiplier not used in position sizing code
# ─────────────────────────────────────────────────────────────────────────────
def test_no_correlation_multiplier_in_sizing_logic():
    """The calculate_adaptive_position_size method should not apply correlation_multiplier."""
    source = inspect.getsource(DynamicRiskManager.calculate_adaptive_position_size)
    # The comment about removal is fine, but operational code should not reference it
    assert 'final_size *= correlation_multiplier' not in source, (
        "correlation_multiplier still being applied in position sizing"
    )
    assert "market_state.get('correlation_multiplier'" not in source, (
        "correlation_multiplier still being read from market_state"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Position size ignores correlation_multiplier in market_state
# ─────────────────────────────────────────────────────────────────────────────
def test_size_ignores_correlation_multiplier():
    """Setting correlation_multiplier in market_state should NOT affect position size."""
    rm = _make_risk_manager()

    # Size with multiplier = 1.0
    rm.market_state['correlation_multiplier'] = 1.0
    size_full = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
    )

    # Size with multiplier = 0.3 (would have reduced by 70% before Sprint 7)
    rm.market_state['correlation_multiplier'] = 0.3
    size_reduced = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
    )

    assert size_full > 0
    assert size_full == size_reduced, (
        f"Position sizes should be identical now that correlation_multiplier "
        f"is removed, got {size_full} vs {size_reduced}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Position sizing still works correctly (regression)
# ─────────────────────────────────────────────────────────────────────────────
def test_position_sizing_regression():
    """Verify triple constraint system still produces valid sizes."""
    rm = _make_risk_manager()

    size = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
    )
    assert size > 0, "Position size should be positive with positive edge"

    # Verify leverage constraint works
    max_leverage_size = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0, max_leverage=0.5
    )
    assert max_leverage_size <= 100000 * 0.5 / 2000.0 * 1.01, (
        "Size should respect leverage limit"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Kelly criterion still works (regression)
# ─────────────────────────────────────────────────────────────────────────────
def test_kelly_still_works():
    """Verify Kelly criterion still drives position sizing."""
    rm = _make_risk_manager()

    # Higher win prob → larger size
    size_high = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.70, 2.0, current_price=2000.0
    )
    size_low = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.40, 2.0, current_price=2000.0
    )

    # With win_prob=0.40, R:R=2.0: Kelly = (2*0.4 - 0.6)/2 = 0.1 → positive
    # With win_prob=0.70, R:R=2.0: Kelly = (2*0.7 - 0.3)/2 = 0.55 → larger
    assert size_high >= size_low, (
        f"Higher win prob should give >= position size: {size_high} vs {size_low}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Training/live mode still works (Sprint 5 regression)
# ─────────────────────────────────────────────────────────────────────────────
def test_training_live_mode_still_works():
    """Verify Sprint 5 conditional Kelly floor is preserved."""
    rm = _make_risk_manager()

    # Training mode: Kelly floor at 0.02 even with zero edge
    size_training = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.30, 1.0,  # Kelly = 0 (no edge)
        current_price=2000.0, training_mode=True
    )

    # Live mode: Kelly=0 → no trade
    size_live = rm.calculate_adaptive_position_size(
        "test", 100000, 50.0, 0.30, 1.0,
        current_price=2000.0, training_mode=False
    )

    assert size_training > 0, "Training mode should force minimum position"
    assert size_live == 0.0, "Live mode with no edge should return 0"


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: comment about removal exists as documentation
# ─────────────────────────────────────────────────────────────────────────────
def test_removal_documented():
    """Sprint 7 removal should be documented in the source."""
    source = inspect.getsource(DynamicRiskManager.calculate_adaptive_position_size)
    assert 'Sprint 7' in source, "Sprint 7 removal should be documented"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
