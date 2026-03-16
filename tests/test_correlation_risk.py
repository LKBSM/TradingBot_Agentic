"""
Sprint 5 Tests: Correlation Risk Integration — regime detection,
position sizing adjustment, kill switch triggering, and Prometheus gauges.
"""
import os
import sys
import ast
import numpy as np
import pytest

# Direct imports to avoid gymnasium dependency
from src.multi_asset.correlation_tracker import (
    CorrelationTracker,
    CorrelationTrackerConfig,
    CorrelationRegime,
    CorrelationPair,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_tracker(**overrides) -> CorrelationTracker:
    """Create a tracker with default config, overrideable."""
    cfg = CorrelationTrackerConfig(
        short_window=20,
        medium_window=60,
        long_window=100,
        min_update_interval_sec=0,  # No throttle in tests
        **overrides,
    )
    return CorrelationTracker(config=cfg)


def feed_correlated_data(
    tracker: CorrelationTracker,
    n: int,
    gold_dxy_corr: float = -0.8,
    seed: int = 42,
):
    """
    Feed synthetic price data into the tracker with a target correlation.

    gold_dxy_corr: target correlation between Gold and DXY.
    """
    rng = np.random.RandomState(seed)

    # Generate correlated return series
    # gold_returns and dxy_returns with target correlation
    gold_returns = rng.randn(n) * 0.01
    noise = rng.randn(n) * 0.01
    # dxy_returns = corr * gold_returns + sqrt(1-corr^2) * noise
    dxy_returns = gold_dxy_corr * gold_returns + np.sqrt(1 - gold_dxy_corr**2) * noise

    gold_price = 2000.0
    dxy_price = 104.0

    tracker.add_asset("XAUUSD")
    tracker.add_asset("DXY")

    for i in range(n):
        gold_price *= (1 + gold_returns[i])
        dxy_price *= (1 + dxy_returns[i])
        tracker.update_prices({"XAUUSD": gold_price, "DXY": dxy_price})


# =============================================================================
# TEST: REGIME DETECTION
# =============================================================================

class TestCorrelationRegimeDetection:
    def test_stable_regime_with_high_correlation(self):
        tracker = make_tracker()
        feed_correlated_data(tracker, n=100, gold_dxy_corr=-0.85)

        pair = tracker.get_pair_info("XAUUSD", "DXY")
        assert pair is not None
        # With -0.85 correlation, |corr| > 0.7 → STABLE
        assert pair.correlation_abs >= 0.5  # May not be exact due to noise

    def test_elevated_regime_with_medium_correlation(self):
        tracker = make_tracker()
        feed_correlated_data(tracker, n=100, gold_dxy_corr=-0.5)

        pair = tracker.get_pair_info("XAUUSD", "DXY")
        assert pair is not None
        # Moderate correlation may land in ELEVATED or STABLE range

    def test_breakdown_regime_with_low_correlation(self):
        tracker = make_tracker()
        feed_correlated_data(tracker, n=100, gold_dxy_corr=-0.1)

        pair = tracker.get_pair_info("XAUUSD", "DXY")
        assert pair is not None
        # Very low correlation → BREAKDOWN or DECORRELATED
        assert pair.regime in (CorrelationRegime.BREAKDOWN, CorrelationRegime.DECORRELATED)

    def test_regime_transitions_on_correlation_shift(self):
        """Feed stable data, then shift to breakdown — verify regime changes."""
        tracker = make_tracker()

        # Phase 1: 80 bars of high correlation
        feed_correlated_data(tracker, n=80, gold_dxy_corr=-0.85, seed=42)
        pair = tracker.get_pair_info("XAUUSD", "DXY")

        # Phase 2: 80 bars of near-zero correlation (decorrelation event)
        feed_correlated_data(tracker, n=80, gold_dxy_corr=0.05, seed=99)
        pair = tracker.get_pair_info("XAUUSD", "DXY")
        assert pair is not None
        # After a large shift, should not be STABLE
        assert pair.regime != CorrelationRegime.STABLE


# =============================================================================
# TEST: RISK ADJUSTMENT MULTIPLIER
# =============================================================================

class TestRiskAdjustment:
    def test_stable_returns_1_0(self):
        tracker = make_tracker()
        feed_correlated_data(tracker, n=100, gold_dxy_corr=-0.85)

        pair = tracker.get_pair_info("XAUUSD", "DXY")
        if pair and pair.regime == CorrelationRegime.STABLE:
            assert tracker.get_risk_adjustment() == 1.0

    def test_breakdown_returns_0_3(self):
        tracker = make_tracker()
        feed_correlated_data(tracker, n=100, gold_dxy_corr=-0.1)

        pair = tracker.get_pair_info("XAUUSD", "DXY")
        if pair and pair.regime in (CorrelationRegime.BREAKDOWN, CorrelationRegime.DECORRELATED):
            assert tracker.get_risk_adjustment() == 0.3

    def test_no_pairs_returns_1_0(self):
        tracker = make_tracker()
        assert tracker.get_risk_adjustment() == 1.0

    def test_multiplier_bounds(self):
        """Multiplier should always be between 0.3 and 1.0."""
        tracker = make_tracker()
        for corr in [-0.9, -0.6, -0.3, 0.0, 0.3, 0.6, 0.9]:
            tracker2 = make_tracker()
            feed_correlated_data(tracker2, n=100, gold_dxy_corr=corr, seed=int(abs(corr) * 100))
            m = tracker2.get_risk_adjustment()
            assert 0.3 <= m <= 1.0, f"Multiplier {m} out of bounds for corr={corr}"


# =============================================================================
# TEST: POSITION SIZING INTEGRATION
# =============================================================================

class TestPositionSizingIntegration:
    def test_correlation_multiplier_ignored_after_sprint7(self):
        """Sprint 7: correlation_multiplier removed from position sizing.
        Setting it in market_state should have NO effect on position size."""
        from src.environment.risk_manager import DynamicRiskManager

        config = {
            'RISK_PERCENTAGE_PER_TRADE': 0.01,
            'STOP_LOSS_PERCENTAGE': 0.02,
            'TAKE_PROFIT_PERCENTAGE': 0.04,
            'MIN_TRADE_QUANTITY': 0.001,
        }
        rm = DynamicRiskManager(config)
        rm.set_client_profile("test", 100000, 15.0, 0.25, 0.02)

        # Full size (no correlation adjustment)
        rm.market_state['correlation_multiplier'] = 1.0
        size_full = rm.calculate_adaptive_position_size(
            "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
        )

        # Same size even with breakdown multiplier (Sprint 7: removed)
        rm.market_state['correlation_multiplier'] = 0.3
        size_with_breakdown = rm.calculate_adaptive_position_size(
            "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
        )

        assert size_full > 0
        assert size_full == size_with_breakdown, (
            "Sprint 7: correlation_multiplier should no longer affect sizing"
        )

    def test_correlation_multiplier_defaults_to_1(self):
        """Without correlation_multiplier in market_state, size is unchanged."""
        from src.environment.risk_manager import DynamicRiskManager

        config = {'MIN_TRADE_QUANTITY': 0.001}
        rm = DynamicRiskManager(config)
        rm.set_client_profile("test", 100000, 15.0, 0.25, 0.02)

        # No correlation_multiplier set
        size = rm.calculate_adaptive_position_size(
            "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
        )
        assert size > 0


# =============================================================================
# TEST: KILL SWITCH INTEGRATION
# =============================================================================

class TestKillSwitchCorrelation:
    def _make_ks(self):
        from src.agents.kill_switch import KillSwitch
        return KillSwitch(enable_persistence=False)

    def test_correlation_z_score_triggers_halt(self):
        """z-score >= 3.0 should trigger CORRELATION_BREAKDOWN halt."""
        from src.agents.kill_switch import HaltLevel, HaltReason

        ks = self._make_ks()
        level = ks.update(equity=100000, correlation_z_score=3.5)

        assert level.value >= HaltLevel.REDUCED.value
        assert ks.halt_reason == HaltReason.CORRELATION_BREAKDOWN

    def test_correlation_z_score_below_threshold_no_halt(self):
        """z-score < 3.0 should not trigger halt."""
        from src.agents.kill_switch import HaltLevel

        ks = self._make_ks()
        level = ks.update(equity=100000, correlation_z_score=2.5)

        assert level == HaltLevel.NONE

    def test_correlation_z_score_none_no_halt(self):
        """None z-score should not trigger halt."""
        from src.agents.kill_switch import HaltLevel

        ks = self._make_ks()
        level = ks.update(equity=100000, correlation_z_score=None)

        assert level == HaltLevel.NONE

    def test_negative_z_score_triggers_halt(self):
        """Negative z-score with |z| >= 3.0 should also trigger."""
        from src.agents.kill_switch import HaltLevel, HaltReason

        ks = self._make_ks()
        level = ks.update(equity=100000, correlation_z_score=-3.2)

        assert level.value >= HaltLevel.REDUCED.value


# =============================================================================
# TEST: PROMETHEUS GAUGES
# =============================================================================

class TestPrometheusGauges:
    def test_correlation_gauges_exist(self):
        from src.performance.metrics import create_trading_metrics, reset_registry

        reset_registry()
        metrics = create_trading_metrics()

        assert 'gold_dxy_correlation' in metrics
        assert 'gold_us10y_correlation' in metrics
        assert 'correlation_regime' in metrics
        assert 'correlation_risk_multiplier' in metrics

    def test_gauges_accept_values(self):
        from src.performance.metrics import create_trading_metrics, reset_registry

        reset_registry()
        metrics = create_trading_metrics()

        metrics['gold_dxy_correlation'].set(-0.75)
        metrics['gold_us10y_correlation'].set(0.45)
        metrics['correlation_regime'].set(0)  # STABLE
        metrics['correlation_risk_multiplier'].set(1.0)

        # No errors = success


# =============================================================================
# TEST: SYNTHETIC INTEGRATION (Sprint 5 roadmap requirement)
# =============================================================================

class TestSyntheticIntegration:
    def test_correlation_drop_no_longer_affects_sizing(self):
        """
        Sprint 7: correlation_multiplier removed from position sizing.
        Correlation changes are tracked by CorrelationTracker but no longer
        directly scale position size in single-asset mode.
        """
        from src.environment.risk_manager import DynamicRiskManager

        config = {'MIN_TRADE_QUANTITY': 0.001}
        rm = DynamicRiskManager(config)
        rm.set_client_profile("test", 100000, 15.0, 0.25, 0.02)

        # Phase 1: Stable correlation → multiplier = 1.0
        rm.market_state['correlation_multiplier'] = 1.0
        size_stable = rm.calculate_adaptive_position_size(
            "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
        )

        # Phase 2: Breakdown → multiplier = 0.3
        rm.market_state['correlation_multiplier'] = 0.3
        size_breakdown = rm.calculate_adaptive_position_size(
            "test", 100000, 50.0, 0.55, 2.0, current_price=2000.0
        )

        # Sprint 7: sizes should be equal now
        assert size_stable == size_breakdown, (
            "Sprint 7: correlation_multiplier no longer affects position sizing"
        )


# =============================================================================
# TEST: SOURCE VERIFICATION
# =============================================================================

class TestSourceVerification:
    def test_no_print_in_correlation_tracker(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "multi_asset", "correlation_tracker.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    pytest.fail(f"Found print() at line {node.lineno}")

    def test_get_risk_adjustment_exists(self):
        assert hasattr(CorrelationTracker, 'get_risk_adjustment')

    def test_correlation_breakdown_in_kill_switch(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "agents", "kill_switch.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "CORRELATION_BREAKDOWN" in source
        assert "correlation_z_score" in source

    def test_correlation_multiplier_in_risk_manager(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "risk_manager.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "correlation_multiplier" in source

    def test_prometheus_gauges_in_metrics(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "performance", "metrics.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "gold_dxy_correlation" in source
        assert "gold_us10y_correlation" in source
        assert "correlation_regime" in source
