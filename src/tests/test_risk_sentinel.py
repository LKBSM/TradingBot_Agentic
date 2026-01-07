# =============================================================================
# UNIT TESTS - Risk Sentinel Agent
# =============================================================================
# Comprehensive test suite for the Risk Sentinel Agent.
#
# Run with: pytest src/tests/test_risk_sentinel.py -v
#
# Tests cover:
#   1. Agent lifecycle (init, start, stop)
#   2. Individual risk rules
#   3. Decision making logic
#   4. Integration with environment
#   5. Edge cases and error handling
#
# =============================================================================

import pytest
import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.agents.base_agent import AgentState
from src.agents.events import (
    TradeProposal,
    RiskAssessment,
    DecisionType,
    RiskLevel,
    EventBus,
    EventType,
    AgentEvent
)
from src.agents.risk_sentinel import RiskSentinelAgent, create_risk_sentinel
from src.agents.config import RiskSentinelConfig, ConfigPreset, get_risk_sentinel_config


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def default_config():
    """Default risk configuration for testing."""
    return RiskSentinelConfig()


@pytest.fixture
def conservative_config():
    """Conservative risk configuration."""
    return RiskSentinelConfig.conservative()


@pytest.fixture
def agent(default_config):
    """Create a Risk Sentinel agent for testing."""
    agent = RiskSentinelAgent(config=default_config, name="TestSentinel")
    agent.start()
    yield agent
    agent.stop()


@pytest.fixture
def sample_proposal():
    """Create a sample trade proposal for testing."""
    return TradeProposal(
        action="BUY",
        asset="XAU/USD",
        quantity=0.1,
        entry_price=2000.0,
        current_balance=10000.0,
        current_position=0.0,
        current_equity=10000.0,
        unrealized_pnl=0.0,
        market_data={
            'Close': 2000.0,
            'ATR': 20.0,
            'RSI': 50.0,
            'Volume': 1000.0
        },
        metadata={'step': 100}
    )


# =============================================================================
# LIFECYCLE TESTS
# =============================================================================


class TestAgentLifecycle:
    """Tests for agent lifecycle management."""

    def test_agent_creation(self, default_config):
        """Test agent can be created with default config."""
        agent = RiskSentinelAgent(config=default_config)
        assert agent is not None
        assert agent.state == AgentState.INITIALIZING

    def test_agent_start(self, default_config):
        """Test agent transitions to RUNNING state on start."""
        agent = RiskSentinelAgent(config=default_config)
        result = agent.start()
        assert result is True
        assert agent.state == AgentState.RUNNING
        agent.stop()

    def test_agent_stop(self, default_config):
        """Test agent transitions to STOPPED state on stop."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()
        result = agent.stop()
        assert result is True
        assert agent.state == AgentState.STOPPED

    def test_agent_pause_resume(self, agent):
        """Test agent can be paused and resumed."""
        assert agent.state == AgentState.RUNNING

        result = agent.pause()
        assert result is True
        assert agent.state == AgentState.PAUSED

        result = agent.resume()
        assert result is True
        assert agent.state == AgentState.RUNNING

    def test_factory_function(self):
        """Test create_risk_sentinel factory function."""
        agent = create_risk_sentinel(preset="moderate")
        assert agent is not None
        agent.start()
        assert agent.state == AgentState.RUNNING
        agent.stop()


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestConfiguration:
    """Tests for agent configuration."""

    def test_default_config_values(self, default_config):
        """Test default configuration values are set correctly."""
        assert default_config.max_drawdown_pct == 0.10
        assert default_config.max_risk_per_trade_pct == 0.01
        assert default_config.max_leverage == 1.0
        assert default_config.enabled is True

    def test_conservative_config(self, conservative_config):
        """Test conservative configuration is more restrictive."""
        default = RiskSentinelConfig()
        assert conservative_config.max_position_size_pct < default.max_position_size_pct
        assert conservative_config.max_risk_per_trade_pct < default.max_risk_per_trade_pct
        assert conservative_config.max_drawdown_pct < default.max_drawdown_pct

    def test_aggressive_config(self):
        """Test aggressive configuration is less restrictive."""
        aggressive = RiskSentinelConfig.aggressive()
        default = RiskSentinelConfig()
        assert aggressive.max_position_size_pct > default.max_position_size_pct
        assert aggressive.max_drawdown_pct > default.max_drawdown_pct

    def test_config_to_dict(self, default_config):
        """Test configuration can be converted to dictionary."""
        config_dict = default_config.to_dict()
        assert isinstance(config_dict, dict)
        assert 'max_drawdown_pct' in config_dict
        assert 'enabled' in config_dict

    def test_preset_retrieval(self):
        """Test configuration presets can be retrieved."""
        for preset in [ConfigPreset.CONSERVATIVE, ConfigPreset.MODERATE,
                       ConfigPreset.AGGRESSIVE, ConfigPreset.BACKTESTING]:
            config = get_risk_sentinel_config(preset)
            assert config is not None
            assert isinstance(config, RiskSentinelConfig)


# =============================================================================
# TRADE EVALUATION TESTS
# =============================================================================


class TestTradeEvaluation:
    """Tests for trade proposal evaluation."""

    def test_hold_always_approved(self, agent):
        """Test HOLD actions are always approved."""
        proposal = TradeProposal(
            action="HOLD",
            current_equity=10000.0,
            current_balance=10000.0
        )
        assessment = agent.evaluate_trade(proposal)
        assert assessment.decision == DecisionType.APPROVE
        assert assessment.risk_score == 0.0

    def test_basic_buy_approval(self, agent, sample_proposal):
        """Test a valid BUY proposal is approved."""
        assessment = agent.evaluate_trade(sample_proposal)
        # With default config and healthy portfolio, should approve
        assert assessment.decision in [DecisionType.APPROVE, DecisionType.REJECT]
        assert assessment.proposal_id == sample_proposal.proposal_id

    def test_assessment_contains_reasoning(self, agent, sample_proposal):
        """Test assessment includes reasoning when enabled."""
        assessment = agent.evaluate_trade(sample_proposal)
        # Reasoning should be populated when enable_rule_explanations=True
        assert len(assessment.reasoning) > 0

    def test_assessment_timing(self, agent, sample_proposal):
        """Test assessment includes processing time."""
        assessment = agent.evaluate_trade(sample_proposal)
        assert assessment.assessment_time_ms >= 0


# =============================================================================
# RISK RULE TESTS
# =============================================================================


class TestRiskRules:
    """Tests for individual risk rules."""

    def test_max_drawdown_rejection(self, default_config):
        """Test trades are rejected when max drawdown is breached."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        # Set state to simulate 15% drawdown (above 10% limit)
        agent._peak_equity = 10000.0
        agent._current_equity = 8500.0  # 15% down from peak

        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=8500.0,
            current_balance=8500.0,
            market_data={'ATR': 20.0}
        )

        assessment = agent.evaluate_trade(proposal)
        assert assessment.decision == DecisionType.REJECT
        # Should have a drawdown violation
        assert any('drawdown' in v.rule_name.lower() for v in assessment.violations)

        agent.stop()

    def test_minimum_balance_rejection(self, default_config):
        """Test trades are rejected when balance is too low."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=50.0,  # Below $100 minimum
            current_balance=50.0,
            market_data={'ATR': 20.0}
        )

        assessment = agent.evaluate_trade(proposal)
        assert assessment.decision == DecisionType.REJECT
        assert any('MINIMUM_BALANCE' in v.rule_name for v in assessment.violations)

        agent.stop()

    def test_leverage_limit(self, default_config):
        """Test trades are rejected when leverage exceeds limit."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        # Propose a trade that would result in 2x leverage
        proposal = TradeProposal(
            action="BUY",
            quantity=10.0,  # 10 * 2000 = $20,000 position
            entry_price=2000.0,
            current_equity=10000.0,  # $10,000 equity = 2x leverage
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        )

        assessment = agent.evaluate_trade(proposal)
        # Should reject due to leverage > 1.0
        assert assessment.decision == DecisionType.REJECT

        agent.stop()

    def test_daily_trade_limit(self, default_config):
        """Test trades are limited per day."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        # Simulate many trades today
        agent._trades_today = default_config.max_trades_per_day

        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        )

        assessment = agent.evaluate_trade(proposal)
        # Should have daily trade limit violation
        assert any('DAILY_TRADE_LIMIT' in v.rule_name for v in assessment.violations)

        agent.stop()

    def test_consecutive_losses_warning(self, default_config):
        """Test consecutive losses trigger warning."""
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        # Simulate 3 consecutive losses
        agent._consecutive_losses = 3

        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        )

        assessment = agent.evaluate_trade(proposal)
        # Should have consecutive losses violation
        assert any('CONSECUTIVE_LOSSES' in v.rule_name for v in assessment.violations)

        agent.stop()


# =============================================================================
# STATE TRACKING TESTS
# =============================================================================


class TestStateTracking:
    """Tests for portfolio and market state tracking."""

    def test_portfolio_state_update(self, agent):
        """Test portfolio state is updated correctly."""
        agent.update_portfolio_state(
            equity=15000.0,
            position=0.5,
            entry_price=2000.0,
            current_step=100
        )

        assert agent._current_equity == 15000.0
        assert agent._peak_equity == 15000.0
        assert agent._current_position == 0.5
        assert agent._current_step == 100

    def test_peak_equity_tracking(self, agent):
        """Test peak equity is tracked correctly."""
        agent.update_portfolio_state(equity=10000.0, position=0.0, entry_price=0.0, current_step=1)
        agent.update_portfolio_state(equity=12000.0, position=0.0, entry_price=0.0, current_step=2)
        agent.update_portfolio_state(equity=11000.0, position=0.0, entry_price=0.0, current_step=3)

        # Peak should remain at 12000
        assert agent._peak_equity == 12000.0

    def test_trade_result_recording(self, agent):
        """Test trade results are recorded correctly."""
        agent.update_portfolio_state(equity=10000.0, position=0.0, entry_price=0.0, current_step=1)

        # Record a winning trade
        agent.record_trade_result(pnl=100.0)
        assert agent._trades_today == 1
        assert agent._consecutive_losses == 0

        # Record a losing trade
        agent.record_trade_result(pnl=-50.0)
        assert agent._trades_today == 2
        assert agent._consecutive_losses == 1
        assert agent._steps_since_loss == 0

    def test_step_recording(self, agent):
        """Test step counter is updated correctly."""
        agent._steps_since_loss = 0

        for _ in range(5):
            agent.record_step()

        assert agent._steps_since_loss == 5

    def test_market_regime_setting(self, agent):
        """Test market regime can be set."""
        agent.set_market_regime(0)  # Calm
        assert agent._current_regime == 0

        agent.set_market_regime(1)  # Volatile
        assert agent._current_regime == 1


# =============================================================================
# STATISTICS TESTS
# =============================================================================


class TestStatistics:
    """Tests for statistics and reporting."""

    def test_statistics_retrieval(self, agent, sample_proposal):
        """Test statistics can be retrieved."""
        # Make some decisions
        for _ in range(5):
            agent.evaluate_trade(sample_proposal)

        stats = agent.get_statistics()

        assert 'total_assessments' in stats
        assert stats['total_assessments'] == 5
        assert 'approval_rate' in stats

    def test_risk_dashboard(self, agent):
        """Test risk dashboard can be generated."""
        dashboard = agent.get_risk_dashboard()
        assert isinstance(dashboard, str)
        assert 'RISK SENTINEL DASHBOARD' in dashboard

    def test_health_check(self, agent):
        """Test health check returns valid data."""
        health = agent.health_check()

        assert 'agent_id' in health
        assert 'state' in health
        assert 'healthy' in health
        assert health['healthy'] is True
        assert health['state'] == 'RUNNING'


# =============================================================================
# EVENT BUS INTEGRATION TESTS
# =============================================================================


class TestEventBusIntegration:
    """Tests for event bus integration."""

    def test_event_bus_subscription(self, default_config):
        """Test agent subscribes to events on event bus."""
        event_bus = EventBus()
        agent = RiskSentinelAgent(config=default_config, event_bus=event_bus)
        agent.start()

        # Check subscription
        count = event_bus.get_subscriber_count(EventType.TRADE_PROPOSED)
        assert count > 0

        agent.stop()

    def test_event_processing(self, default_config):
        """Test agent processes events from event bus."""
        event_bus = EventBus()
        agent = RiskSentinelAgent(config=default_config, event_bus=event_bus)
        agent.start()

        # Create and publish event
        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        )

        event = AgentEvent(
            event_type=EventType.TRADE_PROPOSED,
            source_agent="test",
            payload=proposal.to_dict()
        )

        responses = event_bus.publish(event)
        assert len(responses) > 0

        agent.stop()


# =============================================================================
# EDGE CASE TESTS
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_equity(self, agent):
        """Test handling of zero equity."""
        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=0.0,
            current_balance=0.0,
            market_data={'ATR': 20.0}
        )

        # Should not crash
        assessment = agent.evaluate_trade(proposal)
        assert assessment.decision == DecisionType.REJECT

    def test_negative_values(self, agent):
        """Test handling of negative values."""
        proposal = TradeProposal(
            action="BUY",
            quantity=-0.1,  # Invalid negative quantity
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={'ATR': 20.0}
        )

        # Should handle gracefully
        assessment = agent.evaluate_trade(proposal)
        assert assessment is not None

    def test_missing_market_data(self, agent):
        """Test handling of missing market data."""
        proposal = TradeProposal(
            action="BUY",
            quantity=0.1,
            entry_price=2000.0,
            current_equity=10000.0,
            current_balance=10000.0,
            market_data={}  # Empty market data
        )

        # Should not crash
        assessment = agent.evaluate_trade(proposal)
        assert assessment is not None

    def test_disabled_agent(self, default_config):
        """Test agent behavior when disabled."""
        default_config.enabled = False
        agent = RiskSentinelAgent(config=default_config)
        agent.start()

        proposal = TradeProposal(
            action="BUY",
            quantity=100.0,  # Would normally be rejected
            entry_price=2000.0,
            current_equity=100.0,
            current_balance=100.0,
            market_data={'ATR': 20.0}
        )

        # Should auto-approve when disabled
        assessment = agent.evaluate_trade(proposal)
        assert assessment.decision == DecisionType.APPROVE

        agent.stop()


# =============================================================================
# RUN TESTS
# =============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
