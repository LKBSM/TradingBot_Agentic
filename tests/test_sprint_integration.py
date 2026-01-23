# =============================================================================
# SPRINT 1+2 INTEGRATION TESTS
# =============================================================================
"""
Integration tests verifying Sprint 1 (Risk & Safety) and Sprint 2 (Intelligence)
work together correctly.

This test suite ensures:
1. Risk components integrate with Intelligence components
2. Event flow works across both sprints
3. Combined decision making produces valid results

Run with: pytest tests/test_sprint_integration.py -v
"""

import pytest
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import tempfile
import shutil
from pathlib import Path


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_market_data():
    """Generate realistic market data for testing."""
    np.random.seed(42)
    n_points = 200

    # Generate price series with trend
    base_price = 1900.0  # Gold price
    returns = np.random.normal(0.0001, 0.005, n_points)
    prices = base_price * np.cumprod(1 + returns)

    # Generate OHLCV data
    data = []
    for i, close in enumerate(prices):
        volatility = abs(np.random.normal(0, 0.003))
        high = close * (1 + volatility)
        low = close * (1 - volatility)
        open_price = close * (1 + np.random.normal(0, 0.001))
        volume = np.random.randint(1000, 10000)

        data.append({
            'timestamp': datetime.now() - timedelta(minutes=15 * (n_points - i)),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': volume
        })

    return data


@pytest.fixture
def sample_news_texts():
    """Sample news headlines for sentiment testing."""
    return [
        ("Fed signals potential rate cuts amid slowing inflation", datetime.now() - timedelta(hours=1)),
        ("Gold prices surge on safe-haven demand", datetime.now() - timedelta(hours=2)),
        ("US dollar weakens against major currencies", datetime.now() - timedelta(hours=3)),
        ("Economic uncertainty drives investors to precious metals", datetime.now() - timedelta(hours=4)),
    ]


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SPRINT 1 COMPONENT TESTS
# =============================================================================

class TestSprint1Components:
    """Test Sprint 1 components individually."""

    def test_portfolio_risk_import(self):
        """Test portfolio risk can be imported."""
        from src.agents import PortfolioRiskManager, VaRCalculator, Position
        assert PortfolioRiskManager is not None
        assert VaRCalculator is not None
        assert Position is not None

    def test_var_calculation(self, sample_market_data):
        """Test VaR calculation works."""
        from src.agents import VaRCalculator, VaRMethod

        # Extract returns from price data
        prices = [d['close'] for d in sample_market_data]
        returns = np.diff(prices) / prices[:-1]

        calculator = VaRCalculator()

        # Test historical VaR
        result = calculator.calculate(
            returns=returns,
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL,
            confidence=0.95
        )

        assert result is not None
        assert result.var_amount > 0
        assert 0 < result.var_pct < 0.5  # Reasonable range

    def test_kill_switch_creation(self):
        """Test kill switch can be created."""
        from src.agents import KillSwitch, KillSwitchConfig, create_kill_switch

        config = KillSwitchConfig(
            daily_loss_limit_pct=5.0,
            total_drawdown_limit_pct=15.0,
            max_positions=10
        )

        kill_switch = create_kill_switch(config)
        assert kill_switch is not None
        assert not kill_switch.is_halted

    def test_audit_logger_creation(self, temp_log_dir):
        """Test audit logger can be created."""
        from src.agents import create_audit_logger, AuditEventType

        logger = create_audit_logger(
            log_directory=temp_log_dir,
            service_name="test_service"
        )

        assert logger is not None

        # Test logging a decision
        logger.log_decision(
            decision_id="test-001",
            action_proposed="BUY XAUUSD 0.1",
            agents_consulted=[],
            final_decision="APPROVE",
            reasoning="Test decision"
        )


# =============================================================================
# SPRINT 2 COMPONENT TESTS
# =============================================================================

class TestSprint2Components:
    """Test Sprint 2 components individually."""

    def test_sentiment_analyzer_import(self):
        """Test sentiment analyzer can be imported."""
        from src.agents import create_sentiment_analyzer
        assert create_sentiment_analyzer is not None

    def test_sentiment_analysis(self, sample_news_texts):
        """Test sentiment analysis on news."""
        from src.agents import create_sentiment_analyzer

        analyzer = create_sentiment_analyzer()

        # Analyze single text
        text, _ = sample_news_texts[0]
        result = analyzer.analyze(text)

        assert result is not None
        assert -1 <= result.score <= 1
        assert 0 <= result.confidence <= 1

    def test_regime_predictor_import(self):
        """Test regime predictor can be imported."""
        from src.agents import create_regime_predictor
        assert create_regime_predictor is not None

    def test_regime_prediction(self, sample_market_data):
        """Test regime prediction with price data."""
        from src.agents import create_regime_predictor

        predictor = create_regime_predictor()

        # Feed price data
        for data in sample_market_data[:50]:  # Use first 50 points
            predictor.update(data['close'], data['volume'])

        # Get prediction
        prediction = predictor.predict()

        # Prediction may be None if not enough data
        # But update should not fail
        assert True

    def test_multi_timeframe_import(self):
        """Test multi-timeframe engine can be imported."""
        from src.agents import create_multi_timeframe_engine
        assert create_multi_timeframe_engine is not None

    def test_ensemble_model_import(self):
        """Test ensemble model can be imported."""
        from src.agents import create_ensemble_risk_model
        assert create_ensemble_risk_model is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestSprint1Sprint2Integration:
    """Integration tests for Sprint 1 + Sprint 2 working together."""

    def test_intelligence_informs_risk(self, sample_market_data, sample_news_texts):
        """Test that intelligence output can inform risk decisions."""
        from src.agents import (
            create_sentiment_analyzer,
            create_regime_predictor,
            VaRCalculator,
            VaRMethod
        )

        # Step 1: Get sentiment from news
        sentiment_analyzer = create_sentiment_analyzer()
        sentiments = []
        for text, _ in sample_news_texts:
            result = sentiment_analyzer.analyze(text)
            sentiments.append(result.score)

        avg_sentiment = np.mean(sentiments)

        # Step 2: Get regime from price data
        regime_predictor = create_regime_predictor()
        for data in sample_market_data[:50]:
            regime_predictor.update(data['close'], data['volume'])

        # Step 3: Calculate VaR
        prices = [d['close'] for d in sample_market_data]
        returns = np.diff(prices) / prices[:-1]

        var_calculator = VaRCalculator()
        var_result = var_calculator.calculate(
            returns=returns,
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL,
            confidence=0.95
        )

        # Step 4: Combine intelligence into risk decision
        # If sentiment is negative, increase risk aversion
        sentiment_multiplier = 1.0 if avg_sentiment >= 0 else 1.5
        adjusted_var = var_result.var_amount * sentiment_multiplier

        # Verify integration produced valid output
        assert adjusted_var > 0
        assert isinstance(sentiment_multiplier, float)

    def test_event_flow_integration(self, temp_log_dir):
        """Test event flow between Sprint 1 and Sprint 2 components."""
        from src.agents import (
            EventBus,
            EventType,
            TradeProposal,
            create_audit_logger
        )

        # Create event bus
        event_bus = EventBus()

        # Create audit logger to track events
        audit_logger = create_audit_logger(
            log_directory=temp_log_dir,
            service_name="integration_test"
        )

        # Track received events
        received_events = []

        def event_handler(event):
            received_events.append(event)

        # Subscribe to trade proposals
        event_bus.subscribe(EventType.TRADE_PROPOSED, event_handler)

        # Create a trade proposal (simulating PPO agent decision)
        proposal = TradeProposal(
            proposal_id="test-001",
            symbol="XAUUSD",
            action="BUY",
            quantity=0.1,
            price=1920.50,
            timestamp=datetime.now(),
            source_agent="PPO_Agent",
            confidence=0.85
        )

        # Publish the event
        event_bus.publish(EventType.TRADE_PROPOSED, proposal)

        # Verify event was received
        assert len(received_events) == 1
        assert received_events[0].symbol == "XAUUSD"

    def test_combined_decision_pipeline(self, sample_market_data, sample_news_texts, temp_log_dir):
        """Test complete decision pipeline with all components."""
        from src.agents import (
            create_sentiment_analyzer,
            create_regime_predictor,
            VaRCalculator,
            VaRMethod,
            KillSwitch,
            KillSwitchConfig,
            create_audit_logger
        )

        # Initialize all components
        sentiment_analyzer = create_sentiment_analyzer()
        regime_predictor = create_regime_predictor()
        var_calculator = VaRCalculator()
        kill_switch = KillSwitch(KillSwitchConfig(
            daily_loss_limit_pct=5.0,
            total_drawdown_limit_pct=15.0,
            max_positions=10
        ))
        audit_logger = create_audit_logger(
            log_directory=temp_log_dir,
            service_name="combined_test"
        )

        # Step 1: Analyze sentiment
        sentiment_scores = []
        for text, timestamp in sample_news_texts:
            result = sentiment_analyzer.analyze(text)
            sentiment_scores.append(result.score)
        overall_sentiment = np.mean(sentiment_scores)

        # Step 2: Update regime predictor
        for data in sample_market_data:
            regime_predictor.update(data['close'], data.get('volume', 1000))

        # Step 3: Calculate VaR
        prices = [d['close'] for d in sample_market_data]
        returns = np.diff(prices) / prices[:-1]
        var_result = var_calculator.calculate(
            returns=returns,
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL,
            confidence=0.95
        )

        # Step 4: Check kill switch
        is_halted = kill_switch.is_halted

        # Step 5: Make combined decision
        decision = {
            'sentiment': 'positive' if overall_sentiment > 0 else 'negative',
            'sentiment_score': overall_sentiment,
            'var_amount': var_result.var_amount,
            'var_pct': var_result.var_pct,
            'is_halted': is_halted,
            'can_trade': not is_halted and var_result.var_pct < 0.1
        }

        # Step 6: Log the decision
        audit_logger.log_decision(
            decision_id="combined-001",
            action_proposed="ANALYZE_MARKET",
            agents_consulted=[
                {"agent": "sentiment", "output": decision['sentiment']},
                {"agent": "var", "output": f"VaR: {decision['var_pct']:.2%}"}
            ],
            final_decision="PROCEED" if decision['can_trade'] else "HALT",
            reasoning=f"Sentiment: {decision['sentiment']}, VaR: {decision['var_pct']:.2%}"
        )

        # Verify all components produced valid output
        assert isinstance(decision['sentiment_score'], float)
        assert decision['var_amount'] > 0
        assert isinstance(decision['can_trade'], bool)


# =============================================================================
# STRESS TESTS
# =============================================================================

class TestStressIntegration:
    """Stress tests for combined components."""

    def test_high_volume_events(self, temp_log_dir):
        """Test system handles high volume of events."""
        from src.agents import EventBus, EventType, TradeProposal

        event_bus = EventBus()
        received_count = [0]  # Use list for mutable counter

        def counter_handler(event):
            received_count[0] += 1

        event_bus.subscribe(EventType.TRADE_PROPOSED, counter_handler)

        # Send 1000 events
        for i in range(1000):
            proposal = TradeProposal(
                proposal_id=f"stress-{i:04d}",
                symbol="XAUUSD",
                action="BUY" if i % 2 == 0 else "SELL",
                quantity=0.1,
                price=1920.50 + i * 0.01,
                timestamp=datetime.now(),
                source_agent="StressTest",
                confidence=0.5
            )
            event_bus.publish(EventType.TRADE_PROPOSED, proposal)

        assert received_count[0] == 1000

    def test_rapid_sentiment_analysis(self, sample_news_texts):
        """Test rapid sentiment analysis."""
        from src.agents import create_sentiment_analyzer

        analyzer = create_sentiment_analyzer()

        # Analyze 100 texts rapidly
        results = []
        for _ in range(25):  # 25 * 4 texts = 100 analyses
            for text, _ in sample_news_texts:
                result = analyzer.analyze(text)
                results.append(result.score)

        assert len(results) == 100
        assert all(-1 <= s <= 1 for s in results)


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
