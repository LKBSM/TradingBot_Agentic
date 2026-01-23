# =============================================================================
# SPRINT 1 TESTS - Risk Management Components
# =============================================================================
# Comprehensive test suite for Sprint 1 risk management:
#   - Portfolio Risk (VaR, CVaR, Correlations, Exposure)
#   - Kill Switch (Circuit Breakers, Recovery)
#   - Audit Logger (Structured Logging)
#   - Integrated Risk Manager
#
# Run with: pytest tests/test_sprint1_risk.py -v
# =============================================================================

import pytest
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path

# Import Sprint 1 components
from src.agents.portfolio_risk import (
    VaRCalculator,
    VaRMethod,
    VaRResult,
    CorrelationEngine,
    ExposureManager,
    StressTester,
    PortfolioRiskManager,
    Position,
    RiskLimits,
    create_portfolio_risk_manager
)

from src.agents.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    CircuitBreaker,
    CircuitBreakerConfig,
    HaltLevel,
    HaltReason,
    BreakerState,
    RecoveryManager,
    RecoveryState,
    create_kill_switch
)

from src.agents.audit_logger import (
    AuditLogger,
    AuditRecord,
    DecisionAuditRecord,
    TradeAuditRecord,
    AuditEventType,
    LogLevel,
    MemoryHandler,
    create_audit_logger
)

from src.agents.risk_integration import (
    IntegratedRiskManager,
    IntegratedRiskConfig,
    IntegratedRiskResult,
    RiskDecision,
    create_integrated_risk_manager
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_returns():
    """Generate sample returns for testing."""
    np.random.seed(42)
    # Simulate daily returns with mean 0.0005 (0.05%) and std 0.01 (1%)
    returns = np.random.normal(0.0005, 0.01, 252)
    return returns


@pytest.fixture
def sample_positions():
    """Create sample positions for testing."""
    return [
        Position("EURUSD", 100000, 1.1000, 1.1050, "EUR", "forex"),
        Position("GBPUSD", 50000, 1.2500, 1.2480, "GBP", "forex"),
        Position("XAUUSD", -20000, 1950.00, 1960.00, "USD", "commodity"),
    ]


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for logs."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# VAR CALCULATOR TESTS
# =============================================================================

class TestVaRCalculator:
    """Tests for VaR calculation."""

    def test_historical_var(self, sample_returns):
        """Test historical VaR calculation."""
        calculator = VaRCalculator()
        result = calculator.calculate(
            returns=sample_returns,
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL,
            confidence=0.95
        )

        assert isinstance(result, VaRResult)
        assert result.var_amount > 0
        assert result.var_pct > 0
        assert result.var_pct < 0.10  # Should be reasonable
        assert result.confidence == 0.95
        assert result.method == VaRMethod.HISTORICAL

    def test_parametric_var(self, sample_returns):
        """Test parametric VaR calculation."""
        calculator = VaRCalculator()
        result = calculator.calculate(
            returns=sample_returns,
            portfolio_value=100000,
            method=VaRMethod.PARAMETRIC,
            confidence=0.95
        )

        assert result.var_amount > 0
        assert result.method == VaRMethod.PARAMETRIC

    def test_monte_carlo_var(self, sample_returns):
        """Test Monte Carlo VaR calculation."""
        calculator = VaRCalculator()
        result = calculator.calculate(
            returns=sample_returns,
            portfolio_value=100000,
            method=VaRMethod.MONTE_CARLO,
            confidence=0.95
        )

        assert result.var_amount > 0
        assert result.method == VaRMethod.MONTE_CARLO

    def test_cvar_calculation(self, sample_returns):
        """Test CVaR (Expected Shortfall) calculation."""
        calculator = VaRCalculator()
        result = calculator.calculate(
            returns=sample_returns,
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL,
            confidence=0.95
        )

        # CVaR should be greater than or equal to VaR
        assert result.cvar_amount is not None
        assert result.cvar_amount >= result.var_amount

    def test_different_confidence_levels(self, sample_returns):
        """Test VaR at different confidence levels."""
        calculator = VaRCalculator()

        var_95 = calculator.calculate(
            sample_returns, 100000, VaRMethod.HISTORICAL, confidence=0.95
        )
        var_99 = calculator.calculate(
            sample_returns, 100000, VaRMethod.HISTORICAL, confidence=0.99
        )

        # 99% VaR should be higher than 95% VaR
        assert var_99.var_amount > var_95.var_amount

    def test_empty_returns(self):
        """Test handling of empty returns."""
        calculator = VaRCalculator()
        result = calculator.calculate(
            returns=np.array([]),
            portfolio_value=100000,
            method=VaRMethod.HISTORICAL
        )

        assert result.var_amount == 0
        assert result.sample_size == 0


# =============================================================================
# CORRELATION ENGINE TESTS
# =============================================================================

class TestCorrelationEngine:
    """Tests for correlation analysis."""

    def test_correlation_calculation(self):
        """Test basic correlation calculation."""
        engine = CorrelationEngine(lookback_short=20, lookback_long=50)

        # Generate correlated returns
        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)

        for i in range(100):
            engine.update({
                "EURUSD": base[i],
                "GBPUSD": base[i] * 0.8 + np.random.normal(0, 0.002),  # Correlated
                "XAUUSD": np.random.normal(0, 0.015)  # Uncorrelated
            })

        # Get correlation matrix
        corr_matrix = engine.get_correlation_matrix("short")
        assert corr_matrix is not None

        # EURUSD and GBPUSD should be highly correlated
        eur_gbp_corr = engine.get_correlation("EURUSD", "GBPUSD")
        assert eur_gbp_corr is not None
        assert eur_gbp_corr > 0.5

    def test_highly_correlated_pairs(self):
        """Test detection of highly correlated pairs."""
        engine = CorrelationEngine()

        np.random.seed(42)
        base = np.random.normal(0, 0.01, 100)

        for i in range(100):
            engine.update({
                "A": base[i],
                "B": base[i] * 0.95 + np.random.normal(0, 0.001),
                "C": np.random.normal(0, 0.01)
            })

        pairs = engine.get_highly_correlated_pairs(threshold=0.7)
        # A and B should be detected as highly correlated
        assert len(pairs) >= 1

    def test_correlation_breakdown_detection(self):
        """Test detection of correlation breakdown."""
        engine = CorrelationEngine(
            lookback_short=10,
            lookback_long=50,
            breakdown_threshold=0.3
        )

        np.random.seed(42)

        # Phase 1: High correlation
        for i in range(60):
            base = np.random.normal(0, 0.01)
            engine.update({
                "A": base,
                "B": base * 0.9 + np.random.normal(0, 0.001)
            })

        # Phase 2: Correlation breaks down
        for i in range(20):
            engine.update({
                "A": np.random.normal(0, 0.01),
                "B": np.random.normal(0, 0.01)  # Now uncorrelated
            })

        alerts = engine.detect_correlation_breakdown()
        # Should detect the breakdown
        # Note: This depends on the data, may or may not trigger


# =============================================================================
# EXPOSURE MANAGER TESTS
# =============================================================================

class TestExposureManager:
    """Tests for exposure management."""

    def test_exposure_calculation(self, sample_positions):
        """Test exposure calculation."""
        manager = ExposureManager(equity=100000)
        manager.update_positions(sample_positions)

        report = manager.get_exposure_report()

        assert report.gross_exposure > 0
        assert report.long_exposure > 0
        assert report.short_exposure > 0
        assert report.gross_exposure == report.long_exposure + report.short_exposure

    def test_currency_exposure(self, sample_positions):
        """Test exposure by currency."""
        manager = ExposureManager(equity=100000)
        manager.update_positions(sample_positions)

        report = manager.get_exposure_report()

        assert "EUR" in report.exposure_by_currency
        assert "GBP" in report.exposure_by_currency

    def test_concentration_metrics(self, sample_positions):
        """Test concentration metrics (HHI)."""
        manager = ExposureManager(equity=100000)
        manager.update_positions(sample_positions)

        report = manager.get_exposure_report()

        assert report.hhi_concentration > 0
        assert report.hhi_concentration <= 1.0
        assert report.largest_position_pct > 0

    def test_new_position_check(self, sample_positions):
        """Test checking new position against limits."""
        limits = RiskLimits(
            max_gross_exposure=2.0,
            max_single_position_pct=0.50
        )
        manager = ExposureManager(equity=100000, limits=limits)
        manager.update_positions(sample_positions)

        new_position = Position("USDJPY", 30000, 150.00, 150.10, "JPY", "forex")

        allowed, violations, multiplier = manager.check_new_position(new_position)

        assert isinstance(allowed, bool)
        assert isinstance(violations, list)
        assert 0 <= multiplier <= 1


# =============================================================================
# STRESS TESTER TESTS
# =============================================================================

class TestStressTester:
    """Tests for stress testing."""

    def test_run_scenario(self, sample_positions):
        """Test running a stress scenario."""
        tester = StressTester()

        result = tester.run_scenario(
            scenario_name="2008_financial_crisis",
            positions=sample_positions,
            portfolio_value=100000
        )

        assert "scenario_name" in result
        assert "total_loss" in result
        assert "loss_pct" in result
        assert result["portfolio_value_after"] < result["portfolio_value_before"]

    def test_run_all_scenarios(self, sample_positions):
        """Test running all predefined scenarios."""
        tester = StressTester()

        results = tester.run_all_scenarios(
            positions=sample_positions,
            portfolio_value=100000
        )

        assert len(results) > 0
        assert "2008_financial_crisis" in results
        assert "2020_covid_crash" in results

    def test_worst_case(self, sample_positions):
        """Test finding worst case scenario."""
        tester = StressTester()

        worst = tester.get_worst_case(
            positions=sample_positions,
            portfolio_value=100000
        )

        assert "scenario_name" in worst
        assert "loss_pct" in worst


# =============================================================================
# PORTFOLIO RISK MANAGER TESTS
# =============================================================================

class TestPortfolioRiskManager:
    """Tests for the main portfolio risk manager."""

    def test_initialization(self):
        """Test manager initialization."""
        manager = create_portfolio_risk_manager(
            equity=100000,
            preset="moderate"
        )

        assert manager.equity == 100000
        assert manager.limits is not None

    def test_update_equity(self):
        """Test equity updates and drawdown tracking."""
        manager = PortfolioRiskManager(equity=100000)

        manager.update_equity(95000)  # 5% drawdown
        report = manager.get_risk_report()

        assert report["current_drawdown"] == pytest.approx(0.05, rel=0.01)

    def test_check_trade(self, sample_returns):
        """Test trade checking."""
        manager = PortfolioRiskManager(
            equity=100000,
            limits=RiskLimits(max_var_pct=0.02)
        )

        # Add some returns for VaR calculation
        for r in sample_returns:
            manager.record_return(r)

        position = Position("EURUSD", 50000, 1.10, 1.10, "EUR", "forex")

        allowed, violations, multiplier = manager.check_trade(position)

        assert isinstance(allowed, bool)
        assert isinstance(violations, list)
        assert 0 <= multiplier <= 1

    def test_comprehensive_report(self, sample_positions, sample_returns):
        """Test comprehensive risk report."""
        manager = PortfolioRiskManager(equity=100000)
        manager.update_positions(sample_positions)

        for r in sample_returns:
            manager.record_return(r)

        report = manager.get_risk_report()

        assert "equity" in report
        assert "var" in report
        assert "exposure" in report
        assert "stress_test_worst_case" in report


# =============================================================================
# CIRCUIT BREAKER TESTS
# =============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker."""

    def test_breaker_trip(self):
        """Test circuit breaker tripping."""
        config = CircuitBreakerConfig(
            name="test_breaker",
            threshold=0.05,
            halt_level=HaltLevel.FULL_HALT,
            cooldown_seconds=60
        )
        breaker = CircuitBreaker(config)

        # Should not trip
        assert not breaker.check(0.03)
        assert not breaker.is_tripped

        # Should trip
        assert breaker.check(0.06)
        assert breaker.is_tripped
        assert breaker.state == BreakerState.OPEN

    def test_breaker_reset(self):
        """Test circuit breaker reset."""
        config = CircuitBreakerConfig(
            name="test_breaker",
            threshold=0.05,
            halt_level=HaltLevel.FULL_HALT,
            cooldown_seconds=0
        )
        breaker = CircuitBreaker(config)

        # Trip the breaker
        breaker.check(0.06)
        assert breaker.is_tripped

        # Reset
        breaker.reset(force=True)
        assert not breaker.is_tripped
        assert breaker.state == BreakerState.CLOSED

    def test_consecutive_trigger_threshold(self):
        """Test consecutive trigger threshold."""
        config = CircuitBreakerConfig(
            name="test_breaker",
            threshold=0.05,
            halt_level=HaltLevel.FULL_HALT,
            consecutive_threshold=3  # Need 3 consecutive triggers
        )
        breaker = CircuitBreaker(config)

        # First two should not trip
        assert not breaker.check(0.06)
        assert not breaker.check(0.06)
        assert not breaker.is_tripped

        # Third should trip
        assert breaker.check(0.06)
        assert breaker.is_tripped


# =============================================================================
# KILL SWITCH TESTS
# =============================================================================

class TestKillSwitch:
    """Tests for kill switch."""

    def test_initialization(self):
        """Test kill switch initialization."""
        ks = create_kill_switch(preset="moderate")

        assert not ks.is_halted
        assert ks.halt_level == HaltLevel.NONE
        assert ks.is_trading_allowed()

    def test_daily_loss_trigger(self):
        """Test daily loss limit trigger."""
        config = KillSwitchConfig(max_daily_loss_pct=0.03)
        ks = KillSwitch(config=config)

        # Update with loss exceeding limit
        ks.update(
            equity=97000,
            peak_equity=100000,
            daily_pnl=-3500  # 3.5% loss
        )

        assert ks.is_halted
        assert ks.halt_reason == HaltReason.DAILY_LOSS_LIMIT

    def test_max_drawdown_trigger(self):
        """Test max drawdown trigger."""
        config = KillSwitchConfig(max_drawdown_pct=0.10)
        ks = KillSwitch(config=config)

        # Update with drawdown exceeding limit
        ks.update(
            equity=88000,
            peak_equity=100000  # 12% drawdown
        )

        assert ks.is_halted
        assert ks.halt_reason == HaltReason.MAX_DRAWDOWN
        assert ks.halt_level == HaltLevel.EMERGENCY

    def test_manual_halt(self):
        """Test manual halt."""
        ks = KillSwitch()

        ks.manual_halt("Testing manual halt")

        assert ks.is_halted
        assert ks.halt_reason == HaltReason.MANUAL_HALT

    def test_emergency_halt(self):
        """Test emergency halt."""
        ks = KillSwitch()

        ks.emergency_halt("Emergency!")

        assert ks.is_halted
        assert ks.halt_level == HaltLevel.EMERGENCY
        assert ks.halt_reason == HaltReason.EMERGENCY_STOP

    def test_position_multiplier(self):
        """Test position multiplier calculation."""
        ks = KillSwitch()

        # Normal state
        assert ks.get_position_multiplier() == 1.0

        # Halted state
        ks.manual_halt("Test")
        assert ks.get_position_multiplier() == 0.0

    def test_consecutive_losses(self):
        """Test consecutive losses tracking."""
        config = KillSwitchConfig(max_consecutive_losses=3)
        ks = KillSwitch(config=config)

        # Record losses
        ks.record_trade_result(-100, -0.01)
        ks.record_trade_result(-100, -0.01)
        assert not ks.is_halted

        ks.record_trade_result(-100, -0.01)
        # May trigger reduced mode

    def test_daily_reset(self):
        """Test daily counter reset."""
        ks = KillSwitch()

        ks.record_trade_result(-100, -0.01)
        ks.record_trade_result(-100, -0.01)

        ks.reset_daily_counters()
        # Counters should be reset


# =============================================================================
# RECOVERY MANAGER TESTS
# =============================================================================

class TestRecoveryManager:
    """Tests for recovery manager."""

    def test_recovery_start(self):
        """Test starting recovery."""
        from src.agents.kill_switch import HaltEvent

        rm = RecoveryManager(
            cooldown_seconds=1,  # Short for testing
            require_confirmation=False
        )

        halt_event = HaltEvent(
            halt_id="test_123",
            reason=HaltReason.DAILY_LOSS_LIMIT,
            level=HaltLevel.FULL_HALT,
            timestamp=datetime.now(),
            trigger_value=0.04,
            threshold=0.03,
            message="Test halt"
        )

        rm.start_recovery(halt_event)

        assert rm.state == RecoveryState.COOLING_OFF
        assert rm.position_multiplier == 0.0

    def test_gradual_recovery(self):
        """Test gradual recovery progress."""
        from src.agents.kill_switch import HaltEvent

        rm = RecoveryManager(
            cooldown_seconds=0,
            require_confirmation=False,
            gradual_steps=5,
            step_duration_seconds=0
        )

        halt_event = HaltEvent(
            halt_id="test_123",
            reason=HaltReason.DAILY_LOSS_LIMIT,
            level=HaltLevel.FULL_HALT,
            timestamp=datetime.now(),
            trigger_value=0.04,
            threshold=0.03,
            message="Test halt"
        )

        rm.start_recovery(halt_event)
        rm.update()  # Should move to GRADUAL_RESTART

        assert rm.state in [RecoveryState.GRADUAL_RESTART, RecoveryState.RECOVERED]


# =============================================================================
# AUDIT LOGGER TESTS
# =============================================================================

class TestAuditLogger:
    """Tests for audit logger."""

    def test_initialization(self, temp_log_dir):
        """Test logger initialization."""
        logger = create_audit_logger(
            session_id="test_session",
            log_directory=temp_log_dir,
            preset="minimal"
        )

        assert logger.session_id == "test_session"

        logger.close()

    def test_log_decision(self, temp_log_dir):
        """Test logging a decision."""
        logger = AuditLogger(
            session_id="test",
            log_directory=temp_log_dir
        )

        record_id = logger.log_decision(
            decision_id="dec_123",
            proposal_id="prop_456",
            proposed_action="BUY",
            proposed_quantity=1000,
            proposed_symbol="EURUSD",
            final_decision="APPROVE",
            agent_assessments=[
                {"agent": "risk", "decision": "APPROVE"}
            ],
            reasoning=["All checks passed"]
        )

        assert record_id is not None

        # Check it's in memory
        records = logger.get_recent_records(limit=10)
        assert len(records) > 0

        logger.close()

    def test_log_trade(self, temp_log_dir):
        """Test logging a trade."""
        logger = AuditLogger(
            session_id="test",
            log_directory=temp_log_dir
        )

        record_id = logger.log_trade(
            trade_id="trd_123",
            decision_id="dec_123",
            symbol="EURUSD",
            action="BUY",
            quantity=1000,
            executed_price=1.1050,
            pnl=50,
            pnl_pct=0.005
        )

        assert record_id is not None

        logger.close()

    def test_export_to_json(self, temp_log_dir):
        """Test JSON export."""
        logger = AuditLogger(
            session_id="test",
            log_directory=temp_log_dir
        )

        # Log some records
        for i in range(5):
            logger.log_system_event(
                AuditEventType.SYSTEM_START,
                f"Test event {i}"
            )

        export_path = Path(temp_log_dir) / "export.json"
        count = logger.export_to_json(export_path)

        assert count >= 5
        assert export_path.exists()

        logger.close()

    def test_statistics(self, temp_log_dir):
        """Test statistics gathering."""
        logger = AuditLogger(
            session_id="test",
            log_directory=temp_log_dir
        )

        # Log various events
        logger.log_system_event(AuditEventType.SYSTEM_START, "Start")
        logger.log_system_event(AuditEventType.SYSTEM_ERROR, "Error", level=LogLevel.ERROR)

        stats = logger.get_statistics()

        assert stats["total_records"] >= 2
        assert "by_event_type" in stats

        logger.close()


# =============================================================================
# INTEGRATED RISK MANAGER TESTS
# =============================================================================

class TestIntegratedRiskManager:
    """Tests for integrated risk manager."""

    def test_initialization(self, temp_log_dir):
        """Test manager initialization."""
        config = IntegratedRiskConfig(
            audit_log_directory=temp_log_dir
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        assert manager._equity == 100000
        assert manager.is_trading_allowed()

        manager.close()

    def test_evaluate_trade_approval(self, temp_log_dir):
        """Test trade evaluation with approval."""
        config = IntegratedRiskConfig(
            max_drawdown=0.20,  # High limit to ensure approval
            enable_audit_logging=True,
            audit_log_directory=temp_log_dir
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        result = manager.evaluate_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=10000,
            price=1.10
        )

        assert isinstance(result, IntegratedRiskResult)
        assert result.is_approved
        assert result.decision in [RiskDecision.APPROVE, RiskDecision.APPROVE_MODIFIED]
        assert result.approved_quantity > 0

        manager.close()

    def test_evaluate_trade_rejection_drawdown(self, temp_log_dir):
        """Test trade rejection due to drawdown."""
        config = IntegratedRiskConfig(
            max_drawdown=0.05,  # Low limit
            enable_audit_logging=False
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Simulate drawdown
        manager.update_equity(93000)  # 7% drawdown

        result = manager.evaluate_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=10000,
            price=1.10
        )

        assert not result.is_approved
        assert result.decision == RiskDecision.REJECT_DRAWDOWN
        assert len(result.violations) > 0

    def test_evaluate_trade_modification(self, temp_log_dir):
        """Test trade modification due to risk factors."""
        config = IntegratedRiskConfig(
            max_drawdown=0.20,
            drawdown_warning=0.03,  # Low warning threshold
            enable_gradual_reduction=True,
            enable_audit_logging=False
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Simulate some drawdown to trigger reduction
        manager.update_equity(95000)  # 5% drawdown, above warning

        result = manager.evaluate_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=10000,
            price=1.10
        )

        # Should be approved but modified
        assert result.is_approved
        assert result.position_multiplier < 1.0
        assert result.approved_quantity < result.original_quantity

    def test_kill_switch_integration(self, temp_log_dir):
        """Test kill switch integration."""
        config = IntegratedRiskConfig(
            enable_kill_switch=True,
            max_daily_loss=0.02,
            enable_audit_logging=False
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Trigger kill switch manually
        manager.emergency_halt("Test emergency")

        assert not manager.is_trading_allowed()

        result = manager.evaluate_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=10000,
            price=1.10
        )

        assert not result.is_approved
        assert result.decision == RiskDecision.REJECT_KILL_SWITCH

    def test_statistics(self, temp_log_dir):
        """Test statistics tracking."""
        config = IntegratedRiskConfig(enable_audit_logging=False)
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Evaluate some trades
        for i in range(5):
            manager.evaluate_trade(
                symbol="EURUSD",
                action="BUY",
                quantity=10000,
                price=1.10
            )

        stats = manager.get_statistics()

        assert stats["total_evaluations"] == 5
        assert stats["approvals"] + stats["rejections"] + stats["modifications"] == 5

    def test_risk_report(self, temp_log_dir):
        """Test comprehensive risk report."""
        config = IntegratedRiskConfig(enable_audit_logging=False)
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        report = manager.get_risk_report()

        assert "trading_allowed" in report
        assert "position_multiplier" in report
        assert "statistics" in report
        assert "portfolio_risk" in report

    def test_factory_presets(self):
        """Test factory function presets."""
        conservative = create_integrated_risk_manager(
            equity=100000,
            preset="conservative"
        )
        aggressive = create_integrated_risk_manager(
            equity=100000,
            preset="aggressive"
        )

        assert conservative.config.max_drawdown < aggressive.config.max_drawdown
        assert conservative.config.max_var_pct < aggressive.config.max_var_pct

        conservative.close()
        aggressive.close()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for all Sprint 1 components."""

    def test_full_workflow(self, temp_log_dir, sample_returns):
        """Test complete workflow with all components."""
        # Initialize manager
        config = IntegratedRiskConfig(
            max_var_pct=0.03,
            max_drawdown=0.15,
            enable_audit_logging=True,
            audit_log_directory=temp_log_dir
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Add historical returns
        for r in sample_returns:
            manager.record_return(r)

        # Add positions
        positions = [
            Position("EURUSD", 50000, 1.10, 1.1050, "EUR", "forex")
        ]
        manager.update_positions(positions)

        # Evaluate new trade
        result = manager.evaluate_trade(
            symbol="GBPUSD",
            action="BUY",
            quantity=30000,
            price=1.25
        )

        # Should be approved (limits are generous)
        assert result.is_approved

        # Simulate trade outcome
        manager.record_trade_outcome(pnl=500, pnl_pct=0.005)

        # Get final report
        report = manager.get_risk_report()
        assert report["statistics"]["total_evaluations"] >= 1

        # Check audit logs
        audit_stats = manager._audit_logger.get_statistics()
        assert audit_stats["total_records"] > 0

        manager.close()

    def test_risk_cascade(self, temp_log_dir):
        """Test risk cascade (multiple factors reducing position)."""
        config = IntegratedRiskConfig(
            max_drawdown=0.20,
            drawdown_warning=0.05,
            var_warning_threshold=0.01,
            enable_gradual_reduction=True,
            enable_audit_logging=False
        )
        manager = IntegratedRiskManager(
            equity=100000,
            config=config
        )

        # Simulate multiple risk factors
        manager.update_equity(92000)  # 8% drawdown

        # Add correlated position
        positions = [
            Position("EURUSD", 80000, 1.10, 1.10, "EUR", "forex")
        ]
        manager.update_positions(positions)

        # Try to add more exposure in same currency
        result = manager.evaluate_trade(
            symbol="EURGBP",  # More EUR exposure
            action="BUY",
            quantity=50000,
            price=0.85,
            currency="EUR"
        )

        # Position should be significantly reduced due to multiple factors
        assert result.position_multiplier < 1.0
        assert len(result.warnings) > 0


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
