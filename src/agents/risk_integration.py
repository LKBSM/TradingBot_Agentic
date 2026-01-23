# =============================================================================
# RISK INTEGRATION - Sprint 1 Complete Integration Module
# =============================================================================
# This module integrates all Sprint 1 risk management components:
#
#   1. PortfolioRiskManager - VaR, CVaR, Correlations, Exposure
#   2. KillSwitch - Emergency halt system with circuit breakers
#   3. AuditLogger - Complete audit trail
#   4. IntelligentRiskSentinel - Enhanced with portfolio risk
#
# The IntegratedRiskManager provides a unified interface that:
#   - Combines all risk checks into a single evaluation
#   - Manages position sizing with all factors
#   - Provides comprehensive reporting
#   - Maintains full audit trail
#
# Architecture:
#   ┌─────────────────────────────────────────────────────────────────┐
#   │                  INTEGRATED RISK MANAGER                        │
#   │                                                                 │
#   │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
#   │  │ Portfolio   │ │    Kill     │ │   Audit     │ │   Risk    │ │
#   │  │ Risk Mgr    │ │   Switch    │ │   Logger    │ │ Sentinel  │ │
#   │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └─────┬─────┘ │
#   │         │               │               │              │       │
#   │         └───────────────┴───────────────┴──────────────┘       │
#   │                         │                                       │
#   │                         ▼                                       │
#   │              ┌─────────────────────┐                           │
#   │              │  Unified Risk API   │                           │
#   │              └─────────────────────┘                           │
#   └─────────────────────────────────────────────────────────────────┘
#
# =============================================================================

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Security hardening
from src.agents.security_hardening import InputValidator, ValidationError

# Import Sprint 1 components
from src.agents.portfolio_risk import (
    PortfolioRiskManager,
    Position,
    RiskLimits,
    VaRMethod,
    VaRResult,
    ExposureReport,
    CorrelationEngine,
    create_portfolio_risk_manager
)

from src.agents.kill_switch import (
    KillSwitch,
    KillSwitchConfig,
    HaltLevel,
    HaltReason,
    create_kill_switch
)

from src.agents.audit_logger import (
    AuditLogger,
    AuditEventType,
    LogLevel,
    DecisionAuditRecord,
    create_audit_logger,
    get_audit_logger,
    set_audit_logger
)

# Import existing components
from src.agents.events import (
    TradeProposal,
    RiskAssessment,
    DecisionType,
    RiskLevel,
    RiskViolation
)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class RiskDecision(Enum):
    """Integrated risk decision types."""
    APPROVE = "approve"
    APPROVE_MODIFIED = "approve_modified"
    REJECT_RISK = "reject_risk"
    REJECT_EXPOSURE = "reject_exposure"
    REJECT_VAR = "reject_var"
    REJECT_CORRELATION = "reject_correlation"
    REJECT_KILL_SWITCH = "reject_kill_switch"
    REJECT_DRAWDOWN = "reject_drawdown"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class IntegratedRiskResult:
    """
    Complete risk evaluation result from all components.
    """
    # Decision
    decision: RiskDecision
    is_approved: bool
    decision_id: str = field(default_factory=lambda: f"risk_{uuid.uuid4().hex[:12]}")

    # Position sizing
    original_quantity: float = 0.0
    approved_quantity: float = 0.0
    position_multiplier: float = 1.0

    # Component results
    var_check_passed: bool = True
    exposure_check_passed: bool = True
    correlation_check_passed: bool = True
    kill_switch_check_passed: bool = True
    drawdown_check_passed: bool = True

    # Risk metrics at decision time
    portfolio_var_pct: float = 0.0
    portfolio_cvar_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    gross_exposure_pct: float = 0.0
    net_exposure_pct: float = 0.0

    # Multipliers breakdown
    var_multiplier: float = 1.0
    exposure_multiplier: float = 1.0
    correlation_multiplier: float = 1.0
    drawdown_multiplier: float = 1.0
    kill_switch_multiplier: float = 1.0

    # Violations and warnings
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Reasoning
    reasoning: List[str] = field(default_factory=list)

    # Performance
    evaluation_time_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision": self.decision.value,
            "is_approved": self.is_approved,
            "decision_id": self.decision_id,
            "original_quantity": self.original_quantity,
            "approved_quantity": self.approved_quantity,
            "position_multiplier": round(self.position_multiplier, 4),
            "checks": {
                "var": self.var_check_passed,
                "exposure": self.exposure_check_passed,
                "correlation": self.correlation_check_passed,
                "kill_switch": self.kill_switch_check_passed,
                "drawdown": self.drawdown_check_passed
            },
            "metrics": {
                "var_pct": round(self.portfolio_var_pct * 100, 2),
                "cvar_pct": round(self.portfolio_cvar_pct * 100, 2),
                "drawdown_pct": round(self.current_drawdown_pct * 100, 2),
                "gross_exposure_pct": round(self.gross_exposure_pct * 100, 2),
                "net_exposure_pct": round(self.net_exposure_pct * 100, 2)
            },
            "multipliers": {
                "var": round(self.var_multiplier, 2),
                "exposure": round(self.exposure_multiplier, 2),
                "correlation": round(self.correlation_multiplier, 2),
                "drawdown": round(self.drawdown_multiplier, 2),
                "kill_switch": round(self.kill_switch_multiplier, 2),
                "final": round(self.position_multiplier, 2)
            },
            "violations": self.violations,
            "warnings": self.warnings,
            "reasoning": self.reasoning,
            "evaluation_time_ms": round(self.evaluation_time_ms, 2),
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class IntegratedRiskConfig:
    """
    Configuration for the integrated risk manager.
    """
    # VaR limits
    max_var_pct: float = 0.02               # 2% daily VaR
    max_cvar_pct: float = 0.03              # 3% daily CVaR
    var_warning_threshold: float = 0.015    # 1.5% warning

    # Exposure limits
    max_gross_exposure: float = 1.5         # 150%
    max_net_exposure: float = 1.0           # 100%
    max_single_position: float = 0.20       # 20%
    max_currency_concentration: float = 0.50  # 50%

    # Correlation limits
    high_correlation_threshold: float = 0.70
    max_correlated_exposure: float = 0.30

    # Drawdown limits
    max_drawdown: float = 0.10              # 10%
    drawdown_warning: float = 0.07          # 7%
    max_daily_loss: float = 0.03            # 3%
    max_weekly_loss: float = 0.05           # 5%

    # Kill switch
    enable_kill_switch: bool = True
    kill_switch_preset: str = "moderate"

    # Position sizing
    min_position_multiplier: float = 0.1    # Minimum 10%
    enable_gradual_reduction: bool = True

    # Audit
    enable_audit_logging: bool = True
    audit_log_directory: str = "./logs/audit"

    def to_risk_limits(self) -> RiskLimits:
        """Convert to RiskLimits for PortfolioRiskManager."""
        return RiskLimits(
            max_var_pct=self.max_var_pct,
            max_cvar_pct=self.max_cvar_pct,
            max_gross_exposure=self.max_gross_exposure,
            max_net_exposure=self.max_net_exposure,
            max_single_position_pct=self.max_single_position,
            max_currency_exposure=self.max_currency_concentration,
            correlation_threshold=self.high_correlation_threshold,
            max_correlated_exposure=self.max_correlated_exposure,
            max_drawdown=self.max_drawdown,
            max_daily_loss=self.max_daily_loss,
            max_weekly_loss=self.max_weekly_loss
        )

    def to_kill_switch_config(self) -> KillSwitchConfig:
        """Convert to KillSwitchConfig."""
        return KillSwitchConfig(
            max_daily_loss_pct=self.max_daily_loss,
            max_weekly_loss_pct=self.max_weekly_loss,
            max_drawdown_pct=self.max_drawdown,
            max_var_pct=self.max_var_pct,
            max_gross_exposure_pct=self.max_gross_exposure
        )


# =============================================================================
# INTEGRATED RISK MANAGER
# =============================================================================

class IntegratedRiskManager:
    """
    Unified risk management combining all Sprint 1 components.

    This class provides a single interface for:
    - VaR/CVaR-based risk checks
    - Correlation-adjusted position sizing
    - Exposure management
    - Kill switch protection
    - Complete audit trail

    Example:
        manager = IntegratedRiskManager(
            equity=100000,
            config=IntegratedRiskConfig(max_drawdown=0.10)
        )

        # Evaluate a trade
        result = manager.evaluate_trade(
            symbol="EURUSD",
            action="BUY",
            quantity=50000,
            price=1.10
        )

        if result.is_approved:
            # Execute with adjusted quantity
            execute_trade(quantity=result.approved_quantity)
        else:
            print(f"Trade rejected: {result.violations}")

        # Record outcome for learning
        manager.record_trade_outcome(pnl=250, pnl_pct=0.0025)
    """

    def __init__(
        self,
        equity: float,
        config: Optional[IntegratedRiskConfig] = None,
        session_id: Optional[str] = None
    ):
        """
        Initialize integrated risk manager.

        Args:
            equity: Initial account equity
            config: Risk configuration
            session_id: Session ID for audit logging
        """
        self.config = config or IntegratedRiskConfig()
        self.session_id = session_id or f"session_{uuid.uuid4().hex[:8]}"

        # Initialize components
        self._portfolio_risk = PortfolioRiskManager(
            equity=equity,
            limits=self.config.to_risk_limits()
        )

        if self.config.enable_kill_switch:
            self._kill_switch = KillSwitch(
                config=self.config.to_kill_switch_config()
            )
        else:
            self._kill_switch = None

        if self.config.enable_audit_logging:
            self._audit_logger = create_audit_logger(
                session_id=self.session_id,
                log_directory=self.config.audit_log_directory
            )
            set_audit_logger(self._audit_logger)
        else:
            self._audit_logger = None

        # State
        self._equity = equity
        self._peak_equity = equity
        self._current_drawdown = 0.0
        self._daily_pnl = 0.0
        self._weekly_pnl = 0.0
        self._positions: List[Position] = []
        self._returns_history: List[float] = []

        # Statistics
        self._total_evaluations = 0
        self._total_approvals = 0
        self._total_rejections = 0
        self._total_modifications = 0

        self._logger = logging.getLogger("risk_integration")
        self._logger.info(f"Integrated Risk Manager initialized. Session: {self.session_id}")

    # =========================================================================
    # STATE UPDATES
    # =========================================================================

    def update_equity(self, equity: float) -> None:
        """
        Update account equity.

        Args:
            equity: New equity value
        """
        pnl = equity - self._equity
        self._equity = equity
        self._daily_pnl += pnl
        self._weekly_pnl += pnl

        if equity > self._peak_equity:
            self._peak_equity = equity

        self._current_drawdown = (
            (self._peak_equity - equity) / self._peak_equity
            if self._peak_equity > 0 else 0
        )

        # Update components
        self._portfolio_risk.update_equity(equity)

        if self._kill_switch:
            self._kill_switch.update(
                equity=equity,
                peak_equity=self._peak_equity,
                daily_pnl=self._daily_pnl,
                weekly_pnl=self._weekly_pnl
            )

    def update_positions(self, positions: List[Position]) -> None:
        """Update current positions."""
        self._positions = positions
        self._portfolio_risk.update_positions(positions)

    def record_return(self, portfolio_return: float) -> None:
        """
        Record a portfolio return for VaR calculation.

        Args:
            portfolio_return: Period return as decimal
        """
        self._returns_history.append(portfolio_return)
        self._portfolio_risk.record_return(portfolio_return)

        # Update correlation engine if we have asset returns
        # (Would need asset-level returns in production)

    def record_trade_outcome(
        self,
        pnl: float,
        pnl_pct: float,
        is_win: Optional[bool] = None
    ) -> None:
        """
        Record a trade outcome.

        Args:
            pnl: Trade P&L in currency
            pnl_pct: Trade P&L as percentage
            is_win: Whether trade was profitable
        """
        if self._kill_switch:
            self._kill_switch.record_trade_result(pnl, pnl_pct)

        if self._audit_logger:
            self._audit_logger.log_trade(
                trade_id=f"trd_{uuid.uuid4().hex[:8]}",
                decision_id="",
                symbol="",
                action="CLOSED",
                quantity=0,
                executed_price=0,
                pnl=pnl,
                pnl_pct=pnl_pct
            )

    # =========================================================================
    # MAIN EVALUATION METHOD
    # =========================================================================

    def evaluate_trade(
        self,
        symbol: str,
        action: str,
        quantity: float,
        price: float,
        currency: str = "USD",
        asset_class: str = "forex",
        stop_loss_pct: Optional[float] = None
    ) -> IntegratedRiskResult:
        """
        Evaluate a proposed trade against all risk criteria.

        This is the main method that integrates all risk checks.

        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            action: Trade action (BUY, SELL, etc.)
            quantity: Proposed quantity
            price: Current/entry price
            currency: Position currency
            asset_class: Asset classification
            stop_loss_pct: Optional stop loss percentage

        Returns:
            IntegratedRiskResult with decision and adjusted parameters

        Raises:
            ValidationError: If inputs fail validation
        """
        start_time = time.time()
        self._total_evaluations += 1

        # =====================================================================
        # INPUT VALIDATION (SECURITY)
        # =====================================================================
        try:
            symbol = InputValidator.validate_symbol(symbol)
            action = InputValidator.validate_action(action)
            quantity = InputValidator.validate_quantity(quantity)
            price = InputValidator.validate_price(price)

            # Validate optional stop_loss_pct
            if stop_loss_pct is not None:
                if not isinstance(stop_loss_pct, (int, float)):
                    raise ValidationError(f"stop_loss_pct must be numeric, got {type(stop_loss_pct)}")
                if stop_loss_pct <= 0 or stop_loss_pct > 1:
                    raise ValidationError(f"stop_loss_pct must be between 0 and 1, got {stop_loss_pct}")

            # Validate currency
            if not isinstance(currency, str) or len(currency) != 3:
                raise ValidationError(f"Invalid currency: {currency}")
            currency = currency.upper()

            # Validate asset_class
            valid_asset_classes = {"forex", "crypto", "equity", "commodity", "index"}
            if asset_class.lower() not in valid_asset_classes:
                raise ValidationError(
                    f"Invalid asset_class: {asset_class}. "
                    f"Valid: {', '.join(sorted(valid_asset_classes))}"
                )

        except ValidationError as e:
            self._logger.error(f"Input validation failed: {e}")
            result = IntegratedRiskResult(
                original_quantity=quantity if isinstance(quantity, (int, float)) else 0,
                approved_quantity=0,
                is_approved=False,
                decision=RiskDecision.REJECT_RISK
            )
            result.violations.append(f"Validation failed: {e}")
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            self._total_rejections += 1
            return result

        # Create result object
        result = IntegratedRiskResult(
            original_quantity=quantity,
            approved_quantity=quantity
        )

        # Create position for analysis
        proposed_position = Position(
            symbol=symbol,
            quantity=quantity if action in ["BUY", "OPEN_LONG"] else -quantity,
            entry_price=price,
            current_price=price,
            currency=currency,
            asset_class=asset_class
        )

        # =====================================================================
        # CHECK 1: KILL SWITCH
        # =====================================================================
        if self._kill_switch:
            if not self._kill_switch.is_trading_allowed():
                result.kill_switch_check_passed = False
                result.kill_switch_multiplier = 0.0
                result.decision = RiskDecision.REJECT_KILL_SWITCH
                result.is_approved = False
                result.approved_quantity = 0
                result.violations.append(
                    f"Kill switch active: {self._kill_switch.halt_reason.value if self._kill_switch.halt_reason else 'Unknown'}"
                )
                result.reasoning.append("Trade blocked by kill switch")

                self._log_decision(result, proposed_position)
                self._total_rejections += 1
                result.evaluation_time_ms = (time.time() - start_time) * 1000
                return result

            # Get kill switch position multiplier
            result.kill_switch_multiplier = self._kill_switch.get_position_multiplier()
            if result.kill_switch_multiplier < 1.0:
                result.warnings.append(
                    f"Kill switch reducing position to {result.kill_switch_multiplier:.0%}"
                )

        # =====================================================================
        # CHECK 2: DRAWDOWN
        # =====================================================================
        result.current_drawdown_pct = self._current_drawdown

        if self._current_drawdown >= self.config.max_drawdown:
            result.drawdown_check_passed = False
            result.drawdown_multiplier = 0.0
            result.decision = RiskDecision.REJECT_DRAWDOWN
            result.is_approved = False
            result.approved_quantity = 0
            result.violations.append(
                f"Max drawdown exceeded: {self._current_drawdown:.1%} >= {self.config.max_drawdown:.1%}"
            )
            result.reasoning.append("Trade blocked due to excessive drawdown")

            self._log_decision(result, proposed_position)
            self._total_rejections += 1
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        # Calculate drawdown multiplier (gradual reduction)
        if self.config.enable_gradual_reduction:
            if self._current_drawdown >= self.config.drawdown_warning:
                dd_ratio = (self._current_drawdown - self.config.drawdown_warning) / (
                    self.config.max_drawdown - self.config.drawdown_warning
                )
                result.drawdown_multiplier = max(0.3, 1.0 - dd_ratio * 0.7)
                result.warnings.append(
                    f"Drawdown warning: reducing position to {result.drawdown_multiplier:.0%}"
                )

        # =====================================================================
        # CHECK 3: VAR
        # =====================================================================
        if len(self._returns_history) >= 30:
            var_result = self._portfolio_risk.calculate_var(VaRMethod.HISTORICAL)
            result.portfolio_var_pct = var_result.var_pct
            result.portfolio_cvar_pct = var_result.cvar_pct or 0

            if var_result.var_pct > self.config.max_var_pct:
                result.var_check_passed = False
                result.var_multiplier = 0.0
                result.decision = RiskDecision.REJECT_VAR
                result.is_approved = False
                result.approved_quantity = 0
                result.violations.append(
                    f"VaR limit exceeded: {var_result.var_pct:.2%} > {self.config.max_var_pct:.1%}"
                )
                result.reasoning.append("Trade blocked due to VaR limit")

                self._log_decision(result, proposed_position)
                self._total_rejections += 1
                result.evaluation_time_ms = (time.time() - start_time) * 1000
                return result

            # Calculate VaR multiplier
            if var_result.var_pct > self.config.var_warning_threshold:
                var_ratio = (var_result.var_pct - self.config.var_warning_threshold) / (
                    self.config.max_var_pct - self.config.var_warning_threshold
                )
                result.var_multiplier = max(0.5, 1.0 - var_ratio * 0.5)
                result.warnings.append(
                    f"VaR warning: reducing position to {result.var_multiplier:.0%}"
                )

        # =====================================================================
        # CHECK 4: EXPOSURE
        # =====================================================================
        exp_allowed, exp_violations, exp_multiplier = self._portfolio_risk.exposure_manager.check_new_position(
            proposed_position,
            self._portfolio_risk.correlation_engine
        )

        exposure_report = self._portfolio_risk.exposure_manager.get_exposure_report()
        result.gross_exposure_pct = exposure_report.gross_exposure / self._equity if self._equity > 0 else 0
        result.net_exposure_pct = exposure_report.net_exposure / self._equity if self._equity > 0 else 0

        if not exp_allowed and exp_multiplier < self.config.min_position_multiplier:
            result.exposure_check_passed = False
            result.exposure_multiplier = 0.0
            result.decision = RiskDecision.REJECT_EXPOSURE
            result.is_approved = False
            result.approved_quantity = 0
            result.violations.extend(exp_violations)
            result.reasoning.append("Trade blocked due to exposure limits")

            self._log_decision(result, proposed_position)
            self._total_rejections += 1
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        result.exposure_multiplier = exp_multiplier
        if exp_violations:
            result.warnings.extend(exp_violations)

        # =====================================================================
        # CHECK 5: CORRELATION
        # =====================================================================
        corr_weights = self._portfolio_risk.correlation_engine.calculate_correlation_adjusted_weights(
            self._positions + [proposed_position],
            self.config.high_correlation_threshold
        )

        result.correlation_multiplier = corr_weights.get(symbol, 1.0)

        if result.correlation_multiplier < self.config.min_position_multiplier:
            result.correlation_check_passed = False
            result.decision = RiskDecision.REJECT_CORRELATION
            result.is_approved = False
            result.approved_quantity = 0
            result.violations.append(
                f"Position too correlated with existing positions"
            )
            result.reasoning.append("Trade blocked due to correlation limits")

            self._log_decision(result, proposed_position)
            self._total_rejections += 1
            result.evaluation_time_ms = (time.time() - start_time) * 1000
            return result

        if result.correlation_multiplier < 1.0:
            result.warnings.append(
                f"Correlation adjustment: reducing position to {result.correlation_multiplier:.0%}"
            )

        # =====================================================================
        # CALCULATE FINAL POSITION SIZE
        # =====================================================================
        final_multiplier = min(
            result.kill_switch_multiplier,
            result.drawdown_multiplier,
            result.var_multiplier,
            result.exposure_multiplier,
            result.correlation_multiplier
        )

        result.position_multiplier = max(
            self.config.min_position_multiplier,
            final_multiplier
        )

        result.approved_quantity = quantity * result.position_multiplier

        # =====================================================================
        # DETERMINE FINAL DECISION
        # =====================================================================
        if result.position_multiplier >= 0.99:
            result.decision = RiskDecision.APPROVE
            result.is_approved = True
            self._total_approvals += 1
        else:
            result.decision = RiskDecision.APPROVE_MODIFIED
            result.is_approved = True
            self._total_modifications += 1

        result.reasoning.append(
            f"Trade approved with {result.position_multiplier:.0%} position size"
        )

        # =====================================================================
        # LOG AND RETURN
        # =====================================================================
        result.evaluation_time_ms = (time.time() - start_time) * 1000
        self._log_decision(result, proposed_position)

        return result

    def _log_decision(
        self,
        result: IntegratedRiskResult,
        position: Position
    ) -> None:
        """Log the decision to audit trail."""
        if not self._audit_logger:
            return

        self._audit_logger.log_decision(
            decision_id=result.decision_id,
            proposal_id=f"prop_{uuid.uuid4().hex[:8]}",
            proposed_action="BUY" if position.is_long else "SELL",
            proposed_quantity=result.original_quantity,
            proposed_symbol=position.symbol,
            final_decision="APPROVE" if result.is_approved else "REJECT",
            final_quantity=result.approved_quantity,
            position_multiplier=result.position_multiplier,
            agent_assessments=[
                {
                    "agent": "var_check",
                    "passed": result.var_check_passed,
                    "multiplier": result.var_multiplier
                },
                {
                    "agent": "exposure_check",
                    "passed": result.exposure_check_passed,
                    "multiplier": result.exposure_multiplier
                },
                {
                    "agent": "correlation_check",
                    "passed": result.correlation_check_passed,
                    "multiplier": result.correlation_multiplier
                },
                {
                    "agent": "kill_switch_check",
                    "passed": result.kill_switch_check_passed,
                    "multiplier": result.kill_switch_multiplier
                },
                {
                    "agent": "drawdown_check",
                    "passed": result.drawdown_check_passed,
                    "multiplier": result.drawdown_multiplier
                }
            ],
            reasoning=result.reasoning,
            decision_time_ms=result.evaluation_time_ms,
            portfolio_state={
                "equity": self._equity,
                "drawdown_pct": self._current_drawdown,
                "var_pct": result.portfolio_var_pct,
                "gross_exposure_pct": result.gross_exposure_pct
            }
        )

    # =========================================================================
    # CONTROL METHODS
    # =========================================================================

    def emergency_halt(self, reason: str = "Manual emergency halt") -> None:
        """Trigger emergency trading halt."""
        if self._kill_switch:
            self._kill_switch.emergency_halt(reason)

        if self._audit_logger:
            self._audit_logger.log_kill_switch_event(
                triggered=True,
                reason=reason,
                halt_level="EMERGENCY"
            )

    def manual_halt(self, reason: str = "Manual halt") -> None:
        """Trigger manual trading halt."""
        if self._kill_switch:
            self._kill_switch.manual_halt(reason)

    def request_reset(self) -> Optional[str]:
        """Request a reset token."""
        if self._kill_switch:
            return self._kill_switch.request_reset()
        return None

    def confirm_reset(self, token: str) -> bool:
        """Confirm reset with token."""
        if self._kill_switch:
            success = self._kill_switch.confirm_reset(token)
            if success and self._audit_logger:
                self._audit_logger.log_kill_switch_event(
                    triggered=False,
                    reason="Manual reset confirmed",
                    halt_level="NONE"
                )
            return success
        return False

    def reset_daily_counters(self) -> None:
        """Reset daily counters."""
        self._daily_pnl = 0.0
        if self._kill_switch:
            self._kill_switch.reset_daily_counters()
        self._portfolio_risk.reset_daily_pnl()

    def reset_weekly_counters(self) -> None:
        """Reset weekly counters."""
        self._weekly_pnl = 0.0
        if self._kill_switch:
            self._kill_switch.reset_weekly_counters()

    # =========================================================================
    # QUERIES
    # =========================================================================

    def is_trading_allowed(self) -> bool:
        """Check if trading is currently allowed."""
        if self._kill_switch and not self._kill_switch.is_trading_allowed():
            return False
        if self._current_drawdown >= self.config.max_drawdown:
            return False
        return True

    def get_position_multiplier(self) -> float:
        """Get current position size multiplier."""
        multiplier = 1.0

        # Kill switch
        if self._kill_switch:
            multiplier = min(multiplier, self._kill_switch.get_position_multiplier())

        # Drawdown
        if self._current_drawdown >= self.config.drawdown_warning:
            dd_mult = max(0.3, 1.0 - (
                (self._current_drawdown - self.config.drawdown_warning) /
                (self.config.max_drawdown - self.config.drawdown_warning)
            ) * 0.7)
            multiplier = min(multiplier, dd_mult)

        return max(self.config.min_position_multiplier, multiplier)

    def get_risk_report(self) -> Dict[str, Any]:
        """Get comprehensive risk report."""
        portfolio_report = self._portfolio_risk.get_risk_report()
        kill_switch_status = self._kill_switch.get_status() if self._kill_switch else None

        return {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "trading_allowed": self.is_trading_allowed(),
            "position_multiplier": self.get_position_multiplier(),
            "statistics": {
                "total_evaluations": self._total_evaluations,
                "approvals": self._total_approvals,
                "rejections": self._total_rejections,
                "modifications": self._total_modifications,
                "approval_rate": self._total_approvals / max(1, self._total_evaluations)
            },
            "portfolio_risk": portfolio_report,
            "kill_switch": kill_switch_status,
            "config": {
                "max_var_pct": self.config.max_var_pct,
                "max_drawdown": self.config.max_drawdown,
                "max_gross_exposure": self.config.max_gross_exposure
            }
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get evaluation statistics."""
        return {
            "total_evaluations": self._total_evaluations,
            "approvals": self._total_approvals,
            "rejections": self._total_rejections,
            "modifications": self._total_modifications,
            "approval_rate": self._total_approvals / max(1, self._total_evaluations),
            "modification_rate": self._total_modifications / max(1, self._total_evaluations),
            "rejection_rate": self._total_rejections / max(1, self._total_evaluations)
        }

    # =========================================================================
    # DASHBOARD
    # =========================================================================

    def get_dashboard(self) -> str:
        """Generate comprehensive text dashboard."""
        stats = self.get_statistics()
        is_halted = not self.is_trading_allowed()

        # Status indicator
        if is_halted:
            status = "[X] HALTED"
        elif self.get_position_multiplier() < 1.0:
            status = "[!] REDUCED"
        else:
            status = "[OK] NORMAL"

        portfolio_dashboard = self._portfolio_risk.get_dashboard()
        kill_switch_dashboard = self._kill_switch.get_dashboard() if self._kill_switch else ""

        return f"""
################################################################################
#                    INTEGRATED RISK MANAGEMENT DASHBOARD                       #
################################################################################

  SESSION: {self.session_id}
  STATUS:  {status}
  POSITION MULTIPLIER: {self.get_position_multiplier():.0%}

  EVALUATION STATISTICS
  ─────────────────────────────────────────────────────────────────────────────
  Total Evaluations:   {stats['total_evaluations']:>10,}
  Approvals:           {stats['approvals']:>10,}   ({stats['approval_rate']:.1%})
  Modifications:       {stats['modifications']:>10,}   ({stats['modification_rate']:.1%})
  Rejections:          {stats['rejections']:>10,}   ({stats['rejection_rate']:.1%})

{portfolio_dashboard}

{kill_switch_dashboard}

################################################################################
"""

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def close(self) -> None:
        """Close and cleanup resources."""
        if self._audit_logger:
            self._audit_logger.close()

        self._logger.info(
            f"Integrated Risk Manager closed. "
            f"Evaluations: {self._total_evaluations}, "
            f"Approvals: {self._total_approvals}, "
            f"Rejections: {self._total_rejections}"
        )


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_integrated_risk_manager(
    equity: float,
    preset: str = "moderate",
    session_id: Optional[str] = None
) -> IntegratedRiskManager:
    """
    Create an IntegratedRiskManager with preset configuration.

    Args:
        equity: Initial equity
        preset: "conservative", "moderate", "aggressive"
        session_id: Optional session ID

    Returns:
        Configured IntegratedRiskManager
    """
    presets = {
        "conservative": IntegratedRiskConfig(
            max_var_pct=0.01,
            max_cvar_pct=0.015,
            max_drawdown=0.05,
            max_daily_loss=0.02,
            max_gross_exposure=1.0,
            kill_switch_preset="conservative"
        ),
        "moderate": IntegratedRiskConfig(
            max_var_pct=0.02,
            max_cvar_pct=0.03,
            max_drawdown=0.10,
            max_daily_loss=0.03,
            max_gross_exposure=1.5,
            kill_switch_preset="moderate"
        ),
        "aggressive": IntegratedRiskConfig(
            max_var_pct=0.03,
            max_cvar_pct=0.05,
            max_drawdown=0.15,
            max_daily_loss=0.05,
            max_gross_exposure=2.0,
            kill_switch_preset="aggressive"
        )
    }

    config = presets.get(preset, presets["moderate"])

    return IntegratedRiskManager(
        equity=equity,
        config=config,
        session_id=session_id
    )


# =============================================================================
# ADAPTER FOR EXISTING SYSTEM
# =============================================================================

class RiskSentinelAdapter:
    """
    Adapter to integrate IntegratedRiskManager with existing
    IntelligentRiskSentinel and Orchestrator.

    This allows gradual migration without breaking existing code.
    """

    def __init__(self, integrated_manager: IntegratedRiskManager):
        self.manager = integrated_manager

    def evaluate_trade(self, proposal: TradeProposal) -> RiskAssessment:
        """
        Evaluate trade proposal using integrated risk manager.

        Args:
            proposal: TradeProposal from existing system

        Returns:
            RiskAssessment compatible with existing system
        """
        # Map proposal to integrated manager format
        result = self.manager.evaluate_trade(
            symbol=proposal.symbol,
            action=proposal.action,
            quantity=proposal.quantity,
            price=proposal.entry_price,
            currency="USD",
            asset_class="forex"
        )

        # Map result back to RiskAssessment
        if result.is_approved:
            if result.position_multiplier >= 0.99:
                decision = DecisionType.APPROVE
            else:
                decision = DecisionType.MODIFY
        else:
            decision = DecisionType.REJECT

        # Map risk level
        if result.decision in [RiskDecision.REJECT_KILL_SWITCH, RiskDecision.REJECT_DRAWDOWN]:
            risk_level = RiskLevel.CRITICAL
        elif result.decision in [RiskDecision.REJECT_VAR, RiskDecision.REJECT_EXPOSURE]:
            risk_level = RiskLevel.HIGH
        elif result.decision == RiskDecision.APPROVE_MODIFIED:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Create violations
        violations = []
        for v in result.violations:
            violations.append(RiskViolation(
                rule_name=result.decision.value,
                rule_description=v,
                severity=risk_level,
                current_value=0,
                threshold=0,
                recommendation="Reduce position size or wait"
            ))

        return RiskAssessment(
            proposal_id=proposal.proposal_id,
            decision=decision,
            risk_score=50 if result.is_approved else 80,
            risk_level=risk_level,
            violations=violations,
            reasoning=result.reasoning,
            modified_params={
                "suggested_quantity": result.approved_quantity,
                "position_multiplier": result.position_multiplier
            } if decision == DecisionType.MODIFY else None,
            assessment_time_ms=result.evaluation_time_ms
        )
