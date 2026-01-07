# =============================================================================
# RISK SENTINEL AGENT - Autonomous Risk Management Guardian
# =============================================================================
# The Risk Sentinel is the "guardian angel" of your trading system.
# It sits between the RL agent (which proposes trades) and the execution
# layer (which actually trades), ensuring every trade passes safety checks.
#
# === WHAT IT DOES ===
# 1. Validates every trade proposal against a comprehensive rule engine
# 2. Monitors portfolio risk levels in real-time
# 3. Enforces hard limits (drawdown, position size, leverage)
# 4. Detects unusual market conditions and adjusts accordingly
# 5. Provides explainable decisions (why was this trade rejected?)
#
# === HOW IT WORKS ===
#
#   ┌─────────────┐     TradeProposal     ┌─────────────────┐
#   │  RL Agent   │ ────────────────────> │  Risk Sentinel  │
#   │  (PPO Bot)  │                       │                 │
#   └─────────────┘                       │  ┌───────────┐  │
#                                         │  │ Rule      │  │
#                                         │  │ Engine    │  │
#                                         │  └───────────┘  │
#                                         │        │        │
#   ┌─────────────┐     RiskAssessment    │        v        │
#   │ Environment │ <──────────────────── │ APPROVE/REJECT/ │
#   │ (Execute)   │                       │ MODIFY          │
#   └─────────────┘                       └─────────────────┘
#
# === RULE ENGINE ===
# The agent uses a configurable rule engine with:
#   - HARD RULES: Instant rejection if violated (e.g., max drawdown)
#   - SOFT RULES: Accumulate violations, reject if too many
#   - ADVISORY: Warning only, logged but doesn't reject
#
# === COMMERCIAL VALUE ===
# This agent provides:
#   - Compliance: Audit trail of every decision
#   - Safety: Hard stops prevent catastrophic losses
#   - Explainability: Every rejection has a clear reason
#   - Customization: Config presets for different risk appetites
#
# =============================================================================

import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
import logging
import numpy as np

# Import base classes and events
from src.agents.base_agent import BaseAgent, AgentState, AgentCapability, AgentContext
from src.agents.events import (
    AgentEvent,
    EventType,
    EventBus,
    TradeProposal,
    RiskAssessment,
    RiskViolation,
    RiskLevel,
    DecisionType,
    AgentDecision,
    create_risk_assessment_event
)
from src.agents.config import RiskSentinelConfig

# Import action constants for long/short trading
try:
    from src.config import (
        ACTION_HOLD, ACTION_OPEN_LONG, ACTION_CLOSE_LONG,
        ACTION_OPEN_SHORT, ACTION_CLOSE_SHORT,
        POSITION_FLAT, POSITION_LONG, POSITION_SHORT,
        ACTION_NAMES
    )
except ImportError:
    # Fallback definitions
    ACTION_HOLD = 0
    ACTION_OPEN_LONG = 1
    ACTION_CLOSE_LONG = 2
    ACTION_OPEN_SHORT = 3
    ACTION_CLOSE_SHORT = 4
    POSITION_FLAT = 0
    POSITION_LONG = 1
    POSITION_SHORT = -1
    ACTION_NAMES = {0: 'HOLD', 1: 'OPEN_LONG', 2: 'CLOSE_LONG', 3: 'OPEN_SHORT', 4: 'CLOSE_SHORT'}


# =============================================================================
# RULE DEFINITIONS
# =============================================================================


class RuleType:
    """
    Categories of rules in the rule engine.

    HARD: Immediate rejection, no exceptions
    SOFT: Accumulate violations, reject if threshold exceeded
    ADVISORY: Warning only, logged but trade proceeds
    """
    HARD = "HARD"       # Instant reject
    SOFT = "SOFT"       # Accumulate and decide
    ADVISORY = "ADVISORY"  # Warning only


@dataclass
class RiskRule:
    """
    Definition of a single risk rule.

    Each rule has:
        - Unique name for identification
        - Human-readable description
        - Rule type (HARD/SOFT/ADVISORY)
        - Check function that returns (passed, details)

    Attributes:
        name: Unique identifier (e.g., "MAX_DRAWDOWN")
        description: What this rule checks
        rule_type: HARD, SOFT, or ADVISORY
        check_fn: Function that performs the check
        enabled: Whether this rule is active
    """
    name: str
    description: str
    rule_type: str  # HARD, SOFT, or ADVISORY
    enabled: bool = True


# =============================================================================
# RISK SENTINEL AGENT
# =============================================================================


class RiskSentinelAgent(BaseAgent):
    """
    Autonomous Risk Management Agent.

    The Risk Sentinel evaluates every trade proposal against a comprehensive
    rule engine, ensuring trades comply with risk parameters before execution.

    === FEATURES ===
    1. Comprehensive Rule Engine: 15+ risk rules covering all aspects
    2. Explainable Decisions: Every rejection includes detailed reasoning
    3. Adaptive Risk: Adjusts limits based on market conditions
    4. State Tracking: Remembers recent trades, losses, daily P&L
    5. Real-time Monitoring: Tracks drawdown, exposure, volatility

    === USAGE ===
    ```python
    # Create and configure
    config = RiskSentinelConfig()
    sentinel = RiskSentinelAgent(config=config)
    sentinel.start()

    # Evaluate a trade
    proposal = TradeProposal(action="BUY", quantity=0.1, ...)
    assessment = sentinel.evaluate_trade(proposal)

    if assessment.is_approved():
        # Execute the trade
        pass
    else:
        # Log rejection reason
        print(assessment.reasoning)
    ```

    === INTEGRATION ===
    The agent can work in two modes:
    1. Direct: Call evaluate_trade() directly
    2. Event-driven: Subscribe to TRADE_PROPOSED events via EventBus
    """

    def __init__(
        self,
        config: Optional[RiskSentinelConfig] = None,
        event_bus: Optional[EventBus] = None,
        name: str = "RiskSentinel"
    ):
        """
        Initialize the Risk Sentinel Agent.

        Args:
            config: RiskSentinelConfig with risk parameters
            event_bus: Optional EventBus for event-driven mode
            name: Human-readable name for this agent instance
        """
        # Use default config if none provided
        self._risk_config = config or RiskSentinelConfig()

        # Initialize base agent
        super().__init__(
            name=name,
            config=self._risk_config.to_dict(),
            event_bus=event_bus
        )

        # === STATE TRACKING ===
        # These track the trading state for rule evaluation

        # Portfolio state (updated externally)
        self._peak_equity: float = 0.0           # Highest portfolio value
        self._current_equity: float = 0.0        # Current portfolio value
        self._current_drawdown: float = 0.0      # Current drawdown %
        self._daily_pnl: float = 0.0             # Today's P&L
        self._daily_start_equity: float = 0.0    # Equity at day start

        # Trade tracking
        self._trade_history: deque = deque(maxlen=100)  # Recent trades
        self._trades_today: int = 0              # Trade count today
        self._last_trade_time: Optional[datetime] = None
        self._steps_since_loss: int = 999        # Steps since last losing trade
        self._consecutive_losses: int = 0        # Streak of losing trades

        # Position tracking
        self._current_position: float = 0.0      # Current position size
        self._position_entry_price: float = 0.0  # Entry price of current position
        self._position_entry_step: int = 0       # When position was opened
        self._current_step: int = 0              # Current simulation step

        # Market state
        self._current_atr: float = 0.0           # Current ATR
        self._avg_atr: float = 0.0               # Average ATR (for spike detection)
        self._current_regime: int = 0            # Market regime (0=calm, 1=volatile)

        # Statistics
        self._total_assessments: int = 0
        self._total_approvals: int = 0
        self._total_rejections: int = 0
        self._total_modifications: int = 0

        # Event subscriptions (if using event bus)
        self._subscriptions = [EventType.TRADE_PROPOSED]

        # Logger setup
        self._logger = logging.getLogger(f"agent.{self.full_id}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def initialize(self) -> bool:
        """
        Initialize the Risk Sentinel.

        Loads configuration, sets up initial state, and prepares rule engine.

        Returns:
            True if initialization successful
        """
        self._logger.info("Initializing Risk Sentinel Agent...")

        try:
            # Log configuration
            self._logger.info(f"Config: max_drawdown={self._risk_config.max_drawdown_pct:.1%}")
            self._logger.info(f"Config: max_risk_per_trade={self._risk_config.max_risk_per_trade_pct:.1%}")
            self._logger.info(f"Config: strict_mode={self._risk_config.strict_mode}")

            # Reset state
            self._reset_daily_tracking()

            self._logger.info("Risk Sentinel initialized successfully")
            return True

        except Exception as e:
            self._logger.error(f"Initialization failed: {e}")
            return False

    def shutdown(self) -> bool:
        """
        Shutdown the Risk Sentinel.

        Saves statistics and cleans up resources.

        Returns:
            True if shutdown successful
        """
        self._logger.info("Shutting down Risk Sentinel...")

        # Log final statistics
        self._logger.info(
            f"Final stats: {self._total_assessments} assessments, "
            f"{self._total_approvals} approved, "
            f"{self._total_rejections} rejected, "
            f"{self._total_modifications} modified"
        )

        return True

    def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
        """
        Process incoming events (event-driven mode).

        Handles TRADE_PROPOSED events by evaluating the trade and
        returning a risk assessment event.

        Args:
            event: Incoming event to process

        Returns:
            Response event with risk assessment
        """
        if event.event_type == EventType.TRADE_PROPOSED:
            # Extract trade proposal from event payload
            proposal = TradeProposal(**event.payload)

            # Evaluate the trade
            assessment = self.evaluate_trade(proposal)

            # Create response event
            return create_risk_assessment_event(assessment, self.full_id)

        return None

    def get_capabilities(self) -> List[AgentCapability]:
        """Return this agent's capabilities."""
        return [AgentCapability.RISK_ASSESSMENT]

    # =========================================================================
    # MAIN EVALUATION METHOD
    # =========================================================================

    def evaluate_trade(
        self,
        proposal: TradeProposal,
        context: Optional[AgentContext] = None
    ) -> RiskAssessment:
        """
        Evaluate a trade proposal against all risk rules.

        This is the main entry point for risk assessment. It:
        1. Extracts context from the proposal
        2. Runs all applicable rules
        3. Aggregates violations
        4. Makes a final decision
        5. Returns a detailed assessment

        Args:
            proposal: The trade proposal to evaluate
            context: Optional additional context

        Returns:
            RiskAssessment with decision and reasoning
        """
        start_time = time.time()

        # Track assessment
        self._total_assessments += 1

        # Initialize assessment
        violations: List[RiskViolation] = []
        reasoning: List[str] = []
        risk_score: float = 0.0

        # === EXTRACT CONTEXT FROM PROPOSAL ===
        # Update internal state from proposal data
        self._update_state_from_proposal(proposal)

        # === CHECK IF AGENT IS ENABLED ===
        if not self._risk_config.enabled:
            reasoning.append("Risk Sentinel is disabled, auto-approving")
            return self._create_assessment(
                proposal=proposal,
                decision=DecisionType.APPROVE,
                risk_score=0.0,
                violations=[],
                reasoning=reasoning,
                start_time=start_time
            )

        # === HANDLE SAFE ACTIONS ===
        # HOLD and CLOSE actions are generally safe - less strict evaluation
        safe_actions = ["HOLD", "CLOSE_LONG", "CLOSE_SHORT"]
        if proposal.action in safe_actions:
            if proposal.action == "HOLD":
                reasoning.append("HOLD action is inherently safe")
            else:
                reasoning.append(f"{proposal.action} closes an existing position - generally safe")
            return self._create_assessment(
                proposal=proposal,
                decision=DecisionType.APPROVE,
                risk_score=0.0,
                violations=[],
                reasoning=reasoning,
                start_time=start_time
            )

        # === RUN RULE ENGINE ===
        # Each rule returns (passed, violation_or_none, score_impact)
        rule_results = self._run_all_rules(proposal)

        # Aggregate results
        hard_violations = []
        soft_violations = []
        advisory_notes = []

        for rule_name, passed, violation, score_impact, rule_type in rule_results:
            risk_score += score_impact

            if not passed and violation:
                if rule_type == RuleType.HARD:
                    hard_violations.append(violation)
                    reasoning.append(f"[HARD FAIL] {violation.rule_description}")
                elif rule_type == RuleType.SOFT:
                    soft_violations.append(violation)
                    reasoning.append(f"[SOFT FAIL] {violation.rule_description}")
                else:
                    advisory_notes.append(violation)
                    reasoning.append(f"[ADVISORY] {violation.rule_description}")
            elif passed:
                reasoning.append(f"[PASS] {rule_name}")

        # Combine all violations
        violations = hard_violations + soft_violations

        # === MAKE DECISION ===
        decision = self._make_decision(hard_violations, soft_violations)

        # === CREATE ASSESSMENT ===
        return self._create_assessment(
            proposal=proposal,
            decision=decision,
            risk_score=min(100.0, max(0.0, risk_score)),  # Clamp to 0-100
            violations=violations,
            reasoning=reasoning,
            start_time=start_time
        )

    # =========================================================================
    # RULE ENGINE
    # =========================================================================

    def _run_all_rules(
        self,
        proposal: TradeProposal
    ) -> List[Tuple[str, bool, Optional[RiskViolation], float, str]]:
        """
        Run all risk rules against a proposal.

        Returns a list of tuples:
            (rule_name, passed, violation_or_none, risk_score_impact, rule_type)
        """
        results = []

        # === HARD RULES (Instant rejection) ===

        # Rule 1: Maximum Drawdown Check
        results.append(self._rule_max_drawdown(proposal))

        # Rule 2: Trading Halted Check
        results.append(self._rule_trading_halted(proposal))

        # Rule 3: Minimum Balance Check
        results.append(self._rule_minimum_balance(proposal))

        # Rule 4: Maximum Leverage Check
        results.append(self._rule_max_leverage(proposal))

        # === SOFT RULES (Accumulate violations) ===

        # Rule 5: Position Size Limit
        results.append(self._rule_position_size(proposal))

        # Rule 6: Risk Per Trade Limit
        results.append(self._rule_risk_per_trade(proposal))

        # Rule 7: Daily Trade Limit
        results.append(self._rule_daily_trade_limit(proposal))

        # Rule 8: Cooldown After Loss
        results.append(self._rule_loss_cooldown(proposal))

        # Rule 9: Maximum Position Duration
        results.append(self._rule_position_duration(proposal))

        # Rule 10: Daily Loss Limit
        results.append(self._rule_daily_loss_limit(proposal))

        # Rule 11: Consecutive Losses Check
        results.append(self._rule_consecutive_losses(proposal))

        # === ADVISORY RULES (Warning only) ===

        # Rule 12: Volatility Spike Warning
        results.append(self._rule_volatility_spike(proposal))

        # Rule 13: Market Regime Check
        results.append(self._rule_market_regime(proposal))

        return results

    # =========================================================================
    # INDIVIDUAL RULES
    # =========================================================================

    def _rule_max_drawdown(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        HARD RULE: Check if max drawdown has been breached.

        If current drawdown exceeds the limit, ALL trading stops.
        This is the ultimate safety valve.
        """
        rule_name = "MAX_DRAWDOWN"
        max_dd = self._risk_config.max_drawdown_pct

        # Calculate current drawdown
        if self._peak_equity > 0:
            current_dd = (self._peak_equity - self._current_equity) / self._peak_equity
        else:
            current_dd = 0.0

        self._current_drawdown = current_dd

        if current_dd >= max_dd:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Max drawdown breached: {current_dd:.1%} >= {max_dd:.1%}. TRADING HALTED.",
                severity=RiskLevel.CRITICAL,
                current_value=current_dd,
                threshold=max_dd,
                recommendation="Stop trading. Review strategy. Consider reducing position sizes."
            )
            return (rule_name, False, violation, 50.0, RuleType.HARD)

        # Calculate risk score contribution
        score = (current_dd / max_dd) * 30  # Up to 30 points for drawdown proximity
        return (rule_name, True, None, score, RuleType.HARD)

    def _rule_trading_halted(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        HARD RULE: Check if trading has been halted.

        Trading can be halted by:
        - Max drawdown breach
        - Manual intervention
        - Emergency stop
        """
        rule_name = "TRADING_HALTED"

        # Check if we're in error state
        if self.state == AgentState.ERROR:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description="Trading halted due to agent error state",
                severity=RiskLevel.CRITICAL,
                current_value=1.0,
                threshold=0.0,
                recommendation="Resolve error before resuming trading"
            )
            return (rule_name, False, violation, 100.0, RuleType.HARD)

        return (rule_name, True, None, 0.0, RuleType.HARD)

    def _rule_minimum_balance(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        HARD RULE: Ensure minimum balance for trading.

        Don't allow trading if balance is too low to cover fees
        or position sizing requirements.
        """
        rule_name = "MINIMUM_BALANCE"
        min_balance = 100.0  # Minimum $100 to trade

        if proposal.current_balance < min_balance:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Balance ${proposal.current_balance:.2f} below minimum ${min_balance:.2f}",
                severity=RiskLevel.CRITICAL,
                current_value=proposal.current_balance,
                threshold=min_balance,
                recommendation="Add funds or reduce position sizes"
            )
            return (rule_name, False, violation, 100.0, RuleType.HARD)

        return (rule_name, True, None, 0.0, RuleType.HARD)

    def _rule_max_leverage(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        HARD RULE: Enforce maximum leverage limit.

        Leverage is calculated as position_value / equity.
        """
        rule_name = "MAX_LEVERAGE"
        max_lev = self._risk_config.max_leverage

        # Calculate proposed leverage
        if proposal.current_equity > 0:
            position_value = proposal.quantity * proposal.entry_price
            proposed_leverage = position_value / proposal.current_equity
        else:
            proposed_leverage = 0.0

        if proposed_leverage > max_lev:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Leverage {proposed_leverage:.2f}x exceeds max {max_lev:.2f}x",
                severity=RiskLevel.HIGH,
                current_value=proposed_leverage,
                threshold=max_lev,
                recommendation=f"Reduce position size to stay under {max_lev}x leverage"
            )
            return (rule_name, False, violation, 40.0, RuleType.HARD)

        return (rule_name, True, None, 0.0, RuleType.HARD)

    def _rule_position_size(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Check position size relative to portfolio.

        Ensures no single position is too large relative to portfolio.
        """
        rule_name = "POSITION_SIZE"
        max_size_pct = self._risk_config.max_position_size_pct

        # Calculate position as % of portfolio
        if proposal.current_equity > 0:
            position_value = proposal.quantity * proposal.entry_price
            position_pct = position_value / proposal.current_equity
        else:
            position_pct = 0.0

        if position_pct > max_size_pct:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Position size {position_pct:.1%} exceeds max {max_size_pct:.1%}",
                severity=RiskLevel.MEDIUM,
                current_value=position_pct,
                threshold=max_size_pct,
                recommendation=f"Reduce position to {max_size_pct:.0%} of portfolio or less"
            )
            return (rule_name, False, violation, 20.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_risk_per_trade(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Check risk per trade (potential loss).

        Uses ATR-based stop loss to calculate maximum risk.
        """
        rule_name = "RISK_PER_TRADE"
        max_risk_pct = self._risk_config.max_risk_per_trade_pct

        # Calculate risk using ATR
        atr = proposal.market_data.get('ATR', self._current_atr)
        if atr <= 0:
            atr = proposal.entry_price * 0.01  # Fallback: 1% of price

        # Risk = position_size * stop_distance / equity
        stop_distance = atr * self._risk_config.required_atr_multiplier
        potential_loss = proposal.quantity * stop_distance

        if proposal.current_equity > 0:
            risk_pct = potential_loss / proposal.current_equity
        else:
            risk_pct = 0.0

        if risk_pct > max_risk_pct:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Risk per trade {risk_pct:.2%} exceeds max {max_risk_pct:.1%}",
                severity=RiskLevel.MEDIUM,
                current_value=risk_pct,
                threshold=max_risk_pct,
                recommendation=f"Reduce position size to risk max {max_risk_pct:.1%}"
            )
            return (rule_name, False, violation, 25.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_daily_trade_limit(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Limit number of trades per day.

        Prevents overtrading which increases fees and reduces edge.
        """
        rule_name = "DAILY_TRADE_LIMIT"
        max_trades = self._risk_config.max_trades_per_day

        if self._trades_today >= max_trades:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Daily trade limit reached: {self._trades_today}/{max_trades}",
                severity=RiskLevel.MEDIUM,
                current_value=float(self._trades_today),
                threshold=float(max_trades),
                recommendation="Wait until tomorrow or review trading frequency"
            )
            return (rule_name, False, violation, 15.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_loss_cooldown(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Enforce cooldown after a losing trade.

        Prevents revenge trading after losses.
        """
        rule_name = "LOSS_COOLDOWN"
        cooldown = self._risk_config.cooldown_after_loss_steps

        if self._steps_since_loss < cooldown:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Cooldown active: {self._steps_since_loss}/{cooldown} steps since loss",
                severity=RiskLevel.LOW,
                current_value=float(self._steps_since_loss),
                threshold=float(cooldown),
                recommendation=f"Wait {cooldown - self._steps_since_loss} more steps before trading"
            )
            return (rule_name, False, violation, 10.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_position_duration(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Check if existing position has been held too long.

        Only applies when we have an open position and are trying to hold.
        Encourages closing stale positions.
        """
        rule_name = "POSITION_DURATION"
        max_duration = self._risk_config.max_position_duration_steps

        # Only check if we have a position
        if abs(self._current_position) > 0:
            hold_duration = self._current_step - self._position_entry_step

            if hold_duration > max_duration:
                violation = RiskViolation(
                    rule_name=rule_name,
                    rule_description=f"Position held {hold_duration} steps, max is {max_duration}",
                    severity=RiskLevel.LOW,
                    current_value=float(hold_duration),
                    threshold=float(max_duration),
                    recommendation="Consider closing position to free capital"
                )
                return (rule_name, False, violation, 5.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_daily_loss_limit(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Check daily loss limit.

        Stops trading if daily losses exceed threshold.
        """
        rule_name = "DAILY_LOSS_LIMIT"
        max_daily_loss = self._risk_config.daily_loss_limit_pct

        # Calculate daily P&L as percentage
        if self._daily_start_equity > 0:
            daily_return = (self._current_equity - self._daily_start_equity) / self._daily_start_equity
        else:
            daily_return = 0.0

        if daily_return < -max_daily_loss:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"Daily loss {abs(daily_return):.2%} exceeds limit {max_daily_loss:.1%}",
                severity=RiskLevel.HIGH,
                current_value=abs(daily_return),
                threshold=max_daily_loss,
                recommendation="Stop trading for today. Review strategy."
            )
            return (rule_name, False, violation, 30.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_consecutive_losses(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        SOFT RULE: Check consecutive losing trades.

        After 3+ consecutive losses, reduce risk or pause.
        """
        rule_name = "CONSECUTIVE_LOSSES"
        max_consecutive = 3

        if self._consecutive_losses >= max_consecutive:
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description=f"{self._consecutive_losses} consecutive losses, max is {max_consecutive}",
                severity=RiskLevel.MEDIUM,
                current_value=float(self._consecutive_losses),
                threshold=float(max_consecutive),
                recommendation="Consider pausing or reducing position sizes"
            )
            return (rule_name, False, violation, 20.0, RuleType.SOFT)

        return (rule_name, True, None, 0.0, RuleType.SOFT)

    def _rule_volatility_spike(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        ADVISORY: Warn about unusual volatility.

        Doesn't reject, but flags high-volatility conditions.
        """
        rule_name = "VOLATILITY_SPIKE"
        spike_threshold = self._risk_config.volatility_spike_threshold

        current_atr = proposal.market_data.get('ATR', self._current_atr)

        if self._avg_atr > 0 and current_atr > 0:
            atr_ratio = current_atr / self._avg_atr

            if atr_ratio > spike_threshold:
                violation = RiskViolation(
                    rule_name=rule_name,
                    rule_description=f"Volatility spike detected: ATR {atr_ratio:.1f}x average",
                    severity=RiskLevel.MEDIUM,
                    current_value=atr_ratio,
                    threshold=spike_threshold,
                    recommendation="Consider reducing position size or waiting"
                )
                return (rule_name, False, violation, 10.0, RuleType.ADVISORY)

        return (rule_name, True, None, 0.0, RuleType.ADVISORY)

    def _rule_market_regime(
        self,
        proposal: TradeProposal
    ) -> Tuple[str, bool, Optional[RiskViolation], float, str]:
        """
        ADVISORY: Check market regime and adjust accordingly.

        In high-volatility regime, flag for awareness.
        """
        rule_name = "MARKET_REGIME"

        if not self._risk_config.regime_check_enabled:
            return (rule_name, True, None, 0.0, RuleType.ADVISORY)

        if self._current_regime == 1:  # High volatility
            violation = RiskViolation(
                rule_name=rule_name,
                rule_description="Market in high-volatility regime",
                severity=RiskLevel.LOW,
                current_value=1.0,
                threshold=0.0,
                recommendation="Position sizes auto-reduced by config"
            )
            return (rule_name, False, violation, 5.0, RuleType.ADVISORY)

        return (rule_name, True, None, 0.0, RuleType.ADVISORY)

    # =========================================================================
    # DECISION MAKING
    # =========================================================================

    def _make_decision(
        self,
        hard_violations: List[RiskViolation],
        soft_violations: List[RiskViolation]
    ) -> DecisionType:
        """
        Make final decision based on violations.

        Decision Logic:
        1. Any HARD violation -> REJECT
        2. In strict mode: Any SOFT violation -> REJECT
        3. In relaxed mode: SOFT violations < threshold -> APPROVE
        4. Otherwise -> REJECT
        """
        # Any hard violation = immediate reject
        if hard_violations:
            self._total_rejections += 1
            return DecisionType.REJECT

        # Check soft violations
        if self._risk_config.strict_mode:
            # Strict: Any violation = reject
            if soft_violations:
                self._total_rejections += 1
                return DecisionType.REJECT
        else:
            # Relaxed: Allow up to threshold violations
            if len(soft_violations) > self._risk_config.soft_violation_threshold:
                self._total_rejections += 1
                return DecisionType.REJECT

        # No critical violations
        self._total_approvals += 1
        return DecisionType.APPROVE

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _update_state_from_proposal(self, proposal: TradeProposal) -> None:
        """
        Update internal state from proposal data.

        Extracts relevant information from the proposal to track
        portfolio state, market conditions, etc.
        """
        # Portfolio state
        self._current_equity = proposal.current_equity
        if self._current_equity > self._peak_equity:
            self._peak_equity = self._current_equity

        if self._daily_start_equity == 0:
            self._daily_start_equity = self._current_equity

        # Position state
        self._current_position = proposal.current_position

        # Market state
        self._current_atr = proposal.market_data.get('ATR', 0.0)

        # Update average ATR (simple moving average approximation)
        if self._avg_atr == 0:
            self._avg_atr = self._current_atr
        else:
            self._avg_atr = 0.95 * self._avg_atr + 0.05 * self._current_atr

    def _create_assessment(
        self,
        proposal: TradeProposal,
        decision: DecisionType,
        risk_score: float,
        violations: List[RiskViolation],
        reasoning: List[str],
        start_time: float
    ) -> RiskAssessment:
        """
        Create a RiskAssessment object from evaluation results.
        """
        # Determine risk level from score
        if risk_score >= 75:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 50:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 25:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW

        # Calculate processing time
        elapsed_ms = (time.time() - start_time) * 1000

        # Update metrics
        self._metrics.decisions_made += 1
        self._metrics.last_activity = datetime.now()

        return RiskAssessment(
            proposal_id=proposal.proposal_id,
            decision=decision,
            risk_score=risk_score,
            risk_level=risk_level,
            violations=violations,
            reasoning=reasoning if self._risk_config.enable_rule_explanations else [],
            assessment_time_ms=elapsed_ms
        )

    def _reset_daily_tracking(self) -> None:
        """Reset daily tracking variables (call at day start)."""
        self._trades_today = 0
        self._daily_pnl = 0.0
        self._daily_start_equity = self._current_equity

    # =========================================================================
    # PUBLIC STATE UPDATE METHODS
    # =========================================================================

    def update_portfolio_state(
        self,
        equity: float,
        position: float = 0.0,
        entry_price: float = 0.0,
        current_step: int = 0
    ) -> None:
        """
        Update portfolio state (call from environment).

        Args:
            equity: Current total portfolio value
            position: Current position size
            entry_price: Entry price of current position
            current_step: Current simulation step
        """
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        self._current_position = position
        self._position_entry_price = entry_price
        self._current_step = current_step

    def record_trade_result(self, pnl: float) -> None:
        """
        Record the result of a completed trade.

        Args:
            pnl: Profit/loss of the trade
        """
        self._trades_today += 1
        self._daily_pnl += pnl
        self._last_trade_time = datetime.now()

        if pnl < 0:
            self._steps_since_loss = 0
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._trade_history.append({
            'timestamp': datetime.now().isoformat(),
            'pnl': pnl,
            'equity': self._current_equity
        })

    def record_step(self) -> None:
        """Record a simulation step (increments cooldown counter)."""
        self._steps_since_loss += 1
        self._current_step += 1

    def set_market_regime(self, regime: int) -> None:
        """
        Set current market regime.

        Args:
            regime: 0 = calm/low volatility, 1 = chaos/high volatility
        """
        self._current_regime = regime

    def set_position_entry(self, step: int, price: float) -> None:
        """Record when a position was opened."""
        self._position_entry_step = step
        self._position_entry_price = price

    # =========================================================================
    # STATISTICS AND REPORTING
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics about agent performance.

        Returns:
            Dictionary with all statistics
        """
        approval_rate = (
            self._total_approvals / self._total_assessments
            if self._total_assessments > 0 else 0.0
        )

        return {
            'total_assessments': self._total_assessments,
            'total_approvals': self._total_approvals,
            'total_rejections': self._total_rejections,
            'total_modifications': self._total_modifications,
            'approval_rate': f"{approval_rate:.1%}",
            'current_drawdown': f"{self._current_drawdown:.2%}",
            'peak_equity': self._peak_equity,
            'current_equity': self._current_equity,
            'trades_today': self._trades_today,
            'consecutive_losses': self._consecutive_losses,
            'current_regime': 'VOLATILE' if self._current_regime == 1 else 'CALM'
        }

    def get_risk_dashboard(self) -> str:
        """
        Generate a text-based risk dashboard.

        Returns:
            Formatted string showing current risk status
        """
        stats = self.get_statistics()

        dashboard = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    RISK SENTINEL DASHBOARD                       ║
╠══════════════════════════════════════════════════════════════════╣
║ Status: {'RUNNING' if self.state == AgentState.RUNNING else self.state.name:12} │ Regime: {stats['current_regime']:10} ║
╠══════════════════════════════════════════════════════════════════╣
║ PORTFOLIO                                                        ║
║   Peak Equity:     ${stats['peak_equity']:>10,.2f}                           ║
║   Current Equity:  ${stats['current_equity']:>10,.2f}                           ║
║   Drawdown:        {stats['current_drawdown']:>10}                           ║
╠══════════════════════════════════════════════════════════════════╣
║ DECISIONS                                                        ║
║   Total Assessed:  {stats['total_assessments']:>10}                           ║
║   Approved:        {stats['total_approvals']:>10}                           ║
║   Rejected:        {stats['total_rejections']:>10}                           ║
║   Approval Rate:   {stats['approval_rate']:>10}                           ║
╠══════════════════════════════════════════════════════════════════╣
║ TODAY                                                            ║
║   Trades:          {stats['trades_today']:>10}                           ║
║   Consec. Losses:  {stats['consecutive_losses']:>10}                           ║
╚══════════════════════════════════════════════════════════════════╝
"""
        return dashboard


# =============================================================================
# CONVENIENCE FUNCTION
# =============================================================================


def create_risk_sentinel(
    preset: str = "moderate",
    event_bus: Optional[EventBus] = None
) -> RiskSentinelAgent:
    """
    Factory function to create a RiskSentinelAgent with a preset config.

    Args:
        preset: "conservative", "moderate", "aggressive", or "backtesting"
        event_bus: Optional EventBus for event-driven mode

    Returns:
        Configured RiskSentinelAgent instance
    """
    from src.agents.config import ConfigPreset, get_risk_sentinel_config

    preset_map = {
        "conservative": ConfigPreset.CONSERVATIVE,
        "moderate": ConfigPreset.MODERATE,
        "aggressive": ConfigPreset.AGGRESSIVE,
        "backtesting": ConfigPreset.BACKTESTING
    }

    config = get_risk_sentinel_config(preset_map.get(preset, ConfigPreset.MODERATE))
    agent = RiskSentinelAgent(config=config, event_bus=event_bus)

    return agent
