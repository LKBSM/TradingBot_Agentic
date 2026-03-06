# =============================================================================
# TRADING ORCHESTRATOR - Centralized Multi-Agent Coordination
# =============================================================================
# This module implements the central coordinator for all trading agents.
# It provides:
#   1. Agent registration and lifecycle management
#   2. Hierarchical decision coordination with priority rules
#   3. Position size aggregation (most conservative)
#   4. Health monitoring and fallback logic
#
# Priority Rules:
#   CRITICAL (News BLOCK) > HIGH (Risk REJECT) > NORMAL (Analysis)
#
# Decision Flow:
#   1. Collect assessments from all registered agents
#   2. Apply priority rules (highest priority wins on conflicts)
#   3. Aggregate position size recommendations (use minimum)
#   4. Generate final OrchestratedDecision with full audit trail
#
# =============================================================================

from typing import Dict, Any, Optional, List, Tuple, Set, Protocol, runtime_checkable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import time
import uuid
from collections import defaultdict
from threading import Lock, RLock, Thread
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.agents.base_agent import BaseAgent, AgentCapability, AgentState
from src.agents.events import (
    AgentEvent, EventType, EventBus, DecisionType,
    TradeProposal, RiskAssessment
)
from src.agents.config import AgentConfig


# =============================================================================
# AGENT QUERY PROTOCOLS - Formal interfaces for agent capabilities
# =============================================================================
# Using Protocol (PEP 544) for structural subtyping: agents don't need to
# explicitly inherit from these protocols, they just need to implement
# the required methods. This is checked at registration time.

@runtime_checkable
class NewsEvaluator(Protocol):
    """Protocol for agents that can evaluate news impact on trades."""
    def evaluate_news_impact(self, proposal: TradeProposal) -> Any: ...

@runtime_checkable
class TradeEvaluator(Protocol):
    """Protocol for agents that can evaluate trade proposals for risk."""
    def evaluate_trade(self, proposal: TradeProposal) -> Any: ...

@runtime_checkable
class MarketAnalyzer(Protocol):
    """Protocol for agents that analyze market conditions."""
    def analyze(self, market_data: Any) -> Any: ...


# =============================================================================
# ORCHESTRATOR-SPECIFIC ENUMS AND DATA CLASSES
# =============================================================================


class AgentPriority(Enum):
    """
    Priority levels for agent decision coordination.

    Higher priority agents can override lower priority decisions.
    """
    CRITICAL = 1    # News blocking - highest priority, can halt all trading
    HIGH = 2        # Risk management - can reject/modify trades
    NORMAL = 3      # Market analysis - provides context
    LOW = 4         # Advisory only - does not affect decisions


@dataclass
class AgentRegistration:
    """
    Registration information for an agent in the orchestrator.

    Attributes:
        agent: The agent instance
        priority: Decision priority level
        capabilities: What the agent can do
        is_critical: If True, failure halts trading
        fallback_decision: Decision to use if agent fails
    """
    agent: BaseAgent
    priority: AgentPriority
    capabilities: List[AgentCapability] = field(default_factory=list)
    is_critical: bool = False
    fallback_decision: DecisionType = DecisionType.REJECT
    registered_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'agent_id': self.agent.full_id,
            'agent_name': self.agent.name,
            'priority': self.priority.name,
            'capabilities': [c.name for c in self.capabilities],
            'is_critical': self.is_critical,
            'fallback_decision': self.fallback_decision.name,
            'state': self.agent.state.name,
            'registered_at': self.registered_at.isoformat()
        }


@dataclass
class OrchestratedDecision:
    """
    Final coordinated decision from all agents.

    This is what the environment uses to execute trades.

    Attributes:
        final_decision: APPROVE, REJECT, or MODIFY
        final_position_size: Adjusted position size
        confidence: Aggregate confidence (0-1)
        agent_decisions: Per-agent decision breakdown
        agent_assessments: Full assessment from each agent
        reasoning: Human-readable explanation
        blocking_agent: Which agent blocked (if any)
    """
    final_decision: DecisionType          # APPROVE, REJECT, MODIFY
    final_position_size: float            # Adjusted position size
    original_position_size: float         # Original proposed size
    confidence: float                     # Aggregate confidence (0-1)

    # Per-agent breakdown (for debugging/compliance)
    agent_decisions: Dict[str, str] = field(default_factory=dict)
    agent_assessments: Dict[str, Any] = field(default_factory=dict)

    # Decision reasoning
    reasoning: List[str] = field(default_factory=list)
    blocking_agent: Optional[str] = None  # Which agent blocked (if any)
    modifying_agents: List[str] = field(default_factory=list)

    # Position sizing breakdown
    position_multipliers: Dict[str, float] = field(default_factory=dict)

    # Metadata
    orchestration_time_ms: float = 0.0
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'decision_id': self.decision_id,
            'final_decision': self.final_decision.name,
            'final_position_size': round(self.final_position_size, 6),
            'original_position_size': round(self.original_position_size, 6),
            'position_change': round(
                (self.final_position_size / self.original_position_size - 1) * 100, 1
            ) if self.original_position_size > 0 else 0,
            'confidence': round(self.confidence, 3),
            'agent_decisions': self.agent_decisions,
            'reasoning': self.reasoning,
            'blocking_agent': self.blocking_agent,
            'modifying_agents': self.modifying_agents,
            'position_multipliers': {
                k: round(v, 3) for k, v in self.position_multipliers.items()
            },
            'orchestration_time_ms': round(self.orchestration_time_ms, 2),
            'timestamp': self.timestamp.isoformat()
        }

    def is_approved(self) -> bool:
        """Check if trade was approved (with or without modifications)."""
        return self.final_decision in [DecisionType.APPROVE, DecisionType.MODIFY]

    def get_summary(self) -> str:
        """Get a one-line summary."""
        icon = {
            DecisionType.APPROVE: "OK",
            DecisionType.REJECT: "XX",
            DecisionType.MODIFY: "~~",
            DecisionType.DEFER: "??"
        }
        size_change = (
            (self.final_position_size / self.original_position_size - 1) * 100
            if self.original_position_size > 0 else 0
        )
        return (
            f"[{icon.get(self.final_decision, '??')}] "
            f"{self.final_decision.name} | "
            f"Size: {size_change:+.0f}% | "
            f"Agents: {len(self.agent_decisions)} | "
            f"Blocked by: {self.blocking_agent or 'None'}"
        )


@dataclass
class OrchestratorConfig(AgentConfig):
    """Configuration for TradingOrchestrator."""

    # === DECISION RULES ===
    require_unanimous_approval: bool = False    # All agents must approve
    allow_modified_trades: bool = True          # Accept MODIFY decisions

    # === POSITION SIZING ===
    position_aggregation: str = "minimum"       # "minimum", "average", "weighted"
    min_position_multiplier: float = 0.1        # Floor for position size
    max_position_multiplier: float = 1.5        # Ceiling for position size

    # === HEALTH MONITORING ===
    heartbeat_interval_sec: int = 30
    agent_timeout_sec: int = 60
    auto_disable_failed_agents: bool = True

    # === SAFETY ===
    halt_on_critical_failure: bool = True       # Stop trading if critical agent fails
    require_news_agent: bool = True             # Trading disabled without news agent
    require_risk_agent: bool = True             # Trading disabled without risk agent

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        base = super().to_dict()
        base.update({
            'require_unanimous_approval': self.require_unanimous_approval,
            'allow_modified_trades': self.allow_modified_trades,
            'position_aggregation': self.position_aggregation,
            'min_position_multiplier': self.min_position_multiplier,
            'max_position_multiplier': self.max_position_multiplier,
            'heartbeat_interval_sec': self.heartbeat_interval_sec,
            'agent_timeout_sec': self.agent_timeout_sec,
            'auto_disable_failed_agents': self.auto_disable_failed_agents,
            'halt_on_critical_failure': self.halt_on_critical_failure,
            'require_news_agent': self.require_news_agent,
            'require_risk_agent': self.require_risk_agent,
        })
        return base


# =============================================================================
# TRADING ORCHESTRATOR
# =============================================================================


class TradingOrchestrator:
    """
    Centralized coordinator for all trading agents.

    Responsibilities:
        1. Agent lifecycle management (register, start, stop)
        2. Decision coordination (collect, prioritize, aggregate)
        3. Health monitoring (heartbeat, fallback)
        4. Event routing (via EventBus)

    Priority Rules (in order):
        1. CRITICAL agent returns BLOCK -> REJECT (safety first)
        2. HIGH agent returns REJECT -> REJECT (hard rules violated)
        3. HIGH agent returns MODIFY -> MODIFY (combine modifications)
        4. All agents approve -> APPROVE

    Position Sizing:
        Uses minimum of all agent recommendations (most conservative)
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        event_bus: Optional[EventBus] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Orchestrator configuration
            event_bus: Event bus for communication
        """
        self._config = config or OrchestratorConfig()
        self._event_bus = event_bus or EventBus()

        # === AGENT REGISTRY ===
        self._agents: Dict[str, AgentRegistration] = {}
        self._agents_by_priority: Dict[AgentPriority, List[str]] = defaultdict(list)
        self._lock = Lock()
        self._decision_lock = RLock()  # Thread-safe decision coordination

        # === STATE ===
        self._is_running: bool = False
        self._trading_enabled: bool = True
        self._global_block_reason: Optional[str] = None

        # === HEALTH MONITORING ===
        self._agent_health: Dict[str, Dict[str, Any]] = {}
        self._last_heartbeat: Dict[str, datetime] = {}
        self._failed_agents: Set[str] = set()

        # === CIRCUIT BREAKER ===
        # Track agent failures and implement circuit breaker pattern
        self._agent_failure_counts: Dict[str, int] = defaultdict(int)
        self._agent_circuit_open: Dict[str, datetime] = {}  # When circuit opened
        self._circuit_failure_threshold = 5  # Open circuit after N failures
        self._circuit_reset_timeout = timedelta(minutes=2)  # Reset after this time

        # === SHARED THREAD POOL ===
        # FIX: Use a shared ThreadPoolExecutor instead of creating one per _query_agent call.
        # This avoids thread creation/destruction overhead in high-frequency trading.
        self._query_executor = ThreadPoolExecutor(
            max_workers=self._config.max_agents if hasattr(self._config, 'max_agents') else 8,
            thread_name_prefix="orchestrator-agent-query"
        )

        # === STATISTICS ===
        self._total_decisions: int = 0
        self._decisions_by_outcome: Dict[str, int] = defaultdict(int)
        self._start_time: Optional[datetime] = None

        self._logger = logging.getLogger("orchestrator")

    # =========================================================================
    # AGENT MANAGEMENT
    # =========================================================================

    def register_agent(
        self,
        agent: BaseAgent,
        priority: AgentPriority = AgentPriority.NORMAL,
        is_critical: bool = False,
        fallback_decision: DecisionType = DecisionType.REJECT
    ) -> bool:
        """
        Register an agent with the orchestrator.

        Args:
            agent: Agent instance to register
            priority: Decision priority level
            is_critical: If True, failure halts trading
            fallback_decision: Decision when agent fails (default: REJECT = fail-safe)

        Returns:
            True if registration successful
        """
        with self._lock:
            agent_id = agent.full_id

            if agent_id in self._agents:
                self._logger.warning(f"Agent {agent_id} already registered")
                return False

            registration = AgentRegistration(
                agent=agent,
                priority=priority,
                capabilities=agent.get_capabilities(),
                is_critical=is_critical,
                fallback_decision=fallback_decision
            )

            self._agents[agent_id] = registration
            self._agents_by_priority[priority].append(agent_id)

            self._logger.info(
                f"Registered agent: {agent_id} "
                f"(priority: {priority.name}, critical: {is_critical})"
            )

            return True

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Remove an agent from the orchestrator.

        Args:
            agent_id: ID of agent to remove

        Returns:
            True if successfully removed
        """
        with self._lock:
            if agent_id not in self._agents:
                self._logger.warning(f"Agent {agent_id} not found")
                return False

            registration = self._agents.pop(agent_id)
            priority = registration.priority

            if agent_id in self._agents_by_priority[priority]:
                self._agents_by_priority[priority].remove(agent_id)

            self._logger.info(f"Unregistered agent: {agent_id}")
            return True

    def start_all(self) -> bool:
        """
        Start all registered agents.

        Returns:
            True if all agents started successfully
        """
        self._logger.info("Starting all agents...")
        self._start_time = datetime.now()
        all_started = True

        for agent_id, registration in self._agents.items():
            try:
                if registration.agent.state != AgentState.RUNNING:
                    success = registration.agent.start()
                    if not success:
                        self._logger.error(f"Failed to start agent: {agent_id}")
                        all_started = False
                        if registration.is_critical:
                            self._global_block_reason = f"Critical agent {agent_id} failed to start"
                    else:
                        self._logger.info(f"Started agent: {agent_id}")
            except Exception as e:
                self._logger.error(f"Error starting agent {agent_id}: {e}")
                all_started = False

        self._is_running = all_started
        return all_started

    def stop_all(self) -> bool:
        """
        Stop all registered agents gracefully.

        Returns:
            True if all agents stopped successfully
        """
        self._logger.info("Stopping all agents...")
        all_stopped = True

        for agent_id, registration in self._agents.items():
            try:
                if registration.agent.state == AgentState.RUNNING:
                    success = registration.agent.stop()
                    if not success:
                        self._logger.warning(f"Agent {agent_id} did not stop cleanly")
                        all_stopped = False
                    else:
                        self._logger.info(f"Stopped agent: {agent_id}")
            except Exception as e:
                self._logger.error(f"Error stopping agent {agent_id}: {e}")
                all_stopped = False

        self._is_running = False

        # FIX: Shutdown the shared thread pool executor on stop
        try:
            self._query_executor.shutdown(wait=True, cancel_futures=False)
        except Exception as e:
            self._logger.error(f"Error shutting down query executor: {e}")

        return all_stopped

    # =========================================================================
    # CIRCUIT BREAKER
    # =========================================================================

    def _is_circuit_open(self, agent_id: str) -> bool:
        """
        Check if circuit breaker is open for an agent.

        THREAD SAFETY FIX: Now uses lock to protect shared state.
        """
        with self._lock:
            if agent_id not in self._agent_circuit_open:
                return False

            # Check if timeout has passed
            opened_at = self._agent_circuit_open[agent_id]
            if datetime.now() - opened_at >= self._circuit_reset_timeout:
                # Allow one probe request (half-open state)
                self._logger.info(f"Circuit breaker half-open for {agent_id}, allowing probe")
                return False

            return True

    def _record_agent_success(self, agent_id: str) -> None:
        """
        Record successful agent query, reset circuit breaker.

        THREAD SAFETY FIX: Now uses lock to protect shared state.
        """
        with self._lock:
            self._agent_failure_counts[agent_id] = 0
            if agent_id in self._agent_circuit_open:
                del self._agent_circuit_open[agent_id]
                self._logger.info(f"Circuit breaker closed for {agent_id}")
            # Remove from failed agents if it was there
            self._failed_agents.discard(agent_id)

    def _record_agent_failure(self, agent_id: str) -> None:
        """
        Record agent query failure, potentially open circuit breaker.

        THREAD SAFETY FIX: Now uses lock to protect shared state.
        """
        with self._lock:
            self._agent_failure_counts[agent_id] += 1

            if self._agent_failure_counts[agent_id] >= self._circuit_failure_threshold:
                self._agent_circuit_open[agent_id] = datetime.now()
                self._failed_agents.add(agent_id)  # SECURITY FIX: Also add to failed agents
                self._logger.warning(
                    f"Circuit breaker OPEN for {agent_id} after "
                    f"{self._agent_failure_counts[agent_id]} failures"
                )

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers."""
        now = datetime.now()
        status = {}
        for agent_id in self._agents:
            failures = self._agent_failure_counts.get(agent_id, 0)
            is_open = agent_id in self._agent_circuit_open
            time_until_reset = None
            if is_open:
                opened_at = self._agent_circuit_open[agent_id]
                remaining = self._circuit_reset_timeout - (now - opened_at)
                time_until_reset = max(0, remaining.total_seconds())

            status[agent_id] = {
                'failures': failures,
                'circuit_open': is_open,
                'time_until_reset_sec': time_until_reset
            }
        return status

    # =========================================================================
    # DECISION COORDINATION
    # =========================================================================

    def coordinate_decision(
        self,
        proposal: TradeProposal,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestratedDecision:
        """
        Coordinate decision across all agents.

        This is the main entry point for trade evaluation.
        Thread-safe: Uses RLock to prevent race conditions.

        Process:
            1. Query all agents in priority order
            2. Apply blocking rules (highest priority wins)
            3. Aggregate position size recommendations
            4. Generate final decision with reasoning

        Args:
            proposal: Trade proposal to evaluate
            context: Optional additional context

        Returns:
            OrchestratedDecision with final verdict
        """
        with self._decision_lock:
            return self._coordinate_decision_internal(proposal, context)

    def _coordinate_decision_internal(
        self,
        proposal: TradeProposal,
        context: Optional[Dict[str, Any]] = None
    ) -> OrchestratedDecision:
        """Internal decision coordination (must be called with lock held)."""
        start_time = time.time()
        self._total_decisions += 1

        reasoning: List[str] = []
        agent_decisions: Dict[str, str] = {}
        agent_assessments: Dict[str, Any] = {}
        position_multipliers: Dict[str, float] = {}
        modifying_agents: List[str] = []
        blocking_agent: Optional[str] = None

        original_size = proposal.quantity

        # --- Check if trading is enabled ---
        if not self._trading_enabled:
            reasoning.append(f"Trading disabled: {self._global_block_reason or 'Unknown reason'}")
            return self._create_decision(
                DecisionType.REJECT,
                0.0, original_size,
                0.0,
                agent_decisions, agent_assessments,
                reasoning, "orchestrator",
                modifying_agents, position_multipliers,
                start_time
            )

        # --- Collect assessments from all agents (by priority) ---
        for priority in sorted(AgentPriority, key=lambda x: x.value):
            agent_ids = self._agents_by_priority.get(priority, [])

            for agent_id in agent_ids:
                if agent_id in self._failed_agents:
                    continue

                # CIRCUIT BREAKER: Skip if circuit is open
                if self._is_circuit_open(agent_id):
                    reasoning.append(f"Agent {agent_id} skipped (circuit breaker open)")
                    continue

                registration = self._agents.get(agent_id)
                if not registration:
                    continue

                try:
                    assessment = self._query_agent(registration, proposal)

                    if assessment:
                        # Record success for circuit breaker
                        self._record_agent_success(agent_id)
                        agent_assessments[agent_id] = assessment

                        # Extract decision and multiplier from assessment
                        decision, multiplier = self._extract_decision(assessment)
                        agent_decisions[agent_id] = decision.name
                        position_multipliers[agent_id] = multiplier

                        # Check for blocking (CRITICAL priority)
                        if priority == AgentPriority.CRITICAL:
                            if decision == DecisionType.REJECT:
                                blocking_agent = agent_id
                                reasoning.append(
                                    f"BLOCKED by {agent_id}: {self._get_reason(assessment)}"
                                )
                                # Immediate return for critical blocks
                                return self._create_decision(
                                    DecisionType.REJECT,
                                    0.0, original_size,
                                    0.0,
                                    agent_decisions, agent_assessments,
                                    reasoning, blocking_agent,
                                    modifying_agents, position_multipliers,
                                    start_time
                                )

                        # Check for rejection (HIGH priority)
                        if priority == AgentPriority.HIGH:
                            if decision == DecisionType.REJECT:
                                blocking_agent = agent_id
                                reasoning.append(
                                    f"REJECTED by {agent_id}: {self._get_reason(assessment)}"
                                )
                            elif decision == DecisionType.MODIFY:
                                modifying_agents.append(agent_id)
                                reasoning.append(
                                    f"MODIFIED by {agent_id}: size={multiplier:.0%}"
                                )

                except Exception as e:
                    self._logger.error(f"Error querying agent {agent_id}: {e}")
                    # Record failure for circuit breaker
                    self._record_agent_failure(agent_id)

                    if registration.is_critical:
                        # THREAD SAFETY FIX: Use lock when modifying shared state
                        with self._lock:
                            self._failed_agents.add(agent_id)
                        if self._config.halt_on_critical_failure:
                            blocking_agent = agent_id
                            reasoning.append(f"CRITICAL agent {agent_id} failed: {e}")

        # --- Check for rejection from HIGH priority agents ---
        if blocking_agent:
            self._decisions_by_outcome['REJECT'] += 1
            return self._create_decision(
                DecisionType.REJECT,
                0.0, original_size,
                0.0,
                agent_decisions, agent_assessments,
                reasoning, blocking_agent,
                modifying_agents, position_multipliers,
                start_time
            )

        # --- Aggregate position sizes ---
        final_multiplier = self._aggregate_position_sizes(position_multipliers)

        # Clamp to valid range BEFORE calculating final size
        final_multiplier = max(
            self._config.min_position_multiplier,
            min(self._config.max_position_multiplier, final_multiplier)
        )
        final_size = original_size * final_multiplier

        # --- Determine final decision ---
        if modifying_agents:
            final_decision = DecisionType.MODIFY
            self._decisions_by_outcome['MODIFY'] += 1
            reasoning.append(
                f"Position adjusted: {original_size:.4f} -> {final_size:.4f} "
                f"({final_multiplier:.0%})"
            )
        else:
            final_decision = DecisionType.APPROVE
            self._decisions_by_outcome['APPROVE'] += 1
            if not reasoning:
                reasoning.append("All agents approved")

        # Calculate confidence
        confidence = self._calculate_confidence(agent_assessments)

        return self._create_decision(
            final_decision,
            final_size, original_size,
            confidence,
            agent_decisions, agent_assessments,
            reasoning, blocking_agent,
            modifying_agents, position_multipliers,
            start_time
        )

    def _query_agent(
        self,
        registration: AgentRegistration,
        proposal: TradeProposal
    ) -> Optional[Dict[str, Any]]:
        """
        Query an agent for its assessment with timeout protection.

        Handles different agent types (news, risk, regime).
        Uses timeout to prevent hung agents from blocking decisions.
        """
        agent = registration.agent

        # Thread-safe state check using agent's state lock if available
        try:
            if hasattr(agent, '_state_lock'):
                with agent._state_lock:
                    current_state = agent.state
            else:
                current_state = agent.state

            if current_state != AgentState.RUNNING:
                self._logger.warning(f"Agent {agent.full_id} not running (state: {current_state})")
                return None
        except Exception as e:
            self._logger.error(f"Error checking agent state: {e}")
            return None

        # Execute query with timeout
        timeout_sec = self._config.agent_timeout_sec

        def _do_query():
            # Use Protocol-based dispatch for type-safe agent querying.
            # Each agent is checked against formal Protocols defined at module
            # level, replacing fragile hasattr() duck-typing.
            if isinstance(agent, NewsEvaluator):
                assessment = agent.evaluate_news_impact(proposal)
                return assessment.to_dict() if hasattr(assessment, 'to_dict') else assessment

            if isinstance(agent, TradeEvaluator):
                assessment = agent.evaluate_trade(proposal)
                return assessment.to_dict() if hasattr(assessment, 'to_dict') else assessment

            if isinstance(agent, MarketAnalyzer):
                # Regime agent needs market data, not trade proposal
                return None

            self._logger.warning(
                f"Agent {agent.full_id} does not implement any known evaluator protocol "
                f"(NewsEvaluator, TradeEvaluator, MarketAnalyzer). Skipping."
            )
            return None

        try:
            future = self._query_executor.submit(_do_query)
            return future.result(timeout=timeout_sec)
        except FuturesTimeoutError:
            self._logger.error(
                f"Agent {agent.full_id} query timed out after {timeout_sec}s"
            )
            return None
        except Exception as e:
            self._logger.error(f"Agent {agent.full_id} query failed: {e}")
            return None

    def _extract_decision(
        self,
        assessment: Dict[str, Any]
    ) -> Tuple[DecisionType, float]:
        """
        Extract decision type and position multiplier from assessment.

        Handles different assessment formats (news, risk, regime).
        """
        # Try to get decision
        decision_str = assessment.get('decision', 'APPROVE')

        # Handle NewsAssessment format
        if decision_str in ['block', 'BLOCK']:
            return DecisionType.REJECT, 0.0
        elif decision_str in ['reduce', 'REDUCE']:
            multiplier = assessment.get('position_multiplier', 0.5)
            return DecisionType.MODIFY, multiplier
        elif decision_str in ['allow', 'ALLOW']:
            multiplier = assessment.get('position_multiplier', 1.0)
            return DecisionType.APPROVE, multiplier

        # Handle RiskAssessment format
        if decision_str in ['REJECT', 'reject']:
            return DecisionType.REJECT, 0.0
        elif decision_str in ['MODIFY', 'modify']:
            # Get modified size from modified_params
            modified = assessment.get('modified_params', {})
            if 'position_size' in modified:
                # Calculate multiplier (need original size context)
                return DecisionType.MODIFY, 0.5  # Default reduction
            return DecisionType.MODIFY, 0.8

        # Default: approved
        return DecisionType.APPROVE, 1.0

    def _get_reason(self, assessment: Dict[str, Any]) -> str:
        """Extract human-readable reason from assessment."""
        # Try different formats
        if 'reasoning' in assessment:
            reasons = assessment['reasoning']
            if isinstance(reasons, list) and reasons:
                return reasons[0]
            return str(reasons)

        if 'violations' in assessment:
            violations = assessment['violations']
            if violations:
                return violations[0].get('rule_description', 'Rule violation')

        return "No specific reason"

    def _aggregate_position_sizes(
        self,
        multipliers: Dict[str, float]
    ) -> float:
        """
        Aggregate position size multipliers from all agents.

        Strategy determined by config:
            - "minimum": Use smallest multiplier (most conservative)
            - "average": Use average of all multipliers
            - "weighted": Weight by agent priority
        """
        if not multipliers:
            return 1.0

        values = list(multipliers.values())

        if self._config.position_aggregation == "minimum":
            return min(values)
        elif self._config.position_aggregation == "average":
            return sum(values) / len(values)
        elif self._config.position_aggregation == "weighted":
            # Weight by priority (lower priority number = higher weight)
            weighted_sum = 0.0
            total_weight = 0.0
            for agent_id, mult in multipliers.items():
                reg = self._agents.get(agent_id)
                if reg:
                    weight = 1.0 / reg.priority.value
                    weighted_sum += mult * weight
                    total_weight += weight
            return weighted_sum / total_weight if total_weight > 0 else 1.0

        return min(values)  # Default to minimum

    def _calculate_confidence(self, assessments: Dict[str, Any]) -> float:
        """Calculate aggregate confidence from all assessments."""
        confidences = []

        for assessment in assessments.values():
            if 'confidence' in assessment:
                confidences.append(assessment['confidence'])
            elif 'sentiment_confidence' in assessment:
                confidences.append(assessment['sentiment_confidence'])
            elif 'risk_score' in assessment:
                # Convert risk score to confidence (inverse)
                confidences.append(1.0 - assessment['risk_score'] / 100)

        if not confidences:
            return 0.5

        return sum(confidences) / len(confidences)

    def _create_decision(
        self,
        decision: DecisionType,
        final_size: float,
        original_size: float,
        confidence: float,
        agent_decisions: Dict[str, str],
        agent_assessments: Dict[str, Any],
        reasoning: List[str],
        blocking_agent: Optional[str],
        modifying_agents: List[str],
        position_multipliers: Dict[str, float],
        start_time: float
    ) -> OrchestratedDecision:
        """Create an OrchestratedDecision object."""
        elapsed_ms = (time.time() - start_time) * 1000

        return OrchestratedDecision(
            final_decision=decision,
            final_position_size=final_size,
            original_position_size=original_size,
            confidence=confidence,
            agent_decisions=agent_decisions,
            agent_assessments=agent_assessments,
            reasoning=reasoning,
            blocking_agent=blocking_agent,
            modifying_agents=modifying_agents,
            position_multipliers=position_multipliers,
            orchestration_time_ms=elapsed_ms
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def enable_trading(self) -> None:
        """Enable trading (after manual review)."""
        self._trading_enabled = True
        self._global_block_reason = None
        self._logger.info("Trading enabled")

    def disable_trading(self, reason: str) -> None:
        """Disable all trading (emergency stop)."""
        self._trading_enabled = False
        self._global_block_reason = reason
        self._logger.warning(f"Trading disabled: {reason}")

    def is_trading_enabled(self) -> bool:
        """Check if trading is currently enabled."""
        return self._trading_enabled

    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID."""
        reg = self._agents.get(agent_id)
        return reg.agent if reg else None

    def get_all_agents(self) -> Dict[str, AgentRegistration]:
        """Get all registered agents."""
        return self._agents.copy()

    def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status and agent health."""
        agent_statuses = {}
        for agent_id, reg in self._agents.items():
            agent_statuses[agent_id] = {
                'state': reg.agent.state.name,
                'priority': reg.priority.name,
                'is_critical': reg.is_critical,
                'is_failed': agent_id in self._failed_agents
            }

        return {
            'is_running': self._is_running,
            'trading_enabled': self._trading_enabled,
            'global_block_reason': self._global_block_reason,
            'total_agents': len(self._agents),
            'failed_agents': list(self._failed_agents),
            'agents': agent_statuses,
            'uptime_seconds': (
                (datetime.now() - self._start_time).total_seconds()
                if self._start_time else 0
            )
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get decision statistics."""
        total = self._total_decisions or 1
        return {
            'total_decisions': self._total_decisions,
            'approvals': self._decisions_by_outcome.get('APPROVE', 0),
            'rejections': self._decisions_by_outcome.get('REJECT', 0),
            'modifications': self._decisions_by_outcome.get('MODIFY', 0),
            'approval_rate': f"{self._decisions_by_outcome.get('APPROVE', 0) / total * 100:.1f}%",
            'rejection_rate': f"{self._decisions_by_outcome.get('REJECT', 0) / total * 100:.1f}%",
            'modification_rate': f"{self._decisions_by_outcome.get('MODIFY', 0) / total * 100:.1f}%"
        }

    def get_dashboard(self) -> str:
        """Get a text-based dashboard for monitoring."""
        status = self.get_status()
        stats = self.get_statistics()

        trading_status = "ENABLED" if self._trading_enabled else "DISABLED"

        agent_lines = []
        for agent_id, info in status.get('agents', {}).items():
            state_icon = "OK" if info['state'] == 'RUNNING' else "!!"
            crit_icon = "*" if info['is_critical'] else " "
            agent_lines.append(
                f"   [{state_icon}]{crit_icon} {agent_id[:30]:30} | "
                f"{info['priority']:8} | {info['state']}"
            )

        agents_str = "\n".join(agent_lines) if agent_lines else "   No agents registered"

        return f"""
================================================================================
                        TRADING ORCHESTRATOR DASHBOARD
================================================================================
 Status:           {'RUNNING' if status['is_running'] else 'STOPPED':12}
 Trading:          {trading_status}
 Block Reason:     {status['global_block_reason'] or 'None'}

 Statistics:
   Total Decisions: {stats['total_decisions']:>10}
   Approved:        {stats['approvals']:>10} ({stats['approval_rate']})
   Rejected:        {stats['rejections']:>10} ({stats['rejection_rate']})
   Modified:        {stats['modifications']:>10} ({stats['modification_rate']})

 Registered Agents ({status['total_agents']}):
{agents_str}

 Failed Agents:    {', '.join(status['failed_agents']) or 'None'}
================================================================================
"""


    # =========================================================================
    # SPRINT 2: INTELLIGENCE INTEGRATION
    # =========================================================================

    def get_intelligence_report(
        self,
        market_data: Optional[Dict[str, Any]] = None,
        news_texts: Optional[List[str]] = None
    ) -> Optional['IntelligenceReportEvent']:
        """
        Get unified intelligence report from Sprint 2 components.

        This method queries all Sprint 2 intelligence components and
        produces an aggregated IntelligenceReportEvent.

        Args:
            market_data: Optional market data for regime/timeframe analysis
            news_texts: Optional news texts for sentiment analysis

        Returns:
            IntelligenceReportEvent with aggregated intelligence
        """
        from src.agents.events import (
            IntelligenceReportEvent,
            SentimentEvent,
            RegimeEvent,
            TimeframeAlignmentEvent,
            EnsemblePredictionEvent,
            RiskLevel
        )

        try:
            report = IntelligenceReportEvent()

            # Query Sprint 2 components if available
            sentiment = self._get_sentiment_signal(news_texts)
            regime = self._get_regime_signal(market_data)
            timeframe = self._get_timeframe_signal(market_data)
            ensemble = self._get_ensemble_signal(market_data)

            report.sentiment = sentiment
            report.regime = regime
            report.timeframe = timeframe
            report.ensemble = ensemble

            # Aggregate signals
            signals = []
            confidences = []
            multipliers = []

            if sentiment:
                signals.append(1 if sentiment.score > 0.1 else (-1 if sentiment.score < -0.1 else 0))
                confidences.append(sentiment.confidence)

            if regime:
                multipliers.append(regime.position_multiplier)
                confidences.append(regime.regime_confidence)

            if timeframe:
                if timeframe.signal == "BUY":
                    signals.append(1)
                elif timeframe.signal == "SELL":
                    signals.append(-1)
                else:
                    signals.append(0)
                confidences.append(timeframe.confidence)
                multipliers.append(timeframe.position_multiplier)

            if ensemble:
                multipliers.append(ensemble.position_multiplier)
                confidences.append(ensemble.confidence)

            # Calculate overall signal
            if signals:
                avg_signal = sum(signals) / len(signals)
                if avg_signal > 0.3:
                    report.overall_signal = "BUY"
                elif avg_signal < -0.3:
                    report.overall_signal = "SELL"
                else:
                    report.overall_signal = "HOLD"

            # Calculate overall confidence
            if confidences:
                report.overall_confidence = sum(confidences) / len(confidences)

            # Calculate position multiplier
            if multipliers:
                report.position_multiplier = min(multipliers)
                report.recommended_position_pct = report.position_multiplier * 100

            # Determine if we should trade
            report.should_trade = (
                report.overall_signal != "HOLD" and
                report.overall_confidence > 0.5 and
                report.position_multiplier > 0.3
            )

            # Add reasoning
            if sentiment and sentiment.is_significant:
                report.reasoning.append(f"Sentiment: {sentiment.label} ({sentiment.score:.2f})")
            if regime:
                report.reasoning.append(f"Regime: {regime.current_regime}")
            if timeframe and timeframe.has_conflict:
                report.reasoning.append(f"Timeframe conflict: {timeframe.conflict_description}")
            if ensemble:
                report.reasoning.append(f"Ensemble risk: {ensemble.risk_category}")

            return report

        except Exception as e:
            self._logger.error(f"Error generating intelligence report: {e}")
            return None

    def _get_sentiment_signal(
        self,
        news_texts: Optional[List[str]]
    ) -> Optional['SentimentEvent']:
        """Get sentiment signal from Sprint 2 sentiment analyzer."""
        from src.agents.events import SentimentEvent

        try:
            from src.agents import create_sentiment_analyzer
            if create_sentiment_analyzer is None:
                return None

            if not news_texts:
                return None

            # FIX: Cache analyzer instance to avoid recreation on every call.
            # Analyzers are stateless so a single instance is safe to reuse.
            if not hasattr(self, '_cached_sentiment_analyzer') or self._cached_sentiment_analyzer is None:
                self._cached_sentiment_analyzer = create_sentiment_analyzer()
            analyzer = self._cached_sentiment_analyzer
            results = [analyzer.analyze(text) for text in news_texts]

            if not results:
                return None

            avg_score = sum(r.score for r in results) / len(results)
            avg_confidence = sum(r.confidence for r in results) / len(results)

            label = "BULLISH" if avg_score > 0.1 else ("BEARISH" if avg_score < -0.1 else "NEUTRAL")

            return SentimentEvent(
                score=avg_score,
                confidence=avg_confidence,
                label=label,
                source_count=len(results),
                is_significant=abs(avg_score) > 0.3
            )
        except Exception as e:
            self._logger.debug(f"Sentiment analysis not available: {e}")
            return None

    def _get_regime_signal(
        self,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional['RegimeEvent']:
        """Get regime signal from Sprint 2 regime predictor."""
        from src.agents.events import RegimeEvent, RiskLevel

        try:
            from src.agents import create_regime_predictor
            if create_regime_predictor is None:
                return None

            if not market_data or 'prices' not in market_data:
                return None

            # FIX: Cache predictor instance to avoid recreation on every call.
            if not hasattr(self, '_cached_regime_predictor') or self._cached_regime_predictor is None:
                self._cached_regime_predictor = create_regime_predictor()
            predictor = self._cached_regime_predictor
            prices = market_data.get('prices', [])
            volumes = market_data.get('volumes', [1000] * len(prices))

            for price, vol in zip(prices[-100:], volumes[-100:]):
                predictor.update(price, vol)

            prediction = predictor.predict()
            if prediction is None:
                return None

            return RegimeEvent(
                current_regime=prediction.current_regime.name if hasattr(prediction.current_regime, 'name') else str(prediction.current_regime),
                regime_confidence=prediction.confidence,
                predicted_regime=prediction.predicted_regime.name if hasattr(prediction.predicted_regime, 'name') else str(prediction.predicted_regime),
                transition_probability=prediction.transition_probability,
                position_multiplier=prediction.position_multiplier if hasattr(prediction, 'position_multiplier') else 1.0,
                risk_level=RiskLevel.HIGH if prediction.confidence < 0.5 else RiskLevel.MEDIUM
            )
        except Exception as e:
            self._logger.debug(f"Regime prediction not available: {e}")
            return None

    def _get_timeframe_signal(
        self,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional['TimeframeAlignmentEvent']:
        """Get timeframe alignment signal from Sprint 2 multi-timeframe engine."""
        from src.agents.events import TimeframeAlignmentEvent

        try:
            from src.agents import create_multi_timeframe_engine
            if create_multi_timeframe_engine is None:
                return None

            if not market_data:
                return None

            # Multi-timeframe requires OHLCV data for multiple timeframes
            # For now, return None if not enough data
            return None

        except Exception as e:
            self._logger.debug(f"Multi-timeframe analysis not available: {e}")
            return None

    def _get_ensemble_signal(
        self,
        market_data: Optional[Dict[str, Any]]
    ) -> Optional['EnsemblePredictionEvent']:
        """Get ensemble model signal from Sprint 2 ensemble model."""
        from src.agents.events import EnsemblePredictionEvent

        try:
            from src.agents import create_ensemble_risk_model
            if create_ensemble_risk_model is None:
                return None

            if not market_data:
                return None

            # Ensemble model requires features
            # For now, return None if not enough data
            return None

        except Exception as e:
            self._logger.debug(f"Ensemble model not available: {e}")
            return None

    def coordinate_with_intelligence(
        self,
        proposal: TradeProposal,
        market_data: Optional[Dict[str, Any]] = None,
        news_texts: Optional[List[str]] = None
    ) -> OrchestratedDecision:
        """
        Coordinate decision with Sprint 2 intelligence enhancement.

        This extends coordinate_decision by first getting an intelligence
        report and incorporating its recommendations.

        Args:
            proposal: Trade proposal to evaluate
            market_data: Optional market data for intelligence
            news_texts: Optional news texts for sentiment

        Returns:
            OrchestratedDecision with intelligence-enhanced verdict
        """
        # Get intelligence report
        intelligence = self.get_intelligence_report(market_data, news_texts)

        # Prepare context with intelligence
        context = {'intelligence': intelligence.to_dict() if intelligence else None}

        # Get base decision
        decision = self.coordinate_decision(proposal, context)

        # Enhance with intelligence recommendations
        if intelligence:
            # Adjust position size based on intelligence
            if intelligence.position_multiplier < 1.0:
                decision.final_position_size *= intelligence.position_multiplier
                decision.position_multipliers['intelligence'] = intelligence.position_multiplier
                decision.reasoning.append(
                    f"Intelligence adjusted: x{intelligence.position_multiplier:.2f}"
                )

            # Block if intelligence suggests not to trade
            if not intelligence.should_trade and decision.is_approved():
                decision.reasoning.append(
                    f"Intelligence caution: {', '.join(intelligence.reasoning)}"
                )

        return decision


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_trading_orchestrator(
    require_news: bool = True,
    require_risk: bool = True,
    position_aggregation: str = "minimum",
    event_bus: Optional[EventBus] = None
) -> TradingOrchestrator:
    """
    Factory function to create a configured TradingOrchestrator.

    Args:
        require_news: Require news agent for trading
        require_risk: Require risk agent for trading
        position_aggregation: How to aggregate position sizes
        event_bus: Event bus for communication

    Returns:
        Configured TradingOrchestrator instance
    """
    config = OrchestratorConfig(
        require_news_agent=require_news,
        require_risk_agent=require_risk,
        position_aggregation=position_aggregation
    )

    return TradingOrchestrator(config=config, event_bus=event_bus)
