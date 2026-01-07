# =============================================================================
# AGENT EVENT SYSTEM - Inter-Agent Communication Framework
# =============================================================================
# This module implements the event-driven communication system that allows
# agents to communicate without direct coupling. Key concepts:
#
#   Event: A message with a type, payload, and metadata
#   EventBus: Central hub that routes events to subscribed agents
#   TradeProposal: A proposed trade that needs risk approval
#   RiskAssessment: The result of risk analysis on a proposal
#   AgentDecision: Final decision (APPROVE, REJECT, MODIFY)
#
# Communication Flow:
#   1. RL Agent proposes trade -> TradeProposal event
#   2. EventBus routes to RiskSentinel
#   3. RiskSentinel evaluates -> RiskAssessment event
#   4. RiskAssessment contains AgentDecision (APPROVE/REJECT/MODIFY)
#   5. Environment executes (or doesn't) based on decision
#
# Benefits:
#   - Loose coupling: Agents don't know about each other directly
#   - Scalability: Add new agents without changing existing ones
#   - Auditability: All events are logged with timestamps
#   - Testability: Easy to mock events for unit tests
#
# =============================================================================

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Union
from datetime import datetime
import uuid
import json
import logging
from threading import Lock
from collections import defaultdict

# =============================================================================
# ENUMS - Event Types and Decision Types
# =============================================================================


class EventType(Enum):
    """
    Categories of events that can flow through the system.

    Event types determine:
        1. Which agents receive the event (routing)
        2. What payload structure to expect
        3. What response is expected

    Naming Convention: DOMAIN_ACTION (e.g., TRADE_PROPOSED, RISK_ASSESSED)
    """
    # --- Trade Lifecycle Events ---
    TRADE_PROPOSED = auto()       # New trade request from RL agent
    TRADE_APPROVED = auto()       # Trade passed all checks
    TRADE_REJECTED = auto()       # Trade failed risk checks
    TRADE_MODIFIED = auto()       # Trade parameters adjusted
    TRADE_EXECUTED = auto()       # Trade was executed
    TRADE_CLOSED = auto()         # Position was closed

    # --- Risk Events ---
    RISK_ASSESSED = auto()        # Risk evaluation completed
    RISK_ALERT = auto()           # Risk threshold breached
    RISK_LIMIT_UPDATED = auto()   # Risk parameters changed
    DRAWDOWN_WARNING = auto()     # Approaching max drawdown
    DRAWDOWN_BREACH = auto()      # Max drawdown exceeded

    # --- Market Events ---
    MARKET_DATA_UPDATE = auto()   # New price data available
    REGIME_CHANGE = auto()        # Market regime shifted
    VOLATILITY_SPIKE = auto()     # Unusual volatility detected
    NEWS_ALERT = auto()           # Important news detected

    # --- System Events ---
    AGENT_STARTED = auto()        # Agent came online
    AGENT_STOPPED = auto()        # Agent went offline
    HEARTBEAT = auto()            # Health check ping
    ERROR = auto()                # Error occurred


class DecisionType(Enum):
    """
    Possible decisions an agent can make about a trade proposal.

    APPROVE: Trade is acceptable as-is, proceed with execution
    REJECT: Trade violates rules, do NOT execute
    MODIFY: Trade is acceptable with modifications (see modified_params)
    DEFER: Agent cannot decide, pass to human or another agent
    """
    APPROVE = auto()   # Trade approved as-is
    REJECT = auto()    # Trade rejected, do not execute
    MODIFY = auto()    # Trade approved with modifications
    DEFER = auto()     # Cannot decide, need human input


class RiskLevel(Enum):
    """
    Risk severity levels for categorization.

    Used for:
        - Dashboard color coding (GREEN -> RED)
        - Alert prioritization
        - Decision weighting
    """
    LOW = auto()       # Green: Normal operations
    MEDIUM = auto()    # Yellow: Heightened awareness
    HIGH = auto()      # Orange: Caution advised
    CRITICAL = auto()  # Red: Immediate attention required


# =============================================================================
# DATA CLASSES - Event Payloads
# =============================================================================


@dataclass
class AgentEvent:
    """
    Base event class for all inter-agent communication.

    Every event has:
        - Unique ID for tracking
        - Type for routing
        - Source agent ID
        - Timestamp for ordering
        - Payload with event-specific data
        - Optional correlation ID to link related events

    Example:
        event = AgentEvent(
            event_type=EventType.TRADE_PROPOSED,
            source_agent="RL-Agent-1",
            payload={'action': 'BUY', 'quantity': 0.1}
        )
    """
    event_type: EventType                           # What kind of event
    source_agent: str                               # Who sent it
    payload: Dict[str, Any] = field(default_factory=dict)  # Event data
    timestamp: datetime = field(default_factory=datetime.now)  # When created
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    correlation_id: Optional[str] = None            # Links related events
    priority: int = 5                               # 1=highest, 10=lowest

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization/logging."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'source_agent': self.source_agent,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'priority': self.priority,
            'payload': self.payload
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentEvent':
        """Create event from dictionary."""
        return cls(
            event_type=EventType[data['event_type']],
            source_agent=data['source_agent'],
            payload=data.get('payload', {}),
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_id=data.get('event_id', str(uuid.uuid4())[:12]),
            correlation_id=data.get('correlation_id'),
            priority=data.get('priority', 5)
        )


@dataclass
class TradeProposal:
    """
    A proposed trade that needs risk evaluation.

    This is the primary input to the Risk Sentinel Agent. Contains all
    information needed to assess the trade's risk profile.

    Attributes:
        action: Trade direction (BUY, SELL, HOLD)
        asset: Asset being traded (e.g., "XAU/USD")
        quantity: Proposed position size
        entry_price: Current/expected entry price
        current_balance: Available capital
        current_position: Existing position size
        market_data: Current market indicators (RSI, ATR, etc.)
        metadata: Additional context (strategy name, confidence, etc.)
    """
    action: str                                     # BUY, SELL, HOLD
    asset: str = "XAU/USD"                         # Trading pair
    quantity: float = 0.0                          # Position size
    entry_price: float = 0.0                       # Entry price
    current_balance: float = 0.0                   # Available cash
    current_position: float = 0.0                  # Current holdings
    current_equity: float = 0.0                    # Total portfolio value
    unrealized_pnl: float = 0.0                    # Open position P&L
    market_data: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert proposal to dictionary."""
        return {
            'proposal_id': self.proposal_id,
            'action': self.action,
            'asset': self.asset,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_balance': self.current_balance,
            'current_position': self.current_position,
            'current_equity': self.current_equity,
            'unrealized_pnl': self.unrealized_pnl,
            'market_data': self.market_data,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RiskViolation:
    """
    Represents a single risk rule violation.

    When a trade proposal violates a risk rule, we create one of these
    to document exactly what was violated and why.

    Attributes:
        rule_name: Identifier of the violated rule
        rule_description: Human-readable explanation
        severity: How serious is this violation
        current_value: What the actual value is
        threshold: What the limit is
        recommendation: What should be done
    """
    rule_name: str                                  # e.g., "MAX_POSITION_SIZE"
    rule_description: str                           # Human-readable
    severity: RiskLevel                             # How bad is it
    current_value: float                            # Actual value
    threshold: float                                # Limit value
    recommendation: str                             # What to do

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            'rule_name': self.rule_name,
            'rule_description': self.rule_description,
            'severity': self.severity.name,
            'current_value': self.current_value,
            'threshold': self.threshold,
            'recommendation': self.recommendation
        }


@dataclass
class RiskAssessment:
    """
    Result of risk analysis on a trade proposal.

    This is the primary output of the Risk Sentinel Agent. Contains:
        - The decision (APPROVE, REJECT, MODIFY)
        - Risk score (0-100)
        - List of violations (if any)
        - Reasoning chain (for explainability)
        - Modified parameters (if decision is MODIFY)

    Attributes:
        proposal_id: Links to the original TradeProposal
        decision: Final verdict on the trade
        risk_score: Overall risk score (0=safe, 100=maximum risk)
        risk_level: Categorical risk level
        violations: List of rule violations
        reasoning: Human-readable explanation of decision
        modified_params: If MODIFY, what changes are recommended
        assessment_time_ms: How long the assessment took
    """
    proposal_id: str                                # Links to original proposal
    decision: DecisionType                          # APPROVE, REJECT, MODIFY
    risk_score: float = 0.0                        # 0-100 risk score
    risk_level: RiskLevel = RiskLevel.LOW          # Categorical level
    violations: List[RiskViolation] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)  # Decision explanation
    modified_params: Dict[str, Any] = field(default_factory=dict)
    assessment_time_ms: float = 0.0                # Processing time
    timestamp: datetime = field(default_factory=datetime.now)
    assessment_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            'assessment_id': self.assessment_id,
            'proposal_id': self.proposal_id,
            'decision': self.decision.name,
            'risk_score': round(self.risk_score, 2),
            'risk_level': self.risk_level.name,
            'violations': [v.to_dict() for v in self.violations],
            'reasoning': self.reasoning,
            'modified_params': self.modified_params,
            'assessment_time_ms': round(self.assessment_time_ms, 2),
            'timestamp': self.timestamp.isoformat()
        }

    def is_approved(self) -> bool:
        """Check if the trade was approved (with or without modifications)."""
        return self.decision in [DecisionType.APPROVE, DecisionType.MODIFY]

    def get_summary(self) -> str:
        """Get a one-line summary of the assessment."""
        emoji = {
            DecisionType.APPROVE: "OK",
            DecisionType.REJECT: "XX",
            DecisionType.MODIFY: "~~",
            DecisionType.DEFER: "??"
        }
        return (
            f"[{emoji.get(self.decision, '??')}] "
            f"{self.decision.name} | "
            f"Risk: {self.risk_score:.0f}/100 ({self.risk_level.name}) | "
            f"Violations: {len(self.violations)}"
        )


@dataclass
class AgentDecision:
    """
    Wrapper for any agent decision with full context.

    This is the final output that goes back to the environment.
    It wraps whatever specific assessment was made with metadata
    for tracking and auditing.
    """
    agent_id: str                                   # Which agent decided
    decision_type: DecisionType                     # What was decided
    confidence: float = 1.0                         # 0-1 confidence score
    assessment: Optional[RiskAssessment] = None     # Detailed assessment
    raw_action: Any = None                          # Original action requested
    approved_action: Any = None                     # Action after processing
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert decision to dictionary."""
        return {
            'agent_id': self.agent_id,
            'decision_type': self.decision_type.name,
            'confidence': self.confidence,
            'assessment': self.assessment.to_dict() if self.assessment else None,
            'raw_action': self.raw_action,
            'approved_action': self.approved_action,
            'timestamp': self.timestamp.isoformat()
        }


# =============================================================================
# EVENT BUS - Central Message Router
# =============================================================================


class EventBus:
    """
    Central hub for event-driven communication between agents.

    The EventBus implements the publish-subscribe pattern:
        - Agents subscribe to event types they care about
        - When an event is published, all subscribers are notified
        - Events are logged for audit purposes

    Thread-Safety:
        All operations are thread-safe using locks.

    Example Usage:
        bus = EventBus()

        # Subscribe to events
        bus.subscribe(EventType.TRADE_PROPOSED, my_handler)

        # Publish an event
        event = AgentEvent(EventType.TRADE_PROPOSED, "agent-1", {...})
        responses = bus.publish(event)

        # Unsubscribe
        bus.unsubscribe(EventType.TRADE_PROPOSED, my_handler)
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the event bus.

        Args:
            enable_logging: Whether to log all events (for audit)
        """
        # --- Subscription Storage ---
        # Maps event types to lists of handler functions
        # Example: {EventType.TRADE_PROPOSED: [handler1, handler2]}
        self._subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self._lock: Lock = Lock()

        # --- Event History ---
        # Stores all events for audit purposes (limited size)
        self._event_history: List[Dict[str, Any]] = []
        self._max_history_size: int = 10000  # Prevent memory bloat

        # --- Logging ---
        self._enable_logging = enable_logging
        self._logger = logging.getLogger("event_bus")

    def subscribe(
        self,
        event_type: EventType,
        handler: Callable[['AgentEvent'], Optional['AgentEvent']]
    ) -> None:
        """
        Subscribe a handler to an event type.

        Args:
            event_type: The type of event to listen for
            handler: Function to call when event occurs.
                     Signature: (AgentEvent) -> Optional[AgentEvent]
        """
        with self._lock:
            if handler not in self._subscribers[event_type]:
                self._subscribers[event_type].append(handler)
                self._logger.debug(
                    f"Subscribed handler to {event_type.name}"
                )

    def unsubscribe(
        self,
        event_type: EventType,
        handler: Callable
    ) -> None:
        """
        Remove a handler from an event type.

        Args:
            event_type: The type of event
            handler: The handler function to remove
        """
        with self._lock:
            if handler in self._subscribers[event_type]:
                self._subscribers[event_type].remove(handler)
                self._logger.debug(
                    f"Unsubscribed handler from {event_type.name}"
                )

    def publish(
        self,
        event: AgentEvent,
        wait_for_response: bool = True
    ) -> List[Optional[AgentEvent]]:
        """
        Publish an event to all subscribers.

        Args:
            event: The event to publish
            wait_for_response: Whether to collect and return responses

        Returns:
            List of response events from handlers (if wait_for_response=True)
        """
        responses: List[Optional[AgentEvent]] = []

        with self._lock:
            handlers = self._subscribers.get(event.event_type, []).copy()

        # Log the event
        if self._enable_logging:
            self._log_event(event)

        # Call each handler
        for handler in handlers:
            try:
                response = handler(event)
                if wait_for_response:
                    responses.append(response)
            except Exception as e:
                self._logger.error(
                    f"Handler error for {event.event_type.name}: {e}"
                )
                if wait_for_response:
                    responses.append(None)

        return responses

    def _log_event(self, event: AgentEvent) -> None:
        """
        Log an event for audit purposes.

        Maintains a circular buffer of recent events.
        """
        event_record = {
            **event.to_dict(),
            'logged_at': datetime.now().isoformat()
        }

        self._event_history.append(event_record)

        # Trim history if too large
        if len(self._event_history) > self._max_history_size:
            self._event_history = self._event_history[-self._max_history_size // 2:]

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get recent event history for auditing.

        Args:
            event_type: Filter by event type (None = all)
            limit: Maximum number of events to return

        Returns:
            List of event records (newest first)
        """
        history = self._event_history.copy()

        if event_type:
            history = [
                e for e in history
                if e.get('event_type') == event_type.name
            ]

        return history[-limit:][::-1]  # Newest first

    def get_subscriber_count(self, event_type: EventType) -> int:
        """Get the number of subscribers for an event type."""
        with self._lock:
            return len(self._subscribers.get(event_type, []))

    def clear_history(self) -> None:
        """Clear the event history (for testing or memory management)."""
        self._event_history.clear()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def create_trade_proposal_event(
    proposal: TradeProposal,
    source_agent: str = "rl_agent"
) -> AgentEvent:
    """
    Create an AgentEvent from a TradeProposal.

    Helper function to wrap a TradeProposal in the standard event format.

    Args:
        proposal: The trade proposal to wrap
        source_agent: ID of the agent making the proposal

    Returns:
        AgentEvent ready to publish
    """
    return AgentEvent(
        event_type=EventType.TRADE_PROPOSED,
        source_agent=source_agent,
        payload=proposal.to_dict(),
        correlation_id=proposal.proposal_id
    )


def create_risk_assessment_event(
    assessment: RiskAssessment,
    source_agent: str
) -> AgentEvent:
    """
    Create an AgentEvent from a RiskAssessment.

    Helper function to wrap a RiskAssessment in the standard event format.

    Args:
        assessment: The risk assessment to wrap
        source_agent: ID of the agent that made the assessment

    Returns:
        AgentEvent ready to publish
    """
    event_type = {
        DecisionType.APPROVE: EventType.TRADE_APPROVED,
        DecisionType.REJECT: EventType.TRADE_REJECTED,
        DecisionType.MODIFY: EventType.TRADE_MODIFIED,
        DecisionType.DEFER: EventType.RISK_ASSESSED
    }.get(assessment.decision, EventType.RISK_ASSESSED)

    return AgentEvent(
        event_type=event_type,
        source_agent=source_agent,
        payload=assessment.to_dict(),
        correlation_id=assessment.proposal_id
    )
