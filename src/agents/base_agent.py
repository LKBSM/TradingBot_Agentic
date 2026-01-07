# =============================================================================
# BASE AGENT - Abstract Foundation for All Trading Agents
# =============================================================================
# This module defines the foundational abstract class that ALL agents must
# inherit from. It provides:
#   - Standardized lifecycle management (init -> run -> shutdown)
#   - Event subscription and publishing capabilities
#   - State management with audit logging
#   - Health monitoring and heartbeat
#   - Graceful degradation patterns
#
# Design Principles:
#   1. Single Responsibility: Each agent does ONE thing well
#   2. Loose Coupling: Agents communicate only via events
#   3. Fail-Safe: Agents fail gracefully, never crash the system
#   4. Observable: All decisions are logged and explainable
#
# =============================================================================

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid
import json
from threading import Lock

# =============================================================================
# ENUMS - Agent States and Capabilities
# =============================================================================


class AgentState(Enum):
    """
    Lifecycle states for an agent. Follows a strict state machine:

    INITIALIZING -> READY -> RUNNING -> (PAUSED) -> STOPPING -> STOPPED
                       |                    ^
                       +--------------------+

    State Transitions:
        INITIALIZING: Agent is loading config, connecting to resources
        READY: Agent has initialized successfully, waiting to start
        RUNNING: Agent is actively processing events and making decisions
        PAUSED: Agent is temporarily inactive (can resume quickly)
        STOPPING: Agent is gracefully shutting down
        STOPPED: Agent has fully terminated
        ERROR: Agent encountered a critical error (can attempt recovery)
    """
    INITIALIZING = auto()  # Loading configuration and resources
    READY = auto()         # Initialized, waiting to start
    RUNNING = auto()       # Actively processing
    PAUSED = auto()        # Temporarily inactive
    STOPPING = auto()      # Graceful shutdown in progress
    STOPPED = auto()       # Fully terminated
    ERROR = auto()         # Critical error state


class AgentCapability(Enum):
    """
    Declares what an agent can do. Used for:
        1. Service discovery (find agents with specific capabilities)
        2. Permission checking (can this agent perform this action?)
        3. Orchestrator routing (send events to capable agents)

    Example: A RiskSentinelAgent has RISK_ASSESSMENT capability,
             so the orchestrator routes all risk-related queries to it.
    """
    RISK_ASSESSMENT = auto()      # Can evaluate risk of proposed trades
    TRADE_EXECUTION = auto()      # Can execute trades on exchanges
    MARKET_ANALYSIS = auto()      # Can analyze market conditions
    PORTFOLIO_MANAGEMENT = auto()  # Can manage portfolio allocation
    BACKTESTING = auto()          # Can run historical simulations
    SENTIMENT_ANALYSIS = auto()    # Can analyze news/social sentiment
    REGIME_DETECTION = auto()      # Can detect market regimes


# =============================================================================
# DATA CLASSES - Structured Data for Agent Communication
# =============================================================================


@dataclass
class AgentMetrics:
    """
    Performance metrics tracked by every agent. Used for:
        1. Health monitoring (is the agent performing well?)
        2. Performance dashboards (how many decisions per second?)
        3. Audit trails (what did this agent do today?)

    Attributes:
        events_processed: Total events this agent has handled
        decisions_made: Total decisions (approve/reject/modify)
        avg_response_time_ms: Average time to process an event
        uptime_seconds: How long the agent has been running
        error_count: Number of errors encountered
        last_activity: Timestamp of last meaningful action
    """
    events_processed: int = 0
    decisions_made: int = 0
    avg_response_time_ms: float = 0.0
    uptime_seconds: float = 0.0
    error_count: int = 0
    last_activity: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for JSON serialization."""
        return {
            'events_processed': self.events_processed,
            'decisions_made': self.decisions_made,
            'avg_response_time_ms': round(self.avg_response_time_ms, 2),
            'uptime_seconds': round(self.uptime_seconds, 2),
            'error_count': self.error_count,
            'last_activity': self.last_activity.isoformat() if self.last_activity else None
        }


@dataclass
class AgentContext:
    """
    Runtime context passed to agents during decision-making.

    This encapsulates everything an agent needs to make a decision:
        - Current market state (prices, indicators)
        - Portfolio state (balance, positions)
        - Historical context (recent trades, performance)
        - Environment metadata (timestamp, episode info)

    Attributes:
        timestamp: Current simulation/live timestamp
        market_data: Dict containing OHLCV and indicators
        portfolio: Dict containing balance, positions, equity
        metadata: Additional context (episode number, step, etc.)
    """
    timestamp: datetime
    market_data: Dict[str, Any] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert context to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            'market_data': self.market_data,
            'portfolio': self.portfolio,
            'metadata': self.metadata
        }


# =============================================================================
# BASE AGENT CLASS - Abstract Foundation
# =============================================================================


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.

    This class provides the foundational structure that all agents must follow.
    It handles:
        - Unique identification and naming
        - State management with thread-safety
        - Event subscription and publishing
        - Metrics collection
        - Logging with structured output

    Subclasses MUST implement:
        - initialize(): Setup resources, load models
        - process_event(): Handle incoming events
        - shutdown(): Clean up resources

    Subclasses SHOULD implement:
        - get_capabilities(): Declare agent capabilities
        - health_check(): Custom health validation

    Example Usage:
        class MyAgent(BaseAgent):
            def initialize(self) -> bool:
                # Load ML model, connect to DB, etc.
                return True

            def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
                # Handle the event and return response
                return response_event

            def shutdown(self) -> bool:
                # Close connections, save state
                return True
    """

    def __init__(
        self,
        name: str,
        config: Optional[Dict[str, Any]] = None,
        event_bus: Optional['EventBus'] = None
    ):
        """
        Initialize the base agent.

        Args:
            name: Human-readable name for this agent (e.g., "RiskSentinel-1")
            config: Configuration dictionary with agent-specific settings
            event_bus: Optional shared event bus for inter-agent communication
        """
        # --- Identity ---
        # Each agent gets a unique ID (UUID) for tracking across the system.
        # The name is human-readable, ID is for programmatic references.
        self.id: str = str(uuid.uuid4())[:8]  # Short UUID for readability
        self.name: str = name
        self.full_id: str = f"{name}-{self.id}"  # e.g., "RiskSentinel-a1b2c3d4"

        # --- Configuration ---
        # Config is immutable after initialization to prevent runtime surprises.
        self._config: Dict[str, Any] = config or {}

        # --- State Management ---
        # Thread-safe state transitions using a lock.
        self._state: AgentState = AgentState.INITIALIZING
        self._state_lock: Lock = Lock()
        self._state_history: List[Dict[str, Any]] = []  # Audit trail

        # --- Event System ---
        # Agents communicate via events, not direct method calls.
        self._event_bus: Optional['EventBus'] = event_bus
        self._subscriptions: List[str] = []  # Event types we're listening to

        # --- Metrics ---
        # Performance tracking for monitoring and optimization.
        self._metrics: AgentMetrics = AgentMetrics()
        self._start_time: Optional[datetime] = None

        # --- Logging ---
        # Each agent gets its own logger with structured output.
        self._logger = logging.getLogger(f"agent.{self.full_id}")
        self._setup_logging()

        # Record initial state
        self._record_state_change(AgentState.INITIALIZING, "Agent created")

    def _setup_logging(self) -> None:
        """
        Configure structured logging for this agent.

        Log format includes:
            - Timestamp
            - Agent ID
            - Log level
            - Message

        Example output:
            2024-01-15 10:30:45 | RiskSentinel-a1b2c3d4 | INFO | Trade approved
        """
        if not self._logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    @property
    def state(self) -> AgentState:
        """
        Get current agent state (thread-safe).

        Returns:
            Current AgentState enum value
        """
        with self._state_lock:
            return self._state

    def _set_state(self, new_state: AgentState, reason: str = "") -> bool:
        """
        Transition to a new state with validation (thread-safe).

        State transitions are validated to prevent illegal jumps:
            - Can't go from STOPPED to RUNNING (must reinitialize)
            - Can't skip states (INITIALIZING -> RUNNING skips READY)

        Args:
            new_state: Target state to transition to
            reason: Human-readable explanation for the transition

        Returns:
            True if transition was successful, False otherwise
        """
        with self._state_lock:
            # Validate transition (basic state machine rules)
            valid_transitions = {
                AgentState.INITIALIZING: [AgentState.READY, AgentState.ERROR],
                AgentState.READY: [AgentState.RUNNING, AgentState.STOPPING, AgentState.ERROR],
                AgentState.RUNNING: [AgentState.PAUSED, AgentState.STOPPING, AgentState.ERROR],
                AgentState.PAUSED: [AgentState.RUNNING, AgentState.STOPPING, AgentState.ERROR],
                AgentState.STOPPING: [AgentState.STOPPED, AgentState.ERROR],
                AgentState.STOPPED: [AgentState.INITIALIZING],  # Can restart
                AgentState.ERROR: [AgentState.INITIALIZING, AgentState.STOPPING],
            }

            if new_state not in valid_transitions.get(self._state, []):
                self._logger.warning(
                    f"Invalid state transition: {self._state.name} -> {new_state.name}"
                )
                return False

            old_state = self._state
            self._state = new_state
            self._record_state_change(new_state, reason)
            self._logger.info(
                f"State: {old_state.name} -> {new_state.name} | Reason: {reason}"
            )
            return True

    def _record_state_change(self, state: AgentState, reason: str) -> None:
        """
        Record state change for audit trail.

        Every state transition is logged with:
            - Timestamp (when it happened)
            - Previous state
            - New state
            - Reason for change

        This creates a complete audit trail for compliance and debugging.
        """
        self._state_history.append({
            'timestamp': datetime.now().isoformat(),
            'state': state.name,
            'reason': reason
        })

    # =========================================================================
    # LIFECYCLE METHODS (Template Method Pattern)
    # =========================================================================

    def start(self) -> bool:
        """
        Start the agent's main operation.

        This is the public entry point for starting an agent. It:
            1. Calls initialize() for setup
            2. Transitions to READY state
            3. Transitions to RUNNING state
            4. Subscribes to events

        Returns:
            True if agent started successfully, False otherwise

        Raises:
            None - errors are logged and False is returned
        """
        try:
            self._logger.info("Starting agent...")
            self._start_time = datetime.now()

            # Call subclass initialization
            if not self.initialize():
                self._set_state(AgentState.ERROR, "Initialization failed")
                return False

            # Transition through states
            if not self._set_state(AgentState.READY, "Initialization complete"):
                return False

            if not self._set_state(AgentState.RUNNING, "Agent started"):
                return False

            # Subscribe to events if event bus exists
            if self._event_bus:
                self._subscribe_to_events()

            self._logger.info(f"Agent {self.full_id} started successfully")
            return True

        except Exception as e:
            self._logger.error(f"Failed to start agent: {e}")
            self._set_state(AgentState.ERROR, str(e))
            return False

    def stop(self) -> bool:
        """
        Gracefully stop the agent.

        This is the public entry point for stopping an agent. It:
            1. Transitions to STOPPING state
            2. Unsubscribes from events
            3. Calls shutdown() for cleanup
            4. Transitions to STOPPED state

        Returns:
            True if agent stopped successfully, False otherwise
        """
        try:
            self._logger.info("Stopping agent...")

            if not self._set_state(AgentState.STOPPING, "Stop requested"):
                return False

            # Unsubscribe from events
            if self._event_bus:
                self._unsubscribe_from_events()

            # Call subclass shutdown
            if not self.shutdown():
                self._logger.warning("Shutdown returned False, continuing anyway")

            self._set_state(AgentState.STOPPED, "Agent stopped")
            self._logger.info(f"Agent {self.full_id} stopped successfully")
            return True

        except Exception as e:
            self._logger.error(f"Error during shutdown: {e}")
            self._set_state(AgentState.ERROR, str(e))
            return False

    def pause(self) -> bool:
        """Pause the agent temporarily (can resume quickly)."""
        return self._set_state(AgentState.PAUSED, "Paused by request")

    def resume(self) -> bool:
        """Resume a paused agent."""
        return self._set_state(AgentState.RUNNING, "Resumed from pause")

    # =========================================================================
    # EVENT HANDLING
    # =========================================================================

    def _subscribe_to_events(self) -> None:
        """Subscribe to relevant event types on the event bus."""
        if self._event_bus and self._subscriptions:
            for event_type in self._subscriptions:
                self._event_bus.subscribe(event_type, self._handle_event)
            self._logger.info(f"Subscribed to events: {self._subscriptions}")

    def _unsubscribe_from_events(self) -> None:
        """Unsubscribe from all events on the event bus."""
        if self._event_bus and self._subscriptions:
            for event_type in self._subscriptions:
                self._event_bus.unsubscribe(event_type, self._handle_event)
            self._logger.info("Unsubscribed from all events")

    def _handle_event(self, event: 'AgentEvent') -> Optional['AgentEvent']:
        """
        Internal event handler wrapper with metrics and error handling.

        This wraps the subclass's process_event() method to add:
            - Timing metrics
            - Error handling
            - Logging

        Args:
            event: The incoming event to process

        Returns:
            Response event from the agent, or None
        """
        import time

        if self.state != AgentState.RUNNING:
            self._logger.warning(f"Received event while in {self.state.name} state")
            return None

        start = time.time()
        try:
            # Call subclass implementation
            response = self.process_event(event)

            # Update metrics
            elapsed_ms = (time.time() - start) * 1000
            self._metrics.events_processed += 1
            self._metrics.last_activity = datetime.now()

            # Update rolling average response time
            n = self._metrics.events_processed
            old_avg = self._metrics.avg_response_time_ms
            self._metrics.avg_response_time_ms = old_avg + (elapsed_ms - old_avg) / n

            return response

        except Exception as e:
            self._logger.error(f"Error processing event: {e}")
            self._metrics.error_count += 1
            return None

    # =========================================================================
    # METRICS AND HEALTH
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics.

        Returns:
            Dictionary containing all performance metrics
        """
        # Update uptime
        if self._start_time:
            self._metrics.uptime_seconds = (
                datetime.now() - self._start_time
            ).total_seconds()

        return self._metrics.to_dict()

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the agent.

        Override in subclasses for custom health checks (e.g., DB connection,
        API availability, model loaded, etc.)

        Returns:
            Dictionary with health status and details
        """
        return {
            'agent_id': self.full_id,
            'state': self.state.name,
            'healthy': self.state == AgentState.RUNNING,
            'metrics': self.get_metrics(),
            'capabilities': [c.name for c in self.get_capabilities()]
        }

    def get_state_history(self) -> List[Dict[str, Any]]:
        """
        Get the complete state transition history.

        Returns:
            List of state change records for audit purposes
        """
        return self._state_history.copy()

    # =========================================================================
    # ABSTRACT METHODS (Must be implemented by subclasses)
    # =========================================================================

    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize agent resources.

        This method is called once during startup. Subclasses should:
            - Load ML models
            - Connect to databases
            - Load configuration
            - Validate dependencies

        Returns:
            True if initialization successful, False otherwise
        """
        pass

    @abstractmethod
    def process_event(self, event: 'AgentEvent') -> Optional['AgentEvent']:
        """
        Process an incoming event and return a response.

        This is the main decision-making method. Subclasses should:
            - Validate the event
            - Make decisions based on agent logic
            - Return a response event (or None)

        Args:
            event: The incoming event to process

        Returns:
            Response event, or None if no response needed
        """
        pass

    @abstractmethod
    def shutdown(self) -> bool:
        """
        Clean up agent resources.

        This method is called during graceful shutdown. Subclasses should:
            - Save state if needed
            - Close connections
            - Release resources

        Returns:
            True if shutdown successful, False otherwise
        """
        pass

    # =========================================================================
    # OPTIONAL OVERRIDES
    # =========================================================================

    def get_capabilities(self) -> List[AgentCapability]:
        """
        Declare the capabilities of this agent.

        Override in subclasses to declare what this agent can do.
        Used by the orchestrator for routing decisions.

        Returns:
            List of AgentCapability enums
        """
        return []

    def get_config(self) -> Dict[str, Any]:
        """
        Get a copy of the agent's configuration.

        Returns:
            Copy of the configuration dictionary
        """
        return self._config.copy()

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"<{self.__class__.__name__}(id={self.full_id}, state={self.state.name})>"
