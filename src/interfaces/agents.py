# =============================================================================
# AGENT INTERFACES
# =============================================================================
# Abstract interfaces for trading agents.
#
# These interfaces define the contract that all agents must follow,
# enabling dependency injection and easy testing with mock implementations.
#
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


# =============================================================================
# COMMON TYPES
# =============================================================================

class AgentState(Enum):
    """State of an agent."""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class DecisionType(Enum):
    """Type of agent decision."""
    APPROVE = "approve"
    REJECT = "reject"
    MODIFY = "modify"
    ABSTAIN = "abstain"


class Priority(Enum):
    """Priority level for decisions."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class TradeProposal:
    """
    A proposed trade for evaluation by agents.

    This is the input that agents receive and must evaluate.
    """
    proposal_id: str
    action: int  # 0=HOLD, 1=OPEN_LONG, 2=CLOSE_LONG, 3=OPEN_SHORT, 4=CLOSE_SHORT
    symbol: str
    current_price: float
    proposed_quantity: float
    current_position: float
    current_equity: float
    timestamp: datetime
    metadata: Dict[str, Any]


@dataclass
class AgentDecision:
    """
    Decision from an agent about a trade proposal.

    Contains the decision, reasoning, and any modifications.
    """
    agent_name: str
    decision: DecisionType
    priority: Priority
    confidence: float  # 0.0 to 1.0
    reason: str
    modified_quantity: Optional[float] = None
    modified_action: Optional[int] = None
    risk_score: Optional[float] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class AgentHealth:
    """Health status of an agent."""
    agent_name: str
    state: AgentState
    is_healthy: bool
    last_heartbeat: datetime
    error_count: int
    last_error: Optional[str]
    metrics: Dict[str, float]


# =============================================================================
# BASE AGENT INTERFACE
# =============================================================================

class IAgent(ABC):
    """
    Base interface for all trading agents.

    All agents must implement these core methods for lifecycle
    management and health monitoring.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Get agent name."""
        pass

    @property
    @abstractmethod
    def state(self) -> AgentState:
        """Get current agent state."""
        pass

    @abstractmethod
    async def start(self) -> bool:
        """
        Start the agent.

        Returns:
            True if started successfully
        """
        pass

    @abstractmethod
    async def stop(self) -> bool:
        """
        Stop the agent gracefully.

        Returns:
            True if stopped successfully
        """
        pass

    @abstractmethod
    def get_health(self) -> AgentHealth:
        """
        Get agent health status.

        Returns:
            Current health status
        """
        pass

    @abstractmethod
    def heartbeat(self) -> bool:
        """
        Send heartbeat to indicate agent is alive.

        Returns:
            True if heartbeat recorded
        """
        pass


# =============================================================================
# SPECIALIZED AGENT INTERFACES
# =============================================================================

class IRiskAgent(IAgent):
    """
    Interface for risk management agents.

    Risk agents evaluate trade proposals for risk compliance
    and can approve, reject, or modify proposed trades.
    """

    @abstractmethod
    async def evaluate_trade(self, proposal: TradeProposal) -> AgentDecision:
        """
        Evaluate a trade proposal for risk compliance.

        Args:
            proposal: Trade proposal to evaluate

        Returns:
            Decision with risk assessment
        """
        pass

    @abstractmethod
    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.

        Returns:
            Dictionary of risk metrics (VaR, exposure, etc.)
        """
        pass

    @abstractmethod
    def set_risk_limits(self, limits: Dict[str, float]) -> None:
        """
        Set risk limits.

        Args:
            limits: Dictionary of limit names to values
        """
        pass


class INewsAgent(IAgent):
    """
    Interface for news/sentiment analysis agents.

    News agents analyze market news and events to determine
    if trading should be blocked or modified.
    """

    @abstractmethod
    async def evaluate_trade(self, proposal: TradeProposal) -> AgentDecision:
        """
        Evaluate trade against current news/events.

        Args:
            proposal: Trade proposal to evaluate

        Returns:
            Decision based on news analysis
        """
        pass

    @abstractmethod
    async def get_current_sentiment(self) -> Dict[str, float]:
        """
        Get current market sentiment.

        Returns:
            Dictionary with sentiment scores by source/topic
        """
        pass

    @abstractmethod
    def get_blocked_events(self) -> List[Dict[str, Any]]:
        """
        Get list of events that block trading.

        Returns:
            List of blocking events with details
        """
        pass


class IMarketRegimeAgent(IAgent):
    """
    Interface for market regime detection agents.

    Market regime agents classify current market conditions
    to inform trading strategy adjustments.
    """

    class Regime(Enum):
        """Market regime types."""
        TRENDING_UP = "trending_up"
        TRENDING_DOWN = "trending_down"
        RANGING = "ranging"
        VOLATILE = "volatile"
        LOW_VOLATILITY = "low_volatility"
        BREAKOUT = "breakout"
        REVERSAL = "reversal"
        UNKNOWN = "unknown"

    @abstractmethod
    async def evaluate_trade(self, proposal: TradeProposal) -> AgentDecision:
        """
        Evaluate trade against current market regime.

        Args:
            proposal: Trade proposal to evaluate

        Returns:
            Decision based on regime analysis
        """
        pass

    @abstractmethod
    def get_current_regime(self) -> 'IMarketRegimeAgent.Regime':
        """
        Get current detected market regime.

        Returns:
            Current market regime
        """
        pass

    @abstractmethod
    def get_regime_confidence(self) -> float:
        """
        Get confidence in current regime detection.

        Returns:
            Confidence score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    def get_regime_history(self, periods: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent regime history.

        Args:
            periods: Number of periods to return

        Returns:
            List of regime records with timestamps
        """
        pass


# =============================================================================
# MOCK IMPLEMENTATIONS (for testing)
# =============================================================================

class MockRiskAgent(IRiskAgent):
    """Mock risk agent that always approves trades."""

    def __init__(self, name: str = "mock_risk"):
        self._name = name
        self._state = AgentState.READY
        self._limits: Dict[str, float] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> AgentState:
        return self._state

    async def start(self) -> bool:
        self._state = AgentState.RUNNING
        return True

    async def stop(self) -> bool:
        self._state = AgentState.STOPPED
        return True

    def get_health(self) -> AgentHealth:
        return AgentHealth(
            agent_name=self._name,
            state=self._state,
            is_healthy=True,
            last_heartbeat=datetime.utcnow(),
            error_count=0,
            last_error=None,
            metrics={}
        )

    def heartbeat(self) -> bool:
        return True

    async def evaluate_trade(self, proposal: TradeProposal) -> AgentDecision:
        return AgentDecision(
            agent_name=self._name,
            decision=DecisionType.APPROVE,
            priority=Priority.NORMAL,
            confidence=1.0,
            reason="Mock agent: auto-approve",
            risk_score=0.1
        )

    def get_risk_metrics(self) -> Dict[str, float]:
        return {'var_95': 0.02, 'exposure': 0.5}

    def set_risk_limits(self, limits: Dict[str, float]) -> None:
        self._limits = limits


class MockNewsAgent(INewsAgent):
    """Mock news agent that never blocks trades."""

    def __init__(self, name: str = "mock_news"):
        self._name = name
        self._state = AgentState.READY

    @property
    def name(self) -> str:
        return self._name

    @property
    def state(self) -> AgentState:
        return self._state

    async def start(self) -> bool:
        self._state = AgentState.RUNNING
        return True

    async def stop(self) -> bool:
        self._state = AgentState.STOPPED
        return True

    def get_health(self) -> AgentHealth:
        return AgentHealth(
            agent_name=self._name,
            state=self._state,
            is_healthy=True,
            last_heartbeat=datetime.utcnow(),
            error_count=0,
            last_error=None,
            metrics={}
        )

    def heartbeat(self) -> bool:
        return True

    async def evaluate_trade(self, proposal: TradeProposal) -> AgentDecision:
        return AgentDecision(
            agent_name=self._name,
            decision=DecisionType.APPROVE,
            priority=Priority.NORMAL,
            confidence=1.0,
            reason="Mock agent: no news blocking"
        )

    async def get_current_sentiment(self) -> Dict[str, float]:
        return {'overall': 0.5, 'news': 0.5, 'social': 0.5}

    def get_blocked_events(self) -> List[Dict[str, Any]]:
        return []
