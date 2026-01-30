# =============================================================================
# RISK MANAGEMENT INTERFACES
# =============================================================================
# Abstract interfaces for risk management components.
#
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple


# =============================================================================
# RISK MANAGER INTERFACE
# =============================================================================

@dataclass
class PositionSizeResult:
    """Result of position sizing calculation."""
    recommended_size: float
    max_size: float
    risk_per_trade: float
    stop_loss_price: float
    take_profit_price: float
    risk_reward_ratio: float
    reasoning: str


class IRiskManager(ABC):
    """
    Interface for risk management.

    Handles position sizing, stop-loss/take-profit calculations,
    and risk limit enforcement.
    """

    @abstractmethod
    def calculate_position_size(
        self,
        entry_price: float,
        current_equity: float,
        volatility: float,
        direction: int  # 1 for long, -1 for short
    ) -> PositionSizeResult:
        """
        Calculate recommended position size.

        Args:
            entry_price: Proposed entry price
            current_equity: Current account equity
            volatility: Current market volatility (ATR or similar)
            direction: Trade direction (1=long, -1=short)

        Returns:
            Position sizing result with recommendations
        """
        pass

    @abstractmethod
    def check_risk_limits(
        self,
        proposed_size: float,
        entry_price: float,
        current_equity: float,
        current_position: float
    ) -> Tuple[bool, str]:
        """
        Check if a trade would violate risk limits.

        Args:
            proposed_size: Proposed position size
            entry_price: Proposed entry price
            current_equity: Current account equity
            current_position: Current position size

        Returns:
            Tuple of (is_allowed, reason)
        """
        pass

    @abstractmethod
    def update_stops(
        self,
        current_price: float,
        entry_price: float,
        position_size: float,
        current_pnl: float
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Update stop-loss and take-profit levels.

        Implements trailing stop logic and dynamic TP adjustment.

        Args:
            current_price: Current market price
            entry_price: Position entry price
            position_size: Current position size
            current_pnl: Current unrealized P&L

        Returns:
            Tuple of (new_stop_loss, new_take_profit) or None if unchanged
        """
        pass

    @abstractmethod
    def get_current_risk_metrics(self) -> Dict[str, float]:
        """
        Get current risk metrics.

        Returns:
            Dictionary with metrics like:
            - current_var: Value at Risk
            - current_exposure: Total exposure
            - current_leverage: Current leverage ratio
            - drawdown_pct: Current drawdown percentage
        """
        pass


# =============================================================================
# KILL SWITCH INTERFACE
# =============================================================================

class HaltLevel(Enum):
    """Severity levels for trading halt."""
    NONE = 0
    CAUTION = 1
    REDUCED = 2
    NEW_ONLY = 3
    CLOSE_ONLY = 4
    FULL_HALT = 5
    EMERGENCY = 6


@dataclass
class KillSwitchStatus:
    """Current status of the kill switch."""
    halt_level: HaltLevel
    is_halted: bool
    is_trading_allowed: bool
    is_closing_allowed: bool
    position_multiplier: float
    halt_reason: Optional[str]
    halt_message: str
    halt_time: Optional[datetime]
    recovery_eta: Optional[datetime]


class IKillSwitch(ABC):
    """
    Interface for kill switch / circuit breaker.

    Provides emergency trading halt capabilities and
    gradual recovery mechanisms.
    """

    @abstractmethod
    def update(
        self,
        equity: float,
        daily_pnl: float,
        weekly_pnl: float,
        var_pct: Optional[float] = None
    ) -> HaltLevel:
        """
        Update kill switch with current portfolio state.

        Args:
            equity: Current account equity
            daily_pnl: Today's P&L
            weekly_pnl: This week's P&L
            var_pct: Current VaR percentage

        Returns:
            Current halt level
        """
        pass

    @abstractmethod
    def get_status(self) -> KillSwitchStatus:
        """
        Get current kill switch status.

        Returns:
            Current status including halt level and recovery info
        """
        pass

    @abstractmethod
    def is_trading_allowed(self) -> bool:
        """Check if new trades are allowed."""
        pass

    @abstractmethod
    def is_closing_allowed(self) -> bool:
        """Check if closing positions is allowed."""
        pass

    @abstractmethod
    def get_position_multiplier(self) -> float:
        """
        Get position size multiplier based on current state.

        Returns:
            Multiplier from 0.0 (no trading) to 1.0 (full size)
        """
        pass

    @abstractmethod
    def emergency_halt(self, reason: str) -> None:
        """Trigger emergency halt."""
        pass

    @abstractmethod
    def request_reset(self) -> bool:
        """Request reset of kill switch."""
        pass

    @abstractmethod
    def record_trade_result(self, pnl: float, pnl_pct: float) -> None:
        """Record a trade result for tracking."""
        pass
