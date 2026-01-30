# =============================================================================
# TRADE LOGGER INTERFACE
# =============================================================================
# Abstract interface for trade logging implementations.
#
# This allows swapping logging backends without changing the trading code:
# - FileTradeLogger: Logs to local files
# - DatabaseTradeLogger: Logs to SQLite/PostgreSQL
# - CloudTradeLogger: Logs to cloud services
# - NullTradeLogger: No logging (for tests)
#
# =============================================================================

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class TradeType(Enum):
    """Type of trade action."""
    OPEN_LONG = "open_long"
    CLOSE_LONG = "close_long"
    OPEN_SHORT = "open_short"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"


class TradeStatus(Enum):
    """Status of a trade."""
    PENDING = "pending"
    EXECUTED = "executed"
    REJECTED = "rejected"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class TradeRecord:
    """
    Immutable record of a trade event.

    This dataclass captures all relevant information about a trade
    for logging, auditing, and analysis purposes.
    """
    trade_id: str
    timestamp: datetime
    trade_type: TradeType
    status: TradeStatus
    symbol: str
    entry_price: float
    exit_price: Optional[float]
    quantity: float
    pnl: float
    pnl_pct: float
    commission: float
    slippage: float
    duration_steps: int
    episode: int
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'trade_type': self.trade_type.value,
            'status': self.status.value,
            'symbol': self.symbol,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'quantity': self.quantity,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'slippage': self.slippage,
            'duration_steps': self.duration_steps,
            'episode': self.episode,
            'metadata': self.metadata,
        }


@dataclass
class TradeSummary:
    """Summary statistics for a trading session."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_commission: float
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    max_consecutive_wins: int
    max_consecutive_losses: int
    sharpe_ratio: Optional[float]
    max_drawdown_pct: float


class ITradeLogger(ABC):
    """
    Abstract interface for trade logging.

    Implementations must provide methods for:
    - Logging individual trades
    - Managing episodes (trading sessions)
    - Retrieving trade history
    - Computing summary statistics

    Example usage:
        logger: ITradeLogger = FileTradeLogger("./logs/trades.json")

        logger.new_episode()
        logger.log_trade(trade_record)

        summary = logger.get_summary()
        print(f"Win rate: {summary.win_rate:.1%}")
    """

    @abstractmethod
    def log_trade(self, trade: TradeRecord) -> bool:
        """
        Log a trade event.

        Args:
            trade: Trade record to log

        Returns:
            True if logged successfully
        """
        pass

    @abstractmethod
    def log_trade_simple(
        self,
        trade_id: str,
        trade_type: TradeType,
        entry_price: float,
        quantity: float,
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
        **kwargs
    ) -> bool:
        """
        Simplified trade logging with minimal required fields.

        Args:
            trade_id: Unique trade identifier
            trade_type: Type of trade action
            entry_price: Entry price
            quantity: Trade quantity
            exit_price: Exit price (for closed trades)
            pnl: Profit/loss
            **kwargs: Additional metadata

        Returns:
            True if logged successfully
        """
        pass

    @abstractmethod
    def new_episode(self) -> int:
        """
        Start a new trading episode.

        Returns:
            New episode number
        """
        pass

    @abstractmethod
    def get_current_episode(self) -> int:
        """Get current episode number."""
        pass

    @abstractmethod
    def get_trades(
        self,
        episode: Optional[int] = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        """
        Get trade history.

        Args:
            episode: Filter by episode (None = all episodes)
            limit: Maximum number of trades to return

        Returns:
            List of trade records
        """
        pass

    @abstractmethod
    def get_summary(self, episode: Optional[int] = None) -> TradeSummary:
        """
        Get summary statistics.

        Args:
            episode: Filter by episode (None = all episodes)

        Returns:
            Trade summary statistics
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered logs to storage."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources."""
        pass


# =============================================================================
# NULL IMPLEMENTATION (for testing)
# =============================================================================

class NullTradeLogger(ITradeLogger):
    """
    Null implementation that does nothing.

    Use this in tests or when logging is not needed.
    """

    def __init__(self):
        self._episode = 0

    def log_trade(self, trade: TradeRecord) -> bool:
        return True

    def log_trade_simple(
        self,
        trade_id: str,
        trade_type: TradeType,
        entry_price: float,
        quantity: float,
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
        **kwargs
    ) -> bool:
        return True

    def new_episode(self) -> int:
        self._episode += 1
        return self._episode

    def get_current_episode(self) -> int:
        return self._episode

    def get_trades(
        self,
        episode: Optional[int] = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        return []

    def get_summary(self, episode: Optional[int] = None) -> TradeSummary:
        return TradeSummary(
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            total_pnl=0.0,
            total_commission=0.0,
            win_rate=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
            max_consecutive_wins=0,
            max_consecutive_losses=0,
            sharpe_ratio=None,
            max_drawdown_pct=0.0,
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        pass


# =============================================================================
# IN-MEMORY IMPLEMENTATION (for testing and simple use cases)
# =============================================================================

class InMemoryTradeLogger(ITradeLogger):
    """
    In-memory implementation for testing and simple use cases.

    Stores trades in memory - data is lost when the process exits.
    """

    def __init__(self):
        self._trades: List[TradeRecord] = []
        self._episode = 0

    def log_trade(self, trade: TradeRecord) -> bool:
        self._trades.append(trade)
        return True

    def log_trade_simple(
        self,
        trade_id: str,
        trade_type: TradeType,
        entry_price: float,
        quantity: float,
        exit_price: Optional[float] = None,
        pnl: float = 0.0,
        **kwargs
    ) -> bool:
        trade = TradeRecord(
            trade_id=trade_id,
            timestamp=datetime.utcnow(),
            trade_type=trade_type,
            status=TradeStatus.EXECUTED,
            symbol=kwargs.get('symbol', 'XAU/USD'),
            entry_price=entry_price,
            exit_price=exit_price,
            quantity=quantity,
            pnl=pnl,
            pnl_pct=kwargs.get('pnl_pct', 0.0),
            commission=kwargs.get('commission', 0.0),
            slippage=kwargs.get('slippage', 0.0),
            duration_steps=kwargs.get('duration_steps', 0),
            episode=self._episode,
            metadata=kwargs,
        )
        return self.log_trade(trade)

    def new_episode(self) -> int:
        self._episode += 1
        return self._episode

    def get_current_episode(self) -> int:
        return self._episode

    def get_trades(
        self,
        episode: Optional[int] = None,
        limit: int = 100
    ) -> List[TradeRecord]:
        trades = self._trades
        if episode is not None:
            trades = [t for t in trades if t.episode == episode]
        return trades[-limit:]

    def get_summary(self, episode: Optional[int] = None) -> TradeSummary:
        trades = self.get_trades(episode=episode, limit=10000)

        if not trades:
            return TradeSummary(
                total_trades=0, winning_trades=0, losing_trades=0,
                total_pnl=0.0, total_commission=0.0, win_rate=0.0,
                avg_win=0.0, avg_loss=0.0, profit_factor=0.0,
                max_consecutive_wins=0, max_consecutive_losses=0,
                sharpe_ratio=None, max_drawdown_pct=0.0,
            )

        winning = [t for t in trades if t.pnl > 0]
        losing = [t for t in trades if t.pnl < 0]

        total_wins = sum(t.pnl for t in winning)
        total_losses = abs(sum(t.pnl for t in losing))

        return TradeSummary(
            total_trades=len(trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            total_pnl=sum(t.pnl for t in trades),
            total_commission=sum(t.commission for t in trades),
            win_rate=len(winning) / len(trades) if trades else 0.0,
            avg_win=total_wins / len(winning) if winning else 0.0,
            avg_loss=total_losses / len(losing) if losing else 0.0,
            profit_factor=total_wins / total_losses if total_losses > 0 else float('inf'),
            max_consecutive_wins=0,  # Simplified
            max_consecutive_losses=0,  # Simplified
            sharpe_ratio=None,
            max_drawdown_pct=0.0,
        )

    def flush(self) -> None:
        pass

    def close(self) -> None:
        self._trades.clear()
