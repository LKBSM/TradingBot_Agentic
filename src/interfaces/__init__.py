# =============================================================================
# INTERFACES MODULE
# =============================================================================
# Abstract interfaces for dependency injection pattern.
#
# Using interfaces (abstract base classes) allows:
# - Swapping implementations without changing dependent code
# - Easy mocking for unit tests
# - Clear contracts between components
# - Loose coupling for maintainability
#
# =============================================================================

from src.interfaces.trade_logger import ITradeLogger
from src.interfaces.agents import (
    IAgent,
    IRiskAgent,
    INewsAgent,
    IMarketRegimeAgent,
)
from src.interfaces.risk import IRiskManager, IKillSwitch

__all__ = [
    'ITradeLogger',
    'IAgent',
    'IRiskAgent',
    'INewsAgent',
    'IMarketRegimeAgent',
    'IRiskManager',
    'IKillSwitch',
]
