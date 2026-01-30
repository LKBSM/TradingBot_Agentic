# =============================================================================
# LIVE TRADING MODULE - Production MT5 Integration
# =============================================================================
# This module provides real-time trading capabilities via MetaTrader 5.
#
# Components:
#   - mt5_connector.py: Core MT5 API wrapper
#   - order_manager.py: Order lifecycle management
#   - position_sync.py: Position synchronization
#   - alerting.py: Multi-channel alerting system
#   - live_risk_manager.py: Real-time risk enforcement
#
# =============================================================================

from .mt5_connector import (
    MT5Connector,
    MT5ConnectionError,
    MT5OrderError,
    OrderResult,
    PositionInfo,
    AccountInfo
)

from .order_manager import (
    OrderManager,
    OrderType,
    OrderStatus,
    ManagedOrder
)

from .alerting import (
    AlertManager,
    AlertLevel,
    AlertChannel,
    TelegramAlert,
    DiscordAlert,
    EmailAlert
)

from .live_risk_manager import (
    LiveRiskManager,
    LiveRiskConfig
)

__all__ = [
    # MT5 Connector
    'MT5Connector',
    'MT5ConnectionError',
    'MT5OrderError',
    'OrderResult',
    'PositionInfo',
    'AccountInfo',

    # Order Manager
    'OrderManager',
    'OrderType',
    'OrderStatus',
    'ManagedOrder',

    # Alerting
    'AlertManager',
    'AlertLevel',
    'AlertChannel',
    'TelegramAlert',
    'DiscordAlert',
    'EmailAlert',

    # Risk
    'LiveRiskManager',
    'LiveRiskConfig',
]
