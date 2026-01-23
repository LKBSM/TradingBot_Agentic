# =============================================================================
# ASSET CONFIGURATION - Multi-Asset Trading Profiles
# =============================================================================
"""
Configuration for all supported trading assets.

Each asset has specific characteristics:
- Trading sessions (when most liquid)
- Typical spreads and volatility
- Correlation groups
- News sensitivity
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from enum import Enum, auto
from datetime import time


class AssetClass(Enum):
    """Asset class categories."""
    FOREX_MAJOR = auto()        # EUR, GBP, JPY, CHF vs USD
    FOREX_MINOR = auto()        # Crosses without USD
    FOREX_EXOTIC = auto()       # Emerging market currencies
    COMMODITY_METAL = auto()    # Gold, Silver, Platinum
    COMMODITY_ENERGY = auto()   # Oil, Natural Gas
    INDEX_US = auto()           # US stock indices
    INDEX_EU = auto()           # European indices
    INDEX_ASIA = auto()         # Asian indices
    CRYPTO = auto()             # Cryptocurrencies


class TradingSession(Enum):
    """Major trading sessions."""
    SYDNEY = auto()     # 22:00 - 07:00 UTC
    TOKYO = auto()      # 00:00 - 09:00 UTC
    LONDON = auto()     # 08:00 - 17:00 UTC
    NEW_YORK = auto()   # 13:00 - 22:00 UTC


@dataclass
class SessionTime:
    """Trading session time window."""
    session: TradingSession
    start_utc: time
    end_utc: time
    is_primary: bool = False  # Primary session for this asset


@dataclass
class AssetConfig:
    """Configuration for a tradeable asset."""
    # Identity
    symbol: str                         # e.g., "XAUUSD"
    name: str                           # e.g., "Gold vs US Dollar"
    asset_class: AssetClass

    # Trading characteristics
    pip_value: float                    # Value of 1 pip movement
    pip_digits: int                     # Decimal places for pip
    lot_size: int                       # Standard lot size
    min_lot: float                      # Minimum tradeable lot
    max_lot: float                      # Maximum lot size

    # Cost structure
    typical_spread_pips: float          # Average spread in pips

    # Volatility profile
    avg_daily_range_pips: float         # Average daily range
    high_volatility_threshold: float    # ATR multiplier for high vol
    low_volatility_threshold: float     # ATR multiplier for low vol

    # Fields with defaults must come after fields without defaults
    commission_per_lot: float = 0.0     # Commission if any

    # Trading sessions
    sessions: List[SessionTime] = field(default_factory=list)
    best_trading_hours_utc: List[int] = field(default_factory=list)

    # Correlations
    positive_correlations: List[str] = field(default_factory=list)
    negative_correlations: List[str] = field(default_factory=list)
    correlation_group: str = ""         # e.g., "USD_BASED", "RISK_ON"

    # News sensitivity
    high_impact_events: List[str] = field(default_factory=list)
    news_sensitivity: float = 1.0       # 0.5=low, 1.0=normal, 2.0=high

    # Risk parameters
    max_position_lots: float = 10.0     # Max position size
    max_daily_trades: int = 10          # Max trades per day
    min_trade_interval_min: int = 5     # Minimum time between trades

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'asset_class': self.asset_class.name,
            'pip_value': self.pip_value,
            'typical_spread_pips': self.typical_spread_pips,
            'avg_daily_range_pips': self.avg_daily_range_pips,
            'correlation_group': self.correlation_group,
            'news_sensitivity': self.news_sensitivity
        }


# =============================================================================
# ASSET CONFIGURATIONS
# =============================================================================

ASSET_CONFIGS: Dict[str, AssetConfig] = {
    # -------------------------------------------------------------------------
    # COMMODITIES - METALS
    # -------------------------------------------------------------------------
    'XAUUSD': AssetConfig(
        symbol='XAUUSD',
        name='Gold vs US Dollar',
        asset_class=AssetClass.COMMODITY_METAL,
        pip_value=0.01,
        pip_digits=2,
        lot_size=100,  # 100 oz per lot
        min_lot=0.01,
        max_lot=50.0,
        typical_spread_pips=30,  # ~$0.30
        avg_daily_range_pips=2000,  # ~$20
        high_volatility_threshold=1.5,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0), is_primary=True),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0), is_primary=True),
        ],
        best_trading_hours_utc=[8, 9, 10, 13, 14, 15, 16],
        positive_correlations=['XAGUSD', 'EURUSD'],
        negative_correlations=['DXY', 'USDJPY'],
        correlation_group='SAFE_HAVEN',
        high_impact_events=['FOMC', 'NFP', 'CPI', 'Fed Speech', 'Geopolitical'],
        news_sensitivity=2.0,  # Very sensitive
        max_position_lots=20.0,
        max_daily_trades=8,
        min_trade_interval_min=15
    ),

    'XAGUSD': AssetConfig(
        symbol='XAGUSD',
        name='Silver vs US Dollar',
        asset_class=AssetClass.COMMODITY_METAL,
        pip_value=0.001,
        pip_digits=3,
        lot_size=5000,  # 5000 oz per lot
        min_lot=0.01,
        max_lot=30.0,
        typical_spread_pips=30,
        avg_daily_range_pips=500,
        high_volatility_threshold=1.5,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0), is_primary=True),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0)),
        ],
        best_trading_hours_utc=[8, 9, 10, 13, 14, 15],
        positive_correlations=['XAUUSD', 'EURUSD'],
        negative_correlations=['DXY'],
        correlation_group='SAFE_HAVEN',
        high_impact_events=['FOMC', 'NFP', 'Industrial Data'],
        news_sensitivity=1.8,
        max_position_lots=15.0
    ),

    # -------------------------------------------------------------------------
    # FOREX - MAJORS
    # -------------------------------------------------------------------------
    'EURUSD': AssetConfig(
        symbol='EURUSD',
        name='Euro vs US Dollar',
        asset_class=AssetClass.FOREX_MAJOR,
        pip_value=0.0001,
        pip_digits=4,
        lot_size=100000,
        min_lot=0.01,
        max_lot=100.0,
        typical_spread_pips=1.0,
        avg_daily_range_pips=80,
        high_volatility_threshold=1.3,
        low_volatility_threshold=0.6,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0), is_primary=True),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0)),
        ],
        best_trading_hours_utc=[8, 9, 10, 11, 13, 14, 15],
        positive_correlations=['GBPUSD', 'XAUUSD'],
        negative_correlations=['USDCHF', 'DXY'],
        correlation_group='USD_BASED',
        high_impact_events=['FOMC', 'ECB', 'NFP', 'Eurozone CPI'],
        news_sensitivity=1.5,
        max_position_lots=50.0,
        max_daily_trades=15
    ),

    'GBPUSD': AssetConfig(
        symbol='GBPUSD',
        name='British Pound vs US Dollar',
        asset_class=AssetClass.FOREX_MAJOR,
        pip_value=0.0001,
        pip_digits=4,
        lot_size=100000,
        min_lot=0.01,
        max_lot=100.0,
        typical_spread_pips=1.5,
        avg_daily_range_pips=100,
        high_volatility_threshold=1.4,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0), is_primary=True),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0)),
        ],
        best_trading_hours_utc=[8, 9, 10, 11, 13, 14, 15],
        positive_correlations=['EURUSD'],
        negative_correlations=['USDCHF', 'DXY'],
        correlation_group='USD_BASED',
        high_impact_events=['FOMC', 'BOE', 'NFP', 'UK CPI', 'UK GDP'],
        news_sensitivity=1.6,
        max_position_lots=50.0
    ),

    'USDJPY': AssetConfig(
        symbol='USDJPY',
        name='US Dollar vs Japanese Yen',
        asset_class=AssetClass.FOREX_MAJOR,
        pip_value=0.01,
        pip_digits=2,
        lot_size=100000,
        min_lot=0.01,
        max_lot=100.0,
        typical_spread_pips=1.0,
        avg_daily_range_pips=70,
        high_volatility_threshold=1.3,
        low_volatility_threshold=0.6,
        sessions=[
            SessionTime(TradingSession.TOKYO, time(0, 0), time(9, 0), is_primary=True),
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0)),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0)),
        ],
        best_trading_hours_utc=[0, 1, 2, 3, 8, 9, 13, 14],
        positive_correlations=['DXY', 'US10Y'],
        negative_correlations=['XAUUSD', 'EURUSD'],
        correlation_group='RISK_SENTIMENT',
        high_impact_events=['FOMC', 'BOJ', 'NFP', 'Japan CPI'],
        news_sensitivity=1.4,
        max_position_lots=50.0
    ),

    'USDCHF': AssetConfig(
        symbol='USDCHF',
        name='US Dollar vs Swiss Franc',
        asset_class=AssetClass.FOREX_MAJOR,
        pip_value=0.0001,
        pip_digits=4,
        lot_size=100000,
        min_lot=0.01,
        max_lot=100.0,
        typical_spread_pips=1.5,
        avg_daily_range_pips=60,
        high_volatility_threshold=1.3,
        low_volatility_threshold=0.6,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0), is_primary=True),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0)),
        ],
        best_trading_hours_utc=[8, 9, 10, 13, 14, 15],
        positive_correlations=['DXY'],
        negative_correlations=['EURUSD', 'XAUUSD'],
        correlation_group='SAFE_HAVEN',
        high_impact_events=['FOMC', 'SNB', 'NFP'],
        news_sensitivity=1.3,
        max_position_lots=50.0
    ),

    'AUDUSD': AssetConfig(
        symbol='AUDUSD',
        name='Australian Dollar vs US Dollar',
        asset_class=AssetClass.FOREX_MAJOR,
        pip_value=0.0001,
        pip_digits=4,
        lot_size=100000,
        min_lot=0.01,
        max_lot=100.0,
        typical_spread_pips=1.2,
        avg_daily_range_pips=70,
        high_volatility_threshold=1.4,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.SYDNEY, time(22, 0), time(7, 0)),
            SessionTime(TradingSession.TOKYO, time(0, 0), time(9, 0), is_primary=True),
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0)),
        ],
        best_trading_hours_utc=[0, 1, 2, 3, 8, 9],
        positive_correlations=['NZDUSD', 'XAUUSD'],
        negative_correlations=['DXY'],
        correlation_group='RISK_ON',
        high_impact_events=['FOMC', 'RBA', 'China Data', 'Australia Employment'],
        news_sensitivity=1.4,
        max_position_lots=50.0
    ),

    # -------------------------------------------------------------------------
    # INDICES
    # -------------------------------------------------------------------------
    'US30': AssetConfig(
        symbol='US30',
        name='Dow Jones Industrial Average',
        asset_class=AssetClass.INDEX_US,
        pip_value=1.0,
        pip_digits=0,
        lot_size=1,
        min_lot=0.1,
        max_lot=50.0,
        typical_spread_pips=3.0,
        avg_daily_range_pips=300,
        high_volatility_threshold=1.5,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.NEW_YORK, time(13, 30), time(20, 0), is_primary=True),
        ],
        best_trading_hours_utc=[13, 14, 15, 16, 17, 18, 19],
        positive_correlations=['SPX500', 'NAS100'],
        negative_correlations=['VIX'],
        correlation_group='RISK_ON',
        high_impact_events=['FOMC', 'NFP', 'US GDP', 'Earnings'],
        news_sensitivity=1.5,
        max_position_lots=20.0
    ),

    'SPX500': AssetConfig(
        symbol='SPX500',
        name='S&P 500 Index',
        asset_class=AssetClass.INDEX_US,
        pip_value=0.1,
        pip_digits=1,
        lot_size=1,
        min_lot=0.1,
        max_lot=50.0,
        typical_spread_pips=5.0,
        avg_daily_range_pips=500,
        high_volatility_threshold=1.5,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.NEW_YORK, time(13, 30), time(20, 0), is_primary=True),
        ],
        best_trading_hours_utc=[13, 14, 15, 16, 17, 18, 19],
        positive_correlations=['US30', 'NAS100'],
        negative_correlations=['VIX', 'XAUUSD'],
        correlation_group='RISK_ON',
        high_impact_events=['FOMC', 'NFP', 'US GDP', 'CPI'],
        news_sensitivity=1.6,
        max_position_lots=20.0
    ),

    'NAS100': AssetConfig(
        symbol='NAS100',
        name='Nasdaq 100 Index',
        asset_class=AssetClass.INDEX_US,
        pip_value=0.1,
        pip_digits=1,
        lot_size=1,
        min_lot=0.1,
        max_lot=50.0,
        typical_spread_pips=10.0,
        avg_daily_range_pips=200,
        high_volatility_threshold=1.6,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.NEW_YORK, time(13, 30), time(20, 0), is_primary=True),
        ],
        best_trading_hours_utc=[13, 14, 15, 16, 17, 18, 19],
        positive_correlations=['SPX500', 'US30'],
        negative_correlations=['VIX'],
        correlation_group='RISK_ON',
        high_impact_events=['FOMC', 'Tech Earnings', 'NFP'],
        news_sensitivity=1.7,
        max_position_lots=15.0
    ),

    # -------------------------------------------------------------------------
    # ENERGY
    # -------------------------------------------------------------------------
    'USOIL': AssetConfig(
        symbol='USOIL',
        name='West Texas Intermediate Crude Oil',
        asset_class=AssetClass.COMMODITY_ENERGY,
        pip_value=0.01,
        pip_digits=2,
        lot_size=1000,  # barrels
        min_lot=0.01,
        max_lot=50.0,
        typical_spread_pips=4.0,
        avg_daily_range_pips=200,
        high_volatility_threshold=1.6,
        low_volatility_threshold=0.5,
        sessions=[
            SessionTime(TradingSession.LONDON, time(8, 0), time(17, 0)),
            SessionTime(TradingSession.NEW_YORK, time(13, 0), time(22, 0), is_primary=True),
        ],
        best_trading_hours_utc=[13, 14, 15, 16, 17],
        positive_correlations=['UKOIL', 'USDCAD'],
        negative_correlations=['DXY'],
        correlation_group='ENERGY',
        high_impact_events=['EIA Inventory', 'OPEC', 'Geopolitical'],
        news_sensitivity=2.0,
        max_position_lots=20.0
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_asset_config(symbol: str) -> Optional[AssetConfig]:
    """
    Get configuration for an asset.

    Args:
        symbol: Asset symbol (e.g., 'XAUUSD')

    Returns:
        AssetConfig or None if not found
    """
    return ASSET_CONFIGS.get(symbol.upper())


def get_all_assets() -> List[str]:
    """Get list of all supported asset symbols."""
    return list(ASSET_CONFIGS.keys())


def get_assets_by_class(asset_class: AssetClass) -> List[str]:
    """
    Get assets by asset class.

    Args:
        asset_class: Asset class to filter by

    Returns:
        List of asset symbols
    """
    return [
        symbol for symbol, config in ASSET_CONFIGS.items()
        if config.asset_class == asset_class
    ]


def get_correlated_assets(symbol: str, min_correlation: float = 0.5) -> Dict[str, float]:
    """
    Get assets correlated with the given symbol.

    Args:
        symbol: Asset symbol
        min_correlation: Minimum correlation threshold

    Returns:
        Dictionary of {symbol: correlation_estimate}
    """
    config = get_asset_config(symbol)
    if not config:
        return {}

    result = {}

    # Add positive correlations (estimated at 0.7)
    for corr_symbol in config.positive_correlations:
        if corr_symbol in ASSET_CONFIGS:
            result[corr_symbol] = 0.7

    # Add negative correlations (estimated at -0.7)
    for corr_symbol in config.negative_correlations:
        if corr_symbol in ASSET_CONFIGS:
            result[corr_symbol] = -0.7

    return result


def is_trading_session_active(symbol: str, hour_utc: int) -> bool:
    """
    Check if any trading session is active for the asset.

    Args:
        symbol: Asset symbol
        hour_utc: Current hour in UTC

    Returns:
        True if in active trading session
    """
    config = get_asset_config(symbol)
    if not config:
        return True  # Default to active

    for session in config.sessions:
        start_hour = session.start_utc.hour
        end_hour = session.end_utc.hour

        if start_hour <= end_hour:
            if start_hour <= hour_utc < end_hour:
                return True
        else:  # Session spans midnight
            if hour_utc >= start_hour or hour_utc < end_hour:
                return True

    return False
