# =============================================================================
# MULTI-ASSET MODULE - Sprint 3 Multi-Asset Support
# =============================================================================
"""
Multi-asset trading support for the trading bot.

Enables trading across multiple asset classes:
- Forex: EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD
- Commodities: XAUUSD (Gold), XAGUSD (Silver), Oil
- Indices: US30, SPX500, NAS100

Each asset has:
- Asset-specific regime detection
- Correlation tracking with other assets
- Custom volatility profiles
- Spread/cost considerations
"""

from .asset_config import (
    AssetConfig,
    AssetClass,
    TradingSession,
    get_asset_config,
    get_all_assets,
    get_assets_by_class
)

from .asset_manager import (
    MultiAssetManager,
    AssetState,
    create_multi_asset_manager
)

from .correlation_tracker import (
    CorrelationTracker,
    CorrelationMatrix,
    create_correlation_tracker
)

__all__ = [
    # Config
    'AssetConfig',
    'AssetClass',
    'TradingSession',
    'get_asset_config',
    'get_all_assets',
    'get_assets_by_class',
    # Manager
    'MultiAssetManager',
    'AssetState',
    'create_multi_asset_manager',
    # Correlation
    'CorrelationTracker',
    'CorrelationMatrix',
    'create_correlation_tracker',
]

__version__ = '1.0.0'
