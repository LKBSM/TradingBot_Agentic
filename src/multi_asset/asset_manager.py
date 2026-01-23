# =============================================================================
# ASSET MANAGER - Multi-Asset Portfolio Management
# =============================================================================
"""
Manages multiple assets for trading, including:
- Asset state tracking (regime, volatility, positions)
- Position sizing across assets
- Exposure management
- Cross-asset risk limits
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import numpy as np

from .asset_config import (
    AssetConfig,
    AssetClass,
    get_asset_config,
    get_all_assets,
    ASSET_CONFIGS
)

logger = logging.getLogger(__name__)


class AssetRegime(Enum):
    """Market regime for an asset."""
    STRONG_UPTREND = auto()
    UPTREND = auto()
    RANGING = auto()
    DOWNTREND = auto()
    STRONG_DOWNTREND = auto()
    HIGH_VOLATILITY = auto()
    UNKNOWN = auto()


@dataclass
class AssetState:
    """Current state of an asset."""
    symbol: str
    last_price: float = 0.0
    bid: float = 0.0
    ask: float = 0.0
    spread_pips: float = 0.0

    # Regime
    regime: AssetRegime = AssetRegime.UNKNOWN
    regime_strength: float = 0.0  # 0-1

    # Volatility
    current_atr: float = 0.0
    volatility_percentile: float = 50.0  # 0-100

    # Position
    position_lots: float = 0.0
    position_direction: int = 0  # 1=long, -1=short, 0=flat
    unrealized_pnl: float = 0.0
    entry_price: float = 0.0

    # Session
    is_session_active: bool = True
    session_name: str = ""

    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)
    last_trade: Optional[datetime] = None

    # Daily stats
    daily_trades: int = 0
    daily_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'last_price': self.last_price,
            'spread_pips': self.spread_pips,
            'regime': self.regime.name,
            'regime_strength': self.regime_strength,
            'volatility_percentile': self.volatility_percentile,
            'position_lots': self.position_lots,
            'position_direction': self.position_direction,
            'unrealized_pnl': self.unrealized_pnl,
            'is_session_active': self.is_session_active,
            'daily_trades': self.daily_trades,
            'daily_pnl': self.daily_pnl
        }


@dataclass
class PortfolioLimits:
    """Portfolio-wide risk limits."""
    max_total_exposure_usd: float = 100000.0
    max_single_asset_exposure_pct: float = 30.0  # % of total
    max_correlated_exposure_pct: float = 50.0    # % in correlated assets
    max_daily_loss_usd: float = 5000.0
    max_open_positions: int = 5
    max_trades_per_day: int = 20


class MultiAssetManager:
    """
    Manages multiple trading assets.

    Tracks state, enforces limits, and coordinates trading across assets.
    """

    def __init__(
        self,
        assets: Optional[List[str]] = None,
        limits: Optional[PortfolioLimits] = None
    ):
        """
        Initialize multi-asset manager.

        Args:
            assets: List of asset symbols to manage
            limits: Portfolio-wide limits
        """
        self._logger = logging.getLogger("multi_asset.manager")

        # Portfolio limits
        self.limits = limits or PortfolioLimits()

        # Initialize assets
        self._assets: Dict[str, AssetConfig] = {}
        self._states: Dict[str, AssetState] = {}

        # Portfolio tracking
        self._total_exposure_usd: float = 0.0
        self._daily_pnl: float = 0.0
        self._daily_trades: int = 0
        self._last_reset: datetime = datetime.now()

        # Price history for correlation calculation
        self._price_history: Dict[str, List[float]] = {}
        self._history_length: int = 100

        # Add assets after all attributes are initialized
        asset_list = assets or ['XAUUSD']  # Default to gold
        for symbol in asset_list:
            self.add_asset(symbol)

    def add_asset(self, symbol: str) -> bool:
        """
        Add an asset to management.

        Args:
            symbol: Asset symbol

        Returns:
            True if added successfully
        """
        config = get_asset_config(symbol)
        if not config:
            self._logger.warning(f"Unknown asset: {symbol}")
            return False

        if symbol in self._assets:
            self._logger.debug(f"Asset {symbol} already managed")
            return True

        self._assets[symbol] = config
        self._states[symbol] = AssetState(symbol=symbol)
        self._price_history[symbol] = []

        self._logger.info(f"Added asset: {symbol} ({config.name})")
        return True

    def remove_asset(self, symbol: str) -> bool:
        """
        Remove an asset from management.

        Args:
            symbol: Asset symbol

        Returns:
            True if removed
        """
        if symbol not in self._assets:
            return False

        # Check for open positions
        state = self._states.get(symbol)
        if state and state.position_lots != 0:
            self._logger.warning(f"Cannot remove {symbol}: has open position")
            return False

        del self._assets[symbol]
        del self._states[symbol]
        if symbol in self._price_history:
            del self._price_history[symbol]

        self._logger.info(f"Removed asset: {symbol}")
        return True

    def update_price(
        self,
        symbol: str,
        bid: float,
        ask: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """
        Update price for an asset.

        Args:
            symbol: Asset symbol
            bid: Bid price
            ask: Ask price
            timestamp: Price timestamp
        """
        if symbol not in self._states:
            return

        state = self._states[symbol]
        config = self._assets.get(symbol)

        state.bid = bid
        state.ask = ask
        state.last_price = (bid + ask) / 2
        state.spread_pips = (ask - bid) / config.pip_value if config else 0
        state.last_update = timestamp or datetime.now()

        # Update price history
        if symbol in self._price_history:
            self._price_history[symbol].append(state.last_price)
            if len(self._price_history[symbol]) > self._history_length:
                self._price_history[symbol] = self._price_history[symbol][-self._history_length:]

        # Update unrealized PnL if in position
        if state.position_lots != 0 and state.entry_price > 0:
            if state.position_direction > 0:  # Long
                state.unrealized_pnl = (state.last_price - state.entry_price) * state.position_lots * (config.lot_size if config else 1)
            else:  # Short
                state.unrealized_pnl = (state.entry_price - state.last_price) * state.position_lots * (config.lot_size if config else 1)

    def update_regime(
        self,
        symbol: str,
        regime: AssetRegime,
        strength: float = 0.5
    ) -> None:
        """
        Update market regime for an asset.

        Args:
            symbol: Asset symbol
            regime: Current regime
            strength: Regime strength (0-1)
        """
        if symbol in self._states:
            self._states[symbol].regime = regime
            self._states[symbol].regime_strength = strength

    def update_volatility(
        self,
        symbol: str,
        atr: float,
        percentile: float = 50.0
    ) -> None:
        """
        Update volatility metrics for an asset.

        Args:
            symbol: Asset symbol
            atr: Current ATR
            percentile: Volatility percentile (0-100)
        """
        if symbol in self._states:
            self._states[symbol].current_atr = atr
            self._states[symbol].volatility_percentile = percentile

    def get_state(self, symbol: str) -> Optional[AssetState]:
        """Get current state of an asset."""
        return self._states.get(symbol)

    def get_all_states(self) -> Dict[str, AssetState]:
        """Get states of all managed assets."""
        return self._states.copy()

    def get_config(self, symbol: str) -> Optional[AssetConfig]:
        """Get configuration for an asset."""
        return self._assets.get(symbol)

    # -------------------------------------------------------------------------
    # POSITION MANAGEMENT
    # -------------------------------------------------------------------------

    def can_open_position(
        self,
        symbol: str,
        direction: int,
        lots: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.

        Args:
            symbol: Asset symbol
            direction: 1=long, -1=short
            lots: Position size in lots

        Returns:
            Tuple of (can_open, reason)
        """
        # Check daily reset
        self._check_daily_reset()

        config = self._assets.get(symbol)
        state = self._states.get(symbol)

        if not config or not state:
            return False, f"Unknown asset: {symbol}"

        # Check if session is active
        if not state.is_session_active:
            return False, f"Trading session not active for {symbol}"

        # Check lot size limits
        if lots < config.min_lot:
            return False, f"Lot size {lots} below minimum {config.min_lot}"
        if lots > config.max_lot:
            return False, f"Lot size {lots} above maximum {config.max_lot}"

        # Check max position for this asset
        total_lots = abs(state.position_lots) + lots
        if total_lots > config.max_position_lots:
            return False, f"Would exceed max position {config.max_position_lots} lots for {symbol}"

        # Check daily trade limit for this asset
        if state.daily_trades >= config.max_daily_trades:
            return False, f"Daily trade limit ({config.max_daily_trades}) reached for {symbol}"

        # Check min trade interval
        if state.last_trade:
            elapsed = (datetime.now() - state.last_trade).total_seconds() / 60
            if elapsed < config.min_trade_interval_min:
                return False, f"Min trade interval not met ({config.min_trade_interval_min} min)"

        # Check portfolio limits
        open_positions = sum(1 for s in self._states.values() if s.position_lots != 0)
        if state.position_lots == 0 and open_positions >= self.limits.max_open_positions:
            return False, f"Max open positions ({self.limits.max_open_positions}) reached"

        if self._daily_trades >= self.limits.max_trades_per_day:
            return False, f"Daily portfolio trade limit ({self.limits.max_trades_per_day}) reached"

        # Check daily loss limit
        if self._daily_pnl <= -self.limits.max_daily_loss_usd:
            return False, f"Daily loss limit (${self.limits.max_daily_loss_usd}) reached"

        # Check exposure limits
        exposure_usd = lots * (config.lot_size * state.last_price if state.last_price > 0 else 10000)
        if self._total_exposure_usd + exposure_usd > self.limits.max_total_exposure_usd:
            return False, f"Would exceed max total exposure ${self.limits.max_total_exposure_usd}"

        # Check single asset exposure
        asset_exposure_pct = (exposure_usd / self.limits.max_total_exposure_usd) * 100
        if asset_exposure_pct > self.limits.max_single_asset_exposure_pct:
            return False, f"Would exceed max single asset exposure {self.limits.max_single_asset_exposure_pct}%"

        return True, "OK"

    def record_position_open(
        self,
        symbol: str,
        direction: int,
        lots: float,
        entry_price: float
    ) -> None:
        """
        Record a position opening.

        Args:
            symbol: Asset symbol
            direction: 1=long, -1=short
            lots: Position size
            entry_price: Entry price
        """
        state = self._states.get(symbol)
        config = self._assets.get(symbol)

        if not state or not config:
            return

        # Update state
        if state.position_direction == direction:
            # Adding to position
            total_lots = state.position_lots + lots
            state.entry_price = (
                (state.entry_price * state.position_lots + entry_price * lots) / total_lots
            )
            state.position_lots = total_lots
        elif state.position_lots == 0:
            # New position
            state.position_lots = lots
            state.position_direction = direction
            state.entry_price = entry_price
        else:
            # Partial close or reversal
            remaining = state.position_lots - lots
            if remaining >= 0:
                state.position_lots = remaining
                if remaining == 0:
                    state.position_direction = 0
                    state.entry_price = 0
            else:
                # Reversal
                state.position_lots = abs(remaining)
                state.position_direction = direction
                state.entry_price = entry_price

        state.last_trade = datetime.now()
        state.daily_trades += 1
        self._daily_trades += 1

        # Update exposure
        self._update_total_exposure()

    def record_position_close(
        self,
        symbol: str,
        lots: float,
        close_price: float
    ) -> float:
        """
        Record a position closing.

        Args:
            symbol: Asset symbol
            lots: Lots closed
            close_price: Close price

        Returns:
            Realized PnL
        """
        state = self._states.get(symbol)
        config = self._assets.get(symbol)

        if not state or not config:
            return 0.0

        if state.position_lots == 0:
            return 0.0

        # Calculate PnL
        if state.position_direction > 0:  # Long
            pnl_per_lot = (close_price - state.entry_price) * config.lot_size
        else:  # Short
            pnl_per_lot = (state.entry_price - close_price) * config.lot_size

        realized_pnl = pnl_per_lot * min(lots, state.position_lots)

        # Update state
        state.position_lots -= lots
        if state.position_lots <= 0:
            state.position_lots = 0
            state.position_direction = 0
            state.entry_price = 0
            state.unrealized_pnl = 0

        state.last_trade = datetime.now()
        state.daily_pnl += realized_pnl
        self._daily_pnl += realized_pnl

        # Update exposure
        self._update_total_exposure()

        return realized_pnl

    def _update_total_exposure(self) -> None:
        """Update total portfolio exposure."""
        total = 0.0
        for symbol, state in self._states.items():
            if state.position_lots > 0:
                config = self._assets.get(symbol)
                if config and state.last_price > 0:
                    total += state.position_lots * config.lot_size * state.last_price
        self._total_exposure_usd = total

    def _check_daily_reset(self) -> None:
        """Reset daily counters if new day."""
        now = datetime.now()
        if now.date() != self._last_reset.date():
            self._daily_pnl = 0.0
            self._daily_trades = 0
            for state in self._states.values():
                state.daily_trades = 0
                state.daily_pnl = 0.0
            self._last_reset = now

    # -------------------------------------------------------------------------
    # CORRELATION
    # -------------------------------------------------------------------------

    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """
        Get correlation between two assets.

        Args:
            symbol1: First asset
            symbol2: Second asset

        Returns:
            Correlation coefficient (-1 to 1)
        """
        prices1 = self._price_history.get(symbol1, [])
        prices2 = self._price_history.get(symbol2, [])

        if len(prices1) < 20 or len(prices2) < 20:
            # Not enough data, use config estimates
            config1 = self._assets.get(symbol1)
            if config1:
                if symbol2 in config1.positive_correlations:
                    return 0.7
                if symbol2 in config1.negative_correlations:
                    return -0.7
            return 0.0

        # Calculate returns
        min_len = min(len(prices1), len(prices2))
        returns1 = np.diff(prices1[-min_len:]) / prices1[-min_len:-1]
        returns2 = np.diff(prices2[-min_len:]) / prices2[-min_len:-1]

        # Calculate correlation
        if len(returns1) > 0 and len(returns2) > 0:
            return float(np.corrcoef(returns1, returns2)[0, 1])

        return 0.0

    def get_correlated_exposure(self, symbol: str, threshold: float = 0.5) -> float:
        """
        Get total exposure in assets correlated with the given symbol.

        Args:
            symbol: Asset symbol
            threshold: Correlation threshold

        Returns:
            Total correlated exposure in USD
        """
        total_correlated = 0.0
        state = self._states.get(symbol)
        config = self._assets.get(symbol)

        if not state or not config:
            return 0.0

        for other_symbol, other_state in self._states.items():
            if other_symbol == symbol:
                continue
            if other_state.position_lots == 0:
                continue

            corr = self.get_correlation(symbol, other_symbol)
            if abs(corr) >= threshold:
                other_config = self._assets.get(other_symbol)
                if other_config and other_state.last_price > 0:
                    exposure = other_state.position_lots * other_config.lot_size * other_state.last_price
                    # Weight by correlation
                    total_correlated += exposure * abs(corr)

        return total_correlated

    # -------------------------------------------------------------------------
    # PORTFOLIO STATUS
    # -------------------------------------------------------------------------

    def get_portfolio_status(self) -> Dict[str, Any]:
        """Get comprehensive portfolio status."""
        self._check_daily_reset()

        open_positions = []
        for symbol, state in self._states.items():
            if state.position_lots > 0:
                open_positions.append({
                    'symbol': symbol,
                    'direction': 'LONG' if state.position_direction > 0 else 'SHORT',
                    'lots': state.position_lots,
                    'entry_price': state.entry_price,
                    'current_price': state.last_price,
                    'unrealized_pnl': state.unrealized_pnl
                })

        return {
            'managed_assets': list(self._assets.keys()),
            'total_exposure_usd': self._total_exposure_usd,
            'daily_pnl': self._daily_pnl,
            'daily_trades': self._daily_trades,
            'open_positions': open_positions,
            'limits': {
                'max_exposure': self.limits.max_total_exposure_usd,
                'max_daily_loss': self.limits.max_daily_loss_usd,
                'max_positions': self.limits.max_open_positions
            }
        }


def create_multi_asset_manager(
    assets: Optional[List[str]] = None,
    max_exposure_usd: float = 100000.0
) -> MultiAssetManager:
    """
    Factory function to create a multi-asset manager.

    Args:
        assets: List of asset symbols to manage
        max_exposure_usd: Maximum total exposure

    Returns:
        Configured MultiAssetManager instance
    """
    limits = PortfolioLimits(max_total_exposure_usd=max_exposure_usd)
    return MultiAssetManager(assets=assets, limits=limits)
