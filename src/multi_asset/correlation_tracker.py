# =============================================================================
# CORRELATION TRACKER - Real-time Cross-Asset Correlation Monitoring
# =============================================================================
"""
Tracks correlations between assets in real-time.

Features:
- Rolling correlation calculation
- Correlation breakdown detection (regime change signal)
- Exposure adjustment recommendations
- Historical correlation analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from enum import Enum, auto
import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class CorrelationRegime(Enum):
    """Correlation regime states."""
    STABLE = auto()         # Correlations within normal range
    ELEVATED = auto()       # Correlations higher than normal
    BREAKDOWN = auto()      # Correlations breaking down (crisis signal)
    DECORRELATED = auto()   # Assets moving independently


@dataclass
class CorrelationPair:
    """Correlation data for a pair of assets."""
    asset1: str
    asset2: str

    # Current correlation
    correlation: float = 0.0
    correlation_abs: float = 0.0

    # Rolling correlations
    corr_20: float = 0.0    # 20-bar correlation
    corr_50: float = 0.0    # 50-bar correlation
    corr_100: float = 0.0   # 100-bar correlation

    # Historical stats
    historical_mean: float = 0.0
    historical_std: float = 0.0

    # Z-score (how many std deviations from mean)
    z_score: float = 0.0

    # Regime
    regime: CorrelationRegime = CorrelationRegime.STABLE

    # Timestamps
    last_update: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'pair': f"{self.asset1}/{self.asset2}",
            'correlation': round(self.correlation, 3),
            'corr_20': round(self.corr_20, 3),
            'corr_50': round(self.corr_50, 3),
            'corr_100': round(self.corr_100, 3),
            'z_score': round(self.z_score, 2),
            'regime': self.regime.name
        }


@dataclass
class CorrelationMatrix:
    """Full correlation matrix for all tracked assets."""
    assets: List[str]
    matrix: np.ndarray
    timestamp: datetime = field(default_factory=datetime.now)

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """Get correlation between two assets."""
        try:
            i = self.assets.index(asset1)
            j = self.assets.index(asset2)
            return float(self.matrix[i, j])
        except (ValueError, IndexError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {'assets': self.assets, 'correlations': {}}
        for i, asset1 in enumerate(self.assets):
            for j, asset2 in enumerate(self.assets):
                if i < j:
                    key = f"{asset1}/{asset2}"
                    result['correlations'][key] = round(float(self.matrix[i, j]), 3)
        return result


@dataclass
class CorrelationTrackerConfig:
    """Configuration for correlation tracker."""
    # Rolling windows
    short_window: int = 20
    medium_window: int = 50
    long_window: int = 100

    # Historical lookback for regime detection
    historical_window: int = 252  # ~1 year of daily data

    # Regime thresholds
    breakdown_z_threshold: float = 2.0      # Z-score for breakdown detection
    elevated_z_threshold: float = 1.5       # Z-score for elevated correlation

    # Update frequency
    min_update_interval_sec: int = 60

    # Alerts
    alert_on_breakdown: bool = True
    alert_on_elevated: bool = True


class CorrelationTracker:
    """
    Real-time correlation tracking between assets.

    Monitors correlations for risk management and regime detection.
    """

    def __init__(self, config: Optional[CorrelationTrackerConfig] = None):
        """
        Initialize correlation tracker.

        Args:
            config: Tracker configuration
        """
        self.config = config or CorrelationTrackerConfig()
        self._logger = logging.getLogger("correlation.tracker")

        # Price history for each asset
        self._prices: Dict[str, deque] = {}

        # Return history for each asset
        self._returns: Dict[str, deque] = {}

        # Correlation pairs
        self._pairs: Dict[str, CorrelationPair] = {}

        # Historical correlations for regime detection
        self._historical_corr: Dict[str, deque] = {}

        # Last update time
        self._last_update: Optional[datetime] = None

        # Current correlation matrix
        self._current_matrix: Optional[CorrelationMatrix] = None

    def add_asset(self, symbol: str) -> None:
        """
        Add an asset to track.

        Args:
            symbol: Asset symbol
        """
        if symbol not in self._prices:
            max_len = max(
                self.config.long_window,
                self.config.historical_window
            ) + 10

            self._prices[symbol] = deque(maxlen=max_len)
            self._returns[symbol] = deque(maxlen=max_len)

            # Create pairs with existing assets
            for existing in self._prices.keys():
                if existing != symbol:
                    pair_key = self._get_pair_key(symbol, existing)
                    if pair_key not in self._pairs:
                        self._pairs[pair_key] = CorrelationPair(
                            asset1=min(symbol, existing),
                            asset2=max(symbol, existing)
                        )
                        self._historical_corr[pair_key] = deque(
                            maxlen=self.config.historical_window
                        )

            self._logger.debug(f"Added asset {symbol} to correlation tracker")

    def remove_asset(self, symbol: str) -> None:
        """
        Remove an asset from tracking.

        Args:
            symbol: Asset symbol
        """
        if symbol in self._prices:
            del self._prices[symbol]
            del self._returns[symbol]

            # Remove pairs involving this asset
            pairs_to_remove = [
                k for k in self._pairs.keys()
                if symbol in k
            ]
            for pair_key in pairs_to_remove:
                del self._pairs[pair_key]
                if pair_key in self._historical_corr:
                    del self._historical_corr[pair_key]

    def update_price(self, symbol: str, price: float) -> None:
        """
        Update price for an asset.

        Args:
            symbol: Asset symbol
            price: Current price
        """
        if symbol not in self._prices:
            self.add_asset(symbol)

        prices = self._prices[symbol]

        # Calculate return if we have previous price
        if len(prices) > 0 and prices[-1] > 0:
            ret = (price - prices[-1]) / prices[-1]
            self._returns[symbol].append(ret)

        prices.append(price)

    def update_prices(self, prices: Dict[str, float]) -> None:
        """
        Update prices for multiple assets.

        Args:
            prices: Dictionary of {symbol: price}
        """
        for symbol, price in prices.items():
            self.update_price(symbol, price)

        # Recalculate correlations if enough time has passed
        now = datetime.now()
        if (self._last_update is None or
            (now - self._last_update).total_seconds() >= self.config.min_update_interval_sec):
            self._recalculate_correlations()
            self._last_update = now

    def _recalculate_correlations(self) -> None:
        """Recalculate all correlations."""
        for pair_key, pair in self._pairs.items():
            self._update_pair_correlation(pair_key, pair)

        # Update correlation matrix
        self._update_matrix()

    def _update_pair_correlation(
        self,
        pair_key: str,
        pair: CorrelationPair
    ) -> None:
        """Update correlation for a single pair."""
        returns1 = list(self._returns.get(pair.asset1, []))
        returns2 = list(self._returns.get(pair.asset2, []))

        min_len = min(len(returns1), len(returns2))
        if min_len < 10:
            return

        # Align returns
        r1 = np.array(returns1[-min_len:])
        r2 = np.array(returns2[-min_len:])

        # Calculate rolling correlations
        if min_len >= self.config.short_window:
            pair.corr_20 = self._calculate_correlation(
                r1[-self.config.short_window:],
                r2[-self.config.short_window:]
            )

        if min_len >= self.config.medium_window:
            pair.corr_50 = self._calculate_correlation(
                r1[-self.config.medium_window:],
                r2[-self.config.medium_window:]
            )

        if min_len >= self.config.long_window:
            pair.corr_100 = self._calculate_correlation(
                r1[-self.config.long_window:],
                r2[-self.config.long_window:]
            )

        # Set current correlation (use medium window)
        pair.correlation = pair.corr_50 if pair.corr_50 != 0 else pair.corr_20
        pair.correlation_abs = abs(pair.correlation)

        # Update historical and calculate z-score
        self._historical_corr[pair_key].append(pair.correlation)
        hist = list(self._historical_corr[pair_key])

        if len(hist) >= 20:
            pair.historical_mean = float(np.mean(hist))
            pair.historical_std = float(np.std(hist))

            if pair.historical_std > 0.01:
                pair.z_score = (pair.correlation - pair.historical_mean) / pair.historical_std
            else:
                pair.z_score = 0.0

        # Determine regime
        pair.regime = self._determine_regime(pair)
        pair.last_update = datetime.now()

    def _calculate_correlation(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray
    ) -> float:
        """Calculate correlation between two return series."""
        if len(returns1) < 2 or len(returns2) < 2:
            return 0.0

        try:
            corr_matrix = np.corrcoef(returns1, returns2)
            corr = corr_matrix[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0

    def _determine_regime(self, pair: CorrelationPair) -> CorrelationRegime:
        """Determine correlation regime for a pair."""
        z = pair.z_score

        # Check for breakdown (correlation moving away from historical norm)
        if abs(z) >= self.config.breakdown_z_threshold:
            return CorrelationRegime.BREAKDOWN

        # Check for elevated correlation
        if z >= self.config.elevated_z_threshold:
            return CorrelationRegime.ELEVATED

        # Check for decorrelation
        if pair.correlation_abs < 0.2:
            return CorrelationRegime.DECORRELATED

        return CorrelationRegime.STABLE

    def _update_matrix(self) -> None:
        """Update the full correlation matrix."""
        assets = sorted(set(self._prices.keys()))
        n = len(assets)

        if n < 2:
            return

        matrix = np.eye(n)

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    pair_key = self._get_pair_key(asset1, asset2)
                    pair = self._pairs.get(pair_key)
                    if pair:
                        matrix[i, j] = pair.correlation
                        matrix[j, i] = pair.correlation

        self._current_matrix = CorrelationMatrix(
            assets=assets,
            matrix=matrix,
            timestamp=datetime.now()
        )

    def _get_pair_key(self, asset1: str, asset2: str) -> str:
        """Get consistent pair key regardless of order."""
        return f"{min(asset1, asset2)}/{max(asset1, asset2)}"

    # -------------------------------------------------------------------------
    # PUBLIC API
    # -------------------------------------------------------------------------

    def get_correlation(self, asset1: str, asset2: str) -> float:
        """
        Get current correlation between two assets.

        Args:
            asset1: First asset
            asset2: Second asset

        Returns:
            Correlation coefficient (-1 to 1)
        """
        if asset1 == asset2:
            return 1.0

        pair_key = self._get_pair_key(asset1, asset2)
        pair = self._pairs.get(pair_key)

        return pair.correlation if pair else 0.0

    def get_pair_info(self, asset1: str, asset2: str) -> Optional[CorrelationPair]:
        """
        Get detailed correlation info for a pair.

        Args:
            asset1: First asset
            asset2: Second asset

        Returns:
            CorrelationPair or None
        """
        pair_key = self._get_pair_key(asset1, asset2)
        return self._pairs.get(pair_key)

    def get_correlation_matrix(self) -> Optional[CorrelationMatrix]:
        """Get current correlation matrix."""
        return self._current_matrix

    def get_breakdown_alerts(self) -> List[CorrelationPair]:
        """
        Get pairs with correlation breakdown.

        Returns:
            List of pairs in breakdown regime
        """
        return [
            pair for pair in self._pairs.values()
            if pair.regime == CorrelationRegime.BREAKDOWN
        ]

    def get_high_correlation_pairs(
        self,
        threshold: float = 0.7
    ) -> List[CorrelationPair]:
        """
        Get pairs with high correlation.

        Args:
            threshold: Minimum correlation (absolute value)

        Returns:
            List of highly correlated pairs
        """
        return [
            pair for pair in self._pairs.values()
            if pair.correlation_abs >= threshold
        ]

    def get_exposure_multiplier(
        self,
        asset: str,
        existing_positions: Dict[str, float]
    ) -> float:
        """
        Get recommended position size multiplier based on correlations.

        If adding to positions in correlated assets, reduce size.

        Args:
            asset: Asset to trade
            existing_positions: Dict of {symbol: position_size}

        Returns:
            Multiplier (0.0 to 1.0)
        """
        if not existing_positions:
            return 1.0

        total_correlation_weight = 0.0

        for existing_asset, position in existing_positions.items():
            if position == 0 or existing_asset == asset:
                continue

            corr = self.get_correlation(asset, existing_asset)

            # Weight by absolute correlation and position size
            weight = abs(corr) * abs(position)
            total_correlation_weight += weight

        # Convert to multiplier (higher correlation = lower multiplier)
        # Max reduction of 50%
        if total_correlation_weight > 0:
            reduction = min(0.5, total_correlation_weight * 0.1)
            return 1.0 - reduction

        return 1.0

    def get_status(self) -> Dict[str, Any]:
        """Get tracker status."""
        breakdown_pairs = self.get_breakdown_alerts()
        high_corr_pairs = self.get_high_correlation_pairs()

        return {
            'tracked_assets': list(self._prices.keys()),
            'tracked_pairs': len(self._pairs),
            'last_update': self._last_update.isoformat() if self._last_update else None,
            'breakdown_alerts': len(breakdown_pairs),
            'high_correlation_pairs': len(high_corr_pairs),
            'matrix': self._current_matrix.to_dict() if self._current_matrix else None
        }


def create_correlation_tracker(
    assets: Optional[List[str]] = None
) -> CorrelationTracker:
    """
    Factory function to create a correlation tracker.

    Args:
        assets: Initial list of assets to track

    Returns:
        Configured CorrelationTracker instance
    """
    tracker = CorrelationTracker()

    if assets:
        for asset in assets:
            tracker.add_asset(asset)

    return tracker
