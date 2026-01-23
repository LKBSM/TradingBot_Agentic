# =============================================================================
# MARKET REGIME DETECTION AGENT - Intelligent Market Context Classification
# =============================================================================
# This agent is CRITICAL for profitability. It answers the most important
# trading question: "What type of market are we in right now?"
#
# === WHY THIS MATTERS ===
# - Trend-following strategies FAIL in ranging markets
# - Mean-reversion strategies FAIL in trending markets
# - Without regime awareness, you're wrong 50% of the time
#
# === REGIMES DETECTED ===
# 1. STRONG_UPTREND:   Clear bullish momentum, buy dips
# 2. WEAK_UPTREND:     Bullish bias but choppy
# 3. STRONG_DOWNTREND: Clear bearish momentum, sell rallies
# 4. WEAK_DOWNTREND:   Bearish bias but choppy
# 5. RANGING:          No clear direction, mean reversion works
# 6. HIGH_VOLATILITY:  Dangerous, reduce position size
# 7. LOW_VOLATILITY:   Breakout imminent, prepare
# 8. TRANSITION:       Regime changing, be cautious
#
# === INDICATORS USED ===
# - ADX (Average Directional Index) for trend strength
# - Bollinger Band Width for volatility
# - Moving Average alignment (9/21/50)
# - RSI for momentum confirmation
# - Price structure (higher highs/lows)
# - Volume profile for confirmation
#
# =============================================================================

import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from collections import deque
from datetime import datetime
from enum import Enum, auto
import logging

# Import base classes
from src.agents.base_agent import BaseAgent, AgentState, AgentCapability
from src.agents.events import AgentEvent, EventType


# =============================================================================
# MARKET REGIME ENUMS
# =============================================================================

class RegimeType(Enum):
    """Detailed market regime classification."""
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    WEAK_DOWNTREND = "weak_downtrend"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"
    UNKNOWN = "unknown"


class TrendDirection(Enum):
    """Simple trend direction."""
    UP = 1
    DOWN = -1
    NEUTRAL = 0


class VolatilityState(Enum):
    """Volatility classification."""
    VERY_LOW = "very_low"
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    EXTREME = "extreme"


# =============================================================================
# REGIME ANALYSIS RESULT
# =============================================================================

@dataclass
class RegimeAnalysis:
    """Complete regime analysis result."""
    regime: RegimeType
    confidence: float  # 0-1
    trend_direction: TrendDirection
    trend_strength: float  # 0-1
    volatility_state: VolatilityState
    volatility_percentile: float  # 0-100
    momentum: float  # -1 to 1
    is_transitioning: bool
    recommended_strategy: str
    position_size_multiplier: float  # 0.25-1.5
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime': self.regime.value,
            'confidence': self.confidence,
            'trend_direction': self.trend_direction.value,
            'trend_strength': self.trend_strength,
            'volatility_state': self.volatility_state.value,
            'volatility_percentile': self.volatility_percentile,
            'momentum': self.momentum,
            'is_transitioning': self.is_transitioning,
            'recommended_strategy': self.recommended_strategy,
            'position_size_multiplier': self.position_size_multiplier,
            'details': self.details
        }


# =============================================================================
# TECHNICAL INDICATORS CALCULATOR
# =============================================================================

class TechnicalIndicators:
    """
    Calculate technical indicators for regime detection.

    Uses numpy for speed - no pandas dependency in hot path.
    """

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> float:
        """Simple Moving Average."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0
        return np.mean(prices[-period:])

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(prices) < period:
            return prices

        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[0] = prices[0]

        for i in range(1, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]

        return ema

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> float:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return 50  # Neutral

        deltas = np.diff(prices[-period-1:])
        gains = np.maximum(deltas, 0)
        losses = np.abs(np.minimum(deltas, 0))

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
        """Average True Range."""
        if len(closes) < period + 1:
            return np.mean(highs - lows) if len(highs) > 0 else 0

        tr_list = []
        for i in range(1, min(period + 1, len(closes))):
            high_low = highs[-i] - lows[-i]
            high_close = abs(highs[-i] - closes[-i-1])
            low_close = abs(lows[-i] - closes[-i-1])
            tr_list.append(max(high_low, high_close, low_close))

        return np.mean(tr_list)

    @staticmethod
    def adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[float, float, float]:
        """
        Average Directional Index.

        Returns: (ADX, +DI, -DI)
        """
        if len(closes) < period + 1:
            return 0, 0, 0

        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        tr_list = []

        for i in range(1, min(period + 1, len(closes))):
            up_move = highs[-i] - highs[-i-1]
            down_move = lows[-i-1] - lows[-i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

            # True Range
            high_low = highs[-i] - lows[-i]
            high_close = abs(highs[-i] - closes[-i-1])
            low_close = abs(lows[-i] - closes[-i-1])
            tr_list.append(max(high_low, high_close, low_close))

        # Average TR
        avg_tr = np.mean(tr_list)
        if avg_tr == 0:
            return 0, 0, 0

        # +DI and -DI
        plus_di = 100 * np.mean(plus_dm) / avg_tr
        minus_di = 100 * np.mean(minus_dm) / avg_tr

        # DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            return 0, plus_di, minus_di

        dx = 100 * abs(plus_di - minus_di) / di_sum

        # ADX is smoothed DX (simplified)
        adx = dx

        return adx, plus_di, minus_di

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float, float]:
        """
        Bollinger Bands.

        Returns: (upper, middle, lower, bandwidth_percentile)
        """
        if len(prices) < period:
            return prices[-1] * 1.02, prices[-1], prices[-1] * 0.98, 50

        middle = np.mean(prices[-period:])
        std = np.std(prices[-period:])

        upper = middle + std_dev * std
        lower = middle - std_dev * std

        # Bandwidth as percentage
        bandwidth = (upper - lower) / middle * 100

        return upper, middle, lower, bandwidth

    @staticmethod
    def price_position(current_price: float, sma_short: float, sma_long: float) -> float:
        """
        Price position relative to moving averages.

        Returns: -1 to 1 (negative = below MAs, positive = above MAs)
        """
        avg_ma = (sma_short + sma_long) / 2
        distance = (current_price - avg_ma) / avg_ma
        return np.clip(distance * 10, -1, 1)

    @staticmethod
    def higher_highs_higher_lows(highs: np.ndarray, lows: np.ndarray, lookback: int = 10) -> Tuple[bool, bool]:
        """
        Detect if we have higher highs and higher lows (uptrend structure).

        Returns: (has_higher_highs, has_higher_lows)
        """
        if len(highs) < lookback:
            return False, False

        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]

        # Find local peaks and troughs
        mid = lookback // 2

        first_half_high = np.max(recent_highs[:mid])
        second_half_high = np.max(recent_highs[mid:])

        first_half_low = np.min(recent_lows[:mid])
        second_half_low = np.min(recent_lows[mid:])

        higher_highs = second_half_high > first_half_high
        higher_lows = second_half_low > first_half_low

        return higher_highs, higher_lows


# =============================================================================
# MARKET REGIME DETECTION AGENT
# =============================================================================

class MarketRegimeAgent(BaseAgent):
    """
    Intelligent Market Regime Detection Agent.

    This agent continuously analyzes market data to classify the current
    trading environment. It provides actionable regime information that
    other agents (like RiskSentinel) use to adjust their behavior.

    === KEY FEATURES ===
    1. Multi-indicator regime detection
    2. Confidence scoring for each classification
    3. Regime transition detection
    4. Strategy recommendations per regime
    5. Position size multipliers

    === USAGE ===
    ```python
    agent = MarketRegimeAgent()
    agent.start()

    # Update with new price data
    analysis = agent.analyze(
        prices=price_array,
        highs=high_array,
        lows=low_array,
        volumes=volume_array
    )

    print(f"Regime: {analysis.regime.value}")
    print(f"Strategy: {analysis.recommended_strategy}")
    print(f"Size multiplier: {analysis.position_size_multiplier}")
    ```
    """

    def __init__(self, name: str = "MarketRegimeAgent", config: Optional[Dict] = None):
        """Initialize the Market Regime Agent."""
        super().__init__(name=name, config=config or {})

        # Configuration
        self._config = config or {}
        self._lookback = self._config.get('lookback', 100)
        self._min_data_points = self._config.get('min_data_points', 50)

        # Data buffers
        self._prices: deque = deque(maxlen=self._lookback * 2)
        self._highs: deque = deque(maxlen=self._lookback * 2)
        self._lows: deque = deque(maxlen=self._lookback * 2)
        self._volumes: deque = deque(maxlen=self._lookback * 2)
        self._atr_history: deque = deque(maxlen=self._lookback)

        # Regime tracking
        self._current_regime = RegimeType.UNKNOWN
        self._regime_history: deque = deque(maxlen=50)
        self._regime_duration = 0
        self._last_regime_change_step = 0
        self._current_step = 0

        # Transition detection
        self._regime_stability_threshold = 5  # Min bars before regime can change
        self._transition_warning_count = 0

        # Volatility percentile history (for relative comparison)
        self._volatility_history: deque = deque(maxlen=500)

        # Logging
        self._logger = logging.getLogger(f"agent.{self.full_id}")

    # =========================================================================
    # LIFECYCLE METHODS
    # =========================================================================

    def initialize(self) -> bool:
        """Initialize the agent."""
        self._logger.info("Initializing Market Regime Agent...")
        self._logger.info(f"  - Lookback: {self._lookback} bars")
        self._logger.info(f"  - Min data points: {self._min_data_points}")
        return True

    def shutdown(self) -> bool:
        """Shutdown the agent."""
        self._logger.info("Shutting down Market Regime Agent...")
        return True

    def process_event(self, event: AgentEvent) -> Optional[AgentEvent]:
        """Process incoming events."""
        if event.event_type == EventType.MARKET_DATA_UPDATE:
            # Extract price data from event
            data = event.payload
            analysis = self.analyze(
                prices=np.array(data.get('closes', [])),
                highs=np.array(data.get('highs', [])),
                lows=np.array(data.get('lows', [])),
                volumes=np.array(data.get('volumes', []))
            )

            # Return regime change event if regime changed
            if analysis.is_transitioning:
                return AgentEvent(
                    event_type=EventType.REGIME_CHANGE,
                    source_agent_id=self.full_id,
                    payload=analysis.to_dict()
                )

        return None

    def get_capabilities(self) -> List[AgentCapability]:
        """Return agent capabilities."""
        return [AgentCapability.REGIME_DETECTION, AgentCapability.MARKET_ANALYSIS]

    # =========================================================================
    # MAIN ANALYSIS METHOD
    # =========================================================================

    def analyze(
        self,
        prices: np.ndarray,
        highs: Optional[np.ndarray] = None,
        lows: Optional[np.ndarray] = None,
        volumes: Optional[np.ndarray] = None,
        current_price: Optional[float] = None
    ) -> RegimeAnalysis:
        """
        Analyze market regime from price data.

        Args:
            prices: Array of close prices (most recent last)
            highs: Array of high prices (optional, uses prices if not provided)
            lows: Array of low prices (optional, uses prices if not provided)
            volumes: Array of volumes (optional)
            current_price: Current price (optional, uses last price if not provided)

        Returns:
            RegimeAnalysis with complete regime classification
        """
        self._current_step += 1

        # Update buffers
        if len(prices) > 0:
            for p in prices[-10:]:  # Add last 10 prices to buffer
                self._prices.append(p)

        if highs is not None and len(highs) > 0:
            for h in highs[-10:]:
                self._highs.append(h)
        else:
            self._highs = self._prices.copy()

        if lows is not None and len(lows) > 0:
            for l in lows[-10:]:
                self._lows.append(l)
        else:
            self._lows = self._prices.copy()

        if volumes is not None and len(volumes) > 0:
            for v in volumes[-10:]:
                self._volumes.append(v)

        # Check minimum data
        if len(self._prices) < self._min_data_points:
            return self._create_unknown_analysis("Insufficient data")

        # Convert to numpy arrays
        price_arr = np.array(list(self._prices))
        high_arr = np.array(list(self._highs))
        low_arr = np.array(list(self._lows))
        vol_arr = np.array(list(self._volumes)) if len(self._volumes) > 0 else None

        curr_price = current_price if current_price else price_arr[-1]

        # Calculate all indicators
        indicators = self._calculate_indicators(price_arr, high_arr, low_arr, vol_arr)

        # Determine regime
        regime, confidence = self._classify_regime(indicators, curr_price)

        # Check for transition
        is_transitioning = self._check_transition(regime)

        # Update regime if stable enough
        if not is_transitioning and regime != self._current_regime:
            if self._regime_duration >= self._regime_stability_threshold:
                self._current_regime = regime
                self._regime_duration = 0
                self._last_regime_change_step = self._current_step
                self._regime_history.append({
                    'regime': regime.value,
                    'step': self._current_step,
                    'confidence': confidence
                })

        self._regime_duration += 1

        # Get strategy recommendation
        strategy, size_mult = self._get_strategy_recommendation(regime, indicators)

        return RegimeAnalysis(
            regime=regime,
            confidence=confidence,
            trend_direction=indicators['trend_direction'],
            trend_strength=indicators['trend_strength'],
            volatility_state=indicators['volatility_state'],
            volatility_percentile=indicators['volatility_percentile'],
            momentum=indicators['momentum'],
            is_transitioning=is_transitioning,
            recommended_strategy=strategy,
            position_size_multiplier=size_mult,
            details=indicators
        )

    def update_single(
        self,
        price: float,
        high: Optional[float] = None,
        low: Optional[float] = None,
        volume: float = 0
    ) -> RegimeAnalysis:
        """
        Update with a single new bar and return analysis.

        More efficient than analyze() when processing bar-by-bar.
        """
        self._prices.append(price)
        self._highs.append(high if high else price)
        self._lows.append(low if low else price)
        self._volumes.append(volume)

        if len(self._prices) < self._min_data_points:
            return self._create_unknown_analysis("Insufficient data")

        return self.analyze(
            prices=np.array(list(self._prices)),
            highs=np.array(list(self._highs)),
            lows=np.array(list(self._lows)),
            volumes=np.array(list(self._volumes)) if self._volumes else None,
            current_price=price
        )

    # =========================================================================
    # INDICATOR CALCULATION
    # =========================================================================

    def _calculate_indicators(
        self,
        prices: np.ndarray,
        highs: np.ndarray,
        lows: np.ndarray,
        volumes: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """Calculate all technical indicators for regime detection."""
        ti = TechnicalIndicators

        # Moving Averages
        sma_9 = ti.sma(prices, 9)
        sma_21 = ti.sma(prices, 21)
        sma_50 = ti.sma(prices, 50)

        current_price = prices[-1]

        # Price position relative to MAs
        price_position = ti.price_position(current_price, sma_9, sma_50)

        # MA alignment
        ma_alignment = 0
        if sma_9 > sma_21 > sma_50:
            ma_alignment = 1  # Bullish alignment
        elif sma_9 < sma_21 < sma_50:
            ma_alignment = -1  # Bearish alignment

        # ADX for trend strength
        adx, plus_di, minus_di = ti.adx(highs, lows, prices)

        # RSI for momentum
        rsi = ti.rsi(prices)
        momentum = (rsi - 50) / 50  # Normalize to -1 to 1

        # ATR for volatility
        atr = ti.atr(highs, lows, prices)
        self._atr_history.append(atr)

        # Bollinger Bands for volatility context
        bb_upper, bb_middle, bb_lower, bb_width = ti.bollinger_bands(prices)

        # Volatility percentile
        atr_pct = atr / current_price * 100  # ATR as percentage
        self._volatility_history.append(atr_pct)

        if len(self._volatility_history) > 20:
            volatility_percentile = (
                np.sum(np.array(list(self._volatility_history)) < atr_pct) /
                len(self._volatility_history) * 100
            )
        else:
            volatility_percentile = 50

        # Price structure
        higher_highs, higher_lows = ti.higher_highs_higher_lows(highs, lows)
        lower_highs = not higher_highs
        lower_lows = not higher_lows

        # Trend direction
        if higher_highs and higher_lows and ma_alignment == 1:
            trend_direction = TrendDirection.UP
        elif lower_highs and lower_lows and ma_alignment == -1:
            trend_direction = TrendDirection.DOWN
        else:
            trend_direction = TrendDirection.NEUTRAL

        # Trend strength (0-1)
        trend_strength = min(1.0, adx / 50)  # ADX > 50 = very strong

        # Volatility state
        if volatility_percentile > 90:
            volatility_state = VolatilityState.EXTREME
        elif volatility_percentile > 75:
            volatility_state = VolatilityState.HIGH
        elif volatility_percentile > 25:
            volatility_state = VolatilityState.NORMAL
        elif volatility_percentile > 10:
            volatility_state = VolatilityState.LOW
        else:
            volatility_state = VolatilityState.VERY_LOW

        return {
            'sma_9': sma_9,
            'sma_21': sma_21,
            'sma_50': sma_50,
            'price_position': price_position,
            'ma_alignment': ma_alignment,
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'rsi': rsi,
            'momentum': momentum,
            'atr': atr,
            'atr_pct': atr_pct,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower,
            'bb_width': bb_width,
            'volatility_percentile': volatility_percentile,
            'volatility_state': volatility_state,
            'higher_highs': higher_highs,
            'higher_lows': higher_lows,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }

    # =========================================================================
    # REGIME CLASSIFICATION
    # =========================================================================

    def _classify_regime(self, indicators: Dict, current_price: float) -> Tuple[RegimeType, float]:
        """
        Classify market regime based on indicators.

        Returns: (regime, confidence)
        """
        adx = indicators['adx']
        trend_direction = indicators['trend_direction']
        trend_strength = indicators['trend_strength']
        volatility_state = indicators['volatility_state']
        ma_alignment = indicators['ma_alignment']
        momentum = indicators['momentum']

        # Start with volatility check (overrides trend in extreme cases)
        if volatility_state == VolatilityState.EXTREME:
            return RegimeType.HIGH_VOLATILITY, 0.9

        if volatility_state == VolatilityState.VERY_LOW:
            return RegimeType.LOW_VOLATILITY, 0.8

        # Check for strong trends
        if adx > 30:  # Strong trend
            if trend_direction == TrendDirection.UP:
                confidence = min(0.95, 0.6 + trend_strength * 0.3)
                return RegimeType.STRONG_UPTREND, confidence
            elif trend_direction == TrendDirection.DOWN:
                confidence = min(0.95, 0.6 + trend_strength * 0.3)
                return RegimeType.STRONG_DOWNTREND, confidence

        # Check for weak trends
        if adx > 15:
            if ma_alignment == 1 and momentum > 0:
                return RegimeType.WEAK_UPTREND, 0.6 + trend_strength * 0.2
            elif ma_alignment == -1 and momentum < 0:
                return RegimeType.WEAK_DOWNTREND, 0.6 + trend_strength * 0.2

        # Ranging market
        if adx < 20 and volatility_state in [VolatilityState.NORMAL, VolatilityState.LOW]:
            return RegimeType.RANGING, 0.7

        # Transition (conflicting signals)
        if self._check_conflicting_signals(indicators):
            return RegimeType.TRANSITION, 0.5

        return RegimeType.UNKNOWN, 0.3

    def _check_conflicting_signals(self, indicators: Dict) -> bool:
        """Check if indicators are giving conflicting signals."""
        conflicts = 0

        # MA alignment vs price position
        if indicators['ma_alignment'] == 1 and indicators['price_position'] < -0.3:
            conflicts += 1
        if indicators['ma_alignment'] == -1 and indicators['price_position'] > 0.3:
            conflicts += 1

        # Momentum vs trend direction
        if indicators['trend_direction'] == TrendDirection.UP and indicators['momentum'] < -0.3:
            conflicts += 1
        if indicators['trend_direction'] == TrendDirection.DOWN and indicators['momentum'] > 0.3:
            conflicts += 1

        # ADX suggesting trend but no clear direction
        if indicators['adx'] > 25 and indicators['trend_direction'] == TrendDirection.NEUTRAL:
            conflicts += 1

        return conflicts >= 2

    def _check_transition(self, new_regime: RegimeType) -> bool:
        """Check if we're transitioning between regimes."""
        if new_regime == self._current_regime:
            self._transition_warning_count = 0
            return False

        self._transition_warning_count += 1

        # Need consecutive signals before confirming transition
        return self._transition_warning_count < 3

    # =========================================================================
    # STRATEGY RECOMMENDATIONS
    # =========================================================================

    def _get_strategy_recommendation(
        self,
        regime: RegimeType,
        indicators: Dict
    ) -> Tuple[str, float]:
        """
        Get strategy recommendation and position size multiplier for regime.

        Returns: (strategy_description, size_multiplier)
        """
        recommendations = {
            RegimeType.STRONG_UPTREND: (
                "TREND_FOLLOW_LONG: Buy dips to MA, trail stops, no shorts",
                1.2
            ),
            RegimeType.WEAK_UPTREND: (
                "CAUTIOUS_LONG: Smaller positions, tighter stops, quick profits",
                0.8
            ),
            RegimeType.STRONG_DOWNTREND: (
                "TREND_FOLLOW_SHORT: Sell rallies, trail stops, no longs",
                1.2
            ),
            RegimeType.WEAK_DOWNTREND: (
                "CAUTIOUS_SHORT: Smaller positions, tighter stops, quick profits",
                0.8
            ),
            RegimeType.RANGING: (
                "MEAN_REVERSION: Fade extremes, target middle, tight stops",
                0.9
            ),
            RegimeType.HIGH_VOLATILITY: (
                "REDUCE_RISK: Cut positions 50%, widen stops, reduce frequency",
                0.4
            ),
            RegimeType.LOW_VOLATILITY: (
                "PREPARE_BREAKOUT: Small positions, wait for expansion, no chase",
                0.6
            ),
            RegimeType.TRANSITION: (
                "WAIT_AND_SEE: Reduce exposure, no new positions, observe",
                0.3
            ),
            RegimeType.UNKNOWN: (
                "NO_TRADE: Insufficient confidence, stay flat",
                0.0
            )
        }

        return recommendations.get(regime, ("UNKNOWN", 0.5))

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _create_unknown_analysis(self, reason: str) -> RegimeAnalysis:
        """Create an unknown regime analysis."""
        return RegimeAnalysis(
            regime=RegimeType.UNKNOWN,
            confidence=0.0,
            trend_direction=TrendDirection.NEUTRAL,
            trend_strength=0.0,
            volatility_state=VolatilityState.NORMAL,
            volatility_percentile=50,
            momentum=0.0,
            is_transitioning=False,
            recommended_strategy=f"NO_TRADE: {reason}",
            position_size_multiplier=0.0,
            details={'reason': reason}
        )

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def get_current_regime(self) -> RegimeType:
        """Get current detected regime."""
        return self._current_regime

    def get_regime_duration(self) -> int:
        """Get how long current regime has been active."""
        return self._regime_duration

    def get_regime_history(self) -> List[Dict]:
        """Get history of regime changes."""
        return list(self._regime_history)

    def get_statistics(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            'current_regime': self._current_regime.value,
            'regime_duration': self._regime_duration,
            'total_regime_changes': len(self._regime_history),
            'data_points': len(self._prices),
            'current_step': self._current_step,
            'recent_regimes': [r['regime'] for r in list(self._regime_history)[-5:]]
        }

    def reset(self) -> None:
        """Reset agent state."""
        self._prices.clear()
        self._highs.clear()
        self._lows.clear()
        self._volumes.clear()
        self._atr_history.clear()
        self._volatility_history.clear()
        self._regime_history.clear()
        self._current_regime = RegimeType.UNKNOWN
        self._regime_duration = 0
        self._current_step = 0


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_market_regime_agent(config: Optional[Dict] = None) -> MarketRegimeAgent:
    """
    Create a MarketRegimeAgent with optional configuration.

    Args:
        config: Optional configuration dict with:
            - lookback: Number of bars for analysis (default 100)
            - min_data_points: Minimum data required (default 50)

    Returns:
        Configured MarketRegimeAgent
    """
    return MarketRegimeAgent(config=config)
