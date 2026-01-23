# =============================================================================
# MULTI-TIMEFRAME ANALYSIS ENGINE - Sprint 2: Intelligence Enhancement
# =============================================================================
"""
Professional Multi-Timeframe Analysis Engine for Institutional Trading

This module implements sophisticated multi-timeframe analysis combining signals
from multiple time horizons with weighted aggregation and conflict detection.

Key Features:
- 4 timeframes: Weekly (strategic), Daily (tactical), 4H (swing), 1H (timing)
- Weighted signal aggregation with configurable weights
- Timeframe alignment scoring
- Conflict detection and resolution
- Trend, momentum, and volatility analysis per timeframe
- Fractal pattern recognition
- Higher timeframe confirmation rules

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MULTI-TIMEFRAME ANALYSIS ENGINE                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Weekly    │  │    Daily    │  │      4H     │  │      1H     │        │
│  │   (40%)     │  │    (30%)    │  │    (20%)    │  │    (10%)    │        │
│  │  Strategic  │  │   Tactical  │  │    Swing    │  │   Timing    │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │    ALIGNMENT CALCULATOR     │                         │
│                    │  - Trend Alignment Score    │                         │
│                    │  - Momentum Confluence      │                         │
│                    │  - Conflict Detection       │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │     SIGNAL AGGREGATOR       │                         │
│                    │  - Weighted Combination     │                         │
│                    │  - Higher TF Confirmation   │                         │
│                    │  - Final Signal Output      │                         │
│                    └─────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘

Version: 2.0.0
Author: TradingBot Team
License: Proprietary - Commercial Use
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class Timeframe(Enum):
    """Supported timeframes for analysis."""
    M1 = "1m"      # 1 minute
    M5 = "5m"      # 5 minutes
    M15 = "15m"    # 15 minutes
    M30 = "30m"    # 30 minutes
    H1 = "1h"      # 1 hour
    H4 = "4h"      # 4 hours
    D1 = "1d"      # Daily
    W1 = "1w"      # Weekly
    MN = "1M"      # Monthly

    @property
    def minutes(self) -> int:
        """Get timeframe duration in minutes."""
        mapping = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440, "1w": 10080, "1M": 43200
        }
        return mapping.get(self.value, 60)

    def __lt__(self, other: 'Timeframe') -> bool:
        return self.minutes < other.minutes


class TrendState(Enum):
    """Trend direction state."""
    STRONG_BULLISH = auto()
    BULLISH = auto()
    WEAK_BULLISH = auto()
    NEUTRAL = auto()
    WEAK_BEARISH = auto()
    BEARISH = auto()
    STRONG_BEARISH = auto()

    @property
    def score(self) -> float:
        """Numerical score from -1 (bearish) to +1 (bullish)."""
        scores = {
            TrendState.STRONG_BULLISH: 1.0,
            TrendState.BULLISH: 0.7,
            TrendState.WEAK_BULLISH: 0.3,
            TrendState.NEUTRAL: 0.0,
            TrendState.WEAK_BEARISH: -0.3,
            TrendState.BEARISH: -0.7,
            TrendState.STRONG_BEARISH: -1.0
        }
        return scores.get(self, 0.0)

    @property
    def is_bullish(self) -> bool:
        return self.score > 0

    @property
    def is_bearish(self) -> bool:
        return self.score < 0


class MomentumState(Enum):
    """Momentum state."""
    STRONG_POSITIVE = auto()
    POSITIVE = auto()
    WEAK_POSITIVE = auto()
    NEUTRAL = auto()
    WEAK_NEGATIVE = auto()
    NEGATIVE = auto()
    STRONG_NEGATIVE = auto()

    @property
    def score(self) -> float:
        scores = {
            MomentumState.STRONG_POSITIVE: 1.0,
            MomentumState.POSITIVE: 0.7,
            MomentumState.WEAK_POSITIVE: 0.3,
            MomentumState.NEUTRAL: 0.0,
            MomentumState.WEAK_NEGATIVE: -0.3,
            MomentumState.NEGATIVE: -0.7,
            MomentumState.STRONG_NEGATIVE: -1.0
        }
        return scores.get(self, 0.0)


class SignalStrength(Enum):
    """Signal strength classification."""
    VERY_STRONG = auto()
    STRONG = auto()
    MODERATE = auto()
    WEAK = auto()
    NEUTRAL = auto()
    CONFLICTING = auto()

    @property
    def confidence(self) -> float:
        confidence_map = {
            SignalStrength.VERY_STRONG: 0.95,
            SignalStrength.STRONG: 0.80,
            SignalStrength.MODERATE: 0.65,
            SignalStrength.WEAK: 0.50,
            SignalStrength.NEUTRAL: 0.30,
            SignalStrength.CONFLICTING: 0.10
        }
        return confidence_map.get(self, 0.30)


class SignalType(Enum):
    """Trading signal type."""
    STRONG_BUY = auto()
    BUY = auto()
    WEAK_BUY = auto()
    HOLD = auto()
    WEAK_SELL = auto()
    SELL = auto()
    STRONG_SELL = auto()
    BLOCKED = auto()  # Conflicting signals - no action

    @property
    def score(self) -> float:
        scores = {
            SignalType.STRONG_BUY: 1.0,
            SignalType.BUY: 0.7,
            SignalType.WEAK_BUY: 0.3,
            SignalType.HOLD: 0.0,
            SignalType.WEAK_SELL: -0.3,
            SignalType.SELL: -0.7,
            SignalType.STRONG_SELL: -1.0,
            SignalType.BLOCKED: 0.0
        }
        return scores.get(self, 0.0)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OHLCV:
    """OHLCV candle data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def body_size(self) -> float:
        """Absolute body size."""
        return abs(self.close - self.open)

    @property
    def upper_wick(self) -> float:
        """Upper wick size."""
        return self.high - max(self.open, self.close)

    @property
    def lower_wick(self) -> float:
        """Lower wick size."""
        return min(self.open, self.close) - self.low

    @property
    def range(self) -> float:
        """Total candle range."""
        return self.high - self.low

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class TimeframeAnalysis:
    """Analysis results for a single timeframe."""
    timeframe: Timeframe
    timestamp: datetime

    # Trend analysis
    trend_state: TrendState
    trend_strength: float  # 0-1
    trend_duration: int    # Number of periods

    # Moving averages
    sma_fast: float
    sma_slow: float
    ema_fast: float
    ema_slow: float
    price_vs_sma: float  # % above/below

    # Momentum
    momentum_state: MomentumState
    rsi: float
    rsi_divergence: bool = False
    macd_histogram: float = 0.0
    macd_signal: float = 0.0

    # Volatility
    atr: float = 0.0
    atr_percentile: float = 50.0  # 0-100
    bollinger_position: float = 0.0  # -1 to +1 (lower band to upper band)

    # Structure
    higher_high: bool = False
    higher_low: bool = False
    lower_high: bool = False
    lower_low: bool = False

    # Support/Resistance
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0
    distance_to_support_pct: float = 0.0
    distance_to_resistance_pct: float = 0.0

    # Signal
    signal: SignalType = SignalType.HOLD
    signal_confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "trend_state": self.trend_state.name,
            "trend_strength": self.trend_strength,
            "trend_duration": self.trend_duration,
            "momentum_state": self.momentum_state.name,
            "rsi": self.rsi,
            "atr": self.atr,
            "signal": self.signal.name,
            "signal_confidence": self.signal_confidence
        }


@dataclass
class AlignmentResult:
    """Multi-timeframe alignment analysis result."""
    timestamp: datetime

    # Individual timeframe analyses
    analyses: Dict[Timeframe, TimeframeAnalysis]

    # Alignment scores
    trend_alignment: float         # -1 to +1 (all bearish to all bullish)
    momentum_alignment: float      # -1 to +1
    overall_alignment: float       # 0 to 1 (conflicting to aligned)

    # Conflict detection
    has_conflict: bool
    conflict_description: str = ""
    conflicting_timeframes: List[Tuple[Timeframe, Timeframe]] = field(default_factory=list)

    # Aggregated signal
    aggregated_signal: SignalType = SignalType.HOLD
    signal_strength: SignalStrength = SignalStrength.WEAK
    confidence: float = 0.5

    # Position sizing recommendation
    position_size_multiplier: float = 1.0

    # Explanation
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "analyses": {tf.value: a.to_dict() for tf, a in self.analyses.items()},
            "trend_alignment": self.trend_alignment,
            "momentum_alignment": self.momentum_alignment,
            "overall_alignment": self.overall_alignment,
            "has_conflict": self.has_conflict,
            "conflict_description": self.conflict_description,
            "aggregated_signal": self.aggregated_signal.name,
            "signal_strength": self.signal_strength.name,
            "confidence": self.confidence,
            "position_size_multiplier": self.position_size_multiplier,
            "explanation": self.explanation
        }


@dataclass
class MultiTimeframeConfig:
    """Configuration for multi-timeframe analysis."""
    # Active timeframes and their weights
    timeframes: Dict[Timeframe, float] = field(default_factory=lambda: {
        Timeframe.W1: 0.40,  # Weekly - Strategic
        Timeframe.D1: 0.30,  # Daily - Tactical
        Timeframe.H4: 0.20,  # 4H - Swing
        Timeframe.H1: 0.10   # 1H - Timing
    })

    # Indicator periods
    sma_fast_period: int = 10
    sma_slow_period: int = 50
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    rsi_period: int = 14
    atr_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0

    # Thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    trend_strength_threshold: float = 0.5
    conflict_threshold: float = 0.4  # Alignment below this = conflict

    # Higher timeframe confirmation
    require_htf_confirmation: bool = True
    htf_confirmation_timeframes: int = 2  # How many higher TFs must confirm

    # Minimum data requirements
    min_candles_per_timeframe: int = 100

    # Signal blocking
    block_on_conflict: bool = True
    reduce_size_on_weak_alignment: bool = True

    def validate(self) -> bool:
        """Validate configuration."""
        if not self.timeframes:
            raise ValueError("At least one timeframe must be configured")

        total_weight = sum(self.timeframes.values())
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Timeframe weights must sum to 1.0, got {total_weight}")

        return True


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class TechnicalIndicators:
    """Collection of technical indicator calculations."""

    @staticmethod
    def sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Simple Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        result = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            result[i] = np.mean(prices[i - period + 1:i + 1])
        return result

    @staticmethod
    def ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Exponential Moving Average."""
        if len(prices) < period:
            return np.full(len(prices), np.nan)

        alpha = 2 / (period + 1)
        result = np.zeros(len(prices))
        result[0] = prices[0]

        for i in range(1, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]

        # Set NaN for initial warmup period
        result[:period - 1] = np.nan
        return result

    @staticmethod
    def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Relative Strength Index."""
        if len(prices) < period + 1:
            return np.full(len(prices), np.nan)

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        result = np.full(len(prices), np.nan)

        # Initial average
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        for i in range(period, len(prices)):
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))

            if i < len(gains):
                avg_gain = (avg_gain * (period - 1) + gains[i]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        return result

    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """MACD (line, signal, histogram)."""
        ema_fast = TechnicalIndicators.ema(prices, fast)
        ema_slow = TechnicalIndicators.ema(prices, slow)

        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line[~np.isnan(macd_line)], signal)

        # Pad signal line to match length
        full_signal = np.full(len(prices), np.nan)
        full_signal[-len(signal_line):] = signal_line

        histogram = macd_line - full_signal

        return macd_line, full_signal, histogram

    @staticmethod
    def atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
            period: int = 14) -> np.ndarray:
        """Average True Range."""
        if len(high) < period + 1:
            return np.full(len(high), np.nan)

        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i - 1])
            lc = abs(low[i] - close[i - 1])
            tr[i] = max(hl, hc, lc)

        return TechnicalIndicators.sma(tr, period)

    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20,
                        std_dev: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands (upper, middle, lower)."""
        middle = TechnicalIndicators.sma(prices, period)

        std = np.full(len(prices), np.nan)
        for i in range(period - 1, len(prices)):
            std[i] = np.std(prices[i - period + 1:i + 1])

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    @staticmethod
    def bollinger_position(price: float, upper: float, lower: float) -> float:
        """Calculate position within Bollinger Bands (-1 to +1)."""
        if upper == lower or np.isnan(upper) or np.isnan(lower):
            return 0.0

        band_range = upper - lower
        position = (price - lower) / band_range
        return (position * 2) - 1  # Convert 0-1 to -1 to +1

    @staticmethod
    def find_swings(prices: np.ndarray, lookback: int = 5) -> Tuple[List[int], List[int]]:
        """Find swing highs and lows."""
        swing_highs = []
        swing_lows = []

        for i in range(lookback, len(prices) - lookback):
            # Check for swing high
            is_swing_high = all(prices[i] > prices[i - j] for j in range(1, lookback + 1)) and \
                           all(prices[i] > prices[i + j] for j in range(1, lookback + 1))
            if is_swing_high:
                swing_highs.append(i)

            # Check for swing low
            is_swing_low = all(prices[i] < prices[i - j] for j in range(1, lookback + 1)) and \
                          all(prices[i] < prices[i + j] for j in range(1, lookback + 1))
            if is_swing_low:
                swing_lows.append(i)

        return swing_highs, swing_lows

    @staticmethod
    def detect_divergence(prices: np.ndarray, indicator: np.ndarray,
                          lookback: int = 10) -> Tuple[bool, bool]:
        """Detect bullish and bearish divergence."""
        if len(prices) < lookback * 2:
            return False, False

        recent_prices = prices[-lookback:]
        older_prices = prices[-lookback * 2:-lookback]
        recent_ind = indicator[-lookback:]
        older_ind = indicator[-lookback * 2:-lookback]

        # Skip if NaN values
        if np.any(np.isnan(recent_ind)) or np.any(np.isnan(older_ind)):
            return False, False

        price_higher = np.max(recent_prices) > np.max(older_prices)
        price_lower = np.min(recent_prices) < np.min(older_prices)
        ind_higher = np.max(recent_ind) > np.max(older_ind)
        ind_lower = np.min(recent_ind) < np.min(older_ind)

        # Bullish divergence: price lower low, indicator higher low
        bullish_div = price_lower and not ind_lower

        # Bearish divergence: price higher high, indicator lower high
        bearish_div = price_higher and not ind_higher

        return bullish_div, bearish_div


# =============================================================================
# TIMEFRAME ANALYZER
# =============================================================================

class TimeframeAnalyzer:
    """Analyzes a single timeframe and produces signals."""

    def __init__(self, timeframe: Timeframe, config: MultiTimeframeConfig):
        self.timeframe = timeframe
        self.config = config

        # Data storage
        self.candles: deque = deque(maxlen=500)

        # Cached calculations
        self._last_analysis: Optional[TimeframeAnalysis] = None
        self._last_analysis_time: Optional[datetime] = None

    def add_candle(self, candle: OHLCV) -> None:
        """Add a new candle."""
        self.candles.append(candle)
        self._last_analysis = None  # Invalidate cache

    def add_candles(self, candles: List[OHLCV]) -> None:
        """Add multiple candles."""
        for candle in candles:
            self.candles.append(candle)
        self._last_analysis = None

    def has_sufficient_data(self) -> bool:
        """Check if we have enough data for analysis."""
        return len(self.candles) >= self.config.min_candles_per_timeframe

    def analyze(self) -> Optional[TimeframeAnalysis]:
        """Perform comprehensive analysis for this timeframe."""
        if not self.has_sufficient_data():
            logger.warning(f"Insufficient data for {self.timeframe.value}: "
                          f"{len(self.candles)}/{self.config.min_candles_per_timeframe}")
            return None

        # Convert to arrays
        closes = np.array([c.close for c in self.candles])
        highs = np.array([c.high for c in self.candles])
        lows = np.array([c.low for c in self.candles])
        opens = np.array([c.open for c in self.candles])
        volumes = np.array([c.volume for c in self.candles])

        latest = self.candles[-1]
        current_price = closes[-1]

        # Calculate indicators
        sma_fast = TechnicalIndicators.sma(closes, self.config.sma_fast_period)
        sma_slow = TechnicalIndicators.sma(closes, self.config.sma_slow_period)
        ema_fast = TechnicalIndicators.ema(closes, self.config.ema_fast_period)
        ema_slow = TechnicalIndicators.ema(closes, self.config.ema_slow_period)

        rsi = TechnicalIndicators.rsi(closes, self.config.rsi_period)
        macd_line, macd_signal, macd_hist = TechnicalIndicators.macd(
            closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )

        atr = TechnicalIndicators.atr(highs, lows, closes, self.config.atr_period)
        bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
            closes, self.config.bollinger_period, self.config.bollinger_std
        )

        # Get latest values
        current_sma_fast = sma_fast[-1]
        current_sma_slow = sma_slow[-1]
        current_ema_fast = ema_fast[-1]
        current_ema_slow = ema_slow[-1]
        current_rsi = rsi[-1] if not np.isnan(rsi[-1]) else 50.0
        current_atr = atr[-1] if not np.isnan(atr[-1]) else 0.0
        current_macd_hist = macd_hist[-1] if not np.isnan(macd_hist[-1]) else 0.0
        current_macd_signal = macd_signal[-1] if not np.isnan(macd_signal[-1]) else 0.0

        # Calculate price vs SMA
        price_vs_sma = ((current_price - current_sma_slow) / current_sma_slow * 100) \
            if current_sma_slow > 0 and not np.isnan(current_sma_slow) else 0.0

        # Bollinger position
        bb_position = TechnicalIndicators.bollinger_position(
            current_price, bb_upper[-1], bb_lower[-1]
        )

        # ATR percentile
        valid_atr = atr[~np.isnan(atr)]
        atr_percentile = (np.sum(valid_atr < current_atr) / len(valid_atr) * 100) \
            if len(valid_atr) > 0 else 50.0

        # Trend analysis
        trend_state, trend_strength, trend_duration = self._analyze_trend(
            closes, sma_fast, sma_slow, ema_fast, ema_slow
        )

        # Momentum analysis
        momentum_state = self._analyze_momentum(current_rsi, current_macd_hist, macd_hist)

        # RSI divergence
        bullish_div, bearish_div = TechnicalIndicators.detect_divergence(
            closes, rsi, lookback=10
        )

        # Market structure (higher highs/lows)
        swing_highs, swing_lows = TechnicalIndicators.find_swings(closes, lookback=5)
        hh, hl, lh, ll = self._analyze_structure(closes, swing_highs, swing_lows)

        # Support/Resistance
        support, resistance = self._find_support_resistance(closes, swing_highs, swing_lows)
        dist_support = ((current_price - support) / current_price * 100) if support > 0 else 0.0
        dist_resistance = ((resistance - current_price) / current_price * 100) if resistance > 0 else 0.0

        # Generate signal
        signal, confidence = self._generate_signal(
            trend_state, trend_strength, momentum_state,
            current_rsi, bb_position, hh, hl, lh, ll,
            bullish_div, bearish_div
        )

        analysis = TimeframeAnalysis(
            timeframe=self.timeframe,
            timestamp=latest.timestamp,
            trend_state=trend_state,
            trend_strength=trend_strength,
            trend_duration=trend_duration,
            sma_fast=current_sma_fast,
            sma_slow=current_sma_slow,
            ema_fast=current_ema_fast,
            ema_slow=current_ema_slow,
            price_vs_sma=price_vs_sma,
            momentum_state=momentum_state,
            rsi=current_rsi,
            rsi_divergence=bullish_div or bearish_div,
            macd_histogram=current_macd_hist,
            macd_signal=current_macd_signal,
            atr=current_atr,
            atr_percentile=atr_percentile,
            bollinger_position=bb_position,
            higher_high=hh,
            higher_low=hl,
            lower_high=lh,
            lower_low=ll,
            nearest_support=support,
            nearest_resistance=resistance,
            distance_to_support_pct=dist_support,
            distance_to_resistance_pct=dist_resistance,
            signal=signal,
            signal_confidence=confidence
        )

        self._last_analysis = analysis
        self._last_analysis_time = datetime.now()

        return analysis

    def _analyze_trend(self, closes: np.ndarray, sma_fast: np.ndarray,
                       sma_slow: np.ndarray, ema_fast: np.ndarray,
                       ema_slow: np.ndarray) -> Tuple[TrendState, float, int]:
        """Analyze trend state, strength, and duration."""
        current_price = closes[-1]

        # Get valid values
        if np.isnan(sma_slow[-1]) or np.isnan(ema_slow[-1]):
            return TrendState.NEUTRAL, 0.0, 0

        # Calculate trend score components
        scores = []

        # Price vs SMAs
        if current_price > sma_slow[-1]:
            scores.append(0.5 if current_price > sma_fast[-1] else 0.25)
        else:
            scores.append(-0.5 if current_price < sma_fast[-1] else -0.25)

        # SMA alignment (fast vs slow)
        if sma_fast[-1] > sma_slow[-1]:
            scores.append(0.25)
        else:
            scores.append(-0.25)

        # EMA alignment
        if ema_fast[-1] > ema_slow[-1]:
            scores.append(0.25)
        else:
            scores.append(-0.25)

        # Trend slope (compare to 10 bars ago)
        if len(sma_slow) > 10 and not np.isnan(sma_slow[-10]):
            slope = (sma_slow[-1] - sma_slow[-10]) / sma_slow[-10]
            if slope > 0.02:
                scores.append(0.25)
            elif slope < -0.02:
                scores.append(-0.25)
            else:
                scores.append(0.0)

        trend_score = sum(scores) / len(scores) if scores else 0.0
        trend_strength = abs(trend_score)

        # Determine trend state
        if trend_score >= 0.6:
            state = TrendState.STRONG_BULLISH
        elif trend_score >= 0.35:
            state = TrendState.BULLISH
        elif trend_score >= 0.1:
            state = TrendState.WEAK_BULLISH
        elif trend_score <= -0.6:
            state = TrendState.STRONG_BEARISH
        elif trend_score <= -0.35:
            state = TrendState.BEARISH
        elif trend_score <= -0.1:
            state = TrendState.WEAK_BEARISH
        else:
            state = TrendState.NEUTRAL

        # Calculate trend duration
        duration = self._calculate_trend_duration(closes, sma_fast)

        return state, trend_strength, duration

    def _calculate_trend_duration(self, closes: np.ndarray, sma: np.ndarray) -> int:
        """Calculate how long the current trend has lasted."""
        if len(closes) < 2 or np.isnan(sma[-1]):
            return 0

        is_above = closes[-1] > sma[-1]
        duration = 0

        for i in range(len(closes) - 1, -1, -1):
            if np.isnan(sma[i]):
                break
            if (closes[i] > sma[i]) == is_above:
                duration += 1
            else:
                break

        return duration

    def _analyze_momentum(self, rsi: float, macd_hist: float,
                          macd_hist_history: np.ndarray) -> MomentumState:
        """Analyze momentum state."""
        scores = []

        # RSI contribution
        if rsi >= 70:
            scores.append(0.8)  # Overbought but bullish momentum
        elif rsi >= 60:
            scores.append(0.5)
        elif rsi >= 50:
            scores.append(0.2)
        elif rsi >= 40:
            scores.append(-0.2)
        elif rsi >= 30:
            scores.append(-0.5)
        else:
            scores.append(-0.8)  # Oversold but bearish momentum

        # MACD histogram contribution
        if macd_hist > 0:
            scores.append(0.5 if macd_hist > 0.001 else 0.2)
        else:
            scores.append(-0.5 if macd_hist < -0.001 else -0.2)

        # MACD histogram trend (accelerating/decelerating)
        valid_hist = macd_hist_history[~np.isnan(macd_hist_history)]
        if len(valid_hist) >= 3:
            hist_change = valid_hist[-1] - valid_hist[-3]
            if hist_change > 0.0005:
                scores.append(0.3)
            elif hist_change < -0.0005:
                scores.append(-0.3)
            else:
                scores.append(0.0)

        momentum_score = np.mean(scores)

        if momentum_score >= 0.6:
            return MomentumState.STRONG_POSITIVE
        elif momentum_score >= 0.35:
            return MomentumState.POSITIVE
        elif momentum_score >= 0.1:
            return MomentumState.WEAK_POSITIVE
        elif momentum_score <= -0.6:
            return MomentumState.STRONG_NEGATIVE
        elif momentum_score <= -0.35:
            return MomentumState.NEGATIVE
        elif momentum_score <= -0.1:
            return MomentumState.WEAK_NEGATIVE
        else:
            return MomentumState.NEUTRAL

    def _analyze_structure(self, closes: np.ndarray, swing_highs: List[int],
                           swing_lows: List[int]) -> Tuple[bool, bool, bool, bool]:
        """Analyze market structure (HH, HL, LH, LL)."""
        hh = hl = lh = ll = False

        if len(swing_highs) >= 2:
            last_high = closes[swing_highs[-1]]
            prev_high = closes[swing_highs[-2]]
            hh = last_high > prev_high
            lh = last_high < prev_high

        if len(swing_lows) >= 2:
            last_low = closes[swing_lows[-1]]
            prev_low = closes[swing_lows[-2]]
            hl = last_low > prev_low
            ll = last_low < prev_low

        return hh, hl, lh, ll

    def _find_support_resistance(self, closes: np.ndarray, swing_highs: List[int],
                                  swing_lows: List[int]) -> Tuple[float, float]:
        """Find nearest support and resistance levels."""
        current_price = closes[-1]

        # Get recent swing levels
        support = 0.0
        resistance = float('inf')

        for idx in swing_lows[-5:]:  # Last 5 swing lows
            level = closes[idx]
            if level < current_price and level > support:
                support = level

        for idx in swing_highs[-5:]:  # Last 5 swing highs
            level = closes[idx]
            if level > current_price and level < resistance:
                resistance = level

        if resistance == float('inf'):
            resistance = current_price * 1.02  # Default 2% above

        if support == 0.0:
            support = current_price * 0.98  # Default 2% below

        return support, resistance

    def _generate_signal(self, trend: TrendState, trend_strength: float,
                         momentum: MomentumState, rsi: float, bb_position: float,
                         hh: bool, hl: bool, lh: bool, ll: bool,
                         bullish_div: bool, bearish_div: bool) -> Tuple[SignalType, float]:
        """Generate trading signal with confidence."""
        score = 0.0
        factors = []

        # Trend contribution (40%)
        trend_score = trend.score * 0.4
        score += trend_score
        factors.append(("trend", trend_score))

        # Momentum contribution (30%)
        momentum_score = momentum.score * 0.3
        score += momentum_score
        factors.append(("momentum", momentum_score))

        # RSI extremes (10%)
        if rsi >= 80:
            rsi_score = -0.1  # Extremely overbought
        elif rsi >= 70:
            rsi_score = -0.05
        elif rsi <= 20:
            rsi_score = 0.1  # Extremely oversold
        elif rsi <= 30:
            rsi_score = 0.05
        else:
            rsi_score = 0.0
        score += rsi_score
        factors.append(("rsi", rsi_score))

        # Bollinger position (10%)
        if bb_position >= 0.8:
            bb_score = -0.1  # Near upper band
        elif bb_position <= -0.8:
            bb_score = 0.1   # Near lower band
        else:
            bb_score = 0.0
        score += bb_score
        factors.append(("bollinger", bb_score))

        # Market structure (10%)
        structure_score = 0.0
        if hh and hl:
            structure_score = 0.1  # Bullish structure
        elif lh and ll:
            structure_score = -0.1  # Bearish structure
        score += structure_score
        factors.append(("structure", structure_score))

        # Divergence bonus
        if bullish_div:
            score += 0.15
        if bearish_div:
            score -= 0.15

        # Convert score to signal
        if score >= 0.6:
            signal = SignalType.STRONG_BUY
        elif score >= 0.35:
            signal = SignalType.BUY
        elif score >= 0.15:
            signal = SignalType.WEAK_BUY
        elif score <= -0.6:
            signal = SignalType.STRONG_SELL
        elif score <= -0.35:
            signal = SignalType.SELL
        elif score <= -0.15:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.HOLD

        # Calculate confidence (clamp to [0, 0.95] to prevent negative/overflow)
        raw_confidence = 0.5 + abs(score) * 0.5 + abs(trend_strength) * 0.2
        confidence = max(0.0, min(0.95, raw_confidence))

        return signal, confidence


# =============================================================================
# MULTI-TIMEFRAME ENGINE
# =============================================================================

class MultiTimeframeEngine:
    """
    Multi-Timeframe Analysis Engine.

    Combines analysis from multiple timeframes with weighted aggregation
    and conflict detection.
    """

    def __init__(self, config: Optional[MultiTimeframeConfig] = None):
        """Initialize multi-timeframe engine."""
        self.config = config or MultiTimeframeConfig()
        self.config.validate()

        # Create analyzer for each timeframe
        self.analyzers: Dict[Timeframe, TimeframeAnalyzer] = {}
        for tf in self.config.timeframes.keys():
            self.analyzers[tf] = TimeframeAnalyzer(tf, self.config)

        # Analysis history
        self.history: deque = deque(maxlen=100)

        logger.info(f"MultiTimeframeEngine initialized with {len(self.analyzers)} timeframes")

    def add_data(self, timeframe: Timeframe, candles: List[OHLCV]) -> None:
        """Add candle data for a specific timeframe."""
        if timeframe not in self.analyzers:
            logger.warning(f"Timeframe {timeframe.value} not configured, skipping")
            return

        self.analyzers[timeframe].add_candles(candles)
        logger.debug(f"Added {len(candles)} candles to {timeframe.value}")

    def add_candle(self, timeframe: Timeframe, candle: OHLCV) -> None:
        """Add a single candle for a specific timeframe."""
        if timeframe in self.analyzers:
            self.analyzers[timeframe].add_candle(candle)

    def analyze(self) -> Optional[AlignmentResult]:
        """
        Perform multi-timeframe analysis.

        Returns:
            AlignmentResult with aggregated signals and alignment scores,
            or None if insufficient data.
        """
        # Analyze each timeframe
        analyses: Dict[Timeframe, TimeframeAnalysis] = {}

        for tf, analyzer in self.analyzers.items():
            analysis = analyzer.analyze()
            if analysis is not None:
                analyses[tf] = analysis

        if len(analyses) < 2:
            logger.warning(f"Insufficient timeframe data: {len(analyses)}/{len(self.analyzers)}")
            return None

        # Calculate alignments
        trend_alignment = self._calculate_trend_alignment(analyses)
        momentum_alignment = self._calculate_momentum_alignment(analyses)
        overall_alignment = (abs(trend_alignment) + abs(momentum_alignment)) / 2

        # Detect conflicts
        has_conflict, conflict_desc, conflicting_pairs = self._detect_conflicts(analyses)

        # Aggregate signals
        if has_conflict and self.config.block_on_conflict:
            aggregated_signal = SignalType.BLOCKED
            signal_strength = SignalStrength.CONFLICTING
            confidence = 0.1
        else:
            aggregated_signal, signal_strength, confidence = self._aggregate_signals(
                analyses, trend_alignment, overall_alignment
            )

        # Calculate position size multiplier
        position_multiplier = self._calculate_position_multiplier(
            overall_alignment, has_conflict, signal_strength
        )

        # Generate explanation
        explanation = self._generate_explanation(
            analyses, trend_alignment, momentum_alignment,
            has_conflict, aggregated_signal
        )

        result = AlignmentResult(
            timestamp=datetime.now(),
            analyses=analyses,
            trend_alignment=trend_alignment,
            momentum_alignment=momentum_alignment,
            overall_alignment=overall_alignment,
            has_conflict=has_conflict,
            conflict_description=conflict_desc,
            conflicting_timeframes=conflicting_pairs,
            aggregated_signal=aggregated_signal,
            signal_strength=signal_strength,
            confidence=confidence,
            position_size_multiplier=position_multiplier,
            explanation=explanation
        )

        self.history.append(result)
        return result

    def _calculate_trend_alignment(self, analyses: Dict[Timeframe, TimeframeAnalysis]) -> float:
        """Calculate weighted trend alignment score."""
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, analysis in analyses.items():
            weight = self.config.timeframes.get(tf, 0.0)
            weighted_sum += analysis.trend_state.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _calculate_momentum_alignment(self, analyses: Dict[Timeframe, TimeframeAnalysis]) -> float:
        """Calculate weighted momentum alignment score."""
        weighted_sum = 0.0
        total_weight = 0.0

        for tf, analysis in analyses.items():
            weight = self.config.timeframes.get(tf, 0.0)
            weighted_sum += analysis.momentum_state.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def _detect_conflicts(self, analyses: Dict[Timeframe, TimeframeAnalysis]
                          ) -> Tuple[bool, str, List[Tuple[Timeframe, Timeframe]]]:
        """Detect conflicting signals between timeframes."""
        conflicts = []
        sorted_tfs = sorted(analyses.keys(), key=lambda x: x.minutes, reverse=True)

        for i, tf1 in enumerate(sorted_tfs):
            for tf2 in sorted_tfs[i + 1:]:
                analysis1 = analyses[tf1]
                analysis2 = analyses[tf2]

                # Check trend conflict (opposite directions)
                trend_conflict = (
                    (analysis1.trend_state.is_bullish and analysis2.trend_state.is_bearish) or
                    (analysis1.trend_state.is_bearish and analysis2.trend_state.is_bullish)
                )

                # Conflict is significant if both have strong conviction
                if trend_conflict:
                    strength1 = analysis1.trend_strength
                    strength2 = analysis2.trend_strength
                    if strength1 > 0.3 and strength2 > 0.3:
                        conflicts.append((tf1, tf2))

        has_conflict = len(conflicts) > 0

        if has_conflict:
            # Higher timeframe conflicts are more serious
            htf_conflicts = [(tf1, tf2) for tf1, tf2 in conflicts
                            if tf1.minutes >= Timeframe.D1.minutes]

            if htf_conflicts:
                desc = f"Major conflict: {htf_conflicts[0][0].value} vs {htf_conflicts[0][1].value}"
            else:
                desc = f"Minor conflict: {conflicts[0][0].value} vs {conflicts[0][1].value}"
        else:
            desc = ""

        return has_conflict, desc, conflicts

    def _aggregate_signals(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                           trend_alignment: float, overall_alignment: float
                           ) -> Tuple[SignalType, SignalStrength, float]:
        """Aggregate signals from all timeframes with weighting."""
        weighted_score = 0.0
        total_weight = 0.0
        weighted_confidence = 0.0

        for tf, analysis in analyses.items():
            weight = self.config.timeframes.get(tf, 0.0)
            weighted_score += analysis.signal.score * weight
            weighted_confidence += analysis.signal_confidence * weight
            total_weight += weight

        if total_weight > 0:
            avg_score = weighted_score / total_weight
            avg_confidence = weighted_confidence / total_weight
        else:
            return SignalType.HOLD, SignalStrength.NEUTRAL, 0.3

        # Higher timeframe confirmation
        if self.config.require_htf_confirmation:
            htf_confirms = self._check_htf_confirmation(analyses, avg_score)
            if not htf_confirms:
                avg_score *= 0.5  # Reduce signal strength
                avg_confidence *= 0.7

        # Convert to signal type
        if avg_score >= 0.6:
            signal = SignalType.STRONG_BUY
        elif avg_score >= 0.35:
            signal = SignalType.BUY
        elif avg_score >= 0.15:
            signal = SignalType.WEAK_BUY
        elif avg_score <= -0.6:
            signal = SignalType.STRONG_SELL
        elif avg_score <= -0.35:
            signal = SignalType.SELL
        elif avg_score <= -0.15:
            signal = SignalType.WEAK_SELL
        else:
            signal = SignalType.HOLD

        # Determine signal strength
        if overall_alignment >= 0.8 and abs(avg_score) >= 0.5:
            strength = SignalStrength.VERY_STRONG
        elif overall_alignment >= 0.6 and abs(avg_score) >= 0.35:
            strength = SignalStrength.STRONG
        elif overall_alignment >= 0.4:
            strength = SignalStrength.MODERATE
        elif overall_alignment >= 0.2:
            strength = SignalStrength.WEAK
        else:
            strength = SignalStrength.NEUTRAL

        # Final confidence
        confidence = min(0.95, avg_confidence * overall_alignment)

        return signal, strength, confidence

    def _check_htf_confirmation(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                                 signal_direction: float) -> bool:
        """Check if higher timeframes confirm the signal direction."""
        # Sort timeframes by size (largest first)
        sorted_tfs = sorted(analyses.keys(), key=lambda x: x.minutes, reverse=True)

        confirming = 0
        required = min(self.config.htf_confirmation_timeframes, len(sorted_tfs) - 1)

        for tf in sorted_tfs[:required + 1]:
            analysis = analyses[tf]
            # Check if trend direction matches signal direction
            if signal_direction > 0 and analysis.trend_state.is_bullish:
                confirming += 1
            elif signal_direction < 0 and analysis.trend_state.is_bearish:
                confirming += 1

        return confirming >= required

    def _calculate_position_multiplier(self, alignment: float, has_conflict: bool,
                                        strength: SignalStrength) -> float:
        """Calculate position size multiplier based on alignment."""
        if has_conflict:
            return 0.0  # No position on conflict

        if not self.config.reduce_size_on_weak_alignment:
            return 1.0

        # Base multiplier on alignment
        multiplier = alignment

        # Adjust by signal strength
        multiplier *= strength.confidence

        # Clamp to reasonable range
        return max(0.25, min(1.5, multiplier))

    def _generate_explanation(self, analyses: Dict[Timeframe, TimeframeAnalysis],
                               trend_alignment: float, momentum_alignment: float,
                               has_conflict: bool, signal: SignalType) -> str:
        """Generate human-readable explanation of the analysis."""
        lines = []

        # Overall summary
        direction = "bullish" if trend_alignment > 0 else "bearish" if trend_alignment < 0 else "neutral"
        lines.append(f"Overall bias: {direction.upper()} (trend: {trend_alignment:.2f}, "
                    f"momentum: {momentum_alignment:.2f})")

        if has_conflict:
            lines.append("WARNING: Timeframe conflict detected - signal blocked")

        # Per-timeframe summary
        lines.append("\nTimeframe breakdown:")
        for tf in sorted(analyses.keys(), key=lambda x: x.minutes, reverse=True):
            a = analyses[tf]
            weight = self.config.timeframes.get(tf, 0) * 100
            lines.append(f"  {tf.value} ({weight:.0f}%): {a.trend_state.name} | "
                        f"RSI: {a.rsi:.1f} | Signal: {a.signal.name}")

        lines.append(f"\nFinal signal: {signal.name}")

        return "\n".join(lines)

    def get_quick_signal(self) -> Tuple[SignalType, float]:
        """Get quick aggregated signal without full analysis."""
        result = self.analyze()
        if result is None:
            return SignalType.HOLD, 0.3
        return result.aggregated_signal, result.confidence

    def reset(self) -> None:
        """Reset all analyzers and history."""
        for analyzer in self.analyzers.values():
            analyzer.candles.clear()
            analyzer._last_analysis = None
        self.history.clear()
        logger.info("MultiTimeframeEngine reset")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_multi_timeframe_engine(
    timeframes: Optional[Dict[Timeframe, float]] = None,
    require_htf_confirmation: bool = True,
    block_on_conflict: bool = True
) -> MultiTimeframeEngine:
    """
    Create a configured multi-timeframe engine.

    Args:
        timeframes: Dictionary of timeframe to weight mappings.
                   Defaults to W1:40%, D1:30%, H4:20%, H1:10%
        require_htf_confirmation: Whether higher timeframes must confirm signals
        block_on_conflict: Whether to block signals on timeframe conflict

    Returns:
        Configured MultiTimeframeEngine instance

    Example:
        >>> engine = create_multi_timeframe_engine(
        ...     timeframes={Timeframe.D1: 0.5, Timeframe.H4: 0.3, Timeframe.H1: 0.2}
        ... )
        >>> engine.add_data(Timeframe.D1, daily_candles)
        >>> result = engine.analyze()
    """
    config = MultiTimeframeConfig()

    if timeframes:
        config.timeframes = timeframes

    config.require_htf_confirmation = require_htf_confirmation
    config.block_on_conflict = block_on_conflict

    return MultiTimeframeEngine(config)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resample_to_timeframe(candles: List[OHLCV], source_tf: Timeframe,
                          target_tf: Timeframe) -> List[OHLCV]:
    """
    Resample candles from source timeframe to target timeframe.

    Args:
        candles: Source candles
        source_tf: Source timeframe
        target_tf: Target timeframe (must be larger)

    Returns:
        Resampled candles in target timeframe
    """
    if target_tf.minutes <= source_tf.minutes:
        raise ValueError("Target timeframe must be larger than source")

    ratio = target_tf.minutes // source_tf.minutes
    resampled = []

    for i in range(0, len(candles) - ratio + 1, ratio):
        group = candles[i:i + ratio]
        if len(group) < ratio:
            break

        new_candle = OHLCV(
            timestamp=group[0].timestamp,
            open=group[0].open,
            high=max(c.high for c in group),
            low=min(c.low for c in group),
            close=group[-1].close,
            volume=sum(c.volume for c in group)
        )
        resampled.append(new_candle)

    return resampled


def generate_synthetic_candles(base_price: float, volatility: float,
                               trend: float, count: int,
                               timeframe: Timeframe) -> List[OHLCV]:
    """
    Generate synthetic OHLCV data for testing.

    Args:
        base_price: Starting price
        volatility: Price volatility (e.g., 0.02 for 2%)
        trend: Trend direction (-1 to +1)
        count: Number of candles to generate
        timeframe: Timeframe for timestamps

    Returns:
        List of synthetic OHLCV candles
    """
    candles = []
    current_price = base_price
    current_time = datetime.now() - timedelta(minutes=timeframe.minutes * count)

    for _ in range(count):
        # Generate OHLC
        change = np.random.normal(trend * 0.001, volatility)
        open_price = current_price
        close_price = current_price * (1 + change)

        range_size = abs(close_price - open_price) * (1 + np.random.uniform(0.5, 2))
        high = max(open_price, close_price) + range_size * np.random.uniform(0.1, 0.5)
        low = min(open_price, close_price) - range_size * np.random.uniform(0.1, 0.5)

        volume = np.random.uniform(1000, 10000)

        candles.append(OHLCV(
            timestamp=current_time,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume
        ))

        current_price = close_price
        current_time += timedelta(minutes=timeframe.minutes)

    return candles


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'Timeframe',
    'TrendState',
    'MomentumState',
    'SignalStrength',
    'SignalType',

    # Data classes
    'OHLCV',
    'TimeframeAnalysis',
    'AlignmentResult',
    'MultiTimeframeConfig',

    # Classes
    'TechnicalIndicators',
    'TimeframeAnalyzer',
    'MultiTimeframeEngine',

    # Factory
    'create_multi_timeframe_engine',

    # Utilities
    'resample_to_timeframe',
    'generate_synthetic_candles',
]
