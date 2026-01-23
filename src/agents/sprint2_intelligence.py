# =============================================================================
# SPRINT 2 INTELLIGENCE MODULE - Unified Intelligence Layer
# =============================================================================
"""
Unified Intelligence Layer for Institutional Trading - Sprint 2

This module integrates all Sprint 2 intelligence components into a cohesive
system for sophisticated market analysis and decision making.

Integrated Components:
- SentimentAnalyzer: NLP-based news sentiment analysis (FinBERT)
- RegimePredictor: HMM-based market regime detection
- MultiTimeframeEngine: Multi-timeframe technical analysis
- EnsembleRiskModel: ML ensemble for risk prediction

Architecture:
┌─────────────────────────────────────────────────────────────────────────────┐
│                    INTELLIGENT TRADING SYSTEM                               │
│                                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  SENTIMENT  │  │   REGIME    │  │  MULTI-TF   │  │  ENSEMBLE   │        │
│  │  ANALYZER   │  │  PREDICTOR  │  │   ENGINE    │  │    MODEL    │        │
│  │  (FinBERT)  │  │   (HMM)     │  │  (4 TFs)    │  │  (ML)       │        │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘        │
│         │                │                │                │                │
│         └────────────────┴────────────────┴────────────────┘                │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │   INTELLIGENCE AGGREGATOR   │                         │
│                    │  - Signal Fusion            │                         │
│                    │  - Confidence Weighting     │                         │
│                    │  - Conflict Resolution      │                         │
│                    └──────────────┬──────────────┘                         │
│                                   │                                         │
│                    ┌──────────────▼──────────────┐                         │
│                    │    DECISION ENGINE          │                         │
│                    │  - Trade Recommendation     │                         │
│                    │  - Position Sizing          │                         │
│                    │  - Risk Adjustment          │                         │
│                    └─────────────────────────────┘                         │
└─────────────────────────────────────────────────────────────────────────────┘

Version: 2.0.0
Author: TradingBot Team
License: Proprietary - Commercial Use
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from collections import deque
import logging
import json

logger = logging.getLogger(__name__)

# =============================================================================
# LAZY IMPORTS FOR SPRINT 2 COMPONENTS
# =============================================================================

_sentiment_analyzer = None
_regime_predictor = None
_mtf_engine = None
_ensemble_model = None


def _get_sentiment_analyzer():
    """Lazy import for sentiment analyzer."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from .sentiment_analyzer import (
            SentimentAnalyzer, SentimentResult, AggregatedSentiment,
            SentimentLabel, create_sentiment_analyzer
        )
        _sentiment_analyzer = {
            'SentimentAnalyzer': SentimentAnalyzer,
            'SentimentResult': SentimentResult,
            'AggregatedSentiment': AggregatedSentiment,
            'SentimentLabel': SentimentLabel,
            'create_sentiment_analyzer': create_sentiment_analyzer
        }
    return _sentiment_analyzer


def _get_regime_predictor():
    """Lazy import for regime predictor."""
    global _regime_predictor
    if _regime_predictor is None:
        from .regime_predictor import (
            RegimePredictor, RegimePrediction, PredictedRegime,
            create_regime_predictor
        )
        _regime_predictor = {
            'RegimePredictor': RegimePredictor,
            'RegimePrediction': RegimePrediction,
            'PredictedRegime': PredictedRegime,
            'create_regime_predictor': create_regime_predictor
        }
    return _regime_predictor


def _get_mtf_engine():
    """Lazy import for multi-timeframe engine."""
    global _mtf_engine
    if _mtf_engine is None:
        from .multi_timeframe import (
            MultiTimeframeEngine, AlignmentResult, TimeframeAnalysis,
            Timeframe, SignalType, SignalStrength, OHLCV,
            create_multi_timeframe_engine
        )
        _mtf_engine = {
            'MultiTimeframeEngine': MultiTimeframeEngine,
            'AlignmentResult': AlignmentResult,
            'TimeframeAnalysis': TimeframeAnalysis,
            'Timeframe': Timeframe,
            'SignalType': SignalType,
            'SignalStrength': SignalStrength,
            'OHLCV': OHLCV,
            'create_multi_timeframe_engine': create_multi_timeframe_engine
        }
    return _mtf_engine


def _get_ensemble_model():
    """Lazy import for ensemble model."""
    global _ensemble_model
    if _ensemble_model is None:
        from .ensemble_risk_model import (
            EnsembleRiskModel, EnsemblePrediction, RiskCategory,
            PredictionType, create_ensemble_risk_model
        )
        _ensemble_model = {
            'EnsembleRiskModel': EnsembleRiskModel,
            'EnsemblePrediction': EnsemblePrediction,
            'RiskCategory': RiskCategory,
            'PredictionType': PredictionType,
            'create_ensemble_risk_model': create_ensemble_risk_model
        }
    return _ensemble_model


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class TradingAction(Enum):
    """Recommended trading action."""
    STRONG_BUY = auto()
    BUY = auto()
    WEAK_BUY = auto()
    HOLD = auto()
    WEAK_SELL = auto()
    SELL = auto()
    STRONG_SELL = auto()
    CLOSE_LONG = auto()
    CLOSE_SHORT = auto()
    NO_ACTION = auto()
    BLOCKED = auto()

    @property
    def score(self) -> float:
        """Numerical score from -1 to +1."""
        scores = {
            TradingAction.STRONG_BUY: 1.0,
            TradingAction.BUY: 0.7,
            TradingAction.WEAK_BUY: 0.3,
            TradingAction.HOLD: 0.0,
            TradingAction.WEAK_SELL: -0.3,
            TradingAction.SELL: -0.7,
            TradingAction.STRONG_SELL: -1.0,
            TradingAction.CLOSE_LONG: -0.5,
            TradingAction.CLOSE_SHORT: 0.5,
            TradingAction.NO_ACTION: 0.0,
            TradingAction.BLOCKED: 0.0
        }
        return scores.get(self, 0.0)

    @property
    def is_bullish(self) -> bool:
        return self.score > 0

    @property
    def is_bearish(self) -> bool:
        return self.score < 0


class ConfidenceLevel(Enum):
    """Confidence level for decisions."""
    VERY_HIGH = auto()
    HIGH = auto()
    MODERATE = auto()
    LOW = auto()
    VERY_LOW = auto()
    UNCERTAIN = auto()

    @classmethod
    def from_score(cls, score: float) -> 'ConfidenceLevel':
        if score >= 0.85:
            return cls.VERY_HIGH
        elif score >= 0.70:
            return cls.HIGH
        elif score >= 0.55:
            return cls.MODERATE
        elif score >= 0.40:
            return cls.LOW
        elif score >= 0.25:
            return cls.VERY_LOW
        else:
            return cls.UNCERTAIN


class MarketCondition(Enum):
    """Overall market condition assessment."""
    STRONGLY_FAVORABLE = auto()
    FAVORABLE = auto()
    NEUTRAL = auto()
    UNFAVORABLE = auto()
    STRONGLY_UNFAVORABLE = auto()
    CRISIS = auto()


class RiskLevel(Enum):
    """Risk level enumeration."""
    VERY_LOW = auto()
    LOW = auto()
    MODERATE = auto()
    HIGH = auto()
    VERY_HIGH = auto()
    EXTREME = auto()

    @classmethod
    def from_score(cls, score: float) -> 'RiskLevel':
        if score < 0.15:
            return cls.VERY_LOW
        elif score < 0.35:
            return cls.LOW
        elif score < 0.55:
            return cls.MODERATE
        elif score < 0.75:
            return cls.HIGH
        elif score < 0.9:
            return cls.VERY_HIGH
        else:
            return cls.EXTREME


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ComponentSignal:
    """Signal from a single intelligence component."""
    component_name: str
    signal_value: float      # -1 to +1
    confidence: float        # 0 to 1
    weight: float           # Component weight in ensemble
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def weighted_signal(self) -> float:
        return self.signal_value * self.confidence * self.weight


@dataclass
class IntelligenceReport:
    """Comprehensive intelligence report from all components."""
    timestamp: datetime
    symbol: str

    # Individual component results (stored as dicts for flexibility)
    sentiment_result: Optional[Dict[str, Any]] = None
    regime_result: Optional[Dict[str, Any]] = None
    mtf_result: Optional[Dict[str, Any]] = None
    risk_result: Optional[Dict[str, Any]] = None

    # Component signals
    signals: List[ComponentSignal] = field(default_factory=list)

    # Aggregated analysis
    aggregated_signal: float = 0.0
    aggregated_confidence: float = 0.0
    signal_agreement: float = 0.0

    # Trading recommendation
    recommended_action: TradingAction = TradingAction.HOLD
    confidence_level: ConfidenceLevel = ConfidenceLevel.UNCERTAIN
    position_size_factor: float = 1.0

    # Market assessment
    market_condition: MarketCondition = MarketCondition.NEUTRAL
    risk_level: RiskLevel = RiskLevel.MODERATE

    # Warnings and notes
    warnings: List[str] = field(default_factory=list)
    blocking_reasons: List[str] = field(default_factory=list)
    is_blocked: bool = False

    # Explanation
    explanation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "symbol": self.symbol,
            "sentiment": self.sentiment_result,
            "regime": self.regime_result,
            "multi_timeframe": self.mtf_result,
            "risk_prediction": self.risk_result,
            "signals": [
                {
                    "component": s.component_name,
                    "value": s.signal_value,
                    "confidence": s.confidence,
                    "weighted": s.weighted_signal
                } for s in self.signals
            ],
            "aggregated_signal": self.aggregated_signal,
            "aggregated_confidence": self.aggregated_confidence,
            "signal_agreement": self.signal_agreement,
            "recommended_action": self.recommended_action.name,
            "confidence_level": self.confidence_level.name,
            "position_size_factor": self.position_size_factor,
            "market_condition": self.market_condition.name,
            "risk_level": self.risk_level.name,
            "warnings": self.warnings,
            "blocking_reasons": self.blocking_reasons,
            "is_blocked": self.is_blocked,
            "explanation": self.explanation
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class Sprint2Config:
    """Configuration for Sprint 2 intelligence system."""
    # Component weights (must sum to 1.0)
    sentiment_weight: float = 0.20
    regime_weight: float = 0.25
    multi_timeframe_weight: float = 0.35
    risk_model_weight: float = 0.20

    # Thresholds
    min_confidence_to_trade: float = 0.5
    min_agreement_to_trade: float = 0.4
    high_risk_threshold: float = 0.7
    crisis_risk_threshold: float = 0.85

    # Blocking conditions
    block_on_crisis: bool = True
    block_on_regime_transition: bool = True
    block_on_low_confidence: bool = True

    # Position sizing
    use_dynamic_sizing: bool = True
    max_position_factor: float = 2.0
    min_position_factor: float = 0.25

    # Regime adjustments
    reduce_in_volatile_regime: float = 0.5
    reduce_in_crisis_regime: float = 0.25

    # News impact
    news_impact_duration_hours: float = 4.0

    def validate(self) -> bool:
        """Validate configuration."""
        total_weight = (self.sentiment_weight + self.regime_weight +
                       self.multi_timeframe_weight + self.risk_model_weight)
        if not 0.99 <= total_weight <= 1.01:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")
        return True


# =============================================================================
# SPRINT 2 INTELLIGENCE SYSTEM
# =============================================================================

class Sprint2Intelligence:
    """
    Unified Intelligence System combining all Sprint 2 components.

    This is the main entry point for Sprint 2 intelligence features.
    """

    def __init__(self, config: Optional[Sprint2Config] = None):
        """Initialize Sprint 2 Intelligence System."""
        self.config = config or Sprint2Config()
        self.config.validate()

        # Component instances (lazy loaded)
        self._sentiment_analyzer = None
        self._regime_predictor = None
        self._mtf_engine = None
        self._risk_model = None

        # State tracking
        self.last_report: Optional[IntelligenceReport] = None
        self.report_history: deque = deque(maxlen=100)

        # News cache
        self.recent_news: deque = deque(maxlen=50)
        self.news_timestamps: deque = deque(maxlen=50)

        # Statistics
        self.decisions_made: int = 0
        self.blocked_decisions: int = 0

        logger.info("Sprint2Intelligence initialized")

    def _init_sentiment(self):
        """Initialize sentiment analyzer."""
        if self._sentiment_analyzer is None:
            mods = _get_sentiment_analyzer()
            self._sentiment_analyzer = mods['create_sentiment_analyzer']()

    def _init_regime(self):
        """Initialize regime predictor."""
        if self._regime_predictor is None:
            mods = _get_regime_predictor()
            self._regime_predictor = mods['create_regime_predictor']()

    def _init_mtf(self):
        """Initialize multi-timeframe engine."""
        if self._mtf_engine is None:
            mods = _get_mtf_engine()
            self._mtf_engine = mods['create_multi_timeframe_engine']()

    def _init_risk_model(self):
        """Initialize risk model."""
        if self._risk_model is None:
            mods = _get_ensemble_model()
            self._risk_model = mods['create_ensemble_risk_model']()

    # =========================================================================
    # DATA INPUT METHODS
    # =========================================================================

    def add_news(self, headlines: List[str],
                 timestamp: Optional[datetime] = None) -> None:
        """Add news headlines for sentiment analysis."""
        self._init_sentiment()
        ts = timestamp or datetime.now()
        for headline in headlines:
            self.recent_news.append(headline)
            self.news_timestamps.append(ts)

    def add_price(self, price: float, volume: float = 0.0) -> None:
        """Update regime predictor with new price."""
        self._init_regime()
        self._regime_predictor.update(price, volume)

    def add_candles(self, timeframe: str, candles: List[Dict[str, Any]]) -> None:
        """
        Add OHLCV candles for multi-timeframe analysis.

        Args:
            timeframe: One of "1h", "4h", "1d", "1w"
            candles: List of dicts with keys: timestamp, open, high, low, close, volume
        """
        self._init_mtf()
        mods = _get_mtf_engine()

        # Convert timeframe string to enum
        tf_map = {
            "1m": mods['Timeframe'].M1,
            "5m": mods['Timeframe'].M5,
            "15m": mods['Timeframe'].M15,
            "30m": mods['Timeframe'].M30,
            "1h": mods['Timeframe'].H1,
            "4h": mods['Timeframe'].H4,
            "1d": mods['Timeframe'].D1,
            "1w": mods['Timeframe'].W1
        }

        tf = tf_map.get(timeframe.lower())
        if tf is None:
            logger.warning(f"Unknown timeframe: {timeframe}")
            return

        # Convert to OHLCV objects
        ohlcv_list = []
        for c in candles:
            ohlcv = mods['OHLCV'](
                timestamp=c.get('timestamp', datetime.now()),
                open=float(c['open']),
                high=float(c['high']),
                low=float(c['low']),
                close=float(c['close']),
                volume=float(c.get('volume', 0))
            )
            ohlcv_list.append(ohlcv)

        self._mtf_engine.add_data(tf, ohlcv_list)

    def train_risk_model(self, X: np.ndarray, y: np.ndarray,
                         feature_names: Optional[List[str]] = None) -> None:
        """Train the ensemble risk model."""
        self._init_risk_model()
        self._risk_model.fit(X, y, feature_names)
        logger.info("Risk model trained successfully")

    # =========================================================================
    # ANALYSIS METHODS
    # =========================================================================

    def analyze(self, symbol: str = "UNKNOWN",
                features: Optional[np.ndarray] = None) -> IntelligenceReport:
        """
        Perform comprehensive market analysis.

        Args:
            symbol: Trading symbol
            features: Optional features for risk model

        Returns:
            IntelligenceReport with complete analysis
        """
        self.decisions_made += 1
        timestamp = datetime.now()

        report = IntelligenceReport(
            timestamp=timestamp,
            symbol=symbol
        )

        signals: List[ComponentSignal] = []

        # 1. Sentiment Analysis
        sentiment_signal = self._analyze_sentiment()
        if sentiment_signal:
            signals.append(sentiment_signal)
            report.sentiment_result = sentiment_signal.metadata

        # 2. Regime Prediction
        regime_signal = self._analyze_regime()
        if regime_signal:
            signals.append(regime_signal)
            report.regime_result = regime_signal.metadata

        # 3. Multi-Timeframe Analysis
        mtf_signal = self._analyze_mtf()
        if mtf_signal:
            signals.append(mtf_signal)
            report.mtf_result = mtf_signal.metadata

        # 4. Risk Model
        if features is not None and self._risk_model is not None:
            risk_signal = self._analyze_risk(features)
            if risk_signal:
                signals.append(risk_signal)
                report.risk_result = risk_signal.metadata

        report.signals = signals

        # Aggregate
        if signals:
            report.aggregated_signal = self._aggregate_signals(signals)
            report.aggregated_confidence = self._calculate_confidence(signals)
            report.signal_agreement = self._calculate_agreement(signals)

        # Check blocking
        report.blocking_reasons = self._check_blocking(report)
        report.is_blocked = len(report.blocking_reasons) > 0

        if report.is_blocked:
            self.blocked_decisions += 1
            report.recommended_action = TradingAction.BLOCKED
        else:
            report.recommended_action = self._generate_action(report)

        report.confidence_level = ConfidenceLevel.from_score(report.aggregated_confidence)
        report.position_size_factor = self._calculate_position_size(report)
        report.market_condition = self._assess_market(report)
        report.risk_level = self._assess_risk(report)
        report.warnings = self._generate_warnings(report)
        report.explanation = self._generate_explanation(report)

        self.last_report = report
        self.report_history.append(report)

        return report

    def _analyze_sentiment(self) -> Optional[ComponentSignal]:
        """Analyze sentiment from news."""
        if not self.recent_news or self._sentiment_analyzer is None:
            return None

        cutoff = datetime.now() - timedelta(hours=self.config.news_impact_duration_hours)
        recent_texts = [t for t, ts in zip(self.recent_news, self.news_timestamps) if ts >= cutoff]

        if not recent_texts:
            return None

        recent_times = [ts for ts in self.news_timestamps if ts >= cutoff]
        aggregated = self._sentiment_analyzer.analyze_batch(recent_texts, recent_times)

        return ComponentSignal(
            component_name="sentiment",
            signal_value=aggregated.overall_score,
            confidence=aggregated.confidence,
            weight=self.config.sentiment_weight,
            metadata={
                "category": aggregated.overall_label.name,
                "score": aggregated.overall_score,
                "confidence": aggregated.confidence,
                "article_count": aggregated.num_texts
            }
        )

    def _analyze_regime(self) -> Optional[ComponentSignal]:
        """Analyze market regime."""
        if self._regime_predictor is None:
            return None

        prediction = self._regime_predictor.predict()
        if prediction is None:
            return None

        mods = _get_regime_predictor()
        PredictedRegime = mods['PredictedRegime']

        # Map PredictedRegime to signal values
        regime_signals = {
            PredictedRegime.BULL_TREND: 0.7,
            PredictedRegime.BEAR_TREND: -0.7,
            PredictedRegime.RANGE_BOUND: 0.0,
            PredictedRegime.HIGH_VOLATILITY: -0.3,
            PredictedRegime.TRANSITION: 0.0
        }

        signal_value = regime_signals.get(prediction.current_regime, 0.0)

        return ComponentSignal(
            component_name="regime",
            signal_value=signal_value,
            confidence=prediction.current_probability,
            weight=self.config.regime_weight,
            metadata={
                "regime": prediction.current_regime.name,
                "probability": prediction.current_probability,
                "transition_probability": prediction.transition_probability,
                "prediction_confidence": prediction.prediction_confidence,
                "stability": 1.0 - prediction.transition_probability
            }
        )

    def _analyze_mtf(self) -> Optional[ComponentSignal]:
        """Analyze multi-timeframe signals."""
        if self._mtf_engine is None:
            return None

        result = self._mtf_engine.analyze()
        if result is None:
            return None

        return ComponentSignal(
            component_name="multi_timeframe",
            signal_value=result.aggregated_signal.score,
            confidence=result.confidence,
            weight=self.config.multi_timeframe_weight,
            metadata={
                "signal": result.aggregated_signal.name,
                "strength": result.signal_strength.name,
                "trend_alignment": result.trend_alignment,
                "momentum_alignment": result.momentum_alignment,
                "has_conflict": result.has_conflict,
                "overall_alignment": result.overall_alignment
            }
        )

    def _analyze_risk(self, features: np.ndarray) -> Optional[ComponentSignal]:
        """Analyze risk using ML model."""
        if self._risk_model is None or not self._risk_model.fitted:
            return None

        mods = _get_ensemble_model()
        prediction = self._risk_model.predict(features, mods['PredictionType'].RISK_LEVEL)

        # Higher risk = more bearish
        signal_value = -((prediction.ensemble_prediction - 0.5) * 2)
        signal_value = np.clip(signal_value, -1, 1)

        return ComponentSignal(
            component_name="risk_model",
            signal_value=float(signal_value),
            confidence=prediction.ensemble_confidence,
            weight=self.config.risk_model_weight,
            metadata={
                "risk_category": prediction.risk_category.name,
                "risk_score": prediction.ensemble_prediction,
                "models_used": prediction.models_used,
                "prediction_std": prediction.prediction_std
            }
        )

    def _aggregate_signals(self, signals: List[ComponentSignal]) -> float:
        """Aggregate signals using weighted average."""
        if not signals:
            return 0.0

        total_weighted = sum(s.weighted_signal for s in signals)
        total_weight = sum(s.weight * s.confidence for s in signals)

        return total_weighted / total_weight if total_weight > 0 else 0.0

    def _calculate_confidence(self, signals: List[ComponentSignal]) -> float:
        """Calculate overall confidence."""
        if not signals:
            return 0.0

        weighted_conf = sum(s.confidence * s.weight for s in signals)
        total_weight = sum(s.weight for s in signals)
        base_confidence = weighted_conf / total_weight if total_weight > 0 else 0.0

        agreement = self._calculate_agreement(signals)
        confidence = base_confidence * (0.7 + 0.3 * agreement)

        return min(0.95, confidence)

    def _calculate_agreement(self, signals: List[ComponentSignal]) -> float:
        """Calculate signal agreement (0-1)."""
        if len(signals) < 2:
            return 1.0

        signs = [1 if s.signal_value > 0 else -1 if s.signal_value < 0 else 0
                 for s in signals]

        positive = sum(1 for s in signs if s > 0)
        negative = sum(1 for s in signs if s < 0)
        total_directional = positive + negative

        if total_directional == 0:
            return 1.0

        return max(positive, negative) / total_directional

    def _check_blocking(self, report: IntelligenceReport) -> List[str]:
        """Check blocking conditions."""
        reasons = []

        if report.aggregated_confidence < self.config.min_confidence_to_trade:
            if self.config.block_on_low_confidence:
                reasons.append(f"Low confidence ({report.aggregated_confidence:.2f})")

        if report.signal_agreement < self.config.min_agreement_to_trade:
            reasons.append(f"Signal disagreement ({report.signal_agreement:.2f})")

        if report.regime_result:
            if report.regime_result.get('regime') == 'CRISIS':
                if self.config.block_on_crisis:
                    reasons.append("Crisis regime detected")
            if report.regime_result.get('stability', 1.0) < 0.5:
                if self.config.block_on_regime_transition:
                    reasons.append("Regime instability detected")

        if report.mtf_result and report.mtf_result.get('has_conflict'):
            reasons.append("Timeframe conflict detected")

        return reasons

    def _generate_action(self, report: IntelligenceReport) -> TradingAction:
        """Generate trading action."""
        signal = report.aggregated_signal
        confidence = report.aggregated_confidence

        if signal >= 0.6 and confidence >= 0.7:
            return TradingAction.STRONG_BUY
        elif signal >= 0.35 and confidence >= 0.55:
            return TradingAction.BUY
        elif signal >= 0.15 and confidence >= 0.45:
            return TradingAction.WEAK_BUY
        elif signal <= -0.6 and confidence >= 0.7:
            return TradingAction.STRONG_SELL
        elif signal <= -0.35 and confidence >= 0.55:
            return TradingAction.SELL
        elif signal <= -0.15 and confidence >= 0.45:
            return TradingAction.WEAK_SELL
        else:
            return TradingAction.HOLD

    def _calculate_position_size(self, report: IntelligenceReport) -> float:
        """Calculate position size multiplier."""
        if not self.config.use_dynamic_sizing:
            return 1.0

        factor = 1.0
        factor *= (0.5 + 0.5 * report.aggregated_confidence)
        factor *= (0.6 + 0.4 * report.signal_agreement)

        if report.regime_result:
            regime = report.regime_result.get('regime', '')
            if 'VOLATILE' in regime:
                factor *= self.config.reduce_in_volatile_regime
            elif regime == 'CRISIS':
                factor *= self.config.reduce_in_crisis_regime

        return max(self.config.min_position_factor,
                   min(self.config.max_position_factor, factor))

    def _assess_market(self, report: IntelligenceReport) -> MarketCondition:
        """Assess market condition."""
        score = report.aggregated_signal

        if score >= 0.6:
            return MarketCondition.STRONGLY_FAVORABLE
        elif score >= 0.25:
            return MarketCondition.FAVORABLE
        elif score >= -0.25:
            return MarketCondition.NEUTRAL
        elif score >= -0.6:
            return MarketCondition.UNFAVORABLE
        else:
            return MarketCondition.STRONGLY_UNFAVORABLE

    def _assess_risk(self, report: IntelligenceReport) -> RiskLevel:
        """Assess risk level."""
        if report.risk_result:
            return RiskLevel.from_score(report.risk_result.get('risk_score', 0.5))
        return RiskLevel.MODERATE

    def _generate_warnings(self, report: IntelligenceReport) -> List[str]:
        """Generate warnings."""
        warnings = []

        if report.aggregated_confidence < 0.5:
            warnings.append("Low confidence - reduce position size")

        if report.signal_agreement < 0.6:
            warnings.append("Signals not aligned - proceed with caution")

        if report.regime_result:
            if report.regime_result.get('stability', 1) < 0.6:
                warnings.append("Regime may be transitioning")

        return warnings

    def _generate_explanation(self, report: IntelligenceReport) -> str:
        """Generate explanation."""
        lines = [
            f"=== Intelligence Report for {report.symbol} ===",
            f"Timestamp: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "COMPONENT SIGNALS:"
        ]

        for s in report.signals:
            direction = "BULLISH" if s.signal_value > 0 else "BEARISH" if s.signal_value < 0 else "NEUTRAL"
            lines.append(f"  {s.component_name}: {direction} ({s.signal_value:+.2f}), "
                        f"confidence: {s.confidence:.1%}")

        lines.extend([
            "",
            f"AGGREGATED: {report.aggregated_signal:+.2f} "
            f"(confidence: {report.aggregated_confidence:.1%})",
            "",
            f"RECOMMENDATION: {report.recommended_action.name}",
            f"Position Factor: {report.position_size_factor:.2f}x"
        ])

        if report.is_blocked:
            lines.append("\nBLOCKED:")
            for reason in report.blocking_reasons:
                lines.append(f"  - {reason}")

        return "\n".join(lines)

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_quick_signal(self, symbol: str = "UNKNOWN") -> Tuple[TradingAction, float]:
        """Get quick signal without full report."""
        report = self.analyze(symbol)
        return report.recommended_action, report.aggregated_confidence

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return {
            "decisions_made": self.decisions_made,
            "blocked_decisions": self.blocked_decisions,
            "block_rate": self.blocked_decisions / max(1, self.decisions_made),
            "reports_in_history": len(self.report_history),
            "components_initialized": {
                "sentiment": self._sentiment_analyzer is not None,
                "regime": self._regime_predictor is not None,
                "mtf": self._mtf_engine is not None,
                "risk_model": self._risk_model is not None
            }
        }

    def reset(self) -> None:
        """Reset system state."""
        self.recent_news.clear()
        self.news_timestamps.clear()
        self.report_history.clear()

        if self._regime_predictor:
            self._regime_predictor.reset()
        if self._mtf_engine:
            self._mtf_engine.reset()

        self.last_report = None
        self.decisions_made = 0
        self.blocked_decisions = 0

        logger.info("Sprint2Intelligence reset")


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_sprint2_intelligence(
    sentiment_weight: float = 0.20,
    regime_weight: float = 0.25,
    multi_timeframe_weight: float = 0.35,
    risk_model_weight: float = 0.20
) -> Sprint2Intelligence:
    """
    Create a Sprint 2 Intelligence System.

    Args:
        sentiment_weight: Weight for sentiment (0-1)
        regime_weight: Weight for regime (0-1)
        multi_timeframe_weight: Weight for MTF (0-1)
        risk_model_weight: Weight for risk model (0-1)

    Returns:
        Configured Sprint2Intelligence

    Example:
        >>> intel = create_sprint2_intelligence()
        >>> intel.add_news(["Fed raises rates"])
        >>> intel.add_price(1.0850)
        >>> report = intel.analyze("EURUSD")
        >>> print(report.recommended_action)
    """
    config = Sprint2Config(
        sentiment_weight=sentiment_weight,
        regime_weight=regime_weight,
        multi_timeframe_weight=multi_timeframe_weight,
        risk_model_weight=risk_model_weight
    )

    return Sprint2Intelligence(config)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'TradingAction',
    'ConfidenceLevel',
    'MarketCondition',
    'RiskLevel',

    # Data classes
    'ComponentSignal',
    'IntelligenceReport',
    'Sprint2Config',

    # Main class
    'Sprint2Intelligence',

    # Factory
    'create_sprint2_intelligence',
]
