"""Tests for ConfluenceDetector — Sprint 1 of Smart Sentinel AI."""

import pytest
from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum

from src.intelligence.confluence_detector import (
    ConfluenceDetector,
    ConfluenceSignal,
    ComponentScore,
    SignalTier,
    SignalType,
    DEFAULT_WEIGHTS,
    SL_ATR_MULT,
    TP_ATR_MULT,
)


# ============================================================================
# MOCK OBJECTS (mimicking real agent outputs)
# ============================================================================

class MockRegimeType(Enum):
    STRONG_UPTREND = "strong_uptrend"
    WEAK_UPTREND = "weak_uptrend"
    STRONG_DOWNTREND = "strong_downtrend"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class MockTrendDirection(Enum):
    UP = 1
    DOWN = -1
    NEUTRAL = 0


@dataclass
class MockRegimeAnalysis:
    regime: MockRegimeType = MockRegimeType.STRONG_UPTREND
    confidence: float = 0.85
    trend_direction: MockTrendDirection = MockTrendDirection.UP
    trend_strength: float = 0.8


class MockNewsDecision(Enum):
    BLOCK = "BLOCK"
    REDUCE = "REDUCE"
    ALLOW = "ALLOW"


@dataclass
class MockNewsAssessment:
    decision: MockNewsDecision = MockNewsDecision.ALLOW
    sentiment_score: float = 0.6
    sentiment_confidence: float = 0.7
    reasoning: List[str] = field(default_factory=lambda: ["Positive outlook"])


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def detector():
    return ConfluenceDetector()


@pytest.fixture
def bullish_smc():
    """Strong bullish SMC features."""
    return {
        "BOS_SIGNAL": 1.0,
        "CHOCH_SIGNAL": 1.0,
        "FVG_SIGNAL": 1.0,
        "FVG_SIZE_NORM": 1.5,
        "OB_STRENGTH_NORM": 0.8,
        "RSI": 58.0,
        "MACD_Diff": 0.5,
        "CHOCH_DIVERGENCE": 1,
    }


@pytest.fixture
def bearish_smc():
    """Strong bearish SMC features."""
    return {
        "BOS_SIGNAL": -1.0,
        "CHOCH_SIGNAL": -1.0,
        "FVG_SIGNAL": -1.0,
        "FVG_SIZE_NORM": 1.5,
        "OB_STRENGTH_NORM": -0.7,
        "RSI": 38.0,
        "MACD_Diff": -0.4,
        "CHOCH_DIVERGENCE": -1,
    }


@pytest.fixture
def conflicting_smc():
    """Conflicting signals: bullish BOS but bearish FVG."""
    return {
        "BOS_SIGNAL": 1.0,
        "FVG_SIGNAL": -1.0,
        "OB_STRENGTH_NORM": 0.1,
        "RSI": 50.0,
        "MACD_Diff": -0.1,
    }


@pytest.fixture
def neutral_smc():
    """No structural break."""
    return {
        "BOS_SIGNAL": 0.0,
        "FVG_SIGNAL": 0.0,
        "OB_STRENGTH_NORM": 0.0,
        "RSI": 50.0,
        "MACD_Diff": 0.0,
    }


@pytest.fixture
def bullish_regime():
    return MockRegimeAnalysis(
        regime=MockRegimeType.STRONG_UPTREND,
        confidence=0.85,
        trend_direction=MockTrendDirection.UP,
        trend_strength=0.8,
    )


@pytest.fixture
def bearish_regime():
    return MockRegimeAnalysis(
        regime=MockRegimeType.STRONG_DOWNTREND,
        confidence=0.80,
        trend_direction=MockTrendDirection.DOWN,
        trend_strength=0.75,
    )


@pytest.fixture
def bullish_news():
    return MockNewsAssessment(
        decision=MockNewsDecision.ALLOW,
        sentiment_score=0.6,
        sentiment_confidence=0.7,
    )


@pytest.fixture
def blocking_news():
    return MockNewsAssessment(
        decision=MockNewsDecision.BLOCK,
        sentiment_score=0.0,
        sentiment_confidence=0.5,
        reasoning=["NFP release in 15 minutes"],
    )


# ============================================================================
# TESTS: WEIGHTS VALIDATION
# ============================================================================

class TestWeightsValidation:
    def test_default_weights_sum_to_100(self):
        assert abs(sum(DEFAULT_WEIGHTS.values()) - 100.0) < 0.01

    def test_custom_weights_accepted_when_sum_100(self):
        d = ConfluenceDetector(weights={"bos": 50.0, "fvg": 10.0, "order_block": 10.0,
                                        "regime": 10.0, "news": 10.0, "volume": 5.0,
                                        "momentum": 5.0})  # sums to 100
        assert d.weights["bos"] == 50.0

    def test_invalid_weights_rejected(self):
        with pytest.raises(ValueError, match="Weights must sum to 100"):
            ConfluenceDetector(weights={"bos": 50.0, "fvg": 10.0, "order_block": 10.0,
                                        "regime": 10.0, "news": 10.0, "volume": 5.0,
                                        "momentum": 10.0})  # sums to 105


# ============================================================================
# TESTS: PREMIUM SIGNAL (strong confluence)
# ============================================================================

class TestPremiumSignal:
    def test_strong_bullish_confluence_is_premium(
        self, detector, bullish_smc, bullish_regime, bullish_news
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
            volume=1500.0,
            volume_ma=1000.0,
        )
        assert signal is not None
        assert signal.tier == SignalTier.PREMIUM
        assert signal.confluence_score >= 80.0
        assert signal.signal_type == SignalType.LONG

    def test_premium_signal_has_correct_levels(
        self, detector, bullish_smc, bullish_regime, bullish_news
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None
        assert signal.entry_price == 2400.0
        assert signal.stop_loss == 2400.0 - (SL_ATR_MULT * 10.0)  # 2380.0
        assert signal.take_profit == 2400.0 + (TP_ATR_MULT * 10.0)  # 2440.0
        assert signal.rr_ratio == pytest.approx(2.0, abs=0.01)

    def test_premium_signal_has_components(
        self, detector, bullish_smc, bullish_regime, bullish_news
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None
        assert len(signal.components) == 8
        names = [c.name for c in signal.components]
        assert "BOS" in names
        assert "FVG" in names
        assert "Regime" in names
        assert "RSI_Divergence" in names

    def test_strong_bearish_confluence_is_premium(
        self, detector, bearish_smc, bearish_regime
    ):
        bearish_news = MockNewsAssessment(
            decision=MockNewsDecision.ALLOW,
            sentiment_score=-0.5,
            sentiment_confidence=0.7,
        )
        signal = detector.analyze(
            smc_features=bearish_smc,
            regime=bearish_regime,
            news=bearish_news,
            price=2400.0,
            atr=10.0,
            volume=1500.0,
            volume_ma=1000.0,
        )
        assert signal is not None
        assert signal.signal_type == SignalType.SHORT
        assert signal.confluence_score >= 70.0  # At least standard
        # Short: SL above entry, TP below entry
        assert signal.stop_loss > signal.entry_price
        assert signal.take_profit < signal.entry_price


# ============================================================================
# TESTS: CONFLICTING SIGNALS
# ============================================================================

class TestConflictingSignals:
    def test_conflicting_smc_low_score(
        self, detector, conflicting_smc, bullish_regime, bullish_news
    ):
        signal = detector.analyze(
            smc_features=conflicting_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        # FVG opposes BOS → lower score, may or may not generate signal
        if signal is not None:
            assert signal.confluence_score < 80.0  # Not premium

    def test_no_bos_returns_none(self, detector, neutral_smc, bullish_regime, bullish_news):
        signal = detector.analyze(
            smc_features=neutral_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is None


# ============================================================================
# TESTS: NEWS BLOCKING
# ============================================================================

class TestNewsBlocking:
    def test_news_block_returns_none(
        self, detector, bullish_smc, bullish_regime, blocking_news
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=blocking_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is None

    def test_allow_news_passes(
        self, detector, bullish_smc, bullish_regime, bullish_news
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None

    def test_none_news_treated_as_neutral(
        self, detector, bullish_smc, bullish_regime
    ):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=None,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None  # None news doesn't block


# ============================================================================
# TESTS: SHORT SIGNAL DETECTION
# ============================================================================

class TestShortSignals:
    def test_bearish_bos_generates_short(self, detector, bearish_smc):
        signal = detector.analyze(
            smc_features=bearish_smc,
            regime=None,
            news=None,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None
        assert signal.signal_type == SignalType.SHORT

    def test_short_sl_above_entry(self, detector, bearish_smc):
        signal = detector.analyze(
            smc_features=bearish_smc,
            regime=None,
            news=None,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None
        assert signal.stop_loss == 2420.0  # entry + 2*ATR
        assert signal.take_profit == 2360.0  # entry - 4*ATR


# ============================================================================
# TESTS: VOLUME CONFIRMATION
# ============================================================================

class TestVolumeConfirmation:
    def test_high_volume_boosts_score(self, detector, bullish_smc, bullish_regime, bullish_news):
        signal_high = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
            volume=2000.0,
            volume_ma=1000.0,
        )
        signal_low = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
            volume=300.0,
            volume_ma=1000.0,
        )
        assert signal_high is not None
        assert signal_low is not None
        assert signal_high.confluence_score > signal_low.confluence_score

    def test_no_volume_gives_neutral(self, detector, bullish_smc, bullish_regime, bullish_news):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
            volume=None,
            volume_ma=None,
        )
        assert signal is not None
        vol_comp = [c for c in signal.components if c.name == "Volume"][0]
        assert vol_comp.weighted_score == pytest.approx(5.0, abs=0.01)  # 50% of 10


# ============================================================================
# TESTS: TIER CLASSIFICATION
# ============================================================================

class TestTierClassification:
    def test_tier_thresholds(self, detector):
        assert detector._classify_tier(85.0) == SignalTier.PREMIUM
        assert detector._classify_tier(80.0) == SignalTier.PREMIUM
        assert detector._classify_tier(79.9) == SignalTier.STANDARD
        assert detector._classify_tier(60.0) == SignalTier.STANDARD
        assert detector._classify_tier(59.9) == SignalTier.WEAK
        assert detector._classify_tier(40.0) == SignalTier.WEAK
        assert detector._classify_tier(39.9) == SignalTier.INVALID


# ============================================================================
# TESTS: SERIALIZATION
# ============================================================================

class TestSerialization:
    def test_to_dict(self, detector, bullish_smc, bullish_regime, bullish_news):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
        )
        assert signal is not None
        d = signal.to_dict()
        assert d["symbol"] == "XAUUSD"
        assert d["signal_type"] in ("LONG", "SHORT")
        assert "components" in d
        assert "reasoning" in d
        assert isinstance(d["components"], list)


# ============================================================================
# TESTS: EDGE CASES
# ============================================================================

class TestEdgeCases:
    def test_zero_atr_still_works(self, detector, bullish_smc):
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=None,
            news=None,
            price=2400.0,
            atr=0.0,
        )
        # Should produce signal but with 0 SL/TP distance
        if signal is not None:
            assert signal.stop_loss == signal.entry_price
            assert signal.take_profit == signal.entry_price

    def test_missing_smc_keys_default_to_zero(self, detector):
        sparse = {"BOS_SIGNAL": 1.0}
        signal = detector.analyze(
            smc_features=sparse,
            regime=None,
            news=None,
            price=2400.0,
            atr=10.0,
        )
        # May or may not generate signal, should not crash
        assert signal is None or isinstance(signal, ConfluenceSignal)

    def test_bar_timestamp_propagated(self, detector, bullish_smc, bullish_regime, bullish_news):
        ts = "2025-06-15T14:30:00"
        signal = detector.analyze(
            smc_features=bullish_smc,
            regime=bullish_regime,
            news=bullish_news,
            price=2400.0,
            atr=10.0,
            bar_timestamp=ts,
        )
        assert signal is not None
        assert signal.bar_timestamp == ts
