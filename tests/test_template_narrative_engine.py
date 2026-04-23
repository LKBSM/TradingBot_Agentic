"""Tests for TemplateNarrativeEngine — algorithmic replacement for LLMNarrativeEngine."""

from dataclasses import dataclass, field
from typing import List

import pytest

from src.intelligence.llm_narrative_engine import NarrativeTier, SignalNarrative
from src.intelligence.template_narrative_engine import (
    MIN_RR_VALID,
    MIN_SCORE_VALID,
    TemplateNarrativeEngine,
)


STANDARD_TIER_MIN = 60.0  # aligned with ConfluenceDetector.SignalTier.STANDARD


# =============================================================================
# MOCK SIGNAL TYPES (duck-typed to ConfluenceSignal / ComponentScore)
# =============================================================================

@dataclass
class MockComponent:
    name: str
    weighted_score: float
    weight: float
    reasoning: str = ""


@dataclass
class MockSignal:
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    confluence_score: float = 82.0
    tier: str = "PREMIUM"
    entry_price: float = 2400.00
    stop_loss: float = 2380.00
    take_profit: float = 2440.00
    rr_ratio: float = 2.0
    atr: float = 10.0
    components: List[MockComponent] = field(default_factory=lambda: [
        MockComponent("BOS", 14.0, 15.0, "Bullish break of prior swing high"),
        MockComponent("FVG", 13.0, 15.0, "Bullish FVG aligning with direction"),
        MockComponent("OrderBlock", 9.0, 10.0, "Unmitigated bullish OB at entry"),
        MockComponent("Regime", 22.0, 25.0, "Strong uptrend"),
        MockComponent("Volume", 6.0, 10.0, "Above-average volume"),
        MockComponent("Momentum", 5.0, 10.0, "RSI rising from 45"),
    ])
    vol_forecast_atr: float = 11.5
    vol_regime: str = "normal"
    vol_confidence_lower: float = 9.5
    vol_confidence_upper: float = 13.5


@pytest.fixture
def engine() -> TemplateNarrativeEngine:
    return TemplateNarrativeEngine()


@pytest.fixture
def strong_signal() -> MockSignal:
    return MockSignal()


# =============================================================================
# VISUAL TIER
# =============================================================================

def test_visual_tier_returns_no_narrative(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.VISUAL)
    assert isinstance(result, SignalNarrative)
    assert result.tier == NarrativeTier.VISUAL
    assert result.is_valid is True
    assert result.full_narrative == ""
    assert result.cost_usd == 0.0
    assert result.latency_ms >= 0.0


def test_visual_tier_is_default(engine, strong_signal):
    result = engine.generate_narrative(strong_signal)
    assert result.tier == NarrativeTier.VISUAL


# =============================================================================
# VALIDATOR TIER
# =============================================================================

def test_validator_accepts_strong_signal(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.tier == NarrativeTier.VALIDATOR
    assert result.is_valid is True
    assert "validated" in result.validation_reason.lower()
    assert "LONG" in result.validation_reason
    assert result.cost_usd == 0.0


def test_validator_rejects_low_score(engine, strong_signal):
    strong_signal.confluence_score = 30.0
    strong_signal.tier = "INVALID"
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is False
    assert "below" in result.validation_reason.lower()


def test_validator_rejects_weak_tier(engine, strong_signal):
    """WEAK-tier signals (40 <= score < 60) should be rejected."""
    strong_signal.confluence_score = 50.0
    strong_signal.tier = "WEAK"
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is False
    assert "below" in result.validation_reason.lower()


def test_validator_accepts_standard_tier(engine, strong_signal):
    """STANDARD-tier signals (score >= 60) should be accepted."""
    strong_signal.confluence_score = 62.0
    strong_signal.tier = "STANDARD"
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is True


def test_validator_rejects_low_rr(engine, strong_signal):
    strong_signal.rr_ratio = 1.0
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is False
    assert "risk/reward" in result.validation_reason.lower()


def test_validator_rejects_no_components(engine, strong_signal):
    strong_signal.components = []
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is False
    assert "component" in result.validation_reason.lower()


def test_validator_rejects_weak_components(engine, strong_signal):
    # All components at 50% of weight — no dominant confluence
    strong_signal.components = [
        MockComponent("BOS", 7.5, 15.0, "weak"),
        MockComponent("FVG", 7.5, 15.0, "weak"),
        MockComponent("Regime", 12.5, 25.0, "neutral"),
    ]
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    assert result.is_valid is False


def test_validator_does_not_second_guess_detector(engine, strong_signal):
    """Regime veto is ConfluenceDetector's job — narrator doesn't re-check it."""
    strong_signal.components = [
        MockComponent("BOS", 14.0, 15.0, "Strong BOS"),
        MockComponent("FVG", 13.0, 15.0, "Strong FVG"),
        MockComponent("Regime", 3.0, 25.0, "Weak regime alignment"),  # 12% of weight
    ]
    result = engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    # Dominant structure passes — detector already decided regime wasn't fatal.
    assert result.is_valid is True


# =============================================================================
# NARRATOR TIER
# =============================================================================

def test_narrator_returns_three_paragraphs(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert result.tier == NarrativeTier.NARRATOR
    assert result.is_valid is True
    paragraphs = result.full_narrative.split("\n\n")
    assert len(paragraphs) == 3
    assert "Market Setup" in paragraphs[0]
    assert "Key Confluences" in paragraphs[1]
    assert "Risk Considerations" in paragraphs[2]


def test_narrator_includes_key_fields(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    text = result.full_narrative
    assert "XAUUSD" in text
    assert "LONG" in text or "long" in text
    assert "82" in text  # score
    assert "2400.00" in text  # entry
    assert "2380.00" in text  # stop
    assert "2440.00" in text  # target
    assert "2.00" in text  # R:R


def test_narrator_reflects_dominant_components(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    confluences = result.key_confluences.lower()
    # Strongest component (BOS at 93%) must appear in the top-3 list
    assert "break of structure" in confluences
    # Narrative should indicate multiple dominant factors and cite weight ratios
    assert "dominant" in confluences
    assert "%" in confluences


def test_narrator_rejects_invalid_signal(engine, strong_signal):
    strong_signal.rr_ratio = 0.5
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert result.is_valid is False
    assert result.full_narrative == ""


def test_narrator_short_direction(engine, strong_signal):
    strong_signal.signal_type = "SHORT"
    strong_signal.entry_price = 2400.00
    strong_signal.stop_loss = 2420.00
    strong_signal.take_profit = 2360.00
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert result.is_valid is True
    assert "SHORT" in result.full_narrative or "short" in result.full_narrative
    # Invalidation direction: close ABOVE stop for shorts
    assert "above" in result.risk_warnings.lower()


def test_narrator_long_invalidation_direction(engine, strong_signal):
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert "below" in result.risk_warnings.lower()


# =============================================================================
# VOLATILITY REGIME VARIATIONS
# =============================================================================

def test_high_vol_regime_guidance(engine, strong_signal):
    strong_signal.vol_regime = "high"
    strong_signal.vol_forecast_atr = 15.0
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    risk = result.risk_warnings.lower()
    assert "elevated" in risk or "wider" in risk or "position size" in risk


def test_low_vol_regime_guidance(engine, strong_signal):
    strong_signal.vol_regime = "low"
    strong_signal.vol_forecast_atr = 7.5
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    setup = result.full_narrative.lower()
    assert "compressed" in setup or "tight" in setup


def test_vol_expansion_detection(engine, strong_signal):
    strong_signal.atr = 10.0
    strong_signal.vol_forecast_atr = 13.0  # +30%
    strong_signal.vol_regime = "high"
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    setup = result.full_narrative.lower()
    assert "expansion" in setup or "expanding" in setup or "+30%" in result.full_narrative


def test_vol_compression_detection(engine, strong_signal):
    strong_signal.atr = 10.0
    strong_signal.vol_forecast_atr = 7.0  # -30%
    strong_signal.vol_regime = "low"
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert "compression" in result.full_narrative.lower()


def test_missing_vol_data_handled(engine, strong_signal):
    strong_signal.vol_regime = None
    strong_signal.vol_forecast_atr = None
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert result.is_valid is True
    assert result.full_narrative != ""


# =============================================================================
# PRICE FORMATTING (multi-instrument)
# =============================================================================

def test_jpy_uses_three_decimals(engine, strong_signal):
    strong_signal.symbol = "USDJPY"
    strong_signal.entry_price = 150.123
    strong_signal.stop_loss = 149.823
    strong_signal.take_profit = 150.723
    strong_signal.atr = 0.200
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert "150.123" in result.full_narrative
    assert "149.823" in result.full_narrative


def test_fx_uses_five_decimals(engine, strong_signal):
    strong_signal.symbol = "EURUSD"
    strong_signal.entry_price = 1.08500
    strong_signal.stop_loss = 1.08200
    strong_signal.take_profit = 1.09100
    strong_signal.atr = 0.00150
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert "1.08500" in result.full_narrative


# =============================================================================
# INTERFACE PARITY
# =============================================================================

def test_to_dict_shape(engine, strong_signal):
    """Narrative dict must match what telegram_notifier and signal_store expect."""
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    d = result.to_dict()
    for key in (
        "tier", "is_valid", "validation_reason", "full_narrative",
        "key_confluences", "risk_warnings", "cost_usd", "model_used", "latency_ms",
    ):
        assert key in d


def test_get_stats(engine, strong_signal):
    engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    engine.generate_narrative(strong_signal, NarrativeTier.VALIDATOR)
    stats = engine.get_stats()
    assert stats["total_calls"] == 2
    assert stats["total_cost_usd"] == 0.0
    assert stats["avg_cost_per_call"] == 0.0


def test_cost_is_zero(engine, strong_signal):
    for tier in (NarrativeTier.VISUAL, NarrativeTier.VALIDATOR, NarrativeTier.NARRATOR):
        result = engine.generate_narrative(strong_signal, tier)
        assert result.cost_usd == 0.0


def test_latency_is_subsecond(engine, strong_signal):
    """Algorithmic engine should be far under LLM latency."""
    result = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert result.latency_ms < 100  # generous upper bound


def test_determinism(engine, strong_signal):
    r1 = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    r2 = engine.generate_narrative(strong_signal, NarrativeTier.NARRATOR)
    assert r1.full_narrative == r2.full_narrative


# =============================================================================
# THRESHOLD CONSTANTS
# =============================================================================

def test_thresholds_are_reasonable():
    assert MIN_SCORE_VALID == STANDARD_TIER_MIN
    assert MIN_RR_VALID >= 1.0
