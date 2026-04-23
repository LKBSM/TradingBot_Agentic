"""Tests for LLMNarrativeEngine — Sprint 2 of Smart Sentinel AI."""

import pytest
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, patch

from src.intelligence.llm_narrative_engine import (
    LLMNarrativeEngine,
    NarrativeTier,
    SignalNarrative,
    SMC_SYSTEM_PROMPT,
    COST_PER_1M,
)


# ============================================================================
# MOCK SIGNAL
# ============================================================================

@dataclass
class MockComponentScore:
    name: str
    weighted_score: float
    weight: float
    reasoning: str


@dataclass
class MockSignal:
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    confluence_score: float = 82.5
    tier: str = "PREMIUM"
    entry_price: float = 2400.0
    stop_loss: float = 2380.0
    take_profit: float = 2440.0
    rr_ratio: float = 2.0
    atr: float = 10.0
    components: List[MockComponentScore] = field(default_factory=lambda: [
        MockComponentScore("BOS", 15.0, 15.0, "Bullish BOS"),
        MockComponentScore("FVG", 15.0, 15.0, "Bullish FVG"),
        MockComponentScore("Regime", 20.0, 25.0, "Uptrend"),
    ])


# ============================================================================
# MOCK ANTHROPIC RESPONSE
# ============================================================================

def make_mock_response(text: str, input_tokens: int = 500, output_tokens: int = 100,
                       cache_read: int = 0):
    """Create a mock Anthropic API response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    response.usage = usage

    return response


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def signal():
    return MockSignal()


@pytest.fixture
def engine_no_key():
    """Engine without API key — visual-only mode."""
    return LLMNarrativeEngine(api_key=None)


@pytest.fixture
def engine_with_mock():
    """Engine with mocked Anthropic client."""
    engine = LLMNarrativeEngine(api_key=None)
    engine._client = MagicMock()
    return engine


# ============================================================================
# TESTS: VISUAL TIER (Layer 1)
# ============================================================================

class TestVisualTier:
    def test_visual_returns_zero_cost(self, engine_no_key, signal):
        result = engine_no_key.generate_narrative(signal, NarrativeTier.VISUAL)
        assert result.cost_usd == 0.0
        assert result.tier == NarrativeTier.VISUAL
        assert result.is_valid is True

    def test_visual_no_api_call(self, engine_no_key, signal):
        result = engine_no_key.generate_narrative(signal, NarrativeTier.VISUAL)
        assert result.model_used == ""
        assert result.full_narrative == ""

    def test_visual_fallback_when_no_client(self, engine_no_key, signal):
        """When requesting VALIDATOR but no API key, falls back to visual."""
        result = engine_no_key.generate_narrative(signal, NarrativeTier.VALIDATOR)
        assert result.tier == NarrativeTier.VISUAL
        assert result.cost_usd == 0.0


# ============================================================================
# TESTS: VALIDATOR TIER (Layer 2 — Haiku)
# ============================================================================

class TestValidatorTier:
    def test_haiku_valid_signal(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|Strong bullish confluence with BOS+FVG alignment"
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        assert result.tier == NarrativeTier.VALIDATOR
        assert result.is_valid is True
        assert "bullish" in result.validation_reason.lower()
        assert result.cost_usd > 0

    def test_haiku_invalid_signal(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "INVALID|Ranging market contradicts directional trade"
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        assert result.tier == NarrativeTier.VALIDATOR
        assert result.is_valid is False
        assert "ranging" in result.validation_reason.lower()

    def test_haiku_only_called_once(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK"
        )
        engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)
        assert engine_with_mock._client.messages.create.call_count == 1

    def test_haiku_cost_calculation(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK", input_tokens=600, output_tokens=20, cache_read=400
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        # Cost: (600-400)*0.80/1M + 400*0.08/1M + 20*4.00/1M
        expected = (200 * 0.80 + 400 * 0.08 + 20 * 4.00) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, abs=1e-8)

    def test_haiku_cache_hit_detected(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK", cache_read=500
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)
        assert result.cache_hit is True


# ============================================================================
# TESTS: NARRATOR TIER (Layer 3 — Haiku + Sonnet cascade)
# ============================================================================

class TestNarratorTier:
    def test_full_cascade_valid(self, engine_with_mock, signal):
        """Haiku validates, then Sonnet narrates."""
        engine_with_mock._client.messages.create.side_effect = [
            make_mock_response("VALID|Strong setup"),
            make_mock_response(
                "Gold is testing key resistance at 2400.\n\n"
                "BOS and FVG align bullishly with regime confirmation.\n\n"
                "Risk: SL at 2380 (2×ATR), TP at 2440. R:R = 2:1."
            ),
        ]
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.NARRATOR)

        assert result.tier == NarrativeTier.NARRATOR
        assert result.is_valid is True
        assert len(result.full_narrative) > 0
        assert len(result.key_confluences) > 0
        assert len(result.risk_warnings) > 0
        assert result.cost_usd > 0

    def test_haiku_rejects_no_sonnet_call(self, engine_with_mock, signal):
        """If Haiku says INVALID, Sonnet is never called → cost savings."""
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "INVALID|Counter-trend, low conviction"
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.NARRATOR)

        assert result.is_valid is False
        assert engine_with_mock._client.messages.create.call_count == 1  # Only Haiku

    def test_cascade_combines_costs(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.side_effect = [
            make_mock_response("VALID|OK", input_tokens=500, output_tokens=20),
            make_mock_response("Narrative text.", input_tokens=800, output_tokens=200),
        ]
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.NARRATOR)

        # Total cost = haiku cost + sonnet cost
        assert result.cost_usd > 0
        haiku_cost = (500 * 0.80 + 20 * 4.00) / 1_000_000
        sonnet_cost = (800 * 3.00 + 200 * 15.00) / 1_000_000
        assert result.cost_usd == pytest.approx(haiku_cost + sonnet_cost, abs=1e-7)


# ============================================================================
# TESTS: ERROR HANDLING
# ============================================================================

class TestErrorHandling:
    def test_api_error_fails_open(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.side_effect = Exception("API down")
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        assert result.is_valid is True  # Fail-open
        assert "error" in result.validation_reason.lower()
        assert result.cost_usd == 0.0


# ============================================================================
# TESTS: CSV SERIALIZATION
# ============================================================================

class TestCSVSerialization:
    def test_signal_to_csv_compact(self, signal):
        csv = LLMNarrativeEngine._signal_to_csv(signal)
        assert "sym=XAUUSD" in csv
        assert "dir=LONG" in csv
        assert "score=82.5" in csv
        assert "entry=2400.00" in csv
        assert "BOS=15.0/15" in csv

    def test_csv_shorter_than_json(self, signal):
        import json
        csv = LLMNarrativeEngine._signal_to_csv(signal)
        json_str = json.dumps({
            "symbol": signal.symbol,
            "signal_type": signal.signal_type,
            "confluence_score": signal.confluence_score,
            "entry_price": signal.entry_price,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        })
        assert len(csv) < len(json_str) * 2  # CSV should be reasonably compact


# ============================================================================
# TESTS: SYSTEM PROMPT
# ============================================================================

class TestSystemPrompt:
    def test_system_prompt_contains_smc_rules(self):
        assert "BOS" in SMC_SYSTEM_PROMPT
        assert "FVG" in SMC_SYSTEM_PROMPT
        assert "Order Block" in SMC_SYSTEM_PROMPT

    def test_system_prompt_contains_risk_rules(self):
        assert "2×ATR" in SMC_SYSTEM_PROMPT
        assert "4×ATR" in SMC_SYSTEM_PROMPT
        assert "Kelly" in SMC_SYSTEM_PROMPT


# ============================================================================
# TESTS: STATS TRACKING
# ============================================================================

class TestStatsTracking:
    def test_stats_after_calls(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response("VALID|OK")
        engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)
        engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        stats = engine_with_mock.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_cost_usd"] > 0

    def test_initial_stats_zero(self, engine_no_key):
        stats = engine_no_key.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_cost_usd"] == 0
