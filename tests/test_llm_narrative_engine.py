"""Tests for LLMNarrativeEngine — 4-tier post-cascade refactor."""

import pytest
from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock

from src.intelligence.llm_narrative_engine import (
    LLMNarrativeEngine,
    NarrativeTier,
    SignalNarrative,
    SMC_SYSTEM_PROMPT,
    COST_PER_1M,
    DEFAULT_VALIDATOR_MODEL,
    DEFAULT_NARRATOR_MODEL,
    DEFAULT_INSTITUTIONAL_MODEL,
    TIER_MODEL_MAP,
    model_for_tier,
    _model_family,
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

def make_mock_response(
    text: str,
    input_tokens: int = 500,
    output_tokens: int = 100,
    cache_read: int = 0,
    cache_write: int = 0,
):
    """Create a mock Anthropic API response."""
    response = MagicMock()
    content_block = MagicMock()
    content_block.text = text
    response.content = [content_block]

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_read_input_tokens = cache_read
    usage.cache_creation_input_tokens = cache_write
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
        # input=600 with cache_read=400 → billed_input = 600 (cache_read is excluded
        # already). Cost: 600*0.80/1M + 400*0.08/1M + 20*4.00/1M
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK", input_tokens=600, output_tokens=20, cache_read=400
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        expected = (600 * 0.80 + 400 * 0.08 + 20 * 4.00) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, abs=1e-8)

    def test_haiku_cost_with_cache_write(self, engine_with_mock, signal):
        # First call populates cache: input=2500, cache_write=2200, output=20.
        # Anthropic reports input_tokens INCLUDING cache_write tokens, so we
        # subtract them to avoid double-billing.
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK", input_tokens=2500, output_tokens=20, cache_write=2200
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)

        billed_input = 2500 - 2200
        expected = (
            billed_input * 0.80 + 2200 * 1.00 + 20 * 4.00
        ) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, abs=1e-8)

    def test_haiku_cache_hit_detected(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "VALID|OK", cache_read=500
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.VALIDATOR)
        assert result.cache_hit is True


# ============================================================================
# TESTS: NARRATOR TIER (Layer 3 — Sonnet single call, no cascade)
# ============================================================================

class TestNarratorTier:
    def test_narrator_single_call(self, engine_with_mock, signal):
        """Sonnet narrates directly — no Haiku gate (algo gates upstream)."""
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "Gold is testing key resistance at 2400.\n\n"
            "BOS and FVG align bullishly with regime confirmation.\n\n"
            "Risk: SL at 2380 (2×ATR), TP at 2440. R:R = 2:1."
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.NARRATOR)

        assert result.tier == NarrativeTier.NARRATOR
        assert result.is_valid is True
        assert len(result.full_narrative) > 0
        assert len(result.key_confluences) > 0
        assert len(result.risk_warnings) > 0
        assert result.cost_usd > 0
        # Single call — no Haiku gate
        assert engine_with_mock._client.messages.create.call_count == 1
        assert result.model_used == DEFAULT_NARRATOR_MODEL

    def test_narrator_uses_sonnet_pricing(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "p1\n\np2\n\np3", input_tokens=800, output_tokens=200
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.NARRATOR)

        expected = (800 * 3.00 + 200 * 15.00) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, abs=1e-7)


# ============================================================================
# TESTS: INSTITUTIONAL TIER (Layer 4 — Opus single call)
# ============================================================================

class TestInstitutionalTier:
    def test_institutional_uses_opus(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "Setup\n\nConfluences\n\nVolatility\n\nRisk Frame\n\nInvalidation"
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.INSTITUTIONAL)

        assert result.tier == NarrativeTier.INSTITUTIONAL
        assert result.model_used == DEFAULT_INSTITUTIONAL_MODEL
        assert engine_with_mock._client.messages.create.call_count == 1

    def test_institutional_uses_opus_pricing(self, engine_with_mock, signal):
        engine_with_mock._client.messages.create.return_value = make_mock_response(
            "x", input_tokens=1000, output_tokens=500
        )
        result = engine_with_mock.generate_narrative(signal, NarrativeTier.INSTITUTIONAL)
        expected = (1000 * 15.00 + 500 * 75.00) / 1_000_000
        assert result.cost_usd == pytest.approx(expected, abs=1e-7)


# ============================================================================
# TESTS: TIER → MODEL ROUTING
# ============================================================================

class TestTierRouting:
    def test_free_returns_empty_model(self):
        assert model_for_tier("FREE") == ""

    def test_analyst_routes_to_haiku(self):
        assert model_for_tier("ANALYST") == DEFAULT_VALIDATOR_MODEL
        assert "haiku" in model_for_tier("ANALYST")

    def test_strategist_routes_to_sonnet(self):
        assert model_for_tier("STRATEGIST") == DEFAULT_NARRATOR_MODEL
        assert "sonnet" in model_for_tier("STRATEGIST")

    def test_institutional_routes_to_opus(self):
        assert model_for_tier("INSTITUTIONAL") == DEFAULT_INSTITUTIONAL_MODEL
        assert "opus" in model_for_tier("INSTITUTIONAL")

    def test_unknown_tier_falls_back_to_sonnet(self):
        assert model_for_tier("UNKNOWN_TIER") == DEFAULT_NARRATOR_MODEL

    def test_narrative_tier_enum_routes(self):
        assert model_for_tier(NarrativeTier.VALIDATOR) == DEFAULT_VALIDATOR_MODEL
        assert model_for_tier(NarrativeTier.NARRATOR) == DEFAULT_NARRATOR_MODEL
        assert model_for_tier(NarrativeTier.INSTITUTIONAL) == DEFAULT_INSTITUTIONAL_MODEL


class TestModelFamily:
    def test_haiku_family(self):
        assert _model_family("claude-haiku-4-5-20251001") == "haiku"

    def test_sonnet_family(self):
        assert _model_family("claude-sonnet-4-6") == "sonnet"

    def test_opus_family(self):
        assert _model_family("claude-opus-4-7") == "opus"


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

    def test_system_prompt_contains_anti_hallucination(self):
        assert "Anti-Hallucination" in SMC_SYSTEM_PROMPT
        assert "Never invent" in SMC_SYSTEM_PROMPT

    def test_system_prompt_contains_examples(self):
        assert "Example A" in SMC_SYSTEM_PROMPT
        assert "Example B" in SMC_SYSTEM_PROMPT
        assert "VALID|" in SMC_SYSTEM_PROMPT
        assert "INVALID|" in SMC_SYSTEM_PROMPT

    def test_system_prompt_long_enough_for_haiku_cache(self):
        """Must be ≥ 2048 tokens for Haiku cache to engage.

        Use a conservative chars/4 estimate to bound this without requiring
        tiktoken at test time.
        """
        est_tokens_lower = len(SMC_SYSTEM_PROMPT) / 4.5  # conservative
        assert est_tokens_lower >= 2048, (
            f"System prompt too short for Haiku cache: "
            f"~{est_tokens_lower:.0f} tokens, need ≥ 2048"
        )


# ============================================================================
# TESTS: COST TABLE
# ============================================================================

class TestCostTable:
    def test_all_three_families_present(self):
        for family in ("haiku", "sonnet", "opus"):
            assert f"{family}_input" in COST_PER_1M
            assert f"{family}_output" in COST_PER_1M
            assert f"{family}_cache_read" in COST_PER_1M
            assert f"{family}_cache_write" in COST_PER_1M

    def test_cache_read_is_cheaper_than_input(self):
        for family in ("haiku", "sonnet", "opus"):
            assert COST_PER_1M[f"{family}_cache_read"] < COST_PER_1M[f"{family}_input"]


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
