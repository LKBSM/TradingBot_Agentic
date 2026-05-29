"""Tests for HaikuDescriptionEngine + HaikuDescriptionCacheStore (Chantier 2 Étape 5)."""

import tempfile
from pathlib import Path

import pytest

from src.intelligence.haiku_description_engine import (
    DEFAULT_MAX_TOKENS,
    HaikuDescriptionEngine,
    SYSTEM_PROMPT,
)
from src.intelligence.market_reading_schema import MarketReadingRegime
from src.storage.haiku_description_cache_store import HaikuDescriptionCacheStore


# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockContentBlock:
    def __init__(self, text: str):
        self.text = text


class _MockResponse:
    def __init__(self, text: str):
        self.content = [_MockContentBlock(text)]


class _MockMessages:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise RuntimeError("No more mocked responses")
        return _MockResponse(self._responses.pop(0))


class _MockAnthropicClient:
    def __init__(self, responses: list[str]):
        self.messages = _MockMessages(responses)


@pytest.fixture
def regime() -> MarketReadingRegime:
    return MarketReadingRegime(
        trend="bullish",
        volatility_observed="elevated",
        market_phase="expansion",
        mtf_confluence={"h1": "bullish", "h4": "bullish"},
    )


@pytest.fixture
def cache_store(tmp_path: Path) -> HaikuDescriptionCacheStore:
    db = tmp_path / "haiku_cache.db"
    return HaikuDescriptionCacheStore(db_path=str(db))


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


def test_cache_hit_does_not_call_llm(cache_store, regime):
    client = _MockAnthropicClient(["Tendance haussière, volatilité élevée."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    tags = ["volatility_elevated", "trend_bullish"]

    # First call: cache miss → LLM
    desc1, src1 = engine.generate(tags, regime)
    assert src1 == "haiku_generated"
    assert desc1 == "Tendance haussière, volatilité élevée."
    assert len(client.messages.calls) == 1

    # Second call (same tags, same regime): cache hit → no LLM
    desc2, src2 = engine.generate(tags, regime)
    assert desc2 == desc1
    assert src2 == "haiku_generated"
    assert len(client.messages.calls) == 1  # unchanged


def test_cache_key_independent_of_tag_order(cache_store, regime):
    client = _MockAnthropicClient(["Description from LLM."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc1, _ = engine.generate(["a", "b", "c"], regime)
    desc2, _ = engine.generate(["c", "a", "b"], regime)
    assert desc1 == desc2
    assert len(client.messages.calls) == 1  # second call hit cache


# ---------------------------------------------------------------------------
# LLM call shape
# ---------------------------------------------------------------------------


def test_llm_called_with_correct_model_and_short_prompt(cache_store, regime):
    client = _MockAnthropicClient(["Clean output."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    engine.generate(["t1"], regime)

    call = client.messages.calls[0]
    assert call["model"].startswith("claude-haiku")
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["max_tokens"] <= 200  # Lever 3 budget
    assert call["system"] == SYSTEM_PROMPT
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    # System prompt itself is short
    assert len(SYSTEM_PROMPT) < 1500  # ~rough proxy for < 200 tokens FR


# ---------------------------------------------------------------------------
# CRITICAL: forbidden-token contamination → fallback template, no cache write
# ---------------------------------------------------------------------------


def test_contaminated_haiku_output_falls_back_no_cache(cache_store, regime):
    """If Haiku emits a forbidden token, fallback template is used AND nothing is cached."""
    client = _MockAnthropicClient(["Conseille d'acheter maintenant."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc, src = engine.generate(["t1"], regime)

    assert src == "template_fallback"
    # The returned description must NOT contain the contaminated tokens
    assert "conseille" not in desc.lower()
    assert "achète" not in desc.lower()
    assert "acheter" not in desc.lower()
    # Cache must be empty for this combination (contaminated output never persisted)
    assert cache_store.size() == 0

    # Subsequent call: still a cache miss (because we didn't cache).
    # If the client is exhausted, the engine should still produce a fallback.
    desc2, src2 = engine.generate(["t1"], regime)
    # The client raised RuntimeError("No more mocked responses") → engine falls back
    assert src2 == "template_fallback"


# ---------------------------------------------------------------------------
# No client injected → fallback template
# ---------------------------------------------------------------------------


def test_no_client_returns_template_fallback(cache_store, regime):
    engine = HaikuDescriptionEngine(anthropic_client=None, cache_store=cache_store)
    desc, src = engine.generate(["volatility_elevated"], regime)
    assert src == "template_fallback"
    assert "tendance" in desc.lower()  # template uses this vocabulary
    assert cache_store.size() == 0


# ---------------------------------------------------------------------------
# Truncation to 280 chars
# ---------------------------------------------------------------------------


def test_haiku_output_truncated_to_schema_max_length(cache_store, regime):
    long_output = "Tendance haussière " * 50  # ~950 chars
    client = _MockAnthropicClient([long_output])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    desc, _ = engine.generate(["t"], regime)
    assert len(desc) <= 280


# ---------------------------------------------------------------------------
# Cache store basics
# ---------------------------------------------------------------------------


def test_cache_store_get_returns_none_on_miss(cache_store):
    assert cache_store.get("nonexistent_hash") is None


def test_cache_store_put_then_get_roundtrip(cache_store):
    cache_store.put("k1", "Description un.", "haiku_generated")
    assert cache_store.get("k1") == ("Description un.", "haiku_generated")
    cache_store.put("k1", "Description deux.", "haiku_generated")  # replace
    assert cache_store.get("k1") == ("Description deux.", "haiku_generated")


def test_cache_store_uses_env_var_for_path(tmp_path: Path, monkeypatch):
    custom = tmp_path / "via_env.db"
    monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(custom))
    store = HaikuDescriptionCacheStore()  # no explicit path
    store.put("k", "d", "haiku_generated")
    assert custom.exists()


def test_cache_hash_changes_when_regime_changes(cache_store):
    regime_a = MarketReadingRegime(
        trend="bullish", volatility_observed="normal", market_phase="trend",
        mtf_confluence={"h1": "bullish"},
    )
    regime_b = MarketReadingRegime(
        trend="bearish", volatility_observed="normal", market_phase="trend",
        mtf_confluence={"h1": "bullish"},
    )
    h_a = HaikuDescriptionEngine._compute_hash(["t1"], regime_a)
    h_b = HaikuDescriptionEngine._compute_hash(["t1"], regime_b)
    assert h_a != h_b


# ---------------------------------------------------------------------------
# LLM exception → fallback (not propagated)
# ---------------------------------------------------------------------------


def test_llm_exception_returns_fallback(cache_store, regime):
    class _BoomMessages:
        def create(self, **kwargs):
            raise ConnectionError("network down")

    class _BoomClient:
        messages = _BoomMessages()

    engine = HaikuDescriptionEngine(anthropic_client=_BoomClient(), cache_store=cache_store)
    desc, src = engine.generate(["t1"], regime)
    assert src == "template_fallback"
    assert len(desc) > 0
