"""Tests for HaikuDescriptionEngine + HaikuDescriptionCacheStore.

Narrated-reading upgrade: the engine now generates a fact-anchored narration
(tags + regime + structure + price), validated against the engine FACTS
(forbidden tokens AND known levels), retried once, then deterministic template.
"""

from pathlib import Path

import pytest

from src.intelligence.haiku_description_engine import (
    DEFAULT_MAX_TOKENS,
    HaikuDescriptionEngine,
    SYSTEM_PROMPT,
)
from src.intelligence.market_reading_schema import (
    MarketReadingRegime,
    MarketReadingStructure,
    OrderBlock,
)
from src.intelligence.narrated_reading import (
    NARRATION_MAX_LENGTH,
    build_reading_facts,
)
from src.storage.haiku_description_cache_store import HaikuDescriptionCacheStore

PRICE = 2000.0
INSTRUMENT = "XAUUSD"


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
def structure() -> MarketReadingStructure:
    return MarketReadingStructure()  # empty — no zones / breaks / retest


@pytest.fixture
def cache_store(tmp_path: Path) -> HaikuDescriptionCacheStore:
    db = tmp_path / "haiku_cache.db"
    return HaikuDescriptionCacheStore(db_path=str(db))


def _gen(engine, tags, regime, structure):
    return engine.generate(tags, regime, structure, PRICE, INSTRUMENT)


# ---------------------------------------------------------------------------
# Cache hit / miss
# ---------------------------------------------------------------------------


def test_cache_hit_does_not_call_llm(cache_store, regime, structure):
    client = _MockAnthropicClient(["Tendance haussière, volatilité élevée."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    tags = ["volatility_elevated", "trend_bullish"]

    desc1, src1 = _gen(engine, tags, regime, structure)
    assert src1 == "haiku_generated"
    assert desc1 == "Tendance haussière, volatilité élevée."
    assert len(client.messages.calls) == 1

    # Same facts → cache hit → no new LLM call.
    desc2, src2 = _gen(engine, tags, regime, structure)
    assert desc2 == desc1
    assert src2 == "haiku_generated"
    assert len(client.messages.calls) == 1


def test_cache_key_independent_of_tag_order(cache_store, regime, structure):
    client = _MockAnthropicClient(["Description from LLM."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc1, _ = _gen(engine, ["a", "b", "c"], regime, structure)
    desc2, _ = _gen(engine, ["c", "a", "b"], regime, structure)
    assert desc1 == desc2
    assert len(client.messages.calls) == 1  # second call hit cache


def test_cache_key_excludes_raw_price(cache_store, regime, structure):
    """A quiet tick (price moves, structure unchanged) must NOT bust the cache."""
    client = _MockAnthropicClient(["Tendance haussière observée."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    engine.generate(["t"], regime, structure, 2000.0, INSTRUMENT)
    engine.generate(["t"], regime, structure, 2001.4, INSTRUMENT)  # price moved
    assert len(client.messages.calls) == 1  # structural fingerprint unchanged


# ---------------------------------------------------------------------------
# LLM call shape
# ---------------------------------------------------------------------------


def test_llm_called_with_correct_model_and_short_prompt(cache_store, regime, structure):
    client = _MockAnthropicClient(["Clean output."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    _gen(engine, ["t1"], regime, structure)

    call = client.messages.calls[0]
    assert call["model"].startswith("claude-haiku")
    assert call["max_tokens"] == DEFAULT_MAX_TOKENS
    assert call["max_tokens"] <= 400  # Lever 3 budget (paragraph)
    assert call["system"] == SYSTEM_PROMPT
    assert len(call["messages"]) == 1
    assert call["messages"][0]["role"] == "user"
    assert len(SYSTEM_PROMPT) < 1800  # short system prompt (~250 tokens FR)


# ---------------------------------------------------------------------------
# Forbidden-token contamination → fallback template, no cache write
# ---------------------------------------------------------------------------


def test_contaminated_haiku_output_falls_back_no_cache(cache_store, regime, structure):
    # Both attempts contaminated (forbidden token) → template fallback, nothing cached.
    client = _MockAnthropicClient(
        ["Conseille d'acheter maintenant.", "Je conseille de sortir vite."]
    )
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc, src = _gen(engine, ["t1"], regime, structure)

    assert src == "template_fallback"
    assert "conseille" not in desc.lower()
    assert cache_store.size() == 0


def test_retry_once_then_succeeds(cache_store, regime, structure):
    """First attempt contaminated, second clean → haiku_generated (one retry)."""
    client = _MockAnthropicClient(
        ["Conseille d'acheter.", "Tendance haussière, volatilité élevée."]
    )
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc, src = _gen(engine, ["t1"], regime, structure)
    assert src == "haiku_generated"
    assert desc == "Tendance haussière, volatilité élevée."
    assert len(client.messages.calls) == 2  # one retry


# ---------------------------------------------------------------------------
# Level-anchoring: a narration citing a foreign level is rejected
# ---------------------------------------------------------------------------


def test_unanchored_level_rejected_then_fallback(cache_store, regime):
    """A narration citing a price absent from the facts is rejected (both tries)."""
    structure = MarketReadingStructure(
        order_blocks=[
            OrderBlock(
                id="ob1",
                direction="bearish",
                level_high=2010.0,
                level_low=2005.0,
                importance="medium",
                status="active",
                created_at="2026-06-20T00:00:00Z",
                tested=False,
            )
        ]
    )
    # 2222.22 is NOT a level the engine produced → must be rejected.
    client = _MockAnthropicClient(
        ["Le prix teste 2222.22.", "Order Block actif à 9999.99."]
    )
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc, src = engine.generate(["t1"], regime, structure, PRICE, INSTRUMENT)
    assert src == "template_fallback"
    assert "2222.22" not in desc
    assert "9999.99" not in desc
    assert cache_store.size() == 0


def test_anchored_level_accepted(cache_store, regime):
    """A narration reusing a real engine level passes anchoring."""
    structure = MarketReadingStructure(
        order_blocks=[
            OrderBlock(
                id="ob1",
                direction="bearish",
                level_high=2010.0,
                level_low=2005.0,
                importance="medium",
                status="active",
                created_at="2026-06-20T00:00:00Z",
                tested=False,
            )
        ]
    )
    facts = build_reading_facts(structure, regime, PRICE, INSTRUMENT)
    assert "2005.00" in {*[z.low for z in facts.zones]}  # sanity: real level
    client = _MockAnthropicClient(["Un Order Block actif borne 2005.00–2010.00."])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)

    desc, src = engine.generate(["t1"], regime, structure, PRICE, INSTRUMENT)
    assert src == "haiku_generated"
    assert "2005.00" in desc


# ---------------------------------------------------------------------------
# No client injected → deterministic template
# ---------------------------------------------------------------------------


def test_no_client_returns_template_fallback(cache_store, regime, structure):
    engine = HaikuDescriptionEngine(anthropic_client=None, cache_store=cache_store)
    desc, src = _gen(engine, ["volatility_elevated"], regime, structure)
    assert src == "template_fallback"
    assert "tendance" in desc.lower()
    assert cache_store.size() == 0


# ---------------------------------------------------------------------------
# Truncation to the narration max length
# ---------------------------------------------------------------------------


def test_haiku_output_truncated_to_schema_max_length(cache_store, regime, structure):
    long_output = "Tendance haussière " * 60  # ~1140 chars, no numbers
    client = _MockAnthropicClient([long_output])
    engine = HaikuDescriptionEngine(anthropic_client=client, cache_store=cache_store)
    desc, _ = _gen(engine, ["t"], regime, structure)
    assert len(desc) <= NARRATION_MAX_LENGTH


# ---------------------------------------------------------------------------
# Cache store basics
# ---------------------------------------------------------------------------


def test_cache_store_get_returns_none_on_miss(cache_store):
    assert cache_store.get("nonexistent_hash") is None


def test_cache_store_put_then_get_roundtrip(cache_store):
    cache_store.put("k1", "Description un.", "haiku_generated")
    assert cache_store.get("k1") == ("Description un.", "haiku_generated")
    cache_store.put("k1", "Description deux.", "haiku_generated")
    assert cache_store.get("k1") == ("Description deux.", "haiku_generated")


def test_cache_store_uses_env_var_for_path(tmp_path: Path, monkeypatch):
    custom = tmp_path / "via_env.db"
    monkeypatch.setenv("MARKET_READINGS_DB_PATH", str(custom))
    store = HaikuDescriptionCacheStore()
    store.put("k", "d", "haiku_generated")
    assert custom.exists()


def test_cache_hash_changes_when_regime_changes(structure):
    regime_a = MarketReadingRegime(
        trend="bullish", volatility_observed="normal", market_phase="trend",
        mtf_confluence={"h1": "bullish"},
    )
    regime_b = MarketReadingRegime(
        trend="bearish", volatility_observed="normal", market_phase="trend",
        mtf_confluence={"h1": "bullish"},
    )
    facts_a = build_reading_facts(structure, regime_a, PRICE, INSTRUMENT)
    facts_b = build_reading_facts(structure, regime_b, PRICE, INSTRUMENT)
    h_a = HaikuDescriptionEngine._compute_hash(["t1"], facts_a)
    h_b = HaikuDescriptionEngine._compute_hash(["t1"], facts_b)
    assert h_a != h_b


# ---------------------------------------------------------------------------
# LLM exception → fallback (not propagated)
# ---------------------------------------------------------------------------


def test_llm_exception_returns_fallback(cache_store, regime, structure):
    class _BoomMessages:
        def create(self, **kwargs):
            raise ConnectionError("network down")

    class _BoomClient:
        messages = _BoomMessages()

    engine = HaikuDescriptionEngine(anthropic_client=_BoomClient(), cache_store=cache_store)
    desc, src = _gen(engine, ["t1"], regime, structure)
    assert src == "template_fallback"
    assert len(desc) > 0
