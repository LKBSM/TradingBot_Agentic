"""Tests for the LLM-2B.8 cost-optimization caches.

Covers:
- TTL+LRU eviction semantics on the generic cache
- QueryEmbeddingCache key normalisation (case + whitespace)
- AnswerCache invalidation on corpus fingerprint change
- RAGPipeline embedding cache integration (hit count grows with repeated queries)
- RAGPipeline answer cache short-circuit (LLM is NOT called on a hit)
- CostTracker pricing math + per-tier aggregation
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.intelligence.rag import (
    AnswerCache,
    CachedAnswer,
    CostTracker,
    HashEmbedder,
    QueryEmbeddingCache,
    RAGPipeline,
)
from src.intelligence.rag.cache import _TTLLRUCache, _normalise_text
from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.cost_tracker import (
    EMBEDDING_PRICING_PER_1M_TOKENS,
    MODEL_PRICING,
)


# ---------------------------------------------------------------------------
# Generic _TTLLRUCache
# ---------------------------------------------------------------------------


def test_ttllru_get_returns_value_within_ttl():
    c: _TTLLRUCache[int] = _TTLLRUCache(max_size=2, ttl_seconds=10.0)
    c.put("a", 1)
    assert c.get("a") == 1


def test_ttllru_evicts_after_ttl():
    # Windows time.sleep granularity is ~15ms; pick a TTL/sleep pair that
    # clears it comfortably so the test doesn't go flaky in CI.
    c: _TTLLRUCache[int] = _TTLLRUCache(max_size=4, ttl_seconds=0.05)
    c.put("a", 1)
    time.sleep(0.15)
    assert c.get("a") is None
    assert c.stats().evictions == 1


def test_ttllru_lru_eviction_when_over_capacity():
    c: _TTLLRUCache[int] = _TTLLRUCache(max_size=2, ttl_seconds=60.0)
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)  # 'a' evicted (LRU)
    assert c.get("a") is None
    assert c.get("b") == 2
    assert c.get("c") == 3


def test_ttllru_get_promotes_to_mru():
    c: _TTLLRUCache[int] = _TTLLRUCache(max_size=2, ttl_seconds=60.0)
    c.put("a", 1)
    c.put("b", 2)
    c.get("a")  # promotes a
    c.put("c", 3)  # 'b' evicted, not 'a'
    assert c.get("a") == 1
    assert c.get("b") is None


def test_ttllru_invalid_args_raise():
    with pytest.raises(ValueError):
        _TTLLRUCache(max_size=0, ttl_seconds=1)
    with pytest.raises(ValueError):
        _TTLLRUCache(max_size=10, ttl_seconds=0)


def test_stats_track_hit_ratio():
    c: _TTLLRUCache[int] = _TTLLRUCache(max_size=4, ttl_seconds=60.0)
    c.put("x", 1)
    c.get("x")  # hit
    c.get("y")  # miss
    s = c.stats()
    assert s.hits == 1
    assert s.misses == 1
    assert s.hit_ratio == 0.5
    assert s.size == 1
    assert s.capacity == 4


# ---------------------------------------------------------------------------
# QueryEmbeddingCache
# ---------------------------------------------------------------------------


def test_normalise_text_collapses_whitespace_and_case():
    assert _normalise_text("  Hello  WORLD ") == "hello world"
    assert _normalise_text("Hello World") == _normalise_text("hello\tworld")


def test_query_embedding_cache_round_trip():
    cache = QueryEmbeddingCache(max_size=4, ttl_seconds=60.0)
    vec = np.array([0.1, 0.9], dtype=np.float32)
    cache.put("Hello world", vec)
    got = cache.get("hello   world")  # different case + spacing → same slot
    assert got is not None
    np.testing.assert_array_equal(got, vec)


def test_query_embedding_cache_signature_invalidation():
    """Switching the embedder signature should bypass cached entries."""
    cache_a = QueryEmbeddingCache(embedder_signature="hash-256-seed1")
    cache_b = QueryEmbeddingCache(embedder_signature="voyage-3-large")
    cache_a.put("q", np.array([1.0, 0.0], dtype=np.float32))
    # Same query but different signature ⇒ no hit
    assert cache_b.get("q") is None


# ---------------------------------------------------------------------------
# AnswerCache
# ---------------------------------------------------------------------------


def test_answer_cache_round_trip():
    c = AnswerCache(max_size=4, ttl_seconds=60.0)
    val = CachedAnswer(answer="hello", retrieved_chunk_ids=["c1"])
    c.put("query A", "en", 5, "size=10", val)
    got = c.get("query A", "en", 5, "size=10")
    assert got is val


def test_answer_cache_invalidates_on_corpus_change():
    c = AnswerCache()
    val = CachedAnswer(answer="hello", retrieved_chunk_ids=[])
    c.put("query", "en", 5, "size=10", val)
    # Corpus grew ⇒ different fingerprint ⇒ no hit
    assert c.get("query", "en", 5, "size=11") is None


def test_answer_cache_separates_languages():
    c = AnswerCache()
    fr = CachedAnswer(answer="bonjour", retrieved_chunk_ids=[])
    en = CachedAnswer(answer="hello", retrieved_chunk_ids=[])
    c.put("greeting", "fr", 5, "fp", fr)
    c.put("greeting", "en", 5, "fp", en)
    assert c.get("greeting", "fr", 5, "fp").answer == "bonjour"
    assert c.get("greeting", "en", 5, "fp").answer == "hello"


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def _make_chunks() -> list[Chunk]:
    return [
        Chunk(text="HAR-RV decomposes vol", source_id="paper_har", chunk_index=0,
              metadata={"type": "paper"}),
        Chunk(text="CFTC COT report Friday", source_id="data_cot", chunk_index=0,
              metadata={"type": "data"}),
        Chunk(text="VIX is the fear gauge", source_id="edu_vix", chunk_index=0,
              metadata={"type": "education"}),
    ]


def test_pipeline_uses_embedding_cache_on_repeated_queries():
    cache = QueryEmbeddingCache(embedder_signature="hash-128")
    pipe = RAGPipeline(
        embedder=HashEmbedder(dimension=128, seed=7),
        embedding_cache=cache,
    )
    pipe.ingest(_make_chunks())

    pipe.retrieve("HAR-RV vol")
    pipe.retrieve("har-rv VOL")  # same after normalisation
    pipe.retrieve("HAR-RV vol")
    stats = cache.stats()
    # 2 hits (calls 2 and 3) and 1 miss (call 1).
    assert stats.hits == 2
    assert stats.misses == 1


def test_pipeline_embedding_cache_optional():
    """Without a cache, retrieve still works."""
    pipe = RAGPipeline(embedder=HashEmbedder(dimension=64, seed=1))
    pipe.ingest(_make_chunks())
    out = pipe.retrieve("VIX")
    assert len(out) > 0


def test_pipeline_answer_cache_short_circuits_llm():
    """Identical (query, lang, top_k) on a corpus fingerprint should hit."""
    answer_cache = AnswerCache(max_size=8, ttl_seconds=60.0)
    pipe = RAGPipeline(
        embedder=HashEmbedder(dimension=128, seed=2),
        answer_cache=answer_cache,
    )
    pipe.ingest(_make_chunks())

    llm_call_count = {"n": 0}

    def stub_llm(system: str, user: str) -> str:
        llm_call_count["n"] += 1
        return f"answer #{llm_call_count['n']}"

    r1 = pipe.query("HAR-RV vol", llm=stub_llm)
    r2 = pipe.query("HAR-RV vol", llm=stub_llm)
    r3 = pipe.query("har-rv  VOL", llm=stub_llm)  # normalises to same key

    # LLM was called exactly once; r2/r3 are cache replays.
    assert llm_call_count["n"] == 1
    assert r1.answer == r2.answer == r3.answer
    assert "cache_hit" in r2.elapsed_seconds
    assert "cache_hit" in r3.elapsed_seconds
    # Stats sanity
    s = answer_cache.stats()
    assert s.hits == 2
    assert s.misses == 1


def test_pipeline_answer_cache_invalidates_on_ingest():
    """Re-ingesting changes the corpus fingerprint ⇒ cache cold again."""
    pipe = RAGPipeline(
        embedder=HashEmbedder(dimension=128, seed=3),
        answer_cache=AnswerCache(),
    )
    pipe.ingest(_make_chunks())
    n_calls = {"n": 0}

    def llm(system, user):
        n_calls["n"] += 1
        return "x"

    pipe.query("HAR-RV vol", llm=llm)
    assert n_calls["n"] == 1
    pipe.query("HAR-RV vol", llm=llm)
    assert n_calls["n"] == 1  # cache hit
    # Add a new chunk → fingerprint changes → cache miss again.
    pipe.ingest(
        [Chunk(text="EUR/USD level 1.1", source_id="extra", chunk_index=0)]
    )
    pipe.query("HAR-RV vol", llm=llm)
    assert n_calls["n"] == 2


def test_pipeline_does_not_cache_when_no_llm():
    """Stub-only path (no LLM provided) should not poison the cache with empty answers."""
    cache = AnswerCache()
    pipe = RAGPipeline(
        embedder=HashEmbedder(dimension=64, seed=5), answer_cache=cache
    )
    pipe.ingest(_make_chunks())
    pipe.query("HAR-RV vol", llm=None)
    pipe.query("HAR-RV vol", llm=None)
    assert cache.stats().size == 0  # never wrote anything


# ---------------------------------------------------------------------------
# CostTracker
# ---------------------------------------------------------------------------


def test_cost_tracker_haiku_pricing():
    tr = CostTracker()
    tr.record_llm("claude-haiku-4-5", input_tokens=1_000_000, output_tokens=200_000, tier="STRATEGIST")
    s = tr.summary()
    # Haiku: $1.00/1M input, $5.00/1M output
    expected = 1.0 + (200_000 / 1_000_000) * 5.0
    assert abs(s.llm_usd - expected) < 1e-6
    assert s.by_tier_usd["STRATEGIST"] == pytest.approx(expected)
    assert s.n_llm_calls == 1


def test_cost_tracker_voyage_pricing():
    tr = CostTracker()
    tr.record_embedding("voyage-3-large", n_tokens=10_000_000, tier="INSTITUTIONAL")
    s = tr.summary()
    # voyage-3-large = $0.18 / 1M
    assert s.embedding_usd == pytest.approx(10 * 0.18, abs=1e-6)
    assert s.n_embedding_calls == 1


def test_cost_tracker_unknown_model_zero_cost():
    """Don't fabricate prices for unrecognised models — return 0 and let callers alert."""
    tr = CostTracker()
    tr.record_llm("future-model-xyz", input_tokens=1000, output_tokens=1000)
    assert tr.summary().llm_usd == 0.0


def test_cost_tracker_aggregates_per_tier_and_model():
    tr = CostTracker()
    tr.record_llm("claude-haiku-4-5", 1_000_000, 0, tier="ANALYST")
    tr.record_llm("claude-sonnet-4-6", 1_000_000, 0, tier="STRATEGIST")
    tr.record_embedding("voyage-3-large", 1_000_000, tier="STRATEGIST")
    s = tr.summary()
    assert s.by_tier_usd["ANALYST"] == pytest.approx(1.00)
    assert s.by_tier_usd["STRATEGIST"] == pytest.approx(3.00 + 0.18, abs=1e-6)
    assert s.by_model_usd["claude-haiku-4-5"] == pytest.approx(1.00)
    assert s.by_model_usd["claude-sonnet-4-6"] == pytest.approx(3.00)
    assert s.total_usd == pytest.approx(1.00 + 3.00 + 0.18, abs=1e-6)


def test_cost_tracker_reset():
    tr = CostTracker()
    tr.record_llm("claude-haiku-4-5", 1000, 1000)
    tr.reset()
    assert tr.summary().total_usd == 0.0
    assert tr.summary().n_llm_calls == 0


def test_pricing_table_includes_2026_q2_models():
    """Sanity: the pricing table must enumerate at least the 3 Claude tiers."""
    for m in ("claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"):
        assert m in MODEL_PRICING
        assert MODEL_PRICING[m]["input"] > 0
        assert MODEL_PRICING[m]["output"] > MODEL_PRICING[m]["input"]


def test_pricing_voyage_models_present():
    assert "voyage-3-large" in EMBEDDING_PRICING_PER_1M_TOKENS
    assert "voyage-3-lite" in EMBEDDING_PRICING_PER_1M_TOKENS
