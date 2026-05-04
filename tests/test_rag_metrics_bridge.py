"""Tests for the OBS-2B.1 metrics bridge.

Verifies that the bridge correctly mirrors RAG cache + cost tracker
state into the central MetricsRegistry, so Prometheus scrapes on
/metrics surface RAG observability.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.rag import (
    AnswerCache,
    CachedAnswer,
    CostTracker,
    QueryEmbeddingCache,
)
from src.intelligence.rag.metrics_bridge import (
    RagMetricsHandles,
    register_rag_metrics,
    snapshot,
)
from src.performance.metrics import MetricsRegistry


@pytest.fixture
def registry() -> MetricsRegistry:
    return MetricsRegistry()


@pytest.fixture
def handles(registry) -> RagMetricsHandles:
    return register_rag_metrics(registry)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_returns_handles_for_all_gauges(handles):
    for attr in (
        "cache_hits",
        "cache_misses",
        "cache_evictions",
        "cache_size",
        "cache_hit_ratio",
        "cost_usd_total",
        "llm_calls",
        "embedding_calls",
    ):
        assert getattr(handles, attr) is not None


def test_register_is_idempotent(registry):
    h1 = register_rag_metrics(registry)
    h2 = register_rag_metrics(registry)
    # Same underlying gauge object on a second call.
    assert h1.cache_hits is h2.cache_hits
    assert h1.cost_usd_total is h2.cost_usd_total


# ---------------------------------------------------------------------------
# snapshot()
# ---------------------------------------------------------------------------


def test_snapshot_mirrors_answer_cache_stats(handles):
    cache = AnswerCache()
    val = CachedAnswer(answer="x", retrieved_chunk_ids=[])
    cache.put("q1", "en", 5, "fp", val)
    cache.get("q1", "en", 5, "fp")  # hit
    cache.get("q2", "en", 5, "fp")  # miss

    snapshot(handles, answer_cache=cache)
    labels = {"cache": "answer"}
    assert handles.cache_hits.get(labels) == 1
    assert handles.cache_misses.get(labels) == 1
    assert handles.cache_size.get(labels) == 1
    assert handles.cache_hit_ratio.get(labels) == 0.5


def test_snapshot_mirrors_embedding_cache_stats(handles):
    cache = QueryEmbeddingCache()
    cache.put("q", np.array([1.0, 0.0], dtype=np.float32))
    cache.get("q")  # hit
    cache.get("q")  # hit
    cache.get("missing")  # miss

    snapshot(handles, embedding_cache=cache)
    labels = {"cache": "embedding"}
    assert handles.cache_hits.get(labels) == 2
    assert handles.cache_misses.get(labels) == 1
    assert handles.cache_hit_ratio.get(labels) == pytest.approx(2 / 3)


def test_snapshot_mirrors_cost_tracker_breakdown(handles):
    tr = CostTracker()
    tr.record_llm("claude-haiku-4-5", 1_000_000, 0, tier="STRATEGIST")
    tr.record_embedding("voyage-3-large", 2_000_000, tier="STRATEGIST")
    snapshot(handles, cost_tracker=tr)

    assert handles.cost_usd_total.get({"kind": "llm"}) == pytest.approx(1.00)
    assert handles.cost_usd_total.get({"kind": "embedding"}) == pytest.approx(0.36)
    assert handles.cost_usd_total.get({"kind": "all"}) == pytest.approx(1.36)
    assert handles.llm_calls.get() == 1
    assert handles.embedding_calls.get() == 1


def test_snapshot_handles_partial_inputs(handles):
    """Calling snapshot with only one source of truth shouldn't error."""
    snapshot(handles, cost_tracker=CostTracker())  # cache args omitted
    snapshot(handles, answer_cache=AnswerCache())  # cost_tracker omitted


def test_snapshot_distinguishes_caches_via_label(handles):
    """answer + embedding caches must NOT collide on the gauge series."""
    answer = AnswerCache()
    answer.put("q", "en", 5, "fp", CachedAnswer(answer="a", retrieved_chunk_ids=[]))
    answer.get("q", "en", 5, "fp")
    embed = QueryEmbeddingCache()
    embed.put("q", np.array([1.0], dtype=np.float32))
    embed.get("q")

    snapshot(handles, answer_cache=answer, embedding_cache=embed)
    assert handles.cache_hits.get({"cache": "answer"}) == 1
    assert handles.cache_hits.get({"cache": "embedding"}) == 1


# ---------------------------------------------------------------------------
# Prometheus exposition
# ---------------------------------------------------------------------------


def test_metrics_appear_in_prometheus_text(registry, handles):
    cache = AnswerCache()
    cache.put("q", "en", 5, "fp", CachedAnswer(answer="x", retrieved_chunk_ids=[]))
    cache.get("q", "en", 5, "fp")
    snapshot(handles, answer_cache=cache)

    text = registry.to_prometheus()
    assert "rag_cache_hits" in text
    assert 'cache="answer"' in text
