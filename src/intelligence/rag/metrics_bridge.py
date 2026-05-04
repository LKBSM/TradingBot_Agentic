"""Bridge RAG cache + cost-tracker stats to the MetricsRegistry — Sprint OBS-2B.1.

Why a separate module
---------------------
``cache.py`` and ``cost_tracker.py`` are deliberately framework-agnostic
(they don't import from ``src.performance``). The /metrics endpoint
needs those numbers though, so this module is the explicit bridge —
register a fixed set of gauges on a registry, then call ``snapshot()``
to refresh them from the live counters.

Pull, not push
--------------
We don't want every cache hit to acquire a metrics lock. Instead, the
metrics handler / a periodic timer calls ``snapshot()`` which reads each
stat once and ``set()``s the gauge. CacheStats is already lock-protected
so this stays consistent.

Metric names (Prometheus-style)
-------------------------------
- ``rag_cache_hits``     {cache="answer"|"embedding"}
- ``rag_cache_misses``   {cache="answer"|"embedding"}
- ``rag_cache_evictions``{cache="answer"|"embedding"}
- ``rag_cache_size``     {cache="answer"|"embedding"}
- ``rag_cache_hit_ratio``{cache="answer"|"embedding"} (0.0-1.0)
- ``rag_cost_usd_total`` {kind="llm"|"embedding"|"all"}
- ``rag_llm_calls``
- ``rag_embedding_calls``

Gauges (not counters) since we're mirroring counters that already exist
upstream — ``set()`` is the right operation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from src.intelligence.rag.cache import AnswerCache, QueryEmbeddingCache
from src.intelligence.rag.cost_tracker import CostTracker
from src.performance.metrics import Gauge, MetricsRegistry

logger = logging.getLogger(__name__)


@dataclass
class RagMetricsHandles:
    """Direct handles to the gauges registered for the RAG subsystem."""

    cache_hits: Gauge
    cache_misses: Gauge
    cache_evictions: Gauge
    cache_size: Gauge
    cache_hit_ratio: Gauge
    cost_usd_total: Gauge
    llm_calls: Gauge
    embedding_calls: Gauge


def register_rag_metrics(registry: MetricsRegistry) -> RagMetricsHandles:
    """Register the RAG metric series on ``registry`` once.

    Idempotent: a second call returns handles to the same series. The
    underlying ``MetricsRegistry`` deduplicates by name.
    """
    handles = RagMetricsHandles(
        cache_hits=registry.gauge(
            "rag_cache_hits",
            "RAG cache hit count (snapshot from in-memory CacheStats).",
        ),
        cache_misses=registry.gauge(
            "rag_cache_misses",
            "RAG cache miss count (snapshot from in-memory CacheStats).",
        ),
        cache_evictions=registry.gauge(
            "rag_cache_evictions",
            "RAG cache evictions (LRU + TTL combined).",
        ),
        cache_size=registry.gauge(
            "rag_cache_size",
            "Current RAG cache occupancy.",
        ),
        cache_hit_ratio=registry.gauge(
            "rag_cache_hit_ratio",
            "RAG cache hit ratio in [0,1] over the lifetime of the process.",
        ),
        cost_usd_total=registry.gauge(
            "rag_cost_usd_total",
            "Cumulative USD cost recorded by the RAG CostTracker.",
        ),
        llm_calls=registry.gauge(
            "rag_llm_calls",
            "Total LLM calls recorded by the RAG CostTracker.",
        ),
        embedding_calls=registry.gauge(
            "rag_embedding_calls",
            "Total embedding calls recorded by the RAG CostTracker.",
        ),
    )
    return handles


def snapshot(
    handles: RagMetricsHandles,
    *,
    answer_cache: Optional[AnswerCache] = None,
    embedding_cache: Optional[QueryEmbeddingCache] = None,
    cost_tracker: Optional[CostTracker] = None,
) -> None:
    """Refresh every gauge from the source-of-truth counters.

    Call before every /metrics scrape, or on a periodic timer. Caches /
    trackers that aren't provided are simply skipped; their gauges keep
    whatever value was last set (or 0).
    """
    if answer_cache is not None:
        s = answer_cache.stats()
        labels = {"cache": "answer"}
        handles.cache_hits.set(s.hits, labels=labels)
        handles.cache_misses.set(s.misses, labels=labels)
        handles.cache_evictions.set(s.evictions, labels=labels)
        handles.cache_size.set(s.size, labels=labels)
        handles.cache_hit_ratio.set(s.hit_ratio, labels=labels)

    if embedding_cache is not None:
        s = embedding_cache.stats()
        labels = {"cache": "embedding"}
        handles.cache_hits.set(s.hits, labels=labels)
        handles.cache_misses.set(s.misses, labels=labels)
        handles.cache_evictions.set(s.evictions, labels=labels)
        handles.cache_size.set(s.size, labels=labels)
        handles.cache_hit_ratio.set(s.hit_ratio, labels=labels)

    if cost_tracker is not None:
        summary = cost_tracker.summary()
        handles.cost_usd_total.set(summary.total_usd, labels={"kind": "all"})
        handles.cost_usd_total.set(summary.llm_usd, labels={"kind": "llm"})
        handles.cost_usd_total.set(summary.embedding_usd, labels={"kind": "embedding"})
        handles.llm_calls.set(summary.n_llm_calls)
        handles.embedding_calls.set(summary.n_embedding_calls)


__all__ = [
    "RagMetricsHandles",
    "register_rag_metrics",
    "snapshot",
]
