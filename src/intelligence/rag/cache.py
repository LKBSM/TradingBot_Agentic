"""TTL-bounded caches for the RAG pipeline — Sprint LLM-2B.8.

Two caches with disjoint scopes:

- :class:`QueryEmbeddingCache` — keyed by ``(text,)``. The embedding of a
  given query string is independent of the corpus, so it's safe to cache
  for as long as the embedder model version is unchanged. Saves Voyage
  tokens on identical broker payloads (B2B reconciliation traffic).
- :class:`AnswerCache` — keyed by ``(query_norm, language, top_k,
  corpus_fingerprint)``. The full RAG response is cached so identical
  questions don't re-run BM25 + dense + LLM. Invalidated automatically
  when the corpus changes (size delta).

Both caches:
- LRU eviction with ``max_size`` bound.
- TTL with monotonic clock (configurable, default 1h).
- Thread-safe via a single ``threading.Lock``.
- Stats: ``hits``, ``misses``, ``evictions``, ``hit_ratio``.

The pipeline integration is opt-in: ``RAGPipeline(cache=...)``. Without
a cache, behaviour is unchanged.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Generic, Hashable, Optional, TypeVar

import numpy as np


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class CacheStats:
    """Snapshot of cache performance counters."""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0
    capacity: int = 0

    @property
    def total(self) -> int:
        return self.hits + self.misses

    @property
    def hit_ratio(self) -> float:
        return self.hits / self.total if self.total else 0.0


# ---------------------------------------------------------------------------
# Generic TTL+LRU cache
# ---------------------------------------------------------------------------


@dataclass
class _Entry(Generic[T]):
    value: T
    expires_at: float


class _TTLLRUCache(Generic[T]):
    """Internal: LRU cache with TTL eviction. Thread-safe."""

    def __init__(self, max_size: int, ttl_seconds: float):
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._store: "OrderedDict[Hashable, _Entry[T]]" = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _now(self) -> float:
        return time.monotonic()

    def get(self, key: Hashable) -> Optional[T]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.expires_at < self._now():
                # Expired — evict and miss.
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                return None
            # Promote to MRU.
            self._store.move_to_end(key, last=True)
            self._hits += 1
            return entry.value

    def put(self, key: Hashable, value: T) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key, last=True)
            self._store[key] = _Entry(value=value, expires_at=self._now() + self._ttl)
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
                size=len(self._store),
                capacity=self._max_size,
            )

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# ---------------------------------------------------------------------------
# Query embedding cache
# ---------------------------------------------------------------------------


def _normalise_text(text: str) -> str:
    """Whitespace-collapsed lowercase form used for both embedding and answer
    cache keys. Two variants of the same question with extra spaces / casing
    therefore share a cache slot."""
    return " ".join(text.lower().split())


def _hash_key(*parts: Any) -> str:
    """SHA-256 hex of the concatenated string parts. Stable across processes."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()


class QueryEmbeddingCache:
    """Cache the embedding of a query string. Independent of corpus.

    Capacity defaults to 1024 entries — typical query frequency tail; tune
    higher for B2B traffic with many distinct broker contexts.
    """

    DEFAULT_MAX_SIZE = 1024
    DEFAULT_TTL_SECONDS = 3600.0

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        embedder_signature: str = "default",
    ):
        self._cache: _TTLLRUCache[np.ndarray] = _TTLLRUCache(max_size, ttl_seconds)
        # Bumping the signature (e.g. when switching from Hash → Voyage) is
        # how callers invalidate without scanning the cache.
        self._embedder_signature = embedder_signature

    def get(self, query: str) -> Optional[np.ndarray]:
        key = _hash_key(self._embedder_signature, _normalise_text(query))
        return self._cache.get(key)

    def put(self, query: str, embedding: np.ndarray) -> None:
        key = _hash_key(self._embedder_signature, _normalise_text(query))
        self._cache.put(key, embedding)

    def stats(self) -> CacheStats:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


# ---------------------------------------------------------------------------
# Answer cache
# ---------------------------------------------------------------------------


@dataclass
class CachedAnswer:
    """The pieces of a RAGResponse worth caching.

    We don't cache the ``RAGResponse`` directly because it contains a
    fresh ``elapsed_seconds`` map; replaying that as if it were the
    original timing would mislead observability. Callers reconstruct
    a response with cache-hit timing instead.
    """

    answer: str
    retrieved_chunk_ids: list[str]
    retrieved_chunks_text: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)


class AnswerCache:
    """Cache the answer for ``(query, lang, top_k)`` bound to a corpus."""

    DEFAULT_MAX_SIZE = 512
    DEFAULT_TTL_SECONDS = 1800.0  # 30 minutes — narrative freshness ceiling

    def __init__(
        self,
        max_size: int = DEFAULT_MAX_SIZE,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
    ):
        self._cache: _TTLLRUCache[CachedAnswer] = _TTLLRUCache(max_size, ttl_seconds)

    @staticmethod
    def make_key(
        query: str, language: str, top_k: int, corpus_fingerprint: str
    ) -> str:
        return _hash_key(
            _normalise_text(query),
            language,
            int(top_k),
            corpus_fingerprint,
        )

    def get(
        self, query: str, language: str, top_k: int, corpus_fingerprint: str
    ) -> Optional[CachedAnswer]:
        return self._cache.get(self.make_key(query, language, top_k, corpus_fingerprint))

    def put(
        self,
        query: str,
        language: str,
        top_k: int,
        corpus_fingerprint: str,
        value: CachedAnswer,
    ) -> None:
        self._cache.put(
            self.make_key(query, language, top_k, corpus_fingerprint), value
        )

    def stats(self) -> CacheStats:
        return self._cache.stats()

    def clear(self) -> None:
        self._cache.clear()


__all__ = [
    "AnswerCache",
    "CachedAnswer",
    "CacheStats",
    "QueryEmbeddingCache",
]
