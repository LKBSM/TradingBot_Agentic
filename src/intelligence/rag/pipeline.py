"""Top-level RAG pipeline.

Hybrid retrieval (BM25 sparse + dense embeddings) → reciprocal rank fusion
→ optional rerank → assembled prompt → optional LLM call.

Designed so callers can:
- ingest a corpus once, query many times
- swap embedders (HashEmbedder for tests, VoyageEmbedder for prod)
- skip the LLM call to inspect the assembled prompt (cheap CI eval)
- swap in a real LLM client (Anthropic, OpenAI, etc.) via a callable

Latency budget per LLM-2B.1: p99 < 4s end-to-end. With HashEmbedder +
in-memory store the retrieval portion is ~5ms; the LLM call dominates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Optional

import numpy as np

from src.intelligence.rag.bm25 import BM25Index, BM25Hit
from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.embedders import Embedder
from src.intelligence.rag.prompts import RAGPromptBundle, build_prompt_bundle
from src.intelligence.rag.vector_store import DenseHit, InMemoryVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RetrievedChunk:
    """One chunk after fusion of sparse + dense retrievers."""

    chunk: Chunk
    fused_score: float
    bm25_score: float = 0.0
    dense_score: float = 0.0


@dataclass
class RAGResponse:
    """End-to-end RAG output."""

    query: str
    retrieved: list[RetrievedChunk]
    prompt_bundle: RAGPromptBundle
    answer: str = ""
    elapsed_seconds: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reciprocal rank fusion
# ---------------------------------------------------------------------------


def reciprocal_rank_fusion(
    bm25_hits: list[BM25Hit],
    dense_hits: list[DenseHit],
    k_smoothing: int = 60,
) -> list[RetrievedChunk]:
    """Merge two ranked lists into one, weighted by reciprocal-rank.

    RRF (Cormack et al. 2009) is the canonical "no-tuning" hybrid fusion:
    score(d) = sum over rankers r of 1 / (k + rank_r(d)). Robust to
    different score scales (BM25 unbounded vs cosine in [-1, 1]) without
    requiring score normalisation.

    `k_smoothing=60` is the value Cormack recommended; safe default.
    """
    fused: dict[str, RetrievedChunk] = {}
    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit.chunk.chunk_id
        contrib = 1.0 / (k_smoothing + rank)
        fused.setdefault(
            cid,
            RetrievedChunk(chunk=hit.chunk, fused_score=0.0, bm25_score=hit.score),
        )
        fused[cid].fused_score += contrib
        fused[cid].bm25_score = hit.score
    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit.chunk.chunk_id
        contrib = 1.0 / (k_smoothing + rank)
        fused.setdefault(
            cid,
            RetrievedChunk(chunk=hit.chunk, fused_score=0.0, dense_score=hit.score),
        )
        fused[cid].fused_score += contrib
        fused[cid].dense_score = hit.score
    ranked = sorted(fused.values(), key=lambda r: -r.fused_score)
    return ranked


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


# Type alias for a function that takes (system, user) and returns answer text.
# Lets callers plug Claude / OpenAI / a stub.
LLMCallable = Callable[[str, str], str]


class RAGPipeline:
    """Hybrid RAG pipeline.

    Lifecycle:
        pipe = RAGPipeline(embedder=HashEmbedder())
        pipe.ingest(chunks)
        response = pipe.query("Why is XAU bullish today?", llm=my_llm_fn)

    Without an `llm` argument, `query()` returns the assembled prompt
    bundle without making a network call — useful for unit tests and for
    CI eval where the LLM is replaced by a stub.
    """

    def __init__(
        self,
        embedder: Embedder,
        bm25_k: int = 20,
        dense_k: int = 20,
        final_k: int = 5,
        max_context_chars: int = 6000,
        language: str = "fr",
    ):
        self.embedder = embedder
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.final_k = final_k
        self.max_context_chars = max_context_chars
        self.language = language

        self._bm25 = BM25Index()
        self._vector_store = InMemoryVectorStore(dimension=embedder.dimension)

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, chunks: Iterable[Chunk]) -> int:
        """Add chunks to both indices. Returns the number ingested."""
        chunk_list = list(chunks)
        if not chunk_list:
            return 0
        # Embed in one batch (callers can stream by ingest()-ing slices)
        texts = [c.text for c in chunk_list]
        embeddings = self.embedder.embed(texts)
        self._bm25.add(chunk_list)
        self._vector_store.add(chunk_list, embeddings)
        logger.info("RAG ingested %d chunks (corpus size: %d)", len(chunk_list), self.size)
        return len(chunk_list)

    @property
    def size(self) -> int:
        return self._bm25.size

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Hybrid retrieval only — no LLM call."""
        bm25_hits = self._bm25.search(query, k=self.bm25_k)
        query_embedding = self.embedder.embed([query])[0]
        dense_hits = self._vector_store.search(query_embedding, k=self.dense_k)
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits)
        return fused[: self.final_k]

    def query(
        self,
        query: str,
        llm: Optional[LLMCallable] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline. ``llm`` is optional: when None, the
        response carries the prompt bundle for inspection but no answer."""
        timings: dict[str, float] = {}

        t0 = time.perf_counter()
        retrieved = self.retrieve(query)
        timings["retrieve"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        ctx = [
            (rc.chunk.chunk_id, rc.chunk.text, rc.chunk.metadata) for rc in retrieved
        ]
        prompt = build_prompt_bundle(
            query, ctx, language=self.language, max_context_chars=self.max_context_chars
        )
        timings["assemble"] = time.perf_counter() - t0

        answer = ""
        if llm is not None:
            t0 = time.perf_counter()
            answer = llm(prompt.system, prompt.user)
            timings["llm"] = time.perf_counter() - t0

        return RAGResponse(
            query=query,
            retrieved=retrieved,
            prompt_bundle=prompt,
            answer=answer,
            elapsed_seconds=timings,
        )
