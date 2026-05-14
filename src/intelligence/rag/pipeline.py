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

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import numpy as np

from src.intelligence.rag.bm25 import BM25Index, BM25Hit
from src.intelligence.rag.cache import AnswerCache, CachedAnswer, QueryEmbeddingCache
from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.embedders import Embedder
from src.intelligence.rag.prompts import RAGPromptBundle, build_prompt_bundle
from src.intelligence.rag.vector_store import DenseHit, InMemoryVectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Corpus fingerprint (DATA-2B.8)
# ---------------------------------------------------------------------------


class CorpusDriftError(RuntimeError):
    """Raised when BM25 and the vector store disagree on the corpus.

    Distinct from generic ``RuntimeError`` so /health/deep can report a
    specific reason and ops can write a kill-switch alert against it.
    """


@dataclass(frozen=True)
class CorpusFingerprint:
    """Detail of the BM25 + vector store corpus alignment.

    Fields:
        fingerprint   - 16-char hex digest of the joined chunk_ids when
                        aligned, or the discrepancy summary when not.
                        Used as the AnswerCache key fragment.
        bm25_size     - chunks indexed in BM25.
        vector_size   - chunks indexed in the vector store.
        aligned       - both indexes hold the same chunk_ids in the same
                        order. False is a hard error in production.
        drift_reason  - human-readable explanation when ``aligned=False``.
    """

    fingerprint: str
    bm25_size: int
    vector_size: int
    aligned: bool
    drift_reason: str = ""


def _compute_corpus_fingerprint(
    bm25: "BM25Index", vector_store: "InMemoryVectorStore"
) -> CorpusFingerprint:
    bm_ids = bm25.chunk_ids() if hasattr(bm25, "chunk_ids") else []
    vs_ids = (
        vector_store.chunk_ids() if hasattr(vector_store, "chunk_ids") else []
    )
    bm_size, vs_size = len(bm_ids), len(vs_ids)

    if bm_size != vs_size:
        return CorpusFingerprint(
            fingerprint=f"drift:size:{bm_size}!={vs_size}",
            bm25_size=bm_size,
            vector_size=vs_size,
            aligned=False,
            drift_reason=(
                f"size mismatch: BM25={bm_size} chunks, "
                f"vector_store={vs_size} chunks"
            ),
        )
    if bm_ids != vs_ids:
        # Same count but at least one id differs — find the first to make
        # the error readable (with a cap so a 10k-chunk corpus doesn't
        # spam the log).
        first_diff = next(
            (
                f"#{i}: bm25={a!r}, vector={b!r}"
                for i, (a, b) in enumerate(zip(bm_ids, vs_ids))
                if a != b
            ),
            "",
        )
        return CorpusFingerprint(
            fingerprint=f"drift:order:{bm_size}",
            bm25_size=bm_size,
            vector_size=vs_size,
            aligned=False,
            drift_reason=f"chunk-id ordering differs ({first_diff})",
        )

    h = hashlib.sha1()
    for cid in bm_ids:
        h.update(cid.encode("utf-8"))
        h.update(b"\x1f")
    return CorpusFingerprint(
        fingerprint=h.hexdigest()[:16],
        bm25_size=bm_size,
        vector_size=vs_size,
        aligned=True,
    )


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
    # LLM-2B.11 — frozen audit metadata for the prompt template that was
    # used to render this response. ``None`` only if the prompt registry
    # wasn't configured (legacy paths, unit-test stubs). When wired,
    # this is the {template_id, version, sha256} triple the audit
    # ledger entry can stamp so a post-hoc regression analysis can pin
    # the exact prompt version that generated each insight.
    prompt_audit: Optional[dict] = None


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
        embedding_cache: Optional[QueryEmbeddingCache] = None,
        answer_cache: Optional[AnswerCache] = None,
        prompt_registry: Optional[Any] = None,
    ):
        self.embedder = embedder
        self.bm25_k = bm25_k
        self.dense_k = dense_k
        self.final_k = final_k
        self.max_context_chars = max_context_chars
        self.language = language
        self.embedding_cache = embedding_cache
        self.answer_cache = answer_cache
        # LLM-2B.11: when wired, the registry stamps a frozen
        # {template_id, version, sha256} triple onto every RAGResponse
        # so the audit ledger can track *which* prompt version
        # generated *which* insight.
        self.prompt_registry = prompt_registry

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

    # ------------------------------------------------------------------
    # Idempotent rebuild — DATA-2B.10
    # ------------------------------------------------------------------

    @staticmethod
    def rebuild_token(chunks: Iterable[Chunk]) -> str:
        """Compute a deterministic token for a *proposed* chunk set.

        Two callers that ingest the same chunks in the same order get
        the same token. Uses the same byte-stream layout as
        ``_compute_corpus_fingerprint`` (each chunk_id followed by
        ``\\x1f``), so a post-ingest fingerprint comparison against
        the pre-ingest token confirms the index landed correctly.
        """
        import hashlib
        h = hashlib.sha1()
        for c in chunks:
            h.update(c.chunk_id.encode("utf-8"))
            h.update(b"\x1f")
        return h.hexdigest()[:16]

    def ingest_idempotent(self, chunks: Iterable[Chunk]) -> dict:
        """Ingest only if the current corpus doesn't already match.

        DATA-2B.10: a crashed rebuild that left the corpus
        half-populated can re-run this without producing a duplicated
        chunk set. Compares the rebuild_token of the proposed chunks
        against the current corpus fingerprint:

        - **match** → no-op, return ``{"status": "no_op", ...}``.
        - **empty corpus or different fingerprint** → ingest as normal,
          return ``{"status": "ingested", "n": N, ...}``.

        The current implementation isn't atomic: a partial ingest
        that crashes mid-flight will produce a third state where the
        BM25 and vector indexes disagree. That state is caught by
        ``assert_indexes_aligned()`` (DATA-2B.8) — callers building
        production pipelines should run that immediately after
        ``ingest_idempotent`` to fail fast.
        """
        chunk_list = list(chunks)
        token = self.rebuild_token(chunk_list)
        current_fp = self.corpus_fingerprint()
        if current_fp.aligned and current_fp.fingerprint == token:
            return {
                "status": "no_op",
                "token": token,
                "n": 0,
                "size_before": self.size,
                "size_after": self.size,
            }
        size_before = self.size
        n = self.ingest(chunk_list)
        return {
            "status": "ingested",
            "token": token,
            "n": n,
            "size_before": size_before,
            "size_after": self.size,
        }

    @property
    def size(self) -> int:
        return self._bm25.size

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _corpus_fingerprint(self) -> str:
        """Stable identifier for the answer-cache key.

        DATA-2B.8: hash the ordered chunk_ids of BM25 + vector store. This
        catches three drift shapes:

        - corpus grew (new ingest) ⇒ different fingerprint ⇒ cache miss.
        - chunks reshuffled (re-ingest with same total count but different
          ids) ⇒ different fingerprint.
        - one of the two indexes drifted (BM25 has 500 chunks, vector
          store has 499 because an embed call failed silently) ⇒ different
          fingerprint, AND ``corpus_drift_seq`` flags it explicitly.
        """
        return self.corpus_fingerprint().fingerprint

    def corpus_fingerprint(self) -> "CorpusFingerprint":
        """Detailed fingerprint suitable for the deep-health probe."""
        return _compute_corpus_fingerprint(self._bm25, self._vector_store)

    def assert_indexes_aligned(self) -> None:
        """Boot guard: fail fast if BM25 and vector store disagree.

        Called automatically by the deep-health endpoint. Production
        wiring should call it once at app startup so a botched
        re-index crashes the pod rather than serving wrong answers.
        """
        fp = self.corpus_fingerprint()
        if not fp.aligned:
            raise CorpusDriftError(fp.drift_reason)

    def _prompt_audit(self) -> Optional[dict]:
        """Resolve the {template_id, version, sha256} for the active language.

        Returns ``None`` (legacy behaviour) when no registry is wired.
        When the registry knows the prompt but the active language tag
        isn't registered (e.g. exotic Accept-Language), falls back to
        the English template — same fallback ``build_prompt_bundle``
        uses, so audit and prompt stay aligned.
        """
        if self.prompt_registry is None:
            return None
        primary = f"rag.system.{self.language}"
        fallback = "rag.system.en"
        try:
            return self.prompt_registry.to_audit_dict(primary)
        except KeyError:
            try:
                return self.prompt_registry.to_audit_dict(fallback)
            except KeyError:
                return None

    def _embed_query(self, query: str) -> np.ndarray:
        if self.embedding_cache is not None:
            cached = self.embedding_cache.get(query)
            if cached is not None:
                return cached
        embedding = self.embedder.embed([query])[0]
        if self.embedding_cache is not None:
            self.embedding_cache.put(query, embedding)
        return embedding

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """Hybrid retrieval only — no LLM call."""
        bm25_hits = self._bm25.search(query, k=self.bm25_k)
        query_embedding = self._embed_query(query)
        dense_hits = self._vector_store.search(query_embedding, k=self.dense_k)
        fused = reciprocal_rank_fusion(bm25_hits, dense_hits)
        return fused[: self.final_k]

    def query(
        self,
        query: str,
        llm: Optional[LLMCallable] = None,
    ) -> RAGResponse:
        """Run the full RAG pipeline. ``llm`` is optional: when None, the
        response carries the prompt bundle for inspection but no answer.

        When an :class:`AnswerCache` is configured, identical
        ``(query, language, top_k)`` triples short-circuit the BM25 +
        dense + LLM stack and replay the cached answer + chunk IDs.
        ``elapsed_seconds`` then carries a single ``cache_hit`` entry so
        observability can distinguish hits from cold runs.
        """
        timings: dict[str, float] = {}

        # ─── Answer cache short-circuit ────────────────────────────────
        if self.answer_cache is not None and llm is not None:
            t0 = time.perf_counter()
            cached = self.answer_cache.get(
                query, self.language, self.final_k, self._corpus_fingerprint()
            )
            timings["cache_lookup"] = time.perf_counter() - t0
            if cached is not None:
                # Rehydrate retrieved chunks from cached IDs by querying the
                # corpus. We need the texts back for the prompt bundle to
                # stay coherent — at this point the BM25 index is the
                # authoritative source of chunk content.
                retrieved = self._rehydrate_cached(cached)
                ctx = [
                    (rc.chunk.chunk_id, rc.chunk.text, rc.chunk.metadata)
                    for rc in retrieved
                ]
                prompt = build_prompt_bundle(
                    query, ctx, language=self.language,
                    max_context_chars=self.max_context_chars,
                )
                timings["cache_hit"] = 1.0  # marker for observability
                return RAGResponse(
                    query=query,
                    retrieved=retrieved,
                    prompt_bundle=prompt,
                    answer=cached.answer,
                    elapsed_seconds=timings,
                    prompt_audit=self._prompt_audit(),
                )

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

            # Only cache real LLM answers — stub paths are deterministic
            # and cheap enough that caching adds noise without saving cost.
            if self.answer_cache is not None and answer:
                self.answer_cache.put(
                    query,
                    self.language,
                    self.final_k,
                    self._corpus_fingerprint(),
                    CachedAnswer(
                        answer=answer,
                        retrieved_chunk_ids=[rc.chunk.chunk_id for rc in retrieved],
                        retrieved_chunks_text=[rc.chunk.text for rc in retrieved],
                        extras={"source_ids": [rc.chunk.source_id for rc in retrieved]},
                    ),
                )

        return RAGResponse(
            query=query,
            retrieved=retrieved,
            prompt_bundle=prompt,
            answer=answer,
            elapsed_seconds=timings,
            prompt_audit=self._prompt_audit(),
        )

    def _rehydrate_cached(self, cached: "CachedAnswer") -> list["RetrievedChunk"]:
        """Reconstruct ``RetrievedChunk`` objects from a cache entry.

        Looks the chunks up by chunk_id in the BM25 index. Falls back to
        synthesising a Chunk from the cached text when the corpus has
        rotated (rare — should not happen due to the corpus fingerprint
        invalidation, but defensive).
        """
        out: list[RetrievedChunk] = []
        index = {c.chunk_id: c for c in self._bm25.chunks}
        for chunk_id, text in zip(
            cached.retrieved_chunk_ids, cached.retrieved_chunks_text
        ):
            chunk = index.get(chunk_id)
            if chunk is None:
                chunk = Chunk(text=text, source_id=chunk_id, chunk_index=0)
            out.append(RetrievedChunk(chunk=chunk, fused_score=0.0))
        return out
