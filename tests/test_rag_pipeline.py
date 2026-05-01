"""Tests for the LLM-2B.1 RAG pipeline.

Per DoD: F1 sourcing > 0.85 + hallucination < 5% will be measured by the
extended eval harness in LLM-2B.3 (Phase 2B follow-up). Here we test the
infrastructure contract: chunking, hybrid retrieval, RRF fusion,
prompt assembly, and latency.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.intelligence.rag import (
    BM25Index,
    Chunk,
    HashEmbedder,
    InMemoryVectorStore,
    RAGPipeline,
    chunk_text,
)
from src.intelligence.rag.bm25 import BM25Hit
from src.intelligence.rag.pipeline import (
    RetrievedChunk,
    reciprocal_rank_fusion,
)
from src.intelligence.rag.prompts import (
    SYSTEM_PROMPT_FR,
    SYSTEM_PROMPT_EN,
    assemble_user_prompt,
    build_prompt_bundle,
)
from src.intelligence.rag.vector_store import DenseHit


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def test_chunk_text_respects_overlap():
    text = " ".join(str(i) for i in range(2000))  # 2000 tokens
    chunks = chunk_text(text, source_id="t", chunk_tokens=500, overlap_tokens=100)
    # 2000 tokens, step 400 → 5 chunks: [0..499], [400..899], [800..1299], [1200..1699], [1600..1999]
    assert len(chunks) == 5
    # Verify overlap: last 100 tokens of chunk 0 == first 100 of chunk 1
    tokens0 = chunks[0].text.split()
    tokens1 = chunks[1].text.split()
    assert tokens0[-100:] == tokens1[:100]


def test_chunk_text_empty_returns_empty():
    assert chunk_text("", source_id="t") == []
    assert chunk_text("   ", source_id="t") == []


def test_chunk_text_respects_short_input():
    chunks = chunk_text("short text", source_id="s", chunk_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].text == "short text"


def test_chunk_id_is_deterministic():
    c1 = Chunk(text="abc", source_id="s1", chunk_index=0)
    c2 = Chunk(text="abc", source_id="s1", chunk_index=0)
    assert c1.chunk_id == c2.chunk_id
    # Different content ⇒ different id
    c3 = Chunk(text="abd", source_id="s1", chunk_index=0)
    assert c1.chunk_id != c3.chunk_id


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------


def test_hash_embedder_l2_normalised():
    emb = HashEmbedder(dimension=128)
    vecs = emb.embed(["the quick brown fox", "another sentence"])
    assert vecs.shape == (2, 128)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)


def test_hash_embedder_identical_text_gives_identical_embedding():
    emb = HashEmbedder()
    a = emb.embed(["same text"])[0]
    b = emb.embed(["same text"])[0]
    np.testing.assert_array_equal(a, b)


def test_hash_embedder_disjoint_vocab_low_similarity():
    emb = HashEmbedder(dimension=512)  # high dim ⇒ fewer hash collisions
    a = emb.embed(["cat dog mouse rabbit"])[0]
    b = emb.embed(["volatility regime macroeconomic"])[0]
    cosine = float(np.dot(a, b))
    assert cosine < 0.3


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


def test_bm25_returns_top_k_chunks_for_keyword():
    chunks = [
        Chunk(text="The Fed raised rates today", source_id="s1", chunk_index=0),
        Chunk(text="Gold price reached 2350", source_id="s2", chunk_index=0),
        Chunk(text="Federal Reserve monetary policy", source_id="s3", chunk_index=0),
    ]
    idx = BM25Index()
    idx.add(chunks)
    hits = idx.search("Fed monetary", k=2)
    assert len(hits) == 2
    # The exact "Fed" match should rank above generic gold price text
    ids = [h.chunk.source_id for h in hits]
    assert "s1" in ids or "s3" in ids


def test_bm25_returns_empty_on_no_match():
    idx = BM25Index()
    idx.add([Chunk(text="cat dog", source_id="x", chunk_index=0)])
    assert idx.search("completely_unrelated_term") == []


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------


def test_vector_store_search_returns_top_k():
    store = InMemoryVectorStore(dimension=4)
    chunks = [
        Chunk(text="a", source_id="s1", chunk_index=0),
        Chunk(text="b", source_id="s2", chunk_index=0),
        Chunk(text="c", source_id="s3", chunk_index=0),
    ]
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.7, 0.0],
        ],
        dtype=np.float32,
    )
    # L2-normalise
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    store.add(chunks, embeddings)

    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    hits = store.search(query, k=2)
    assert len(hits) == 2
    # First chunk is exact match ⇒ score = 1.0
    assert hits[0].chunk.source_id == "s1"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_vector_store_dimension_mismatch_raises():
    store = InMemoryVectorStore(dimension=4)
    chunks = [Chunk(text="a", source_id="s1", chunk_index=0)]
    bad = np.zeros((1, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="dim"):
        store.add(chunks, bad)


# ---------------------------------------------------------------------------
# Reciprocal Rank Fusion
# ---------------------------------------------------------------------------


def _mk_chunk(cid: str) -> Chunk:
    c = Chunk(text=cid, source_id=cid, chunk_index=0)
    return c


def test_rrf_promotes_chunks_in_both_lists():
    a, b, c = _mk_chunk("a"), _mk_chunk("b"), _mk_chunk("c")
    bm25 = [BM25Hit(chunk=a, score=10.0), BM25Hit(chunk=b, score=5.0)]
    dense = [DenseHit(chunk=b, score=0.9), DenseHit(chunk=c, score=0.5)]
    fused = reciprocal_rank_fusion(bm25, dense)
    # b appears in both ⇒ should rank first
    assert fused[0].chunk.chunk_id == b.chunk_id


def test_rrf_handles_disjoint_lists():
    a, b = _mk_chunk("a"), _mk_chunk("b")
    bm25 = [BM25Hit(chunk=a, score=10.0)]
    dense = [DenseHit(chunk=b, score=0.9)]
    fused = reciprocal_rank_fusion(bm25, dense)
    assert {r.chunk.chunk_id for r in fused} == {a.chunk_id, b.chunk_id}


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


def test_assemble_user_prompt_includes_query_and_chunks():
    chunks_ctx = [
        ("abc1", "First source text", {"type": "paper", "label": "Source A"}),
        ("abc2", "Second source text", {"type": "data", "label": "Source B"}),
    ]
    user = assemble_user_prompt("What is X?", chunks_ctx)
    assert "What is X?" in user
    assert "[source:abc1]" in user
    assert "[source:abc2]" in user
    assert "First source text" in user
    assert "Second source text" in user


def test_assemble_user_prompt_truncates_at_budget():
    long_text = "x" * 10_000
    chunks_ctx = [("a", long_text, {})]
    user = assemble_user_prompt("Q", chunks_ctx, max_context_chars=500)
    assert len(user) < 1500  # query + headers + truncated chunk
    assert "..." in user


def test_build_prompt_bundle_picks_language():
    chunks_ctx = [("a", "text", {})]
    fr = build_prompt_bundle("Q", chunks_ctx, language="fr")
    en = build_prompt_bundle("Q", chunks_ctx, language="en")
    assert fr.system == SYSTEM_PROMPT_FR
    assert en.system == SYSTEM_PROMPT_EN
    assert fr.cited_chunk_ids == ["a"]


def test_system_prompts_carry_anti_hallucination_rules():
    """The hard-rule block must be present in both languages."""
    for prompt in (SYSTEM_PROMPT_FR, SYSTEM_PROMPT_EN):
        assert "[source:" in prompt
        assert "insuffisant" in prompt or "insufficient" in prompt


def test_system_prompts_block_calls_to_action():
    """Compliance gate (UE 2024/2811): forbidden phrases must be listed."""
    for prompt in (SYSTEM_PROMPT_FR, SYSTEM_PROMPT_EN):
        # FR list: "achetez", "vendez", "100% sûr", "garanti"
        # EN list: "buy", "sell", "100% sure", "guaranteed"
        text = prompt.lower()
        assert "achetez" in text or "buy" in text


# ---------------------------------------------------------------------------
# End-to-end pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def populated_pipeline() -> RAGPipeline:
    pipe = RAGPipeline(embedder=HashEmbedder(dimension=256, seed=1))
    chunks = []
    chunks.extend(
        chunk_text(
            "Smart Money Concepts: Break of Structure is a confirmed close above "
            "the most recent swing high. After BOS, look for a retest of the FVG.",
            source_id="smc_001",
            metadata={"type": "education", "label": "SMC glossary"},
        )
    )
    chunks.extend(
        chunk_text(
            "CFTC COT report for Comex Gold (088691) week of 2026-04-22 showed "
            "Managed Money net long at 142503 contracts.",
            source_id="cftc_001",
            metadata={"type": "data", "label": "CFTC COT"},
        )
    )
    chunks.extend(
        chunk_text(
            "HAR-RV model by Corsi 2009 decomposes realised volatility into daily, "
            "weekly, and monthly components for accurate forecasting.",
            source_id="paper_001",
            metadata={"type": "paper", "label": "Corsi 2009"},
        )
    )
    pipe.ingest(chunks)
    return pipe


def test_pipeline_retrieves_relevant_chunks_top_1(populated_pipeline):
    response = populated_pipeline.query("What is a Break of Structure in SMC?")
    assert response.retrieved[0].chunk.source_id == "smc_001"


def test_pipeline_retrieves_har_rv_paper_top_1(populated_pipeline):
    response = populated_pipeline.query("How does HAR-RV decompose volatility?")
    assert response.retrieved[0].chunk.source_id == "paper_001"


def test_pipeline_retrieves_cot_data_top_1(populated_pipeline):
    response = populated_pipeline.query("Latest CFTC COT report on Gold?")
    assert response.retrieved[0].chunk.source_id == "cftc_001"


def test_pipeline_no_llm_returns_empty_answer(populated_pipeline):
    response = populated_pipeline.query("anything")
    assert response.answer == ""
    assert response.prompt_bundle.user.strip()  # but the prompt is built


def test_pipeline_with_stub_llm_returns_answer(populated_pipeline):
    def stub_llm(system: str, user: str) -> str:
        return f"[stub answer derived from {len(user)} chars of user message]"

    response = populated_pipeline.query("test", llm=stub_llm)
    assert response.answer.startswith("[stub answer")
    assert "llm" in response.elapsed_seconds


# ---------------------------------------------------------------------------
# Latency gate (LLM-2B.1 KPI)
# ---------------------------------------------------------------------------


def test_pipeline_retrieve_latency_under_100ms_per_query(populated_pipeline):
    """Plan KPI is end-to-end < 4s (LLM-bound). Retrieval-only should be
    nowhere near that — assert <100ms to catch performance regressions."""
    queries = ["BOS in SMC", "HAR-RV volatility", "COT report", "Gold price"] * 5
    t0 = time.perf_counter()
    for q in queries:
        populated_pipeline.retrieve(q)
    avg_ms = (time.perf_counter() - t0) / len(queries) * 1000
    assert avg_ms < 100.0, f"avg retrieval latency {avg_ms:.1f}ms is concerning"
