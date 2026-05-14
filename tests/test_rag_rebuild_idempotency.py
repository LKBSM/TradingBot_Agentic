"""Tests for the DATA-2B.10 RAG corpus rebuild idempotency token."""

from __future__ import annotations

import pytest

from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.embedders import HashEmbedder
from src.intelligence.rag.pipeline import RAGPipeline


def _chunks(*ids):
    return [Chunk(text=f"body for {i}", source_id=i, chunk_index=0) for i in ids]


@pytest.fixture
def pipeline():
    return RAGPipeline(embedder=HashEmbedder(dimension=32))


# ---------------------------------------------------------------------------
# rebuild_token
# ---------------------------------------------------------------------------


def test_token_deterministic_for_same_chunks(pipeline):
    chs = _chunks("a", "b", "c")
    t1 = pipeline.rebuild_token(chs)
    t2 = pipeline.rebuild_token(chs)
    assert t1 == t2
    assert len(t1) == 16


def test_token_changes_with_chunk_order(pipeline):
    a = pipeline.rebuild_token(_chunks("a", "b"))
    b = pipeline.rebuild_token(_chunks("b", "a"))
    assert a != b


def test_token_changes_with_chunk_membership(pipeline):
    a = pipeline.rebuild_token(_chunks("a", "b"))
    b = pipeline.rebuild_token(_chunks("a", "b", "c"))
    assert a != b


# ---------------------------------------------------------------------------
# ingest_idempotent
# ---------------------------------------------------------------------------


def test_first_call_ingests(pipeline):
    res = pipeline.ingest_idempotent(_chunks("a", "b", "c"))
    assert res["status"] == "ingested"
    assert res["n"] == 3
    assert res["size_before"] == 0
    assert res["size_after"] == 3


def test_second_call_with_same_chunks_is_no_op(pipeline):
    chs = _chunks("a", "b", "c")
    pipeline.ingest_idempotent(chs)
    res = pipeline.ingest_idempotent(chs)
    assert res["status"] == "no_op"
    assert res["n"] == 0
    assert res["size_after"] == 3
    # No accidental duplication
    assert pipeline.size == 3


def test_different_chunks_replaces_token_and_ingests(pipeline):
    pipeline.ingest_idempotent(_chunks("a", "b"))
    res = pipeline.ingest_idempotent(_chunks("a", "b", "c"))
    assert res["status"] == "ingested"
    # ingest is additive in this codebase — the new state shows the
    # extra chunk was added, not that the index was wiped.
    assert pipeline.size == 5  # initial 2 + new ingest of 3 (re-adds a,b,c)


def test_token_reflected_in_response(pipeline):
    chs = _chunks("a", "b")
    res = pipeline.ingest_idempotent(chs)
    assert res["token"] == pipeline.rebuild_token(chs)


# ---------------------------------------------------------------------------
# Empty chunk list
# ---------------------------------------------------------------------------


def test_empty_chunks_on_empty_pipeline_returns_noop():
    p = RAGPipeline(embedder=HashEmbedder(dimension=32))
    res = p.ingest_idempotent([])
    # Empty corpus, empty proposed set → tokens match → no_op
    assert res["status"] == "no_op"
    assert p.size == 0
