"""Tests for the DATA-2B.8 corpus fingerprint guard."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from src.intelligence.rag.bm25 import BM25Index
from src.intelligence.rag.chunking import Chunk
from src.intelligence.rag.embedders import HashEmbedder
from src.intelligence.rag.pipeline import (
    CorpusDriftError,
    CorpusFingerprint,
    RAGPipeline,
    _compute_corpus_fingerprint,
)
from src.intelligence.rag.vector_store import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunks(n: int = 3, prefix: str = "doc") -> list[Chunk]:
    out = []
    for i in range(n):
        out.append(
            Chunk(
                text=f"{prefix} chunk {i} text",
                source_id=f"{prefix}-{i}",
                chunk_index=0,
                metadata={"label": f"L{i}"},
            )
        )
    return out


def _build_pipeline(chunks: list[Chunk]) -> RAGPipeline:
    embedder = HashEmbedder(dimension=64)
    p = RAGPipeline(embedder=embedder, language="en")
    p.ingest(chunks)
    return p


# ---------------------------------------------------------------------------
# _compute_corpus_fingerprint — direct
# ---------------------------------------------------------------------------


def test_aligned_indexes_produce_short_hex_fingerprint():
    chunks = _chunks(3)
    bm25 = BM25Index()
    bm25.add(chunks)
    vec = InMemoryVectorStore(dimension=64)
    embedder = HashEmbedder(dimension=64)
    vec.add(chunks, embedder.embed([c.text for c in chunks]))

    fp = _compute_corpus_fingerprint(bm25, vec)
    assert fp.aligned is True
    assert fp.bm25_size == 3 and fp.vector_size == 3
    assert fp.drift_reason == ""
    # 16-char hex digest
    assert len(fp.fingerprint) == 16
    int(fp.fingerprint, 16)  # parses


def test_size_mismatch_flagged():
    chunks_a = _chunks(3)
    chunks_b = _chunks(2)
    bm25 = BM25Index()
    bm25.add(chunks_a)
    vec = InMemoryVectorStore(dimension=64)
    embedder = HashEmbedder(dimension=64)
    vec.add(chunks_b, embedder.embed([c.text for c in chunks_b]))

    fp = _compute_corpus_fingerprint(bm25, vec)
    assert fp.aligned is False
    assert "size mismatch" in fp.drift_reason
    assert fp.fingerprint.startswith("drift:size:")


def test_id_ordering_mismatch_flagged():
    chunks = _chunks(3)
    bm25 = BM25Index()
    bm25.add(chunks)
    vec = InMemoryVectorStore(dimension=64)
    embedder = HashEmbedder(dimension=64)
    # Same chunks but reversed order
    reversed_chunks = list(reversed(chunks))
    vec.add(reversed_chunks, embedder.embed([c.text for c in reversed_chunks]))

    fp = _compute_corpus_fingerprint(bm25, vec)
    # First chunk_id differs
    assert fp.aligned is False
    assert "ordering differs" in fp.drift_reason
    assert fp.fingerprint.startswith("drift:order:")


def test_fingerprint_changes_when_corpus_grows():
    a = _chunks(3, prefix="a")
    b = _chunks(2, prefix="b")
    bm25_1 = BM25Index()
    bm25_1.add(a)
    vec_1 = InMemoryVectorStore(dimension=64)
    embedder = HashEmbedder(dimension=64)
    vec_1.add(a, embedder.embed([c.text for c in a]))
    fp_before = _compute_corpus_fingerprint(bm25_1, vec_1).fingerprint

    bm25_1.add(b)
    vec_1.add(b, embedder.embed([c.text for c in b]))
    fp_after = _compute_corpus_fingerprint(bm25_1, vec_1).fingerprint

    assert fp_before != fp_after


def test_fingerprint_stable_across_two_constructions():
    """Same chunks → same fingerprint, regardless of object identity."""
    chunks = _chunks(3)

    def fp_for(chunks_):
        embedder = HashEmbedder(dimension=64)
        bm25 = BM25Index()
        bm25.add(chunks_)
        vec = InMemoryVectorStore(dimension=64)
        vec.add(chunks_, embedder.embed([c.text for c in chunks_]))
        return _compute_corpus_fingerprint(bm25, vec).fingerprint

    assert fp_for(chunks) == fp_for(chunks)


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------


def test_pipeline_corpus_fingerprint_returns_dataclass():
    p = _build_pipeline(_chunks(3))
    fp = p.corpus_fingerprint()
    assert isinstance(fp, CorpusFingerprint)
    assert fp.aligned is True


def test_pipeline_underscore_corpus_fingerprint_returns_string():
    """Cache key path expects a string, not a dataclass."""
    p = _build_pipeline(_chunks(3))
    fp_str = p._corpus_fingerprint()
    assert isinstance(fp_str, str)
    assert len(fp_str) == 16


def test_assert_indexes_aligned_raises_on_drift():
    """Inject a drift by adding to BM25 only and assert the boot guard
    catches it."""
    chunks = _chunks(2)
    p = _build_pipeline(chunks)
    # Add one more chunk to BM25 alone — drift introduced.
    extra = _chunks(1, prefix="extra")
    p._bm25.add(extra)

    with pytest.raises(CorpusDriftError, match="size mismatch"):
        p.assert_indexes_aligned()


def test_assert_indexes_aligned_passes_when_aligned():
    p = _build_pipeline(_chunks(3))
    p.assert_indexes_aligned()  # must not raise


# ---------------------------------------------------------------------------
# /health/deep integration
# ---------------------------------------------------------------------------


def test_deep_health_reports_corpus_fingerprint():
    import os
    os.environ.setdefault("SENTINEL_TESTING_MODE", "1")
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    p = _build_pipeline(_chunks(3))
    with patch("src.api.auth.TESTING_MODE", True):
        c = TestClient(create_app(rag_pipeline=p))
        resp = c.get("/api/v1/health/deep")
        assert resp.status_code == 200
        body = resp.json()["checks"]["rag_pipeline"]
        assert body["corpus_aligned"] is True
        assert body["bm25_size"] == 3
        assert body["vector_size"] == 3
        assert "corpus_fingerprint" in body


def test_deep_health_503_when_corpus_drifted():
    import os
    os.environ.setdefault("SENTINEL_TESTING_MODE", "1")
    from unittest.mock import patch
    from fastapi.testclient import TestClient

    from src.api.app import create_app

    p = _build_pipeline(_chunks(3))
    # Drift: extra chunk in BM25 only
    p._bm25.add(_chunks(1, prefix="extra"))

    with patch("src.api.auth.TESTING_MODE", True):
        c = TestClient(create_app(rag_pipeline=p))
        resp = c.get("/api/v1/health/deep")
        assert resp.status_code == 503
        body = resp.json()["checks"]["rag_pipeline"]
        assert body["corpus_aligned"] is False
        assert "size mismatch" in body["drift_reason"]
