"""Tests for the LLM-2B.5 Q&A endpoint.

Verifies:
- Stub mode returns a deterministic answer built from retrieved context
- Real-LLM mode is exercised when an llm callable is wired into the app
- Sources carry registry metadata (label, type, ref, authority_score)
- Validation errors land at 400/422 (not 500)
- 503 when no RAG pipeline is configured
- Per-request top_k and language don't leak into the long-lived pipeline
"""

from __future__ import annotations

import os

# conftest.py already imports src.api.auth before any test module loads, so
# TESTING_MODE is captured at import time. We patch it module-by-module via
# fixture rather than rely on env-var ordering.
os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.qa import build_default_rag_pipeline


@pytest.fixture(autouse=True)
def _force_testing_mode():
    """Patch the captured-at-import TESTING_MODE flags in every module that
    branches on auth, so tier checks are bypassed for the QA endpoint."""
    with patch("src.api.auth.TESTING_MODE", True), patch(
        "src.api.routes.qa.TESTING_MODE", True
    ):
        yield


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_pipeline():
    return build_default_rag_pipeline()


@pytest.fixture
def client_stub(populated_pipeline):
    """App with RAG wired in but no LLM => responses use stub mode."""
    app = create_app(rag_pipeline=populated_pipeline, rag_llm=None)
    return TestClient(app)


@pytest.fixture
def client_with_llm(populated_pipeline):
    """App with a deterministic stub-LLM injected via rag_llm."""

    def stub_llm(system: str, user: str) -> str:
        # Echo a predictable phrase so faithfulness/relevancy tests can detect it.
        return "TEST-LLM answer derived from retrieved sources."

    app = create_app(rag_pipeline=populated_pipeline, rag_llm=stub_llm)
    return TestClient(app)


@pytest.fixture
def client_no_rag():
    """App without a RAG pipeline => /qa returns 503."""
    app = create_app()
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_qa_returns_sources_in_stub_mode(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "What is BOS in Smart Money Concepts?", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "What is BOS in Smart Money Concepts?"
    assert body["language"] == "en"
    assert body["stub_mode"] is True
    assert body["answer"]
    assert len(body["sources"]) == 5  # default top_k
    s0 = body["sources"][0]
    for key in ("source_id", "label", "type", "ref", "authority_score", "fused_score"):
        assert key in s0


def test_qa_uses_real_llm_when_configured(client_with_llm):
    resp = client_with_llm.post(
        "/api/v1/qa",
        json={"query": "Explain HAR-RV volatility forecasting", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stub_mode"] is False
    assert "TEST-LLM" in body["answer"]


def test_qa_french_language_uses_french_disclaimer(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "Qu'est-ce que le COT report ?", "language": "fr"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "fr"
    assert body["disclaimer"]
    # French disclaimer should be in French — sanity check on lexicon.
    text = body["disclaimer"].lower()
    assert any(token in text for token in ["risque", "investissement", "conseil", "garantie"])


def test_qa_top_k_param_respected(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "Why does the dollar matter for gold?", "top_k": 3},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["sources"]) == 3


def test_qa_returns_elapsed_ms(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "What is the VIX?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "retrieve" in body["elapsed_ms"]
    assert "assemble" in body["elapsed_ms"]
    assert body["elapsed_ms"]["retrieve"] >= 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_qa_503_when_pipeline_missing(client_no_rag):
    resp = client_no_rag.post("/api/v1/qa", json={"query": "anything goes here"})
    assert resp.status_code == 503


def test_qa_too_short_query_rejected(client_stub):
    # Pydantic validation: < 3 chars
    resp = client_stub.post("/api/v1/qa", json={"query": "ab"})
    assert resp.status_code == 422


def test_qa_invalid_language_rejected(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "What is gold?", "language": "de"},
    )
    assert resp.status_code == 422


def test_qa_top_k_out_of_range_rejected(client_stub):
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "What is gold?", "top_k": 99},
    )
    assert resp.status_code == 422


def test_qa_does_not_leak_state_across_requests(client_stub, populated_pipeline):
    """Per-request top_k / language must not mutate the underlying pipeline."""
    original_lang = populated_pipeline.language
    original_k = populated_pipeline.final_k

    client_stub.post(
        "/api/v1/qa",
        json={"query": "What is HAR-RV?", "language": "fr", "top_k": 7},
    )

    assert populated_pipeline.language == original_lang
    assert populated_pipeline.final_k == original_k


def test_qa_sanitizes_query(client_stub):
    """The sanitizer should strip control chars and trim whitespace."""
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "  What is the VIX?\x00\x01  "},
    )
    assert resp.status_code == 200
    body = resp.json()
    # No control chars survive
    assert "\x00" not in body["query"]
    assert "\x01" not in body["query"]
    # Whitespace is trimmed
    assert body["query"] == body["query"].strip()


def test_qa_returns_top_source_relevant_to_query(client_stub):
    """End-to-end retrieval quality regression: a CFTC question should
    surface a CFTC-tagged source in the top-3."""
    resp = client_stub.post(
        "/api/v1/qa",
        json={"query": "CFTC COT release schedule for Comex Gold"},
    )
    assert resp.status_code == 200
    body = resp.json()
    top3_ids = [s["source_id"] for s in body["sources"][:3]]
    assert any("cftc" in sid.lower() or "cot" in sid.lower() for sid in top3_ids), (
        f"expected CFTC/COT source in top-3, got {top3_ids}"
    )


def test_qa_stub_answer_when_no_retrieval(populated_pipeline):
    """If the corpus is empty, stub answer falls back to the 'insufficient' message."""
    from src.intelligence.rag import HashEmbedder, RAGPipeline

    empty_pipe = RAGPipeline(embedder=HashEmbedder(dimension=128, seed=1))
    app = create_app(rag_pipeline=empty_pipe)
    client = TestClient(app)
    resp = client.post(
        "/api/v1/qa",
        json={"query": "What is gold?", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["stub_mode"] is True
    assert "insufficient" in body["answer"].lower()
