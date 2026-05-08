"""Tests for the DATA-2B.7 embedder smoke probe."""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.rag.embedders import (
    EmbedderHealthError,
    HashEmbedder,
    embed_health_check,
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_health_check_passes_for_hash_embedder():
    e = HashEmbedder(dimension=128)
    result = embed_health_check(e)
    assert result["ok"] is True
    assert result["configured"] is True
    assert result["dimension"] == 128
    assert result["embedder_class"] == "HashEmbedder"
    # L2 norms should be ~1.0
    for n in result["sample_norms"]:
        assert abs(n - 1.0) < 1e-3
    # Cosine should be < 1 (probe texts are vocab-disjoint)
    assert result["sample_cosine"] < 0.99


def test_health_check_with_expected_dimension_match():
    e = HashEmbedder(dimension=256)
    result = embed_health_check(e, expected_dimension=256)
    assert result["ok"] is True


def test_health_check_returns_unconfigured_for_none():
    result = embed_health_check(None)
    assert result == {"configured": False, "ok": True}


def test_health_check_records_duration():
    e = HashEmbedder(dimension=64)
    result = embed_health_check(e)
    assert result["duration_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_health_check_rejects_dimension_mismatch():
    e = HashEmbedder(dimension=64)
    with pytest.raises(EmbedderHealthError, match="dimension mismatch"):
        embed_health_check(e, expected_dimension=1024)


class _BadShapeEmbedder:
    dimension = 32

    def embed(self, texts):
        # Wrong row count
        return np.ones((1, 32), dtype=np.float32)


def test_health_check_rejects_wrong_shape():
    with pytest.raises(EmbedderHealthError, match="shape"):
        embed_health_check(_BadShapeEmbedder())


class _UnnormalisedEmbedder:
    dimension = 32

    def embed(self, texts):
        # Norms are 5.0 — not L2-normalised
        v = np.zeros((2, 32), dtype=np.float32)
        v[0, 0] = 5.0
        v[1, 1] = 5.0
        return v


def test_health_check_rejects_unnormalised_output():
    with pytest.raises(EmbedderHealthError, match="not L2-normalised"):
        embed_health_check(_UnnormalisedEmbedder())


class _ConstantEmbedder:
    """Returns the same vector for every input — common bug shape."""

    dimension = 32

    def embed(self, texts):
        v = np.zeros((len(texts), 32), dtype=np.float32)
        v[:, 0] = 1.0  # all rows identical, norm = 1
        return v


def test_health_check_rejects_constant_vector_output():
    with pytest.raises(EmbedderHealthError, match="collinear"):
        embed_health_check(_ConstantEmbedder())


class _NaNEmbedder:
    dimension = 32

    def embed(self, texts):
        return np.full((2, 32), np.nan, dtype=np.float32)


def test_health_check_rejects_non_finite_norms():
    with pytest.raises(EmbedderHealthError, match="non-finite"):
        embed_health_check(_NaNEmbedder())


def test_health_check_rejects_too_small_dimension():
    """An embedder with dimension < 16 fails before embed() is called."""

    class _Tiny:
        dimension = 8

        def embed(self, texts):  # pragma: no cover — not reached
            return np.zeros((2, 8), dtype=np.float32)

    with pytest.raises(EmbedderHealthError, match="< 16"):
        embed_health_check(_Tiny())


# ---------------------------------------------------------------------------
# Integration: deep health endpoint surfaces embedder result
# ---------------------------------------------------------------------------


def test_deep_health_includes_embedder_check_when_wired():
    import os

    os.environ.setdefault("SENTINEL_TESTING_MODE", "1")
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from src.api.app import create_app

    with patch("src.api.auth.TESTING_MODE", True):
        c = TestClient(create_app(embedder=HashEmbedder(dimension=128)))
        resp = c.get("/api/v1/health/deep")
        assert resp.status_code == 200
        body = resp.json()
        assert body["checks"]["embedder"]["ok"] is True
        assert body["checks"]["embedder"]["dimension"] == 128


def test_deep_health_503_when_embedder_unhealthy():
    import os

    os.environ.setdefault("SENTINEL_TESTING_MODE", "1")
    from unittest.mock import patch

    from fastapi.testclient import TestClient

    from src.api.app import create_app

    with patch("src.api.auth.TESTING_MODE", True):
        c = TestClient(create_app(embedder=_ConstantEmbedder()))
        resp = c.get("/api/v1/health/deep")
        assert resp.status_code == 503
        assert resp.json()["checks"]["embedder"]["ok"] is False
