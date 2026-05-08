"""Pluggable embedding backends for the RAG pipeline.

Two implementations:

- `HashEmbedder`: deterministic, dependency-free, used in tests and as a
  baseline that lets us validate the pipeline shape before wiring real
  models. Maps each token to a hash bucket in a 256-dim vector.
- `VoyageEmbedder`: live Voyage AI client (`voyage-3-large` per plan),
  initialised lazily so missing API keys don't break imports.

Both implement the same `Embedder` Protocol so the pipeline is agnostic.

DATA-2B.7 also adds ``embed_health_check(embedder)``: a smoke probe used
both at boot (fail fast on dimension mismatch with the configured vector
store) and by the OBS-2B.2 deep health endpoint.
"""

from __future__ import annotations

import hashlib
import logging
import math
import os
import time
from typing import Iterable, Protocol, runtime_checkable

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """Embed texts into a fixed-dim L2-normalised numpy array."""

    dimension: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), self.dimension), L2-normalised rows."""
        ...


# ---------------------------------------------------------------------------
# Hash-based stub embedder
# ---------------------------------------------------------------------------


class HashEmbedder:
    """Deterministic stub for tests / local dev. NOT FOR PRODUCTION.

    Hashes whitespace tokens into buckets and counts. The result is L2-
    normalised so cosine similarity on the same text is 1.0 and on
    completely disjoint vocab is 0.0 — enough to verify the pipeline.
    """

    def __init__(self, dimension: int = 256, seed: int = 0):
        if dimension < 16:
            raise ValueError("dimension must be >= 16 for collision tolerance")
        self.dimension = dimension
        self._seed = seed

    def _bucket(self, token: str) -> int:
        h = hashlib.sha1(f"{self._seed}::{token}".encode("utf-8")).hexdigest()
        return int(h, 16) % self.dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in t.lower().split():
                out[i, self._bucket(tok)] += 1.0
        # L2-normalise
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# Voyage AI live embedder (lazy import)
# ---------------------------------------------------------------------------


class VoyageEmbedder:
    """Adapter over Voyage AI embeddings API.

    Activated only when ``VOYAGE_API_KEY`` is in the environment (or passed
    explicitly). Falls back to a clear error otherwise so misconfiguration
    is loud, not silent.

    Per plan LLM-2B.1: ``voyage-3-large`` at $0.18/1M tokens — cheapest
    high-quality dense retrieval option in 2026.
    """

    DEFAULT_MODEL = "voyage-3-large"
    DIMENSION = 1024  # voyage-3-large default

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        batch_size: int = 64,
    ):
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "VOYAGE_API_KEY missing — pass api_key= explicitly or set env var. "
                "Free tier at https://www.voyageai.com/."
            )
        self._model = model
        self._batch_size = batch_size
        self._client = None  # lazy
        self.dimension = self.DIMENSION

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        try:
            import voyageai  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "voyageai not installed. `pip install voyageai` to enable VoyageEmbedder."
            ) from exc
        self._client = voyageai.Client(api_key=self._api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        self._ensure_client()
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            resp = self._client.embed(  # type: ignore[union-attr]
                texts=batch, model=self._model, input_type="document"
            )
            all_vecs.extend(resp.embeddings)
        out = np.asarray(all_vecs, dtype=np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


# ---------------------------------------------------------------------------
# Health check (DATA-2B.7)
# ---------------------------------------------------------------------------


# Two probe strings deliberately chosen to be:
# - non-empty (so HashEmbedder produces non-zero vectors)
# - vocabulary-disjoint (so the cosine between them stays well below 1.0,
#   confirming the embedder isn't returning a constant vector)
_PROBE_TEXTS: tuple[str, str] = (
    "gold price market structure liquidity sweep",
    "central bank policy rate decision macro narrative",
)


class EmbedderHealthError(RuntimeError):
    """Raised when a probe detects a contract violation.

    Distinct from generic ``RuntimeError`` so callers can ``except
    EmbedderHealthError`` without catching unrelated bugs.
    """


def embed_health_check(
    embedder: Embedder,
    *,
    expected_dimension: int | None = None,
    rtol: float = 1e-3,
) -> dict:
    """Probe an embedder and verify its output contract.

    Checks performed:

    - shape is ``(2, dimension)``
    - both rows are L2-normalised within ``rtol``
    - rows are not identical (would indicate a constant-vector bug)
    - if ``expected_dimension`` is given, asserts ``dimension`` matches

    Returns a dict suitable for direct embedding in the deep-health JSON
    body. Raises :class:`EmbedderHealthError` only when the contract is
    violated (the OBS-2B.2 wrapper catches and downgrades to ok=False).
    """
    if embedder is None:
        return {"configured": False, "ok": True}

    declared_dim = int(getattr(embedder, "dimension", 0))
    if declared_dim < 16:
        raise EmbedderHealthError(
            f"declared dimension {declared_dim} < 16 (collision tolerance floor)"
        )
    if expected_dimension is not None and declared_dim != expected_dimension:
        raise EmbedderHealthError(
            f"dimension mismatch: embedder={declared_dim}, "
            f"expected={expected_dimension}"
        )

    t0 = time.perf_counter()
    vectors = embedder.embed(list(_PROBE_TEXTS))
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

    if vectors.shape != (2, declared_dim):
        raise EmbedderHealthError(
            f"embed() shape {vectors.shape} != (2, {declared_dim})"
        )

    norms = np.linalg.norm(vectors, axis=1)
    if not np.all(np.isfinite(norms)):
        raise EmbedderHealthError("non-finite norms in embedding output")
    if not np.allclose(norms, 1.0, rtol=rtol, atol=rtol):
        raise EmbedderHealthError(
            f"vectors are not L2-normalised: norms={norms.tolist()}"
        )

    cos = float(np.dot(vectors[0], vectors[1]))
    # Vocab-disjoint probe should map to clearly different vectors.
    # 0.99 is a paranoid ceiling; identical vectors give 1.0.
    if cos > 0.99:
        raise EmbedderHealthError(
            f"probe vectors collinear (cos={cos:.4f}) — embedder may be "
            "returning a constant vector"
        )

    return {
        "configured": True,
        "ok": True,
        "dimension": declared_dim,
        "embedder_class": type(embedder).__name__,
        "sample_norms": [round(float(n), 6) for n in norms],
        "sample_cosine": round(cos, 6),
        "duration_ms": elapsed_ms,
    }
