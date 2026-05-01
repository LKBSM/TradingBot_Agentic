"""In-memory dense vector store with cosine search.

For Phase 2B's curated corpus (~500 chunks), in-memory numpy is plenty
fast (cosine search over 500×1024 floats = ~2ms). Migration to ChromaDB /
Qdrant is straightforward later (same interface).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.intelligence.rag.chunking import Chunk


@dataclass
class DenseHit:
    chunk: Chunk
    score: float  # cosine similarity in [-1, 1]


class InMemoryVectorStore:
    """Stores Chunks alongside L2-normalised embedding rows.

    Cosine similarity reduces to dot product on L2-normalised vectors,
    which is what numpy/BLAS optimises best.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self._chunks: list[Chunk] = []
        self._embeddings: np.ndarray | None = None

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"chunks count {len(chunks)} != embeddings rows {embeddings.shape[0]}"
            )
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embeddings dim {embeddings.shape[1]} != store dim {self.dimension}"
            )
        # Defensive copy (caller may keep mutating)
        block = np.asarray(embeddings, dtype=np.float32)
        self._chunks.extend(chunks)
        if self._embeddings is None:
            self._embeddings = block.copy()
        else:
            self._embeddings = np.vstack([self._embeddings, block])

    def search(self, query_embedding: np.ndarray, k: int = 5) -> list[DenseHit]:
        if self._embeddings is None or self.size == 0:
            return []
        q = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        # Caller should pre-normalise; defensive normalisation here too.
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        scores = self._embeddings @ q  # shape (N,)
        if k >= len(scores):
            top = np.argsort(-scores)
        else:
            # argpartition is O(N), then sort the top-k
            unsorted_top = np.argpartition(-scores, k)[:k]
            top = unsorted_top[np.argsort(-scores[unsorted_top])]
        return [DenseHit(chunk=self._chunks[int(i)], score=float(scores[int(i)])) for i in top[:k]]
