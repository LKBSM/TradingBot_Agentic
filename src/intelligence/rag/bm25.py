"""Pure-Python BM25 sparse retriever.

LLM-2B.1: hybrid retrieval = BM25 + dense embeddings. BM25 catches exact
keyword matches (instrument tickers, level numbers, named entities) that
dense embeddings sometimes paraphrase away. Implementation follows the
canonical BM25 formulation (Robertson & Zaragoza 2009) with k1=1.5,
b=0.75 — the defaults in Lucene / rank_bm25.

No external dependency. Designed to be fast on the curated corpus scale
(LLM-2B.2 will index 50 sources × ~10 chunks each = 500 chunks).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.intelligence.rag.chunking import Chunk


_TOKEN_RE = re.compile(r"\b[\w']+\b")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class BM25Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """In-memory BM25 index over a Chunk corpus.

    Build once via `add(chunks)`; query as many times as you like.
    Re-`add()` calls incrementally extend the corpus, but DF tables are
    recomputed on `_finalise()` (called automatically on first `search`).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_freqs: list[Counter] = []
        self._doc_lens: list[int] = []
        self._df: Counter | None = None
        self._avgdl: float = 0.0
        self._N: int = 0

    @property
    def size(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for c in chunks:
            tokens = _tokenize(c.text)
            self._chunks.append(c)
            self._doc_tokens.append(tokens)
            tf = Counter(tokens)
            self._doc_freqs.append(tf)
            self._doc_lens.append(len(tokens))
        self._df = None  # invalidate; will recompute on next search

    def _finalise(self) -> None:
        df: Counter = Counter()
        for tf in self._doc_freqs:
            for term in tf:
                df[term] += 1
        self._df = df
        self._N = len(self._chunks)
        self._avgdl = (sum(self._doc_lens) / self._N) if self._N else 0.0

    def search(self, query: str, k: int = 5) -> list[BM25Hit]:
        """Return top-k chunks by BM25 score for ``query``."""
        if self._df is None:
            self._finalise()
        if self._N == 0:
            return []

        query_terms = _tokenize(query)
        scores = [0.0] * self._N
        for term in query_terms:
            df = self._df.get(term, 0) if self._df else 0
            if df == 0:
                continue
            # IDF: log((N - df + 0.5) / (df + 0.5) + 1) — the +1 keeps it
            # non-negative. Standard Robertson formulation.
            idf = math.log((self._N - df + 0.5) / (df + 0.5) + 1.0)
            for i, tf in enumerate(self._doc_freqs):
                f = tf.get(term, 0)
                if f == 0:
                    continue
                dl = self._doc_lens[i]
                norm = 1.0 - self.b + self.b * (dl / self._avgdl) if self._avgdl else 1.0
                tf_component = f * (self.k1 + 1) / (f + self.k1 * norm)
                scores[i] += idf * tf_component

        # Take top-k by score (descending), break ties by chunk_index.
        ranked = sorted(
            range(self._N),
            key=lambda i: (-scores[i], self._chunks[i].chunk_index),
        )
        return [
            BM25Hit(chunk=self._chunks[i], score=scores[i])
            for i in ranked[:k]
            if scores[i] > 0
        ]
