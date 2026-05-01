"""Document chunking — fixed-size token windows with overlap.

Per LLM-2B.1 spec: 500 tokens per chunk, 100-token overlap. We use a
whitespace tokenizer (≈ 0.75 words = 1 token in English/French) which
overestimates token count slightly compared to BPE — that's deliberate,
keeps chunks safely under any LLM context limit.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Iterable


# Default chunk parameters per plan. Concrete numbers tuned on Markdown /
# institutional report style; adjustable per-source via Chunker(...).
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_OVERLAP_TOKENS = 100

# Whitespace tokenizer — coarse, deterministic, no dependencies. Good
# enough for chunk-boundary purposes.
_TOKEN_RE = re.compile(r"\S+")


@dataclass
class Chunk:
    """One unit of retrievable text + provenance.

    The `chunk_id` is content-addressable (hash of text + source_id) so
    the vector store and BM25 index agree on identity even after
    re-chunking — useful for incremental ingestion.
    """

    text: str
    source_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""

    def __post_init__(self) -> None:
        if not self.chunk_id:
            payload = f"{self.source_id}::{self.chunk_index}::{self.text}".encode("utf-8")
            self.chunk_id = hashlib.sha1(payload).hexdigest()[:16]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def chunk_text(
    text: str,
    source_id: str,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    metadata: dict | None = None,
) -> list[Chunk]:
    """Split `text` into overlapping Chunks.

    Returns at least one chunk for any non-empty input. Pure function —
    deterministic for the same input.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_tokens)")

    tokens = _tokenize(text)
    if not tokens:
        return []

    metadata = metadata or {}
    step = chunk_tokens - overlap_tokens
    chunks: list[Chunk] = []
    idx = 0
    cursor = 0
    while cursor < len(tokens):
        window = tokens[cursor : cursor + chunk_tokens]
        chunk = Chunk(
            text=" ".join(window),
            source_id=source_id,
            chunk_index=idx,
            metadata=dict(metadata),
        )
        chunks.append(chunk)
        idx += 1
        if cursor + chunk_tokens >= len(tokens):
            break
        cursor += step
    return chunks


def chunk_documents(
    docs: Iterable[tuple[str, str, dict]],
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Chunk a stream of (source_id, text, metadata) tuples.

    Convenience wrapper for ingesting curated source corpora (LLM-2B.2).
    """
    out: list[Chunk] = []
    for source_id, text, meta in docs:
        out.extend(chunk_text(text, source_id, chunk_tokens, overlap_tokens, meta))
    return out
