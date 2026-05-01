"""
Retrieval-Augmented Generation (RAG) subsystem.

Sprint LLM-2B.1 (Aisha, 14h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie IV (Phase 2B) Agent 4.

Architecture
------------
- `chunking.py`     — split documents into 500-token chunks with 100 overlap
- `embedders.py`    — pluggable embedding backends (stub + Voyage adapter)
- `bm25.py`         — pure-Python BM25 sparse retriever
- `vector_store.py` — in-memory dense vector store (numpy cosine search)
- `pipeline.py`     — top-level RAG pipeline (hybrid retrieve → assemble)
- `prompts.py`      — anti-hallucination prompt templates
"""

from src.intelligence.rag.bm25 import BM25Index
from src.intelligence.rag.chunking import Chunk, chunk_text
from src.intelligence.rag.embedders import (
    Embedder,
    HashEmbedder,
    VoyageEmbedder,
)
from src.intelligence.rag.pipeline import RAGPipeline, RAGResponse, RetrievedChunk
from src.intelligence.rag.vector_store import InMemoryVectorStore

__all__ = [
    "BM25Index",
    "Chunk",
    "chunk_text",
    "Embedder",
    "HashEmbedder",
    "VoyageEmbedder",
    "InMemoryVectorStore",
    "RAGPipeline",
    "RAGResponse",
    "RetrievedChunk",
]
