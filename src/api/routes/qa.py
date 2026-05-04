"""Q&A endpoint over the curated RAG corpus — Sprint LLM-2B.5.

POST /api/v1/qa
---------------
Accepts a natural-language question, runs the RAG pipeline against the
curated 50-source registry from LLM-2B.2, and returns:

- ``answer``     — the LLM's response, or a deterministic stub when no
                   LLM is wired in (CI / no-key environments).
- ``sources``    — the top-k cited chunks with their metadata.
- ``stub_mode``  — true when the answer was built without a real LLM.
- ``elapsed_ms`` — per-stage retrieval/assembly timings.

Tier policy (UE 2024/2811 + cost discipline)
-------------------------------------------
- TESTING_MODE: any tier, full RAG with LLM if present.
- FREE/ANALYST : retrieval + stub answer (no LLM tokens spent).
- STRATEGIST+ : retrieval + real LLM answer (when ``app.state.rag_llm``
  is configured).

The endpoint is rate-limited by the global per-IP middleware. The query
is sanitized to defeat prompt-injection attempts.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import TESTING_MODE, require_api_key
from src.api.disclaimers import get_disclaimer
from src.api.models import QARequest, QAResponse, QASource
from src.intelligence.security import sanitize_string

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["qa"])


# Tiers allowed to consume the real LLM. Anything below gets the stub.
LLM_TIERS = {"STRATEGIST", "INSTITUTIONAL"}


def _stub_answer(query: str, retrieved: list, language: str) -> str:
    """Build a deterministic stub answer from retrieved chunk metadata.

    Used when no real LLM is plugged in (CI, no API key). Returns a
    concise extract-and-cite-style response so callers can see the
    retrieval working without paying for tokens.
    """
    if not retrieved:
        if language == "fr":
            return "Information insuffisante pour répondre."
        return "Insufficient information to answer."

    if language == "fr":
        header = f"Synthèse à partir de {len(retrieved)} sources curées :\n\n"
    else:
        header = f"Summary from {len(retrieved)} curated sources:\n\n"

    body_lines = []
    for rc in retrieved[:3]:
        meta = rc.chunk.metadata or {}
        label = meta.get("label", rc.chunk.source_id)
        snippet = rc.chunk.text[:240].rstrip()
        if len(rc.chunk.text) > 240:
            snippet += "..."
        body_lines.append(f"[source:{rc.chunk.source_id}] ({label}) {snippet}")
    return header + "\n\n".join(body_lines)


@router.post(
    "/qa",
    response_model=QAResponse,
    responses={
        400: {"description": "Invalid query"},
        503: {"description": "RAG service unavailable"},
    },
)
async def qa(
    body: QARequest,
    request: Request,
    subscriber: dict = Depends(require_api_key),
) -> QAResponse:
    pipeline = getattr(request.app.state.app_state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not configured")

    sanitized_query = sanitize_string(body.query, max_length=1000)
    if len(sanitized_query.strip()) < 3:
        raise HTTPException(status_code=400, detail="Query too short after sanitization")

    tier = subscriber.get("tier", "FREE")
    use_llm = TESTING_MODE or tier in LLM_TIERS
    rag_llm: Optional[Any] = getattr(request.app.state.app_state, "rag_llm", None)
    llm_callable = rag_llm if (use_llm and rag_llm is not None) else None

    # Route mutation: change pipeline language per-request without mutating
    # the long-lived index. RAGPipeline.query() picks language up from
    # ``self.language``, so we briefly swap it under the GIL.
    saved_lang = pipeline.language
    saved_final_k = pipeline.final_k
    pipeline.language = body.language
    pipeline.final_k = body.top_k
    try:
        response = pipeline.query(sanitized_query, llm=llm_callable)
    finally:
        pipeline.language = saved_lang
        pipeline.final_k = saved_final_k

    answer = response.answer
    stub_mode = llm_callable is None
    if stub_mode:
        answer = _stub_answer(sanitized_query, response.retrieved, body.language)

    sources = []
    for rc in response.retrieved:
        meta = rc.chunk.metadata or {}
        sources.append(
            QASource(
                source_id=rc.chunk.source_id,
                label=str(meta.get("label", rc.chunk.source_id)),
                type=str(meta.get("type", "unknown")),
                ref=str(meta.get("ref", "")),
                authority_score=int(meta.get("authority_score", 5)),
                fused_score=float(rc.fused_score),
            )
        )

    elapsed_ms = {k: round(v * 1000, 2) for k, v in response.elapsed_seconds.items()}

    disclaimer = get_disclaimer(body.language)

    return QAResponse(
        query=sanitized_query,
        language=body.language,
        answer=answer,
        stub_mode=stub_mode,
        sources=sources,
        elapsed_ms=elapsed_ms,
        disclaimer=disclaimer,
    )


def build_default_rag_pipeline():
    """Bootstrap a HashEmbedder-backed pipeline with the curated 50 sources.

    Lives here (not in ``app.py``) so callers that want a different embedder
    in production can replace this without touching the API factory.
    """
    from src.intelligence.rag import HashEmbedder, RAGPipeline
    from src.intelligence.rag.sources import all_chunks

    pipe = RAGPipeline(embedder=HashEmbedder(dimension=512, seed=1))
    pipe.ingest(all_chunks())
    logger.info("Default RAG pipeline bootstrapped with %d chunks", pipe.size)
    return pipe
