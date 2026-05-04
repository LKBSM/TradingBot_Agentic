"""B2B enrichment endpoint — Sprint INFRA-2B.5.

POST /api/v1/enrich
-------------------
Accepts a partial signal context from a broker (instrument + timeframe +
direction + optional levels + optional natural-language hint) and returns
an ``InsightSignalV2`` payload enriched with:

- a RAG-sourced ``narrative_long`` (when an LLM is wired in) or a
  deterministic stub from retrieved chunks,
- ``sources_cited`` populated from the curated 50-source registry,
- a ``conviction_0_100`` heuristic derived from the supplied
  reward/risk geometry,
- ``compliance.is_paper_demo=True`` and ``edge_claim=False`` (Phase 2B
  defaults — narrative-first, no edge claim).

The B2B contract is the same Pydantic v2 model used by every B2C surface
(InsightSignalV2), per the dual-architecture design (UX-1.1).

Tier policy
-----------
- TESTING_MODE: any tier.
- STRATEGIST / INSTITUTIONAL : real LLM when ``rag_llm`` is wired in.
- Lower tiers: stub narrative (no LLM tokens spent).
- The endpoint is intended for paying B2B clients; in Phase 2B it lives
  alongside the B2C QA endpoint and shares the same RAG pipeline.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import TESTING_MODE, require_api_key
from src.api.disclaimers import get_disclaimer
from src.api.insight_signal_v2 import (
    ComplianceMeta,
    InsightSignalV2,
    NarrativeLanguage,
    SetupDirection,
    SignalLevels,
    Source,
    SourceType,
    Timeframe,
)
from src.api.models import EnrichRequest
from src.intelligence.security import sanitize_string

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1", tags=["enrich"])


LLM_TIERS = {"STRATEGIST", "INSTITUTIONAL"}


def _build_query(req: EnrichRequest) -> str:
    """Compose a RAG query from the broker's structured signal + free text."""
    pieces = [
        f"{req.direction.replace('_', ' ').lower()} on {req.instrument} {req.timeframe}",
    ]
    if req.entry is not None and req.stop is not None:
        pieces.append(f"entry {req.entry} stop {req.stop}")
    if req.broker_context:
        pieces.append(req.broker_context.strip())
    pieces.append("relevant macro structural and volatility drivers")
    return " — ".join(pieces)


def _rr_ratio(req: EnrichRequest) -> Optional[float]:
    """Reward/risk ratio from the supplied levels, when computable."""
    if req.entry is None or req.stop is None or req.target_1 is None:
        return None
    try:
        risk = abs(req.entry - req.stop)
        reward = abs(req.target_1 - req.entry)
        if risk <= 0:
            return None
        return round(reward / risk, 3)
    except Exception:
        return None


def _conviction(req: EnrichRequest, n_sources: int) -> int:
    """Heuristic 0-100 conviction blending RR geometry + retrieval depth.

    NEUTRAL setups cap at 40 (no actionable thesis). With ≥3 retrieved
    sources and RR≥2 the bullish/bearish setups typically land 60-75.
    Final value is bucketed downstream by ``conviction_to_label``.
    """
    if req.direction == "NEUTRAL":
        # NEUTRAL is by construction non-actionable; bound below the
        # WEAK→MODERATE threshold to keep the rendering honest.
        return min(35, 25 + min(n_sources, 5) * 2)
    base = 40
    rr = _rr_ratio(req)
    if rr is not None:
        base += min(20, int(rr * 8))
    base += min(15, n_sources * 2)
    return max(0, min(100, base))


_NARRATIVE_SHORT_TEMPLATES = {
    "fr": "Setup {dir_label} {instrument} {tf}. Analyse algorithmique, pas un conseil.",
    "en": "{dir_label} setup {instrument} {tf}. Algorithmic analysis, not advice.",
    "de": "{dir_label} Setup {instrument} {tf}. Algorithmische Analyse, keine Beratung.",
    "es": "Setup {dir_label} {instrument} {tf}. Análisis algorítmico, no es asesoramiento.",
}

_DIRECTION_LABEL = {
    ("BULLISH_SETUP", "fr"): "haussier",
    ("BULLISH_SETUP", "en"): "Bullish",
    ("BULLISH_SETUP", "de"): "Bullishes",
    ("BULLISH_SETUP", "es"): "alcista",
    ("BEARISH_SETUP", "fr"): "baissier",
    ("BEARISH_SETUP", "en"): "Bearish",
    ("BEARISH_SETUP", "de"): "Bearisches",
    ("BEARISH_SETUP", "es"): "bajista",
    ("NEUTRAL", "fr"): "neutre",
    ("NEUTRAL", "en"): "Neutral",
    ("NEUTRAL", "de"): "Neutrales",
    ("NEUTRAL", "es"): "neutro",
}


def _narrative_short(req: EnrichRequest) -> str:
    template = _NARRATIVE_SHORT_TEMPLATES.get(
        req.language, _NARRATIVE_SHORT_TEMPLATES["en"]
    )
    return template.format(
        dir_label=_DIRECTION_LABEL.get((req.direction, req.language), req.direction),
        instrument=req.instrument,
        tf=req.timeframe,
    )


def _stub_narrative_long(retrieved: list, language: str) -> str:
    """Concatenated extracts from retrieved chunks as the long narrative."""
    if not retrieved:
        if language == "fr":
            return "Contexte de sources insuffisant pour étayer un récit."
        if language == "de":
            return "Quellenkontext unzureichend für eine Narrative."
        if language == "es":
            return "Contexto de fuentes insuficiente para un análisis narrativo."
        return "Insufficient source context for a narrative."
    parts = []
    for rc in retrieved[:3]:
        meta = rc.chunk.metadata or {}
        label = meta.get("label", rc.chunk.source_id)
        snippet = rc.chunk.text[:300].rstrip()
        if len(rc.chunk.text) > 300:
            snippet += "..."
        parts.append(f"[source:{rc.chunk.source_id}] ({label}) {snippet}")
    return "\n\n".join(parts)


def _to_sources(retrieved: list) -> list[Source]:
    """Map retrieved chunks → Source objects for the v2 payload."""
    out = []
    for rc in retrieved:
        meta = rc.chunk.metadata or {}
        try:
            stype = SourceType(meta.get("type", "education"))
        except ValueError:
            stype = SourceType.EDUCATION
        out.append(
            Source(
                type=stype,
                ref=str(meta.get("ref", "")),
                label=str(meta.get("label", rc.chunk.source_id)),
                quoted_excerpt=rc.chunk.text[:500],
            )
        )
    return out


@router.post(
    "/enrich",
    response_model=InsightSignalV2,
    responses={
        400: {"description": "Invalid payload"},
        503: {"description": "RAG service unavailable"},
    },
)
async def enrich(
    body: EnrichRequest,
    request: Request,
    subscriber: dict = Depends(require_api_key),
) -> InsightSignalV2:
    pipeline = getattr(request.app.state.app_state, "rag_pipeline", None)
    if pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not configured")

    # Safety on the free-form broker hint.
    if body.broker_context:
        body.broker_context = sanitize_string(body.broker_context, max_length=1000)

    tier = subscriber.get("tier", "FREE")
    use_llm = TESTING_MODE or tier in LLM_TIERS
    rag_llm = getattr(request.app.state.app_state, "rag_llm", None)
    llm_callable = rag_llm if (use_llm and rag_llm is not None) else None

    rag_query = _build_query(body)

    # Hold the pipeline language for the duration of the call (see qa.py for
    # rationale). Use a generous top_k for B2B since broker payloads pay for
    # depth.
    saved_lang = pipeline.language
    saved_k = pipeline.final_k
    pipeline.language = body.language
    pipeline.final_k = 8
    try:
        rag_response = pipeline.query(rag_query, llm=llm_callable)
    finally:
        pipeline.language = saved_lang
        pipeline.final_k = saved_k

    retrieved = rag_response.retrieved
    sources = _to_sources(retrieved)

    if llm_callable is not None and rag_response.answer:
        narrative_long = rag_response.answer
        stub_mode = False
    else:
        narrative_long = _stub_narrative_long(retrieved, body.language)
        stub_mode = True

    levels = SignalLevels(
        entry=body.entry,
        stop=body.stop,
        target_1=body.target_1,
        target_2=body.target_2,
    )
    # NEUTRAL must not carry levels per the v2 model_validator. Strip
    # silently rather than reject the broker payload.
    if body.direction == "NEUTRAL":
        levels = SignalLevels()

    # Pre-validate the directional invariants the v2 model enforces, so
    # broker mistakes surface as a clean 422 rather than a 500 from the
    # InsightSignalV2 constructor.
    if (
        body.direction == "BULLISH_SETUP"
        and body.entry is not None
        and body.stop is not None
        and body.stop >= body.entry
    ):
        raise HTTPException(
            status_code=422,
            detail="BULLISH_SETUP requires stop < entry",
        )
    if (
        body.direction == "BEARISH_SETUP"
        and body.entry is not None
        and body.stop is not None
        and body.stop <= body.entry
    ):
        raise HTTPException(
            status_code=422,
            detail="BEARISH_SETUP requires stop > entry",
        )

    payload = InsightSignalV2(
        id=body.client_request_id or str(uuid.uuid4()),
        instrument=body.instrument,
        timeframe=Timeframe(body.timeframe),
        direction=SetupDirection(body.direction),
        conviction_0_100=_conviction(body, len(sources)),
        levels=levels,
        narrative_short=_narrative_short(body),
        narrative_long=narrative_long,
        narrative_language=NarrativeLanguage(body.language),
        sources_cited=sources,
        compliance=ComplianceMeta(
            disclaimer_lang=NarrativeLanguage(body.language),
            edge_claim=False,
            is_paper_demo=True,
        ),
        created_at_utc=datetime.now(timezone.utc),
        extras={
            "stub_mode": stub_mode,
            "rr_ratio": _rr_ratio(body),
            "elapsed_ms": {
                k: round(v * 1000, 2) for k, v in rag_response.elapsed_seconds.items()
            },
            "disclaimer": get_disclaimer(body.language),
            "client_request_id": body.client_request_id,
        },
    )

    # DATA-2B.4 hash-chain: append the delivered insight to the audit ledger
    # if one is configured. The seq + entry_hash bubble back as extras so
    # the broker can store a verifiable receipt alongside their order.
    ledger = getattr(request.app.state.app_state, "audit_ledger", None)
    if ledger is not None:
        try:
            entry = ledger.append(payload)
            payload.extras["audit_seq"] = entry.seq
            payload.extras["audit_entry_hash"] = entry.entry_hash
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("audit ledger append failed: %s", exc)

    return payload
