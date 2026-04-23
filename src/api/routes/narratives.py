"""Narrative endpoints — tier-gated LLM analysis retrieval."""

from __future__ import annotations

import logging
import re
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import require_api_key, TESTING_MODE
from src.api.models import (
    NarrativeResponse,
    ChatRequest,
    ChatResponse,
    ScannerStatusResponse,
)
from src.intelligence.security import sanitize_string

logger = logging.getLogger(__name__)

# Signal ID format: 8-12 hex/alphanumeric chars (uuid4 prefix)
SIGNAL_ID_PATTERN = re.compile(r"^[a-f0-9\-]{8,36}$")

router = APIRouter(prefix="/api/v1", tags=["narratives"])


# =============================================================================
# GET /api/v1/narratives/{signal_id}
# =============================================================================

@router.get(
    "/narratives/{signal_id}",
    response_model=NarrativeResponse,
    responses={
        404: {"description": "Signal not found"},
        403: {"description": "Tier insufficient for narrative access"},
    },
)
async def get_narrative(
    signal_id: str,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """
    Retrieve the narrative for a signal, gated by subscription tier.

    - FREE: confluence_score + entry/SL/TP only (no narrative text)
    - ANALYST: + validation_reason
    - STRATEGIST/INSTITUTIONAL: + full_narrative, key_confluences, risk_warnings
    """
    # Validate signal_id format to prevent injection
    if not SIGNAL_ID_PATTERN.match(signal_id):
        raise HTTPException(status_code=400, detail="Invalid signal ID format")

    store = request.app.state.app_state.signal_store
    record = store.get_by_id(signal_id)

    if record is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")

    tier = subscriber.get("tier", "FREE")

    # Base response (all tiers)
    response = NarrativeResponse(
        signal_id=record.signal_id,
        symbol=record.symbol,
        action=record.action,
        entry_price=record.entry_price,
        stop_loss=record.stop_loss,
        take_profit=record.take_profit,
        rr_ratio=record.rr_ratio,
        confluence_score=record.confluence_score,
        market_context=record.market_context,
    )

    # In testing mode or ANALYST+: add validation
    if TESTING_MODE or tier in ("ANALYST", "STRATEGIST", "INSTITUTIONAL"):
        response.validation_reason = record.validation_reason or ""

    # In testing mode or STRATEGIST+: add full narrative
    if TESTING_MODE or tier in ("STRATEGIST", "INSTITUTIONAL"):
        response.full_narrative = record.narrative or ""
        response.key_confluences = record.key_confluences or ""
        response.risk_warnings = record.risk_warnings or ""

    return response


# =============================================================================
# POST /api/v1/narratives/chat
# =============================================================================

@router.post(
    "/narratives/chat",
    response_model=ChatResponse,
    responses={403: {"description": "Chat requires Institutional tier"}},
)
async def chat_about_signal(
    body: ChatRequest,
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """
    Ask a question about a signal (Institutional tier only).

    Uses Claude to answer contextual questions about the signal's
    market setup, risk profile, or alternative scenarios.
    """
    tier = subscriber.get("tier", "FREE")
    if not TESTING_MODE and tier != "INSTITUTIONAL":
        raise HTTPException(
            status_code=403,
            detail="Chat feature requires Institutional tier ($149/mo)",
        )

    # Validate signal_id format
    if not SIGNAL_ID_PATTERN.match(body.signal_id):
        raise HTTPException(status_code=400, detail="Invalid signal ID format")

    store = request.app.state.app_state.signal_store
    record = store.get_by_id(body.signal_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Signal {body.signal_id} not found")

    # Get LLM engine from app state
    llm_engine = getattr(request.app.state.app_state, "llm_engine", None)
    if llm_engine is None:
        raise HTTPException(status_code=503, detail="Chat service unavailable")

    # Check LLM circuit breaker
    app_state = request.app.state.app_state
    llm_breaker = app_state.circuit_breakers.get("llm") if app_state.circuit_breakers else None
    if llm_breaker is not None:
        from src.intelligence.circuit_breaker import CircuitState
        if llm_breaker.state == CircuitState.OPEN:
            raise HTTPException(
                status_code=503,
                detail="Chat service temporarily unavailable (circuit open)",
            )

    try:
        # Sanitize user question to prevent prompt injection
        sanitized_question = sanitize_string(body.question, max_length=500)

        # Build context from the signal record
        context = (
            f"Signal: {record.action} {record.symbol} at {record.entry_price}, "
            f"SL={record.stop_loss}, TP={record.take_profit}, R:R={record.rr_ratio}, "
            f"Score={record.confluence_score}. "
            f"Narrative: {record.narrative or 'N/A'}"
        )
        prompt = f"Context: {context}\n\nUser question: {sanitized_question}"

        response = llm_engine._call_api(llm_engine._narrator_model, prompt)
        return ChatResponse(
            signal_id=body.signal_id,
            question=sanitized_question,
            answer=response["text"],
            cost_usd=response["cost"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Chat error: %s", e)
        raise HTTPException(status_code=500, detail="Chat processing failed")


# =============================================================================
# GET /api/v1/scanner/status
# =============================================================================

@router.get("/scanner/status", response_model=ScannerStatusResponse)
async def get_scanner_status(
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """Return scanner health and stats."""
    scanner = getattr(request.app.state.app_state, "scanner", None)

    if scanner is None:
        return ScannerStatusResponse(
            running=False,
            uptime_seconds=0,
            bars_scanned=0,
            signals_generated=0,
            cache_hits=0,
            llm_calls=0,
            errors=0,
        )

    stats = scanner.get_stats()
    return ScannerStatusResponse(**stats)
