"""Chatbot endpoint — Chantier 4 — niveau 1.5 strict conversational layer.

POST /api/chatbot/message
  → 200 + ChatbotMessageResponse (content, blocked_reason, tool_calls_made)
  → 422 on invalid body (empty / too-long message, history too long)
  → 503 if the Chatbot was not bootstrapped (CHATBOT_ENABLED=false)
  → 500 on an unexpected internal error (detail never leaked)

The Chatbot lives on ``app.state.app_state.chatbot`` — consistent with how the
MarketReading endpoint reads ``app.state.app_state.market_reading_assembler``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.subscription_gate import enforce_access
from src.api.session_auth import optional_account
from src.intelligence.chatbot.constants import LLM_ERROR_TEMPLATE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])

MAX_MESSAGE_LENGTH = 2000
MAX_HISTORY_ITEMS = 20


class ConversationMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class ChatbotMessageRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_history: list[ConversationMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_ITEMS
    )


class ChatbotMessageResponse(BaseModel):
    content: str
    blocked_reason: Optional[str] = None
    tool_calls_made: list[dict[str, Any]] = Field(default_factory=list)
    # Display-only chart view actions (Couche 4 whitelist). The webapp applies
    # these to the chart RENDER only — they never touch detection. Empty on a
    # plain conversational turn.
    view_actions: list[dict[str, Any]] = Field(default_factory=list)


@router.post("/message", response_model=ChatbotMessageResponse)
async def chatbot_message(
    payload: ChatbotMessageRequest,
    request: Request,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> ChatbotMessageResponse:
    # Paid-only gate (no-op while the gate is OFF): the chat requires an active
    # subscription; subscribers/owner are unlimited, an unsubscribed account is
    # invited to subscribe (402), never errored (PAY-1).
    enforce_access(request, account)

    chatbot = getattr(request.app.state.app_state, "chatbot", None)
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot service not configured")

    try:
        response = chatbot.chat(
            user_message=payload.user_message,
            conversation_history=[m.model_dump() for m in payload.conversation_history],
        )
    except Exception:
        # Log the real cause server-side; never leak internals to the client.
        logger.exception("chatbot.chat failed")
        raise HTTPException(status_code=500, detail="Internal chatbot error")

    return ChatbotMessageResponse(
        content=response.content,
        blocked_reason=response.blocked_reason,
        tool_calls_made=response.tool_calls_made,
        view_actions=getattr(response, "view_actions", []),
    )


def _sse(event: dict[str, Any]) -> str:
    """Encode one orchestration event as an SSE frame."""
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/stream")
async def chatbot_stream(
    payload: ChatbotMessageRequest,
    request: Request,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> StreamingResponse:
    """MIA-1 — Server-Sent Events variant of ``/message``.

    Streams the SAME ``Chatbot.chat_events`` the blocking endpoint drains, so
    every defence layer runs identically (Couche 1 before the model, Couche 4 on
    each view action, Couche 3 on the COMPLETE text before the terminal
    ``answer`` event). The stream carries only fixed status signals + the final
    validated answer — never unvalidated model prose. Access is gated up front
    (same 402/503 posture as ``/message``); once the body starts, a mid-turn
    failure still terminates with a fail-safe ``answer`` event so the client
    always resolves.
    """
    enforce_access(request, account)

    chatbot = getattr(request.app.state.app_state, "chatbot", None)
    if chatbot is None:
        raise HTTPException(status_code=503, detail="Chatbot service not configured")

    user_message = payload.user_message
    history = [m.model_dump() for m in payload.conversation_history]

    def event_stream():
        try:
            for event in chatbot.chat_events(
                user_message=user_message, conversation_history=history
            ):
                yield _sse(event)
        except Exception:
            # Never leak internals; always give the client a terminal answer so
            # it can resolve and render the fail-safe template.
            logger.exception("chatbot.chat_events failed")
            yield _sse(
                {
                    "event": "answer",
                    "content": LLM_ERROR_TEMPLATE,
                    "tool_calls_made": [],
                    "view_actions": [],
                    "blocked_reason": "llm_error",
                }
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Disable proxy buffering so each event flushes immediately (Nginx).
            "X-Accel-Buffering": "no",
        },
    )
