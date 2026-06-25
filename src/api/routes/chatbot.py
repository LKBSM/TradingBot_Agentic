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

import logging
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.entitlements import enforce_chat_access
from src.api.session_auth import optional_account

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
    # Freemium gate (no-op while the gate is OFF): the free tier gets a small
    # daily message quota; subscribers/owner are unlimited. Counting the turn
    # BEFORE we run it keeps the quota authoritative server-side. 402 on exhaust.
    enforce_chat_access(request, account)

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
