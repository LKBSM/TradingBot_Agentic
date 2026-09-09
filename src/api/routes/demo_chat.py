"""MIA-4S — public showcase endpoint for the landing page's simulated M.I.A.

    POST /api/demo/chat         → 200 + DemoChatResponse
    POST /api/demo/chat/stream  → 200 + SSE (same events as the product chat)
    → 422 on an invalid body
    → 429 when a quota is reached (session / IP / daily budget)
    → 503 when the demo agent is not configured (DEMO_CHAT_ENABLED off, no key)

Why this route exists instead of reusing ``/api/chatbot/*``
----------------------------------------------------------
The product chat is gated by ``enforce_access`` — it answers an account. This
one answers ANYONE, with no session and no card, which changes the threat model
entirely: an unmetered public LLM endpoint is a bill and a shared-quota risk
before it is a feature. So the quotas, the logging and the budget brake live
here, at the only door that is open to the street.

Honesty about the quotas
------------------------
They are in-memory and PER PROCESS, exactly like ``AuthThrottle`` (whose
sliding-window primitive they reuse — three instances of one reviewed class,
no new limiter code). Multi-worker deployments therefore multiply them by the
worker count. This is a brake sized for a showcase, not a billing-grade quota;
a shared store (Redis) is the next layer if the landing ever gets real traffic.

Logging carries NO personal data: the client IP is salted-hashed and truncated,
and the visitor's question text is never written — only its length.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import time
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.auth_throttle import AuthThrottle, client_ip
from src.intelligence.chatbot.constants import LLM_ERROR_TEMPLATE

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/demo", tags=["demo"])

# A showcase question is a sentence, not an essay. Small on purpose: it bounds
# both the prompt-injection surface and the per-turn input cost.
MAX_MESSAGE_LENGTH = 600
MAX_HISTORY_ITEMS = 12

SESSION_COOKIE = "mia_demo_sid"

# Quotas — every one env-tunable, every one a deliberate number.
#  · session: what a curious visitor needs to form an opinion. Past that, the
#    honest answer is "subscribe and get the real thing", not more free tokens.
#  · IP/hour: a human never reaches it; a script reaches it in seconds.
#  · daily: the actual budget brake. At ~0.005 $/message this caps the landing's
#    LLM spend at roughly 10 $/day even if every other limit is circumvented.
DEFAULT_MAX_PER_SESSION = 6
DEFAULT_SESSION_WINDOW_S = 2 * 3600
DEFAULT_MAX_PER_IP_HOUR = 20
DEFAULT_MAX_PER_DAY = 2000


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# The per-session cap is a sliding window too (a visitor who comes back tomorrow
# is a new visitor), so all three reuse the same audited primitive.
_session_quota = AuthThrottle(
    max_attempts=_int_env("DEMO_MAX_PER_SESSION", DEFAULT_MAX_PER_SESSION),
    window_s=_int_env("DEMO_SESSION_WINDOW_S", DEFAULT_SESSION_WINDOW_S),
)
_ip_quota = AuthThrottle(
    max_attempts=_int_env("DEMO_MAX_PER_IP_HOUR", DEFAULT_MAX_PER_IP_HOUR),
    window_s=3600,
)
_daily_budget = AuthThrottle(
    max_attempts=_int_env("DEMO_MAX_PER_DAY", DEFAULT_MAX_PER_DAY),
    window_s=24 * 3600,
)

# {zone id → demo layer} for the frozen scenario, built once on first use.
_ZONE_LAYERS: Optional[Dict[str, str]] = None

# Per-process salt unless one is provided: without a stable salt the hashes are
# not even correlatable across restarts, which is the privacy-preserving default.
_LOG_SALT = os.environ.get("DEMO_LOG_SALT") or secrets.token_hex(8)

# What the visitor is told when a quota is reached. Free of forbidden tokens, so
# these never trip Couche 3 — same discipline as the production templates.
QUOTA_TEMPLATES: Dict[str, str] = {
    "session_limit": (
        "On s'arrête ici pour la démonstration : elle est limitée à quelques "
        "questions par visiteur. Ce que tu viens de voir tourne sur un scénario "
        "figé — dans le produit, M.I.A lit les marchés réels et la conversation "
        "n'est pas limitée."
    ),
    "ip_limit": (
        "Beaucoup de questions sont arrivées depuis cette connexion en peu de "
        "temps. La démonstration se met en pause un moment."
    ),
    "daily_budget": (
        "La démonstration a atteint son quota du jour. Les onglets de la page "
        "restent utilisables, et le produit, lui, n'est pas concerné."
    ),
}


class DemoMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)


class DemoChatRequest(BaseModel):
    user_message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    conversation_history: List[DemoMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_ITEMS
    )
    # The landing ships in nine languages. An unknown value falls back to French
    # in the registry rather than being rejected — a bad locale is not an error
    # worth showing a visitor.
    locale: Optional[str] = Field(default=None, max_length=8)


class DemoChatResponse(BaseModel):
    content: str
    blocked_reason: Optional[str] = None
    # Display-only actions, already through Couche 4 AND narrowed to what the
    # frozen demo chart can actually render (never a claim the demo can't honour).
    view_actions: List[Dict[str, Any]] = Field(default_factory=list)
    messages_left: int = 0


def _hashed_ip(ip: str) -> str:
    return hashlib.sha256(f"{_LOG_SALT}:{ip}".encode()).hexdigest()[:12]


def _session_id(request: Request) -> tuple[str, bool]:
    """Return (session id, is_new). Opaque random value, no personal data."""
    existing = request.cookies.get(SESSION_COOKIE)
    if existing and 16 <= len(existing) <= 64 and existing.isascii():
        return existing, False
    return secrets.token_urlsafe(16), True


def _set_session_cookie(response: Response, sid: str) -> None:
    response.set_cookie(
        SESSION_COOKIE,
        sid,
        max_age=_int_env("DEMO_SESSION_WINDOW_S", DEFAULT_SESSION_WINDOW_S),
        httponly=True,
        samesite="lax",
        secure=os.environ.get("DEMO_COOKIE_SECURE", "1") != "0",
        path="/api/demo",
    )


def _agent(request: Request, locale: Optional[str]) -> Any:
    registry = getattr(request.app.state.app_state, "demo_chatbot", None)
    if registry is None:
        # 503 is the signal the landing degrades on: it falls back to the
        # scripted starters rather than showing a broken tab.
        raise HTTPException(status_code=503, detail="Demo agent not configured")
    return registry.for_locale(locale)


def _check_quotas(sid: str, ip: str) -> None:
    """Raise 429 with a spoken reason, cheapest/broadest check first.

    Ordered so an abusive burst is stopped by the daily brake before it can even
    consume the per-IP window, and so NO quota check ever costs an LLM call.
    """
    for reason, quota, key in (
        ("daily_budget", _daily_budget, "global"),
        ("ip_limit", _ip_quota, ip),
        ("session_limit", _session_quota, sid),
    ):
        allowed, retry_after = quota.hit(key)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    "reason": reason,
                    "message": QUOTA_TEMPLATES[reason],
                    "retry_after": retry_after,
                },
            )


def _messages_left(sid: str) -> int:
    blocked, _ = _session_quota.blocked(sid)
    if blocked:
        return 0
    max_per_session = _int_env("DEMO_MAX_PER_SESSION", DEFAULT_MAX_PER_SESSION)
    used = len(_session_quota._hits.get(sid, ()))  # noqa: SLF001 — read-only count
    return max(0, max_per_session - used)


def _demo_view_actions(actions: Any) -> List[Dict[str, Any]]:
    """Keep only the actions the frozen demo chart can render.

    Couche 4 has already rejected anything off the product whitelist and any
    invented zone id; this second pass is not a security layer, it is honesty:
    the demo must not report a view change it cannot show.
    """
    from src.intelligence.chatbot.demo_agent import (
        DEMO_VIEW_ACTIONS,
        DEMO_VIEW_ZONE_ACTIONS,
        load_illustration,
        zone_layer_index,
    )

    global _ZONE_LAYERS
    if _ZONE_LAYERS is None:
        _ZONE_LAYERS = zone_layer_index(load_illustration())

    kept: List[Dict[str, Any]] = []
    for a in actions or []:
        if not isinstance(a, dict):
            continue
        name = a.get("action")
        if name not in DEMO_VIEW_ACTIONS:
            continue
        if name in DEMO_VIEW_ZONE_ACTIONS:
            params = a.get("params") or {}
            ids = params.get("zone_ids") if isinstance(params, dict) else None
            if ids is None:
                # show_zones with no target = restore everything; that renders.
                kept.append(a)
                continue
            layers = sorted({_ZONE_LAYERS[i] for i in ids if i in _ZONE_LAYERS})
            if not layers:
                # Real zone ids the demo chart cannot express as layers: drop the
                # action rather than let the answer describe a change nobody sees.
                continue
            kept.append({**a, "demo_layers": layers})
            continue
        kept.append(a)
    return kept


def _log_turn(
    *,
    ip: str,
    sid: str,
    question_len: int,
    started: float,
    blocked_reason: Optional[str],
    tool_calls: Any,
    stream: bool,
) -> None:
    """Minimal usage record — no personal data, and never the question text."""
    logger.info(
        "demo_chat ip=%s sid=%s qlen=%d tools=%s blocked=%s stream=%s ms=%d left=%d",
        _hashed_ip(ip),
        sid[:6],
        question_len,
        ",".join(sorted({c.get("name", "?") for c in (tool_calls or []) if isinstance(c, dict)}))
        or "-",
        blocked_reason or "-",
        int(stream),
        int((time.monotonic() - started) * 1000),
        _messages_left(sid),
    )


@router.post("/chat", response_model=DemoChatResponse)
async def demo_chat(
    payload: DemoChatRequest, request: Request, response: Response
) -> DemoChatResponse:
    agent = _agent(request, payload.locale)
    sid, is_new = _session_id(request)
    if is_new:
        _set_session_cookie(response, sid)
    ip = client_ip(request)
    _check_quotas(sid, ip)

    started = time.monotonic()
    try:
        result = agent.chat(
            user_message=payload.user_message,
            conversation_history=[m.model_dump() for m in payload.conversation_history],
        )
    except Exception:
        logger.exception("demo_chat failed")
        raise HTTPException(status_code=500, detail="Internal demo chat error")

    _log_turn(
        ip=ip,
        sid=sid,
        question_len=len(payload.user_message),
        started=started,
        blocked_reason=result.blocked_reason,
        tool_calls=result.tool_calls_made,
        stream=False,
    )
    return DemoChatResponse(
        content=result.content,
        blocked_reason=result.blocked_reason,
        view_actions=_demo_view_actions(getattr(result, "view_actions", [])),
        messages_left=_messages_left(sid),
    )


def _sse(event: Dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False, default=str)}\n\n"


@router.post("/chat/stream")
async def demo_chat_stream(payload: DemoChatRequest, request: Request) -> StreamingResponse:
    """SSE variant — drains the SAME ``chat_events`` generator as the product.

    Every defence layer therefore runs identically: Couche 1 before the model,
    Couche 4 on each view action, Couche 3 on the COMPLETE text before the
    terminal ``answer``. The stream carries status signals and the final
    validated answer — never unvalidated model prose.
    """
    agent = _agent(request, payload.locale)
    sid, is_new = _session_id(request)
    ip = client_ip(request)
    _check_quotas(sid, ip)

    user_message = payload.user_message
    history = [m.model_dump() for m in payload.conversation_history]
    started = time.monotonic()

    def event_stream():
        blocked: Optional[str] = None
        tools: Any = []
        try:
            for event in agent.chat_events(
                user_message=user_message, conversation_history=history
            ):
                if event.get("event") == "answer":
                    blocked = event.get("blocked_reason")
                    tools = event.get("tool_calls_made")
                    event = {
                        **event,
                        "view_actions": _demo_view_actions(event.get("view_actions")),
                        "messages_left": _messages_left(sid),
                    }
                yield _sse(event)
        except Exception:
            logger.exception("demo_chat_events failed")
            blocked = "llm_error"
            yield _sse(
                {
                    "event": "answer",
                    "content": LLM_ERROR_TEMPLATE,
                    "tool_calls_made": [],
                    "view_actions": [],
                    "blocked_reason": "llm_error",
                    "messages_left": _messages_left(sid),
                }
            )
        finally:
            _log_turn(
                ip=ip,
                sid=sid,
                question_len=len(user_message),
                started=started,
                blocked_reason=blocked,
                tool_calls=tools,
                stream=True,
            )

    stream = StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    if is_new:
        _set_session_cookie(stream, sid)
    return stream
