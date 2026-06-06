"""Chantier 4 — Couche 2 — Chatbot orchestrator (Haiku + tool use, doc §4.1/§4.4).

Flow per chat():
  1. Couche 1 — adversarial input filter. A match short-circuits to the
     pedagogical refusal template with NO LLM call.
  2. Couche 2 — Haiku with a strict niveau 1.5 system prompt, the default
     signal_summary injected, and 2 tools (get_market_reading / get_signal_summary).
     Multi-turn tool use is bounded to MAX_TOOL_TURNS rounds.
  3. Couche 3 — output forbidden-token filter. **Wired in Étape 5.** For now the
     LLM text is returned as-is (a clearly-marked hook is left in place).

Fail-safe: any Anthropic exception, or exceeding the tool-turn budget, returns a
template (LLM_ERROR_TEMPLATE) instead of crashing or leaking a partial answer.

The Anthropic client is duck-typed (only ``client.messages.create(...)`` is
used) — consistent with HaikuDescriptionEngine / LLMNarrativeEngine, and trivial
to stub in tests.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
from src.intelligence.chatbot.constants import (
    LLM_ERROR_TEMPLATE,
    REFUSAL_TEMPLATE,
)
from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider

logger = logging.getLogger(__name__)

# Aligned with haiku_description_engine.py (Chantier 2) for inter-module coherence.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
MAX_TOOL_TURNS = 3  # hard cap on tool-use rounds to avoid infinite loops

SUPPORTED_INSTRUMENTS = ("XAUUSD", "EURUSD")
SUPPORTED_TIMEFRAMES = ("M15", "H1", "H4")


TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_market_reading",
        "description": (
            "Lecture complète et factuelle d'une combinaison instrument/timeframe "
            "(structure SMC, régime, news, conditions). À utiliser quand "
            "l'utilisateur demande des détails absents du contexte initial."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "instrument": {
                    "type": "string",
                    "enum": list(SUPPORTED_INSTRUMENTS),
                    "description": "XAUUSD ou EURUSD",
                },
                "timeframe": {
                    "type": "string",
                    "enum": list(SUPPORTED_TIMEFRAMES),
                    "description": "M15, H1 ou H4",
                },
            },
            "required": ["instrument", "timeframe"],
        },
    },
    {
        "name": "get_signal_summary",
        "description": (
            "Résumé condensé des 6 combinaisons suivies (XAUUSD/EURUSD × "
            "M15/H1/H4) : tendance, volatilité, phase, structure, news à venir."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


SYSTEM_PROMPT_TEMPLATE = """Tu es MIA Markets, un outil de compréhension des conditions de marché.

RÈGLES STRICTES :
- Tu décris les conditions observées, jamais ne recommandes une action.
- Tu n'utilises jamais : achète, vends, entre, sors, ouvre, ferme, place, long, short (en impératif), conseille, recommande, suggère, évite, devrais.
- Tu n'utilises jamais : risqué, sûr, dangereux, opportunité, bon moment, mauvais moment, setup parfait.
- Si l'utilisateur insiste pour un conseil, tu redis fermement que tu décris uniquement les conditions : « Je décris les conditions du marché. La décision d'agir t'appartient. »
- Tu n'inventes jamais de données — utilise toujours get_market_reading ou get_signal_summary.
- Tu réponds en français, par défaut concis (2-4 phrases sauf demande explicite de détail).

CONTEXTE INITIAL (signal_summary) :
{signal_summary}

Tu as accès à 2 tools :
- get_market_reading(instrument, timeframe) : lecture complète d'une combinaison.
- get_signal_summary() : résumé des 6 combinaisons (XAUUSD/EURUSD × M15/H1/H4).

Si l'utilisateur pose une question contextuelle nécessitant des détails absents du signal_summary, appelle get_market_reading."""


@dataclass
class ChatResponse:
    """Result of a chatbot turn.

    Attributes:
        content: the text shown to the user (LLM answer, refusal, or fallback).
        tool_calls_made: list of {"name", "input"} for each tool executed.
        blocked_reason: None on a normal answer; otherwise the reason
            (adversarial category, "llm_error", "max_tool_turns_exceeded").
    """

    content: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    blocked_reason: Optional[str] = None


class Chatbot:
    """Niveau 1.5 strict conversational orchestrator (Couche 2)."""

    def __init__(
        self,
        anthropic_client: Any,
        summary_provider: SignalSummaryProvider,
        assembler: Any,
        adversarial_filter: Optional[AdversarialFilter] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_turns: int = MAX_TOOL_TURNS,
    ) -> None:
        self._client = anthropic_client
        self._summary_provider = summary_provider
        self._assembler = assembler
        self._adv_filter = adversarial_filter or AdversarialFilter()
        self._model = model
        self._max_tokens = max_tokens
        self._max_tool_turns = max_tool_turns

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        history = list(conversation_history or [])

        # --- Couche 1 — adversarial input filter (no LLM call on match) ---
        adv = self._adv_filter.check(user_message)
        if adv.triggered:
            return ChatResponse(
                content=REFUSAL_TEMPLATE,
                tool_calls_made=[],
                blocked_reason=adv.category,
            )

        # --- Couche 2 — Haiku + tool use ---
        signal_summary = self._safe_summary()
        system = SYSTEM_PROMPT_TEMPLATE.format(
            signal_summary=json.dumps(signal_summary, ensure_ascii=False, indent=2)
        )
        messages: list[dict[str, Any]] = history + [
            {"role": "user", "content": user_message}
        ]
        tool_calls_made: list[dict[str, Any]] = []

        for _turn in range(self._max_tool_turns):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                )
            except Exception as exc:  # timeout / rate limit / network
                logger.warning("chatbot LLM call failed: %s — fail-safe template", exc)
                return ChatResponse(
                    content=LLM_ERROR_TEMPLATE,
                    tool_calls_made=tool_calls_made,
                    blocked_reason="llm_error",
                )

            if getattr(response, "stop_reason", None) != "tool_use":
                text = self._extract_text(response.content)
                # --- Couche 3 hook (Étape 5) : output filter à insérer ici. ---
                return ChatResponse(
                    content=text,
                    tool_calls_made=tool_calls_made,
                    blocked_reason=None,
                )

            # Tool-use turn: record the assistant message once, then append a
            # single user message bundling every tool_result of this turn.
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                result = self._execute_tool(block.name, dict(block.input))
                tool_calls_made.append({"name": block.name, "input": dict(block.input)})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, ensure_ascii=False, default=str),
                })
            messages.append({"role": "user", "content": tool_results})

        # Tool-turn budget exhausted without a final text answer.
        logger.warning("chatbot exceeded %d tool turns — fail-safe template", self._max_tool_turns)
        return ChatResponse(
            content=LLM_ERROR_TEMPLATE,
            tool_calls_made=tool_calls_made,
            blocked_reason="max_tool_turns_exceeded",
        )

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _safe_summary(self) -> dict[str, Any]:
        try:
            return self._summary_provider.get()
        except Exception as exc:  # never let summary failure abort the turn
            logger.warning("signal_summary provider failed: %s", exc)
            return {"instruments_tracked": []}

    def _execute_tool(self, name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Run a tool; on failure return an ``{"error": ...}`` dict so the LLM
        can recover gracefully rather than the whole turn crashing."""
        try:
            if name == "get_market_reading":
                instrument = tool_input.get("instrument")
                timeframe = tool_input.get("timeframe")
                reading = self._assembler.get_or_generate(instrument, timeframe)
                return reading.model_dump(mode="json")
            if name == "get_signal_summary":
                return self._summary_provider.get()
            return {"error": f"unknown tool: {name}"}
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            return {"error": f"tool execution failed: {exc}"}

    @staticmethod
    def _extract_text(content: Any) -> str:
        """Concatenate the text blocks of an Anthropic response."""
        if isinstance(content, str):
            return content
        parts: list[str] = []
        for block in content or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "\n".join(p for p in parts if p).strip()


__all__ = [
    "Chatbot",
    "ChatResponse",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "MAX_TOOL_TURNS",
    "SYSTEM_PROMPT_TEMPLATE",
    "TOOL_SCHEMAS",
]
