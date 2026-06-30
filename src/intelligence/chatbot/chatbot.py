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
    OUTPUT_CONTAMINATED_TEMPLATE,
    REFUSAL_TEMPLATE,
    VIEW_ACTION_REFUSAL_TEMPLATE,
)
from src.intelligence.chatbot.output_filter import OutputFilter
from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider
from src.intelligence.chatbot.view_action_filter import (
    ALLOWED_ACTIONS,
    ViewActionValidator,
)

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
    {
        "name": "apply_chart_view",
        "description": (
            "Change UNIQUEMENT l'AFFICHAGE du graphique (jamais les données ni la "
            "géométrie d'une zone). Action structurée, liste blanche stricte :\n"
            "- set_layer_visibility {layer: 'fvg'|'ob'|'breaks'|'all', visible: bool} "
            "— masquer/afficher UNE couche (breaks = BOS/CHOCH/retest). Pour "
            "PLUSIEURS couches d'un coup (« enlève les FVG et les OB »), utilise la "
            "forme {layers: ['fvg','ob'], visible: bool} — sous-ensemble de "
            "'fvg'/'ob'/'breaks' ; ne mélange jamais layer et layers.\n"
            "- filter_zones {active_only?: bool, proximity_only?: bool, "
            "proximity_pct?: number, min_size_pct?: number} — filtrer les zones "
            "DÉTECTÉES affichées (actives seules / proches du prix / taille min "
            "en % du prix).\n"
            "- focus_zone {zone_id: str} — se centrer sur une zone DÉTECTÉE "
            "(utilise un id renvoyé par get_market_reading ; jamais un id inventé).\n"
            "- highlight_zone {zone_id: str} — mettre en évidence une zone DÉTECTÉE.\n"
            "- hide_zones {zone_ids: [str]} — RETIRER DE L'AFFICHAGE une ou "
            "plusieurs zones DÉTECTÉES (par leurs ids réels ; réversible). La zone "
            "existe toujours, on masque seulement sa boîte.\n"
            "- isolate_zones {zone_ids: [str]} — n'AFFICHER QUE ces zones DÉTECTÉES "
            "(masque toutes les autres ; réversible).\n"
            "- show_zones {zone_ids?: [str]} — ré-afficher des zones masquées ; sans "
            "zone_ids, tout restaurer (annule hide/isolate).\n"
            "- focus_price {} — se centrer sur le prix courant.\n"
            "- fit_chart {} — ajuster la vue à toutes les bougies.\n"
            "- reset_view {} — réinitialiser l'affichage (couches visibles, sans "
            "filtre ni mise en évidence).\n"
            "- set_instrument_timeframe {instrument: 'XAUUSD'|'EURUSD', "
            "timeframe: 'M15'|'H1'|'H4'} — changer la combinaison affichée.\n"
            "INTERDIT : créer/placer/déplacer/redimensionner une structure, ou "
            "fournir un prix/niveau — ces actions n'existent pas et seront rejetées."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(ALLOWED_ACTIONS),
                    "description": "Action de la liste blanche (vue seule).",
                },
                "params": {
                    "type": "object",
                    "description": (
                        "Paramètres de l'action (voir description). Aucun champ de "
                        "prix/niveau/géométrie n'est admis."
                    ),
                },
            },
            "required": ["action"],
        },
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

CONTRÔLE DE L'AFFICHAGE DU GRAPHIQUE (apply_chart_view) :
- Tu peux changer ce que le graphique AFFICHE, jamais ce que le marché contient.
- Actions possibles uniquement : masquer/afficher une OU plusieurs couches (FVG, OB, BOS/CHOCH ; pour plusieurs couches à la fois utilise set_layer_visibility avec layers: ['fvg','ob']), filtrer les zones DÉTECTÉES (actives seules / proches / taille min), te centrer/zoomer (zone détectée ou prix courant), changer instrument/timeframe, mettre en évidence une zone DÉTECTÉE.
- Pour cibler une zone précise (focus_zone / highlight_zone / hide_zones / isolate_zones), appelle d'abord get_market_reading pour obtenir son id, et n'utilise QUE des ids renvoyés par le moteur — jamais un id ou un prix inventé.
- Pour masquer/isoler « l'OB à 4160 » (ou toute zone désignée par son prix) : lis get_market_reading, trouve la zone RÉELLE dont la bande contient ce prix, et masque/isole SON id. Si AUCUNE zone réelle ne correspond, ne masque rien et dis-le : « Aucune zone détectée ne correspond à ce niveau — je n'affiche que ce que le marché montre. »
- Pour masquer/isoler un GROUPE désigné par un critère factuel (« masque les FVG touchés », « n'affiche que les OB actifs », « cache les zones mitigées ») : lis get_market_reading, sélectionne les zones RÉELLES qui correspondent au critère via leur champ `status` (active / mitigated / partially_filled / filled / invalidated — « touché » = mitigated ou partially_filled), rassemble TOUS leurs ids et passe-les en une seule fois dans zone_ids. Tu ne masques que les zones réellement renvoyées par le moteur ; si aucune ne correspond, ne masque rien et dis-le.
- Masquer retire une zone réelle de l'AFFICHAGE (réversible via show_zones) ; ce n'est jamais inventer, déplacer ou supprimer une structure du marché.
- Tu n'inventes, ne places, ne déplaces et ne redimensionnes JAMAIS une structure. Si on te le demande (« mets un OB à 2000 », « agrandis ce FVG »), tu refuses ainsi : « Je n'invente pas de structure — je n'affiche que ce que le marché montre. Je peux masquer, filtrer, ou me centrer sur les zones détectées. »
- Après une action d'affichage, décris-la comme un changement de VUE, au présent (« j'ai masqué les FVG », « je me centre sur l'OB actif »). N'implique jamais que tu as modifié le marché ou créé une structure.

CONTEXTE INITIAL (signal_summary) :
{signal_summary}

Tu as accès à 3 tools :
- get_market_reading(instrument, timeframe) : lecture complète d'une combinaison.
- get_signal_summary() : résumé des 6 combinaisons (XAUUSD/EURUSD × M15/H1/H4).
- apply_chart_view(action, params) : changer l'AFFICHAGE du graphique (liste blanche, vue seule).

Si l'utilisateur pose une question contextuelle nécessitant des détails absents du signal_summary, appelle get_market_reading."""


@dataclass
class ChatResponse:
    """Result of a chatbot turn.

    Attributes:
        content: the text shown to the user (LLM answer, refusal, or fallback).
        tool_calls_made: list of {"name", "input"} for each tool executed.
        view_actions: display-only chart actions the model emitted AND that passed
            the Couche 4 whitelist (normalised). The frontend applies these to the
            chart RENDER only — they never touch detection. Empty on a plain turn.
        blocked_reason: None on a normal answer; otherwise the reason
            (adversarial category, "llm_error", "max_tool_turns_exceeded").
    """

    content: str
    tool_calls_made: list[dict[str, Any]] = field(default_factory=list)
    view_actions: list[dict[str, Any]] = field(default_factory=list)
    blocked_reason: Optional[str] = None


class Chatbot:
    """Niveau 1.5 strict conversational orchestrator (Couche 2)."""

    def __init__(
        self,
        anthropic_client: Any,
        summary_provider: SignalSummaryProvider,
        assembler: Any,
        adversarial_filter: Optional[AdversarialFilter] = None,
        output_filter: Optional[OutputFilter] = None,
        view_action_validator: Optional[ViewActionValidator] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_turns: int = MAX_TOOL_TURNS,
    ) -> None:
        self._client = anthropic_client
        self._summary_provider = summary_provider
        self._assembler = assembler
        self._adv_filter = adversarial_filter or AdversarialFilter()
        self._output_filter = output_filter or OutputFilter()
        self._view_validator = view_action_validator or ViewActionValidator()
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
        view_actions: list[dict[str, Any]] = []
        # Ids of zones the engine actually emitted this turn — the ONLY zones the
        # model is allowed to focus / highlight (an invented id is rejected).
        known_zone_ids: set[str] = set()

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
                # --- Couche 3 — output forbidden-tokens filter ---
                output_check = self._output_filter.check(text)
                if output_check.contaminated:
                    logger.warning(
                        "chatbot output contaminated (%s: %s) — fallback template",
                        output_check.category, output_check.matched_tokens,
                    )
                    return ChatResponse(
                        content=OUTPUT_CONTAMINATED_TEMPLATE,
                        tool_calls_made=tool_calls_made,
                        view_actions=view_actions,
                        blocked_reason=f"output_contaminated_{output_check.category}",
                    )
                return ChatResponse(
                    content=text,
                    tool_calls_made=tool_calls_made,
                    view_actions=view_actions,
                    blocked_reason=None,
                )

            # Tool-use turn: record the assistant message once, then append a
            # single user message bundling every tool_result of this turn.
            messages.append({"role": "assistant", "content": response.content})
            tool_results: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                tool_input = dict(block.input)
                tool_calls_made.append({"name": block.name, "input": tool_input})

                if block.name == "apply_chart_view":
                    # --- Couche 4 — view-action whitelist (display-only) ---
                    result = self._apply_view_action(
                        tool_input, view_actions, known_zone_ids
                    )
                else:
                    result = self._execute_tool(block.name, tool_input)
                    # Harvest detected zone ids so a later focus/highlight can only
                    # reference a zone the engine actually emitted.
                    self._harvest_zone_ids(result, known_zone_ids)

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
            view_actions=view_actions,
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

    def _apply_view_action(
        self,
        tool_input: dict[str, Any],
        view_actions: list[dict[str, Any]],
        known_zone_ids: set[str],
    ) -> dict[str, Any]:
        """Validate a proposed view action (Couche 4) and record it if admissible.

        Returns the tool_result payload handed back to the model: a success ack on
        a whitelisted action, or a rejection carrying the on-brand refusal text so
        the model phrases the refusal itself. NEVER touches detection — a valid
        action is only *recorded* for the frontend to apply to the render.
        """
        check = self._view_validator.validate(
            tool_input, known_zone_ids=known_zone_ids
        )
        if not check.valid:
            logger.info("view action rejected (%s): %s", check.reason, tool_input)
            return {
                "status": "rejected",
                "reason": check.reason,
                "message": VIEW_ACTION_REFUSAL_TEMPLATE,
            }
        action = check.action or {}
        view_actions.append(action)
        return {"status": "applied", "action": action}

    @staticmethod
    def _harvest_zone_ids(result: Any, known_zone_ids: set[str]) -> None:
        """Collect OB / FVG ids from a get_market_reading result (read-only)."""
        if not isinstance(result, dict):
            return
        structure = result.get("structure")
        if not isinstance(structure, dict):
            return
        for key in ("order_blocks", "fair_value_gaps"):
            for zone in structure.get(key, []) or []:
                if isinstance(zone, dict):
                    zid = zone.get("id")
                    if isinstance(zid, str) and zid:
                        known_zone_ids.add(zid)

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
