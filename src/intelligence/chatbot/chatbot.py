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
    VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE,
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
        "name": "get_ob_diagnostic",
        "description": (
            "Diagnostic FACTUEL du moteur : pourquoi une bougie précise est ou "
            "n'est PAS un Order Block. Fournis instrument + timeframe et UNE "
            "référence : price (le moteur résout la bougie la plus récente dont "
            "l'amplitude contient ce prix) ou ts (horodatage ISO de la bougie). "
            "Le résultat rapporte les critères RÉELS évalués par le moteur "
            "(checks passés/échoués avec label_fr), et si un OB détecté a été "
            "retiré (invalidé par une clôture à travers la zone, ou au-delà de "
            "la limite d'affichage). Statuts : is_order_block / not_candidate / "
            "was_rejected / awaiting_next_candle / unresolved / no_data. "
            "OBLIGATOIRE avant d'expliquer pourquoi une bougie n'est pas un OB — "
            "ne réponds jamais à cette question par la définition générale."
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
                "price": {
                    "type": "number",
                    "description": "Prix approximatif désignant la bougie (ex. 4114).",
                },
                "ts": {
                    "type": "string",
                    "description": (
                        "Horodatage ISO 8601 de la bougie (ex. "
                        "2026-07-01T14:00:00Z). Prioritaire sur price si les deux "
                        "sont fournis."
                    ),
                },
            },
            "required": ["instrument", "timeframe"],
        },
    },
    {
        "name": "apply_chart_view",
        "description": (
            "Change UNIQUEMENT l'AFFICHAGE du graphique (jamais les données ni la "
            "géométrie d'une zone). Action structurée, liste blanche stricte :\n"
            "- set_layer_visibility {layer: 'fvg'|'ob'|'breaks'|'liquidity'|'all', "
            "visible: bool} — masquer/afficher UNE couche (breaks = BOS/CHOCH/"
            "retest ; liquidity = poches de liquidité BSL/SSL). Pour PLUSIEURS "
            "couches d'un coup (« enlève les FVG et les OB »), utilise la forme "
            "{layers: ['fvg','ob'], visible: bool} — sous-ensemble de "
            "'fvg'/'ob'/'breaks'/'liquidity' ; ne mélange jamais layer et layers.\n"
            "- filter_zones {active_only?: bool, proximity_only?: bool, "
            "proximity_pct?: number, min_size_pct?: number} — filtrer les zones "
            "DÉTECTÉES affichées (actives seules / proches du prix / taille min "
            "en % du prix).\n"
            "- focus_zone {zone_id: str} — se centrer sur une zone DÉTECTÉE "
            "(utilise un id renvoyé par get_market_reading ; jamais un id inventé).\n"
            "- highlight_zone {zone_id: str} — mettre en évidence une zone DÉTECTÉE.\n"
            "- hide_zones {zone_ids: [str]} OU {category: "
            "'fvg'|'ob'|'bsl'|'ssl'|'liquidity'} — RETIRER DE L'AFFICHAGE des "
            "structures DÉTECTÉES (par leurs ids réels, ou par catégorie : la "
            "catégorie est résolue côté serveur vers TOUS les ids réellement émis "
            "de ce type — 'ssl' = poches de liquidité vendeuses, 'bsl' = "
            "acheteuses, 'liquidity' = les deux ; réversible). La structure existe "
            "toujours, on masque seulement son tracé. Ne mélange jamais zone_ids "
            "et category.\n"
            "- isolate_zones {zone_ids: [str]} OU {category: ...} — n'AFFICHER QUE "
            "ces structures DÉTECTÉES (masque toutes les autres, poches de "
            "liquidité comprises ; réversible).\n"
            "- show_zones {zone_ids?: [str]} OU {category: ...} — ré-afficher des "
            "structures masquées ; sans zone_ids ni category, tout restaurer "
            "(annule hide/isolate).\n"
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
- Actions possibles uniquement : masquer/afficher une OU plusieurs couches (FVG, OB, BOS/CHOCH, liquidité BSL/SSL ; pour plusieurs couches à la fois utilise set_layer_visibility avec layers: ['fvg','ob']), filtrer les zones DÉTECTÉES (actives seules / proches / taille min), te centrer/zoomer (zone détectée ou prix courant), changer instrument/timeframe, mettre en évidence une zone DÉTECTÉE.
- Les poches de liquidité (BSL/SSL) SONT masquables, exactement comme les OB/FVG : masquer une poche est un filtre d'affichage RÉVERSIBLE — la poche détectée existe toujours, tu ne supprimes ni n'inventes rien. Ne refuse jamais « masque les SSL/BSL/la liquidité » au motif que la liquidité ne serait pas contrôlable.
- Pour « masque les SSL » / « les BSL » / « la liquidité » (ou les FVG/OB en tant que groupe par ids) : appelle d'abord get_market_reading (les poches du tour doivent être connues), puis hide_zones {{category: 'ssl'|'bsl'|'liquidity'|'fvg'|'ob'}} — le serveur résout la catégorie vers TOUS les ids réellement émis, et rien d'autre. Ré-afficher : show_zones {{category: ...}} ou show_zones {{}} pour tout restaurer. Si le moteur n'émet AUCUNE poche de la catégorie, l'action est rejetée : dis-le honnêtement (« le moteur n'émet aucune poche SSL sur cette lecture ») — ne masque rien, n'invente rien.
- Une poche de liquidité se DÉCRIT (côté, niveau, intacte/prise/cassée) — jamais de commentaire prédictif du type « le prix va la chercher ».
- Pour cibler une zone ou une poche précise (focus_zone / highlight_zone / hide_zones / isolate_zones), appelle d'abord get_market_reading pour obtenir son id, et n'utilise QUE des ids renvoyés par le moteur — jamais un id ou un prix inventé.
- Pour masquer/isoler « l'OB à 4160 » (ou toute zone désignée par son prix) : lis get_market_reading, trouve la zone RÉELLE dont la bande contient ce prix, et masque/isole SON id. Si AUCUNE zone réelle ne correspond, ne masque rien et dis-le : « Aucune zone détectée ne correspond à ce niveau — je n'affiche que ce que le marché montre. »
- Pour masquer/isoler un GROUPE désigné par un critère factuel (« masque les FVG touchés », « n'affiche que les OB actifs », « cache les zones mitigées ») : lis get_market_reading, sélectionne les zones RÉELLES qui correspondent au critère via leur champ `status` (active / mitigated / partially_filled / filled / invalidated — « touché » = mitigated ou partially_filled), rassemble TOUS leurs ids et passe-les en une seule fois dans zone_ids. Tu ne masques que les zones réellement renvoyées par le moteur ; si aucune ne correspond, ne masque rien et dis-le.
- Masquer retire une zone réelle de l'AFFICHAGE (réversible via show_zones) ; ce n'est jamais inventer, déplacer ou supprimer une structure du marché.
- Tu n'inventes, ne places, ne déplaces et ne redimensionnes JAMAIS une structure. Si on te le demande (« mets un OB à 2000 », « agrandis ce FVG »), tu refuses ainsi : « Je n'invente pas de structure — je n'affiche que ce que le marché montre. Je peux masquer, filtrer, ou me centrer sur les zones détectées. »
- Après une action d'affichage, décris-la comme un changement de VUE, au présent (« j'ai masqué les FVG », « je me centre sur l'OB actif »). N'implique jamais que tu as modifié le marché ou créé une structure.

DIAGNOSTIC ORDER BLOCK (get_ob_diagnostic) :
- Quand l'utilisateur demande pourquoi une bougie ou un niveau n'est PAS un Order Block (« pourquoi la bougie à 4114 n'est pas un OB ? », « pourquoi pas d'OB à 14h ? »), appelle get_ob_diagnostic avec le prix ou l'horodatage. Tu n'expliques JAMAIS un rejet sans ce diagnostic, et tu ne réponds jamais par la définition générale d'un OB.
- Tu rapportes UNIQUEMENT les raisons renvoyées par le moteur : les critères échoués (champs checks → label_fr, avec les valeurs observées), ou la raison de retrait (reject_label_fr, avec la date d'invalidation le cas échéant). Jamais une raison de ton cru.
- status=not_candidate → explique le(s) critère(s) échoué(s), factuellement (« la bougie suivante n'a pas dépassé son plus haut : X contre Y »).
- status=was_rejected → explique la raison réelle : invalidation (une bougie a clôturé à travers la zone, date fournie) ou zone au-delà de la limite d'affichage des zones les plus significatives.
- status=is_order_block → dis que le moteur détecte bien un OB ici (direction, statut du cycle de vie, retesté ou non) ; la confusion vient peut-être de l'affichage.
- status=awaiting_next_candle → explique que le moteur évalue un OB sur la bougie SUIVANTE, qui n'existe pas encore pour la dernière bougie.
- status=unresolved ou no_data → dis honnêtement que tu n'as pas le détail pour cette bougie (hors de la fenêtre analysée, prix jamais touché sur la période, pas de données) — tu ne devines pas.
- Ce diagnostic décrit l'évaluation PASSÉE du moteur — éducatif et factuel. Tu n'en tires jamais une anticipation de ce que le prix fera.

CONTEXTE INITIAL (signal_summary) :
{signal_summary}

Tu as accès à 4 tools :
- get_market_reading(instrument, timeframe) : lecture complète d'une combinaison.
- get_signal_summary() : résumé des 6 combinaisons (XAUUSD/EURUSD × M15/H1/H4).
- get_ob_diagnostic(instrument, timeframe, price|ts) : pourquoi une bougie précise est ou n'est pas un Order Block (raisons réelles du moteur).
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
        # Ids of zones/pockets the engine actually emitted this turn — the ONLY
        # structures the model is allowed to focus / highlight / mask (an
        # invented id is rejected). The category index maps each closed category
        # ('fvg'/'ob'/'bsl'/'ssl'/'liquidity') to those same emitted ids, so a
        # category mask (« masque les SSL ») resolves server-side to real ids.
        known_zone_ids: set[str] = set()
        known_category_ids: dict[str, list[str]] = {}

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
                        tool_input, view_actions, known_zone_ids, known_category_ids
                    )
                else:
                    result = self._execute_tool(block.name, tool_input)
                    # Harvest detected zone/pocket ids so a later focus/highlight/
                    # mask can only reference a structure the engine actually
                    # emitted.
                    self._harvest_zone_ids(result, known_zone_ids, known_category_ids)

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
        known_category_ids: dict[str, list[str]],
    ) -> dict[str, Any]:
        """Validate a proposed view action (Couche 4) and record it if admissible.

        Returns the tool_result payload handed back to the model: a success ack on
        a whitelisted action, or a rejection carrying the on-brand refusal text so
        the model phrases the refusal itself. NEVER touches detection — a valid
        action is only *recorded* for the frontend to apply to the render.
        """
        check = self._view_validator.validate(
            tool_input,
            known_zone_ids=known_zone_ids,
            known_category_ids=known_category_ids,
        )
        if not check.valid:
            logger.info("view action rejected (%s): %s", check.reason, tool_input)
            # A category that resolves to ZERO emitted structures is not an
            # invented-structure attempt — hand back the honest "nothing of that
            # kind on this reading" wording instead of the generic refusal.
            message = (
                VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE
                if check.reason == "empty_category"
                else VIEW_ACTION_REFUSAL_TEMPLATE
            )
            return {
                "status": "rejected",
                "reason": check.reason,
                "message": message,
            }
        action = check.action or {}
        view_actions.append(action)
        return {"status": "applied", "action": action}

    @staticmethod
    def _harvest_zone_ids(
        result: Any,
        known_zone_ids: set[str],
        known_category_ids: Optional[dict[str, list[str]]] = None,
    ) -> None:
        """Collect OB / FVG / liquidity-pocket ids from a get_market_reading
        result (read-only). Each id also lands in its category bucket ('fvg' /
        'ob' / 'bsl' / 'ssl' / 'liquidity') so a category mask resolves to the
        ids ACTUALLY emitted this turn — and nothing else."""
        if not isinstance(result, dict):
            return
        structure = result.get("structure")
        if not isinstance(structure, dict):
            return

        def _bucket(category: str, zid: str) -> None:
            if known_category_ids is None:
                return
            ids = known_category_ids.setdefault(category, [])
            if zid not in ids:
                ids.append(zid)

        for key, category in (("order_blocks", "ob"), ("fair_value_gaps", "fvg")):
            for zone in structure.get(key, []) or []:
                if isinstance(zone, dict):
                    zid = zone.get("id")
                    if isinstance(zid, str) and zid:
                        known_zone_ids.add(zid)
                        _bucket(category, zid)
        for pool in structure.get("liquidity_pools", []) or []:
            if isinstance(pool, dict):
                pid = pool.get("id")
                if isinstance(pid, str) and pid:
                    known_zone_ids.add(pid)
                    side = pool.get("side")
                    if side in ("bsl", "ssl"):
                        _bucket(side, pid)
                    _bucket("liquidity", pid)

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
            if name == "get_ob_diagnostic":
                # Read-only engine diagnostic — the reasons come from the same
                # code path that decided accept/reject (never fabricated here).
                return self._assembler.get_ob_diagnostic(
                    tool_input.get("instrument"),
                    tool_input.get("timeframe"),
                    ts=tool_input.get("ts"),
                    price=tool_input.get("price"),
                )
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
