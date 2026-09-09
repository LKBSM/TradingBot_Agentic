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
used) — consistent with LLMNarrativeEngine, and trivial to stub in tests.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
from src.intelligence.chatbot.constants import (
    LLM_ERROR_TEMPLATE,
    OUTPUT_CONTAMINATED_TEMPLATE,
    REFUSAL_TEMPLATE,
    VIEW_ACTION_EMPTY_CATEGORY_TEMPLATE,
    VIEW_ACTION_REFUSAL_TEMPLATE,
    refusal_for,
)
from src.intelligence.chatbot.output_filter import OutputFilter
from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider
from src.intelligence.llm_cost_policy import cache_block_for
from src.intelligence.chatbot.view_action_filter import (
    ALLOWED_ACTIONS,
    ViewActionValidator,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# MIA-2 lever 6 — response-length safety cap. A niveau-1.5 answer is 2-4 phrases
# (~120-180 tokens); 768 leaves ample room for an explicit "détaille" request
# while capping pathological runaway generations (a longer answer is a slower
# answer). Purely a ceiling — it never lengthens a normal reply.
DEFAULT_MAX_TOKENS = 768
# MIA-2 lever G — explicit per-call timeout (the scanner_translator already sets
# one; the chatbot did not, so a hung call fell back to the SDK default ~600s).
# On timeout the existing except-branch returns LLM_ERROR_TEMPLATE unchanged.
DEFAULT_TIMEOUT_S = 20.0
# MIA-2 lever 5a — server-side history depth cap. The client may send up to 20
# messages; the model only needs a recent window for a market-analysis chat.
# Kept as an EVEN number and applied user-first so the slice stays a valid
# alternating transcript (see _truncate_history). Bounds per-turn context.
MAX_MODEL_HISTORY = 12
MAX_TOOL_TURNS = 3  # hard cap on tool-use rounds to avoid infinite loops

# Perimeter derived from the single sources (TF-1): the chat covers every
# displayed instrument/timeframe, never a hand-listed subset.
from src.intelligence import market_registry
from src.intelligence.lookback_config import enabled_timeframes as _enabled_tfs
from src.intelligence.lookback_config import supported_instruments as _supported_instruments

SUPPORTED_INSTRUMENTS = tuple(_supported_instruments())
SUPPORTED_TIMEFRAMES = tuple(_enabled_tfs())

# Windows for the calendar tool, in minutes. A week each way is the product's
# calendar horizon (matches the /api/calendar defaults order of magnitude).
_CAL_WEEK_MIN = 7 * 24 * 60


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
    {
        "name": "list_markets",
        "description": (
            "Catalogue FACTUEL des marchés couverts par le produit (source unique "
            "du registre) et des unités de temps disponibles. À utiliser quand "
            "l'utilisateur demande quels marchés/quelles unités existent, ou avant "
            "d'affirmer qu'un marché est ou n'est pas suivi. Renvoie pour chaque "
            "marché : id, libellé, type, décimales de prix, unités de temps "
            "servies. N'invente jamais un marché absent de cette liste."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_economic_calendar",
        "description": (
            "Publications économiques RÉELLES du calendrier (source officielle), à "
            "venir et/ou récentes, éventuellement filtrées par marché. À utiliser "
            "dès qu'une question porte sur une annonce macro (NFP, CPI, décision de "
            "taux, GDP…). Renvoie une liste d'événements avec leur identifiant "
            "STABLE `event_id`, le nom, la devise, l'organisme, l'horodatage, "
            "l'état de la valeur (published/pending/unfetched/unavailable) et, si "
            "publiée, la valeur `actual`/`previous`. Si aucune publication ne "
            "correspond, renvoie une liste vide — dis-le, n'invente aucun chiffre. "
            "Pour le détail chiffré d'une publication précise, enchaîne avec "
            "get_publication en passant son `event_id`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "market": {
                    "type": "string",
                    "description": (
                        "Filtre optionnel : un id de marché du registre (ex. "
                        "XAUUSD, EURUSD). Un marché absent du registre est rejeté."
                    ),
                },
                "horizon": {
                    "type": "string",
                    "enum": ["upcoming", "recent", "both"],
                    "description": (
                        "upcoming = à venir (7 j), recent = récentes (7 j), both = "
                        "les deux. Défaut : upcoming."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_publication",
        "description": (
            "Détail FACTUEL d'UNE publication du calendrier par son `event_id` "
            "STABLE (celui renvoyé par get_economic_calendar — jamais un id "
            "inventé). Renvoie la valeur publiée et sa révision éventuelle, la "
            "valeur précédente, l'unité, l'organisme et sa licence, l'historique "
            "publié (`value_series`), et — quand elles sont calculables — les "
            "mesures du moteur autour des publications passées de cette série "
            "(calme avant, état de structure, cycle de vie des zones, retour au "
            "calme), chacune avec sa provenance (taille d'échantillon, période, "
            "marché mesuré). Si l'`event_id` n'existe pas, renvoie found=false : "
            "dis-le honnêtement, n'invente rien. Une mesure absente (None) signifie "
            "qu'elle n'est pas calculable de façon fiable — restitue cette absence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {
                    "type": "string",
                    "description": (
                        "Identifiant stable renvoyé par get_economic_calendar, ex. "
                        "'official:us_cpi:2026-09-14'."
                    ),
                },
            },
            "required": ["event_id"],
        },
    },
]


# MIA-2 lever 1 — the STABLE part of the system prompt (identity, rules, chart
# control, OB diagnostic, market-state rules, tool access). It is byte-identical
# on every turn, so it is sent as a cache_control:ephemeral block: Anthropic
# reprocesses it once, then reads it from cache (~90% cheaper/faster prefill) on
# every following turn. The VARIABLE signal_summary is deliberately NOT here —
# it lives in a separate trailing block (SIGNAL_CONTEXT_TEMPLATE) so a 60s-TTL
# change never invalidates this cached prefix. Nothing was removed from the
# prompt: the signal context simply moved to the end.
SYSTEM_PROMPT_STATIC = """Tu es MIA Markets, un outil de compréhension des conditions de marché.

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

ÉTAT DU MARCHÉ (champ market_status de get_market_reading) :
- Chaque lecture porte un champ `market_status.state` : open / closed_weekend / closed_holiday / daily_break / data_lagged. Il fait autorité sur l'actualité des données.
- Quand l'état n'est PAS open, la lecture décrit la DERNIÈRE bougie clôturée (champ last_close_ts), pas le marché en direct. Tu ne décris JAMAIS une structure (OB, FVG, poche) comme « récente », « nouvelle », « qui vient de se former » ou « en train de » dans ce cas : elle date de la dernière clôture et n'a pas changé depuis.
- Si on te demande si le marché est ouvert, réponds simplement d'après cet état : « Marché fermé (week-end), dernière bougie clôturée le … ; réouverture … » (closed_weekend/closed_holiday), « Pause quotidienne, reprise à … » (daily_break), ou « Aucune nouvelle bougie reçue depuis … » (data_lagged). Faits seulement — jamais de pronostic sur ce que fera le prix à la réouverture.
- data_lagged = le calendrier dit ouvert mais plus aucune bougie n'arrive : dis-le franchement, ne le présente pas comme une panne honteuse ni comme de l'activité.

CALENDRIER & PUBLICATIONS (get_economic_calendar / get_publication) :
- Dès qu'une question porte sur une annonce macro (NFP, CPI, taux, GDP, chômage…), appelle get_economic_calendar (filtre par marché si pertinent), puis get_publication avec l'`event_id` renvoyé pour le détail chiffré. Tu ne cites JAMAIS un chiffre de publication sans l'avoir obtenu par ces outils.
- Tu n'utilises QUE des `event_id` renvoyés par get_economic_calendar — jamais un id inventé. Si get_publication renvoie found=false, dis simplement que cette publication n'est pas dans le calendrier ; tu n'inventes ni valeur ni date.
- Une valeur `actual` avec actual_state != "published" n'est PAS un chiffre publié : pending = pas encore sortie, unfetched/unavailable = non récupérée. Restitue l'état, n'invente pas la valeur.
- Les mesures (calme avant, cycle de vie des zones…) décrivent le comportement PASSÉ du marché autour de cette série ; tu les rapportes avec leur provenance (échantillon, période) et tu n'en tires jamais une anticipation. Une mesure à None n'est pas calculable de façon fiable : dis-le.

CATALOGUE DES MARCHÉS (list_markets) :
- Avant d'affirmer qu'un marché ou une unité est suivi ou non, appelle list_markets. Tu ne prétends jamais suivre un marché absent de ce catalogue, et tu ne refuses jamais un marché qui y figure.

RÈGLE D'ABSENCE (vaut pour TOUS les outils) :
- Tu peux parler de tout ce que le produit SAIT, c'est-à-dire de ce qu'un outil te renvoie réellement. Si l'outil ne renvoie rien (marché non couvert, unité non calculée, publication introuvable, mesure None), tu LE DIS. Tu ne combles pas, tu ne raisonnes pas « par analogie », tu ne produis pas une lecture plausible. Un texte qui sonne juste sur une donnée que tu n'as pas est un mensonge.

Tu as accès à 7 tools :
- get_market_reading(instrument, timeframe) : lecture complète d'une combinaison.
- get_signal_summary() : résumé des 6 combinaisons (XAUUSD/EURUSD × M15/H1/H4).
- get_ob_diagnostic(instrument, timeframe, price|ts) : pourquoi une bougie précise est ou n'est pas un Order Block (raisons réelles du moteur).
- apply_chart_view(action, params) : changer l'AFFICHAGE du graphique (liste blanche, vue seule).
- list_markets() : catalogue des marchés et unités de temps couverts.
- get_economic_calendar(market?, horizon?) : publications économiques réelles (à venir/récentes).
- get_publication(event_id) : détail chiffré + mesures d'une publication précise.

Le CONTEXTE INITIAL (signal_summary) des combinaisons suivies est fourni en fin de ce message système. Si l'utilisateur pose une question contextuelle nécessitant des détails absents de ce signal_summary, appelle get_market_reading."""


# MIA-2 lever 1 — the VARIABLE trailing block. Injected AFTER the cached static
# prefix, so its 60s-TTL refresh never busts the cache. Same data as before,
# same wording — only its position in the prompt changed (middle → end).
SIGNAL_CONTEXT_TEMPLATE = """CONTEXTE INITIAL (signal_summary) :
{signal_summary}"""


# Backward-compatible single-string view (the two parts concatenated). Retained
# for callers/tests that inspect the full prompt text; the runtime uses the two
# blocks above so the stable prefix can be cached independently of the summary.
SYSTEM_PROMPT_TEMPLATE = SYSTEM_PROMPT_STATIC + "\n\n" + SIGNAL_CONTEXT_TEMPLATE


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
        calendar_service: Optional[Any] = None,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_tool_turns: int = MAX_TOOL_TURNS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_model_history: int = MAX_MODEL_HISTORY,
        tool_schemas: Optional[list[dict[str, Any]]] = None,
        tool_handlers: Optional[dict[str, Any]] = None,
        extra_system_blocks: Optional[list[str]] = None,
    ) -> None:
        self._client = anthropic_client
        self._summary_provider = summary_provider
        self._assembler = assembler
        self._adv_filter = adversarial_filter or AdversarialFilter()
        self._output_filter = output_filter or OutputFilter()
        self._view_validator = view_action_validator or ViewActionValidator()
        # Calendar source for the calendar/publication tools. Injected in tests;
        # in production it is built lazily on first calendar read (same pattern as
        # the /api/calendar route). ``_calendar_failed`` latches a build failure so
        # a broken calendar degrades to an honest absence, never a per-turn crash.
        self._calendar_service = calendar_service
        self._calendar_failed = False
        self._model = model
        self._max_tokens = max_tokens
        self._max_tool_turns = max_tool_turns
        self._timeout_s = timeout_s
        self._max_model_history = max_model_history
        # MIA-4S — restricted-surface hooks. All three default to the production
        # behaviour, so an unset caller is byte-for-byte the chatbot of before.
        # They exist so a DIFFERENT data surface (the landing's frozen
        # illustration) can be served by THIS orchestrator instead of a second,
        # divergent one: Couche 1 (adversarial input), Couche 3 (forbidden-token
        # output) and Couche 4 (view-action whitelist) are shared code, never
        # re-implemented. What a caller may narrow is the TOOL SURFACE and the
        # system prompt's tail — never a defence layer.
        #   tool_schemas        — the tools declared to the model. A tool absent
        #                         here cannot be called, whatever the prompt says
        #                         (the lock is the code, not the instruction).
        #   tool_handlers       — {name: callable(input) -> dict} consulted BEFORE
        #                         the built-in dispatch, so a restricted build
        #                         serves frozen data without touching the engine.
        #   extra_system_blocks — appended AFTER the cached static prefix and
        #                         BEFORE the variable trailing block, so the
        #                         MIA-2 cache breakpoint keeps working.
        self._tool_schemas = tool_schemas if tool_schemas is not None else TOOL_SCHEMAS
        self._tool_handlers = dict(tool_handlers or {})
        self._extra_system_blocks = list(extra_system_blocks or [])

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def chat(
        self,
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> ChatResponse:
        """Blocking one-shot turn — drains :meth:`chat_events` and returns the
        terminal ``answer`` event as a :class:`ChatResponse`.

        MIA-1: this is a thin wrapper over the SAME generator the SSE endpoint
        streams, so the two paths can never diverge on validation. Every
        defence layer (Couche 1/3/4) runs inside ``chat_events``; the wrapper
        adds nothing and removes nothing.
        """
        answer: Optional[dict[str, Any]] = None
        for event in self.chat_events(user_message, conversation_history):
            if event.get("event") == "answer":
                answer = event
        # chat_events always terminates with exactly one answer event.
        assert answer is not None, "chat_events did not emit an answer event"
        return ChatResponse(
            content=answer["content"],
            tool_calls_made=answer.get("tool_calls_made", []),
            view_actions=answer.get("view_actions", []),
            blocked_reason=answer.get("blocked_reason"),
        )

    def chat_events(
        self,
        user_message: str,
        conversation_history: Optional[list[dict[str, Any]]] = None,
    ) -> Iterator[dict[str, Any]]:
        """MIA-1 — the single source of truth for a chat turn, as a stream of
        events the SSE endpoint forwards verbatim and :meth:`chat` drains.

        Event contract (each a plain JSON-serialisable dict):
          - ``{"event": "activity"}`` — emitted ONCE the turn is cleared by
            Couche 1 and a model call is about to happen. It is a fixed,
            non-LLM signal ("M.I.A is working") — no validation needed. Lets the
            client show honest activity < 200 ms without displaying any
            unvalidated model text.
          - ``{"event": "tool", "tool": <name>, "instrument"?, "timeframe"?}`` —
            emitted immediately BEFORE a data-reading tool actually runs (never
            for a cached/deduped read, never for display-only view actions). The
            fields are the tool's own arguments, so the client narrates the TRUE
            step in progress ("Lecture de XAUUSD M15…") — a fixed template, not
            model prose.
          - ``{"event": "answer", "content", "blocked_reason",
            "tool_calls_made", "view_actions"}`` — the terminal event, ALWAYS
            emitted exactly once. ``content`` is the fully-generated, Couche-3-
            validated text (or a template). Never streamed token-by-token: the
            output filter can only inspect complete text, so a partially-shown
            answer could never be safely retracted (MIA-1 §4/C).

        VALIDATION IS UNCHANGED from the pre-MIA-1 blocking path: Couche 1 still
        short-circuits before any LLM call, Couche 4 still rejects invented ids
        on every view action, and Couche 3 still runs on the COMPLETE text
        before the answer event is emitted. This method only adds streaming of
        honest status, in-turn tool de-duplication, and parallel independent
        reads — none of which touch what the layers check.
        """
        # MIA-2 lever 5a — cap the history depth sent to the model (user-first,
        # valid alternating slice). Bounds per-turn context without touching what
        # any layer checks: the current user_message and the full defence chain
        # are unaffected.
        history = self._truncate_history(conversation_history, self._max_model_history)

        # --- Couche 1 — adversarial input filter (no LLM call on match) ---
        adv = self._adv_filter.check(user_message)
        if adv.triggered:
            yield {
                "event": "answer",
                # The refusal is chosen by bucket: a forecast request deserves an
                # answer ABOUT forecasting, not the generic recommendation notice.
                # Unmapped buckets keep REFUSAL_TEMPLATE, so this changed nothing
                # for the four that existed before.
                "content": refusal_for(adv.category),
                "tool_calls_made": [],
                "view_actions": [],
                "blocked_reason": adv.category,
            }
            return

        # The turn is cleared to hit the model — surface honest activity now.
        yield {"event": "activity"}

        # --- Couche 2 — Haiku + tool use ---
        signal_summary = self._safe_summary()
        # Compact JSON (no indent) — same data, fewer prefill tokens (MIA-1 §4,
        # prompt trim). Purely a token-count optimisation; content is identical.
        # MIA-2 lever 1 — two system blocks: a cached, byte-stable prefix
        # (identity/rules/tools access, incl. the MIA-3 calendar/markets tools) +
        # the variable signal_summary AFTER it. Anthropic reads the cached prefix
        # (~90% cheaper/faster prefill) on every turn; the 60s-TTL summary refresh
        # only re-processes its own small block. (This supersedes MIA-3's inline
        # cache_control — same intent, MIA-2's split is cache-stable.)
        system = self._build_system(signal_summary)
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
        # In-turn de-duplication: a data tool executed with the same arguments is
        # NEVER run twice within a turn (across rounds), even if the model asks
        # again — the recorded result is reused (MIA-1 §5/D). The engine's own
        # candle cache already covers cross-message repeats; this removes the
        # in-turn waste and the redundant status flash.
        tool_cache: dict[tuple[str, str], dict[str, Any]] = {}

        for _turn in range(self._max_tool_turns):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=self._max_tokens,
                    system=system,
                    messages=messages,
                    tools=self._tool_schemas,
                    timeout=self._timeout_s,
                )
            except Exception as exc:  # timeout / rate limit / network
                logger.warning("chatbot LLM call failed: %s — fail-safe template", exc)
                yield {
                    "event": "answer",
                    "content": LLM_ERROR_TEMPLATE,
                    "tool_calls_made": tool_calls_made,
                    "view_actions": view_actions,
                    "blocked_reason": "llm_error",
                }
                return

            if getattr(response, "stop_reason", None) != "tool_use":
                text = self._extract_text(response.content)
                # --- Couche 3 — output forbidden-tokens filter (COMPLETE text) ---
                output_check = self._output_filter.check(text)
                if output_check.contaminated:
                    logger.warning(
                        "chatbot output contaminated (%s: %s) — fallback template",
                        output_check.category, output_check.matched_tokens,
                    )
                    yield {
                        "event": "answer",
                        "content": OUTPUT_CONTAMINATED_TEMPLATE,
                        "tool_calls_made": tool_calls_made,
                        "view_actions": view_actions,
                        "blocked_reason": f"output_contaminated_{output_check.category}",
                    }
                    return
                yield {
                    "event": "answer",
                    "content": text,
                    "tool_calls_made": tool_calls_made,
                    "view_actions": view_actions,
                    "blocked_reason": None,
                }
                return

            # Tool-use turn: record the assistant message once, then append a
            # single user message bundling every tool_result of this turn.
            messages.append({"role": "assistant", "content": response.content})
            tool_blocks = [
                b for b in response.content
                if getattr(b, "type", None) == "tool_use"
            ]
            # Record the calls in the model's original block order (unchanged
            # contract for tool_calls_made).
            for block in tool_blocks:
                tool_calls_made.append({"name": block.name, "input": dict(block.input)})

            # Partition: data reads (independent → parallelisable, dedupable) vs
            # display-only view actions (sequential — they depend on the ids the
            # reads of THIS round just harvested).
            data_blocks = [b for b in tool_blocks if b.name != "apply_chart_view"]
            view_blocks = [b for b in tool_blocks if b.name == "apply_chart_view"]
            results_by_id: dict[str, dict[str, Any]] = {}

            # Resolve dedup cache hits first; collect the reads we must actually
            # run (and emit their honest status BEFORE executing them).
            to_run: list[tuple[Any, tuple[str, str], dict[str, Any]]] = []
            for block in data_blocks:
                tool_input = dict(block.input)
                key = self._tool_key(block.name, tool_input)
                if key in tool_cache:
                    results_by_id[block.id] = tool_cache[key]
                else:
                    to_run.append((block, key, tool_input))
            for block, _key, tool_input in to_run:
                status = self._tool_status(block.name, tool_input)
                if status is not None:
                    yield status

            # Execute the fresh reads — in parallel when there is more than one
            # (MIA-1 §7: independent calls go together, not in a file).
            if len(to_run) == 1:
                block, key, tool_input = to_run[0]
                result = self._execute_tool(block.name, tool_input)
                tool_cache[key] = result
                results_by_id[block.id] = result
            elif to_run:
                with ThreadPoolExecutor(max_workers=min(4, len(to_run))) as pool:
                    futures = {
                        pool.submit(self._execute_tool, b.name, ti): (b, key)
                        for (b, key, ti) in to_run
                    }
                    for future in futures:
                        block, key = futures[future]
                        result = future.result()
                        tool_cache[key] = result
                        results_by_id[block.id] = result

            # Harvest ids from every data result (deterministic block order) so a
            # view action in this same round can only reference emitted ids.
            for block in data_blocks:
                self._harvest_zone_ids(
                    results_by_id.get(block.id), known_zone_ids, known_category_ids
                )

            # --- Couche 4 — view-action whitelist (display-only), sequential ---
            for block in view_blocks:
                results_by_id[block.id] = self._apply_view_action(
                    dict(block.input), view_actions, known_zone_ids, known_category_ids
                )

            # Bundle every tool_result of the turn (original block order).
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(
                        results_by_id[block.id], ensure_ascii=False, default=str
                    ),
                }
                for block in tool_blocks
            ]
            messages.append({"role": "user", "content": tool_results})

        # Tool-turn budget exhausted without a final text answer.
        logger.warning("chatbot exceeded %d tool turns — fail-safe template", self._max_tool_turns)
        yield {
            "event": "answer",
            "content": LLM_ERROR_TEMPLATE,
            "tool_calls_made": tool_calls_made,
            "view_actions": view_actions,
            "blocked_reason": "max_tool_turns_exceeded",
        }

    @staticmethod
    def _tool_key(name: str, tool_input: dict[str, Any]) -> tuple[str, str]:
        """Canonical de-dup key for a data-tool call: name + argument fingerprint
        (order-independent). Two identical reads in one turn collapse to one."""
        return (name, json.dumps(tool_input, ensure_ascii=False, sort_keys=True))

    @staticmethod
    def _tool_status(name: str, tool_input: dict[str, Any]) -> Optional[dict[str, Any]]:
        """Honest activity event for a data read that is ABOUT TO run, or None
        for tools with no meaningful wait (get_signal_summary is a cached, near-
        instant lookup). Structured — the client localises the wording, so no
        French/English status text is hard-coded server-side, and it can never
        carry model prose (MIA-1 §2/C)."""
        if name in ("get_market_reading", "get_ob_diagnostic"):
            status: dict[str, Any] = {"event": "tool", "tool": name}
            instrument = tool_input.get("instrument")
            timeframe = tool_input.get("timeframe")
            if isinstance(instrument, str):
                status["instrument"] = instrument
            if isinstance(timeframe, str):
                status["timeframe"] = timeframe
            return status
        if name in ("get_economic_calendar", "get_publication"):
            # Calendar reads can replay engine measures (a real wait). Narrate an
            # honest, structured status; the client localises the wording. No
            # market/event field is needed for the label — the tool name suffices.
            return {"event": "tool", "tool": name}
        return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _safe_summary(self) -> dict[str, Any]:
        try:
            return self._summary_provider.get()
        except Exception as exc:  # never let summary failure abort the turn
            logger.warning("signal_summary provider failed: %s", exc)
            return {"instruments_tracked": []}

    def _build_system(self, signal_summary: dict[str, Any]) -> Any:
        """MIA-2 lever 1 — assemble the system prompt as [cached static prefix,
        variable signal block]. The static prefix carries cache_control:ephemeral
        so Anthropic caches it (tools + this block) once and reads it back on
        every subsequent turn; the signal_summary block sits AFTER the breakpoint
        so its 60s-TTL refresh never invalidates the cache. Content is identical
        to the old single-string prompt — only the summary's position changed.

        Falls back to a single plain string if the static prefix is somehow below
        the cache threshold (it is ~2k tokens, so this never triggers in practice)
        — same text, just uncached.
        """
        signal_text = SIGNAL_CONTEXT_TEMPLATE.format(
            signal_summary=json.dumps(signal_summary, ensure_ascii=False)
        )
        # MIA-4S — a restricted build appends its own stable blocks (scope of the
        # simulation, product knowledge). They sit AFTER the cached prefix and
        # BEFORE the variable signal block: byte-stable themselves, so they cache
        # too, and they can never edit or weaken the rules above them.
        extra = self._extra_system_blocks
        static_block = cache_block_for(SYSTEM_PROMPT_STATIC)
        if static_block is None:
            return "\n\n".join([SYSTEM_PROMPT_STATIC, *extra, signal_text])
        blocks: list[dict[str, Any]] = [static_block]
        for text in extra:
            blocks.append({"type": "text", "text": text})
        if blocks[-1] is not static_block:
            cached_tail = cache_block_for(blocks[-1]["text"])
            if cached_tail is not None:
                blocks[-1] = cached_tail
        blocks.append({"type": "text", "text": signal_text})
        return blocks

    @staticmethod
    def _truncate_history(
        conversation_history: Optional[list[dict[str, Any]]], max_len: int
    ) -> list[dict[str, Any]]:
        """MIA-2 lever 5a — keep at most ``max_len`` most-recent history messages,
        trimmed to start on a 'user' turn so the slice handed to the model stays a
        valid alternating transcript (Anthropic requires the first message to be
        'user'). Never touches the current user_message (appended by the caller)
        nor any validation — purely a context-size bound.
        """
        history = list(conversation_history or [])
        if max_len is None or max_len <= 0 or len(history) <= max_len:
            return history
        trimmed = history[-max_len:]
        # Drop leading non-'user' messages so the window opens on a user turn.
        while trimmed and (
            trimmed[0].get("role") if isinstance(trimmed[0], dict) else None
        ) != "user":
            trimmed = trimmed[1:]
        return trimmed

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
            # MIA-4S — an injected handler wins over the built-in dispatch, so a
            # restricted build (the landing's frozen illustration) serves its own
            # data through the SAME orchestrator and the SAME defence layers.
            handler = self._tool_handlers.get(name)
            if handler is not None:
                return handler(tool_input)
            # Defence in depth: a build may only EXECUTE what it DECLARES. The
            # model can already only call declared tools, but the built-in
            # dispatch below reaches live sources (engine, calendar) that a
            # restricted build must never touch — so an undeclared name stops
            # here instead of falling through. Production declares all seven, so
            # this changes nothing there.
            if not any(schema.get("name") == name for schema in self._tool_schemas):
                logger.warning("tool %s is not declared on this build — refused", name)
                return {"error": f"tool not available on this build: {name}"}
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
            if name == "list_markets":
                return self._tool_list_markets()
            if name == "get_economic_calendar":
                return self._tool_economic_calendar(tool_input)
            if name == "get_publication":
                return self._tool_publication(tool_input)
            return {"error": f"unknown tool: {name}"}
        except Exception as exc:
            logger.warning("tool %s failed: %s", name, exc)
            return {"error": f"tool execution failed: {exc}"}

    # ------------------------------------------------------------------ #
    # MIA-3 — read-only product-knowledge tools (markets, calendar).
    # The identifier lock is INTRINSIC: an unknown market / event_id makes the
    # tool return an explicit absence (found=false) — the model never receives
    # fabricated data to relay. Absence is data, not an error.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _tool_list_markets() -> dict[str, Any]:
        specs = market_registry.all_specs()
        markets = [
            {
                "id": s.id,
                "label": s.label,
                "type": s.type,
                "price_decimals": s.price_decimals,
                "timeframes": list(s.timeframes),
            }
            for s in specs
        ]
        return {"markets": markets, "timeframes": list(SUPPORTED_TIMEFRAMES)}

    def _tool_economic_calendar(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        svc = self._calendar()
        if svc is None:
            return {"error": "calendar service unavailable"}
        market = tool_input.get("market")
        if isinstance(market, str) and market:
            # Identifier lock: reject a market the registry never emitted rather
            # than silently returning the unfiltered feed as if it matched.
            if not market_registry.has(market):
                return {"found": False, "reason": "unknown_market", "market": market}
        else:
            market = None
        horizon = tool_input.get("horizon")
        if horizon not in ("upcoming", "recent", "both"):
            horizon = "upcoming"
        lookahead = _CAL_WEEK_MIN if horizon in ("upcoming", "both") else 0
        lookback = _CAL_WEEK_MIN if horizon in ("recent", "both") else 0
        resp = svc.get_calendar(lookahead_minutes=lookahead, lookback_minutes=lookback)
        events = [
            _slim_event(e)
            for e in resp.events
            if market is None or market in (getattr(e, "markets", None) or [])
        ]
        return {"market": market, "horizon": horizon, "count": len(events), "events": events}

    def _tool_publication(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        svc = self._calendar()
        if svc is None:
            return {"error": "calendar service unavailable"}
        event_id = tool_input.get("event_id")
        if not isinstance(event_id, str) or not event_id.strip():
            return {"found": False, "reason": "bad_event_id"}
        resp = svc.get_event(event_id)
        if not resp.events:
            # Intrinsic id lock — an event_id the engine never emitted resolves to
            # nothing. Report the absence; do NOT fabricate a value or date.
            return {"found": False, "reason": "unknown_event_id", "event_id": event_id}
        ev = resp.events[0]
        detail = _slim_event(ev)
        detail["actual_initial"] = getattr(ev, "actual_initial", None)
        detail["revised"] = getattr(ev, "revised", False)
        detail["value_unit"] = getattr(ev, "value_unit", None)
        detail["license_label"] = getattr(ev, "license_label", None)
        series = getattr(ev, "value_series", None) or []
        detail["value_series"] = [
            {"period": getattr(p, "period", None), "value": getattr(p, "value", None)}
            for p in series
        ]
        detail["measures"] = self._publication_measures(event_id)
        return {"found": True, "event": detail}

    @staticmethod
    def _publication_measures(event_id: str) -> Optional[dict[str, Any]]:
        """Engine measures for a publication's recurring series, or None when not
        measurable (fewer than the minimum sample, or the series is not measured).
        None is an HONEST absence the model must restitute, never fill in."""
        parts = event_id.split(":")
        event_key = parts[1] if len(parts) >= 2 else None
        if not event_key:
            return None
        try:
            from src.intelligence.publication_measures import (
                MEASURED_MARKETS,
                load_default_measures,
            )
        except Exception:
            return None
        market = MEASURED_MARKETS.get(event_key)
        if market is None:
            return None
        try:
            measures = load_default_measures(event_key, market)
        except Exception as exc:
            logger.warning("publication measures failed for %s: %s", event_key, exc)
            return None
        if measures is None or not measures.has_any():
            return None
        return measures.model_dump(mode="json")

    def _calendar(self) -> Optional[Any]:
        """Return the calendar service, building it lazily once (same pattern as
        the /api/calendar route). A build failure latches so a broken calendar
        degrades every calendar turn to an honest absence, not a crash."""
        if self._calendar_service is not None:
            return self._calendar_service
        if self._calendar_failed:
            return None
        try:
            from src.intelligence.calendar_service import CalendarService

            self._calendar_service = CalendarService()
            return self._calendar_service
        except Exception as exc:  # never let a calendar build abort a turn
            logger.warning("calendar service build failed: %s", exc)
            self._calendar_failed = True
            return None

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


def _iso(value: Any) -> Optional[str]:
    """ISO-8601 string for a datetime, passthrough for a str, None otherwise."""
    if value is None:
        return None
    if isinstance(value, str):
        return value
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)


def _slim_event(ev: Any) -> dict[str, Any]:
    """Compact, JSON-safe view of a CalendarEvent for a tool result — the core
    facts M.I.A needs, without the full model (keeps the tool payload small).
    `actual`/`previous` are echoed VERBATIM (no rounding, no reinterpretation);
    `actual_state` says whether `actual` is actually a published figure."""
    return {
        "event_id": getattr(ev, "event_id", None),
        "event": getattr(ev, "event", None),
        "currency": getattr(ev, "currency", None),
        "organism": getattr(ev, "organism", None),
        "scheduled_at": _iso(getattr(ev, "scheduled_at", None)),
        "actual_state": getattr(ev, "actual_state", None),
        "actual": getattr(ev, "actual", None),
        "previous": getattr(ev, "previous", None),
        "value_unit": getattr(ev, "value_unit", None),
        "markets": list(getattr(ev, "markets", None) or []),
    }


__all__ = [
    "Chatbot",
    "ChatResponse",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_TIMEOUT_S",
    "MAX_MODEL_HISTORY",
    "MAX_TOOL_TURNS",
    "SIGNAL_CONTEXT_TEMPLATE",
    "SYSTEM_PROMPT_STATIC",
    "SYSTEM_PROMPT_TEMPLATE",
    "TOOL_SCHEMAS",
]
