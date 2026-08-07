"""SC-2 — Scanner conversationnel : traduire une phrase en conditions de la palette.

M.I.A EST UN TRADUCTEUR VERS LA PALETTE FERMÉE. ELLE N'EST JAMAIS JUGE.

  · elle transforme une phrase en conditions qui EXISTENT DÉJÀ dans la palette
    (``conditions_scanner.PALETTE``) ;
  · toute sortie non conforme (type hors palette, valeur hors domaine, champ
    inapplicable) est REJETÉE PAR LE CODE, avant toute évaluation — même verrou
    que l'``id`` des zones dans le chatbot ;
  · elle ne consulte AUCUNE donnée de marché pour traduire ;
  · elle n'ordonne rien, ne classe rien, ne recommande rien : une demande de
    classement / prédiction / conseil produit un REFUS, jamais une recherche ;
  · toute valeur DÉDUITE d'un mot vague (« récemment » → 10 bougies) est
    annoncée comme supposition, jamais silencieuse.

Deux barrières successives, la sortie du LLM n'est jamais crue sur parole :
  1. ``tool_choice`` forcé + ``input_schema`` strict (enums = valeurs exactes de
     la palette) → première contrainte au niveau du tool-call ;
  2. :func:`sanitize_translation` re-valide chaque condition contre la palette et
     les domaines côté serveur AVANT qu'elle ne devienne une carte. Une condition
     non conforme est écartée et nommée, jamais coercée vers une voisine.
Filet ultime : même conforme, la liste passe ensuite par le modèle
``ScanCondition`` du endpoint de scan (422 sinon) — rien d'invalide n'atteint
l'évaluateur.

Le client Anthropic est duck-typé (seul ``client.messages.create(...)`` est
utilisé), comme :class:`Chatbot` / :class:`LLMNarrativeEngine` — trivial à
stubber en test.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.intelligence.conditions_scanner import ALLOWED_CONDITION_TYPES, PALETTE

logger = logging.getLogger(__name__)

# Aligned with the chatbot (Chantier 4) — cheapest capable model, low latency.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_S = 20.0

TOOL_NAME = "emit_scanner_translation"

# Outcomes returned to the route (stable machine keys — the webapp localises).
OUTCOME_TRANSLATED = "translated"  # ≥1 condition, nothing left untranslatable
OUTCOME_PARTIAL = "partial"        # ≥1 condition AND ≥1 untranslatable fragment
OUTCOME_NONE = "none"              # nothing usable understood (0 conditions)
OUTCOME_REFUSED = "refused"        # ranking / prediction / advice → not translated
OUTCOME_EMPTY = "empty"            # blank input
OUTCOME_ERROR = "error"            # LLM/transport failure — textbox stays usable

REFUSAL_KINDS = ("ranking", "prediction", "recommendation")

# Categories a fragment can be refused under (stable keys → webapp localises the
# honest sentence). ``unsupported`` is the catch-all for a model condition that
# fails the palette gate (should be rare — the schema constrains the enum).
UNTRANSLATABLE_CATEGORIES = (
    "indicator",               # RSI, MACD, stochastique, moyennes mobiles, Bollinger, ATR…
    "volume",                  # volume / OBV
    "fibonacci",               # retracements / extensions
    "candle_pattern",          # engulfing, doji, pin bar…
    "classic_support_resistance",  # S/R horizontaux classiques (≠ zones SMC)
    "round_number",            # chiffres ronds / round numbers
    "sentiment",               # sentiment / cot / positionnement
    "fundamental",             # macro / news comme condition scannable
    "other",                   # hors périmètre, non classable
    "unsupported",             # sortie modèle rejetée par le gate palette
)

CONTROL_FIELDS = (
    "direction", "max_bars", "trend", "phase", "volatility", "proximity_pct",
    "side", "zone_kind", "event", "age_bucket", "third", "eq_kind", "session",
    "relation", "max_touches",
)


# ── Palette-derived domains (single source of truth = conditions_scanner) ─────
#
# For each condition type: which control fields apply, and the exact value domain
# of each. Any field a translation sets that is NOT here (for that type), or a
# value outside the domain, is a mistranslation → the condition is rejected.
def _build_control_domains() -> Dict[str, Dict[str, Tuple[Any, ...]]]:
    domains: Dict[str, Dict[str, Tuple[Any, ...]]] = {}
    for entry in PALETTE:
        per_type: Dict[str, Tuple[Any, ...]] = {}
        for control in entry.get("controls", []):
            per_type[control["name"]] = tuple(control["values"])
        domains[entry["type"]] = per_type
    return domains


CONTROL_DOMAINS: Dict[str, Dict[str, Tuple[Any, ...]]] = _build_control_domains()


# ── Refusal pre-filter (deterministic, no LLM) ───────────────────────────────
#
# Flagrant ranking / prediction / advice requests are refused BEFORE any LLM
# call — cheaper and guaranteed. The model's own ``refusal`` field is a second
# net for subtler phrasings. Patterns run over accent-folded, lowercased text.
def _fold(text: str) -> str:
    stripped = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in stripped if not unicodedata.combining(c))
    return stripped.lower()


_REFUSAL_PATTERNS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    # ranking / quality ordering ("les meilleurs", "best", "top setups",
    # "le plus sûr / rentable", "lequel trader", "classe les marchés").
    ("ranking", re.compile(
        r"\b(meilleur|meilleurs|meilleure|best|top\s*\d*\s*(setup|trade|march)"
        r"|plus\s+(sur|surs|sure|rentable|fiable|solide|safe|profitable)"
        r"|lequel|laquelle|which\s+(one|market|pair|setup)|classe(r|ment)?"
        r"|rank(ing|ed)?|trie(r|z)?|prioris|hi[ée]rarchi)\b"
    )),
    # prediction ("où va le prix", "va monter", "will go up", "prochain
    # mouvement", "target/objectif de prix", "prédit").
    ("prediction", re.compile(
        r"\b(ou\s+va|va[\s-]*t[\s-]*il|va\s+(monter|descendre|baisser|grimper|chuter)"
        r"|montera|descendra|prochain\s+mouvement|prevoir|prevision|predi(re|t|ction)"
        r"|will\s+(go|rise|drop|fall|move|pump|dump)|price\s+target|objectif\s+de\s+prix"
        r"|forecast|prediction)\b"
    )),
    # recommendation / personalised advice ("devrais-je", "should I", "je
    # devrais", "recommande", "conseil", "dois-je acheter/vendre", "quoi trader").
    ("recommendation", re.compile(
        r"\b(devrais[\s-]*je|je\s+devrais|should\s+i|dois[\s-]*je|recommand"
        r"|conseil(le|les|ler)?|quoi\s+(trader|acheter|vendre)|que\s+(trader|dois)"
        r"|what\s+should\s+i\s+(trade|buy|sell)|acheter\s+ou\s+vendre|buy\s+or\s+sell)\b"
    )),
)


def detect_refusal(text: str) -> Optional[str]:
    """Return the refusal kind (ranking/prediction/recommendation) or None.

    Deterministic screen run before the LLM. First matching bucket wins so the
    reported kind is stable.
    """
    if not text or not text.strip():
        return None
    folded = _fold(text)
    for kind, pattern in _REFUSAL_PATTERNS:
        if pattern.search(folded):
            return kind
    return None


# ── Server-side re-validation of the model output ────────────────────────────
def sanitize_condition(raw: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Validate ONE model-proposed condition against the closed palette.

    Returns ``(clean, None)`` when the condition is representable — ``clean`` is
    a wire-shaped dict holding only ``type`` plus the control fields that APPLY
    to that type, each within its exact domain. Returns ``(None, reason)`` when
    it must be rejected (out of palette / bad value) — never coerced.

    Inapplicable control fields are DROPPED, not rejected: a control that has no
    effect on a given type carries no user intent that changes the search, so
    dropping it is not "substituting an approaching condition".
    """
    if not isinstance(raw, dict):
        return None, "malformed"
    ctype = raw.get("type")
    if not isinstance(ctype, str) or ctype not in ALLOWED_CONDITION_TYPES:
        return None, "out_of_palette"
    allowed = CONTROL_DOMAINS.get(ctype, {})
    clean: Dict[str, Any] = {"type": ctype}
    for field, domain in allowed.items():
        if field not in raw or raw[field] is None:
            continue  # omitted → the ScanCondition model supplies the default
        value = raw[field]
        # Booleans must not slip through as ints (Python: True == 1).
        if isinstance(value, bool):
            return None, "bad_value"
        if value not in domain:
            return None, "bad_value"
        clean[field] = value
    return clean, None


def _sanitize_untranslatable(raw: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    fragment = raw.get("fragment")
    category = raw.get("category")
    if category not in UNTRANSLATABLE_CATEGORIES:
        category = "other"
    frag = fragment if isinstance(fragment, str) and fragment.strip() else None
    return {"fragment": (frag.strip()[:160] if frag else None), "category": category}


def _sanitize_assumption(raw: Any, valid_types: set[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    ctype = raw.get("condition_type")
    control = raw.get("control")
    value = raw.get("value")
    source = raw.get("source_phrase")
    # Only surface an assumption that points at a condition we actually kept and a
    # real control field — otherwise it would describe something not on screen.
    if ctype not in valid_types:
        return None
    if control not in CONTROL_FIELDS:
        return None
    return {
        "condition_type": ctype,
        "control": control,
        "value": None if value is None else str(value)[:60],
        "source_phrase": (str(source)[:120] if isinstance(source, str) and source.strip() else None),
    }


def _dedupe(conditions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for cond in conditions:
        key = repr(sorted(cond.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(cond)
    return out


def sanitize_translation(payload: Any) -> Dict[str, Any]:
    """Turn a raw model tool-input into the validated, wire-safe translation.

    This is the authoritative gate: it never trusts the model. A refusal short
    -circuits everything. Every proposed condition is re-validated against the
    palette; rejected ones are surfaced as ``unsupported`` fragments (honesty),
    never silently dropped into thin air.
    """
    empty = {
        "outcome": OUTCOME_NONE,
        "refusal": None,
        "conditions": [],
        "assumptions": [],
        "untranslatable": [],
    }
    if not isinstance(payload, dict):
        return empty

    # 1) Refusal wins outright.
    refusal = payload.get("refusal")
    if isinstance(refusal, dict) and refusal.get("kind") in REFUSAL_KINDS:
        return {
            "outcome": OUTCOME_REFUSED,
            "refusal": {"kind": refusal["kind"]},
            "conditions": [],
            "assumptions": [],
            "untranslatable": [],
        }

    # 2) Validate each proposed condition.
    conditions: List[Dict[str, Any]] = []
    untranslatable: List[Dict[str, Any]] = []
    raw_conditions = payload.get("conditions")
    if isinstance(raw_conditions, list):
        for raw in raw_conditions:
            clean, reason = sanitize_condition(raw)
            if clean is not None:
                conditions.append(clean)
            elif reason == "out_of_palette" and isinstance(raw, dict):
                # A type the model invented outside the palette — name it rather
                # than swallow it. Bad-value / malformed are silent drops (the
                # model contradicted its own schema; nothing meaningful to show).
                untranslatable.append({
                    "fragment": str(raw.get("type"))[:160],
                    "category": "unsupported",
                })
    conditions = _dedupe(conditions)

    # 3) Untranslatable fragments the model reported.
    raw_untr = payload.get("untranslatable")
    if isinstance(raw_untr, list):
        for raw in raw_untr:
            item = _sanitize_untranslatable(raw)
            if item is not None:
                untranslatable.append(item)

    # 4) Assumptions — only those pointing at a condition we kept.
    valid_types = {c["type"] for c in conditions}
    assumptions: List[Dict[str, Any]] = []
    raw_assumptions = payload.get("assumptions")
    if isinstance(raw_assumptions, list):
        for raw in raw_assumptions:
            item = _sanitize_assumption(raw, valid_types)
            if item is not None:
                assumptions.append(item)

    # 5) Outcome.
    if conditions and untranslatable:
        outcome = OUTCOME_PARTIAL
    elif conditions:
        outcome = OUTCOME_TRANSLATED
    else:
        outcome = OUTCOME_NONE

    return {
        "outcome": outcome,
        "refusal": None,
        "conditions": conditions,
        "assumptions": assumptions,
        "untranslatable": untranslatable,
    }


# ── Anthropic tool schema (forced) ───────────────────────────────────────────
def _condition_item_schema() -> Dict[str, Any]:
    """One condition object — enums pinned to the exact palette domains."""
    return {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": sorted(ALLOWED_CONDITION_TYPES),
                "description": "Clé EXACTE d'une condition de la palette fermée.",
            },
            "direction": {"type": "string", "enum": ["any", "bullish", "bearish"]},
            "trend": {"type": "string", "enum": ["bullish", "bearish", "indeterminate"]},
            "phase": {"type": "string", "enum": ["accumulation", "trend", "ranging", "expansion"]},
            "volatility": {"type": "string", "enum": ["low", "normal", "elevated"]},
            "side": {"type": "string", "enum": ["any", "bsl", "ssl"]},
            "zone_kind": {"type": "string", "enum": ["any", "ob", "fvg"]},
            "event": {"type": "string", "enum": ["bos_up", "bos_down", "choch_up", "choch_down"]},
            "age_bucket": {"type": "string", "enum": ["lt10", "10to50", "gt50"]},
            "third": {"type": "string", "enum": ["bottom", "middle", "top"]},
            "eq_kind": {"type": "string", "enum": ["any", "highs", "lows"]},
            "session": {"type": "string", "enum": ["asia", "london", "new_york", "overlap"]},
            "relation": {"type": "string", "enum": ["same", "opposite"]},
            "max_bars": {"type": "integer", "enum": [5, 10, 20, 50]},
            "max_touches": {"type": "integer", "enum": [1, 2, 3]},
            "proximity_pct": {"type": "number", "enum": [0.1, 0.25, 0.5, 1.0]},
        },
        "required": ["type"],
        "additionalProperties": False,
    }


def build_tool_schema() -> Dict[str, Any]:
    return {
        "name": TOOL_NAME,
        "description": (
            "Émets la traduction structurée de la phrase du trader en conditions "
            "de la palette fermée du scanner. N'invente jamais de condition hors "
            "palette. Si la demande est un classement, une prédiction ou un "
            "conseil, remplis 'refusal' et laisse 'conditions' vide."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "conditions": {
                    "type": "array",
                    "description": (
                        "Conditions comprises, uniquement des clés de la palette. "
                        "Une condition par intention distincte."
                    ),
                    "items": _condition_item_schema(),
                },
                "assumptions": {
                    "type": "array",
                    "description": (
                        "Chaque valeur DÉDUITE d'un mot vague (ex. « récemment » → "
                        "max_bars=10). N'invente pas de condition ici : décris "
                        "seulement une valeur choisie pour une condition comprise."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "condition_type": {"type": "string", "enum": sorted(ALLOWED_CONDITION_TYPES)},
                            "control": {"type": "string", "enum": list(CONTROL_FIELDS)},
                            "value": {"type": "string"},
                            "source_phrase": {"type": "string"},
                        },
                        "required": ["condition_type", "control", "value"],
                        "additionalProperties": False,
                    },
                },
                "untranslatable": {
                    "type": "array",
                    "description": (
                        "Chaque fragment de la phrase qui ne correspond à AUCUNE "
                        "condition de la palette (RSI, volume, Fibonacci, figures "
                        "en chandelier, supports/résistances classiques, chiffres "
                        "ronds, sentiment, macro). Ne propose jamais une condition "
                        "approchante à la place."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "fragment": {"type": "string"},
                            "category": {"type": "string", "enum": list(UNTRANSLATABLE_CATEGORIES)},
                        },
                        "required": ["fragment", "category"],
                        "additionalProperties": False,
                    },
                },
                "refusal": {
                    "type": ["object", "null"],
                    "description": (
                        "Non nul UNIQUEMENT si la phrase demande un classement de "
                        "qualité, une prédiction de prix ou un conseil personnalisé."
                    ),
                    "properties": {
                        "kind": {"type": "string", "enum": list(REFUSAL_KINDS)},
                    },
                    "required": ["kind"],
                    "additionalProperties": False,
                },
            },
            "required": ["conditions", "assumptions", "untranslatable", "refusal"],
            "additionalProperties": False,
        },
    }


SYSTEM_PROMPT = (
    "Tu es M.I.A, un TRADUCTEUR. Ton seul rôle : transformer la phrase d'un "
    "trader en conditions d'une PALETTE FERMÉE de scanner SMC (Smart Money "
    "Concepts). Tu n'es JAMAIS juge.\n\n"
    "RÈGLES ABSOLUES :\n"
    "1. N'émets QUE des conditions dont la clé figure dans l'énumération du "
    "schéma de l'outil. Jamais une condition inventée.\n"
    "2. Ne consulte aucune donnée de marché. Tu traduis une phrase, rien de "
    "plus. Tu ne dis pas si les conditions sont réunies quelque part.\n"
    "3. Si un fragment ne correspond à AUCUNE condition (RSI, MACD, moyennes "
    "mobiles, Bollinger, volume, Fibonacci, figures en chandelier, "
    "supports/résistances horizontaux classiques, chiffres ronds, sentiment, "
    "macro/news), mets-le dans 'untranslatable' avec sa catégorie. NE choisis "
    "JAMAIS une condition qui ressemblerait vaguement.\n"
    "4. Si tu déduis une valeur d'un mot vague (« récemment » → 10 bougies, "
    "« proche » → une distance), déclare-la dans 'assumptions'.\n"
    "5. Si la phrase demande un CLASSEMENT de qualité (« les meilleurs setups », "
    "« le marché le plus sûr »), une PRÉDICTION (« où va le prix »), ou un "
    "CONSEIL (« qu'est-ce que je devrais trader »), remplis 'refusal' avec le "
    "'kind' adéquat (ranking / prediction / recommendation) et laisse "
    "'conditions' vide. Le scanner ne trie jamais les résultats.\n"
    "6. La phrase peut être en français ou en anglais ; les fragments que tu "
    "cites dans 'untranslatable'/'assumptions' doivent reprendre les mots de "
    "l'utilisateur.\n\n"
    "Repères de vocabulaire (non exhaustif) : « Order Block/OB » → price_in_ob ; "
    "« jamais testé/vierge » → zone_untested ; « FVG/Fair Value Gap » → "
    "price_in_fvg ; « tendance haussière/baissière » → trend_is ; « unité "
    "supérieure d'accord / le 1 h d'accord » → higher_tf_agrees ; « BOS récent » "
    "→ bos_recent_confirmed ; « CHOCH récent » → choch_recent_confirmed ; "
    "« liquidité prise/balayée » → liquidity_swept_recent ; « poche cassée » → "
    "liquidity_broken_recent ; « creux/sommets égaux, EQH/EQL » → "
    "equal_levels_present ; « phase d'expansion/accumulation/range » → "
    "market_phase_is ; « volatilité faible/élevée » → volatility_is ; « session "
    "de Londres/New York/Asie » → session_is ; « tiers haut/bas du range » → "
    "price_in_range_third."
)


class ScannerTranslator:
    """Phrase → conditions de la palette. Fail-safe, jamais bloquant."""

    def __init__(
        self,
        anthropic_client: Any,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        breaker: Optional[Any] = None,
    ) -> None:
        self._client = anthropic_client
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_s = timeout_s
        self._breaker = breaker
        self._tool = build_tool_schema()

    def translate(self, text: str, locale: str = "fr") -> Dict[str, Any]:
        cleaned = (text or "").strip()
        if not cleaned:
            return {
                "outcome": OUTCOME_EMPTY,
                "refusal": None,
                "conditions": [],
                "assumptions": [],
                "untranslatable": [],
            }

        # Deterministic refusal screen — no LLM call on a flagrant ask.
        kind = detect_refusal(cleaned)
        if kind is not None:
            return {
                "outcome": OUTCOME_REFUSED,
                "refusal": {"kind": kind},
                "conditions": [],
                "assumptions": [],
                "untranslatable": [],
            }

        try:
            payload = self._call_llm(cleaned, locale)
        except Exception as exc:  # timeout / rate limit / circuit open / network
            logger.warning("scanner translate LLM call failed: %s — fail-safe", exc)
            return {
                "outcome": OUTCOME_ERROR,
                "refusal": None,
                "conditions": [],
                "assumptions": [],
                "untranslatable": [],
            }

        return sanitize_translation(payload)

    def _call_llm(self, text: str, locale: str) -> Dict[str, Any]:
        lang = "en" if str(locale).lower().startswith("en") else "fr"
        user = f"Langue de l'utilisateur : {lang}.\n\nPhrase du trader :\n{text}"

        def _invoke() -> Dict[str, Any]:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user}],
                tools=[self._tool],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                timeout=self._timeout_s,
            )
            return _extract_tool_input(response)

        if self._breaker is not None:
            return self._breaker.call(_invoke)
        return _invoke()


def _extract_tool_input(response: Any) -> Dict[str, Any]:
    """Pull the forced tool-call input out of an Anthropic response."""
    for block in getattr(response, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == TOOL_NAME:
            data = getattr(block, "input", None)
            if isinstance(data, dict):
                return data
    raise ValueError("model did not return the forced scanner-translation tool call")


__all__ = [
    "ScannerTranslator",
    "build_tool_schema",
    "sanitize_translation",
    "sanitize_condition",
    "detect_refusal",
    "CONTROL_DOMAINS",
    "REFUSAL_KINDS",
    "UNTRANSLATABLE_CATEGORIES",
    "OUTCOME_TRANSLATED",
    "OUTCOME_PARTIAL",
    "OUTCOME_NONE",
    "OUTCOME_REFUSED",
    "OUTCOME_EMPTY",
    "OUTCOME_ERROR",
    "TOOL_NAME",
    "DEFAULT_MODEL",
]
