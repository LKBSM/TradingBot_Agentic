"""Chantier 4 — Couche 3 — output forbidden-tokens filter (doc §4.1/§4.2).

Last line of defence: AFTER Haiku produces a final text answer, it is scanned
against the 4 forbidden-token categories. A hit means the answer drifted into
recommendation/judgement territory → the chatbot replaces it with the
pedagogical fallback (OUTPUT_CONTAMINATED_TEMPLATE) instead of leaking it.

Design
------
- Re-uses ``ALL_FORBIDDEN_TOKENS`` (via ``FORBIDDEN_TOKENS_BY_CATEGORY``) — no
  parallel token set.
- Matching is accent-/case-insensitive: both the LLM text and each token are
  passed through ``normalize_text`` before a word-boundary (``\\b…\\b``) regex
  match (consistent with ``contains_forbidden_tokens`` of Chantier 2).
- Priority order action_trading → recommandation → jugement_moment →
  jugement_risque (consistent with AdversarialFilter). First category that
  matches wins; every matching token of that category is reported.

Homonym note (revised — MIA-5)
-----------------------------
Matching is accent-insensitive, so accented adjectives collapse onto their
unaccented homonyms (``risqué`` → ``risque``). That blanket over-block was
originally deliberate on the OUTPUT path — until it was measured: three
legitimate answers in a row ("la gestion **du risque**…") and a factual macro
answer ("le **coût d'opportunité** de détenir de l'or") were replaced by the
contamination template, so the user saw an unexplained refusal instead of a
correct, compliant answer. An over-block is not free: it destroys the product's
honesty just as surely as an under-block destroys its compliance.

Two narrow, evidence-backed carve-outs are therefore applied — and nothing else:

- ``ACCENT_SENSITIVE_TOKENS`` (``risqué``): searched in the accent-PRESERVING
  text, so the adjective is still caught while the noun ``risque`` is not. The
  unaccented adjective stays covered by the phrase tokens ("c'est risque",
  "trop risque", "très risque", "assez risque", "plutôt risque", "moins/plus
  risque"), so no phrasing of the judgement escapes.
- ``HOMONYM_SAFE_EXPRESSIONS`` ("coût d'opportunité"): the fixed macro-economic
  term is neutralised before the search; the bare ``opportunité`` stays
  forbidden everywhere else.

Both are exact and enumerable: no token was removed from any category, and every
other match still wins. The homonym EXCLUSIONS of the INPUT path (Couche 1) are
unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from src.intelligence.chatbot.constants import (
    ACCENT_SENSITIVE_TOKENS,
    FORBIDDEN_TOKENS_BY_CATEGORY,
    HOMONYM_SAFE_EXPRESSIONS,
    fold_case,
    normalize_text,
)

ForbiddenCategory = Literal[
    "action_trading",
    "recommandation",
    "jugement_moment",
    "jugement_risque",
]

# Priority order — mirrors the forbidden-token category declaration.
_CATEGORY_ORDER: tuple[ForbiddenCategory, ...] = (
    "action_trading",
    "recommandation",
    "jugement_moment",
    "jugement_risque",
)


@dataclass(frozen=True)
class OutputCheckResult:
    """Outcome of scanning an LLM answer.

    Attributes:
        contaminated: True if a forbidden token was found.
        category: the winning category (priority order) or None.
        matched_tokens: the normalised tokens that matched, sorted (deterministic).
    """

    contaminated: bool
    category: Optional[ForbiddenCategory] = None
    matched_tokens: tuple[str, ...] = ()


class OutputFilter:
    """Scans an LLM answer for forbidden tokens (Couche 3)."""

    def __init__(self) -> None:
        # Per category, an ordered list of (reported_token, compiled_regex,
        # accent_sensitive). Accent-INSENSITIVE patterns are de-duplicated on the
        # normalised form (e.g. "achète"/"achete" collapse); the few
        # accent-SENSITIVE ones keep their accents and are matched against the
        # accent-preserving text (see the module docstring).
        self._patterns: list[
            tuple[ForbiddenCategory, list[tuple[str, re.Pattern[str], bool]]]
        ] = []
        for category in _CATEGORY_ORDER:
            seen: dict[str, tuple[re.Pattern[str], bool]] = {}
            for token in FORBIDDEN_TOKENS_BY_CATEGORY[category]:
                accent_sensitive = token in ACCENT_SENSITIVE_TOKENS
                key = fold_case(token) if accent_sensitive else normalize_text(token)
                if not key or key in seen:
                    continue
                seen[key] = (re.compile(r"\b" + re.escape(key) + r"\b"), accent_sensitive)
            self._patterns.append(
                (category, [(k, p, a) for k, (p, a) in seen.items()])
            )
        # Fixed expressions in which a forbidden token is not a judgement. Both
        # forms are pre-computed so neutralisation costs nothing per call.
        self._safe_expressions: list[tuple[str, str]] = [
            (normalize_text(expr), fold_case(expr)) for expr in HOMONYM_SAFE_EXPRESSIONS
        ]

    def check(self, llm_response: str) -> OutputCheckResult:
        if not llm_response or not llm_response.strip():
            return OutputCheckResult(contaminated=False)

        normalized = normalize_text(llm_response)
        accented = fold_case(llm_response)
        # Blank out the safe fixed expressions in BOTH views before searching, so
        # « coût d'opportunité » cannot surface the bare « opportunité ». Replaced
        # by spaces of the same length: offsets and word boundaries are preserved.
        for norm_expr, accent_expr in self._safe_expressions:
            normalized = normalized.replace(norm_expr, " " * len(norm_expr))
            accented = accented.replace(accent_expr, " " * len(accent_expr))

        for category, token_patterns in self._patterns:
            matched = [
                token
                for token, pattern, accent_sensitive in token_patterns
                if pattern.search(accented if accent_sensitive else normalized)
            ]
            if matched:
                return OutputCheckResult(
                    contaminated=True,
                    category=category,
                    matched_tokens=tuple(sorted(matched)),
                )

        return OutputCheckResult(contaminated=False)


__all__ = ["ForbiddenCategory", "OutputCheckResult", "OutputFilter"]
