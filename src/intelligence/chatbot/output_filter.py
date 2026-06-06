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

Homonym note
------------
Because matching is accent-insensitive, accented adjectives collapse onto their
unaccented homonyms (``risqué`` → ``risque``). On the OUTPUT path this is an
intentional *over*-block: if Haiku emits "risque" even as a noun, we prefer the
safe fallback to leaking a niveau-1.5 violation. The homonym EXCLUSIONS still
matter on the INPUT path (Couche 1), where over-blocking a user's descriptive
question would degrade UX. Different trade-off, different layer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from src.intelligence.chatbot.constants import (
    FORBIDDEN_TOKENS_BY_CATEGORY,
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
        # Per category, an ordered list of (normalised_token, compiled_regex),
        # de-duplicated on the normalised form (e.g. "achète"/"achete" collapse).
        self._patterns: list[tuple[ForbiddenCategory, list[tuple[str, re.Pattern[str]]]]] = []
        for category in _CATEGORY_ORDER:
            seen: dict[str, re.Pattern[str]] = {}
            for token in FORBIDDEN_TOKENS_BY_CATEGORY[category]:
                norm = normalize_text(token)
                if not norm or norm in seen:
                    continue
                seen[norm] = re.compile(r"\b" + re.escape(norm) + r"\b")
            self._patterns.append((category, list(seen.items())))

    def check(self, llm_response: str) -> OutputCheckResult:
        if not llm_response or not llm_response.strip():
            return OutputCheckResult(contaminated=False)

        normalized = normalize_text(llm_response)

        for category, token_patterns in self._patterns:
            matched = [
                norm for norm, pattern in token_patterns if pattern.search(normalized)
            ]
            if matched:
                return OutputCheckResult(
                    contaminated=True,
                    category=category,
                    matched_tokens=tuple(sorted(matched)),
                )

        return OutputCheckResult(contaminated=False)


__all__ = ["ForbiddenCategory", "OutputCheckResult", "OutputFilter"]
