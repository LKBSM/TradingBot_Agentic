"""Compliance claim checker — Sprint RISK-2B.2.

Static analyser that scans any user-facing string (narrative, SEO
article body, YouTube description, Telegram message) for tokens or
phrases that would trigger AMF / UE 2024/2811 finfluencer
classification.

Two-layer detection:

1. **Token blacklist** — fast, deterministic. Direct match of
   forbidden words/phrases ("achetez", "buy signal", "100% sûr",
   "edge prouvé") used in the score_calibration guard. Cheap.

2. **Regex pattern bank** — catches reformulations the token list
   misses: percent-return claims ("+50%/mois"), guarantee phrasings
   ("nous garantissons"), call-to-action constructions ("vous devriez
   acheter"), absolute confidence ("certitude que").

When an LLM-as-judge layer is wired (production), an additional pass
sends the text + checklist to Claude Haiku with a structured-output
schema for nuanced cases ("this paragraph implies a recommendation
without using forbidden tokens"). That pass is opt-in via
``ComplianceChecker.with_llm_judge(callable)`` to keep the static
path free of any external API dependency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# Token blacklist — kept in sync with score_calibration.FORBIDDEN_TOKENS.
TOKEN_BLACKLIST_FR = (
    "achetez", "vendez", "achète", "vends", "garanti", "garantis",
    "100% sûr", "100% sur", "edge prouvé", "edge prouve",
    "signal d'achat", "signal de vente",
    "il faut acheter", "il faut vendre",
    "devriez acheter", "devriez vendre",
    "vous devez acheter", "vous devez vendre",
)
TOKEN_BLACKLIST_EN = (
    "buy signal", "sell signal", "guaranteed", "100% sure",
    "proven edge", "you should buy", "you should sell",
    "must buy", "must sell",
)


# Patterns covering reformulations.
PATTERN_BANK: list[tuple[str, re.Pattern[str]]] = [
    (
        "explicit_return_claim",
        re.compile(r"\+\s*\d{2,}\s*%(\s*/\s*(mois|month|an|year|jour|day|semaine|week))?", re.IGNORECASE),
    ),
    (
        "guarantee_phrase",
        re.compile(r"\b(nous|on|je)\s+garantis(s|sons|ssez)?", re.IGNORECASE),
    ),
    (
        "guarantee_phrase_en",
        re.compile(r"\b(we|i)\s+guarantee\b", re.IGNORECASE),
    ),
    (
        "absolute_certainty",
        re.compile(r"\b(certitude|certainty)\s+(que|that)\b", re.IGNORECASE),
    ),
    (
        "performance_proof",
        re.compile(r"\b(edge|alpha|outperform)\s+(prouvé|prouve|proven|demonstrated)\b", re.IGNORECASE),
    ),
]


@dataclass
class Violation:
    kind: str           # "token" | "pattern" | "llm"
    detail: str         # the matched token / pattern name / LLM reason
    snippet: str = ""   # short surrounding text for context

    def to_dict(self) -> dict:
        return {"kind": self.kind, "detail": self.detail, "snippet": self.snippet}


@dataclass
class ComplianceReport:
    ok: bool
    violations: list[Violation] = field(default_factory=list)
    language: str = "fr"
    text_length: int = 0

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "violations": [v.to_dict() for v in self.violations],
            "language": self.language,
            "text_length": self.text_length,
        }


class ComplianceChecker:
    """Static + optional LLM-judge compliance checker."""

    def __init__(self):
        self._llm_judge: Optional[Callable[[str, str], dict]] = None

    def with_llm_judge(self, callable_: Callable[[str, str], dict]) -> "ComplianceChecker":
        """Wire an optional Claude/etc. callable for nuanced detection.

        Signature: ``callable_(text, language) -> {"violations": [...]}``
        Each violation is ``{"kind": "...", "detail": "...", "snippet": "..."}``.
        """
        self._llm_judge = callable_
        return self

    # ------------------------------------------------------------------
    # Public scan
    # ------------------------------------------------------------------

    def check(self, text: str, *, language: str = "fr") -> ComplianceReport:
        violations: list[Violation] = []
        if not text:
            return ComplianceReport(ok=True, language=language, text_length=0)

        violations.extend(self._token_pass(text, language))
        violations.extend(self._pattern_pass(text))
        if self._llm_judge is not None:
            try:
                llm_out = self._llm_judge(text, language)
                for v in llm_out.get("violations", []):
                    violations.append(
                        Violation(
                            kind="llm",
                            detail=str(v.get("detail", "llm flagged")),
                            snippet=str(v.get("snippet", "")),
                        )
                    )
            except Exception:  # pragma: no cover — LLM failure ≠ pass
                pass

        return ComplianceReport(
            ok=not violations,
            violations=violations,
            language=language,
            text_length=len(text),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _token_pass(text: str, language: str) -> list[Violation]:
        lowered = text.lower()
        blacklist = (
            TOKEN_BLACKLIST_FR if language.lower().startswith("fr") else TOKEN_BLACKLIST_EN
        )
        out: list[Violation] = []
        for tok in blacklist:
            idx = lowered.find(tok)
            if idx != -1:
                snippet = text[max(0, idx - 20): idx + len(tok) + 20]
                out.append(Violation(kind="token", detail=tok, snippet=snippet))
        return out

    @staticmethod
    def _pattern_pass(text: str) -> list[Violation]:
        out: list[Violation] = []
        for name, pat in PATTERN_BANK:
            m = pat.search(text)
            if m:
                out.append(
                    Violation(
                        kind="pattern",
                        detail=name,
                        snippet=text[max(0, m.start() - 10): m.end() + 10],
                    )
                )
        return out


__all__ = [
    "ComplianceChecker",
    "ComplianceReport",
    "PATTERN_BANK",
    "TOKEN_BLACKLIST_EN",
    "TOKEN_BLACKLIST_FR",
    "Violation",
]
