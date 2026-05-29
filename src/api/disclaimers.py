"""Multi-language risk-disclosure disclaimers — P29 compliance.

Smart Sentinel AI publishes algorithmic market analyses, not personalised
investment advice. Every user-facing surface (Telegram, Discord, API
responses) must carry a jurisdiction-aware disclaimer that:

* clarifies the educational / informational nature of the output,
* names the absence of an investment-advice licence,
* warns about leveraged-product losses,
* points to the canonical Terms & Privacy URLs.

The wording mirrors AMF (FR), BaFin (DE), CNMV (ES) and ESMA (EN)
guidance on retail derivatives marketing. It is intentionally short
enough to fit a Telegram footer (≤ 280 chars).
"""

from __future__ import annotations

from typing import Dict

# ─── Translations ─────────────────────────────────────────────────────────

_DISCLAIMERS: Dict[str, str] = {
    "fr": (
        "⚠️ Analyse algorithmique à but informatif uniquement — pas un conseil "
        "en investissement. 74-89 % des comptes particuliers en CFD perdent "
        "de l'argent. Smart Sentinel AI n'est pas un conseiller en "
        "investissement enregistré. CGU : /api/v1/terms"
    ),
    "en": (
        "⚠️ Algorithmic analysis for informational purposes only — not "
        "investment advice. 74-89% of retail CFD accounts lose money. "
        "Smart Sentinel AI is not a registered investment adviser. "
        "Terms: /api/v1/terms"
    ),
    "de": (
        "⚠️ Algorithmische Analyse nur zu Informationszwecken — keine "
        "Anlageberatung. 74-89% der Privatanleger verlieren Geld mit CFDs. "
        "Smart Sentinel AI ist kein registrierter Anlageberater. "
        "AGB: /api/v1/terms"
    ),
    "es": (
        "⚠️ Análisis algorítmico con fines informativos únicamente — no es "
        "asesoramiento de inversión. El 74-89% de las cuentas minoristas "
        "de CFD pierden dinero. Smart Sentinel AI no es un asesor de "
        "inversión registrado. Términos: /api/v1/terms"
    ),
}

#: Short footer used in Telegram / Discord when space is constrained.
_FOOTERS: Dict[str, str] = {
    "fr": "Smart Sentinel AI — Analyse algorithmique, pas un conseil en investissement.",
    "en": "Smart Sentinel AI — Algorithmic analysis, not investment advice.",
    "de": "Smart Sentinel AI — Algorithmische Analyse, keine Anlageberatung.",
    "es": "Smart Sentinel AI — Análisis algorítmico, no es asesoramiento de inversión.",
}

DEFAULT_LANG = "en"
SUPPORTED_LANGS = tuple(_DISCLAIMERS.keys())


def _normalise(lang: str | None) -> str:
    if not lang:
        return DEFAULT_LANG
    code = lang.strip().lower()[:2]
    return code if code in _DISCLAIMERS else DEFAULT_LANG


def get_disclaimer(lang: str | None = None) -> str:
    """Return the long-form disclaimer for ``lang``.

    Falls back to English on unknown / unsupported codes.
    """
    return _DISCLAIMERS[_normalise(lang)]


def get_footer(lang: str | None = None) -> str:
    """Return the short footer string for ``lang`` (Telegram/Discord)."""
    return _FOOTERS[_normalise(lang)]


def detect_language_from_request(headers: Dict[str, str] | None) -> str:
    """Pick a language from an ``Accept-Language`` header.

    Looks at the highest-q value first occurrence; falls back to default
    when no recognised tag is present.
    """
    if not headers:
        return DEFAULT_LANG
    raw = headers.get("accept-language") or headers.get("Accept-Language")
    if not raw:
        return DEFAULT_LANG
    for token in raw.split(","):
        tag = token.split(";")[0].strip().lower()[:2]
        if tag in _DISCLAIMERS:
            return tag
    return DEFAULT_LANG


__all__ = [
    "DEFAULT_LANG",
    "SUPPORTED_LANGS",
    "get_disclaimer",
    "get_footer",
    "detect_language_from_request",
]
