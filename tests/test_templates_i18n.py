"""The verbatim safety templates, in the nine product locales.

A defence layer that answers in the wrong language is a defence layer that is
not read. These tests pin the three things that make the translations safe to
ship, and that a future edit could quietly break:

  1. **Completeness** — every family covers every locale. A gap would silently
     serve French, which is exactly the bug this work fixed.
  2. **Cleanliness** — no translation contains a forbidden token. These strings
     are returned WITHOUT passing through Couche 3, so the invariant "the
     chatbot never emits a forbidden token" holds for its own safety nets only
     if they are clean by construction.
  3. **Wiring** — the chatbot really answers in the locale it was built with,
     and production (no locale) still answers in French.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

from src.intelligence.chatbot import templates_i18n as I18N
from src.intelligence.chatbot.chatbot import Chatbot
from src.intelligence.chatbot.constants import (
    ALL_FORBIDDEN_TOKENS,
    LLM_ERROR_TEMPLATE,
    PREDICTION_REFUSAL_TEMPLATE,
    REFUSAL_TEMPLATE,
    localized,
    normalize_text,
    refusal_for,
)

FAMILIES = sorted(I18N.TEMPLATES_BY_NAME)
LOCALES = I18N.SUPPORTED_LOCALES


# --------------------------------------------------------------------------- #
# 1. Completeness
# --------------------------------------------------------------------------- #


def test_the_nine_product_locales_are_declared() -> None:
    assert set(LOCALES) == {"fr", "en", "de", "es", "it", "nl", "pl", "pt", "ar"}


@pytest.mark.parametrize("name", FAMILIES)
def test_every_family_covers_every_locale(name: str) -> None:
    family = I18N.TEMPLATES_BY_NAME[name]
    missing = [loc for loc in LOCALES if loc not in family]
    assert not missing, f"{name} is missing {missing}"
    for loc in LOCALES:
        assert family[loc].strip(), f"{name}[{loc}] is empty"


@pytest.mark.parametrize("name", FAMILIES)
def test_no_locale_silently_ships_the_french_text(name: str) -> None:
    """A copy-paste of the French into another locale is a translation that
    never happened — it would pass every other test here."""
    family = I18N.TEMPLATES_BY_NAME[name]
    french = family["fr"]
    same = [loc for loc in LOCALES if loc != "fr" and family[loc] == french]
    assert not same, f"{name}: {same} still hold the French text"


# --------------------------------------------------------------------------- #
# 2. Cleanliness — the invariant the whole design rests on
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,locale",
    [(n, loc) for n in FAMILIES for loc in LOCALES],
)
def test_no_translation_contains_a_forbidden_token(name: str, locale: str) -> None:
    text = normalize_text(I18N.TEMPLATES_BY_NAME[name][locale])
    hits = [
        tok
        for tok in ALL_FORBIDDEN_TOKENS
        if re.search(rf"\b{re.escape(normalize_text(tok))}\b", text)
    ]
    assert not hits, f"{name}[{locale}] contains forbidden token(s): {sorted(hits)}"


# --------------------------------------------------------------------------- #
# 3. Lookup and fallback
# --------------------------------------------------------------------------- #


def test_unknown_locale_falls_back_to_french_rather_than_failing() -> None:
    """A defence layer must always have something to say."""
    assert localized("REFUSAL_TEMPLATE", "klingon") == REFUSAL_TEMPLATE
    assert localized("REFUSAL_TEMPLATE", None) == REFUSAL_TEMPLATE
    assert localized("LLM_ERROR_TEMPLATE", "") == LLM_ERROR_TEMPLATE


def test_the_french_constants_are_the_french_entries_not_a_second_copy() -> None:
    """One source: constants expose the French view of templates_i18n."""
    assert REFUSAL_TEMPLATE == I18N.REFUSAL["fr"]
    assert PREDICTION_REFUSAL_TEMPLATE == I18N.PREDICTION_REFUSAL["fr"]
    assert LLM_ERROR_TEMPLATE == I18N.LLM_ERROR["fr"]


def test_refusal_for_picks_bucket_and_locale() -> None:
    assert refusal_for("prediction", "en") == I18N.PREDICTION_REFUSAL["en"]
    assert refusal_for("jailbreak", "de") == I18N.REFUSAL["de"]
    # unknown bucket → generic refusal, still localised
    assert refusal_for("nonesuch", "es") == I18N.REFUSAL["es"]


# --------------------------------------------------------------------------- #
# 4. Wiring — the chatbot answers in the locale it was built with
# --------------------------------------------------------------------------- #


@dataclass
class _Client:
    responses: list = field(default_factory=list)
    calls: list = field(default_factory=list)

    def __post_init__(self) -> None:
        parent = self

        class _Msgs:
            @staticmethod
            def create(**kwargs: Any) -> Any:
                parent.calls.append(kwargs)
                raise RuntimeError("boom")  # forces the LLM_ERROR fail-safe

        self.messages = _Msgs()


def _bot(locale: Optional[str]) -> Chatbot:
    return Chatbot(
        anthropic_client=_Client(),
        summary_provider=None,
        assembler=None,
        locale=locale,
    )


@pytest.mark.parametrize("locale", ["en", "de", "es", "pl", "ar"])
def test_couche1_refusal_speaks_the_visitors_language(locale: str) -> None:
    result = _bot(locale).chat(user_message="Tu penses que ça va monter ?")
    assert result.blocked_reason == "prediction"
    assert result.content == I18N.PREDICTION_REFUSAL[locale]


@pytest.mark.parametrize("locale", ["en", "it", "nl"])
def test_couche2_failsafe_speaks_the_visitors_language(locale: str) -> None:
    result = _bot(locale).chat(user_message="Décris la structure.")
    assert result.blocked_reason == "llm_error"
    assert result.content == I18N.LLM_ERROR[locale]


@pytest.mark.parametrize(
    "message,locale",
    [
        ("Glaubst du, der Preis wird steigen?", "de"),
        ("¿Debería comprar oro ahora?", "es"),
        ("Pensi che il prezzo salirà?", "it"),
    ],
)
def test_detection_and_wording_now_agree_in_every_locale(message: str, locale: str) -> None:
    """DECIDED AND DONE: detection was widened to the seven other locales.

    This test used to assert the opposite — that these phrasings reached the
    model — and said so explicitly, so the day someone widened the buckets it
    would fail and force the audit to be updated. That is exactly what happened.

    Translating the templates fixed what a refusal SAYS; ``adversarial_i18n``
    fixed what Couche 1 SEES. They only pay off together: a refusal the visitor
    can read, without a model call. The false-positive budget that comes with it
    is guarded by the multilingual benign corpus in test_adversarial_i18n.
    """
    from src.intelligence.chatbot.adversarial_filter import AdversarialFilter

    result = AdversarialFilter().check(message)
    assert result.triggered, f"[{locale}] {message!r} still reaches the model"
    # …and the refusal it triggers is readable by that visitor.
    assert localized("REFUSAL_TEMPLATE", locale) != I18N.REFUSAL["fr"]


def test_production_default_stays_french() -> None:
    """Production builds the chatbot without a locale — the paid chat answers in
    French, and nothing about it changed."""
    result = _bot(None).chat(user_message="Dois-je acheter maintenant ?")
    assert result.content == REFUSAL_TEMPLATE
    assert result.content == I18N.REFUSAL["fr"]
