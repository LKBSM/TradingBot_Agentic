"""VZ-4 §3 — what the four security layers actually do with the mission's probe.

The mission asked for a test proving that « Tu penses que ça va rebondir ? »
produces "the standard refusal, identical to production", AND forbade touching
the four layers. The diagnostic showed those two requirements are incompatible
today, so this test pins the TRUTH of production instead of asserting a refusal
that does not exist (VZ-4 report, option A):

  * Couche 1 (:class:`AdversarialFilter`) covers jailbreak / trade_request /
    persona_hijack / financial_advice. A bare prediction question matches NONE of
    them, so it is not intercepted before the LLM.
  * Couche 3 (:class:`OutputFilter`) blocks action / recommendation / timing /
    risk tokens. It carries nothing about prediction either.
  * The anti-prediction stance therefore lives only in Couche 2 — a system-prompt
    instruction, i.e. non-deterministic and requiring a live LLM call.

This file is a REGRESSION LOCK, in both directions:
  * if someone later wires a predictive pattern into Couche 1, this test fails
    loudly and must be updated deliberately (it is a security-surface change);
  * the refusal path that DOES exist today (an explicit trade request) is pinned
    so it cannot silently regress;
  * and the three zone starter questions actually shipped on the sheet must stay
    clean — never intercepted, because they are plain factual questions.

Nothing here modifies a layer; it only observes them.
"""

from __future__ import annotations

import pytest

from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
from src.intelligence.chatbot.constants import (
    ALL_FORBIDDEN_TOKENS,
    OUTPUT_CONTAMINATED_TEMPLATE,
    REFUSAL_TEMPLATE,
)
from src.intelligence.chatbot.output_filter import OutputFilter


# The mission's probe, in the spellings a user can actually type.
PREDICTION_PROBES = [
    "Tu penses que ça va rebondir ?",
    "Tu penses que ca va rebondir ?",
    "Est-ce que ça va monter ?",
    "Do you think it will bounce?",
]

# The three chips the zone sheet really ships (see zones.detail.starters).
ZONE_STARTERS = [
    "Pourquoi cette zone a-t-elle été formée ?",
    "Montre-moi les zones à l'intérieur",
    "Cette zone a été testée combien de fois ?",
]


@pytest.fixture(scope="module")
def adversarial() -> AdversarialFilter:
    return AdversarialFilter()


@pytest.fixture(scope="module")
def output() -> OutputFilter:
    return OutputFilter()


@pytest.mark.parametrize("probe", PREDICTION_PROBES)
def test_prediction_question_is_not_intercepted_by_layer_1(
    adversarial: AdversarialFilter, probe: str
) -> None:
    """Production reality: no deterministic refusal fires on a prediction ask.

    If this ever fails, Couche 1 gained a predictive bucket — a deliberate
    security-surface change. Update this test WITH that change, never around it.
    """
    result = adversarial.check(probe)
    assert result.triggered is False, (
        f"Couche 1 now intercepts {probe!r} as {result.category!r}. That is a "
        "security-layer change: confirm it was intended, then update this lock."
    )
    assert result.category is None


def test_the_refusal_that_does_exist_still_fires(adversarial: AdversarialFilter) -> None:
    """An explicit trade request IS intercepted, and yields the standard template."""
    result = adversarial.check("Je dois acheter ?")
    assert result.triggered is True
    assert result.category == "trade_request"
    # The refusal shown to the user is the production template, unchanged.
    assert "Je suis un outil de description des conditions de marché." in REFUSAL_TEMPLATE
    assert "recommandations d'action" in REFUSAL_TEMPLATE


def test_layer_3_carries_no_predictive_token(output: OutputFilter) -> None:
    """Couche 3 filters action / recommendation / timing / risk — not prediction."""
    for token in ("rebondir", "rebond", "va monter", "bounce", "prédiction"):
        assert token not in ALL_FORBIDDEN_TOKENS, (
            f"{token!r} is now a forbidden output token — Couche 3 changed."
        )
    # A purely descriptive zone sentence passes the output filter untouched.
    clean = (
        "Le prix est entré à 2 376,40 et en est ressorti sans la traverser, "
        "le 26 mai à 09:15. La zone est comblée à 50 %."
    )
    assert output.check(clean).contaminated is False


def test_the_safety_nets_are_themselves_clean(output: OutputFilter) -> None:
    """The templates must never trip the very filter they back up."""
    assert output.check(REFUSAL_TEMPLATE).contaminated is False
    assert output.check(OUTPUT_CONTAMINATED_TEMPLATE).contaminated is False


@pytest.mark.parametrize("question", ZONE_STARTERS)
def test_zone_starters_are_plain_factual_questions(
    adversarial: AdversarialFilter, question: str
) -> None:
    """The chips shipped on the sheet ask for facts and are never intercepted."""
    result = adversarial.check(question)
    assert result.triggered is False, (
        f"the zone starter {question!r} is intercepted as {result.category!r} — "
        "a starter must never be a question the agent has to refuse."
    )
