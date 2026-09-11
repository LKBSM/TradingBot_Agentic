"""VZ-4 §3 — la question de test du refus, sur la fiche de zone.

Historique de cette mission, parce qu'il explique la forme du fichier : au moment
du diagnostic VZ-4, « Tu penses que ça va rebondir ? » n'était interceptée par
AUCUNE couche déterministe (Couche 1 ne couvrait que jailbreak / trade_request /
persona_hijack / financial_advice, Couche 3 ne filtre que les jetons d'action /
recommandation / moment / risque). L'anti-prédiction ne vivait qu'en Couche 2,
c'est-à-dire une consigne de prompt — non déterministe.

Entre-temps, `main` a livré le seau ``prediction`` et son refus dédié
(9eb35c6). La question EST donc désormais interceptée, avec un refus propre à la
demande de pronostic. Ce fichier teste ce que la mission demandait à l'origine :

  * la sonde de la maquette produit le refus standard de production ;
  * ce refus est celui du seau ``prediction`` (pas le générique) — il parle bien
    de l'avenir du prix, pas seulement de recommandations ;
  * le refus qui existait déjà (demande de trade explicite) n'a pas régressé ;
  * les 4 puces expédiées sur la fiche se comportent comme prévu : les 3 puces
    factuelles atteignent l'agent, la 4ᵉ (la sonde) est refusée ;
  * une question FACTUELLE qui contient un futur grammatical (« quand le marché
    va-t-il rouvrir ? ») n'est PAS avalée par le seau prédiction — sinon la fiche
    de zone perdrait des réponses légitimes.

Rien ici ne modifie une couche ; le fichier ne fait que les observer.
"""

from __future__ import annotations

import pytest

from src.intelligence.chatbot.adversarial_filter import AdversarialFilter
from src.intelligence.chatbot.constants import (
    ALL_FORBIDDEN_TOKENS,
    OUTPUT_CONTAMINATED_TEMPLATE,
    PREDICTION_REFUSAL_TEMPLATE,
    REFUSAL_TEMPLATE,
    REFUSAL_TEMPLATE_BY_CATEGORY,
)
from src.intelligence.chatbot.output_filter import OutputFilter


# La sonde de la maquette, dans les orthographes qu'un utilisateur tape vraiment.
PREDICTION_PROBES = [
    "Tu penses que ça va rebondir ?",
    "Tu penses que ca va rebondir ?",
    "Est-ce que ça va monter ?",
    "Do you think it will bounce?",
]

# Les 3 puces factuelles de la fiche (zones.detail.starters).
ZONE_STARTERS = [
    "Pourquoi cette zone a-t-elle été formée ?",
    "Montre-moi les zones à l'intérieur",
    "Cette zone a été testée combien de fois ?",
]

# Questions FACTUELLES portant un futur grammatical — elles doivent passer.
FACTUAL_WITH_FUTURE = [
    "Quand le marché va-t-il rouvrir ?",
    "Y a-t-il une news bientôt ?",
]


@pytest.fixture(scope="module")
def adversarial() -> AdversarialFilter:
    return AdversarialFilter()


@pytest.fixture(scope="module")
def output() -> OutputFilter:
    return OutputFilter()


@pytest.mark.parametrize("probe", PREDICTION_PROBES)
def test_the_mockup_probe_produces_the_standard_refusal(
    adversarial: AdversarialFilter, probe: str
) -> None:
    """« Tu penses que ça va rebondir ? » → refus déterministe, avant tout LLM."""
    result = adversarial.check(probe)
    assert result.triggered is True, f"{probe!r} n'est plus intercepté par la Couche 1"
    assert result.category == "prediction"


def test_the_prediction_refusal_is_the_dedicated_one(
    output: OutputFilter,
) -> None:
    """Le refus servi est celui du seau ``prediction``, pas le générique.

    La distinction compte sur la fiche de zone : la question porte sur l'AVENIR du
    prix, et la réponse honnête doit le dire — pas seulement « je ne donne pas de
    recommandations ».
    """
    assert REFUSAL_TEMPLATE_BY_CATEGORY["prediction"] == PREDICTION_REFUSAL_TEMPLATE
    assert PREDICTION_REFUSAL_TEMPLATE != REFUSAL_TEMPLATE
    assert PREDICTION_REFUSAL_TEMPLATE.strip() != ""
    # Un filet de sécurité ne doit jamais déclencher le filtre qu'il épaule.
    assert output.check(PREDICTION_REFUSAL_TEMPLATE).contaminated is False


def test_the_pre_existing_trade_refusal_has_not_regressed(
    adversarial: AdversarialFilter,
) -> None:
    """La demande de trade explicite reste interceptée, avec le refus générique."""
    result = adversarial.check("Je dois acheter ?")
    assert result.triggered is True
    assert result.category == "trade_request"
    assert "recommandations d'action" in REFUSAL_TEMPLATE


@pytest.mark.parametrize("question", ZONE_STARTERS)
def test_the_three_factual_chips_reach_the_agent(
    adversarial: AdversarialFilter, question: str
) -> None:
    """Les puces factuelles de la fiche ne sont jamais refusées."""
    result = adversarial.check(question)
    assert result.triggered is False, (
        f"la puce {question!r} est interceptée comme {result.category!r} — "
        "une puce factuelle ne doit jamais être une question à refuser."
    )


@pytest.mark.parametrize("question", FACTUAL_WITH_FUTURE)
def test_a_grammatical_future_is_not_mistaken_for_a_forecast(
    adversarial: AdversarialFilter, question: str
) -> None:
    """Le seau prédiction ne doit pas avaler une question factuelle au futur.

    Sinon la fiche de zone perdrait des réponses légitimes (« quand rouvre le
    marché ? ») au profit d'un refus hors sujet.
    """
    result = adversarial.check(question)
    assert result.triggered is False, (
        f"{question!r} est refusée comme {result.category!r} alors qu'elle est "
        "factuelle — le seau prédiction sur-bloque."
    )


def test_layer_3_still_carries_no_predictive_token(output: OutputFilter) -> None:
    """Couche 3 filtre action / recommandation / moment / risque — pas le pronostic.

    Le blocage du pronostic est une affaire d'ENTRÉE (Couche 1), pas de sortie :
    une phrase descriptive contenant « rebondi » au passé doit passer.
    """
    for token in ("rebondir", "rebond", "bounce"):
        assert token not in ALL_FORBIDDEN_TOKENS
    clean = (
        "Le prix est entré à 2 376,40 et en est ressorti sans la traverser, "
        "le 26 mai à 09:15. La zone est comblée à 50 %."
    )
    assert output.check(clean).contaminated is False


def test_the_safety_nets_are_themselves_clean(output: OutputFilter) -> None:
    """Aucun gabarit de refus ne doit trébucher sur le filtre de sortie."""
    assert output.check(REFUSAL_TEMPLATE).contaminated is False
    assert output.check(OUTPUT_CONTAMINATED_TEMPLATE).contaminated is False
