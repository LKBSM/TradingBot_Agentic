"""MIA-5 — précision de la Couche 3 : les deux faux positifs prouvés, et la
garantie que rien n'a été desserré au passage.

Contexte : en mesurant la concision, DEUX réponses factuelles et conformes ont
été remplacées par le gabarit de contamination (« Je ne peux pas formuler cette
réponse de cette manière ») — le client voyait un refus incompréhensible :

  1. « La gestion **du risque** est un sujet général… » → le jeton nu ``risqué``
     collapse sur le nom commun ``risque`` par la normalisation des accents.
  2. « le **coût d'opportunité** de détenir de l'or » → le terme macro standard
     pour un actif sans rendement déclenchait ``opportunité``.

Les textes ci-dessous sont les SORTIES RÉELLES capturées à ce moment-là, pas des
exemples reconstruits. Le fichier verrouille les deux carve-outs **et** la
non-régression : tous les jugements de valeur restent bloqués, dans toutes leurs
formes accentuées ou non.
"""

from __future__ import annotations

import pytest

from src.intelligence.chatbot.constants import (
    ACCENT_SENSITIVE_TOKENS,
    ALL_FORBIDDEN_TOKENS,
    FORBIDDEN_TOKENS_BY_CATEGORY,
    HOMONYM_SAFE_EXPRESSIONS,
)
from src.intelligence.chatbot.output_filter import OutputFilter


@pytest.fixture(scope="module")
def filt() -> OutputFilter:
    return OutputFilter()


# --------------------------------------------------------------------------- #
# Les deux réponses réelles qui tombaient à tort
# --------------------------------------------------------------------------- #
_REAL_ANSWER_RISK_NOUN = (
    "Je décris les conditions du marché. La gestion du risque en trading est un "
    "sujet général qui sort de mon périmètre — je ne dois pas produire un cours.\n\n"
    "Si tu as une question précise sur une condition du marché (un niveau, une "
    "structure, un état), je peux t'aider."
)

_REAL_ANSWER_MOMENT_RELATIVE = (
    "Un Order Block se crée quand le prix casse un niveau via une forte bougie "
    "directionnelle. Ensuite, le prix s'éloigne et reteste ce niveau : c'est le "
    "moment où les ordres accumulés se manifestent."
)

_REAL_ANSWER_OPPORTUNITY_COST = (
    "La décision de taux du FOMC du 16 septembre n'a pas encore été publiée. Le "
    "calendrier l'annonce pour ce jour à 18h00 UTC, mais la valeur est en "
    "attente.\n\nUne décision de taux affecte le coût d'opportunité de détenir "
    "de l'or (métal sans rendement), et donc la demande d'or."
)


def test_real_answer_about_risk_management_is_not_blocked(filt: OutputFilter) -> None:
    assert filt.check(_REAL_ANSWER_RISK_NOUN).contaminated is False


def test_real_answer_about_opportunity_cost_is_not_blocked(filt: OutputFilter) -> None:
    assert filt.check(_REAL_ANSWER_OPPORTUNITY_COST).contaminated is False


def test_real_explanation_with_a_temporal_clause_is_not_blocked(
    filt: OutputFilter,
) -> None:
    """« c'est le moment OÙ … » décrit un instant du mécanisme ; le jugement
    interdit est « c'est le moment DE / POUR agir »."""
    assert filt.check(_REAL_ANSWER_MOMENT_RELATIVE).contaminated is False


@pytest.mark.parametrize(
    "text",
    [
        "La gestion du risque est un sujet d'éducation générale.",
        "le risque de volatilité autour de la publication",
        "les risques macro de la semaine",
        "le coût d'opportunité de détenir de l'or",
        "le cout d'opportunite de detenir de l'or",  # sans accents
        "bien sûr, voici les niveaux détectés",
        "La tendance est haussière sur XAUUSD H1.",
    ],
)
def test_legitimate_descriptive_text_passes(filt: OutputFilter, text: str) -> None:
    result = filt.check(text)
    assert not result.contaminated, (result.category, result.matched_tokens)


# --------------------------------------------------------------------------- #
# Non-régression — le jugement de valeur reste bloqué sous TOUTES ses formes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        # adjectif accentué, toutes flexions (le pluriel/féminin n'était couvert
        # par AUCUN jeton avant MIA-5 — c'est un durcissement, pas une tolérance)
        "c'est risqué",
        "une position risquée ici",
        "des niveaux risqués",
        "des zones risquées",
        # adjectif NON accentué : couvert par les tournures
        "c'est risque",
        "trop risque",
        "très risque",
        "assez risque",
        "plutot risque",
        "moins risque",
        "plus risque",
        "peu risqué",
        # le reste de la catégorie D
        "sans risque",
        "gain garanti",
        "c'est dangereux",
        "c'est sûr",
        # opportunité hors expression figée
        "c'est une opportunité",
        "opportunite a saisir",
        # le jugement de moment reste bloqué dans sa forme prescriptive
        "c'est le moment d'agir",
        "c'est le moment de regarder ça",
        "c'est le bon moment",
        # les autres catégories ne bougent pas
        "je te recommande d'attendre",
        "achète maintenant",
        "c'est le bon moment",
    ],
)
def test_value_judgements_are_still_blocked(filt: OutputFilter, text: str) -> None:
    assert filt.check(text).contaminated is True, text


def test_no_token_was_removed_from_any_category() -> None:
    """Les carve-outs changent la MANIÈRE de chercher, jamais la liste cherchée.
    Les planchers de sécurité (MIA-2) restent tenus, avec de la marge."""
    floors = {
        "action_trading": 26,
        "recommandation": 41,
        "jugement_moment": 21,
        "jugement_risque": 27,
    }
    for category, floor in floors.items():
        assert len(FORBIDDEN_TOKENS_BY_CATEGORY[category]) >= floor, category
    assert len(ALL_FORBIDDEN_TOKENS) >= 115


def test_carve_outs_stay_enumerable_and_narrow() -> None:
    """Un carve-out qui grossit sans preuve est un trou de conformité : les deux
    listes restent minuscules, explicites, et chacune adossée à une sortie réelle
    capturée."""
    assert ACCENT_SENSITIVE_TOKENS <= {"risqué", "risquée", "risqués", "risquées"}
    assert HOMONYM_SAFE_EXPRESSIONS == {
        "coût d'opportunité", "coûts d'opportunité", "c'est le moment où",
    }
