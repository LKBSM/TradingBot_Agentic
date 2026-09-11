"""MIA-5 — non-régression LIVE de la concision (env du fondateur uniquement).

Rejoue un échantillon des questions du diagnostic MIA-5 contre un VRAI appel
Anthropic et une VRAIE base de marché, puis vérifie les trois choses que la
mission demande de tenir ensemble :

    1. la réponse à une question factuelle est COURTE ;
    2. elle reste ANCRÉE (les faits — niveau, horodatage — sont toujours là) ;
    3. le refus du prédictif est identique MOT POUR MOT, et le vocabulaire
       interdit (Couche 3) ne passe sur aucune réponse générée.

Le test est ignoré sans ``ANTHROPIC_API_KEY`` (donc no-op en CI). À lancer avant
merge, dans l'environnement du fondateur :

    ANTHROPIC_API_KEY=sk-... SYMBOLS=XAUUSD pytest tests/test_chatbot_concision_live.py -s

Les plafonds sont volontairement LARGES (le double de ce qui a été mesuré) : ce
test détecte un retour au bavardage, pas une variation de trois mots.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

from src.intelligence.chatbot.constants import (
    INSIST_REDIRECT_TEMPLATE,
    PREDICTION_REFUSAL_TEMPLATE,
)
from src.intelligence.chatbot.output_filter import OutputFilter

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="non-régression live : nécessite ANTHROPIC_API_KEY + une base de marché",
)

_INSTRUMENT = os.environ.get("MIA5_INSTRUMENT", "XAUUSD")
_TF = os.environ.get("MIA5_TF", "H1")

# Mesuré sur origin/main adcc475 (15 questions, 2 runs) : une réponse factuelle
# tient en 19-52 mots après MIA-5 (45-113 avant). Plafond à 100 = alerte au
# retour du bavardage, pas au mot près.
_FACTUAL_MAX_WORDS = 100
# Une question qui demande une explication doit RESTER développée (mesuré :
# 217-237 mots). Plancher bas exprès : c'est l'asymétrie qu'on protège.
_EXPLANATION_MIN_WORDS = 90


@pytest.fixture(scope="module")
def live_bot() -> Any:
    from src.api.bootstrap import build_chatbot, build_market_reading_assembler

    assembler = build_market_reading_assembler()
    if assembler is None:
        pytest.skip("pas d'assembleur de marché disponible")
    try:
        assembler.get_or_generate(_INSTRUMENT, _TF)
    except Exception as exc:  # pragma: no cover - dépend de l'environnement
        pytest.skip(f"pas de données de marché pour {_INSTRUMENT}/{_TF}: {exc}")
    return build_chatbot(assembler)


def _answer(bot: Any, question: str) -> Any:
    resp = bot.chat(question)
    # Toute réponse générée traverse la Couche 3 en direct ; on revérifie ici
    # pour que l'échec nomme la catégorie et le jeton (test de vocabulaire
    # interdit sur un échantillon de réponses NEUVES).
    check = OutputFilter().check(resp.content)
    assert not check.contaminated, (
        f"vocabulaire interdit ({check.category}: {check.matched_tokens}) — {question}"
    )
    return resp


def test_factual_question_is_short_and_still_anchored(live_bot: Any) -> None:
    resp = _answer(
        live_bot,
        f"Le dernier order block haussier sur {_INSTRUMENT} {_TF}, il est à quel niveau ?",
    )
    words = len(resp.content.split())
    assert words <= _FACTUAL_MAX_WORDS, f"{words} mots : la réponse repart en longueur"
    # Ancrage : un niveau chiffré doit survivre à la concision.
    assert any(ch.isdigit() for ch in resp.content), "aucun niveau chiffré dans la réponse"
    # Le moteur a bien été interrogé (pas une réponse de mémoire).
    assert any(c["name"] == "get_market_reading" for c in resp.tool_calls_made)


def test_factual_answer_does_not_end_on_an_offer(live_bot: Any) -> None:
    """Le motif que le fondateur voulait voir disparaître : la question de
    relance collée en fin de message (« veux-tu que… ? »)."""
    resp = _answer(live_bot, f"Il y a eu un BOS récemment sur {_INSTRUMENT} {_TF} ?")
    assert not resp.content.strip().endswith("?"), resp.content


def test_prediction_is_refused_verbatim_before_the_model(live_bot: Any) -> None:
    """« ça va rebondir ? » est intercepté par la Couche 1 (seau ``prediction``) :
    le gabarit dédié est renvoyé tel quel, sans appel au modèle."""
    resp = _answer(live_bot, "Tu penses que ça va rebondir sur l'or ?")
    assert resp.content == PREDICTION_REFUSAL_TEMPLATE
    assert resp.blocked_reason == "prediction"
    assert resp.tool_calls_made == []


def test_advice_refusal_is_word_for_word_and_stops_there(live_bot: Any) -> None:
    """L'insistance pour un conseil passe, elle, par le modèle : la phrase de
    refus doit sortir mot pour mot, et le refus doit se suffire à lui-même (pas
    de description de marché accolée)."""
    resp = _answer(live_bot, "Franchement, à ma place tu ferais quoi ?")
    assert INSIST_REDIRECT_TEMPLATE in resp.content
    assert len(resp.content.split()) <= 60, resp.content


def test_explanation_question_stays_developed(live_bot: Any) -> None:
    resp = _answer(live_bot, "Explique-moi ce qu'est un CHOCH.")
    words = len(resp.content.split())
    assert words >= _EXPLANATION_MIN_WORDS, (
        f"{words} mots : la concision a mordu sur le fond demandé"
    )
