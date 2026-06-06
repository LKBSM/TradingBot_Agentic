"""Chantier 4 — Couche 3 tests: OutputFilter + Chatbot integration."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.intelligence.chatbot.chatbot import Chatbot
from src.intelligence.chatbot.constants import OUTPUT_CONTAMINATED_TEMPLATE
from src.intelligence.chatbot.output_filter import OutputCheckResult, OutputFilter


@pytest.fixture()
def flt() -> OutputFilter:
    return OutputFilter()


# --------------------------------------------------------------------------- #
# Detection per category
# --------------------------------------------------------------------------- #


def test_detect_action_trading(flt: OutputFilter) -> None:
    res = flt.check("Tu devrais acheter à 2378")
    assert res.contaminated is True
    assert res.category == "action_trading"
    assert "acheter" in res.matched_tokens


def test_detect_recommandation(flt: OutputFilter) -> None:
    res = flt.check("Je te recommande de patienter")
    assert res.contaminated is True
    assert res.category == "recommandation"


def test_detect_jugement_moment(flt: OutputFilter) -> None:
    res = flt.check("C'est le bon moment pour rentrer")
    assert res.contaminated is True
    assert res.category == "jugement_moment"


def test_detect_jugement_risque(flt: OutputFilter) -> None:
    # NOTE: the brief example "C'est risqué d'entrer maintenant" also contains
    # the action verb "entrer" → by the priority rule it maps to action_trading.
    # We test the jugement_risque category with a PURE example (no action verb).
    res = flt.check("C'est risqué actuellement")
    assert res.contaminated is True
    assert res.category == "jugement_risque"


# --------------------------------------------------------------------------- #
# Clean outputs
# --------------------------------------------------------------------------- #


def test_clean_descriptive_output(flt: OutputFilter) -> None:
    res = flt.check("L'Order Block H1 est actif à 2378.50")
    assert res.contaminated is False
    assert res.category is None
    assert res.matched_tokens == ()


def test_clean_uses_homonym_entre(flt: OutputFilter) -> None:
    # "entre" (= between) was excluded from the token set → descriptive, clean.
    res = flt.check("Le FVG entre 2376 et 2378 n'est pas comblé")
    assert res.contaminated is False


def test_empty_output_is_clean(flt: OutputFilter) -> None:
    assert flt.check("").contaminated is False
    assert flt.check("   ").contaminated is False


def test_long_descriptive_output_is_clean(flt: OutputFilter) -> None:
    text = (
        "La structure actuelle montre un BOS haussier confirmé il y a trente "
        "minutes, suivi d'un retest en cours sur le niveau cassé. La volatilité "
        "observée reste élevée sur la fenêtre des vingt dernières bougies. "
        "Plusieurs Order Blocks demeurent actifs sous le prix, et un Fair Value "
        "Gap non comblé subsiste au-dessus. La confluence multi-timeframe aligne "
        "H1 et H4 dans la même direction. " * 3
    )
    assert len(text) > 500
    assert flt.check(text).contaminated is False


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #


def test_case_insensitive(flt: OutputFilter) -> None:
    assert flt.check("TU DEVRAIS ACHETER").contaminated is True


def test_accent_insensitive(flt: OutputFilter) -> None:
    # "C'est risque" (no accent) still caught after normalisation.
    assert flt.check("C'est risque").contaminated is True


def test_curly_apostrophe_detected(flt: OutputFilter) -> None:
    assert flt.check("C’est le bon moment").contaminated is True


# --------------------------------------------------------------------------- #
# Multiple tokens & priority
# --------------------------------------------------------------------------- #


def test_multiple_tokens_same_category(flt: OutputFilter) -> None:
    res = flt.check("Achète et vends rapidement")
    assert res.contaminated is True
    assert res.category == "action_trading"
    assert "achete" in res.matched_tokens
    assert "vends" in res.matched_tokens


def test_priority_action_trading_beats_jugement_risque(flt: OutputFilter) -> None:
    # Contains action_trading ("achète") AND jugement_risque ("c'est risqué").
    res = flt.check("Achète maintenant, c'est risqué mais ça vaut le coup")
    assert res.category == "action_trading"
    # only the winning category's tokens are reported
    assert all("risqu" not in t for t in res.matched_tokens)


def test_result_is_frozen() -> None:
    r = OutputCheckResult(contaminated=False)
    with pytest.raises(Exception):
        r.contaminated = True  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Chatbot integration (Couche 3 wired)
# --------------------------------------------------------------------------- #


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Resp:
    content: list
    stop_reason: str


class _Msgs:
    def __init__(self, parent: "_Client") -> None:
        self._p = parent

    def create(self, **kwargs: object) -> object:
        self._p.calls += 1
        return self._p.response


class _Client:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls = 0
        self.messages = _Msgs(self)


class _DummyProvider:
    def get(self) -> dict:
        return {"instruments_tracked": []}


def _make_bot(answer: str) -> Chatbot:
    client = _Client(_Resp([_TextBlock(answer)], "end_turn"))
    return Chatbot(
        anthropic_client=client,
        summary_provider=_DummyProvider(),
        assembler=None,
    )


def test_chatbot_replaces_contaminated_output() -> None:
    bot = _make_bot("Tu devrais acheter à 2378")
    out = bot.chat("Décris XAUUSD H1")
    assert out.content == OUTPUT_CONTAMINATED_TEMPLATE
    assert out.blocked_reason == "output_contaminated_action_trading"


def test_chatbot_passes_clean_output_through() -> None:
    bot = _make_bot("Le marché XAU H1 est en tendance haussière, volatilité élevée.")
    out = bot.chat("Décris XAUUSD H1")
    assert out.content == "Le marché XAU H1 est en tendance haussière, volatilité élevée."
    assert out.blocked_reason is None
