"""Chantier 4 — bonus stress: 100 contaminated LLM outputs → all scrubbed by Couche 3.

Every output the (stubbed) LLM produces here contains at least one forbidden
token. The chatbot must replace 100% of them with OUTPUT_CONTAMINATED_TEMPLATE,
and the final content returned must itself carry no forbidden token.
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from typing import Any

from src.intelligence.chatbot.chatbot import Chatbot
from src.intelligence.chatbot.constants import OUTPUT_CONTAMINATED_TEMPLATE
from src.intelligence.chatbot.output_filter import OutputFilter


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Resp:
    content: list
    stop_reason: str = "end_turn"


class _Msgs:
    def __init__(self, parent: "_Client") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        return self._p.response


class _Client:
    def __init__(self, text: str) -> None:
        self.response = _Resp([_TextBlock(text)])
        self.messages = _Msgs(self)


class _StubProvider:
    def get(self) -> dict:
        return {"instruments_tracked": []}


# 20 base contaminated outputs (>=1 forbidden token each, across the 4 categories).
_CONTAMINATED_BASES: list[str] = [
    "Tu devrais acheter à 2378",
    "Il faut vendre maintenant",
    "Entrez maintenant sur le marché",
    "Sortez de votre position tout de suite",
    "Achète le gold sans attendre",
    "Je te recommande de patienter",
    "Je conseille d'attendre la cassure",
    "Tu devrais suivre la tendance",
    "Il faudrait observer ce niveau",
    "Je suggère d'entrer ici",
    "C'est le bon moment pour rentrer",
    "Mauvais moment pour agir là",
    "Setup parfait sur l'or",
    "Belle opportunité sur l'EURUSD",
    "Entrée idéale à ce niveau",
    "C'est risqué en ce moment",
    "C'est dangereux ici",
    "C'est sûr d'attendre",
    "Gain garanti sur ce mouvement",
    "C'est sécurisé à ce prix",
]


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _mutate(base: str, rng: random.Random) -> str:
    s = base
    c = rng.randint(0, 3)
    if c == 1:
        s = s.upper()
    elif c == 2:
        s = s.lower()
    elif c == 3:
        s = _strip_accents(s)
    if rng.random() < 0.5:
        s = s.replace("'", "’")
    if rng.random() < 0.4:
        s = s + rng.choice([".", " !", " ...", ""])
    return s


def test_hundred_contaminated_outputs_all_scrubbed() -> None:
    rng = random.Random(424242)
    corpus = [_mutate(_CONTAMINATED_BASES[i % len(_CONTAMINATED_BASES)], rng) for i in range(100)]
    out_filter = OutputFilter()

    not_caught: list[str] = []
    leaked: list[str] = []

    for text in corpus:
        client = _Client(text)
        bot = Chatbot(anthropic_client=client, summary_provider=_StubProvider(), assembler=None)
        resp = bot.chat("Décris-moi les conditions du marché")
        if not (resp.blocked_reason or "").startswith("output_contaminated_"):
            not_caught.append(text)
        if resp.content != OUTPUT_CONTAMINATED_TEMPLATE:
            not_caught.append(text)
        if out_filter.check(resp.content).contaminated:
            leaked.append(text)

    assert not not_caught, f"{len(not_caught)} contaminated outputs not scrubbed: {not_caught[:5]}"
    assert not leaked, f"forbidden token leaked in final content: {leaked[:5]}"
