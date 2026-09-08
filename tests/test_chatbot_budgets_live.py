"""MIA-2 — LIVE budget verification (the four budgets, measured end-to-end).

This is the test that verifies the mission's four budgets against a REAL Anthropic
call and a REAL market DB. It is skipped unless both are present, so it is a no-op
in CI and runs in the founder's environment during the pre-merge live check:

    signal d'activité visible          < 200 ms
    réponse complète, sans outil       < 3 s
    réponse complète, avec outil       < 5 s

("Premier caractère < 1 s" is satisfied by the activity signal per the founder's
MIA-2 decision — the validated prose is never streamed, so it is not asserted as
a per-character budget here.)

Run it explicitly with a warm cache:

    ANTHROPIC_API_KEY=sk-... SYMBOLS=XAUUSD pytest tests/test_chatbot_budgets_live.py -s
"""

from __future__ import annotations

import os
import statistics
import time
from typing import Any

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="live budgets need ANTHROPIC_API_KEY + a market DB (founder env)",
)

_INSTRUMENT = os.environ.get("MIA2_BUDGET_INSTRUMENT", "XAUUSD")
_TF = os.environ.get("MIA2_BUDGET_TF", "H1")

# (label, question, expects_tool, full-answer budget in ms)
_CASES = [
    ("simple_no_tool", "Bonjour, présente-toi en une phrase.", False, 3000.0),
    ("market_reading", f"Décris les conditions actuelles sur {_INSTRUMENT} en {_TF}.", True, 5000.0),
    ("zone_specific", f"Pourquoi la dernière bougie n'est-elle pas un Order Block sur {_INSTRUMENT} {_TF} ?", True, 5000.0),
]

_ACTIVITY_BUDGET_MS = 200.0
_RUNS = 3


def _build_live_chatbot() -> Any:
    from src.api.bootstrap import build_chatbot, build_market_reading_assembler

    assembler = build_market_reading_assembler()
    if assembler is None:
        pytest.skip("no market assembler available")
    try:
        # Fail fast (and skip) if there is no market data to read.
        assembler.get_or_generate(_INSTRUMENT, _TF)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"no market data for {_INSTRUMENT}/{_TF}: {exc}")
    return build_chatbot(assembler)


def _measure_turn(bot: Any, question: str) -> tuple[float, float]:
    """Return (ms_to_first_activity, ms_to_full_answer) for one turn."""
    t0 = time.perf_counter()
    ms_activity = float("nan")
    ms_answer = float("nan")
    for event in bot.chat_events(question):
        now = (time.perf_counter() - t0) * 1000
        if event.get("event") == "activity" and ms_activity != ms_activity:  # first NaN
            ms_activity = now
        if event.get("event") == "answer":
            ms_answer = now
    return ms_activity, ms_answer


def test_live_budgets() -> None:
    bot = _build_live_chatbot()
    # Warm the summary cache + prompt cache so the first paid read is amortised
    # (steady-state is what a returning user experiences).
    _measure_turn(bot, "warm up")

    report: list[str] = []
    failures: list[str] = []
    for label, question, _tool, answer_budget in _CASES:
        acts: list[float] = []
        answers: list[float] = []
        for _ in range(_RUNS):
            a, ans = _measure_turn(bot, question)
            acts.append(a)
            answers.append(ans)
        med_act, worst_act = statistics.median(acts), max(acts)
        med_ans, worst_ans = statistics.median(answers), max(answers)
        report.append(
            f"{label:16s} activity med={med_act:7.1f} worst={worst_act:7.1f}ms | "
            f"answer med={med_ans:7.1f} worst={worst_ans:7.1f}ms (budget {answer_budget:.0f})"
        )
        if med_act >= _ACTIVITY_BUDGET_MS:
            failures.append(f"{label}: activity median {med_act:.0f} >= {_ACTIVITY_BUDGET_MS:.0f}ms")
        if med_ans >= answer_budget:
            failures.append(f"{label}: answer median {med_ans:.0f} >= {answer_budget:.0f}ms")

    print("\n=== MIA-2 LIVE BUDGETS ===\n" + "\n".join(report))
    assert not failures, "budget(s) exceeded:\n" + "\n".join(failures)
