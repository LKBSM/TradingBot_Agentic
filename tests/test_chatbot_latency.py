"""MIA-2 — latency levers + security non-regression.

Covers the four levers wired for MIA-2 (prompt caching, server-side history
truncation, response-length cap, per-call timeout) and locks the invariants the
mission requires:

  * the four budgets are exercised by a test (deterministic overhead here; the
    live end-to-end budgets run under ``test_budgets_live`` when an API key + a
    market DB are present — see ``tests/test_chatbot_budgets_live.py``);
  * a test FAILS if the number of security checks decreases;
  * an invented identifier is still rejected;
  * the same tool is never executed twice for the same context within a turn;
  * validation is untouched — the static prefix is cached, the summary moved,
    nothing was removed from any defence layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

from src.intelligence.chatbot.chatbot import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TIMEOUT_S,
    MAX_MODEL_HISTORY,
    SIGNAL_CONTEXT_TEMPLATE,
    SYSTEM_PROMPT_STATIC,
    Chatbot,
)
from src.intelligence.chatbot.constants import (
    ADVERSARIAL_PATTERNS_BY_CATEGORY,
    FORBIDDEN_TOKENS_BY_CATEGORY,
)
from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider
from src.intelligence.chatbot.view_action_filter import ALLOWED_ACTIONS
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
)


# --------------------------------------------------------------------------- #
# Minimal stubs (self-contained — mirror tests/test_chatbot.py)
# --------------------------------------------------------------------------- #


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    name: str
    input: dict
    id: str = "tu_1"
    type: str = "tool_use"


@dataclass
class StubResponse:
    content: list
    stop_reason: str


class StubMessages:
    def __init__(self, parent: "StubClient") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        if not self._p.responses:
            raise AssertionError("StubClient: no more scripted responses")
        nxt = self._p.responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


class StubClient:
    def __init__(self, responses: list) -> None:
        self.responses = list(responses)
        self.calls: list[dict] = []
        self.messages = StubMessages(self)


def make_reading(instrument: str = "XAUUSD", timeframe: str = "H1") -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument=instrument,
            timeframe=timeframe,
            candle_close_ts=__import__("datetime").datetime(
                2026, 6, 5, 14, 0, tzinfo=__import__("datetime").timezone.utc
            ),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(),
        regime=MarketReadingRegime(
            trend="bullish",
            volatility_observed="elevated",
            market_phase="expansion",
            mtf_confluence={"h1": "bullish", "h4": "bullish"},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=["trend_bullish"],
            description="Tendance haussière.",
            description_source="engine_template",
        ),
    )


class StubAssembler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        self.calls.append((instrument, timeframe))
        return make_reading(instrument, timeframe)


def make_chatbot(
    responses: list, assembler: Optional[StubAssembler] = None
) -> tuple[Chatbot, StubClient, StubAssembler]:
    assembler = assembler or StubAssembler()
    provider = SignalSummaryProvider(assembler)
    client = StubClient(responses)
    bot = Chatbot(anthropic_client=client, summary_provider=provider, assembler=assembler)
    return bot, client, assembler


def _system_blocks(system: Any) -> list[dict]:
    assert isinstance(system, list), "MIA-2: system must be a list of blocks"
    return system


# --------------------------------------------------------------------------- #
# Lever 1 — prompt caching
# --------------------------------------------------------------------------- #


def test_static_prefix_is_marked_for_cache() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("Bonjour")
    blocks = _system_blocks(client.calls[0]["system"])
    static = blocks[0]
    assert static.get("cache_control") == {"type": "ephemeral"}
    # The stable identity/rules/tool-access text lives in the cached prefix…
    assert "MIA Markets" in static["text"]
    assert "get_ob_diagnostic" in static["text"]
    # …and the VARIABLE summary is NOT in the cached prefix (else a 60s refresh
    # would bust the cache on every turn).
    assert "instruments_tracked" not in static["text"]


def test_signal_summary_sits_after_the_cache_breakpoint() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("Bonjour")
    blocks = _system_blocks(client.calls[0]["system"])
    tail = blocks[-1]
    assert tail is not blocks[0]
    assert "instruments_tracked" in tail["text"]
    assert "XAUUSD" in tail["text"]
    # The variable block must NOT carry cache_control.
    assert "cache_control" not in tail


def test_cached_prefix_is_byte_stable_across_turns() -> None:
    # Two independent turns → the cached static block must be identical bytes so
    # Anthropic actually hits the cache.
    r1 = StubResponse([TextBlock("a")], "end_turn")
    r2 = StubResponse([TextBlock("b")], "end_turn")
    bot, client, _ = make_chatbot([r1, r2])
    bot.chat("Question une")
    bot.chat("Question deux")
    s1 = _system_blocks(client.calls[0]["system"])[0]["text"]
    s2 = _system_blocks(client.calls[1]["system"])[0]["text"]
    assert s1 == s2 == SYSTEM_PROMPT_STATIC


def test_no_information_lost_moving_summary() -> None:
    # The concatenation of the two blocks must still contain everything the old
    # single-string prompt did (identity + rules + summary), just reordered.
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("Bonjour")
    blocks = _system_blocks(client.calls[0]["system"])
    joined = "\n".join(b["text"] for b in blocks)
    assert SIGNAL_CONTEXT_TEMPLATE.split("{")[0].strip() in joined  # "CONTEXTE INITIAL"
    assert "instruments_tracked" in joined
    assert "RÈGLES STRICTES" in joined


# --------------------------------------------------------------------------- #
# Lever 5a — server-side history truncation
# --------------------------------------------------------------------------- #


def _history(n: int) -> list[dict]:
    out: list[dict] = []
    for i in range(n):
        out.append({"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"})
    return out


def test_history_is_capped_server_side() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("actuel", conversation_history=_history(40))
    sent = client.calls[0]["messages"]
    # kept history (≤ MAX_MODEL_HISTORY) + the current user message
    assert len(sent) <= MAX_MODEL_HISTORY + 1
    assert sent[-1]["content"] == "actuel"


def test_truncated_history_starts_on_a_user_turn() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    # 41 messages → last MAX_MODEL_HISTORY slice would start on an assistant turn;
    # the truncator must drop it so the transcript opens on 'user'.
    bot.chat("actuel", conversation_history=_history(41))
    sent = client.calls[0]["messages"]
    assert sent[0]["role"] == "user"


def test_short_history_is_untouched() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("actuel", conversation_history=_history(4))
    sent = client.calls[0]["messages"]
    assert len(sent) == 5  # 4 history + current


# --------------------------------------------------------------------------- #
# Lever 6 + G — response cap & timeout
# --------------------------------------------------------------------------- #


def test_response_length_cap_is_applied() -> None:
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    bot.chat("Bonjour")
    assert client.calls[0]["max_tokens"] == DEFAULT_MAX_TOKENS == 768


def test_timeout_is_passed_to_every_model_call() -> None:
    # A tool-use round → two create() calls; both must carry the timeout.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "H1"})], "tool_use")
    r2 = StubResponse([TextBlock("Lecture faite.")], "end_turn")
    bot, client, _ = make_chatbot([r1, r2])
    bot.chat("Décris XAUUSD H1")
    assert len(client.calls) == 2
    for call in client.calls:
        assert call["timeout"] == DEFAULT_TIMEOUT_S


# --------------------------------------------------------------------------- #
# Security non-regression — the count of checks must NEVER decrease
# --------------------------------------------------------------------------- #

# Baselines measured on 271443a (MIA-2 diagnostic). The guard uses >= so ADDING
# a check is allowed; REMOVING one fails the build.
_ADVERSARIAL_BASELINE = {
    "jailbreak": 8,
    "trade_request": 9,
    "persona_hijack": 9,
    "financial_advice": 9,
}
_FORBIDDEN_BASELINE = {
    "action_trading": 26,
    "recommandation": 41,
    "jugement_moment": 21,
    "jugement_risque": 27,
}
_VIEW_ACTIONS_BASELINE = 11


def test_adversarial_check_count_does_not_regress() -> None:
    for cat, floor in _ADVERSARIAL_BASELINE.items():
        assert len(ADVERSARIAL_PATTERNS_BY_CATEGORY[cat]) >= floor, cat
    total = sum(len(v) for v in ADVERSARIAL_PATTERNS_BY_CATEGORY.values())
    assert total >= sum(_ADVERSARIAL_BASELINE.values()) == 35


def test_forbidden_token_count_does_not_regress() -> None:
    for cat, floor in _FORBIDDEN_BASELINE.items():
        assert len(FORBIDDEN_TOKENS_BY_CATEGORY[cat]) >= floor, cat
    total = sum(len(v) for v in FORBIDDEN_TOKENS_BY_CATEGORY.values())
    assert total >= sum(_FORBIDDEN_BASELINE.values()) == 115


def test_view_action_whitelist_does_not_regress() -> None:
    assert len(ALLOWED_ACTIONS) >= _VIEW_ACTIONS_BASELINE


# --------------------------------------------------------------------------- #
# Invented identifier is still rejected (Couche 4)
# --------------------------------------------------------------------------- #


def test_invented_zone_id_is_rejected() -> None:
    # The model tries to focus a zone id the engine never emitted this turn (no
    # get_market_reading was called), then answers. The invented id must NOT be
    # recorded as a view action.
    r1 = StubResponse(
        [ToolUseBlock("apply_chart_view", {"action": "focus_zone", "params": {"zone_id": "ob-INVENTED-999"}})],
        "tool_use",
    )
    r2 = StubResponse([TextBlock("Je n'affiche que ce que le marché montre.")], "end_turn")
    bot, client, _ = make_chatbot([r1, r2])
    out = bot.chat("centre-toi sur l'OB ob-INVENTED-999")
    assert out.view_actions == []
    # And the tool_result handed back to the model reports the rejection.
    tool_result = client.calls[1]["messages"][-1]["content"][0]["content"]
    assert "rejected" in tool_result


# --------------------------------------------------------------------------- #
# In-turn de-duplication — same tool + same context runs once
# --------------------------------------------------------------------------- #


def test_same_read_not_executed_twice_in_a_turn() -> None:
    # The model asks for the SAME reading across two tool-use rounds. The engine
    # must be hit once for that combo (dedup), not twice.
    call = {"instrument": "XAUUSD", "timeframe": "H1"}
    r1 = StubResponse([ToolUseBlock("get_market_reading", dict(call), id="a")], "tool_use")
    r2 = StubResponse([ToolUseBlock("get_market_reading", dict(call), id="b")], "tool_use")
    r3 = StubResponse([TextBlock("ok")], "end_turn")
    assembler = StubAssembler()
    bot, _client, assembler = make_chatbot([r1, r2, r3], assembler=assembler)
    # Prime the 60s summary cache, then isolate the TOOL path from the perimeter
    # pre-read so the count reflects only tool executions.
    bot._summary_provider.get()  # type: ignore[attr-defined]
    assembler.calls.clear()
    bot.chat("décris puis re-décris XAUUSD H1")
    # r1 executes the read; r2 asks for the SAME (instrument, timeframe) a round
    # later and MUST hit the in-turn cache instead of re-executing.
    assert assembler.calls.count(("XAUUSD", "H1")) == 1


# --------------------------------------------------------------------------- #
# Budgets — deterministic overhead & activity-signal latency
# --------------------------------------------------------------------------- #


def test_activity_signal_precedes_any_model_call() -> None:
    # Budget: "signal d'activité visible < 200 ms". The {activity} event must be
    # yielded BEFORE the first model round-trip, so the client can render it
    # without waiting on the network. With an instant stub this is ~0 ms.
    resp = StubResponse([TextBlock("ok")], "end_turn")
    bot, client, _ = make_chatbot([resp])
    t0 = time.perf_counter()
    events = bot.chat_events("Bonjour")
    first = next(events)
    dt_ms = (time.perf_counter() - t0) * 1000
    assert first == {"event": "activity"}
    assert client.calls == []  # no model call has happened yet
    assert dt_ms < 200.0
    # drain the rest so the generator completes cleanly
    for _ in events:
        pass


def test_non_model_overhead_is_negligible() -> None:
    # Budget headroom: everything OUR code does around the model call (Couche 1,
    # summary assembly with a cached provider, prompt build, Couche 3, event
    # plumbing) must be a tiny fraction of the <3s no-tool budget. This isolates
    # our overhead from the (stubbed here) LLM latency. Live end-to-end budgets
    # run in tests/test_chatbot_budgets_live.py under a real key + DB.
    resp = StubResponse([TextBlock("Le marché XAU H1 est en tendance haussière.")], "end_turn")
    bot, _client, _ = make_chatbot([resp])
    bot.chat("warm up the summary cache")  # prime the 60s summary cache
    resp2 = StubResponse([TextBlock("Réponse.")], "end_turn")
    bot._client.responses.append(resp2)  # type: ignore[attr-defined]
    t0 = time.perf_counter()
    out = bot.chat("Quelles conditions sur XAUUSD H1 ?")
    dt_ms = (time.perf_counter() - t0) * 1000
    assert out.blocked_reason is None
    assert dt_ms < 150.0  # our overhead, excluding real network/generation
