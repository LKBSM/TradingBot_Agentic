"""Couche 4 — view-action whitelist tests (display-only chart control).

Two layers under test:
  1. ViewActionValidator — the pure whitelist gate (accept / reject / clamp).
  2. Chatbot integration — the model emits apply_chart_view tool calls; valid
     ones surface as ChatResponse.view_actions, off-list ones get the on-brand
     refusal and NEVER appear in view_actions. Detection is never mutated.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from src.intelligence.chatbot.chatbot import Chatbot
from src.intelligence.chatbot.constants import VIEW_ACTION_REFUSAL_TEMPLATE
from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider
from src.intelligence.chatbot.view_action_filter import ViewActionValidator
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
    OrderBlock,
)

# Reuse the lightweight stubs from the Couche 2 suite.
from tests.test_chatbot import (  # type: ignore
    StubAssembler,
    StubClient,
    TextBlock,
    ToolUseBlock,
    StubResponse,
)


# --------------------------------------------------------------------------- #
# ViewActionValidator — pure whitelist gate
# --------------------------------------------------------------------------- #


def v() -> ViewActionValidator:
    return ViewActionValidator()


def test_set_layer_visibility_ok() -> None:
    res = v().validate({"action": "set_layer_visibility", "params": {"layer": "fvg", "visible": False}})
    assert res.valid
    assert res.action == {"action": "set_layer_visibility", "params": {"layer": "fvg", "visible": False}}


def test_layer_all_is_allowed() -> None:
    res = v().validate({"action": "set_layer_visibility", "params": {"layer": "all", "visible": True}})
    assert res.valid


def test_bad_layer_rejected() -> None:
    res = v().validate({"action": "set_layer_visibility", "params": {"layer": "candles", "visible": True}})
    assert not res.valid
    assert res.reason == "bad_layer"


def test_visible_must_be_bool() -> None:
    res = v().validate({"action": "set_layer_visibility", "params": {"layer": "ob", "visible": "yes"}})
    assert not res.valid
    assert res.reason == "bad_visible"


def test_filter_zones_clamps_thresholds() -> None:
    res = v().validate(
        {"action": "filter_zones", "params": {"active_only": True, "proximity_pct": 999, "min_size_pct": -3}}
    )
    assert res.valid
    p = res.action["params"]
    assert p["active_only"] is True
    assert p["proximity_pct"] == 10.0  # clamped to max
    assert p["min_size_pct"] == 0.0  # clamped to min


def test_empty_filter_rejected() -> None:
    res = v().validate({"action": "filter_zones", "params": {}})
    assert not res.valid
    assert res.reason == "empty_filter"


def test_focus_price_and_fit_and_reset_ok() -> None:
    for action in ("focus_price", "fit_chart", "reset_view"):
        res = v().validate({"action": action})
        assert res.valid, action
        assert res.action == {"action": action, "params": {}}


def test_set_instrument_timeframe_enums() -> None:
    ok = v().validate({"action": "set_instrument_timeframe", "params": {"instrument": "EURUSD", "timeframe": "H4"}})
    assert ok.valid
    bad_i = v().validate({"action": "set_instrument_timeframe", "params": {"instrument": "BTCUSD", "timeframe": "H4"}})
    assert not bad_i.valid and bad_i.reason == "bad_instrument"
    bad_tf = v().validate({"action": "set_instrument_timeframe", "params": {"instrument": "XAUUSD", "timeframe": "M5"}})
    assert not bad_tf.valid and bad_tf.reason == "bad_timeframe"


def test_focus_zone_requires_known_id() -> None:
    known = {"ob_123"}
    ok = v().validate({"action": "focus_zone", "params": {"zone_id": "ob_123"}}, known_zone_ids=known)
    assert ok.valid
    invented = v().validate({"action": "focus_zone", "params": {"zone_id": "ob_at_2000"}}, known_zone_ids=known)
    assert not invented.valid
    assert invented.reason == "unknown_zone_id"


def test_highlight_zone_requires_known_id() -> None:
    res = v().validate({"action": "highlight_zone", "params": {"zone_id": "x"}}, known_zone_ids=set())
    assert not res.valid
    assert res.reason == "unknown_zone_id"


# ---- The inviolable line: no create/move/resize, no geometry ---------------- #


def test_off_list_action_rejected() -> None:
    for action in ("create_zone", "place_ob", "move_zone", "resize_fvg", "set_zone", "delete_zone"):
        res = v().validate({"action": action, "params": {}})
        assert not res.valid, action
        assert res.reason == "action_not_whitelisted"


def test_geometry_param_is_hard_rejected() -> None:
    # Even on a whitelisted action, a price/level/geometry key is refused.
    for key in ("price", "level", "level_high", "level_low", "high", "low"):
        res = v().validate({"action": "focus_zone", "params": {"zone_id": "ob_1", key: 2000}}, known_zone_ids={"ob_1"})
        assert not res.valid, key
        assert res.reason == "geometry_param_forbidden"


def test_non_object_and_bad_params_rejected() -> None:
    assert not v().validate("masque les FVG").valid
    assert not v().validate({"action": "fit_chart", "params": []}).valid


# --------------------------------------------------------------------------- #
# Chatbot integration — view_actions surfaced, refusal on off-list
# --------------------------------------------------------------------------- #


def _reading_with_ob(ob_id: str = "ob_1") -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument="XAUUSD",
            timeframe="M15",
            candle_close_ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(
            order_blocks=[
                OrderBlock(
                    id=ob_id,
                    level_high=2380.0,
                    level_low=2375.0,
                    importance="high",
                    status="active",
                    created_at=datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
                    tested=False,
                    user_flagged=False,
                )
            ]
        ),
        regime=MarketReadingRegime(
            trend="bullish",
            volatility_observed="elevated",
            market_phase="expansion",
            mtf_confluence={},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=[],
            description="desc",
            description_source="template_fallback",
        ),
    )


class _ReadingAssembler:
    """Assembler returning a reading that carries one detected OB."""

    def __init__(self, ob_id: str = "ob_1") -> None:
        self.ob_id = ob_id
        self.calls: list[tuple[str, str]] = []

    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        self.calls.append((instrument, timeframe))
        return _reading_with_ob(self.ob_id)


def _make_bot(responses: list[Any], assembler: Any) -> tuple[Chatbot, StubClient]:
    provider = SignalSummaryProvider(assembler)
    client = StubClient(responses)
    bot = Chatbot(anthropic_client=client, summary_provider=provider, assembler=assembler)
    return bot, client


def test_view_action_layer_toggle_surfaces() -> None:
    r1 = StubResponse(
        [ToolUseBlock("apply_chart_view", {"action": "set_layer_visibility", "params": {"layer": "fvg", "visible": False}}, id="t1")],
        "tool_use",
    )
    r2 = StubResponse([TextBlock("J'ai masqué les FVG.")], "end_turn")
    bot, _ = _make_bot([r1, r2], StubAssembler())
    out = bot.chat("Masque les FVG")
    assert out.blocked_reason is None
    assert out.content == "J'ai masqué les FVG."
    assert out.view_actions == [
        {"action": "set_layer_visibility", "params": {"layer": "fvg", "visible": False}}
    ]


def test_focus_zone_after_reading_is_allowed() -> None:
    # The model reads the combo (harvesting the OB id), THEN focuses it.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "focus_zone", "params": {"zone_id": "ob_1"}}, id="t2")], "tool_use")
    r3 = StubResponse([TextBlock("Je me centre sur l'OB actif.")], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], _ReadingAssembler("ob_1"))
    out = bot.chat("Centre-toi sur l'order block")
    assert out.view_actions == [{"action": "focus_zone", "params": {"zone_id": "ob_1"}}]
    assert out.blocked_reason is None


def test_invented_zone_is_rejected_with_on_brand_message() -> None:
    # "centre-toi sur l'OB à 2000" → the model emits an invented id; it must be
    # rejected (refusal handed back to the model) and NEVER recorded.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "focus_zone", "params": {"zone_id": "ob_2000"}}, id="t2")], "tool_use")
    r3 = StubResponse([TextBlock(VIEW_ACTION_REFUSAL_TEMPLATE)], "end_turn")
    bot, client = _make_bot([r1, r2, r3], _ReadingAssembler("ob_1"))
    out = bot.chat("Centre-toi sur l'OB à 2000")
    assert out.view_actions == []  # invented zone never recorded
    # the tool_result handed back to the model carried the on-brand refusal
    tool_result_msg = client.calls[2]["messages"][-1]
    assert VIEW_ACTION_REFUSAL_TEMPLATE in tool_result_msg["content"][0]["content"]


def test_create_structure_request_not_representable() -> None:
    # Even if the model tried an off-list action, it is rejected and not recorded.
    r1 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "create_zone", "params": {"price": 2000}}, id="t1")], "tool_use")
    r2 = StubResponse([TextBlock(VIEW_ACTION_REFUSAL_TEMPLATE)], "end_turn")
    bot, _ = _make_bot([r1, r2], StubAssembler())
    out = bot.chat("Mets un OB à 2000")
    assert out.view_actions == []


def test_detection_is_never_mutated_by_view_action() -> None:
    # Capture the reading object the assembler hands out; applying a view action
    # must leave its structure byte-for-byte identical (no mutation path).
    assembler = _ReadingAssembler("ob_1")
    reference = _reading_with_ob("ob_1").model_dump(mode="json")
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "highlight_zone", "params": {"zone_id": "ob_1"}}, id="t2")], "tool_use")
    r3 = StubResponse([TextBlock("Mis en évidence.")], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], assembler)
    bot.chat("Mets en évidence l'OB")
    # A fresh generation yields the SAME structure — nothing was written back.
    assert assembler.get_or_generate("XAUUSD", "M15").model_dump(mode="json")["structure"] == reference["structure"]
