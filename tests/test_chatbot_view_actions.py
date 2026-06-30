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
    FairValueGap,
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


def test_multi_layer_toggle_ok_and_dedupes() -> None:
    # « enlève tous les FVG et les OB » → masque deux couches en une seule action.
    res = v().validate(
        {"action": "set_layer_visibility", "params": {"layers": ["fvg", "ob"], "visible": False}}
    )
    assert res.valid
    assert res.action == {
        "action": "set_layer_visibility",
        "params": {"layers": ["fvg", "ob"], "visible": False},
    }
    # Order-preserving dedupe.
    res2 = v().validate(
        {"action": "set_layer_visibility", "params": {"layers": ["ob", "ob", "fvg"], "visible": True}}
    )
    assert res2.valid
    assert res2.action["params"]["layers"] == ["ob", "fvg"]


def test_multi_layer_rejects_bad_inputs() -> None:
    # "all" is not addressable inside an explicit subset (use the single-layer form).
    bad_all = v().validate(
        {"action": "set_layer_visibility", "params": {"layers": ["fvg", "all"], "visible": False}}
    )
    assert not bad_all.valid and bad_all.reason == "bad_layer"
    # Empty list / non-list.
    empty = v().validate({"action": "set_layer_visibility", "params": {"layers": [], "visible": False}})
    assert not empty.valid and empty.reason == "bad_layer"
    not_list = v().validate(
        {"action": "set_layer_visibility", "params": {"layers": "fvg", "visible": False}}
    )
    assert not not_list.valid and not_list.reason == "bad_layer"
    # Mixing layer + layers is ambiguous.
    mixed = v().validate(
        {"action": "set_layer_visibility", "params": {"layer": "fvg", "layers": ["ob"], "visible": False}}
    )
    assert not mixed.valid and mixed.reason == "bad_layer"
    # visible is still required and must be a bool.
    bad_visible = v().validate(
        {"action": "set_layer_visibility", "params": {"layers": ["fvg", "ob"], "visible": "yes"}}
    )
    assert not bad_visible.valid and bad_visible.reason == "bad_visible"


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


# ---- hide / isolate / show by id (display-only masking, reversible) --------- #


def test_hide_zones_requires_known_ids() -> None:
    known = {"ob_1", "fvg_2"}
    ok = v().validate(
        {"action": "hide_zones", "params": {"zone_ids": ["ob_1", "fvg_2"]}},
        known_zone_ids=known,
    )
    assert ok.valid
    assert ok.action == {"action": "hide_zones", "params": {"zone_ids": ["ob_1", "fvg_2"]}}


def test_hide_zones_rejects_any_invented_id() -> None:
    # « masque l'OB à 4160 » when no real OB matches → one invented id rejects the
    # whole action and NOTHING is hidden.
    res = v().validate(
        {"action": "hide_zones", "params": {"zone_ids": ["ob_1", "ob_at_4160"]}},
        known_zone_ids={"ob_1"},
    )
    assert not res.valid
    assert res.reason == "unknown_zone_id"


def test_hide_zones_dedupes_ids() -> None:
    res = v().validate(
        {"action": "hide_zones", "params": {"zone_ids": ["ob_1", "ob_1"]}},
        known_zone_ids={"ob_1"},
    )
    assert res.valid
    assert res.action["params"]["zone_ids"] == ["ob_1"]


def test_hide_zones_empty_list_rejected() -> None:
    res = v().validate({"action": "hide_zones", "params": {"zone_ids": []}}, known_zone_ids={"ob_1"})
    assert not res.valid
    assert res.reason == "empty_zone_ids"


def test_hide_zones_non_list_rejected() -> None:
    res = v().validate({"action": "hide_zones", "params": {"zone_ids": "ob_1"}}, known_zone_ids={"ob_1"})
    assert not res.valid
    assert res.reason == "bad_zone_ids"


def test_isolate_zones_requires_known_ids() -> None:
    ok = v().validate(
        {"action": "isolate_zones", "params": {"zone_ids": ["ob_1"]}}, known_zone_ids={"ob_1"}
    )
    assert ok.valid
    invented = v().validate(
        {"action": "isolate_zones", "params": {"zone_ids": ["nope"]}}, known_zone_ids={"ob_1"}
    )
    assert not invented.valid and invented.reason == "unknown_zone_id"


def test_show_zones_no_ids_restores_all() -> None:
    # show_zones with no ids = restore everything; no id lock needed.
    res = v().validate({"action": "show_zones", "params": {}}, known_zone_ids=set())
    assert res.valid
    assert res.action == {"action": "show_zones", "params": {}}


def test_show_zones_with_ids_validates_them() -> None:
    ok = v().validate({"action": "show_zones", "params": {"zone_ids": ["ob_1"]}}, known_zone_ids={"ob_1"})
    assert ok.valid and ok.action["params"]["zone_ids"] == ["ob_1"]
    bad = v().validate({"action": "show_zones", "params": {"zone_ids": ["ghost"]}}, known_zone_ids={"ob_1"})
    assert not bad.valid and bad.reason == "unknown_zone_id"


def test_show_zones_empty_list_allowed_as_restore() -> None:
    res = v().validate({"action": "show_zones", "params": {"zone_ids": []}}, known_zone_ids=set())
    assert res.valid
    assert res.action["params"]["zone_ids"] == []


def test_mask_actions_reject_geometry_param() -> None:
    # The hard geometry guard still applies — no coordinate may ride along.
    for action in ("hide_zones", "isolate_zones"):
        res = v().validate(
            {"action": action, "params": {"zone_ids": ["ob_1"], "price": 4160}},
            known_zone_ids={"ob_1"},
        )
        assert not res.valid, action
        assert res.reason == "geometry_param_forbidden"


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


def test_multi_layer_toggle_surfaces() -> None:
    # « enlève tous les FVG et les OB » → une seule action multi-couches, validée
    # et remontée telle quelle (les breaks restent visibles).
    r1 = StubResponse(
        [ToolUseBlock("apply_chart_view", {"action": "set_layer_visibility", "params": {"layers": ["fvg", "ob"], "visible": False}}, id="t1")],
        "tool_use",
    )
    r2 = StubResponse([TextBlock("J'ai masqué les FVG et les OB.")], "end_turn")
    bot, _ = _make_bot([r1, r2], StubAssembler())
    out = bot.chat("Enlève tous les FVG et les OB")
    assert out.blocked_reason is None
    assert out.view_actions == [
        {"action": "set_layer_visibility", "params": {"layers": ["fvg", "ob"], "visible": False}}
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


def test_hide_real_zone_by_id_after_reading_surfaces() -> None:
    # The model reads the combo (harvesting the OB id), THEN hides that real zone.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "hide_zones", "params": {"zone_ids": ["ob_1"]}}, id="t2")], "tool_use")
    r3 = StubResponse([TextBlock("J'ai masqué cet order block.")], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], _ReadingAssembler("ob_1"))
    out = bot.chat("Masque l'OB à 2378")
    assert out.view_actions == [{"action": "hide_zones", "params": {"zone_ids": ["ob_1"]}}]
    assert out.blocked_reason is None


def test_hide_invented_zone_is_rejected_and_nothing_hidden() -> None:
    # « masque l'OB à 4160 » with no matching real zone → invented id → rejected,
    # nothing recorded; the on-brand refusal is handed back to the model.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse([ToolUseBlock("apply_chart_view", {"action": "hide_zones", "params": {"zone_ids": ["ob_at_4160"]}}, id="t2")], "tool_use")
    r3 = StubResponse([TextBlock(VIEW_ACTION_REFUSAL_TEMPLATE)], "end_turn")
    bot, client = _make_bot([r1, r2, r3], _ReadingAssembler("ob_1"))
    out = bot.chat("Masque l'OB à 4160")
    assert out.view_actions == []  # invented zone never recorded
    tool_result_msg = client.calls[2]["messages"][-1]
    assert VIEW_ACTION_REFUSAL_TEMPLATE in tool_result_msg["content"][0]["content"]


# ---- Group resolution by factual state (« masque les FVG touchés ») --------- #


def _fvg(fvg_id: str, status: str) -> FairValueGap:
    return FairValueGap(
        id=fvg_id,
        level_high=2380.0,
        level_low=2378.0,
        status=status,  # type: ignore[arg-type]
        created_at=datetime(2026, 6, 5, 12, 0, tzinfo=timezone.utc),
        tested=status != "active",
    )


def _reading_with_fvgs(fvgs: list[FairValueGap]) -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument="XAUUSD",
            timeframe="M15",
            candle_close_ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(fair_value_gaps=fvgs),
        regime=MarketReadingRegime(
            trend="bullish",
            volatility_observed="elevated",
            market_phase="expansion",
            mtf_confluence={},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=[], description="desc", description_source="template_fallback"
        ),
    )


class _FvgAssembler:
    """Assembler returning a reading carrying several FVGs of mixed status."""

    def __init__(self, fvgs: list[FairValueGap]) -> None:
        self._fvgs = fvgs

    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        return _reading_with_fvgs(list(self._fvgs))


def test_reading_exposes_per_zone_status_for_group_resolution() -> None:
    # The group criterion (« les FVG touchés ») is resolvable ONLY because each
    # zone in the reading carries both an id and a status. Guard that contract.
    reading = _reading_with_fvgs(
        [_fvg("fvg_a", "active"), _fvg("fvg_b", "partially_filled")]
    )
    dumped = reading.model_dump(mode="json")["structure"]["fair_value_gaps"]
    assert {z["id"]: z["status"] for z in dumped} == {
        "fvg_a": "active",
        "fvg_b": "partially_filled",
    }


def test_hide_touched_fvg_group_targets_the_right_ids() -> None:
    # « masque les FVG touchés » : the model reads the combo (3 FVGs, 2 of them
    # partially_filled = touched), resolves the GROUP to the two touched ids, and
    # hides them in a SINGLE multi-id call. The untouched/active FVG stays.
    fvgs = [
        _fvg("fvg_active", "active"),
        _fvg("fvg_touch1", "partially_filled"),
        _fvg("fvg_touch2", "partially_filled"),
    ]
    r1 = StubResponse(
        [ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")],
        "tool_use",
    )
    r2 = StubResponse(
        [
            ToolUseBlock(
                "apply_chart_view",
                {"action": "hide_zones", "params": {"zone_ids": ["fvg_touch1", "fvg_touch2"]}},
                id="t2",
            )
        ],
        "tool_use",
    )
    r3 = StubResponse([TextBlock("J'ai masqué les FVG touchés.")], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], _FvgAssembler(fvgs))
    out = bot.chat("Masque les FVG touchés")
    assert out.blocked_reason is None
    assert out.view_actions == [
        {"action": "hide_zones", "params": {"zone_ids": ["fvg_touch1", "fvg_touch2"]}}
    ]


def test_group_with_one_invented_id_rejects_whole_action() -> None:
    # If the model's resolved group contains a single id the engine did not emit
    # (a hallucinated FVG), the WHOLE hide is rejected — nothing is masked.
    fvgs = [_fvg("fvg_touch1", "partially_filled")]
    r1 = StubResponse(
        [ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")],
        "tool_use",
    )
    r2 = StubResponse(
        [
            ToolUseBlock(
                "apply_chart_view",
                {"action": "hide_zones", "params": {"zone_ids": ["fvg_touch1", "fvg_ghost"]}},
                id="t2",
            )
        ],
        "tool_use",
    )
    r3 = StubResponse([TextBlock(VIEW_ACTION_REFUSAL_TEMPLATE)], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], _FvgAssembler(fvgs))
    out = bot.chat("Masque les FVG touchés")
    assert out.view_actions == []  # one invented id → nothing hidden


def test_isolate_then_show_restore_roundtrip() -> None:
    # isolate a real zone, then restore — both are display-only, reversible. Both
    # view actions ride in ONE assistant turn (after the reading) so the tool-turn
    # budget isn't exhausted.
    r1 = StubResponse([ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "M15"}, id="t1")], "tool_use")
    r2 = StubResponse(
        [
            ToolUseBlock("apply_chart_view", {"action": "isolate_zones", "params": {"zone_ids": ["ob_1"]}}, id="t2"),
            ToolUseBlock("apply_chart_view", {"action": "show_zones", "params": {}}, id="t3"),
        ],
        "tool_use",
    )
    r3 = StubResponse([TextBlock("J'ai isolé puis tout réaffiché.")], "end_turn")
    bot, _ = _make_bot([r1, r2, r3], _ReadingAssembler("ob_1"))
    out = bot.chat("Isole l'OB puis réaffiche tout")
    assert out.view_actions == [
        {"action": "isolate_zones", "params": {"zone_ids": ["ob_1"]}},
        {"action": "show_zones", "params": {}},
    ]
    assert out.blocked_reason is None
