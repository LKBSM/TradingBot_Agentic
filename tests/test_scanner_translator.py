"""SC-2 — unit tests for the scanner translator (phrase → closed palette).

These lock the non-negotiable guarantees of §4/§5:
  · a translation carrying an out-of-palette condition is REJECTED by the code;
  · a bad value is rejected, never coerced to a neighbour;
  · ranking / prediction / advice asks produce a refusal, never a search;
  · an untranslatable fragment is named, never replaced by an approaching one;
  · a deduced default is surfaced as an assumption.
The Anthropic client is duck-typed; a scripted stub drives ``translate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.intelligence.conditions_scanner import ALLOWED_CONDITION_TYPES
from src.intelligence.scanner_translator import (
    CONTROL_DOMAINS,
    ScannerTranslator,
    TOOL_NAME,
    build_tool_schema,
    detect_refusal,
    sanitize_condition,
    sanitize_translation,
)


# ── Scripted Anthropic stub (forced tool_use) ────────────────────────────────
@dataclass
class _ToolBlock:
    name: str
    input: dict
    type: str = "tool_use"


@dataclass
class _Resp:
    content: list


class _Msgs:
    def __init__(self, parent: "_Client") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        if not self._p.responses:
            raise AssertionError("no scripted response")
        return self._p.responses.pop(0)


class _Client:
    def __init__(self, responses: list | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.messages = _Msgs(self)


def _tool_resp(payload: dict) -> _Resp:
    return _Resp([_ToolBlock(name=TOOL_NAME, input=payload)])


# ── Palette / schema integrity ───────────────────────────────────────────────
def test_control_domains_cover_every_palette_type() -> None:
    assert set(CONTROL_DOMAINS) == set(ALLOWED_CONDITION_TYPES)


def test_tool_schema_enum_equals_palette() -> None:
    schema = build_tool_schema()
    enum = schema["input_schema"]["properties"]["conditions"]["items"]["properties"]["type"]["enum"]
    assert set(enum) == set(ALLOWED_CONDITION_TYPES)
    # No predictive/blocked type ever leaks into the enum.
    assert "distribution" not in enum
    assert not any("predict" in t or "target" in t for t in enum)


# ── sanitize_condition — the palette gate ────────────────────────────────────
def test_sanitize_accepts_valid_condition() -> None:
    clean, reason = sanitize_condition({"type": "trend_is", "trend": "bullish"})
    assert reason is None
    assert clean == {"type": "trend_is", "trend": "bullish"}


def test_sanitize_rejects_out_of_palette_type() -> None:
    clean, reason = sanitize_condition({"type": "rsi_oversold", "value": 30})
    assert clean is None
    assert reason == "out_of_palette"


def test_sanitize_rejects_predictive_type() -> None:
    clean, reason = sanitize_condition({"type": "price_will_rise"})
    assert clean is None
    assert reason == "out_of_palette"


def test_sanitize_rejects_bad_value_never_coerces() -> None:
    # 0.3 is not a member of the proximity domain — reject, do not snap to 0.25.
    clean, reason = sanitize_condition({"type": "price_near_ob", "proximity_pct": 0.3})
    assert clean is None
    assert reason == "bad_value"


def test_sanitize_drops_inapplicable_field_but_keeps_condition() -> None:
    # ``session`` does not apply to trend_is — dropped, not a rejection.
    clean, reason = sanitize_condition({"type": "trend_is", "trend": "bearish", "session": "london"})
    assert reason is None
    assert clean == {"type": "trend_is", "trend": "bearish"}


def test_sanitize_rejects_bool_as_int() -> None:
    clean, reason = sanitize_condition({"type": "bos_recent_confirmed", "max_bars": True})
    assert clean is None
    assert reason == "bad_value"


# ── sanitize_translation — outcomes ──────────────────────────────────────────
def test_translation_all_recognized_is_translated() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}],
        "assumptions": [],
        "untranslatable": [],
        "refusal": None,
    })
    assert out["outcome"] == "translated"
    assert len(out["conditions"]) == 2


def test_translation_with_untranslatable_is_partial_and_named() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "zone_untested"}],
        "assumptions": [],
        "untranslatable": [{"fragment": "le RSI est en survente", "category": "indicator"}],
        "refusal": None,
    })
    assert out["outcome"] == "partial"
    assert out["untranslatable"][0]["fragment"] == "le RSI est en survente"
    assert out["untranslatable"][0]["category"] == "indicator"
    # The unsupported bit was NOT replaced by an approaching condition.
    assert [c["type"] for c in out["conditions"]] == ["zone_untested"]


def test_out_of_palette_model_condition_is_surfaced_not_swallowed() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "rsi_oversold"}, {"type": "trend_is", "trend": "bullish"}],
        "assumptions": [],
        "untranslatable": [],
        "refusal": None,
    })
    # The invalid one is dropped from conditions AND named as unsupported.
    assert [c["type"] for c in out["conditions"]] == ["trend_is"]
    assert any(u["category"] == "unsupported" for u in out["untranslatable"])
    assert out["outcome"] == "partial"


def test_refusal_short_circuits_conditions() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "trend_is", "trend": "bullish"}],
        "assumptions": [],
        "untranslatable": [],
        "refusal": {"kind": "ranking"},
    })
    assert out["outcome"] == "refused"
    assert out["refusal"] == {"kind": "ranking"}
    assert out["conditions"] == []  # no search from a refusal


def test_assumption_only_kept_when_it_points_at_a_kept_condition() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "liquidity_swept_recent", "max_bars": 10}],
        "assumptions": [
            {"condition_type": "liquidity_swept_recent", "control": "max_bars",
             "value": "10", "source_phrase": "récemment"},
            # dangling assumption pointing at a condition we did not keep → dropped
            {"condition_type": "price_in_fvg", "control": "direction", "value": "any"},
        ],
        "untranslatable": [],
        "refusal": None,
    })
    assert len(out["assumptions"]) == 1
    assert out["assumptions"][0]["source_phrase"] == "récemment"


def test_conditions_are_deduped() -> None:
    out = sanitize_translation({
        "conditions": [{"type": "zone_untested"}, {"type": "zone_untested"}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })
    assert len(out["conditions"]) == 1


def test_empty_understanding_is_none() -> None:
    out = sanitize_translation({
        "conditions": [], "assumptions": [], "untranslatable": [], "refusal": None,
    })
    assert out["outcome"] == "none"


# ── detect_refusal — deterministic pre-filter (fr + en) ──────────────────────
def test_detect_refusal_ranking_fr() -> None:
    assert detect_refusal("Montre-moi les meilleurs setups du moment") == "ranking"


def test_detect_refusal_ranking_en() -> None:
    assert detect_refusal("show me the best setups right now") == "ranking"


def test_detect_refusal_prediction() -> None:
    assert detect_refusal("où va le prix de l'or demain ?") == "prediction"


def test_detect_refusal_advice() -> None:
    assert detect_refusal("qu'est-ce que je devrais trader aujourd'hui ?") == "recommendation"


def test_detect_refusal_clean_sentence_passes() -> None:
    assert detect_refusal("un Order Block jamais testé en tendance haussière") is None


# ── ScannerTranslator.translate — end to end with a stub ─────────────────────
def _translator(responses: list | None = None) -> tuple[ScannerTranslator, _Client]:
    client = _Client(responses)
    return ScannerTranslator(anthropic_client=client), client


def test_translate_refusal_never_calls_llm() -> None:
    tr, client = _translator([])  # no scripted response — must not be needed
    out = tr.translate("Montre-moi les meilleurs marchés")
    assert out["outcome"] == "refused"
    assert out["refusal"]["kind"] == "ranking"
    assert client.calls == []  # LLM never invoked on a flagrant ask


def test_translate_empty_is_empty() -> None:
    tr, client = _translator([])
    out = tr.translate("   ")
    assert out["outcome"] == "empty"
    assert client.calls == []


def test_translate_forces_the_tool_call() -> None:
    tr, client = _translator([_tool_resp({
        "conditions": [{"type": "trend_is", "trend": "bullish"}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })])
    out = tr.translate("tendance haussière")
    assert out["outcome"] == "translated"
    # tool_choice forced the specific tool.
    assert client.calls[0]["tool_choice"] == {"type": "tool", "name": TOOL_NAME}


def test_translate_llm_failure_is_failsafe_error() -> None:
    tr, _ = _translator([])  # create() raises AssertionError (no scripted response)
    out = tr.translate("un order block jamais testé")
    assert out["outcome"] == "error"
    assert out["conditions"] == []


def test_translate_revalidates_model_out_of_palette() -> None:
    # Even if the model emits an out-of-palette type, the code rejects it.
    tr, _ = _translator([_tool_resp({
        "conditions": [{"type": "moving_average_cross"}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })])
    out = tr.translate("croisement de moyennes mobiles")
    assert all(c["type"] in ALLOWED_CONDITION_TYPES for c in out["conditions"])
    assert out["conditions"] == []
    assert any(u["category"] == "unsupported" for u in out["untranslatable"])
