"""SC-4 — the citation that lets live typing respect a manual removal.

Two properties are locked here, and they are the whole reason SC-4 can honour
« elle ne choisit rien à ta place » while translating on every typing pause:

  1. A condition carries the fragment of the USER'S OWN sentence it came from,
     and that fragment is verified verbatim server-side. A citation the model
     invented is reduced to ``None`` — never repaired, never guessed.
  2. ``conditions`` stays byte-for-byte POST-able to /api/conditions-scan. The
     citation rides in a PARALLEL array, because ``ScanCondition`` forbids extra
     fields; a citation leaking into a condition would 422 the scan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from tests.conftest_translate_throttle import fresh_translate_throttle  # noqa: F401

from src.api.app import create_app
from src.api.routes.conditions_scan import ScanCondition
from src.api.signal_store import SignalStore
from src.intelligence.scanner_translator import (
    ScannerTranslator,
    TOOL_NAME,
    build_tool_schema,
    sanitize_translation,
    verify_source_phrase,
)


# ── verify_source_phrase — the citation is checked, never trusted ────────────
def test_citation_present_in_text_is_kept() -> None:
    assert verify_source_phrase(
        "Order Block jamais testé", "Un Order Block jamais testé en tendance haussière"
    ) == "Order Block jamais testé"


def test_citation_is_matched_accent_and_case_insensitively() -> None:
    # The model routinely re-cases and re-accents what it quotes; the WORDS are
    # what must be the user's, not the typography.
    assert verify_source_phrase("TENDANCE HAUSSIERE", "en tendance haussière") == "TENDANCE HAUSSIERE"


def test_citation_is_matched_across_irregular_whitespace() -> None:
    assert verify_source_phrase("Order Block jamais testé", "un Order   Block\njamais testé")


@pytest.mark.parametrize(
    "citation",
    [
        "un fragment que l'utilisateur n'a jamais écrit",
        "",
        "   ",
        None,
        42,
        {"nope": True},
    ],
)
def test_citation_absent_or_malformed_becomes_none(citation: Any) -> None:
    assert verify_source_phrase(citation, "Un Order Block jamais testé") is None


def test_no_source_text_means_no_citation() -> None:
    # Nothing to check against ⇒ we do not believe the model by default.
    assert verify_source_phrase("Order Block", None) is None
    assert verify_source_phrase("Order Block", "") is None


def test_citation_is_truncated_not_rejected_when_overlong() -> None:
    long_text = "ordre " * 60  # 360 chars, well past MAX_SOURCE_PHRASE
    kept = verify_source_phrase(long_text, long_text)
    assert kept is not None and len(kept) <= 120


# ── sanitize_translation — the two arrays stay aligned ───────────────────────
def _payload(*conditions: dict) -> dict:
    return {
        "conditions": list(conditions),
        "assumptions": [],
        "untranslatable": [],
        "refusal": None,
    }


def test_sources_align_with_conditions_index_for_index() -> None:
    out = sanitize_translation(
        _payload(
            {"type": "trend_is", "trend": "bullish", "source_phrase": "tendance haussière"},
            {"type": "price_in_ob", "source_phrase": "jamais dit par l'utilisateur"},
            {"type": "zone_untested", "source_phrase": "jamais testé"},
        ),
        "Un Order Block jamais testé, en tendance haussière",
    )
    assert len(out["condition_sources"]) == len(out["conditions"])
    assert out["condition_sources"] == ["tendance haussière", None, "jamais testé"]


def test_dedupe_keeps_conditions_and_sources_aligned() -> None:
    out = sanitize_translation(
        _payload(
            {"type": "trend_is", "trend": "bullish", "source_phrase": "tendance haussière"},
            {"type": "trend_is", "trend": "bullish", "source_phrase": "haussière"},
            {"type": "zone_untested", "source_phrase": "jamais testé"},
        ),
        "jamais testé, en tendance haussière",
    )
    assert out["conditions"] == [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}]
    # First occurrence wins, citation included.
    assert out["condition_sources"] == ["tendance haussière", "jamais testé"]


def test_citation_never_leaks_into_the_wire_condition() -> None:
    """The guarantee the scan endpoint depends on: ``conditions`` is POST-able."""
    out = sanitize_translation(
        _payload({"type": "trend_is", "trend": "bullish", "source_phrase": "tendance haussière"}),
        "en tendance haussière",
    )
    assert "source_phrase" not in out["conditions"][0]
    # ScanCondition has extra="forbid" — this is the 422 that must never happen.
    ScanCondition(**out["conditions"][0])


@pytest.mark.parametrize("outcome_payload", ["empty", "refused", "error"])
def test_every_early_return_carries_the_key(outcome_payload: str) -> None:
    """A missing key would be an AttributeError on the client, not an absence."""
    stub = _Client([])
    translator = ScannerTranslator(anthropic_client=stub)
    if outcome_payload == "empty":
        result = translator.translate("   ")
    elif outcome_payload == "refused":
        result = translator.translate("quels sont les meilleurs setups")
    else:
        result = translator.translate("un order block")  # stub raises: no response queued
    assert result["condition_sources"] == []


def test_schema_offers_source_phrase_without_requiring_it() -> None:
    """Optional on purpose: an invented citation is worse than an absent one."""
    item = build_tool_schema()["input_schema"]["properties"]["conditions"]["items"]
    assert "source_phrase" in item["properties"]
    assert "source_phrase" not in item["required"]


# ── endpoint — the array survives the belt-and-suspenders revalidation ───────
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
        return self._p.responses.pop(0)


class _Client:
    def __init__(self, responses: Optional[list] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.messages = _Msgs(self)


def _app_with(translator: Any, tmp_path: Any) -> TestClient:
    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    app = create_app(signal_store=store)
    app.state.app_state.scanner_translator = translator
    return TestClient(app)


def test_endpoint_returns_aligned_sources(tmp_path: Any) -> None:
    stub = _Client([_Resp([_ToolBlock(name=TOOL_NAME, input=_payload(
        {"type": "trend_is", "trend": "bullish", "source_phrase": "tendance haussière"},
        {"type": "zone_untested", "source_phrase": "inventé de toutes pièces"},
    ))])])
    client = _app_with(ScannerTranslator(anthropic_client=stub), tmp_path)
    body = client.post(
        "/api/scanner/translate",
        json={"text": "Un Order Block en tendance haussière", "locale": "fr"},
    ).json()
    assert body["conditions"] == [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}]
    assert body["condition_sources"] == ["tendance haussière", None]


def test_a_condition_dropped_by_revalidation_takes_its_source_with_it(tmp_path: Any) -> None:
    """Alignment must survive the drop, or every later citation shifts by one."""
    from src.api.routes import scanner_translate as mod

    conditions = [
        {"type": "trend_is", "trend": "bullish"},
        {"type": "trend_is", "trend": "not_a_real_value"},  # ScanCondition rejects
        {"type": "zone_untested"},
    ]
    sources = ["haussière", "bidon", "jamais testé"]
    safe, safe_sources = mod._revalidate_conditions(conditions, sources)
    assert safe == [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}]
    assert safe_sources == ["haussière", "jamais testé"]


def test_shorter_sources_array_pads_with_none_instead_of_truncating(tmp_path: Any) -> None:
    from src.api.routes import scanner_translate as mod

    safe, safe_sources = mod._revalidate_conditions(
        [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}], ["haussière"]
    )
    assert len(safe) == len(safe_sources) == 2
    assert safe_sources == ["haussière", None]


# -- the SC-4 cap: typing pauses spend money, so the ceiling lives in the code --
def test_translate_is_capped_and_says_when_to_retry(tmp_path: Any, monkeypatch: Any) -> None:
    """Live typing removes human patience as the ceiling; this puts one back.

    The deployed entrypoint builds the app with no per-IP limiter at all
    (``src.api.asgi:app`` -> ``create_app()``), so this endpoint has to cap
    itself or a single open tab can spend without bound.
    """
    from src.api.auth_throttle import AuthThrottle
    from src.api.routes import scanner_translate as mod

    monkeypatch.setattr(mod, "_TRANSLATE_THROTTLE", AuthThrottle(max_attempts=3, window_s=300))

    payload = _payload({"type": "trend_is", "trend": "bullish", "source_phrase": "haussière"})
    stub = _Client([_Resp([_ToolBlock(name=TOOL_NAME, input=payload)]) for _ in range(3)])
    client = _app_with(ScannerTranslator(anthropic_client=stub), tmp_path)

    body = {"text": "une tendance haussière", "locale": "fr"}
    for _ in range(3):
        assert client.post("/api/scanner/translate", json=body).status_code == 200

    refused = client.post("/api/scanner/translate", json=body)
    assert refused.status_code == 429
    assert int(refused.headers["Retry-After"]) >= 1
    # The 4th request never reached the translator — that is the entire point.
    assert len(stub.calls) == 3


def test_cap_can_be_disabled_for_a_fronted_deployment(tmp_path: Any, monkeypatch: Any) -> None:
    """``TRANSLATE_THROTTLE_MAX=0`` is the documented escape hatch."""
    from src.api.auth_throttle import AuthThrottle
    from src.api.routes import scanner_translate as mod

    monkeypatch.setattr(mod, "_TRANSLATE_THROTTLE", AuthThrottle(max_attempts=0))
    payload = _payload({"type": "trend_is", "trend": "bullish", "source_phrase": "haussière"})
    stub = _Client([_Resp([_ToolBlock(name=TOOL_NAME, input=payload)]) for _ in range(6)])
    client = _app_with(ScannerTranslator(anthropic_client=stub), tmp_path)
    for _ in range(6):
        assert client.post(
            "/api/scanner/translate", json={"text": "une tendance haussière", "locale": "fr"}
        ).status_code == 200
