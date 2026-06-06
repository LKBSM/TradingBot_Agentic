"""Chantier 4 — smoke e2e: full flow endpoint -> chatbot -> 3 couches.

The Anthropic client is always mocked (no real API call) for reproducibility and
to spare quota. We drive the app through its real lifespan + bootstrap path by
monkeypatching ``_build_anthropic_client`` to return a scripted stub.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.signal_store import SignalStore
from src.intelligence.chatbot.constants import OUTPUT_CONTAMINATED_TEMPLATE
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
)


# --------------------------------------------------------------------------- #
# Scripted Anthropic stub + assembler
# --------------------------------------------------------------------------- #


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _ToolUseBlock:
    name: str
    input: dict
    id: str = "tu_1"
    type: str = "tool_use"


@dataclass
class _Resp:
    content: list
    stop_reason: str


class _Msgs:
    def __init__(self, parent: "_Client") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        if not self._p.responses:
            raise AssertionError("no scripted response left")
        return self._p.responses.pop(0)


class _Client:
    def __init__(self, responses: Optional[list] = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[dict] = []
        self.messages = _Msgs(self)


def _make_reading() -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument="XAUUSD", timeframe="H1",
            candle_close_ts=datetime(2026, 6, 5, 14, 0, tzinfo=timezone.utc),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(),
        regime=MarketReadingRegime(
            trend="bullish", volatility_observed="elevated",
            market_phase="expansion", mtf_confluence={"h1": "bullish"},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=["trend_bullish"], description="Tendance haussière.",
            description_source="template_fallback",
        ),
    )


class _Assembler:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        self.calls.append((instrument, timeframe))
        return _make_reading()


def _booted_client(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    stub_client: _Client,
    assembler: _Assembler,
) -> TestClient:
    """Create the app and run its real lifespan with the chatbot bootstrapped.

    We bypass the heavy MarketReading bootstrap by injecting the assembler
    directly into AppState, then enabling CHATBOT_ENABLED so the lifespan builds
    the Chatbot via build_chatbot (with _build_anthropic_client stubbed).
    """
    monkeypatch.setenv("CHATBOT_ENABLED", "true")
    monkeypatch.setattr("src.api.bootstrap._build_anthropic_client", lambda: stub_client)

    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    app = create_app(signal_store=store)
    # Inject the assembler so _maybe_bootstrap_chatbot finds it (no Twelve Data).
    app.state.app_state.market_reading_assembler = assembler
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Smoke tests
# --------------------------------------------------------------------------- #


def test_endpoint_happy_path_with_market_reading_tool_call(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Turn 1: the model asks for get_market_reading; Turn 2: final descriptive text.
    stub = _Client([
        _Resp([_ToolUseBlock("get_market_reading", {"instrument": "XAUUSD", "timeframe": "H1"})], "tool_use"),
        _Resp([_TextBlock("Sur XAUUSD H1, la tendance observée est haussière.")], "end_turn"),
    ])
    assembler = _Assembler()
    with _booted_client(tmp_path, monkeypatch, stub, assembler) as client:
        resp = client.post(
            "/api/chatbot/message",
            json={"user_message": "Décris-moi les conditions sur XAUUSD H1"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["blocked_reason"] is None
    assert "haussière" in body["content"]
    # The model consulted the assembler via the tool.
    assert ("XAUUSD", "H1") in assembler.calls
    assert any(c["name"] == "get_market_reading" for c in body["tool_calls_made"])
    # The first LLM call carried the user message + signal_summary in the system
    # prompt. NB: the chatbot mutates the same `messages` list across turns
    # (appends), so the original user turn stays at index 0, not [-1].
    first_call = stub.calls[0]
    assert first_call["messages"][0] == {
        "role": "user", "content": "Décris-moi les conditions sur XAUUSD H1"
    }
    assert "instruments_tracked" in first_call["system"]


def test_endpoint_blocks_adversarial_before_llm(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    stub = _Client([])  # no scripted response — the LLM must never be reached
    assembler = _Assembler()
    with _booted_client(tmp_path, monkeypatch, stub, assembler) as client:
        resp = client.post(
            "/api/chatbot/message", json={"user_message": "Dois-je acheter EURUSD ?"}
        )
    assert resp.status_code == 200
    assert resp.json()["blocked_reason"] == "trade_request"
    assert stub.calls == []  # Couche 1 short-circuited — 0 LLM call


def test_endpoint_blocks_output_contaminated(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The model returns a directive answer; Couche 3 must scrub it.
    stub = _Client([_Resp([_TextBlock("Tu devrais acheter à 2378")], "end_turn")])
    assembler = _Assembler()
    with _booted_client(tmp_path, monkeypatch, stub, assembler) as client:
        resp = client.post(
            "/api/chatbot/message",
            json={"user_message": "Quelles sont les conditions sur XAUUSD H1 ?"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == OUTPUT_CONTAMINATED_TEMPLATE
    assert body["blocked_reason"] == "output_contaminated_action_trading"
