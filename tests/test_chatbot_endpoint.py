"""Chantier 4 — Étape 6 tests: POST /api/chatbot/message + bootstrap."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.bootstrap import BootstrapConfigurationError, build_chatbot
from src.api.dependencies import AppState
from src.api.signal_store import SignalStore
from src.intelligence.chatbot.chatbot import Chatbot
from src.intelligence.chatbot.constants import REFUSAL_TEMPLATE
from src.intelligence.market_reading_schema import (
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
)


# --------------------------------------------------------------------------- #
# Stubs (scripted Anthropic client + assembler)
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

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        if not self._p.responses:
            raise AssertionError("no scripted response")
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
    def get_or_generate(self, instrument: str, timeframe: str) -> MarketReading:
        return _make_reading()


class _RaisingChatbot:
    def chat(self, **kwargs: Any) -> Any:
        raise RuntimeError("secret internal detail xyz")


def _make_real_chatbot(responses: Optional[list] = None) -> tuple[Chatbot, _Client]:
    from src.intelligence.chatbot.signal_summary_provider import SignalSummaryProvider

    client = _Client(responses)
    bot = Chatbot(
        anthropic_client=client,
        summary_provider=SignalSummaryProvider(_Assembler()),
        assembler=_Assembler(),
    )
    return bot, client


def _client_with_chatbot(tmp_path: Any, chatbot: Any) -> TestClient:
    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    app = create_app(signal_store=store)
    app.state.app_state.chatbot = chatbot
    return TestClient(app)


# --------------------------------------------------------------------------- #
# Endpoint — happy / adversarial / errors
# --------------------------------------------------------------------------- #


def test_happy_path_returns_200(tmp_path: Any) -> None:
    bot, _ = _make_real_chatbot([_Resp([_TextBlock("Tendance haussière observée.")], "end_turn")])
    client = _client_with_chatbot(tmp_path, bot)
    resp = client.post("/api/chatbot/message", json={"user_message": "Conditions XAUUSD H1 ?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["content"] == "Tendance haussière observée."
    assert body["blocked_reason"] is None
    assert body["tool_calls_made"] == []


def test_adversarial_blocked_returns_200_with_reason(tmp_path: Any) -> None:
    bot, client_stub = _make_real_chatbot([])  # no LLM response — must not be called
    client = _client_with_chatbot(tmp_path, bot)
    resp = client.post("/api/chatbot/message", json={"user_message": "Dois-je acheter EURUSD ?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["blocked_reason"] == "trade_request"
    assert body["content"] == REFUSAL_TEMPLATE
    assert client_stub.calls == []  # LLM never invoked


def test_503_when_chatbot_not_configured(tmp_path: Any) -> None:
    client = _client_with_chatbot(tmp_path, None)
    resp = client.post("/api/chatbot/message", json={"user_message": "Bonjour"})
    assert resp.status_code == 503


def test_422_empty_message(tmp_path: Any) -> None:
    bot, _ = _make_real_chatbot([])
    client = _client_with_chatbot(tmp_path, bot)
    resp = client.post("/api/chatbot/message", json={"user_message": ""})
    assert resp.status_code == 422


def test_422_message_too_long(tmp_path: Any) -> None:
    bot, _ = _make_real_chatbot([])
    client = _client_with_chatbot(tmp_path, bot)
    resp = client.post("/api/chatbot/message", json={"user_message": "x" * 2001})
    assert resp.status_code == 422


def test_422_history_too_long(tmp_path: Any) -> None:
    bot, _ = _make_real_chatbot([])
    client = _client_with_chatbot(tmp_path, bot)
    history = [{"role": "user", "content": "q"} for _ in range(21)]
    resp = client.post(
        "/api/chatbot/message",
        json={"user_message": "Bonjour", "conversation_history": history},
    )
    assert resp.status_code == 422


def test_500_internal_error_does_not_leak(tmp_path: Any) -> None:
    client = _client_with_chatbot(tmp_path, _RaisingChatbot())
    resp = client.post("/api/chatbot/message", json={"user_message": "Bonjour"})
    assert resp.status_code == 500
    assert resp.json()["detail"] == "Internal chatbot error"
    assert "secret" not in resp.text


def test_conversation_history_transmitted(tmp_path: Any) -> None:
    bot, client_stub = _make_real_chatbot([_Resp([_TextBlock("ok")], "end_turn")])
    client = _client_with_chatbot(tmp_path, bot)
    history = [
        {"role": "user", "content": "Première question"},
        {"role": "assistant", "content": "Première réponse"},
    ]
    resp = client.post(
        "/api/chatbot/message",
        json={"user_message": "Deuxième question", "conversation_history": history},
    )
    assert resp.status_code == 200
    sent = client_stub.calls[0]["messages"]
    assert sent[0] == {"role": "user", "content": "Première question"}
    assert sent[1] == {"role": "assistant", "content": "Première réponse"}
    assert sent[-1] == {"role": "user", "content": "Deuxième question"}


# --------------------------------------------------------------------------- #
# Bootstrap factory
# --------------------------------------------------------------------------- #


def test_build_chatbot_with_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    # Bypass the real Anthropic SDK construction (the `anthropic` package may not
    # be installed in dev/CI). This tests build_chatbot's own wiring: it injects
    # the client + assembler + filters + summary provider into a Chatbot.
    monkeypatch.setattr("src.api.bootstrap._build_anthropic_client", lambda: object())
    bot = build_chatbot(_Assembler())
    assert isinstance(bot, Chatbot)


def test_build_chatbot_without_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BootstrapConfigurationError):
        build_chatbot(_Assembler())


def test_build_chatbot_without_assembler_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(BootstrapConfigurationError):
        build_chatbot(None)


def test_maybe_bootstrap_chatbot_noop_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.api.app import _maybe_bootstrap_chatbot

    monkeypatch.delenv("CHATBOT_ENABLED", raising=False)
    app_state = AppState(signal_store=object())  # type: ignore[arg-type]
    _maybe_bootstrap_chatbot(app_state)
    assert app_state.chatbot is None
