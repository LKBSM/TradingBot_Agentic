"""SC-2 — POST /api/scanner/translate endpoint + bootstrap guard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.signal_store import SignalStore
from src.intelligence.scanner_translator import ScannerTranslator, TOOL_NAME


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


def _tool_resp(payload: dict) -> _Resp:
    return _Resp([_ToolBlock(name=TOOL_NAME, input=payload)])


def _client_with_translator(tmp_path: Any, translator: Any) -> TestClient:
    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    app = create_app(signal_store=store)
    app.state.app_state.scanner_translator = translator
    return TestClient(app)


class _RaisingTranslator:
    def translate(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("secret internal detail xyz")


def test_happy_path_translated(tmp_path: Any) -> None:
    client_stub = _Client([_tool_resp({
        "conditions": [{"type": "trend_is", "trend": "bullish"}, {"type": "zone_untested"}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })])
    tr = ScannerTranslator(anthropic_client=client_stub)
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": "OB jamais testé en tendance haussière"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "translated"
    assert {c["type"] for c in body["conditions"]} == {"trend_is", "zone_untested"}


def test_refusal_returns_refused_without_search(tmp_path: Any) -> None:
    client_stub = _Client([])  # LLM must never be called on a flagrant ask
    tr = ScannerTranslator(anthropic_client=client_stub)
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": "les meilleurs setups du moment"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "refused"
    assert body["refusal"]["kind"] == "ranking"
    assert body["conditions"] == []
    assert client_stub.calls == []


def test_out_of_palette_model_output_is_rejected_by_endpoint(tmp_path: Any) -> None:
    # The model emits an out-of-palette type; the endpoint must strip it so it can
    # never reach the evaluator, and name it as unsupported.
    client_stub = _Client([_tool_resp({
        "conditions": [{"type": "rsi_oversold"}, {"type": "price_in_fvg", "direction": "bullish"}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })])
    tr = ScannerTranslator(anthropic_client=client_stub)
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": "FVG haussier quand RSI en survente"})
    assert resp.status_code == 200
    body = resp.json()
    assert [c["type"] for c in body["conditions"]] == ["price_in_fvg"]
    assert any(u["category"] == "unsupported" for u in body["untranslatable"])


def test_bad_value_is_not_coerced(tmp_path: Any) -> None:
    client_stub = _Client([_tool_resp({
        "conditions": [{"type": "price_near_ob", "proximity_pct": 0.37}],
        "assumptions": [], "untranslatable": [], "refusal": None,
    })])
    tr = ScannerTranslator(anthropic_client=client_stub)
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": "prix à 0,37% d'un OB"})
    body = resp.json()
    # The condition is dropped (never snapped to 0.25).
    assert body["conditions"] == []


def test_503_when_translator_not_configured(tmp_path: Any) -> None:
    client = _client_with_translator(tmp_path, None)
    resp = client.post("/api/scanner/translate", json={"text": "bonjour"})
    assert resp.status_code == 503


def test_422_empty_text(tmp_path: Any) -> None:
    tr = ScannerTranslator(anthropic_client=_Client([]))
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": ""})
    assert resp.status_code == 422


def test_422_text_too_long(tmp_path: Any) -> None:
    tr = ScannerTranslator(anthropic_client=_Client([]))
    client = _client_with_translator(tmp_path, tr)
    resp = client.post("/api/scanner/translate", json={"text": "x" * 501})
    assert resp.status_code == 422


def test_500_internal_error_does_not_leak(tmp_path: Any) -> None:
    client = _client_with_translator(tmp_path, _RaisingTranslator())
    resp = client.post("/api/scanner/translate", json={"text": "bonjour"})
    assert resp.status_code == 500
    assert "secret" not in resp.text
