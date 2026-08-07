"""SC-2 — LIVE degradation at 0 API credits (no network, no spend).

We deliberately run at 0 Anthropic credits for now. This suite proves the LIVE
path behaves safely in that state, exercising the REAL route + REAL
ScannerTranslator + REAL CircuitBreaker (built exactly as bootstrap builds them),
with an Anthropic client that raises the way a 0-credit account does on
``messages.create``:

  · a normal phrase → 200 with outcome "error" (fail-safe) — never a 500, the
    text field stays usable;
  · a ranking/prediction/advice ask → 200 "refused" WITHOUT any API call — the
    refusal path works fully at 0 credits;
  · after the failure threshold, the CircuitBreaker OPENS and stops calling the
    (exhausted) API — protecting the quota instead of hammering it.

When credits are added, the SAME wiring returns real translations (the request
shape — model, forced tool_choice — is pinned in test_scanner_translator.py).
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.signal_store import SignalStore
from src.intelligence.circuit_breaker import CircuitBreaker, CircuitState
from src.intelligence.scanner_translator import ScannerTranslator


class _QuotaError(Exception):
    """Mimics the anthropic error raised by a 0-credit account."""

    def __init__(self) -> None:
        super().__init__("Your credit balance is too low to access the Anthropic API.")


class _Msgs:
    def __init__(self, parent: "_ZeroCreditClient") -> None:
        self._p = parent

    def create(self, **kwargs: Any) -> Any:
        self._p.calls.append(kwargs)
        raise _QuotaError()


class _ZeroCreditClient:
    """Every messages.create raises, as at 0 credits — and counts the attempts."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.messages = _Msgs(self)


def _translator_like_bootstrap() -> tuple[ScannerTranslator, _ZeroCreditClient, CircuitBreaker]:
    # Same construction as src/api/bootstrap.build_scanner_translator.
    client = _ZeroCreditClient()
    breaker = CircuitBreaker(name="scanner_translator", failure_threshold=3, recovery_timeout=60.0)
    return ScannerTranslator(anthropic_client=client, breaker=breaker), client, breaker


def _client_with(tmp_path: Any, translator: Any) -> TestClient:
    store = SignalStore(db_path=str(tmp_path / "signals.db"))
    app = create_app(signal_store=store)
    app.state.app_state.scanner_translator = translator
    return TestClient(app)


def test_normal_phrase_degrades_to_error_not_500(tmp_path: Any) -> None:
    translator, client, _ = _translator_like_bootstrap()
    api = _client_with(tmp_path, translator)
    resp = api.post("/api/scanner/translate", json={"text": "un Order Block jamais testé en tendance haussière"})
    assert resp.status_code == 200  # never a 500 — the textbox stays usable
    body = resp.json()
    assert body["outcome"] == "error"
    assert body["conditions"] == []
    assert len(client.calls) == 1  # the API was attempted once


def test_refusal_works_at_zero_credits_without_api_call(tmp_path: Any) -> None:
    translator, client, _ = _translator_like_bootstrap()
    api = _client_with(tmp_path, translator)
    resp = api.post("/api/scanner/translate", json={"text": "montre-moi les meilleurs marchés"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["outcome"] == "refused"
    assert body["refusal"]["kind"] == "ranking"
    assert client.calls == []  # deterministic pre-filter — no API needed at 0 credits


def test_empty_and_validation_paths_need_no_api(tmp_path: Any) -> None:
    translator, client, _ = _translator_like_bootstrap()
    api = _client_with(tmp_path, translator)
    # 422 on empty (never reaches the translator).
    assert api.post("/api/scanner/translate", json={"text": ""}).status_code == 422
    assert client.calls == []


def test_circuit_breaker_opens_and_stops_hammering_the_quota(tmp_path: Any) -> None:
    translator, client, breaker = _translator_like_bootstrap()
    api = _client_with(tmp_path, translator)
    phrase = {"text": "un FVG haussier avec le 1 h d'accord"}

    # Threshold = 3: the first three attempts hit the API and fail.
    for _ in range(3):
        resp = api.post("/api/scanner/translate", json=phrase)
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "error"
    assert len(client.calls) == 3
    assert breaker.state == CircuitState.OPEN

    # Now OPEN: further attempts short-circuit — still a safe 200 "error", but the
    # exhausted API is NOT called again (quota protected).
    for _ in range(5):
        resp = api.post("/api/scanner/translate", json=phrase)
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "error"
    assert len(client.calls) == 3  # unchanged — breaker absorbed the extra load


def test_bootstrap_fails_fast_without_api_key(monkeypatch: Any) -> None:
    # Config safety: a missing key is a clear boot error, never a half-wired
    # translator that 500s at request time.
    from src.api.bootstrap import BootstrapConfigurationError, build_scanner_translator

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    raised = False
    try:
        build_scanner_translator()
    except BootstrapConfigurationError:
        raised = True
    assert raised
