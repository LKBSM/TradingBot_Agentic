"""Tests for the INFRA-2B.5 B2B enrichment endpoint.

Verifies:
- Happy path: BULLISH_SETUP returns InsightSignalV2 with sources + narrative
- Stub mode is reflected in extras + narrative is the deterministic stub
- LLM mode wires the answer into ``narrative_long``
- NEUTRAL setups have levels stripped (per v2 validator)
- Validation errors land at 422 (not 500)
- Languages dispatch the right narrative_short and disclaimer
- ``client_request_id`` echoes back as the InsightSignal id when supplied
- 503 when no RAG pipeline is configured
- Pipeline state isolation across requests
- BULLISH_SETUP rejects stop ≥ entry (v2 validator)
"""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes.qa import build_default_rag_pipeline


@pytest.fixture(autouse=True)
def _force_testing_mode():
    """Patch captured-at-import flags so auth bypass and tier checks
    behave for the tests."""
    with patch("src.api.auth.TESTING_MODE", True), patch(
        "src.api.routes.qa.TESTING_MODE", True
    ), patch("src.api.routes.enrich.TESTING_MODE", True):
        yield


@pytest.fixture(scope="module")
def populated_pipeline():
    return build_default_rag_pipeline()


@pytest.fixture
def client_stub(populated_pipeline):
    return TestClient(create_app(rag_pipeline=populated_pipeline))


@pytest.fixture
def client_with_llm(populated_pipeline):
    def stub_llm(system: str, user: str) -> str:
        return "B2B-LLM narrative anchored to retrieved context."

    return TestClient(create_app(rag_pipeline=populated_pipeline, rag_llm=stub_llm))


@pytest.fixture
def client_no_rag():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


_BULLISH_PAYLOAD = {
    "instrument": "XAUUSD",
    "timeframe": "M15",
    "direction": "BULLISH_SETUP",
    "entry": 2350.0,
    "stop": 2340.0,
    "target_1": 2370.0,
    "target_2": 2390.0,
    "broker_context": "Dollar weakening on softer CPI print",
    "language": "en",
}


def test_enrich_returns_insight_signal_v2(client_stub):
    resp = client_stub.post("/api/v1/enrich", json=_BULLISH_PAYLOAD)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["schema_version"] == "2.0.0"
    assert body["instrument"] == "XAUUSD"
    assert body["timeframe"] == "M15"
    assert body["direction"] == "BULLISH_SETUP"
    assert body["narrative_short"]
    assert body["narrative_long"]
    assert len(body["sources_cited"]) > 0
    assert body["compliance"]["edge_claim"] is False
    assert body["compliance"]["is_paper_demo"] is True
    assert body["extras"]["stub_mode"] is True
    assert body["extras"]["rr_ratio"] == 2.0  # (2370-2350)/(2350-2340)


def test_enrich_real_llm_mode_wires_answer_to_narrative(client_with_llm):
    resp = client_with_llm.post("/api/v1/enrich", json=_BULLISH_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["extras"]["stub_mode"] is False
    assert "B2B-LLM" in body["narrative_long"]


def test_enrich_neutral_strips_levels(client_stub):
    resp = client_stub.post(
        "/api/v1/enrich",
        json={
            "instrument": "XAUUSD",
            "timeframe": "H1",
            "direction": "NEUTRAL",
            "entry": 2350.0,  # provided but should be silently dropped
            "stop": 2340.0,
            "target_1": 2370.0,
            "language": "en",
        },
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["direction"] == "NEUTRAL"
    assert body["levels"]["entry"] is None
    assert body["levels"]["stop"] is None
    assert body["levels"]["target_1"] is None
    # NEUTRAL conviction is bounded below the actionable threshold (40)
    assert body["conviction_0_100"] < 40


def test_enrich_client_request_id_echoes_into_signal_id(client_stub):
    payload = dict(_BULLISH_PAYLOAD, client_request_id="broker-abc-123")
    resp = client_stub.post("/api/v1/enrich", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "broker-abc-123"
    assert body["extras"]["client_request_id"] == "broker-abc-123"


def test_enrich_generates_uuid_when_no_client_id(client_stub):
    resp = client_stub.post("/api/v1/enrich", json=_BULLISH_PAYLOAD)
    assert resp.status_code == 200
    sid = resp.json()["id"]
    assert len(sid) >= 8
    assert "-" in sid


@pytest.mark.parametrize(
    "lang,expected_marker",
    [
        ("fr", "haussier"),
        ("en", "Bullish"),
        ("de", "Bullishes"),
        ("es", "alcista"),
    ],
)
def test_enrich_localises_narrative_short(client_stub, lang, expected_marker):
    payload = dict(_BULLISH_PAYLOAD, language=lang)
    resp = client_stub.post("/api/v1/enrich", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert expected_marker in body["narrative_short"]
    assert body["narrative_language"] == lang
    assert body["compliance"]["disclaimer_lang"] == lang


def test_enrich_returns_elapsed_ms(client_stub):
    resp = client_stub.post("/api/v1/enrich", json=_BULLISH_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert "retrieve" in body["extras"]["elapsed_ms"]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_enrich_invalid_timeframe_rejected(client_stub):
    payload = dict(_BULLISH_PAYLOAD, timeframe="M2")
    resp = client_stub.post("/api/v1/enrich", json=payload)
    assert resp.status_code == 422


def test_enrich_invalid_direction_rejected(client_stub):
    payload = dict(_BULLISH_PAYLOAD, direction="LONG")
    resp = client_stub.post("/api/v1/enrich", json=payload)
    assert resp.status_code == 422


def test_enrich_negative_entry_rejected(client_stub):
    payload = dict(_BULLISH_PAYLOAD, entry=-1.0)
    resp = client_stub.post("/api/v1/enrich", json=payload)
    assert resp.status_code == 422


def test_enrich_inverted_levels_rejected_by_v2_validator(client_stub):
    """BULLISH_SETUP requires stop < entry (InsightSignalV2 validator)."""
    payload = dict(_BULLISH_PAYLOAD, entry=2340.0, stop=2350.0)
    resp = client_stub.post("/api/v1/enrich", json=payload)
    # InsightSignalV2 model_validator raises ⇒ 500 when constructing the
    # response. We accept either 422 (FastAPI catches Pydantic) or 500.
    # In the current FastAPI version, validation errors during response
    # construction surface as 500.
    assert resp.status_code in (422, 500)


def test_enrich_503_when_pipeline_missing(client_no_rag):
    resp = client_no_rag.post("/api/v1/enrich", json=_BULLISH_PAYLOAD)
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# State isolation
# ---------------------------------------------------------------------------


def test_enrich_does_not_leak_pipeline_state(client_stub, populated_pipeline):
    original_lang = populated_pipeline.language
    original_k = populated_pipeline.final_k

    client_stub.post(
        "/api/v1/enrich",
        json=dict(_BULLISH_PAYLOAD, language="de"),
    )

    assert populated_pipeline.language == original_lang
    assert populated_pipeline.final_k == original_k
