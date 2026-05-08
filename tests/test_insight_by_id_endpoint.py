"""Tests for the API-2B.3 GET /api/v1/insights/{insight_id} endpoint."""

from __future__ import annotations

import json
import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.audit import HashChainLedger


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with (
        patch("src.api.auth.TESTING_MODE", True),
        patch("src.api.routes.insight_history.TESTING_MODE", True),
    ):
        yield


@pytest.fixture
def populated_ledger():
    led = HashChainLedger()
    led.append({"id": "insight-alpha", "value": 1, "narrative_short": "first"})
    led.append({"id": "insight-beta", "value": 2})
    # Same id twice — broker resubmitted with same client_request_id
    led.append({"id": "insight-alpha", "value": 99, "narrative_short": "latest"})
    return led


@pytest.fixture
def client(populated_ledger):
    return TestClient(create_app(audit_ledger=populated_ledger))


@pytest.fixture
def client_no_ledger():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_get_returns_canonical_body(client, populated_ledger):
    resp = client.get("/api/v1/insights/insight-beta")
    assert resp.status_code == 200
    body = resp.json()
    assert body["insight"]["id"] == "insight-beta"
    assert body["insight"]["value"] == 2
    assert body["audit"]["seq"] == 2
    assert body["audit"]["entry_hash"] == populated_ledger.get(2).entry_hash
    assert body["is_latest_of"] == 1


def test_get_returns_latest_when_multiple_entries(client):
    """Two entries with id=insight-alpha — return the latest (seq=3)."""
    resp = client.get("/api/v1/insights/insight-alpha")
    body = resp.json()
    assert body["insight"]["narrative_short"] == "latest"
    assert body["audit"]["seq"] == 3
    assert body["is_latest_of"] == 2


def test_response_has_strong_etag_header(client, populated_ledger):
    resp = client.get("/api/v1/insights/insight-beta")
    etag = resp.headers["etag"]
    # Strong (no W/) and 64+2 chars
    assert not etag.startswith("W/")
    assert etag == f'"{populated_ledger.get(2).entry_hash}"'


def test_cache_control_header_set(client):
    resp = client.get("/api/v1/insights/insight-beta")
    assert "private" in resp.headers["cache-control"]
    assert "max-age=300" in resp.headers["cache-control"]


def test_x_ledger_seq_header(client):
    resp = client.get("/api/v1/insights/insight-alpha")
    # latest seq is 3
    assert resp.headers["x-ledger-seq"] == "3"


# ---------------------------------------------------------------------------
# Conditional GET (304)
# ---------------------------------------------------------------------------


def test_if_none_match_returns_304(client):
    r1 = client.get("/api/v1/insights/insight-beta")
    etag = r1.headers["etag"]
    r2 = client.get(
        "/api/v1/insights/insight-beta", headers={"If-None-Match": etag}
    )
    assert r2.status_code == 304
    assert r2.content == b""
    assert r2.headers["etag"] == etag


def test_if_none_match_with_stale_etag_returns_200(client):
    resp = client.get(
        "/api/v1/insights/insight-beta",
        headers={"If-None-Match": '"deadbeefdeadbeef"'},
    )
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_404_when_insight_id_unknown(client):
    resp = client.get("/api/v1/insights/does-not-exist")
    assert resp.status_code == 404


def test_400_when_insight_id_too_long(client):
    too_long = "a" * 65
    resp = client.get(f"/api/v1/insights/{too_long}")
    assert resp.status_code == 400


def test_503_when_ledger_unconfigured(client_no_ledger):
    resp = client_no_ledger.get("/api/v1/insights/anything")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# Tier gate (lookup is STRATEGIST+ like history)
# ---------------------------------------------------------------------------


def test_history_endpoint_still_works(client):
    """Sanity: the /history endpoint and /{insight_id} coexist; specific
    /history string isn't shadowed by the path param."""
    resp = client.get("/api/v1/insights/history")
    assert resp.status_code == 200
    body = resp.json()
    assert "entries" in body


def test_tier_gate_blocks_free_outside_testing(populated_ledger):
    with (
        patch("src.api.auth.TESTING_MODE", False),
        patch("src.api.routes.insight_history.TESTING_MODE", False),
    ):
        class _FakeStore:
            def verify_key(self, _):
                return {"key_id": 1, "label": "free"}

            def check_rate_limit(self, _):
                return True

            def record_usage(self, *a):
                pass

        c = TestClient(
            create_app(audit_ledger=populated_ledger, key_store=_FakeStore())
        )
        resp = c.get(
            "/api/v1/insights/insight-beta",
            headers={"X-API-Key": "sk_dev_free"},
        )
        assert resp.status_code == 403
