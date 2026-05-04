"""Tests for the DATA-2B.5 audit /verify + /entry endpoints."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.audit import HashChainLedger


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with patch("src.api.auth.TESTING_MODE", True):
        yield


@pytest.fixture
def populated_ledger():
    led = HashChainLedger()
    for i in range(1, 6):
        led.append({"id": f"insight-{i}", "value": i})
    return led


@pytest.fixture
def client(populated_ledger):
    return TestClient(create_app(audit_ledger=populated_ledger))


@pytest.fixture
def client_no_ledger():
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# /verify
# ---------------------------------------------------------------------------


def test_verify_intact_chain_returns_ok(client, populated_ledger):
    resp = client.get("/api/v1/audit/verify")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["n_entries"] == 5
    assert body["head_hash"] == populated_ledger.head_hash
    assert body["broken_at_seq"] is None


def test_verify_503_when_no_ledger(client_no_ledger):
    resp = client_no_ledger.get("/api/v1/audit/verify")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /entry/{seq}
# ---------------------------------------------------------------------------


def test_entry_returns_full_record(client, populated_ledger):
    resp = client.get("/api/v1/audit/entry/3")
    assert resp.status_code == 200
    body = resp.json()
    assert body["seq"] == 3
    assert body["insight_id"] == "insight-3"
    assert len(body["entry_hash"]) == 64
    # The body's prev_hash matches the previous entry's entry_hash
    prev = populated_ledger.get(2)
    assert body["prev_hash"] == prev.entry_hash


def test_entry_404_on_missing_seq(client):
    resp = client.get("/api/v1/audit/entry/9999")
    assert resp.status_code == 404


def test_entry_400_on_invalid_seq(client):
    resp = client.get("/api/v1/audit/entry/0")
    assert resp.status_code == 400


def test_entry_503_when_no_ledger(client_no_ledger):
    resp = client_no_ledger.get("/api/v1/audit/entry/1")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /by-insight/{insight_id}
# ---------------------------------------------------------------------------


def test_by_insight_returns_match(client):
    resp = client.get("/api/v1/audit/by-insight/insight-2")
    assert resp.status_code == 200
    body = resp.json()
    assert body["insight_id"] == "insight-2"
    assert body["n_entries"] == 1
    assert body["entries"][0]["seq"] == 2


def test_by_insight_404_on_unknown_id(client):
    resp = client.get("/api/v1/audit/by-insight/unknown-xyz")
    assert resp.status_code == 404


def test_by_insight_400_on_oversized_id(client):
    resp = client.get("/api/v1/audit/by-insight/" + ("a" * 200))
    assert resp.status_code == 400


def test_by_insight_returns_all_dups(populated_ledger):
    populated_ledger.append({"id": "insight-1", "value": "second-occurrence"})
    app = create_app(audit_ledger=populated_ledger)
    client = TestClient(app)
    resp = client.get("/api/v1/audit/by-insight/insight-1")
    assert resp.status_code == 200
    assert resp.json()["n_entries"] == 2
