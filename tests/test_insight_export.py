"""Tests for the API-2B.6 NDJSON insight history export."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

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
def ledger():
    led = HashChainLedger()
    for i in range(1, 11):  # 10 entries
        led.append({"id": f"ins-{i:03d}", "v": i})
    return led


@pytest.fixture
def client(ledger):
    return TestClient(create_app(audit_ledger=ledger))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def _lines(resp):
    return [line for line in resp.text.split("\n") if line]


def test_export_returns_ndjson_with_one_row_per_line(client):
    resp = client.get("/api/v1/insights/export")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")
    rows = [json.loads(l) for l in _lines(resp)]
    assert len(rows) == 10
    # All rows contain the full body
    assert all("insight" in r and "entry_hash" in r for r in rows)
    # Default order is newest-first (descending seq)
    seqs = [r["seq"] for r in rows]
    assert seqs == sorted(seqs, reverse=True)


def test_export_includes_canonical_body(client):
    resp = client.get("/api/v1/insights/export")
    rows = [json.loads(l) for l in _lines(resp)]
    bodies = {r["insight_id"]: r["insight"] for r in rows}
    assert bodies["ins-005"]["v"] == 5


def test_export_sets_head_hash_header(client, ledger):
    resp = client.get("/api/v1/insights/export")
    assert resp.headers["x-ledger-head-hash"] == ledger.head_hash
    assert resp.headers["x-ledger-head-seq"] == "10"


def test_export_attachment_disposition(client):
    resp = client.get("/api/v1/insights/export")
    cd = resp.headers["content-disposition"]
    assert 'filename="insights_export.ndjson"' in cd


# ---------------------------------------------------------------------------
# Limit
# ---------------------------------------------------------------------------


def test_limit_caps_output(client):
    resp = client.get("/api/v1/insights/export?limit=3")
    rows = _lines(resp)
    assert len(rows) == 3


def test_limit_rejects_zero(client):
    resp = client.get("/api/v1/insights/export?limit=0")
    assert resp.status_code == 422


def test_limit_rejects_above_50000(client):
    resp = client.get("/api/v1/insights/export?limit=50001")
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def test_filter_by_insight_id(client):
    resp = client.get("/api/v1/insights/export?insight_id=ins-007")
    rows = [json.loads(l) for l in _lines(resp)]
    assert len(rows) == 1
    assert rows[0]["insight_id"] == "ins-007"


def test_filter_by_future_since_returns_empty(client):
    resp = client.get("/api/v1/insights/export?since=2099-01-01T00:00:00Z")
    rows = _lines(resp)
    assert rows == []


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_503_when_ledger_not_wired():
    c = TestClient(create_app())
    resp = c.get("/api/v1/insights/export")
    assert resp.status_code == 503
