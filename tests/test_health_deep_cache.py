"""Tests for the INFRA-2B.12 /health/deep response cache."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.routes import health_deep
from src.audit import HashChainLedger


@pytest.fixture
def client():
    app = create_app(audit_ledger=HashChainLedger())
    return TestClient(app)


def test_first_call_is_a_miss(client):
    resp = client.get("/api/v1/health/deep")
    assert resp.headers["x-health-cache"] == "miss"
    assert resp.status_code == 200


def test_second_call_is_a_hit(client):
    client.get("/api/v1/health/deep")
    resp2 = client.get("/api/v1/health/deep")
    assert resp2.headers["x-health-cache"] == "hit"


def test_cache_body_is_identical_across_hit(client):
    r1 = client.get("/api/v1/health/deep")
    r2 = client.get("/api/v1/health/deep")
    assert r1.json() == r2.json()


def test_cache_reports_remaining_ttl(client):
    client.get("/api/v1/health/deep")
    r = client.get("/api/v1/health/deep")
    remaining = float(r.headers["x-health-cache-expires-in"])
    # < TTL (30s) and > 0
    assert 0 < remaining <= 30


def test_cache_expires_after_ttl(client, monkeypatch):
    # Make _now() jump forward past the TTL.
    base = [0.0]

    def fake_now():
        return base[0]

    monkeypatch.setattr(health_deep, "_now", fake_now)

    base[0] = 100.0
    r1 = client.get("/api/v1/health/deep")
    assert r1.headers["x-health-cache"] == "miss"
    base[0] = 110.0  # +10s — still cached
    r2 = client.get("/api/v1/health/deep")
    assert r2.headers["x-health-cache"] == "hit"
    base[0] = 200.0  # +100s — expired
    r3 = client.get("/api/v1/health/deep")
    assert r3.headers["x-health-cache"] == "miss"


def test_cache_preserves_503_status_on_unhealthy(monkeypatch):
    """When the probe returns unhealthy, the cached response keeps the
    503 status — we don't accidentally upgrade to 200 on hits."""
    def fake_build(app_state):
        return {"ok": False, "checks": {"x": {"configured": True, "ok": False}}}, 503

    monkeypatch.setattr(health_deep, "_build_body", fake_build)
    c = TestClient(create_app())
    r1 = c.get("/api/v1/health/deep")
    assert r1.status_code == 503
    r2 = c.get("/api/v1/health/deep")
    assert r2.status_code == 503
    assert r2.headers["x-health-cache"] == "hit"
