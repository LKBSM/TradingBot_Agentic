"""Integration tests for the OBS-2B.4 /api/v1/metrics/latency endpoint."""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import os
import time

os.environ.setdefault("SENTINEL_TESTING_MODE", "0")

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.latency_tracker import LatencyTracker


class _FakeHMAC:
    def __init__(self, secret: bytes = b"sentinel-obs-test"):
        self._secret = secret

    def verify(self, data: bytes, signature: str, key_version=None) -> bool:
        expected = hmac_mod.new(self._secret, data, hashlib.sha256).hexdigest()
        return hmac_mod.compare_digest(expected, signature)

    def sign_now(self) -> dict:
        ts = str(time.time())
        sig = hmac_mod.new(self._secret, ts.encode(), hashlib.sha256).hexdigest()
        return {"X-Admin-Signature": sig, "X-Admin-Timestamp": ts}


@pytest.fixture
def tracker():
    t = LatencyTracker()
    # Seed two routes with distinct shapes so percentile assertions
    # have something to verify.
    for ms in (5, 10, 15, 20, 25):
        t.record("/api/v1/enrich", float(ms), 200)
    t.record("/api/v1/insights/{insight_id}", 100.0, 200)
    t.record("/api/v1/insights/{insight_id}", 200.0, 500)
    return t


@pytest.fixture
def hmac_mgr():
    return _FakeHMAC()


@pytest.fixture
def client(tracker, hmac_mgr):
    app = create_app(latency_tracker=tracker, hmac_manager=hmac_mgr)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


def test_requires_admin_hmac(client):
    resp = client.get("/api/v1/metrics/latency")
    assert resp.status_code == 401


def test_rejects_bad_signature(client):
    resp = client.get(
        "/api/v1/metrics/latency",
        headers={
            "X-Admin-Signature": "deadbeef",
            "X-Admin-Timestamp": str(time.time()),
        },
    )
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_per_route_stats(client, hmac_mgr):
    resp = client.get("/api/v1/metrics/latency", headers=hmac_mgr.sign_now())
    assert resp.status_code == 200
    body = resp.json()
    assert body["window_seconds"] > 0

    paths = {r["path"]: r for r in body["routes"]}
    # The endpoint itself was just hit so the access-log middleware
    # added /api/v1/metrics/latency — that means at least our two
    # seeded routes must be present alongside the live one.
    assert "/api/v1/enrich" in paths
    assert "/api/v1/insights/{insight_id}" in paths

    enrich = paths["/api/v1/enrich"]
    assert enrich["count"] == 5
    assert enrich["max_ms"] == 25.0
    assert enrich["error_rate"] == 0.0

    insights = paths["/api/v1/insights/{insight_id}"]
    assert insights["count"] == 2
    assert insights["error_count"] == 1
    assert insights["error_rate"] == 0.5


def test_totals_aggregate_window_and_lifetime(client, hmac_mgr):
    resp = client.get("/api/v1/metrics/latency", headers=hmac_mgr.sign_now())
    body = resp.json()
    totals = body["totals"]
    # Seeded: 5 enrich + 2 insights = 7. Plus this very request bumps
    # /api/v1/metrics/latency by 1.
    assert totals["count_window"] >= 7
    assert totals["count_lifetime"] >= 7
    assert totals["error_count_window"] >= 1


# ---------------------------------------------------------------------------
# Wiring — access log middleware feeds the tracker on every request
# ---------------------------------------------------------------------------


def test_middleware_records_into_tracker(tracker, hmac_mgr):
    """Hitting the endpoint must show up in subsequent snapshots."""
    app = create_app(latency_tracker=tracker, hmac_manager=hmac_mgr)
    c = TestClient(app)
    # Initial: this route hasn't been observed yet.
    assert tracker.snapshot("/api/v1/metrics/latency").count == 0
    c.get("/api/v1/metrics/latency", headers=hmac_mgr.sign_now())
    c.get("/api/v1/metrics/latency", headers=hmac_mgr.sign_now())
    snap = tracker.snapshot("/api/v1/metrics/latency")
    assert snap.count == 2


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_503_when_tracker_not_wired(hmac_mgr):
    """create_app auto-installs a tracker, so we have to explicitly
    null it out to exercise the 503 path."""
    app = create_app(hmac_manager=hmac_mgr)
    app.state.app_state.latency_tracker = None
    c = TestClient(app)
    resp = c.get("/api/v1/metrics/latency", headers=hmac_mgr.sign_now())
    assert resp.status_code == 503
