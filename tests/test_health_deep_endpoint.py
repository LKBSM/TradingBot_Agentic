"""Tests for the OBS-2B.2 deep health probe."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.audit import HashChainLedger
from src.delivery.webhook_queue import WebhookDeliveryQueue
from src.intelligence.rag.cost_quota import CostQuotaEnforcer
from src.intelligence.rag.tier_rate_limiter import TierRateLimiter


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeRagPipeline:
    """Just enough of RAGPipeline for the deep probe."""

    def __init__(self, *, retrieve_count: int = 3, raises: bool = False):
        self.retrieve_count = retrieve_count
        self.raises = raises

    def retrieve(self, query: str):
        if self.raises:
            raise RuntimeError("index unavailable")
        return list(range(self.retrieve_count))


def _transport_ok(url, headers, body):
    return 200


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with patch("src.api.auth.TESTING_MODE", True):
        yield


@pytest.fixture
def empty_client():
    """No subsystems wired — endpoint should still return 200."""
    return TestClient(create_app())


@pytest.fixture
def healthy_client():
    led = HashChainLedger()
    led.append({"id": "i-1", "value": 1})
    return TestClient(
        create_app(
            audit_ledger=led,
            rag_pipeline=_FakeRagPipeline(retrieve_count=5),
            cost_quota=CostQuotaEnforcer(),
            webhook_queue=WebhookDeliveryQueue(transport=_transport_ok),
            tier_rate_limiter=TierRateLimiter(),
        )
    )


# ---------------------------------------------------------------------------
# Empty deployment
# ---------------------------------------------------------------------------


def test_deep_health_empty_deployment_is_ok(empty_client):
    """An app with no Phase 2B subsystems wired must still return 200 — the
    probe distinguishes 'unconfigured' from 'failed'."""
    resp = empty_client.get("/api/v1/health/deep")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    for name, check in body["checks"].items():
        assert check["ok"] is True
        assert check["configured"] is False


# ---------------------------------------------------------------------------
# Healthy deployment
# ---------------------------------------------------------------------------


def test_deep_health_full_deployment_returns_200(healthy_client):
    resp = healthy_client.get("/api/v1/health/deep")
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["checks"]["audit_ledger"]["configured"] is True
    assert body["checks"]["audit_ledger"]["n_entries"] == 1
    assert body["checks"]["rag_pipeline"]["retrieved_count"] == 5
    assert body["checks"]["cost_quota"]["any_exhausted"] is False
    assert body["checks"]["webhook_queue"]["pending"] == 0
    assert body["checks"]["webhook_queue"]["dead_letter"] == 0
    assert body["checks"]["tier_rate_limiter"]["caps_per_minute"]["FREE"] >= 1


def test_deep_health_reports_total_duration(healthy_client):
    resp = healthy_client.get("/api/v1/health/deep")
    body = resp.json()
    assert body["duration_ms"] >= 0


def test_deep_health_iso_timestamp_z_suffix(healthy_client):
    resp = healthy_client.get("/api/v1/health/deep")
    ts = resp.json()["checked_at_utc"]
    assert ts.endswith("Z")


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_deep_health_503_when_audit_chain_broken():
    """Tamper directly in SQLite so verify() fails — must return 503 with
    audit_ledger.ok=False but not crash."""
    led = HashChainLedger()
    led.append({"id": "i-1", "value": 1})
    led.append({"id": "i-2", "value": 2})
    # Corrupt the canonical_json of seq 2
    led._writer_conn.execute(
        "UPDATE ledger SET canonical_json='{\"id\":\"tampered\"}' WHERE seq=2"
    )

    c = TestClient(create_app(audit_ledger=led))
    resp = c.get("/api/v1/health/deep")
    assert resp.status_code == 503
    body = resp.json()
    assert body["ok"] is False
    assert body["checks"]["audit_ledger"]["ok"] is False
    assert body["checks"]["audit_ledger"]["broken_at_seq"] == 2


def test_deep_health_503_when_rag_pipeline_raises():
    c = TestClient(create_app(rag_pipeline=_FakeRagPipeline(raises=True)))
    resp = c.get("/api/v1/health/deep")
    assert resp.status_code == 503
    assert resp.json()["checks"]["rag_pipeline"]["ok"] is False
    assert "RuntimeError" in resp.json()["checks"]["rag_pipeline"]["error"]


def test_deep_health_503_when_rag_pipeline_returns_empty():
    """An empty index implies BM25/vector store are unhealthy."""
    c = TestClient(create_app(rag_pipeline=_FakeRagPipeline(retrieve_count=0)))
    resp = c.get("/api/v1/health/deep")
    assert resp.status_code == 503
    assert resp.json()["checks"]["rag_pipeline"]["ok"] is False


def test_deep_health_503_when_cost_quota_exhausted():
    quota = CostQuotaEnforcer({"FREE": 0.05})
    # Push spend past cap
    quota.record("FREE", 0.10)
    c = TestClient(create_app(cost_quota=quota))
    resp = c.get("/api/v1/health/deep")
    assert resp.status_code == 503
    body = resp.json()
    assert body["checks"]["cost_quota"]["any_exhausted"] is True
    assert body["checks"]["cost_quota"]["tiers"]["FREE"]["utilisation"] >= 1.0


def test_deep_health_503_when_webhook_dead_letter_threshold():
    queue = WebhookDeliveryQueue(transport=lambda u, h, b: 404)
    # 11 permanent failures ⇒ 11 dead letters ⇒ above the noise threshold.
    for i in range(11):
        queue.enqueue(url="https://x", secret="s", body=f'{{"i":{i}}}')
    queue.drain()

    c = TestClient(create_app(webhook_queue=queue))
    resp = c.get("/api/v1/health/deep")
    assert resp.status_code == 503
    body = resp.json()
    assert body["checks"]["webhook_queue"]["dead_letter"] == 11
    assert body["checks"]["webhook_queue"]["ok"] is False


def test_deep_health_partial_failure_isolates_culprit():
    """One subsystem fails, others pass — body must show which one."""
    bad_rag = _FakeRagPipeline(raises=True)
    led = HashChainLedger()
    led.append({"id": "i-1", "value": 1})
    c = TestClient(create_app(audit_ledger=led, rag_pipeline=bad_rag))
    resp = c.get("/api/v1/health/deep")
    body = resp.json()
    assert body["ok"] is False
    assert body["checks"]["audit_ledger"]["ok"] is True
    assert body["checks"]["rag_pipeline"]["ok"] is False
