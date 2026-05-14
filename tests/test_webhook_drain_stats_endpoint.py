"""Tests for the OBS-2B.6 webhook drain worker stats endpoint."""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import os
import time

os.environ.setdefault("SENTINEL_TESTING_MODE", "0")  # admin requires real HMAC

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.delivery.webhook_drain_worker import WebhookDrainWorker
from src.delivery.webhook_queue import WebhookDeliveryQueue


class _FakeHMAC:
    def __init__(self, secret: bytes = b"obs-test"):
        self._secret = secret

    def verify(self, data, signature, key_version=None):
        expected = hmac_mod.new(self._secret, data, hashlib.sha256).hexdigest()
        return hmac_mod.compare_digest(expected, signature)

    def headers(self):
        ts = str(time.time())
        sig = hmac_mod.new(self._secret, ts.encode(), hashlib.sha256).hexdigest()
        return {"X-Admin-Signature": sig, "X-Admin-Timestamp": ts}


def _ok_transport(*_):
    return 200


@pytest.fixture
def queue():
    return WebhookDeliveryQueue(transport=_ok_transport)


@pytest.fixture
def worker(queue):
    return WebhookDrainWorker(queue)


@pytest.fixture
def hmac_mgr():
    return _FakeHMAC()


@pytest.fixture
def client(worker, hmac_mgr):
    return TestClient(
        create_app(webhook_drain_worker=worker, hmac_manager=hmac_mgr)
    )


# ---------------------------------------------------------------------------
# Auth gate
# ---------------------------------------------------------------------------


def test_requires_admin_hmac(client):
    resp = client.get("/api/v1/metrics/webhook-drain")
    assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_returns_stats_shape(client, hmac_mgr):
    resp = client.get("/api/v1/metrics/webhook-drain", headers=hmac_mgr.headers())
    assert resp.status_code == 200
    body = resp.json()
    expected = {
        "running", "cycles_run", "successes", "retried", "dead_lettered",
        "skipped_not_due", "last_cycle_at", "pending", "dead_letter",
    }
    assert expected.issubset(body.keys())


def test_pending_count_reflects_queue(queue, client, hmac_mgr):
    queue.enqueue(url="https://b.example/hook", secret="s", body="{}")
    queue.enqueue(url="https://b.example/hook", secret="s", body="{}")
    body = client.get(
        "/api/v1/metrics/webhook-drain", headers=hmac_mgr.headers()
    ).json()
    assert body["pending"] == 2


def test_dead_letter_count_reflects_queue(queue, client, hmac_mgr):
    def fail(*_):
        return 400  # 4xx → straight to dead-letter

    q = WebhookDeliveryQueue(transport=fail)
    q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    q.drain()
    w = WebhookDrainWorker(q)
    c = TestClient(create_app(webhook_drain_worker=w, hmac_manager=_FakeHMAC()))
    body = c.get(
        "/api/v1/metrics/webhook-drain",
        headers=_FakeHMAC().headers(),
    ).json()
    assert body["dead_letter"] == 1


def test_running_false_before_start(client, hmac_mgr):
    body = client.get(
        "/api/v1/metrics/webhook-drain", headers=hmac_mgr.headers()
    ).json()
    assert body["running"] is False


# ---------------------------------------------------------------------------
# Misconfiguration
# ---------------------------------------------------------------------------


def test_503_when_worker_not_wired(hmac_mgr):
    app = create_app(hmac_manager=hmac_mgr)
    c = TestClient(app)
    resp = c.get("/api/v1/metrics/webhook-drain", headers=hmac_mgr.headers())
    assert resp.status_code == 503
