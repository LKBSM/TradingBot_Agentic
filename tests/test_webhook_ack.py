"""Tests for the API-2B.5 webhook ack endpoint + queue.cancel()."""

from __future__ import annotations

import os
from unittest.mock import patch

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.audit import AdminActionLog
from src.delivery.webhook_queue import WebhookDeliveryQueue


@pytest.fixture(autouse=True)
def _force_testing_mode():
    # conftest imports src.api.auth before our os.environ.setdefault
    # runs, so the module-level TESTING_MODE has already latched to
    # False in some test runs. Patch it for every test in this file.
    with patch("src.api.auth.TESTING_MODE", True):
        yield


# ---------------------------------------------------------------------------
# WebhookDeliveryQueue.cancel — unit
# ---------------------------------------------------------------------------


def _ok_transport(url, headers, body):
    return 200


def test_cancel_removes_pending_delivery():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    d = q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    state = q.cancel(d.delivery_id)
    assert state == "pending"
    assert q.pending_size == 0


def test_cancel_removes_dead_letter_delivery():
    def fail(*_):
        return 400  # 4xx → straight to dead-letter

    q = WebhookDeliveryQueue(transport=fail)
    d = q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    q.drain()
    assert q.dead_letter_size == 1

    state = q.cancel(d.delivery_id)
    assert state == "dead"
    assert q.dead_letter_size == 0


def test_cancel_unknown_id_returns_not_found():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    state = q.cancel("whk-99999999")
    assert state == "not_found"


def test_cancel_is_idempotent():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    d = q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    assert q.cancel(d.delivery_id) == "pending"
    assert q.cancel(d.delivery_id) == "not_found"  # no raise on second call


def test_cancel_rejects_empty_id():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    with pytest.raises(ValueError):
        q.cancel("")


def test_find_returns_delivery_or_none():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    d = q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    assert q.find(d.delivery_id) is d
    assert q.find("whk-zzz") is None


# ---------------------------------------------------------------------------
# /api/v1/webhooks/deliveries/{id}/ack — endpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def queue_with_one_pending():
    q = WebhookDeliveryQueue(transport=_ok_transport)
    d = q.enqueue(url="https://b.example/hook", secret="s", body="{}")
    return q, d


@pytest.fixture
def app_with_queue(queue_with_one_pending):
    q, _ = queue_with_one_pending
    return create_app(webhook_queue=q, admin_action_log=AdminActionLog())


@pytest.fixture
def client(app_with_queue):
    return TestClient(app_with_queue)


def test_ack_pending_returns_acknowledged(client, queue_with_one_pending):
    q, d = queue_with_one_pending
    resp = client.post(f"/api/v1/webhooks/deliveries/{d.delivery_id}/ack")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {
        "delivery_id": d.delivery_id,
        "state": "pending",
        "acknowledged": True,
    }
    assert q.pending_size == 0


def test_ack_unknown_returns_not_found_not_404(client):
    resp = client.post("/api/v1/webhooks/deliveries/whk-00000099/ack")
    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "not_found"
    assert body["acknowledged"] is False


def test_ack_writes_admin_audit_record(client, queue_with_one_pending, app_with_queue):
    q, d = queue_with_one_pending
    client.post(f"/api/v1/webhooks/deliveries/{d.delivery_id}/ack")
    log = app_with_queue.state.app_state.admin_action_log
    rows = log.query(action="webhook_ack")
    assert len(rows) == 1
    assert rows[0].target == d.delivery_id
    # actor format is "key:<id>" — TESTING_MODE gives key_id=0
    assert rows[0].actor.startswith("key:")


def test_ack_rejects_overlong_id(client):
    long_id = "a" * 65
    resp = client.post(f"/api/v1/webhooks/deliveries/{long_id}/ack")
    assert resp.status_code == 400


def test_503_when_queue_not_wired():
    app = create_app()  # no webhook_queue
    c = TestClient(app)
    resp = c.post("/api/v1/webhooks/deliveries/whk-1/ack")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# GET /api/v1/webhooks/deliveries/{id} — inspect
# ---------------------------------------------------------------------------


def test_inspect_returns_state(client, queue_with_one_pending):
    _, d = queue_with_one_pending
    resp = client.get(f"/api/v1/webhooks/deliveries/{d.delivery_id}")
    assert resp.status_code == 200
    body = resp.json()
    assert body["delivery_id"] == d.delivery_id
    assert body["url"] == "https://b.example/hook"
    assert body["attempts"] == 0


def test_inspect_404_for_unknown(client):
    resp = client.get("/api/v1/webhooks/deliveries/whk-zzz")
    assert resp.status_code == 404
