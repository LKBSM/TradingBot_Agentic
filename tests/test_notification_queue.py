"""Tests for NotificationQueue (Sprint 2.2)."""
from __future__ import annotations

import time

import pytest

from src.intelligence.notification_queue import NotificationQueue


@pytest.fixture
def queue(tmp_path):
    return NotificationQueue(db_path=str(tmp_path / "notif.db"), ttl_seconds=60)


def test_enqueue_basic(queue):
    assert queue.enqueue("sig_1", {"text": "hello"}) is True
    assert queue.size() == 1


def test_enqueue_idempotent_on_signal_id(queue):
    assert queue.enqueue("sig_1", {"text": "first"}) is True
    assert queue.enqueue("sig_1", {"text": "duplicate"}) is False
    assert queue.size() == 1


def test_enqueue_rejects_empty_signal_id(queue):
    assert queue.enqueue("", {"text": "no id"}) is False
    assert queue.size() == 0


def test_replay_success_clears_queue(queue):
    queue.enqueue("sig_1", {"text": "a"})
    queue.enqueue("sig_2", {"text": "b"})
    delivered = []
    out = queue.replay(notify_fn=lambda p: delivered.append(p))
    assert out["replayed"] == 2
    assert out["failed"] == 0
    assert queue.size() == 0
    assert {p["text"] for p in delivered} == {"a", "b"}


def test_replay_failure_keeps_in_queue(queue):
    queue.enqueue("sig_1", {"text": "boom"})

    def boom(_p):
        raise RuntimeError("notifier down")

    out = queue.replay(notify_fn=boom)
    assert out["replayed"] == 0
    assert out["failed"] == 1
    assert queue.size() == 1


def test_replay_increments_attempts(queue):
    queue.enqueue("sig_1", {"text": "x"})

    def boom(_p):
        raise RuntimeError()

    queue.replay(notify_fn=boom)
    queue.replay(notify_fn=boom)

    # internal field via SQL
    conn = queue._get_connection()
    try:
        row = conn.execute("SELECT attempts FROM pending_notifications").fetchone()
        assert row["attempts"] == 2
    finally:
        conn.close()


def test_ttl_expiry_drops_old(tmp_path):
    queue = NotificationQueue(db_path=str(tmp_path / "notif.db"), ttl_seconds=1)
    queue.enqueue("sig_old", {"text": "old"})
    time.sleep(1.2)
    queue.enqueue("sig_new", {"text": "new"})

    delivered = []
    out = queue.replay(notify_fn=lambda p: delivered.append(p))
    assert out["dropped_expired"] == 1
    assert out["replayed"] == 1
    assert delivered[0]["text"] == "new"


def test_corrupt_payload_dropped(queue):
    # Manually insert garbage to simulate disk corruption
    queue.enqueue("sig_ok", {"text": "fine"})
    conn = queue._get_connection()
    try:
        conn.execute(
            "INSERT INTO pending_notifications "
            "(signal_id, payload, enqueued_at, attempts) "
            "VALUES (?, ?, ?, 0)",
            ("sig_corrupt", "<<not json>>", time.time()),
        )
    finally:
        conn.close()

    delivered = []
    out = queue.replay(notify_fn=lambda p: delivered.append(p))
    assert out["replayed"] == 1
    assert out["failed"] == 1  # corrupt one was counted as failed but dropped
    assert queue.size() == 0


def test_get_stats(queue):
    queue.enqueue("a", {})
    queue.enqueue("b", {})
    queue.replay(notify_fn=lambda p: None)
    stats = queue.get_stats()
    assert stats["enqueued_total"] == 2
    assert stats["replayed_total"] == 2
    assert stats["current_depth"] == 0
    assert stats["ttl_seconds"] == 60


def test_clear(queue):
    queue.enqueue("a", {})
    queue.enqueue("b", {})
    deleted = queue.clear()
    assert deleted == 2
    assert queue.size() == 0


def test_max_per_run_caps_batch(queue):
    for i in range(20):
        queue.enqueue(f"sig_{i}", {"i": i})
    out = queue.replay(notify_fn=lambda p: None, max_per_run=5)
    assert out["replayed"] == 5
    assert queue.size() == 15


def test_persistence_across_instances(tmp_path):
    db = str(tmp_path / "notif.db")
    q1 = NotificationQueue(db_path=db, ttl_seconds=60)
    q1.enqueue("persist", {"data": [1, 2, 3]})
    del q1

    q2 = NotificationQueue(db_path=db, ttl_seconds=60)
    assert q2.size() == 1
    delivered = []
    q2.replay(notify_fn=lambda p: delivered.append(p))
    assert delivered[0]["data"] == [1, 2, 3]


def test_cleanup_expired_explicit(tmp_path):
    queue = NotificationQueue(db_path=str(tmp_path / "notif.db"), ttl_seconds=1)
    queue.enqueue("a", {})
    time.sleep(1.2)
    deleted = queue.cleanup_expired()
    assert deleted == 1
    assert queue.size() == 0
