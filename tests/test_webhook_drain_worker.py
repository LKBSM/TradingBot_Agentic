"""Tests for the INFRA-2B.10 webhook drain worker.

We avoid pytest-asyncio (extra dep): each async-bodied test is wrapped
with ``asyncio.run`` via the ``run_async`` helper so the suite stays
under the same minimal dep set as the rest of CI.
"""

from __future__ import annotations

import asyncio

import pytest

from src.delivery.webhook_drain_worker import (
    DEFAULT_MAX_SLEEP_SECONDS,
    DEFAULT_MIN_SLEEP_SECONDS,
    WebhookDrainWorker,
)
from src.delivery.webhook_queue import WebhookDeliveryQueue


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_transport(plan):
    plan = list(plan)
    captured = []

    def transport(url, headers, body):
        captured.append((url, dict(headers), body))
        if not plan:
            return 200
        nxt = plan.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return int(nxt)

    transport.captured = captured  # type: ignore[attr-defined]
    return transport


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_invalid_sleep_bounds_rejected():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    with pytest.raises(ValueError):
        WebhookDrainWorker(q, min_sleep_seconds=0)
    with pytest.raises(ValueError):
        WebhookDrainWorker(q, min_sleep_seconds=10, max_sleep_seconds=5)


def test_idle_worker_not_running_until_started():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    w = WebhookDrainWorker(q)
    assert w.is_running is False
    stats = w.stats()
    assert stats["running"] is False
    assert stats["cycles_run"] == 0


# ---------------------------------------------------------------------------
# Drain a queued delivery
# ---------------------------------------------------------------------------


def test_worker_drains_pending_delivery():
    transport = make_transport([200])
    q = WebhookDeliveryQueue(transport=transport)
    q.enqueue(url="https://x", secret="s", body="{}")
    w = WebhookDrainWorker(q, min_sleep_seconds=0.05, max_sleep_seconds=0.5)

    async def scenario():
        await w.start()
        await asyncio.sleep(0.2)
        await w.stop(drain_final=False)

    run_async(scenario())

    assert w.successes == 1
    assert q.pending_size == 0
    assert q.dead_letter_size == 0
    assert len(transport.captured) == 1


def test_worker_handles_multiple_cycles():
    transport = make_transport([200, 200, 200])
    q = WebhookDeliveryQueue(transport=transport)
    w = WebhookDrainWorker(q, min_sleep_seconds=0.05, max_sleep_seconds=0.5)

    async def scenario():
        await w.start()
        for _ in range(3):
            q.enqueue(url="https://x", secret="s", body="{}")
            await asyncio.sleep(0.1)
        await asyncio.sleep(0.3)
        await w.stop(drain_final=True)

    run_async(scenario())

    assert w.successes == 3
    assert q.pending_size == 0


# ---------------------------------------------------------------------------
# Retry interaction
# ---------------------------------------------------------------------------


def test_worker_respects_retry_backoff():
    transport = make_transport([503, 200])
    q = WebhookDeliveryQueue(
        transport=transport, base_backoff_seconds=0.5
    )
    q.enqueue(url="https://x", secret="s", body="{}")

    w = WebhookDrainWorker(q, min_sleep_seconds=0.05, max_sleep_seconds=1.0)

    async def scenario():
        await w.start()
        await asyncio.sleep(0.2)
        # First attempt fired, retried scheduled
        assert w.retried >= 1
        assert q.pending_size == 1
        # Wait past backoff so second attempt fires.
        await asyncio.sleep(0.7)
        await w.stop(drain_final=False)

    run_async(scenario())

    assert w.successes == 1
    assert q.pending_size == 0


# ---------------------------------------------------------------------------
# Cooperative shutdown
# ---------------------------------------------------------------------------


def test_stop_drains_final_when_requested():
    transport = make_transport([200])
    q = WebhookDeliveryQueue(transport=transport)
    w = WebhookDrainWorker(q, min_sleep_seconds=0.5, max_sleep_seconds=2.0)

    async def scenario():
        await w.start()
        q.enqueue(url="https://x", secret="s", body="{}")
        await w.stop(drain_final=True)

    run_async(scenario())

    assert q.pending_size == 0
    assert w.is_running is False


def test_stop_without_final_drain_leaves_pending_alone():
    transport = make_transport([200])
    q = WebhookDeliveryQueue(transport=transport)
    w = WebhookDrainWorker(q, min_sleep_seconds=5.0, max_sleep_seconds=10.0)

    async def scenario():
        await w.start()
        await asyncio.sleep(0.05)  # let the loop reach the sleep
        q.enqueue(url="https://x", secret="s", body="{}")
        await w.stop(drain_final=False)

    run_async(scenario())

    assert q.pending_size == 1


def test_double_start_is_idempotent():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    w = WebhookDrainWorker(q, min_sleep_seconds=0.1, max_sleep_seconds=0.5)

    async def scenario():
        await w.start()
        task1 = w._task
        await w.start()
        assert w._task is task1
        await w.stop()

    run_async(scenario())


def test_stop_before_start_is_safe():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    w = WebhookDrainWorker(q)

    async def scenario():
        await w.stop()  # must not raise

    run_async(scenario())


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_worker_survives_drain_exception():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    real_drain = q.drain
    state = {"first": True}

    def flaky_drain():
        if state["first"]:
            state["first"] = False
            raise RuntimeError("simulated transient")
        return real_drain()

    q.drain = flaky_drain  # type: ignore[assignment]
    w = WebhookDrainWorker(q, min_sleep_seconds=0.05, max_sleep_seconds=0.2)

    async def scenario():
        await w.start()
        await asyncio.sleep(0.3)
        await w.stop(drain_final=False)

    run_async(scenario())

    assert w.cycles_run >= 1


# ---------------------------------------------------------------------------
# stats() + sleep heuristic (sync — no event loop needed)
# ---------------------------------------------------------------------------


def test_stats_reflects_queue_depth():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    q.enqueue(url="https://x", secret="s", body="{}")
    q.enqueue(url="https://y", secret="s", body="{}")
    w = WebhookDrainWorker(q)
    stats = w.stats()
    assert stats["pending"] == 2
    assert stats["dead_letter"] == 0


def test_next_sleep_idle_returns_max():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    w = WebhookDrainWorker(q, min_sleep_seconds=1.0, max_sleep_seconds=10.0)
    assert w._next_sleep_seconds() == 10.0


def test_next_sleep_clamps_floor_for_due_now():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    q.enqueue(url="https://x", secret="s", body="{}")  # due_at = now
    w = WebhookDrainWorker(q, min_sleep_seconds=0.5, max_sleep_seconds=10.0)
    assert w._next_sleep_seconds() == 0.5


def test_default_constants():
    assert 0 < DEFAULT_MIN_SLEEP_SECONDS < DEFAULT_MAX_SLEEP_SECONDS
