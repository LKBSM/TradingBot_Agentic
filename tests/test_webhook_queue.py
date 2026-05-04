"""Tests for the INFRA-2B.8 webhook delivery queue."""

from __future__ import annotations

import threading
from typing import Callable

import pytest

from src.delivery.webhook_queue import (
    DeliveryReport,
    WebhookDelivery,
    WebhookDeliveryQueue,
    _backoff_seconds,
    _classify,
)
from src.delivery.webhook_signer import SIGNATURE_HEADER, verify_payload


# ---------------------------------------------------------------------------
# Helpers — fake transport + clock
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_transport(plan: list):
    """Replay a sequence of (status_or_exception) results per call."""
    plan = list(plan)
    captured: list[tuple[str, dict, str]] = []

    def transport(url: str, headers: dict, body: str) -> int:
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
# Classifier + backoff unit tests
# ---------------------------------------------------------------------------


def test_classify_success():
    for s in (200, 201, 204, 299):
        assert _classify(s) == "success"


def test_classify_retry():
    for s in (408, 429, 500, 502, 503, 504, 599):
        assert _classify(s) == "retry"


def test_classify_permanent():
    for s in (400, 401, 403, 404, 410, 422):
        assert _classify(s) == "permanent"


def test_backoff_doubles_then_caps():
    assert _backoff_seconds(1, base=1.0, cap=600.0) == 1.0
    assert _backoff_seconds(2, base=1.0, cap=600.0) == 2.0
    assert _backoff_seconds(3, base=1.0, cap=600.0) == 4.0
    assert _backoff_seconds(20, base=1.0, cap=600.0) == 600.0


# ---------------------------------------------------------------------------
# Enqueue
# ---------------------------------------------------------------------------


def test_enqueue_assigns_id_and_due_now():
    transport = make_transport([])
    clock = FakeClock(t=100.0)
    q = WebhookDeliveryQueue(transport=transport, clock=clock)
    d = q.enqueue(url="https://hook.example.com/x", secret="s", body="{}")
    assert d.delivery_id.startswith("whk-")
    assert d.next_attempt_at == 100.0
    assert q.pending_size == 1


def test_enqueue_rejects_empty_url_or_secret():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    with pytest.raises(ValueError):
        q.enqueue(url="", secret="s", body="{}")
    with pytest.raises(ValueError):
        q.enqueue(url="https://x", secret="", body="{}")


# ---------------------------------------------------------------------------
# Drain — happy path
# ---------------------------------------------------------------------------


def test_drain_success_clears_pending():
    transport = make_transport([200])
    q = WebhookDeliveryQueue(transport=transport)
    q.enqueue(url="https://x", secret="s", body="{}")
    report = q.drain()
    assert report.succeeded == 1
    assert report.retried == 0
    assert q.pending_size == 0
    assert q.dead_letter_size == 0


def test_drain_signs_payload_with_x_sentinel_signature_header():
    transport = make_transport([200])
    q = WebhookDeliveryQueue(transport=transport)
    q.enqueue(url="https://x", secret="topsecret", body='{"a":1}')
    q.drain()
    url, headers, body = transport.captured[0]
    assert SIGNATURE_HEADER in headers
    assert headers["Content-Type"] == "application/json"
    # Receiver-side verification round-trips.
    res = verify_payload(body, headers[SIGNATURE_HEADER], "topsecret")
    assert res.ok


# ---------------------------------------------------------------------------
# Drain — retry semantics
# ---------------------------------------------------------------------------


def test_drain_retries_on_5xx():
    transport = make_transport([503, 200])
    clock = FakeClock(t=0.0)
    q = WebhookDeliveryQueue(transport=transport, clock=clock,
                             base_backoff_seconds=1.0)
    q.enqueue(url="https://x", secret="s", body="{}")
    r1 = q.drain()
    assert r1.retried == 1
    assert q.pending_size == 1
    # Not yet due (backoff = 1s for attempt 1).
    r2 = q.drain()
    assert r2.skipped_not_due == 1
    # Advance past the backoff window.
    clock.advance(2.0)
    r3 = q.drain()
    assert r3.succeeded == 1
    assert q.pending_size == 0


def test_drain_dead_letters_after_max_attempts():
    transport = make_transport([503, 503, 503])
    clock = FakeClock(t=0.0)
    q = WebhookDeliveryQueue(
        transport=transport, max_attempts=3, base_backoff_seconds=0.1, clock=clock
    )
    q.enqueue(url="https://x", secret="s", body="{}")
    for _ in range(3):
        q.drain()
        clock.advance(10.0)
    assert q.dead_letter_size == 1
    assert q.pending_size == 0


def test_drain_permanent_failure_skips_retry_chain():
    """A 4xx (other than 408 / 429) goes straight to dead-letter — no
    point retrying a malformed broker URL."""
    transport = make_transport([404])
    q = WebhookDeliveryQueue(transport=transport, max_attempts=5)
    q.enqueue(url="https://x", secret="s", body="{}")
    report = q.drain()
    assert report.dead_lettered == 1
    assert report.retried == 0
    assert q.dead_letter_size == 1


def test_drain_429_is_retried():
    """Rate-limit responses must be retried (subscriber may catch up)."""
    transport = make_transport([429, 200])
    clock = FakeClock(t=0.0)
    q = WebhookDeliveryQueue(
        transport=transport, base_backoff_seconds=0.5, clock=clock
    )
    q.enqueue(url="https://x", secret="s", body="{}")
    q.drain()
    clock.advance(1.0)
    q.drain()
    assert q.dead_letter_size == 0


def test_drain_network_exception_retries():
    transport = make_transport([RuntimeError("connection reset"), 200])
    clock = FakeClock(t=0.0)
    q = WebhookDeliveryQueue(transport=transport, base_backoff_seconds=0.5,
                             clock=clock)
    q.enqueue(url="https://x", secret="s", body="{}")
    r1 = q.drain()
    assert r1.retried == 1
    [pending] = q.pending()
    assert "connection reset" in pending.last_error
    clock.advance(1.0)
    r2 = q.drain()
    assert r2.succeeded == 1


# ---------------------------------------------------------------------------
# Inspection / ops
# ---------------------------------------------------------------------------


def test_next_due_at_reports_min_pending():
    transport = make_transport([])
    clock = FakeClock(t=100.0)
    q = WebhookDeliveryQueue(transport=transport, clock=clock)
    assert q.next_due_at() is None
    q.enqueue(url="https://x", secret="s", body="{}")
    assert q.next_due_at() == 100.0


def test_requeue_dead_letter_revives_delivery():
    transport = make_transport([404])
    q = WebhookDeliveryQueue(transport=transport)
    d = q.enqueue(url="https://x", secret="s", body="{}")
    q.drain()
    assert q.dead_letter_size == 1
    assert q.requeue_dead_letter(d.delivery_id)
    assert q.dead_letter_size == 0
    assert q.pending_size == 1


def test_requeue_dead_letter_unknown_id_returns_false():
    q = WebhookDeliveryQueue(transport=make_transport([]))
    assert not q.requeue_dead_letter("missing")


# ---------------------------------------------------------------------------
# Concurrent enqueue + drain
# ---------------------------------------------------------------------------


def test_concurrent_enqueue_does_not_lose_deliveries():
    """Stress: 8 threads enqueueing while a 9th drains — every successful
    delivery must be accounted for."""
    transport = make_transport([200] * 1000)
    q = WebhookDeliveryQueue(transport=transport)
    n_per = 50
    n_threads = 8
    enqueue_done = threading.Event()

    def enqueue_worker(tid: int):
        for j in range(n_per):
            q.enqueue(url="https://x", secret="s",
                      body=f'{{"t":{tid},"j":{j}}}')

    drained_succeeded = {"n": 0}

    def drainer():
        while not enqueue_done.is_set() or q.pending_size:
            r = q.drain()
            drained_succeeded["n"] += r.succeeded

    threads = [threading.Thread(target=enqueue_worker, args=(i,))
               for i in range(n_threads)]
    drainer_thread = threading.Thread(target=drainer)
    for t in threads:
        t.start()
    drainer_thread.start()
    for t in threads:
        t.join()
    enqueue_done.set()
    drainer_thread.join(timeout=5.0)
    assert drained_succeeded["n"] == n_threads * n_per
    assert q.pending_size == 0
    assert q.dead_letter_size == 0
