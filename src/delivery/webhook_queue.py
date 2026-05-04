"""In-process webhook delivery queue — Sprint INFRA-2B.8.

Built on top of :mod:`src.delivery.webhook_signer` (HMAC v1). The queue
is intentionally in-process and synchronous: no Redis, no Celery, no
background thread by default. Callers ``enqueue()`` deliveries and
``drain()`` the queue when they're ready (typically a background
``asyncio`` task in production, or a CLI runner in dev).

Transport pluggability
----------------------
The actual HTTP send is supplied as a ``transport`` callable
``(url, headers, body) -> int (status_code)``. This keeps the queue
free of an HTTP client dependency and lets tests inject a deterministic
stub. A production caller wires ``httpx.post`` (or ``requests``) +
timeout.

Retry policy
------------
- Up to ``max_attempts`` (default 5).
- Exponential backoff with jitter: ``2**attempt`` seconds clamped to
  ``max_backoff_seconds`` (default 600s = 10min).
- Anything 2xx is success.
- 4xx (client error other than 408 / 429) is permanent failure → straight
  to dead-letter, no retry.
- 408 / 429 / 5xx / network exception → retry.

Dead-letter
-----------
After ``max_attempts`` exhaustion, the delivery moves to the dead-letter
list. Inspectors can read it via ``dead_letter()`` and operators can
``requeue_dead_letter(idx)`` once the upstream is fixed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from src.delivery.webhook_signer import SIGNATURE_HEADER, sign_payload

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


# A transport returns the HTTP status code or raises an exception.
Transport = Callable[[str, dict, str], int]


@dataclass
class WebhookDelivery:
    """One enqueued delivery."""

    delivery_id: str
    url: str
    secret: str
    body: str
    enqueued_at: float
    attempts: int = 0
    last_error: str = ""
    last_status: Optional[int] = None
    next_attempt_at: float = 0.0


@dataclass
class DeliveryReport:
    """Outcome of a ``drain()`` pass."""

    succeeded: int = 0
    retried: int = 0
    dead_lettered: int = 0
    skipped_not_due: int = 0


# ---------------------------------------------------------------------------
# Retry classification
# ---------------------------------------------------------------------------


def _classify(status: int) -> str:
    """Return one of {"success", "retry", "permanent"}."""
    if 200 <= status < 300:
        return "success"
    if status in (408, 429):
        return "retry"
    if 500 <= status < 600:
        return "retry"
    return "permanent"


def _backoff_seconds(attempts: int, *, base: float, cap: float) -> float:
    return min(cap, base * (2 ** max(0, attempts - 1)))


# ---------------------------------------------------------------------------
# Queue
# ---------------------------------------------------------------------------


class WebhookDeliveryQueue:
    """Thread-safe in-process delivery queue with retry + dead-letter."""

    def __init__(
        self,
        *,
        transport: Transport,
        max_attempts: int = 5,
        base_backoff_seconds: float = 1.0,
        max_backoff_seconds: float = 600.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transport = transport
        self._max_attempts = max_attempts
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._pending: list[WebhookDelivery] = []
        self._dead: list[WebhookDelivery] = []
        self._next_id = 0

    # ------------------------------------------------------------------
    # Enqueue
    # ------------------------------------------------------------------

    def enqueue(self, *, url: str, secret: str, body: str) -> WebhookDelivery:
        """Push a delivery onto the queue. First attempt is due immediately."""
        if not url:
            raise ValueError("url is required")
        if not secret:
            raise ValueError("secret is required")
        with self._lock:
            self._next_id += 1
            d = WebhookDelivery(
                delivery_id=f"whk-{self._next_id:08d}",
                url=url,
                secret=secret,
                body=body,
                enqueued_at=self._clock(),
                next_attempt_at=self._clock(),
            )
            self._pending.append(d)
            return d

    # ------------------------------------------------------------------
    # Drain
    # ------------------------------------------------------------------

    def drain(self) -> DeliveryReport:
        """Attempt delivery on every due item exactly once.

        Returns a count of outcomes; the caller decides whether to call
        ``drain()`` again immediately (if there were retries that may
        already be due) or sleep until ``next_due_at()``.
        """
        report = DeliveryReport()
        with self._lock:
            queue_snapshot = list(self._pending)
            self._pending.clear()
            now = self._clock()
            requeue: list[WebhookDelivery] = []

        for d in queue_snapshot:
            if d.next_attempt_at > now:
                requeue.append(d)
                report.skipped_not_due += 1
                continue

            d.attempts += 1
            try:
                status = self._send(d)
                d.last_status = status
                outcome = _classify(status)
            except Exception as exc:  # network / timeout
                d.last_status = None
                d.last_error = str(exc)[:200]
                outcome = "retry"
                logger.warning("webhook %s transport raised: %s", d.delivery_id, exc)

            if outcome == "success":
                report.succeeded += 1
                continue

            if outcome == "permanent":
                report.dead_lettered += 1
                with self._lock:
                    self._dead.append(d)
                continue

            # outcome == "retry"
            if d.attempts >= self._max_attempts:
                report.dead_lettered += 1
                with self._lock:
                    self._dead.append(d)
                continue

            d.next_attempt_at = self._clock() + _backoff_seconds(
                d.attempts, base=self._base_backoff, cap=self._max_backoff
            )
            requeue.append(d)
            report.retried += 1

        with self._lock:
            # New items may have arrived during the drain — preserve them.
            self._pending = requeue + self._pending

        return report

    def _send(self, d: WebhookDelivery) -> int:
        signed = sign_payload(d.body, d.secret)
        headers = {
            "Content-Type": "application/json",
            SIGNATURE_HEADER: signed.header_value,
        }
        return self._transport(d.url, headers, d.body)

    # ------------------------------------------------------------------
    # Inspection / ops
    # ------------------------------------------------------------------

    @property
    def pending_size(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def dead_letter_size(self) -> int:
        with self._lock:
            return len(self._dead)

    def pending(self) -> list[WebhookDelivery]:
        with self._lock:
            return list(self._pending)

    def dead_letter(self) -> list[WebhookDelivery]:
        with self._lock:
            return list(self._dead)

    def next_due_at(self) -> Optional[float]:
        """Earliest ``next_attempt_at`` of any pending delivery, or None
        when the queue is empty."""
        with self._lock:
            if not self._pending:
                return None
            return min(d.next_attempt_at for d in self._pending)

    def requeue_dead_letter(self, delivery_id: str) -> bool:
        """Move one dead-letter delivery back to the pending queue with
        a fresh attempt counter. Returns True if found."""
        with self._lock:
            for i, d in enumerate(self._dead):
                if d.delivery_id == delivery_id:
                    revived = self._dead.pop(i)
                    revived.attempts = 0
                    revived.last_error = ""
                    revived.last_status = None
                    revived.next_attempt_at = self._clock()
                    self._pending.append(revived)
                    return True
            return False


__all__ = [
    "DeliveryReport",
    "Transport",
    "WebhookDelivery",
    "WebhookDeliveryQueue",
]
