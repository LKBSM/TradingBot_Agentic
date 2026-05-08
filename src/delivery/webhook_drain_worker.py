"""Background drain worker for the webhook delivery queue — Sprint INFRA-2B.10.

The INFRA-2B.8 queue is in-process and exposes ``drain()`` as a manual
operation. In production, *something* has to keep calling ``drain()``
on a schedule, react to retry backoff, and shut down cleanly when the
process exits.

This module provides a thin asyncio worker that:

- wakes whenever ``WebhookDeliveryQueue.next_due_at()`` says a delivery
  is due (no busy-loop polling),
- enforces a minimum and maximum sleep so a permanently empty queue
  doesn't hammer the event loop, and a perpetually due queue doesn't
  starve other tasks,
- supports cooperative cancellation: ``stop()`` flips an event, the
  worker drains one final time and exits.

Why asyncio + a tiny manual loop, not aiojobs / Celery?
The queue itself is already in-process and synchronous; we only need a
single bound goroutine. Pulling in a job framework would be operational
overhead with no benefit at this scale (1-1000 webhooks/day).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Optional

from src.delivery.webhook_queue import DeliveryReport, WebhookDeliveryQueue

logger = logging.getLogger(__name__)


DEFAULT_MIN_SLEEP_SECONDS = 1.0
DEFAULT_MAX_SLEEP_SECONDS = 30.0


class WebhookDrainWorker:
    """Single-task drain loop bound to one ``WebhookDeliveryQueue``.

    Lifecycle::

        worker = WebhookDrainWorker(queue)
        await worker.start()         # spawns the task
        # ... application runs ...
        await worker.stop()          # cooperative shutdown + final drain

    The worker tracks aggregate stats (``cycles_run``, ``successes``,
    ``retried``, ``dead_lettered``) so /health/deep can surface drain
    health without poking inside the queue.
    """

    def __init__(
        self,
        queue: WebhookDeliveryQueue,
        *,
        min_sleep_seconds: float = DEFAULT_MIN_SLEEP_SECONDS,
        max_sleep_seconds: float = DEFAULT_MAX_SLEEP_SECONDS,
        clock=time.monotonic,
    ):
        if min_sleep_seconds <= 0:
            raise ValueError("min_sleep_seconds must be positive")
        if max_sleep_seconds < min_sleep_seconds:
            raise ValueError("max_sleep_seconds must be >= min_sleep_seconds")
        self._queue = queue
        self._min_sleep = min_sleep_seconds
        self._max_sleep = max_sleep_seconds
        self._clock = clock

        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None

        # Aggregate counters for observability
        self.cycles_run: int = 0
        self.successes: int = 0
        self.retried: int = 0
        self.dead_lettered: int = 0
        self.skipped_not_due: int = 0
        self.last_cycle_at: Optional[float] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.is_running:
            return
        self._stop_event = asyncio.Event()
        self._task = asyncio.create_task(
            self._run(), name="webhook-drain-worker"
        )

    async def stop(self, *, drain_final: bool = True) -> None:
        """Signal the worker to exit. Returns once the task has finished.

        ``drain_final=True`` runs one last ``drain()`` after the loop has
        been signalled to stop, so a delivery that became due in the
        last sleep window isn't left hanging until the process restarts.
        """
        if self._stop_event is None or not self.is_running:
            return
        self._stop_event.set()
        try:
            await self._task  # type: ignore[arg-type]
        except asyncio.CancelledError:
            pass
        if drain_final:
            try:
                self._record(self._queue.drain())
            except Exception as exc:  # pragma: no cover — defensive
                logger.warning("final drain raised: %s", exc)

    # ------------------------------------------------------------------
    # Core loop
    # ------------------------------------------------------------------

    def _next_sleep_seconds(self) -> float:
        """Decide how long to sleep before the next drain pass.

        Uses ``next_due_at()`` if anything is queued. Falls back to
        ``min_sleep`` when empty so we don't pin a CPU, and clamps to
        ``max_sleep`` so even a stuck delivery wakes us periodically.
        """
        due_at = self._queue.next_due_at()
        if due_at is None:
            return self._max_sleep  # idle — long nap
        wait = due_at - self._clock()
        if wait < self._min_sleep:
            return self._min_sleep
        if wait > self._max_sleep:
            return self._max_sleep
        return wait

    def _record(self, report: DeliveryReport) -> None:
        self.cycles_run += 1
        self.successes += report.succeeded
        self.retried += report.retried
        self.dead_lettered += report.dead_lettered
        self.skipped_not_due += report.skipped_not_due
        self.last_cycle_at = self._clock()

    async def _run(self) -> None:
        assert self._stop_event is not None
        logger.info("webhook drain worker started")
        try:
            while not self._stop_event.is_set():
                try:
                    self._record(self._queue.drain())
                except Exception as exc:  # one bad cycle shouldn't kill the worker
                    logger.exception("drain cycle raised: %s", exc)

                if self._stop_event.is_set():
                    break

                wait = self._next_sleep_seconds()
                try:
                    # Either the timer fires, or stop() flips the event.
                    await asyncio.wait_for(self._stop_event.wait(), timeout=wait)
                    # If we got here without timeout, stop was set ⇒ exit.
                    break
                except asyncio.TimeoutError:
                    continue
        finally:
            logger.info(
                "webhook drain worker stopped after %d cycles (%d ok, %d retried, "
                "%d dead, %d skipped)",
                self.cycles_run,
                self.successes,
                self.retried,
                self.dead_lettered,
                self.skipped_not_due,
            )

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "running": self.is_running,
            "cycles_run": self.cycles_run,
            "successes": self.successes,
            "retried": self.retried,
            "dead_lettered": self.dead_lettered,
            "skipped_not_due": self.skipped_not_due,
            "last_cycle_at": self.last_cycle_at,
            "pending": self._queue.pending_size,
            "dead_letter": self._queue.dead_letter_size,
        }


__all__ = [
    "DEFAULT_MAX_SLEEP_SECONDS",
    "DEFAULT_MIN_SLEEP_SECONDS",
    "WebhookDrainWorker",
]
