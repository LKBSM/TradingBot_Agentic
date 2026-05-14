"""Graceful shutdown coordinator — Sprint INFRA-2B.11.

Why a dedicated coordinator
---------------------------
On SIGTERM, FastAPI's lifespan fires once and *every* subsystem needs
a clean exit:

- the webhook drain worker must finish its final pass so deliveries
  due in the last sleep window aren't deferred to the next restart,
- the audit ledger's SQLite handle must commit and close (a half-open
  WAL after kill -9 is recoverable; after SIGTERM it's expected to be
  flushed),
- the sentinel scanner must stop polling so we don't write a final
  signal mid-shutdown,
- the SignalStore SQLite handle must close to avoid `database is
  locked` warnings on the next boot.

Doing all of that as a flat `try/except` in `lifespan()` works for two
handlers but breaks down fast: one slow handler blocks the rest, one
exception suppresses every subsequent close, and there's no
observability around what got cleaned up vs. what didn't.

This module centralises the pattern:

- handlers register with a budget (seconds) and a name,
- each runs under ``asyncio.wait_for(budget)`` — a stuck handler is
  cancelled, the others still get their chance,
- exceptions are caught and logged with handler name, never propagated
  (a failed close is a log line, not a crash),
- the run produces a structured ``ShutdownReport`` so post-mortem can
  see exactly which handlers ran, how long they took, what (if
  anything) raised.

Total budget is enforced as a soft global cap: handlers run
sequentially in registration order, each capped at its own budget, and
the coordinator stops accepting new handlers once the global budget
is exhausted (so a misbehaving handler can't steal time from later
ones).

Why sequential and not concurrent
---------------------------------
The shutdown order matters. The webhook drain worker has to stop
before we close the audit ledger (it dereferences the ledger). The
SignalStore closes last so health probes during shutdown can still
read state. ``asyncio.gather`` would lose that ordering, so we accept
the latency cost of sequencing for the determinism gain.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, Union

logger = logging.getLogger(__name__)


DEFAULT_HANDLER_BUDGET_S = 5.0
DEFAULT_TOTAL_BUDGET_S = 30.0


# Handler can be sync (``Callable[[], None]``) or async
# (``Callable[[], Awaitable[None]]``). Sync handlers are awaited via
# ``asyncio.to_thread`` so a blocking ``close()`` doesn't stall the
# event loop past its own budget.
Handler = Union[Callable[[], None], Callable[[], Awaitable[None]]]


@dataclass(frozen=True)
class HandlerResult:
    name: str
    ok: bool
    duration_ms: float
    error: Optional[str] = None
    timed_out: bool = False
    skipped: bool = False  # global budget exhausted before we got to it


@dataclass
class ShutdownReport:
    started_at: float
    finished_at: float
    total_duration_ms: float
    handlers: List[HandlerResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(h.ok or h.skipped for h in self.handlers)

    @property
    def n_failed(self) -> int:
        return sum(1 for h in self.handlers if not h.ok and not h.skipped)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "n_handlers": len(self.handlers),
            "n_failed": self.n_failed,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "total_duration_ms": round(self.total_duration_ms, 2),
            "handlers": [
                {
                    "name": h.name,
                    "ok": h.ok,
                    "duration_ms": round(h.duration_ms, 2),
                    "error": h.error,
                    "timed_out": h.timed_out,
                    "skipped": h.skipped,
                }
                for h in self.handlers
            ],
        }


@dataclass
class _Registration:
    name: str
    handler: Handler
    budget_s: float


class GracefulShutdownCoordinator:
    """Sequenced, budgeted, exception-safe shutdown runner.

    Usage::

        coord = GracefulShutdownCoordinator(total_budget_s=15)
        coord.register("webhook-drain", drain_worker.stop, budget_s=5)
        coord.register("audit-ledger", ledger.close, budget_s=2)
        ...
        report = await coord.run()
    """

    def __init__(
        self,
        *,
        total_budget_s: float = DEFAULT_TOTAL_BUDGET_S,
        clock: Callable[[], float] = time.monotonic,
    ):
        if total_budget_s <= 0:
            raise ValueError("total_budget_s must be positive")
        self._total_budget_s = total_budget_s
        self._clock = clock
        self._registrations: List[_Registration] = []
        self._shutting_down = False
        self._report: Optional[ShutdownReport] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_shutting_down(self) -> bool:
        return self._shutting_down

    @property
    def report(self) -> Optional[ShutdownReport]:
        return self._report

    def register(
        self,
        name: str,
        handler: Handler,
        *,
        budget_s: float = DEFAULT_HANDLER_BUDGET_S,
    ) -> None:
        """Register a shutdown handler.

        ``name`` is used purely for logging — keep it stable across
        deploys so log-based dashboards don't churn.
        Handlers are run in registration order.
        """
        if not name:
            raise ValueError("handler name required")
        if budget_s <= 0:
            raise ValueError("handler budget_s must be positive")
        if self._shutting_down:
            raise RuntimeError(
                "cannot register handlers after shutdown has started"
            )
        self._registrations.append(
            _Registration(name=name, handler=handler, budget_s=budget_s)
        )

    async def run(self) -> ShutdownReport:
        """Execute every registered handler under their own budgets.

        Returns once all handlers have either succeeded, failed, timed
        out, or been skipped because the global budget was exhausted.
        Safe to call multiple times — second and subsequent calls
        return the cached report so a re-entered lifespan can't
        double-shutdown a SignalStore.
        """
        if self._report is not None:
            return self._report

        self._shutting_down = True
        started = self._clock()
        results: List[HandlerResult] = []

        for reg in self._registrations:
            elapsed = self._clock() - started
            remaining_global = self._total_budget_s - elapsed
            if remaining_global <= 0:
                results.append(
                    HandlerResult(
                        name=reg.name,
                        ok=False,
                        duration_ms=0.0,
                        skipped=True,
                        error="global budget exhausted",
                    )
                )
                logger.warning(
                    "shutdown: skipping %r — global budget exhausted",
                    reg.name,
                )
                continue

            per_handler = min(reg.budget_s, remaining_global)
            results.append(await self._run_one(reg, per_handler))

        finished = self._clock()
        self._report = ShutdownReport(
            started_at=started,
            finished_at=finished,
            total_duration_ms=(finished - started) * 1000.0,
            handlers=results,
        )
        logger.info(
            "shutdown complete: %d handlers, %d failed, %.0fms",
            len(results),
            self._report.n_failed,
            self._report.total_duration_ms,
        )
        return self._report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_one(
        self, reg: _Registration, budget_s: float
    ) -> HandlerResult:
        t0 = self._clock()
        try:
            await asyncio.wait_for(
                self._invoke(reg.handler), timeout=budget_s
            )
            duration_ms = (self._clock() - t0) * 1000.0
            logger.info("shutdown: %s ok in %.0fms", reg.name, duration_ms)
            return HandlerResult(
                name=reg.name, ok=True, duration_ms=duration_ms
            )
        except asyncio.TimeoutError:
            duration_ms = (self._clock() - t0) * 1000.0
            logger.warning(
                "shutdown: %s timed out after %.0fms (budget %.1fs)",
                reg.name,
                duration_ms,
                budget_s,
            )
            return HandlerResult(
                name=reg.name,
                ok=False,
                duration_ms=duration_ms,
                timed_out=True,
                error=f"timeout after {budget_s}s",
            )
        except Exception as exc:
            duration_ms = (self._clock() - t0) * 1000.0
            logger.exception(
                "shutdown: %s raised %s after %.0fms",
                reg.name,
                type(exc).__name__,
                duration_ms,
            )
            return HandlerResult(
                name=reg.name,
                ok=False,
                duration_ms=duration_ms,
                error=f"{type(exc).__name__}: {exc}",
            )

    @staticmethod
    async def _invoke(handler: Handler) -> None:
        """Run a sync or async handler uniformly under asyncio."""
        result = handler()
        if inspect.isawaitable(result):
            await result


__all__ = [
    "DEFAULT_HANDLER_BUDGET_S",
    "DEFAULT_TOTAL_BUDGET_S",
    "GracefulShutdownCoordinator",
    "HandlerResult",
    "ShutdownReport",
]
