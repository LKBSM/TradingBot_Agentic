"""Error-budget alert watcher — Sprint OBS-2B.5.

Background poller that reads :class:`LatencyTracker` snapshots and
fires structured alerts when a per-route SLO is breached. The watcher
is meant to wake a webhook / on-call dashboard *before* a slow
endpoint takes a customer down, not after.

SLO model
---------
Each route has two thresholds:

- ``p95_ms_warn`` — soft cap. Crossing it once is noise; crossing it
  in N consecutive snapshots fires a ``warn``.
- ``error_rate_warn`` — rolling-window error-rate cap (e.g. 0.05).
  Same N-consecutive logic.

Both thresholds are per-route so noisy debug endpoints don't pollute
the production /enrich SLOs. A built-in DEFAULT_SLOS table covers the
critical Phase 2B surfaces; the rest of the routes fall back to a
relaxed ``unknown`` SLO.

State machine
-------------
Each route tracks ``consecutive_breaches`` counters; an alert *fires*
exactly when the counter reaches ``threshold_count`` (default 3) and
*clears* on the first snapshot that satisfies the SLO again. That
hysteresis keeps a noisy on-call channel from filling with single-
snapshot blips.

Why not Prometheus alertmanager
-------------------------------
Same reason as OBS-2B.4 (the tracker itself): we don't have a
Prometheus side-car on most deployments, and the alerting state has
to live in-process for the watcher to honour the rolling window's
exact semantics. When/if we add Prom, this watcher emits the same
structured events to a metrics gauge and alertmanager handles routing.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


DEFAULT_THRESHOLD_COUNT = 3
DEFAULT_POLL_SECONDS = 30.0


@dataclass(frozen=True)
class SLOSpec:
    """Per-route SLO."""

    p95_ms_warn: float
    error_rate_warn: float


DEFAULT_SLOS: Dict[str, SLOSpec] = {
    # Hot path — broker-facing insight enrichment. p95 < 800ms, errors
    # < 1% are the published numbers in the pricing grid.
    "/api/v1/enrich": SLOSpec(p95_ms_warn=800.0, error_rate_warn=0.01),
    # Insight lookup — should be near-free (audit ledger hit + JSON
    # serialise).
    "/api/v1/insights/{insight_id}": SLOSpec(
        p95_ms_warn=200.0, error_rate_warn=0.005
    ),
    "/api/v1/insights/history": SLOSpec(
        p95_ms_warn=400.0, error_rate_warn=0.01
    ),
    # Audit endpoints (sequential SQLite reads).
    "/api/v1/audit/entry/{seq}": SLOSpec(
        p95_ms_warn=150.0, error_rate_warn=0.005
    ),
    # Webhook ack — should be effectively instant.
    "/api/v1/webhooks/deliveries/{delivery_id}/ack": SLOSpec(
        p95_ms_warn=100.0, error_rate_warn=0.005
    ),
}

UNKNOWN_SLO = SLOSpec(p95_ms_warn=2000.0, error_rate_warn=0.05)


@dataclass
class AlertEvent:
    """One structured alert event. Emitted as JSON to the watcher's logger."""

    evt: str  # "alert_fire" | "alert_clear"
    route: str
    metric: str  # "p95_ms" | "error_rate"
    observed: float
    threshold: float
    consecutive: int
    ts_utc: str

    def to_dict(self) -> dict:
        return {
            "evt": self.evt,
            "route": self.route,
            "metric": self.metric,
            "observed": self.observed,
            "threshold": self.threshold,
            "consecutive": self.consecutive,
            "ts_utc": self.ts_utc,
        }


@dataclass
class _RouteState:
    p95_breaches: int = 0
    error_breaches: int = 0
    p95_firing: bool = False
    error_firing: bool = False


class ErrorBudgetWatcher:
    """Polls a LatencyTracker, fires alerts via a sink callable.

    Parameters
    ----------
    latency_tracker:
        Required. The ``LatencyTracker`` from OBS-2B.4.
    sink:
        Callable ``(AlertEvent) -> None`` invoked on every fire/clear.
        Default appends to the watcher's logger; production wires the
        Telegram on-call bot or Sentry capture.
    slos:
        Per-route ``SLOSpec`` overrides. Missing routes fall back to
        :data:`UNKNOWN_SLO`.
    threshold_count:
        Number of consecutive breach-snapshots required to fire. Lower
        for "the customer is screaming" testing; higher to suppress
        flap on a noisy endpoint.
    """

    def __init__(
        self,
        latency_tracker: Any,
        *,
        sink: Optional[Callable[[AlertEvent], None]] = None,
        slos: Optional[Dict[str, SLOSpec]] = None,
        threshold_count: int = DEFAULT_THRESHOLD_COUNT,
        poll_seconds: float = DEFAULT_POLL_SECONDS,
        clock: Callable[[], float] = time.time,
    ):
        if latency_tracker is None:
            raise ValueError("latency_tracker is required")
        if threshold_count < 1:
            raise ValueError("threshold_count must be >= 1")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        self._tracker = latency_tracker
        self._sink = sink or self._default_sink
        self._slos = dict(slos) if slos is not None else dict(DEFAULT_SLOS)
        self._threshold_count = threshold_count
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._state: Dict[str, _RouteState] = {}

        # Async loop bookkeeping
        self._stop_event: Optional[asyncio.Event] = None
        self._task: Optional[asyncio.Task] = None
        self._fired_events: List[AlertEvent] = []  # last 100 retained

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
            self._run(), name="error-budget-watcher"
        )

    async def stop(self) -> None:
        if not self.is_running:
            return
        assert self._stop_event is not None
        self._stop_event.set()
        try:
            await self._task  # type: ignore[arg-type]
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Core poll
    # ------------------------------------------------------------------

    def evaluate_once(self) -> List[AlertEvent]:
        """Single non-async pass over snapshot_all(). Useful in tests
        and for synchronous callers (the deep-health endpoint can call
        this without spinning up an async task).
        """
        events: List[AlertEvent] = []
        snapshots = self._tracker.snapshot_all()
        # Routes that disappear from the tracker (e.g. evicted under
        # cardinality cap) keep their state — when they come back the
        # counter resumes; we don't get a spurious clear.
        for snap in snapshots:
            slo = self._slos.get(snap.path, UNKNOWN_SLO)
            state = self._state.setdefault(snap.path, _RouteState())

            # Skip routes with no observations in the window — empty
            # buckets shouldn't satisfy or breach the SLO; we want
            # *signal*, not "we haven't seen traffic recently".
            if snap.count == 0:
                continue

            # ── p95 ────────────────────────────────────────────────
            if snap.p95_ms > slo.p95_ms_warn:
                state.p95_breaches += 1
                if (
                    state.p95_breaches >= self._threshold_count
                    and not state.p95_firing
                ):
                    state.p95_firing = True
                    events.append(
                        self._make_event(
                            "alert_fire", snap.path, "p95_ms",
                            snap.p95_ms, slo.p95_ms_warn,
                            state.p95_breaches,
                        )
                    )
            else:
                if state.p95_firing:
                    events.append(
                        self._make_event(
                            "alert_clear", snap.path, "p95_ms",
                            snap.p95_ms, slo.p95_ms_warn,
                            state.p95_breaches,
                        )
                    )
                state.p95_breaches = 0
                state.p95_firing = False

            # ── error rate ─────────────────────────────────────────
            if snap.error_rate > slo.error_rate_warn:
                state.error_breaches += 1
                if (
                    state.error_breaches >= self._threshold_count
                    and not state.error_firing
                ):
                    state.error_firing = True
                    events.append(
                        self._make_event(
                            "alert_fire", snap.path, "error_rate",
                            snap.error_rate, slo.error_rate_warn,
                            state.error_breaches,
                        )
                    )
            else:
                if state.error_firing:
                    events.append(
                        self._make_event(
                            "alert_clear", snap.path, "error_rate",
                            snap.error_rate, slo.error_rate_warn,
                            state.error_breaches,
                        )
                    )
                state.error_breaches = 0
                state.error_firing = False

        for ev in events:
            self._fired_events.append(ev)
            try:
                self._sink(ev)
            except Exception:  # pragma: no cover — sink failure ≠ crash
                logger.exception("alert sink raised on %s", ev.to_dict())
        # Cap the in-memory event ring so /metrics doesn't grow forever.
        if len(self._fired_events) > 100:
            del self._fired_events[: len(self._fired_events) - 100]
        return events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _make_event(
        self,
        evt: str,
        route: str,
        metric: str,
        observed: float,
        threshold: float,
        consecutive: int,
    ) -> AlertEvent:
        return AlertEvent(
            evt=evt,
            route=route,
            metric=metric,
            observed=round(observed, 4),
            threshold=threshold,
            consecutive=consecutive,
            ts_utc=time.strftime(
                "%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(self._clock())
            ),
        )

    @staticmethod
    def _default_sink(event: AlertEvent) -> None:
        import json
        logger.warning(json.dumps({"alert": event.to_dict()}, ensure_ascii=False))

    async def _run(self) -> None:
        assert self._stop_event is not None
        logger.info(
            "error budget watcher started (poll=%.1fs, routes=%d, "
            "threshold_count=%d)",
            self._poll_seconds,
            len(self._slos),
            self._threshold_count,
        )
        try:
            while not self._stop_event.is_set():
                try:
                    self.evaluate_once()
                except Exception:
                    logger.exception("error budget poll raised")
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._poll_seconds
                    )
                    break  # stop was signalled
                except asyncio.TimeoutError:
                    continue
        finally:
            logger.info("error budget watcher stopped")

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def firing_routes(self) -> List[dict]:
        """Routes currently in a firing state for at least one metric."""
        out = []
        for route, st in self._state.items():
            if st.p95_firing or st.error_firing:
                out.append({
                    "route": route,
                    "p95_firing": st.p95_firing,
                    "error_firing": st.error_firing,
                    "p95_consecutive": st.p95_breaches,
                    "error_consecutive": st.error_breaches,
                })
        return out

    def recent_events(self, limit: int = 50) -> List[dict]:
        return [ev.to_dict() for ev in self._fired_events[-limit:]]


__all__ = [
    "AlertEvent",
    "DEFAULT_POLL_SECONDS",
    "DEFAULT_SLOS",
    "DEFAULT_THRESHOLD_COUNT",
    "ErrorBudgetWatcher",
    "SLOSpec",
    "UNKNOWN_SLO",
]
