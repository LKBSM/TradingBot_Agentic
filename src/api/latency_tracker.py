"""Per-endpoint rolling-window latency tracker — Sprint OBS-2B.4.

Why a dedicated tracker
-----------------------
Existing app.py wires a Prometheus-style histogram via the metrics
registry, but two things make that insufficient on its own:

1. ``MetricsRegistry`` is optional in our deployments — when it isn't
   wired in (most local/dev runs) we lose latency observability
   completely;
2. p50/p95/p99 from histograms requires PromQL machinery on the
   downstream side. For a SaaS at our scale, an *in-process* rolling
   window with computed percentiles is cheaper, always available, and
   directly exposable as JSON (``GET /api/v1/metrics/latency``).

This module implements a small, ring-buffer-backed tracker that's
allocation-cheap (~1MB of doubles for 5min @ 1k req/min × 5 routes =
5×5000 floats), thread-safe, and percentile-correct (no approximation
sketches needed at this scale).

Window semantics
----------------
Per-route ``deque`` of ``(timestamp, latency_ms, status)`` triples.
On every read we lazily purge anything older than ``window_seconds``
(default 300 = 5min). A separate per-route counter tracks total
observations + total non-2xx responses since process start so the
``count_total`` and ``error_rate_total`` stats survive eviction.

Path normalisation
------------------
We bucket routes by the registered FastAPI path (with curly braces)
when available, falling back to the raw URL path. ``/api/v1/insights/
abc-123`` and ``/api/v1/insights/xyz-456`` both end up under
``/api/v1/insights/{insight_id}`` — otherwise high-cardinality ids
would explode the per-route table.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Optional


DEFAULT_WINDOW_SECONDS = 300.0
DEFAULT_MAX_ROUTES = 200      # bounds memory for a misbehaving caller
DEFAULT_PER_ROUTE_CAP = 5000  # ~16 req/s per route over a 5-min window


@dataclass(frozen=True)
class LatencySnapshot:
    """Rolling-window stats for one route."""

    path: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    error_rate: float       # rolling — non-2xx / count
    error_count: int
    count_total: int        # since process start
    error_count_total: int  # since process start


class LatencyTracker:
    """Thread-safe rolling latency tracker.

    Public surface:
      - ``record(path, latency_ms, status)``
      - ``snapshot(path) -> LatencySnapshot``
      - ``snapshot_all() -> list[LatencySnapshot]``
      - ``reset()``
    """

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        max_routes: int = DEFAULT_MAX_ROUTES,
        per_route_cap: int = DEFAULT_PER_ROUTE_CAP,
        clock=time.monotonic,
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if per_route_cap < 1:
            raise ValueError("per_route_cap must be >= 1")
        self._window = window_seconds
        self._max_routes = max_routes
        self._per_route_cap = per_route_cap
        self._clock = clock
        self._lock = threading.Lock()
        # path → deque[(ts, latency_ms, status)]
        self._buffers: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=per_route_cap)
        )
        self._count_total: dict[str, int] = defaultdict(int)
        self._error_total: dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _purge(self, buf: deque, now: float) -> None:
        cutoff = now - self._window
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    @staticmethod
    def _percentile(sorted_values: list[float], q: float) -> float:
        """Linear-interpolation percentile on a *sorted* list. q in [0, 1]."""
        if not sorted_values:
            return 0.0
        if q <= 0:
            return sorted_values[0]
        if q >= 1:
            return sorted_values[-1]
        idx = q * (len(sorted_values) - 1)
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return sorted_values[lo]
        return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (idx - lo)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, path: str, latency_ms: float, status: int) -> None:
        if not path:
            return
        if latency_ms < 0:
            return
        with self._lock:
            # Cardinality cap: drop new routes once we hit the ceiling
            # rather than unbounded growth.
            if (
                path not in self._buffers
                and len(self._buffers) >= self._max_routes
            ):
                return
            self._buffers[path].append((self._clock(), latency_ms, status))
            self._count_total[path] += 1
            if status >= 400:
                self._error_total[path] += 1

    def snapshot(self, path: str) -> LatencySnapshot:
        with self._lock:
            buf = self._buffers.get(path)
            count_total = self._count_total.get(path, 0)
            error_total = self._error_total.get(path, 0)
            if not buf:
                return LatencySnapshot(
                    path=path,
                    count=0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    max_ms=0.0,
                    error_rate=0.0,
                    error_count=0,
                    count_total=count_total,
                    error_count_total=error_total,
                )
            now = self._clock()
            self._purge(buf, now)
            if not buf:
                return LatencySnapshot(
                    path=path,
                    count=0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    max_ms=0.0,
                    error_rate=0.0,
                    error_count=0,
                    count_total=count_total,
                    error_count_total=error_total,
                )
            latencies = sorted(item[1] for item in buf)
            errors = sum(1 for item in buf if item[2] >= 400)
            count = len(buf)
            return LatencySnapshot(
                path=path,
                count=count,
                p50_ms=round(self._percentile(latencies, 0.50), 2),
                p95_ms=round(self._percentile(latencies, 0.95), 2),
                p99_ms=round(self._percentile(latencies, 0.99), 2),
                max_ms=round(latencies[-1], 2),
                error_rate=round(errors / count, 4),
                error_count=errors,
                count_total=count_total,
                error_count_total=error_total,
            )

    def snapshot_all(self) -> list[LatencySnapshot]:
        with self._lock:
            paths = list(self._buffers.keys())
        # Drop the lock before computing percentiles for each path; each
        # snapshot() call re-locks briefly. This keeps the critical
        # section short under read-heavy workloads.
        return [self.snapshot(p) for p in sorted(paths)]

    def reset(self) -> None:
        with self._lock:
            self._buffers.clear()
            self._count_total.clear()
            self._error_total.clear()

    @property
    def routes_tracked(self) -> int:
        with self._lock:
            return len(self._buffers)


__all__ = [
    "DEFAULT_MAX_ROUTES",
    "DEFAULT_PER_ROUTE_CAP",
    "DEFAULT_WINDOW_SECONDS",
    "LatencySnapshot",
    "LatencyTracker",
]
