"""Per-endpoint latency metrics — Sprint OBS-2B.4.

A read-only JSON view over the ``LatencyTracker`` ring buffer that the
access-log middleware feeds on every ``/api/v1/*`` request. The legacy
``/metrics`` endpoint exposes Prometheus-style histograms, but two
realities pushed us to add a JSON surface:

1. ``MetricsRegistry`` is optional and absent on most dev deployments;
2. dashboards (Sentry, our own webapp, Telegram on-call bot) prefer to
   poll a small JSON shape rather than parse a Prom exposition.

Exposing this under ``/api/v1/metrics/latency`` keeps it inside the
versioned, authenticated B2B surface. Admin-HMAC gated so it doesn't
leak per-route timings (which can hint at internal architecture) to
unauthenticated traffic.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from src.api.auth import require_admin

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/metrics", tags=["metrics"])


def _get_tracker(request: Request) -> Any:
    tracker = getattr(request.app.state.app_state, "latency_tracker", None)
    if tracker is None:
        raise HTTPException(
            status_code=503, detail="Latency tracker not configured"
        )
    return tracker


@router.get(
    "/error-budget",
    responses={
        503: {"description": "Error budget watcher not configured"},
    },
)
async def get_error_budget(
    request: Request,
    _: bool = Depends(require_admin),
):
    """Returns currently-firing SLO breaches + last 50 fire/clear events.

    The watcher (OBS-2B.5) polls the LatencyTracker every 30s and
    fires alerts via its sink. This endpoint surfaces *state* — what's
    breaching right now — so an on-call dashboard can render it without
    parsing the alert log.
    """
    watcher = getattr(request.app.state.app_state, "error_budget_watcher", None)
    if watcher is None:
        raise HTTPException(
            status_code=503, detail="Error budget watcher not configured"
        )
    return {
        "firing": watcher.firing_routes(),
        "recent_events": watcher.recent_events(limit=50),
        "running": watcher.is_running,
    }


@router.get(
    "/latency",
    responses={
        503: {"description": "Latency tracker not configured"},
    },
)
async def get_latency_metrics(
    request: Request,
    _: bool = Depends(require_admin),
):
    """Returns per-route rolling-window stats + global summary.

    The window is the tracker's configured ``window_seconds`` (default
    5 minutes). The ``totals`` block carries lifetime counters that
    survive eviction so an SLO computed off this endpoint (e.g.
    "error_rate_24h < 1%") doesn't collapse to 0 just because the
    rolling window emptied.
    """
    tracker = _get_tracker(request)
    snapshots = tracker.snapshot_all()

    total_count = sum(s.count for s in snapshots)
    total_errors = sum(s.error_count for s in snapshots)
    total_lifetime = sum(s.count_total for s in snapshots)
    total_errors_lifetime = sum(s.error_count_total for s in snapshots)

    return {
        "window_seconds": tracker._window,  # type: ignore[attr-defined]
        "routes": [
            {
                "path": s.path,
                "count": s.count,
                "p50_ms": s.p50_ms,
                "p95_ms": s.p95_ms,
                "p99_ms": s.p99_ms,
                "max_ms": s.max_ms,
                "error_rate": s.error_rate,
                "error_count": s.error_count,
                "count_total": s.count_total,
                "error_count_total": s.error_count_total,
            }
            for s in snapshots
        ],
        "totals": {
            "count_window": total_count,
            "error_count_window": total_errors,
            "error_rate_window": (
                round(total_errors / total_count, 4) if total_count else 0.0
            ),
            "count_lifetime": total_lifetime,
            "error_count_lifetime": total_errors_lifetime,
            "error_rate_lifetime": (
                round(total_errors_lifetime / total_lifetime, 4)
                if total_lifetime
                else 0.0
            ),
        },
        "routes_tracked": tracker.routes_tracked,
    }
