"""Prometheus scrape endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

router = APIRouter(tags=["prometheus"])


@router.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics(request: Request):
    """Prometheus-compatible text exposition."""
    registry = request.app.state.app_state.metrics_registry
    if registry is None:
        return PlainTextResponse("", media_type="text/plain; charset=utf-8")
    return PlainTextResponse(
        registry.to_prometheus(),
        media_type="text/plain; charset=utf-8",
    )
