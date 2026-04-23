"""Health-check endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Request

from src.api.auth import TESTING_MODE
from src.api.models import ComponentHealth, HealthResponse, SystemStatus

router = APIRouter(tags=["health"])

_BOOT_TIME = time.time()


async def _build_health(request: Request) -> HealthResponse:
    app_state = request.app.state.app_state

    status = SystemStatus.HEALTHY
    kill_switch_level = 0
    is_trading_active = True
    components = []

    # Health monitor (legacy)
    monitor = app_state.health_monitor
    if monitor is not None:
        try:
            agg = await monitor.check_all()
            status = SystemStatus(agg.status.value)
        except Exception:
            status = SystemStatus.UNKNOWN

    # Kill switch
    ks = app_state.kill_switch
    if ks is not None:
        try:
            kill_switch_level = ks.halt_level.value
            is_trading_active = ks.is_trading_allowed()
        except Exception:
            pass

    # Circuit breaker health
    health_checker = getattr(app_state, "health_checker", None)
    if health_checker is not None:
        try:
            result = health_checker.check_all()
            for name, healthy in result.get("checks", {}).items():
                components.append(ComponentHealth(name=name, healthy=healthy))
            if not result.get("healthy", True):
                status = SystemStatus.DEGRADED
        except Exception:
            pass

    # Scanner status
    scanner_running = False
    signals_generated = 0
    scanner = getattr(app_state, "scanner", None)
    if scanner is not None:
        try:
            stats = scanner.get_stats()
            scanner_running = stats.get("running", False)
            signals_generated = stats.get("signals_generated", 0)
        except Exception:
            pass

    return HealthResponse(
        status=status,
        uptime_seconds=round(time.time() - _BOOT_TIME, 2),
        kill_switch_level=kill_switch_level,
        is_trading_active=is_trading_active,
        testing_mode=TESTING_MODE,
        components=components,
        scanner_running=scanner_running,
        signals_generated=signals_generated,
    )


@router.get("/api/v1/health", response_model=HealthResponse)
async def health(request: Request):
    """Public health endpoint."""
    return await _build_health(request)


@router.get("/health", response_model=HealthResponse, include_in_schema=False)
async def health_docker(request: Request):
    """Docker HEALTHCHECK compat — same payload, hidden from OpenAPI."""
    return await _build_health(request)
