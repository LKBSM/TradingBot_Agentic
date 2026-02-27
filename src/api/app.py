"""FastAPI application factory."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import AppState
from src.api.models import ErrorResponse
from src.api.routes import admin, dashboard, health, operator, prometheus, signals
from src.api.signal_store import SignalStore

logger = logging.getLogger(__name__)


def create_app(
    signal_store: Optional[SignalStore] = None,
    metrics_registry: Any = None,
    health_monitor: Any = None,
    kill_switch: Any = None,
    var_engine: Any = None,
    live_risk_manager: Any = None,
    key_store: Any = None,
    hmac_manager: Any = None,
    signal_tracker: Any = None,
) -> FastAPI:
    """
    Build and return a fully-configured FastAPI application.

    All subsystem references are optional — the API returns safe
    defaults when a subsystem is ``None``.
    """
    if signal_store is None:
        signal_store = SignalStore(db_path="./data/signals.db")

    app_state = AppState(
        signal_store=signal_store,
        metrics_registry=metrics_registry,
        health_monitor=health_monitor,
        kill_switch=kill_switch,
        var_engine=var_engine,
        live_risk_manager=live_risk_manager,
        key_store=key_store,
        hmac_manager=hmac_manager,
        signal_tracker=signal_tracker,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("API starting up")
        yield
        logger.info("API shutting down")

    app = FastAPI(
        title="Trading Bot Signal API",
        version="0.11.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Store app_state so routes can access it via request.app.state
    app.state.app_state = app_state

    # ── CORS ──────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost", "http://localhost:3000"],
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ── Request logging middleware ────────────────────────────────────────
    @app.middleware("http")
    async def request_logging(request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        latency = time.time() - start
        logger.debug(
            "%s %s %s %.3fs",
            request.method,
            request.url.path,
            response.status_code,
            latency,
        )
        # Record to metrics histogram if available
        registry = app_state.metrics_registry
        if registry is not None:
            try:
                registry.histogram(
                    "http_request_duration_seconds",
                    "HTTP request latency",
                ).observe(latency, labels={"path": request.url.path})
            except Exception:
                pass
        return response

    # ── Global exception handler ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_server_error",
                detail=str(exc),
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(signals.router)
    app.include_router(health.router)
    app.include_router(operator.router)
    app.include_router(prometheus.router)
    app.include_router(admin.router)
    app.include_router(dashboard.router)

    return app
