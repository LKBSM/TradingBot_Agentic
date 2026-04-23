"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import AppState
from src.api.models import ErrorResponse
from src.api.routes import admin, dashboard, health, narratives, operator, prometheus, signals, state
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
    tier_manager: Any = None,
    llm_engine: Any = None,
    scanner: Any = None,
    circuit_breakers: Optional[Dict[str, Any]] = None,
    health_checker: Any = None,
    rate_limiter: Any = None,
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
        tier_manager=tier_manager,
        llm_engine=llm_engine,
        scanner=scanner,
        circuit_breakers=circuit_breakers or {},
        health_checker=health_checker,
        rate_limiter=rate_limiter,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("API starting up")
        yield
        logger.info("API shutting down")

    app = FastAPI(
        title="Smart Sentinel AI",
        version="1.0.0",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )

    # Store app_state so routes can access it via request.app.state
    app.state.app_state = app_state

    # ── CORS (configurable from env) ──────────────────────────────────────
    cors_origins_str = os.environ.get(
        "CORS_ALLOWED_ORIGINS",
        "http://localhost,http://localhost:3000",
    )
    cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    # ── Request size limit middleware (1 MB) ──────────────────────────────
    MAX_BODY_SIZE = 1_048_576  # 1 MB

    @app.middleware("http")
    async def request_size_limit(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > MAX_BODY_SIZE:
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "detail": "Max body size is 1 MB"},
            )
        return await call_next(request)

    # ── Per-IP rate limiter middleware ─────────────────────────────────────
    @app.middleware("http")
    async def rate_limit_middleware(request: Request, call_next):
        if rate_limiter is not None:
            client_ip = request.client.host if request.client else "unknown"
            if not rate_limiter.allow(client_ip):
                remaining = rate_limiter.remaining(client_ip)
                return JSONResponse(
                    status_code=429,
                    content={
                        "error": "rate_limit_exceeded",
                        "detail": "Too many requests. Please slow down.",
                    },
                    headers={"Retry-After": "60", "X-RateLimit-Remaining": str(remaining)},
                )
        return await call_next(request)

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
                detail="Internal server error",
            ).model_dump(),
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(signals.router)
    app.include_router(state.router)
    app.include_router(health.router)
    app.include_router(operator.router)
    app.include_router(prometheus.router)
    app.include_router(admin.router)
    app.include_router(dashboard.router)
    app.include_router(narratives.router)

    return app
