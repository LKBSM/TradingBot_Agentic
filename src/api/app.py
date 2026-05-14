"""FastAPI application factory."""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from src.api.dependencies import AppState
from src.api.latency_tracker import LatencyTracker
from src.api.middleware.access_log import StructuredAccessLogMiddleware
from src.api.middleware.geo_block import GeoBlockMiddleware
from src.api.models import ErrorResponse
from src.api.routes import admin, admin_audit, audit, dashboard, enrich, health, health_deep, insight_history, legal, metrics_latency, narratives, operator, prometheus, qa, signals, state, webapp
from src.api.shutdown import GracefulShutdownCoordinator
from src.api.signal_store import SignalStore

logger = logging.getLogger(__name__)


def _auto_register_default_handlers(
    coord: GracefulShutdownCoordinator, app_state: AppState
) -> None:
    """Wire each shutdown-capable subsystem on app_state into the coordinator.

    Called from the lifespan after every subsystem is in place. Only
    registers a handler when (a) the subsystem exists, and (b) no
    handler with the same name was already registered (so a caller can
    override with custom budgets).

    Shutdown order is deliberately:

    1. **Pollers/workers first** — sentinel scanner, webhook drain
       worker. They must stop *before* their downstream stores close
       so a final cycle doesn't write into a closed handle.
    2. **Stores last** — audit ledger, admin action log, SignalStore.
       Close after every writer has stopped so the WAL flushes cleanly.

    All registrations are no-ops if the subsystem is None.
    """
    already = {r.name for r in coord._registrations}  # noqa: SLF001

    def _add(name: str, fn, *, budget_s: float) -> None:
        if name in already:
            return
        coord.register(name, fn, budget_s=budget_s)

    # Pollers / workers
    scanner = app_state.scanner
    if scanner is not None and hasattr(scanner, "stop"):
        _add("sentinel-scanner", scanner.stop, budget_s=5.0)

    # The drain worker isn't stored on AppState today — callers that
    # wire it call coord.register("webhook-drain", worker.stop) before
    # startup. Nothing auto here.

    # Stores
    ledger = app_state.audit_ledger
    if ledger is not None and hasattr(ledger, "close"):
        _add("audit-ledger", ledger.close, budget_s=2.0)

    admin_log = app_state.admin_action_log
    if admin_log is not None and hasattr(admin_log, "close"):
        _add("admin-action-log", admin_log.close, budget_s=2.0)

    signal_store = app_state.signal_store
    if signal_store is not None and hasattr(signal_store, "close"):
        _add("signal-store", signal_store.close, budget_s=2.0)


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
    operational_kill_switch: Any = None,
    rag_pipeline: Any = None,
    rag_llm: Any = None,
    audit_ledger: Any = None,
    cost_quota: Any = None,
    tier_rate_limiter: Any = None,
    webhook_queue: Any = None,
    embedder: Any = None,
    idempotency_store: Any = None,
    admin_action_log: Any = None,
    latency_tracker: Any = None,
    shutdown_coordinator: Any = None,
) -> FastAPI:
    """
    Build and return a fully-configured FastAPI application.

    All subsystem references are optional — the API returns safe
    defaults when a subsystem is ``None``.
    """
    if signal_store is None:
        signal_store = SignalStore(db_path="./data/signals.db")
    # Latency tracker is cheap (~1MB max @ default caps) and always
    # useful; default-instantiate so /api/v1/metrics/latency is live
    # even when the caller didn't wire one in.
    if latency_tracker is None:
        latency_tracker = LatencyTracker()
    if shutdown_coordinator is None:
        shutdown_coordinator = GracefulShutdownCoordinator()

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
        operational_kill_switch=operational_kill_switch,
        rag_pipeline=rag_pipeline,
        rag_llm=rag_llm,
        audit_ledger=audit_ledger,
        cost_quota=cost_quota,
        tier_rate_limiter=tier_rate_limiter,
        webhook_queue=webhook_queue,
        embedder=embedder,
        idempotency_store=idempotency_store,
        admin_action_log=admin_action_log,
        latency_tracker=latency_tracker,
        shutdown_coordinator=shutdown_coordinator,
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("API starting up")
        # INFRA-2B.11: auto-register any wired subsystems that expose a
        # shutdown surface. Caller can register more *before* startup —
        # we only add the ones not already registered by name. Order
        # matters: workers stop first, ledgers/stores close last so
        # in-flight log writes complete.
        coord: GracefulShutdownCoordinator = app_state.shutdown_coordinator
        _auto_register_default_handlers(coord, app_state)
        try:
            yield
        finally:
            logger.info("API shutting down — running %d handlers", len(coord._registrations))  # noqa: SLF001
            try:
                await coord.run()
            except Exception:
                logger.exception("shutdown coordinator failed")

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

    # ── Gzip compression (≥1KB) — −60-70% bandwidth /metrics, /equity-curve
    app.add_middleware(GZipMiddleware, minimum_size=1024)

    # ── Structured access log — outermost, so it sees final status codes
    # and total latency through every other middleware (OBS-2B.3).
    # Latency tracker (OBS-2B.4) lives inside the middleware so each
    # request feeds the rolling p50/p95/p99 window.
    app.add_middleware(
        StructuredAccessLogMiddleware, latency_tracker=latency_tracker
    )

    # ── Geo-block (US/QC/UK + OFAC SDN) — P29 compliance ──────────────────
    app.add_middleware(GeoBlockMiddleware)

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

    # ── Security headers (HSTS, CSP, X-Frame, X-Content-Type, Referrer) ──
    # Defence-in-depth against clickjacking, MIME sniffing, downgrade attacks
    # and over-eager referrers. Eval 15 finding #1 (no security headers).
    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=63072000; includeSubDomains",
        )
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        # Tight CSP — JSON API has no inline scripts/styles. /api/docs Swagger
        # UI loads from cdn.jsdelivr.net + fastapi.tiangolo.com favicon.
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; "
            "script-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "style-src 'self' https://cdn.jsdelivr.net 'unsafe-inline'; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "connect-src 'self'; "
            "frame-ancestors 'none'",
        )
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        return response

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
    app.include_router(health_deep.router)
    app.include_router(operator.router)
    app.include_router(prometheus.router)
    app.include_router(admin.router)
    app.include_router(admin_audit.router)
    app.include_router(dashboard.router)
    app.include_router(narratives.router)
    app.include_router(legal.router)
    app.include_router(qa.router)
    app.include_router(enrich.router)
    app.include_router(audit.router)
    app.include_router(insight_history.router)
    app.include_router(metrics_latency.router)
    app.include_router(webapp.router)

    return app
