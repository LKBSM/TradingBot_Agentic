"""Structured JSON access log middleware — Sprint OBS-2B.3.

Why a dedicated middleware
--------------------------
The default request_logging middleware in ``src/api/app.py`` writes a
debug line and feeds the metrics histogram. That's fine for *latency*,
but it loses every other dimension that matters in production:

- which subscriber made the call? (tier + last 4 chars of api_key_id)
- what was the response status?
- which exact path? (raw URL, before route templating)
- which client IP? (already used by per-IP rate limiter — surface it
  here too so abuse triage doesn't require correlating two log streams)
- what's the request_id? (so a single log line tied to a 500 has the
  same id the client received in the response)

This middleware emits one JSON-encoded line per /api/v1/* request. The
JSON shape is intentionally flat — no nesting — so a downstream parser
(Vector / Loki / DataDog) can index every field without schema gymnastics.

Request id
----------
We respect an inbound ``X-Request-Id`` if it looks safe (alphanumeric +
``-_``, ≤ 64 chars), otherwise we mint a fresh ``urandom`` token. The id
is attached to the response as ``X-Request-Id`` and to the log line as
``request_id``.

PII / secrets
-------------
We never log the raw API key, tokens, or request body. ``api_key_id`` is
truncated to its last 4 chars (subscriber identification without the
full hash). Client IPs are logged — they're already in lower layers
(rate limiter, geo-block) and required for abuse triage; suppress
upstream of the SaaS if the deployment legally needs IP minimisation.
"""

from __future__ import annotations

import json
import logging
import os
import re
import secrets
import time
from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("smart_sentinel.access")


REQUEST_ID_HEADER = "X-Request-Id"
_REQUEST_ID_MAX = 64
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")
DEFAULT_LOG_PREFIX = "/api/v1"


def _safe_inbound_request_id(value: Optional[str]) -> Optional[str]:
    """Sanitise an inbound X-Request-Id header.

    Returning ``None`` for invalid shapes lets the caller mint a fresh
    one. We don't propagate weird strings into log files.
    """
    if not value:
        return None
    if len(value) > _REQUEST_ID_MAX:
        return None
    if not _REQUEST_ID_RE.match(value):
        return None
    return value


def _new_request_id() -> str:
    """Short, URL-safe, unguessable. 16 hex chars = 64 bits of entropy."""
    return secrets.token_hex(8)


def _last4(value: Any) -> str:
    """Last 4 chars of a key-id-ish value, for logging without leaking."""
    s = str(value or "")
    return s[-4:] if len(s) >= 4 else s


class StructuredAccessLogMiddleware(BaseHTTPMiddleware):
    """Emits one JSON access-log entry per matching request.

    Parameters
    ----------
    log_prefix:
        Only paths starting with this prefix are logged. Defaults to
        ``/api/v1`` so probes / docs / static assets don't drown the
        signal.
    logger:
        The logging.Logger to write to. Production wires the
        already-installed JSONFormatter via ``LOG_FORMAT=json``;
        tests pass a custom logger so they can capture lines.
    """

    def __init__(
        self,
        app: Any,
        *,
        log_prefix: str = DEFAULT_LOG_PREFIX,
        access_logger: Optional[logging.Logger] = None,
        latency_tracker: Optional[Any] = None,
    ):
        super().__init__(app)
        self._log_prefix = log_prefix
        self._logger = access_logger or logger
        # Optional rolling-window latency tracker (OBS-2B.4). When wired,
        # every request feeds the tracker so /api/v1/metrics/latency can
        # serve p50/p95/p99 without needing a Prometheus side-car.
        self._latency_tracker = latency_tracker

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        path = request.url.path
        if not path.startswith(self._log_prefix):
            return await call_next(request)

        request_id = (
            _safe_inbound_request_id(request.headers.get(REQUEST_ID_HEADER))
            or _new_request_id()
        )
        # Make the id visible to downstream handlers (e.g. log lines they
        # emit themselves can pick it up via request.state.request_id).
        request.state.request_id = request_id

        start = time.perf_counter()
        status = 500
        response: Optional[Response] = None
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            latency_ms = round((time.perf_counter() - start) * 1000, 2)
            client_ip = request.client.host if request.client else "-"

            # Subscriber state is set by the auth dependency. When auth
            # raises (401/429), no subscriber exists — log "-" rather
            # than crashing.
            subscriber = getattr(request.state, "subscriber", None)
            tier = (subscriber or {}).get("tier", "-") if subscriber else "-"
            api_key_id_last4 = (
                _last4((subscriber or {}).get("key_id")) if subscriber else "-"
            )

            # Route template (e.g. ``/api/v1/insights/{insight_id}``) is
            # set on the scope by Starlette's Router during dispatch;
            # using it keeps high-cardinality ids out of the tracker.
            # Falls back to the raw path when no route matched (404).
            route = request.scope.get("route")
            route_path = getattr(route, "path", None) or path

            entry = {
                "evt": "http_access",
                "request_id": request_id,
                "method": request.method,
                "path": path,
                "route": route_path,
                "status": status,
                "latency_ms": latency_ms,
                "client_ip": client_ip,
                "tier": tier,
                "api_key_id_last4": api_key_id_last4,
                "user_agent": request.headers.get("user-agent", "-")[:120],
            }
            # JSON in the message body keeps log shippers happy even when
            # the formatter isn't JSON-aware (LOG_FORMAT=text deployments).
            self._logger.info(json.dumps(entry, ensure_ascii=False, default=str))

            # OBS-2B.4: feed the rolling-window latency tracker so
            # /api/v1/metrics/latency stays current without a separate
            # Prometheus side-car. Bucket by route template to keep
            # cardinality bounded under arbitrary path params.
            if self._latency_tracker is not None:
                try:
                    self._latency_tracker.record(
                        route_path, latency_ms, status
                    )
                except Exception:  # pragma: no cover — tracker bug ≠ user 500
                    logger.exception("latency_tracker.record failed")

            # Attach the id to the response for client correlation. When
            # the downstream raised, ``response`` may be None — Starlette
            # builds its own 500 in that case and we lose the header, but
            # the log line still carries the id.
            if response is not None:
                response.headers.setdefault(REQUEST_ID_HEADER, request_id)


def install_access_log(app: Any, **kwargs: Any) -> None:
    """Convenience hook: ``install_access_log(app)`` from ``create_app``.

    Kept as a free function so create_app can decide *when* to add the
    middleware (after CORS, before the request-size limit).
    """
    app.add_middleware(StructuredAccessLogMiddleware, **kwargs)


__all__ = [
    "DEFAULT_LOG_PREFIX",
    "REQUEST_ID_HEADER",
    "StructuredAccessLogMiddleware",
    "install_access_log",
]
