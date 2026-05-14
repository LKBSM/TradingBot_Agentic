"""Rate-limit response headers middleware — Sprint API-2B.4.

The TierRateLimiter (INFRA-2B.9) keeps per-tier sliding-window counters,
but until now it was *silent* — clients only learned they were near
their cap when a request actually got rejected with 429. Brokers need
to self-throttle before that point.

This middleware reads the limiter's snapshot for the authenticated
subscriber and adds standard headers on every successful response:

    X-RateLimit-Limit:      cap per minute for this tier
    X-RateLimit-Remaining:  how many calls left in the window
    X-RateLimit-Reset:      Unix epoch seconds when the window slides
    X-RateLimit-Policy:     "<cap>;w=60" (IETF draft-aligned)

Naming follows the IETF
``draft-ietf-httpapi-ratelimit-headers``: ``X-RateLimit-Limit`` for the
total cap, ``Remaining`` decrementing, ``Reset`` as an absolute
timestamp (not delta seconds — easier to interpret across clock skew).

Why a separate middleware
-------------------------
We already have a per-IP rate limiter in app.py and the auth dependency
attaches the subscriber to ``request.state``. Bundling header
production into auth would tightly couple read-only header emission
with the actual auth flow; bundling into the per-IP limiter would
expose per-tier caps to unauthenticated requests. A dedicated
middleware decoupled from both stays testable and easy to disable.

Scope
-----
Only ``/api/v1/*`` paths are decorated — health probes, /docs, /metrics
prom don't carry tier semantics. Unauthenticated requests are skipped
(no subscriber on ``request.state`` means no tier to gate against).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)


DEFAULT_PREFIX = "/api/v1"


class RateLimitHeadersMiddleware(BaseHTTPMiddleware):
    """Adds X-RateLimit-* headers to every authenticated /api/v1/* response.

    Parameters
    ----------
    tier_rate_limiter:
        Required — the ``TierRateLimiter`` instance. When ``None``, the
        middleware is a complete no-op (so we don't have to special-case
        the middleware list).
    path_prefix:
        Only paths starting with this prefix get decorated.
    window_seconds:
        The limiter's window — used in the ``X-RateLimit-Policy``
        header so clients know how to interpret the cap. Defaults to
        60s, matching ``TierRateLimiter`` defaults.
    """

    def __init__(
        self,
        app: Any,
        *,
        tier_rate_limiter: Optional[Any] = None,
        path_prefix: str = DEFAULT_PREFIX,
        window_seconds: float = 60.0,
    ):
        super().__init__(app)
        self._limiter = tier_rate_limiter
        self._prefix = path_prefix
        self._window = window_seconds

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        if self._limiter is None:
            return response
        path = request.url.path
        if not path.startswith(self._prefix):
            return response

        subscriber = getattr(request.state, "subscriber", None)
        if not subscriber:
            return response

        tier = subscriber.get("tier")
        api_key = subscriber.get("api_key") or subscriber.get("key_id")
        if not tier or api_key is None:
            return response

        try:
            snap = self._limiter.snapshot(str(api_key), tier)
        except Exception:  # pragma: no cover — limiter bug ≠ user 500
            logger.exception("rate-limit snapshot failed")
            return response

        reset_epoch = int(time.time() + snap.reset_in_seconds)
        response.headers["X-RateLimit-Limit"] = str(snap.cap)
        response.headers["X-RateLimit-Remaining"] = str(snap.remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_epoch)
        response.headers["X-RateLimit-Policy"] = (
            f"{snap.cap};w={int(self._window)}"
        )
        return response


def install_rate_limit_headers(app: Any, **kwargs: Any) -> None:
    app.add_middleware(RateLimitHeadersMiddleware, **kwargs)


__all__ = [
    "DEFAULT_PREFIX",
    "RateLimitHeadersMiddleware",
    "install_rate_limit_headers",
]
