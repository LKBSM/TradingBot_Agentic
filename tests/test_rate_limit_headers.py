"""Tests for the API-2B.4 rate-limit response headers middleware."""

from __future__ import annotations

import os
import time

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest
from fastapi import APIRouter, FastAPI
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from src.api.middleware.rate_limit_headers import RateLimitHeadersMiddleware
from src.intelligence.rag.tier_rate_limiter import TierRateLimiter


def _stub_subscriber_app(
    *,
    limiter: TierRateLimiter,
    subscriber: dict | None,
    prefix_in_route: bool = True,
) -> TestClient:
    """Build a tiny FastAPI app that pretends auth ran (stuffs subscriber
    onto request.state) and returns 200 from /api/v1/echo."""

    app = FastAPI()

    class _AttachSubscriber(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            if subscriber is not None:
                request.state.subscriber = subscriber
            return await call_next(request)

    # Order matters: RateLimitHeaders has to run AFTER subscriber attach
    # (i.e., be added FIRST since BaseHTTPMiddleware stacks LIFO).
    app.add_middleware(RateLimitHeadersMiddleware, tier_rate_limiter=limiter)
    app.add_middleware(_AttachSubscriber)

    router = APIRouter()

    @router.get("/api/v1/echo")
    def echo():
        return {"ok": True}

    @router.get("/health")
    def health():
        return {"ok": True}

    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_headers_present_for_authenticated_request():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_test_abc", "tier": "STRATEGIST"}
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    assert resp.status_code == 200
    assert "x-ratelimit-limit" in resp.headers
    assert "x-ratelimit-remaining" in resp.headers
    assert "x-ratelimit-reset" in resp.headers
    assert "x-ratelimit-policy" in resp.headers
    # STRATEGIST default cap = 200
    assert resp.headers["x-ratelimit-limit"] == "200"
    assert resp.headers["x-ratelimit-policy"].startswith("200;w=")


def test_remaining_reflects_prior_consumption():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_test_abc", "tier": "FREE"}  # cap=5
    limiter.allow("sk_test_abc", "FREE")
    limiter.allow("sk_test_abc", "FREE")
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    # 5 - 2 used = 3 remaining. The header is read-only, so it does NOT
    # decrement on this request.
    assert resp.headers["x-ratelimit-remaining"] == "3"
    # Still 3 after a second read-only call.
    resp2 = c.get("/api/v1/echo")
    assert resp2.headers["x-ratelimit-remaining"] == "3"


def test_remaining_zero_when_capped():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_full", "tier": "FREE"}
    for _ in range(5):
        limiter.allow("sk_full", "FREE")
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    assert resp.headers["x-ratelimit-remaining"] == "0"


# ---------------------------------------------------------------------------
# Reset is an absolute epoch timestamp
# ---------------------------------------------------------------------------


def test_reset_is_future_epoch_seconds():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_x", "tier": "ANALYST"}
    limiter.allow("sk_x", "ANALYST")
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    reset = int(resp.headers["x-ratelimit-reset"])
    now = int(time.time())
    # Window is 60s — reset must lie in [now, now+61]
    assert now <= reset <= now + 61


def test_reset_is_now_when_no_observations():
    """No prior records → snapshot.reset_in_seconds = 0 → header == now."""
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_fresh", "tier": "ANALYST"}
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    reset = int(resp.headers["x-ratelimit-reset"])
    now = int(time.time())
    assert abs(reset - now) <= 1


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------


def test_no_headers_for_unauthenticated_request():
    limiter = TierRateLimiter()
    c = _stub_subscriber_app(limiter=limiter, subscriber=None)
    resp = c.get("/api/v1/echo")
    assert resp.status_code == 200
    assert "x-ratelimit-limit" not in resp.headers


def test_no_headers_for_non_api_path():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_x", "tier": "FREE"}
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/health")
    assert "x-ratelimit-limit" not in resp.headers


def test_no_headers_when_limiter_not_wired():
    sub = {"api_key": "sk_x", "tier": "FREE"}
    app = FastAPI()

    class _AttachSubscriber(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.subscriber = sub
            return await call_next(request)

    app.add_middleware(RateLimitHeadersMiddleware, tier_rate_limiter=None)
    app.add_middleware(_AttachSubscriber)

    @app.get("/api/v1/echo")
    def _e():
        return {"ok": True}

    c = TestClient(app)
    resp = c.get("/api/v1/echo")
    assert "x-ratelimit-limit" not in resp.headers


def test_no_headers_when_subscriber_missing_tier():
    limiter = TierRateLimiter()
    sub = {"api_key": "sk_x"}  # no "tier" field
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    assert "x-ratelimit-limit" not in resp.headers


def test_no_headers_when_subscriber_missing_api_key():
    limiter = TierRateLimiter()
    sub = {"tier": "FREE"}  # no api_key or key_id
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    assert "x-ratelimit-limit" not in resp.headers


def test_falls_back_to_key_id_when_api_key_absent():
    limiter = TierRateLimiter()
    sub = {"key_id": 42, "tier": "FREE"}
    c = _stub_subscriber_app(limiter=limiter, subscriber=sub)
    resp = c.get("/api/v1/echo")
    assert resp.headers["x-ratelimit-limit"] == "5"


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


def test_limiter_exception_does_not_break_response():
    sub = {"api_key": "sk_x", "tier": "FREE"}

    class _Boom:
        def snapshot(self, *a, **kw):
            raise RuntimeError("limiter exploded")

    app = FastAPI()

    class _AttachSubscriber(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            request.state.subscriber = sub
            return await call_next(request)

    app.add_middleware(RateLimitHeadersMiddleware, tier_rate_limiter=_Boom())
    app.add_middleware(_AttachSubscriber)

    @app.get("/api/v1/echo")
    def _e():
        return {"ok": True}

    resp = TestClient(app).get("/api/v1/echo")
    assert resp.status_code == 200
    assert "x-ratelimit-limit" not in resp.headers
