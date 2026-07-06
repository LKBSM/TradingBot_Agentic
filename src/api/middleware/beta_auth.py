"""Private-beta lockdown middleware — a hard login wall over the whole API.

Context
-------
The product ships to a CLOSED beta (a handful of invited testers) before any
commercial launch. During that phase EVERY data / AI / scanner / view-control
endpoint must answer **only** to an authenticated, authorized account — an
anonymous caller hitting the API directly must get ``401`` and **zero data**.

Why a middleware and not per-route dependencies
-----------------------------------------------
The API exposes ~30 routers across two historical auth systems (session cookies
for the webapp, ``X-API-Key`` for the legacy ``/api/v1/*`` surface), and both
are effectively open in the personal-testing configuration
(``SENTINEL_TESTING_MODE=1`` bypasses ``require_api_key``; the freemium guards
are no-ops while ``SUBSCRIPTION_GATE_ENFORCED=0``). Sprinkling a new dependency
on every route is error-prone — one forgotten route is a data leak. A single
middleware in front of the router is *blanket* coverage: a route cannot be
forgotten because nothing reaches the router without passing here first.

This wall is intentionally ORTHOGONAL to the subscription gate. It does not care
about tier or payment — only "is this a valid, active, authorized session?".
The freemium machinery (``entitlements`` / ``subscription_gate``) stays exactly
as-is; flipping this on does not touch it.

Posture
-------
* ``BETA_LOCKDOWN`` unset / ``0`` → the middleware is a **no-op** (pass-through),
  so every existing test and the current personal-testing deployment are
  unaffected. Enforcement is a single deliberate env flip, mirroring the house
  style of ``SENTINEL_TESTING_MODE`` / ``SUBSCRIPTION_GATE_ENFORCED``.
* ``BETA_LOCKDOWN=1`` → any request whose path is not on the public allowlist
  must carry a valid ``mia_session`` cookie resolving to an active account
  (optionally restricted to a set of roles via ``BETA_ALLOWED_ROLES``).

All cookie crypto is reused from :mod:`src.api.session_auth` (itsdangerous
signature) and :class:`src.api.account_store.AccountStore` (opaque token → row).
No new crypto here.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Set

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.api.session_auth import COOKIE_NAME, unsign_session_token

logger = logging.getLogger(__name__)


# Paths that stay public even under lockdown. Kept minimal on purpose: the login
# flow, the public access probe the UI reads, the legal documents, and the
# health checks the orchestrator/Docker uses. Everything else is walled.
#
# NOTE: ``/api/auth/me`` and ``/api/access/me`` are allowlisted but still return
# their own auth-aware payloads (401 / ``authenticated: false``) — the frontend
# relies on reaching them while logged out to decide where to route.
DEFAULT_ALLOWED_EXACT: Set[str] = {
    "/health",
    "/api/access/me",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/me",
    "/api/auth/register",          # reachable, but the route itself 403s under lockdown
    "/api/auth/password-reset/request",
    "/api/auth/password-reset/confirm",
    "/api/v1/terms",
    "/api/v1/privacy",
}

# Prefix allowlist — any path under these is public.
DEFAULT_ALLOWED_PREFIXES: tuple[str, ...] = (
    "/health",        # /health, /health/deep
    "/api/v1/legal",  # legal version + rendered documents
)


def beta_lockdown_enabled() -> bool:
    """Whether the beta lockdown wall is active. Default OFF (fail-open to the
    existing behavior — this wall is opt-in, never silently locking tests).

    Public so other modules (registration route, access probe) share ONE source
    of truth for the flag instead of re-reading the env var independently.
    """
    raw = os.environ.get("BETA_LOCKDOWN", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


# Backwards-friendly private alias used within this module.
_lockdown_enabled = beta_lockdown_enabled


def _allowed_roles() -> Optional[Set[str]]:
    """Optional role restriction. Empty/unset → any active account is allowed.

    ``BETA_ALLOWED_ROLES=tester,owner`` would refuse a bare ``user`` account even
    with a valid session. For the closed beta the default (any active account)
    is sufficient because public registration is disabled under lockdown, so the
    only accounts that exist are the seeded testers + the owner.
    """
    raw = os.environ.get("BETA_ALLOWED_ROLES", "").strip()
    if not raw:
        return None
    roles = {p.strip().lower() for p in raw.split(",") if p.strip()}
    return roles or None


class BetaAuthMiddleware(BaseHTTPMiddleware):
    """Require a valid session on every non-allowlisted request under lockdown."""

    def __init__(
        self,
        app,
        *,
        enabled: Optional[bool] = None,
        allowed_exact: Optional[Iterable[str]] = None,
        allowed_prefixes: Optional[Iterable[str]] = None,
        allowed_roles: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__(app)
        # ``enabled`` overrides the env flag when passed (used by tests). When
        # None we resolve the env flag PER REQUEST so a test can toggle it with
        # monkeypatch after the app is built.
        self._enabled_override = enabled
        self._allowed_exact: Set[str] = (
            set(allowed_exact) if allowed_exact is not None else set(DEFAULT_ALLOWED_EXACT)
        )
        self._allowed_prefixes: tuple[str, ...] = (
            tuple(allowed_prefixes)
            if allowed_prefixes is not None
            else DEFAULT_ALLOWED_PREFIXES
        )
        self._allowed_roles_override: Optional[Set[str]] = (
            {r.lower() for r in allowed_roles} if allowed_roles is not None else None
        )

    # -- helpers ----------------------------------------------------------

    def _is_public(self, path: str) -> bool:
        if path in self._allowed_exact:
            return True
        for prefix in self._allowed_prefixes:
            if path == prefix or path.startswith(prefix + "/"):
                return True
        return False

    def _enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        return _lockdown_enabled()

    def _roles(self) -> Optional[Set[str]]:
        if self._allowed_roles_override is not None:
            return self._allowed_roles_override
        return _allowed_roles()

    @staticmethod
    def _unauthorized(detail: str) -> JSONResponse:
        # WWW-Authenticate advertises the scheme; the body is deliberately terse
        # (no internals) — same discipline as the rest of the API.
        return JSONResponse(
            status_code=401,
            content={"error": "authentication_required", "detail": detail},
            headers={"WWW-Authenticate": "Cookie"},
        )

    # -- dispatch ---------------------------------------------------------

    async def dispatch(self, request: Request, call_next):
        if not self._enabled():
            return await call_next(request)

        # CORS preflight carries no credentials by design — let the CORS layer
        # answer it. A preflight never returns protected data.
        if request.method == "OPTIONS":
            return await call_next(request)

        if self._is_public(request.url.path):
            return await call_next(request)

        # Resolve the signed session cookie → opaque token → account row.
        cookie = request.cookies.get(COOKIE_NAME)
        raw_token = unsign_session_token(cookie) if cookie else None
        if not raw_token:
            return self._unauthorized("Authentification requise (beta privée).")

        store = getattr(request.app.state.app_state, "account_store", None)
        if store is None:
            # Fail CLOSED: if we cannot verify the session we must not serve data.
            logger.error("BETA_LOCKDOWN active but no account_store — denying request")
            return self._unauthorized("Service d'authentification indisponible.")

        account = store.resolve_session(raw_token)
        if account is None:
            return self._unauthorized("Session invalide ou expirée.")

        roles = self._roles()
        if roles is not None and str(account.get("role", "")).lower() not in roles:
            return JSONResponse(
                status_code=403,
                content={
                    "error": "forbidden",
                    "detail": "Compte non autorisé pour la beta privée.",
                },
            )

        # Stash for downstream handlers (avoids a second cookie resolution).
        request.state.account = account
        return await call_next(request)
