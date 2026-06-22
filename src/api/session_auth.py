"""Session cookie plumbing + FastAPI auth dependencies.

The session token is an opaque, server-side, revocable value (see
``AccountStore.create_session``). On top of that, the cookie VALUE is signed
with **itsdangerous** (Pallets — the same signer Flask uses) so a tampered or
forged cookie is rejected before it ever touches the database. Two proven
libraries, zero home-made crypto:

* ``AccountStore`` (argon2-cffi) owns password + token hashing.
* ``itsdangerous.URLSafeTimedSerializer`` owns cookie signing.

Cookie attributes: ``HttpOnly`` (no JS access), ``SameSite=Lax`` (blocks
cross-site POST → CSRF mitigation for state-changing routes), ``Secure`` in
production (toggled off only when ``SESSION_COOKIE_SECURE=0`` for local http).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from src.api.account_store import DEFAULT_SESSION_TTL_S, AccountStore

logger = logging.getLogger(__name__)

COOKIE_NAME = "mia_session"
_SALT = "mia.session.v1"

# Signature lifetime mirrors the DB session TTL. The DB row is still the source
# of truth for revocation; the signed max_age is just defence in depth.
_SIGNATURE_MAX_AGE_S = int(DEFAULT_SESSION_TTL_S)


def _secret() -> str:
    """Resolve the cookie-signing secret from the environment.

    SESSION_SECRET must be set in any real deployment. We fall back to a
    process-local random value ONLY so dev/tests boot without config — that
    fallback rotates every process, which simply invalidates old cookies
    (safe, never insecure).
    """
    secret = os.environ.get("SESSION_SECRET")
    if not secret:
        # Lazy, process-local. Never logged.
        secret = os.environ.setdefault("_SESSION_SECRET_EPHEMERAL", os.urandom(32).hex())
        logger.warning(
            "SESSION_SECRET is not set — using an ephemeral per-process secret. "
            "Set SESSION_SECRET in the environment for stable sessions."
        )
    return secret


def _serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(_secret(), salt=_SALT)


def _cookie_secure() -> bool:
    # Secure by default; only an explicit "0"/"false" disables it (local http).
    raw = os.environ.get("SESSION_COOKIE_SECURE", "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def sign_session_token(raw_token: str) -> str:
    """Wrap the opaque DB token in an itsdangerous signature for the cookie."""
    return _serializer().dumps(raw_token)


def unsign_session_token(cookie_value: str) -> Optional[str]:
    """Verify the cookie signature and return the opaque DB token, or None."""
    if not cookie_value:
        return None
    try:
        return _serializer().loads(cookie_value, max_age=_SIGNATURE_MAX_AGE_S)
    except (BadSignature, SignatureExpired):
        return None
    except Exception:  # pragma: no cover - defensive
        logger.exception("unexpected error unsigning session cookie")
        return None


def set_session_cookie(response: Response, raw_token: str) -> None:
    response.set_cookie(
        key=COOKIE_NAME,
        value=sign_session_token(raw_token),
        max_age=_SIGNATURE_MAX_AGE_S,
        httponly=True,
        secure=_cookie_secure(),
        samesite="lax",
        path="/",
    )


def clear_session_cookie(response: Response) -> None:
    response.delete_cookie(key=COOKIE_NAME, path="/", samesite="lax")


def _get_account_store(request: Request) -> Optional[AccountStore]:
    return getattr(request.app.state.app_state, "account_store", None)


def get_raw_session_token(request: Request) -> Optional[str]:
    cookie_value = request.cookies.get(COOKIE_NAME)
    if not cookie_value:
        return None
    return unsign_session_token(cookie_value)


async def optional_account(request: Request) -> Optional[Dict[str, Any]]:
    """Dependency: the current account, or None if not logged in.

    Never raises — use for routes that adapt to logged-in/out (e.g. /me).
    """
    store = _get_account_store(request)
    if store is None:
        return None
    raw_token = get_raw_session_token(request)
    if not raw_token:
        return None
    account = store.resolve_session(raw_token)
    if account is not None:
        request.state.account = account
    return account


async def require_account(
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> Dict[str, Any]:
    """Dependency: the current account, or 401 if not authenticated."""
    if account is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    return account


async def require_owner(
    account: Dict[str, Any] = Depends(require_account),
) -> Dict[str, Any]:
    """Dependency: the current account must have the ``owner`` role (403 else)."""
    if account.get("role") != "owner":
        raise HTTPException(status_code=403, detail="Owner role required")
    return account
