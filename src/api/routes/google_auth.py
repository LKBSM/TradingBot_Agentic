"""Google OAuth sign-in (PAY-1) — env-gated, no secrets in code.

Flow (hosted by Google — we never see a password):

  GET  /api/auth/google/config    → {enabled} so the UI shows/hides the button
  GET  /api/auth/google/start     → redirect to Google's consent screen (CSRF
                                     state in a signed, HttpOnly cookie)
  GET  /api/auth/google/callback  → verify state, exchange the code, read the
                                     Google-verified email, then:
                                       · existing account → log in (mark verified)
                                       · new email        → carry a short-lived
                                         signed "pending" token to the frontend
                                         consent step (18+/terms/privacy MUST be
                                         accepted before an account is created)
  POST /api/auth/google/finalize  → {token, age_confirmed, accept_terms,
                                     accept_privacy} → create the account
                                     (email pre-verified by Google) + log in

When ``GOOGLE_CLIENT_ID``/``GOOGLE_CLIENT_SECRET`` are unset the router is a
clean no-op: ``/config`` says ``enabled:false`` and the flow endpoints 404. So a
deploy without Google configured simply doesn't offer the button.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from src.api.account_store import AccountError, AccountStore
from src.api.public_urls import api_public_url, app_public_url
from src.api.routes.legal import LAST_UPDATED as LEGAL_VERSION
from src.api.session_auth import set_session_cookie

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/google", tags=["auth-google"])

_AUTHORIZE_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_TOKEN_URL = "https://oauth2.googleapis.com/token"
_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

_STATE_COOKIE = "g_oauth_state"
_STATE_MAX_AGE = 600  # 10 min to complete the round trip
_PENDING_MAX_AGE = 900  # 15 min for the consent step


# =============================================================================
# Config
# =============================================================================

def _client_id() -> Optional[str]:
    return os.environ.get("GOOGLE_CLIENT_ID") or None


def _client_secret() -> Optional[str]:
    return os.environ.get("GOOGLE_CLIENT_SECRET") or None


def is_configured() -> bool:
    return bool(_client_id() and _client_secret())


def _redirect_uri() -> str:
    explicit = os.environ.get("GOOGLE_REDIRECT_URI")
    if explicit:
        return explicit
    # PAY-2: the OAuth callback lands on the BACKEND — single public-URL source.
    return f"{api_public_url()}/api/auth/google/callback"


def _app_url() -> str:
    # PAY-2: "back to the app" is the FRONTEND — single public-URL source.
    return app_public_url()


def _signer() -> URLSafeTimedSerializer:
    # Reuse the session signing secret (fail-fast in prod is enforced elsewhere).
    secret = os.environ.get("SESSION_SECRET") or "dev-ephemeral-oauth-secret"
    return URLSafeTimedSerializer(secret, salt="google-oauth")


def _require_configured() -> None:
    if not is_configured():
        raise HTTPException(status_code=404, detail="Google sign-in is not enabled.")


def _store(request: Request) -> AccountStore:
    store = getattr(request.app.state.app_state, "account_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Account service unavailable")
    return store


# =============================================================================
# HTTP helpers (stdlib — no extra dependency)
# =============================================================================

def _post_form(url: str, data: Dict[str, str]) -> Dict[str, Any]:
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (fixed host)
        return json.loads(resp.read().decode())


def _get_json(url: str, bearer: str) -> Dict[str, Any]:
    req = urllib.request.Request(url, method="GET")
    req.add_header("Authorization", f"Bearer {bearer}")
    req.add_header("Accept", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:  # noqa: S310 (fixed host)
        return json.loads(resp.read().decode())


# =============================================================================
# Routes
# =============================================================================

@router.get("/config")
async def google_config() -> Dict[str, bool]:
    """Public: whether the Google button should be shown."""
    return {"enabled": is_configured()}


@router.get("/start")
async def google_start(response: Response):
    """Redirect to Google's consent screen with a signed CSRF state cookie."""
    _require_configured()
    nonce = secrets.token_urlsafe(16)
    state = _signer().dumps(nonce)
    params = {
        "client_id": _client_id(),
        "redirect_uri": _redirect_uri(),
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
        "prompt": "select_account",
    }
    url = f"{_AUTHORIZE_URL}?{urllib.parse.urlencode(params)}"
    redirect = RedirectResponse(url, status_code=302)
    # HttpOnly state cookie — compared against the ?state on callback (CSRF).
    secure = os.environ.get("SESSION_COOKIE_SECURE", "1") not in ("0", "false", "no")
    redirect.set_cookie(
        _STATE_COOKIE, state, max_age=_STATE_MAX_AGE, httponly=True,
        samesite="lax", secure=secure, path="/api/auth/google",
    )
    return redirect


@router.get("/callback")
async def google_callback(request: Request):
    """Handle Google's redirect: verify state, exchange the code, resolve email."""
    _require_configured()
    err_url = f"{_app_url()}/connexion?error=google"

    state_q = request.query_params.get("state")
    code = request.query_params.get("code")
    state_cookie = request.cookies.get(_STATE_COOKIE)
    if not code or not state_q or not state_cookie or state_q != state_cookie:
        return RedirectResponse(err_url, status_code=302)
    try:
        _signer().loads(state_q, max_age=_STATE_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return RedirectResponse(err_url, status_code=302)

    try:
        token = _post_form(
            _TOKEN_URL,
            {
                "code": code,
                "client_id": _client_id() or "",
                "client_secret": _client_secret() or "",
                "redirect_uri": _redirect_uri(),
                "grant_type": "authorization_code",
            },
        )
        access_token = token.get("access_token")
        if not access_token:
            raise ValueError("no access_token")
        info = _get_json(_USERINFO_URL, access_token)
    except Exception:
        logger.exception("google oauth token/userinfo exchange failed")
        return RedirectResponse(err_url, status_code=302)

    email = (info.get("email") or "").strip()
    google_verified = bool(info.get("email_verified"))
    if not email or not google_verified:
        # We only trust a Google-verified address.
        return RedirectResponse(err_url, status_code=302)

    store = _store(request)
    existing = store.get_account_by_identifier(email)
    if existing is not None:
        # Existing account → log in. Google vouched for the address, so ensure
        # it is marked verified.
        store.mark_email_verified(existing["id"])
        raw_token = store.create_session(existing["id"])
        redirect = RedirectResponse(f"{_app_url()}/app", status_code=302)
        set_session_cookie(redirect, raw_token)
        redirect.delete_cookie(_STATE_COOKIE, path="/api/auth/google")
        return redirect

    # New user → carry a signed pending token to the consent step (18+/terms/
    # privacy MUST be accepted before we create the account — Loi 25 / 18+).
    pending = _signer().dumps({"email": email, "verified": True})
    redirect = RedirectResponse(
        f"{_app_url()}/inscription/google?tok={urllib.parse.quote(pending)}",
        status_code=302,
    )
    redirect.delete_cookie(_STATE_COOKIE, path="/api/auth/google")
    return redirect


class GoogleFinalizeBody(BaseModel):
    token: str = Field(..., min_length=1, max_length=2048)
    username: str = Field(..., min_length=3, max_length=32)
    age_confirmed: bool = False
    accept_terms: bool = False
    accept_privacy: bool = False


@router.post("/finalize")
async def google_finalize(
    payload: GoogleFinalizeBody, request: Request, response: Response
):
    """Create the Google-linked account after the consent step, then log in."""
    _require_configured()
    try:
        data = _signer().loads(payload.token, max_age=_PENDING_MAX_AGE)
    except (BadSignature, SignatureExpired):
        raise HTTPException(status_code=400, detail="Lien Google expiré. Réessaie.")
    email = str(data.get("email") or "")
    if not email:
        raise HTTPException(status_code=400, detail="Lien Google invalide.")
    if not payload.age_confirmed:
        raise HTTPException(status_code=422, detail="Vous devez déclarer avoir 18 ans ou plus.")
    if not (payload.accept_terms and payload.accept_privacy):
        raise HTTPException(
            status_code=422,
            detail="Vous devez accepter les Conditions d'utilisation et la Politique de confidentialité.",
        )

    store = _store(request)
    # If the email got created meanwhile, just log in.
    existing = store.get_account_by_identifier(email)
    if existing is not None:
        store.mark_email_verified(existing["id"])
        raw_token = store.create_session(existing["id"])
        set_session_cookie(response, raw_token)
        return {"ok": True, "id": existing["id"], "email": existing["email"]}

    consents = [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
    # A random, unusable password: Google users authenticate via Google. They can
    # set a password later via the reset flow if they ever want password login.
    random_password = secrets.token_urlsafe(24)
    try:
        account = store.create_account(
            payload.username, email, random_password,
            age_confirmed=True, consents=consents,
        )
    except AccountError as exc:
        code = 409 if exc.code in {"username_taken", "email_taken", "account_conflict"} else 422
        raise HTTPException(status_code=code, detail=str(exc))
    # Google already verified the address.
    store.mark_email_verified(account["id"])
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    return {"ok": True, "id": account["id"], "email": account["email"]}
