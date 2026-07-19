"""Account & authentication endpoints.

  POST   /api/auth/register          create account (18+, consents required)
  POST   /api/auth/login             username OR email + password → session cookie
  POST   /api/auth/logout            revoke session + clear cookie
  GET    /api/auth/me                current account (or 401)
  PATCH  /api/auth/profile           update email
  POST   /api/auth/password-reset/request   issue a single-use reset token
  POST   /api/auth/password-reset/confirm   burn token + set new password

Auth lives on FastAPI (single source of truth alongside tier_manager/KeyStore).
The Next.js frontend reaches these via the same-origin ``/api/*`` rewrite, so
the session cookie is first-party. All crypto is delegated to argon2-cffi
(passwords/tokens) and itsdangerous (cookie signing) — see ``account_store`` and
``session_auth``.

NO payments here. Gated features import ``require_active_subscription`` from
``subscription_gate`` (today a pass-through; owner always allowed).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.api.account_store import AccountError, AccountStore
from src.api.middleware.beta_auth import beta_lockdown_enabled
from src.api.routes.legal import LAST_UPDATED as LEGAL_VERSION
from src.api.session_auth import (
    clear_session_cookie,
    get_raw_session_token,
    optional_account,
    require_account,
    require_owner,
    set_session_cookie,
)
from src.api.subscription_gate import account_has_access, require_active_subscription

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# The legal version recorded against every consent. Single source = legal.py,
# so the consent stamp and the rendered document version can never drift.
TERMS_VERSION = LEGAL_VERSION
PRIVACY_VERSION = LEGAL_VERSION


# =============================================================================
# Schemas
# =============================================================================

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=32)
    # Email format is validated in the store (AccountStore._validate_*) so we
    # don't pull in the optional `email-validator` dependency for EmailStr.
    email: str = Field(..., min_length=3, max_length=320)
    password: str = Field(..., min_length=10, max_length=256)
    # 18+ self-declaration — must be True (Loi 25 / responsible-gaming posture).
    age_confirmed: bool = Field(..., description="Déclare avoir 18 ans ou plus")
    # Explicit, separate consent checkboxes — both REQUIRED.
    accept_terms: bool = Field(..., description="Accepte les Conditions d'utilisation")
    accept_privacy: bool = Field(..., description="Accepte la Politique de confidentialité")


class LoginRequest(BaseModel):
    # Username OR email — a single free-text identifier field.
    identifier: str = Field(..., min_length=1, max_length=320)
    password: str = Field(..., min_length=1, max_length=256)


class ProfileUpdateRequest(BaseModel):
    email: str = Field(..., min_length=3, max_length=320)


class ResetRequestBody(BaseModel):
    identifier: str = Field(..., min_length=1, max_length=320)


class ResetConfirmBody(BaseModel):
    token: str = Field(..., min_length=1, max_length=512)
    new_password: str = Field(..., min_length=10, max_length=256)


class ConsentOut(BaseModel):
    doc: str
    version: str
    accepted_at: str


class AccountOut(BaseModel):
    id: int
    username: str
    email: str
    role: str
    age_confirmed: bool
    created_at: str
    consents: List[ConsentOut] = Field(default_factory=list)


class MessageOut(BaseModel):
    ok: bool = True
    message: str


# =============================================================================
# Helpers
# =============================================================================

def _store(request: Request) -> AccountStore:
    store = getattr(request.app.state.app_state, "account_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Account service unavailable")
    return store


def _account_out(store: AccountStore, account: Dict[str, Any]) -> AccountOut:
    consents = store.get_consents(account["id"])
    return AccountOut(
        id=account["id"],
        username=account["username"],
        email=account["email"],
        role=account["role"],
        age_confirmed=account["age_confirmed"],
        created_at=account["created_at"],
        consents=[ConsentOut(**c) for c in consents],
    )


def _raise_account_error(exc: AccountError) -> None:
    # Map store-level validation/conflict codes to deterministic 4xx. Conflicts
    # → 409, everything else → 422. The message is safe (no internals).
    conflict_codes = {"username_taken", "email_taken", "account_conflict"}
    status = 409 if exc.code in conflict_codes else 422
    raise HTTPException(status_code=status, detail=str(exc))


# =============================================================================
# Routes
# =============================================================================

@router.post("/register", response_model=AccountOut, status_code=201)
async def register(payload: RegisterRequest, request: Request, response: Response):
    store = _store(request)

    # Closed beta: public self-registration is disabled. The only accounts that
    # may exist are the owner (seeded from env) and the testers (seeded by
    # scripts/seed_testers.py). This keeps the access perimeter to invited
    # accounts only, independently of the BetaAuthMiddleware login wall.
    if beta_lockdown_enabled():
        raise HTTPException(
            status_code=403,
            detail=(
                "Les inscriptions sont fermées pendant la beta privée. "
                "L'accès est réservé aux comptes testeurs invités."
            ),
        )

    if not payload.age_confirmed:
        raise HTTPException(
            status_code=422,
            detail="Vous devez déclarer avoir 18 ans ou plus.",
        )
    if not (payload.accept_terms and payload.accept_privacy):
        raise HTTPException(
            status_code=422,
            detail=(
                "Vous devez accepter les Conditions d'utilisation et la "
                "Politique de confidentialité."
            ),
        )

    # Record the SERVER's current version (client cannot backdate consent).
    consents = [("terms", TERMS_VERSION), ("privacy", PRIVACY_VERSION)]
    try:
        account = store.create_account(
            payload.username,
            str(payload.email),
            payload.password,
            age_confirmed=payload.age_confirmed,
            consents=consents,
        )
    except AccountError as exc:
        _raise_account_error(exc)

    # Auto-login on successful registration.
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    logger.info("registered + logged in account id=%s", account["id"])
    return _account_out(store, account)


@router.post("/login", response_model=AccountOut)
async def login(payload: LoginRequest, request: Request, response: Response):
    store = _store(request)
    account = store.verify_credentials(payload.identifier, payload.password)
    if account is None:
        # Single generic message — never reveal which of id/password was wrong.
        raise HTTPException(
            status_code=401, detail="Identifiant ou mot de passe incorrect."
        )
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    return _account_out(store, account)


@router.post("/logout", response_model=MessageOut)
async def logout(request: Request, response: Response):
    store = getattr(request.app.state.app_state, "account_store", None)
    raw_token = get_raw_session_token(request)
    if store is not None and raw_token:
        store.delete_session(raw_token)
    clear_session_cookie(response)
    return MessageOut(message="Déconnecté.")


@router.get("/me", response_model=AccountOut)
async def me(
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
):
    return _account_out(_store(request), account)


@router.patch("/profile", response_model=AccountOut)
async def update_profile(
    payload: ProfileUpdateRequest,
    request: Request,
    response: Response,
    account: Dict[str, Any] = Depends(require_account),
):
    store = _store(request)
    try:
        updated = store.update_email(account["id"], str(payload.email))
    except AccountError as exc:
        _raise_account_error(exc)
    # update_email revoked every session (AUTH-14). Mint a fresh one so the
    # actor stays logged in on THIS device while other devices are signed out.
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    return _account_out(store, updated)


@router.post("/password-reset/request", response_model=MessageOut)
async def password_reset_request(payload: ResetRequestBody, request: Request):
    store = _store(request)
    raw_token = store.create_reset_token(payload.identifier)
    # Anti-enumeration: ALWAYS return the same response whether or not the
    # identifier matched. When it did, the token is dispatched out-of-band.
    if raw_token is not None:
        _dispatch_reset_token(request, payload.identifier, raw_token)
    return MessageOut(
        message=(
            "Si un compte correspond, un lien de réinitialisation a été envoyé."
        )
    )


@router.post("/password-reset/confirm", response_model=MessageOut)
async def password_reset_confirm(payload: ResetConfirmBody, request: Request):
    store = _store(request)
    try:
        ok = store.consume_reset_token(payload.token, payload.new_password)
    except AccountError as exc:
        _raise_account_error(exc)
    if not ok:
        raise HTTPException(
            status_code=400, detail="Jeton invalide ou expiré."
        )
    return MessageOut(message="Mot de passe réinitialisé. Vous pouvez vous connecter.")


@router.get("/access", response_model=Dict[str, Any])
async def access_status(
    request: Request,
    account: Dict[str, Any] = Depends(require_active_subscription),
):
    """Single entitlement probe behind the payment gate.

    Returns the resolved access for the current account. When the gate is
    enforced (``SUBSCRIPTION_GATE_ENFORCED=1``) a non-entitled non-owner account
    gets 402 from the dependency before reaching this body; owner is untouched.
    """
    return {
        "ok": True,
        "role": account["role"],
        "has_access": account_has_access(account, _store(request)),
        "is_owner": account["role"] == "owner",
    }


@router.get("/admin/overview", response_model=Dict[str, Any])
async def admin_overview(
    account: Dict[str, Any] = Depends(require_owner),
):
    """Owner-only surface — seam for the future admin dashboard.

    Proves the role wall: ``owner`` → 200, any normal ``user`` → 403. Returns a
    minimal placeholder payload (no business data yet).
    """
    return {
        "ok": True,
        "owner": account["username"],
        "note": "Admin dashboard placeholder — owner-only access confirmed.",
    }


def _dispatch_reset_token(request: Request, identifier: str, raw_token: str) -> None:
    """Deliver the reset token out-of-band (email).

    Email delivery is mission-④ territory; until an emailer is wired, we log
    that a token was issued WITHOUT logging the token value (never log secrets).
    A wired notifier would be read from app_state here.
    """
    logger.info("password reset token issued for identifier=%r (delivery pending)", identifier)
