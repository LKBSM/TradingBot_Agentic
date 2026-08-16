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
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, Field

from src.api.account_store import AccountError, AccountStore
from src.api.auth_throttle import AuthThrottle, client_ip
from src.api.middleware.beta_auth import beta_lockdown_enabled
from src.api.public_urls import app_public_url
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


class VerifyEmailConfirmBody(BaseModel):
    token: str = Field(..., min_length=1, max_length=512)


class PasswordChangeBody(BaseModel):
    current_password: str = Field(..., min_length=1, max_length=256)
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
    email_verified: bool = True
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


def _throttle(request: Request) -> Optional[AuthThrottle]:
    """The per-app auth throttle (AUTH-HARDENING). None only if an app was built
    without one (older test harnesses); callers must no-op gracefully then."""
    return getattr(request.app.state, "auth_throttle", None)


def _enforce_throttle(request: Request, *keys: str) -> None:
    """429 if ANY key is already at its cap. Read-only (does not record) — callers
    record failures explicitly so a success can clear the counter."""
    throttle = _throttle(request)
    if throttle is None:
        return
    for key in keys:
        blocked, retry_after = throttle.blocked(key)
        if blocked:
            raise HTTPException(
                status_code=429,
                detail="Trop de tentatives. Réessayez dans un instant.",
                headers={"Retry-After": str(retry_after)},
            )


def _record_throttle(request: Request, *keys: str) -> None:
    throttle = _throttle(request)
    if throttle is not None:
        for key in keys:
            throttle.hit(key)


def _clear_throttle(request: Request, *keys: str) -> None:
    throttle = _throttle(request)
    if throttle is not None:
        for key in keys:
            throttle.reset(key)


def _account_out(store: AccountStore, account: Dict[str, Any]) -> AccountOut:
    consents = store.get_consents(account["id"])
    return AccountOut(
        id=account["id"],
        username=account["username"],
        email=account["email"],
        role=account["role"],
        age_confirmed=account["age_confirmed"],
        email_verified=account.get("email_verified", True),
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

    # AUTH-HARDENING — throttle registration per IP (every attempt counts) so the
    # endpoint cannot be used to spray-probe which emails exist, nor to spam
    # account creation. Recorded before any work so conflicts count too.
    ip_key = f"register:ip:{client_ip(request)}"
    _enforce_throttle(request, ip_key)
    _record_throttle(request, ip_key)

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
        # AUTH-HARDENING — anti-enumeration: a conflict returns ONE generic message
        # that does not reveal WHICH of username/email is taken (the old messages
        # "Cette adresse e-mail est déjà utilisée." / "Ce nom d'utilisateur est
        # déjà pris." confirmed a specific email/username exists). Non-conflict
        # validation errors (weak password, bad email) stay specific — they leak
        # nothing about existing accounts. NB: full elimination of email
        # enumeration needs a verify-by-email registration flow; combined with the
        # per-IP throttle above, mass probing is blocked in the meantime.
        conflict_codes = {"username_taken", "email_taken", "account_conflict"}
        if exc.code in conflict_codes:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Ces informations ne peuvent pas être utilisées. Si vous avez "
                    "déjà un compte, connectez-vous ou réinitialisez votre mot de passe."
                ),
            )
        _raise_account_error(exc)

    # Auto-login on successful registration. The account is UNVERIFIED — the
    # access gate blocks market data until the emailed link is confirmed (PAY-1);
    # the session lets the UI show the "verify your email" state + resend.
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    _dispatch_verification_email(request, account["id"], account["email"])
    logger.info("registered + logged in account id=%s (unverified)", account["id"])
    return _account_out(store, account)


@router.post("/login", response_model=AccountOut)
async def login(payload: LoginRequest, request: Request, response: Response):
    store = _store(request)

    # AUTH-HARDENING — brute-force brake. Throttle by IP AND by identifier, so a
    # spray across many IPs against ONE account is also slowed. Only FAILURES
    # count and a success clears both counters, so a legit user who mistypes once
    # is never locked out (the happy path is untouched).
    ip_key = f"login:ip:{client_ip(request)}"
    id_key = f"login:id:{payload.identifier.strip().lower()}"
    _enforce_throttle(request, ip_key, id_key)

    account = store.verify_credentials(payload.identifier, payload.password)
    if account is None:
        _record_throttle(request, ip_key, id_key)
        # Single generic message — never reveal which of id/password was wrong.
        raise HTTPException(
            status_code=401, detail="Identifiant ou mot de passe incorrect."
        )
    _clear_throttle(request, ip_key, id_key)
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

    # AUTH-HARDENING — throttle reset requests per IP (every attempt counts) to
    # brake reset-email spam and identifier probing. The response is already
    # enumeration-safe (same message either way); the throttle bounds the volume.
    ip_key = f"reset:ip:{client_ip(request)}"
    _enforce_throttle(request, ip_key)
    _record_throttle(request, ip_key)

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


# =============================================================================
# Email verification (PAY-1 — mandatory before access)
# =============================================================================

@router.post("/verify-email/confirm", response_model=MessageOut)
async def verify_email_confirm(payload: VerifyEmailConfirmBody, request: Request):
    """Confirm an email-verification token. Single-use; 400 on invalid/expired."""
    store = _store(request)
    if not store.consume_email_verification(payload.token):
        raise HTTPException(status_code=400, detail="Lien de vérification invalide ou expiré.")
    return MessageOut(message="Adresse e-mail confirmée. Ton accès est activé.")


@router.post("/verify-email/resend", response_model=MessageOut)
async def verify_email_resend(
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
):
    """Re-send the verification email to the logged-in account (throttled).

    Always answers the same whether or not a token was needed (already verified →
    no email). Requires a session so it can't be used to spam arbitrary inboxes.
    """
    ip_key = f"verify:ip:{client_ip(request)}"
    _enforce_throttle(request, ip_key)
    _record_throttle(request, ip_key)
    if not account.get("email_verified", False):
        _dispatch_verification_email(request, account["id"], account["email"])
    return MessageOut(
        message="Si une vérification est nécessaire, un e-mail vient d'être envoyé."
    )


# =============================================================================
# Password change (authenticated) + account deletion (Loi 25 erasure)
# =============================================================================

@router.post("/password", response_model=AccountOut)
async def change_password(
    payload: PasswordChangeBody,
    request: Request,
    response: Response,
    account: Dict[str, Any] = Depends(require_account),
):
    """Change the password after verifying the current one.

    All sessions are revoked (other devices signed out); a fresh session is
    minted for the current device so the actor stays logged in.
    """
    store = _store(request)
    # Throttle by IP + account so a hijacked session can't brute-force the
    # current password. Only failures are recorded (a success clears them).
    ip_key = f"pwchange:ip:{client_ip(request)}"
    id_key = f"pwchange:id:{account['id']}"
    _enforce_throttle(request, ip_key, id_key)
    try:
        ok = store.change_password(
            account["id"], payload.current_password, payload.new_password
        )
    except AccountError as exc:
        _raise_account_error(exc)
    if not ok:
        _record_throttle(request, ip_key, id_key)
        raise HTTPException(status_code=400, detail="Mot de passe actuel incorrect.")
    _clear_throttle(request, ip_key, id_key)
    # change_password revoked every session — re-mint one for this device.
    raw_token = store.create_session(account["id"])
    set_session_cookie(response, raw_token)
    fresh = store.get_account(account["id"]) or account
    return _account_out(store, fresh)


@router.delete("/account", response_model=MessageOut)
async def delete_account(
    request: Request,
    response: Response,
    account: Dict[str, Any] = Depends(require_account),
):
    """Delete the account and ALL its data (Loi 25 right to erasure).

    Cancels the Stripe subscription FIRST (so deletion never leaves an active
    subscription billing a deleted user), then hard-deletes the account (cascade
    removes sessions, consents, subscription row, tokens). No card data is stored
    here, so nothing card-related survives. The session cookie is cleared.
    """
    store = _store(request)
    account_id = account["id"]

    # Cancel any live Stripe subscription before erasing the linkage.
    sub_id = store.stripe_subscription_id_for(account_id)
    if sub_id:
        stripe = getattr(request.app.state.app_state, "stripe_client", None)
        if stripe is not None and getattr(stripe, "is_configured", False):
            try:
                stripe.cancel_subscription(sub_id)
            except Exception as exc:
                logger.exception("cancel-on-delete failed for account id=%s", account_id)
                # Do NOT delete while an active subscription may keep billing.
                raise HTTPException(
                    status_code=502,
                    detail="Impossible d'annuler l'abonnement pour le moment. Réessaie.",
                ) from exc

    store.delete_account(account_id)
    clear_session_cookie(response)
    logger.info("account id=%s deleted at user request (Loi 25)", account_id)
    return MessageOut(message="Ton compte et tes données ont été supprimés.")


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


def _reset_base_url() -> str:
    """Public base URL of the frontend for building email links (reset + verify).

    PAY-2: routed through the single public-URL source so verification and reset
    links never hard-code a host. ``FRONTEND_BASE_URL`` still works (it is one of
    the honoured fallbacks) and in production the startup guard forbids localhost.
    """
    return app_public_url()


def _send_reset_email(to_email: str, reset_url: str) -> bool:
    """Send the reset link over SMTP when configured; return False otherwise.

    Env-gated (``SMTP_HOST`` …). With no SMTP configured this is a clean no-op —
    the caller then logs an honest "delivery not configured" line. Never logs the
    token/link. Best-effort: a send failure is surfaced to the caller as False.
    """
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Réinitialisation de votre mot de passe MIA Markets"
    msg["From"] = os.environ.get(
        "SMTP_FROM", os.environ.get("SMTP_USER", "no-reply@mia.markets")
    )
    msg["To"] = to_email
    msg.set_content(
        "Vous avez demandé la réinitialisation de votre mot de passe MIA Markets.\n\n"
        f"Ouvrez ce lien pour choisir un nouveau mot de passe :\n{reset_url}\n\n"
        "Si vous n'êtes pas à l'origine de cette demande, ignorez cet e-mail — "
        "votre mot de passe reste inchangé. Le lien expire après un court délai."
    )
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)
    return True


def _send_verification_email(to_email: str, verify_url: str) -> bool:
    """Send the email-verification link over SMTP when configured (else no-op).

    Same env-gated, best-effort posture as the reset email — never logs the token.
    """
    host = os.environ.get("SMTP_HOST")
    if not host:
        return False
    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = "Confirmez votre adresse e-mail — MIA Markets"
    msg["From"] = os.environ.get(
        "SMTP_FROM", os.environ.get("SMTP_USER", "no-reply@mia.markets")
    )
    msg["To"] = to_email
    msg.set_content(
        "Bienvenue sur MIA Markets.\n\n"
        "Confirmez votre adresse e-mail pour activer l'accès :\n"
        f"{verify_url}\n\n"
        "Si vous n'êtes pas à l'origine de cette inscription, ignorez cet e-mail. "
        "Le lien expire après 24 heures."
    )
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    with smtplib.SMTP(host, port, timeout=10) as server:
        server.starttls()
        if user and password:
            server.login(user, password)
        server.send_message(msg)
    return True


def _dispatch_verification_email(request: Request, account_id: int, email: str) -> None:
    """Issue + email an email-verification token (out-of-band). Degrades to a log
    line when SMTP is not configured (never logs the token)."""
    store = _store(request)
    raw_token = store.create_email_verification(account_id)
    if raw_token is None:
        return  # already verified or inactive — nothing to send
    verify_url = f"{_reset_base_url()}/verifier-email?token={raw_token}"
    if email:
        try:
            if _send_verification_email(email, verify_url):
                logger.info("verification email sent for account id=%s", account_id)
                return
        except Exception:
            logger.exception("verification email send failed for account id=%s", account_id)
    logger.info(
        "verification requested for account id=%s — email delivery not configured "
        "(token not logged)",
        account_id,
    )


def _dispatch_reset_token(request: Request, identifier: str, raw_token: str) -> None:
    """Deliver the reset token out-of-band (email link) — AUTH-02.

    Resolves the recipient email, builds the confirmation link
    (``/mot-de-passe-oublie/confirmer?token=…``) and emails it when SMTP is
    configured. With no emailer configured it degrades cleanly to a log line
    (never logging the token). Anti-enumeration is the caller's job — this is
    only reached when a matching account existed.
    """
    store = _store(request)
    email = store.email_for_identifier(identifier)
    reset_url = f"{_reset_base_url()}/mot-de-passe-oublie/confirmer?token={raw_token}"
    if email:
        try:
            if _send_reset_email(email, reset_url):
                logger.info("password reset email sent for identifier=%r", identifier)
                return
        except Exception:
            logger.exception("password reset email send failed for identifier=%r", identifier)
    logger.info(
        "password reset requested for identifier=%r — email delivery not configured "
        "(token not logged)",
        identifier,
    )
