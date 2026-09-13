"""SMTP-1 — the single place transactional email is sent from.

Why this module exists
----------------------
The same twenty lines of ``smtplib`` were copy-pasted three times: email
verification and password reset in ``routes/accounts.py``, the annual renewal
notice in ``billing/renewal_notices.py``. Each copy decided on its own what to
do when SMTP was unconfigured, and each one answered "return False, quietly".

That duplication is not a style problem, it is why the following went unnoticed
in production: ``SMTP_HOST`` was never declared in ``render.yaml``, so all three
returned False forever. The only trace was an ``INFO`` line.

And the consequence was not "some emails are late". ``EMAIL_VERIFICATION_ENFORCED``
defaults to **on**, ``access.py`` denies ``has_access`` to any unverified
account, and the code that lifts that wall is delivered by… email. So every new
customer who registered was locked out for good — while the owner account, which
is seeded verified and exempt, kept working perfectly. The wall and its key had
drifted apart.

What this module does about it
------------------------------
- **One sender.** ``send_email`` is the only place ``smtplib`` is used for
  transactional mail, so there is one answer to "what happens when SMTP is
  missing", and one place to change it.
- **Loud at boot.** :func:`assert_email_delivery_configured` refuses to start a
  production deploy that raises the verification wall without a way to send the
  key — the same posture as ``assert_public_urls_configured``. The error names
  both ways out.
- **Loud per event.** :func:`report_undeliverable` logs at ``ERROR``, not
  ``INFO``: at that instant a real person is stuck in front of a wall.
- **Visible.** :func:`health_snapshot` puts the state on ``/health``.

Environment
-----------
``SMTP_HOST`` (the switch), ``SMTP_PORT`` (587), ``SMTP_USER``,
``SMTP_PASSWORD``, ``SMTP_FROM``, ``SMTP_TIMEOUT_S`` (10), ``SMTP_STARTTLS``
(on). Credentials live only in the environment. Setup: docs/ops/envoi-courriels.md
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

ENV_HOST = "SMTP_HOST"
ENV_PORT = "SMTP_PORT"
ENV_USER = "SMTP_USER"
ENV_PASSWORD = "SMTP_PASSWORD"
ENV_FROM = "SMTP_FROM"
ENV_TIMEOUT = "SMTP_TIMEOUT_S"
ENV_STARTTLS = "SMTP_STARTTLS"
ENV_VERIFICATION_ENFORCED = "EMAIL_VERIFICATION_ENFORCED"

DEFAULT_PORT = 587
DEFAULT_TIMEOUT_S = 10.0
DEFAULT_FROM = "no-reply@mia.markets"

#: The three transactional mails, and what breaks for the customer when one
#: cannot be delivered. Used verbatim in the boot error and in the logs, so the
#: person reading them does not have to go and work it out.
PURPOSES = {
    "verification": "a new account can never be confirmed, so it never gains access",
    "password_reset": "a locked-out customer has no way back in",
    "renewal_notice": "the 30-day advance notice before an annual charge is not sent",
}


class EmailDeliveryNotConfigured(RuntimeError):
    """Raised at startup when production would lock customers out silently."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

def _env(name: str, default: str = "", env: Optional[dict] = None) -> str:
    return ((env if env is not None else os.environ).get(name) or default).strip()


def _is_production(env: Optional[dict] = None) -> bool:
    return _env("ENVIRONMENT", "", env).lower() in {"production", "prod"}


def smtp_configured(env: Optional[dict] = None) -> bool:
    """``SMTP_HOST`` is the switch — same convention the old copies used."""
    return bool(_env(ENV_HOST, "", env))


def email_verification_enforced(env: Optional[dict] = None) -> bool:
    """Mirror of ``subscription_gate._email_verification_enforced`` (default on).

    Duplicated here rather than imported: this module is called from the boot
    guard and must not drag the subscription gate (and its FastAPI imports) into
    startup. The two are pinned together by
    ``test_wall_flag_reading_matches_the_subscription_gate``.

    Note the ``.get(name, "1")`` rather than ``or "1"``: an **empty** value means
    the wall is DOWN, only an *absent* variable defaults to up. That distinction
    is not pedantry — reading it the other way would make the boot guard refuse
    to start over a wall that is not actually standing.
    """
    source = env if env is not None else os.environ
    raw = source.get(ENV_VERIFICATION_ENFORCED, "1")
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def sender_address(env: Optional[dict] = None) -> str:
    return _env(ENV_FROM, "", env) or _env(ENV_USER, "", env) or DEFAULT_FROM


def _port(env: Optional[dict] = None) -> int:
    raw = _env(ENV_PORT, str(DEFAULT_PORT), env)
    try:
        return int(raw)
    except ValueError:
        logger.warning("mailer: %s=%r is not a number — using %d", ENV_PORT, raw, DEFAULT_PORT)
        return DEFAULT_PORT


def _timeout(env: Optional[dict] = None) -> float:
    raw = _env(ENV_TIMEOUT, str(DEFAULT_TIMEOUT_S), env)
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_TIMEOUT_S


def _starttls(env: Optional[dict] = None) -> bool:
    return _env(ENV_STARTTLS, "1", env).lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Observability — what /health reports
# --------------------------------------------------------------------------- #

@dataclass
class _State:
    sent: int = 0
    failed: int = 0
    undeliverable: int = 0          # attempted while SMTP was not configured
    last_sent_utc: Optional[float] = None
    last_error: Optional[str] = None
    last_error_utc: Optional[float] = None
    last_undeliverable_purpose: Optional[str] = None
    lock: threading.Lock = field(default_factory=threading.Lock)


_state = _State()


def reset_state() -> None:
    """Test helper — production never calls this."""
    global _state
    _state = _State()


def report_undeliverable(purpose: str, *, detail: str = "") -> None:
    """Record + shout that a mail could not be delivered at all.

    ``ERROR``, deliberately. The old code logged this at ``INFO``, which in a
    production JSON log stream is indistinguishable from nothing happening —
    while on the other end a real person is waiting for a code that will never
    arrive.
    """
    consequence = PURPOSES.get(purpose, "the customer does not receive it")
    with _state.lock:
        _state.undeliverable += 1
        _state.last_undeliverable_purpose = purpose
    logger.error(
        "email NOT DELIVERED (%s): email delivery is not configured — %s. "
        "Set SMTP_HOST/SMTP_USER/SMTP_PASSWORD (docs/ops/envoi-courriels.md).%s",
        purpose,
        consequence,
        f" {detail}" if detail else "",
    )


def health_snapshot(env: Optional[dict] = None) -> dict:
    """Compact view for ``/health``. Never raises, never costs a connection."""
    configured = smtp_configured(env)
    wall = email_verification_enforced(env)
    with _state.lock:
        snap = {
            "configured": configured,
            "verification_wall": wall,
            "sent": _state.sent,
            "failed": _state.failed,
            "undeliverable": _state.undeliverable,
            "last_error": _state.last_error,
        }
    # The one field an operator should look at. False means somebody, somewhere,
    # is being asked to confirm an address by a system that cannot write to them.
    snap["healthy"] = configured or not wall
    return snap


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #

def send_email(
    to_email: str,
    subject: str,
    text_body: str,
    *,
    purpose: str = "transactional",
    branded: bool = True,
    env: Optional[dict] = None,
) -> bool:
    """Send one plain-text mail (with a branded HTML alternative).

    Returns ``False`` — without raising — when SMTP is not configured, which is
    the contract the three call sites already had. A genuine delivery failure
    still **raises**, so the caller's ``except`` keeps logging it as before.
    """
    host = _env(ENV_HOST, "", env)
    if not host:
        report_undeliverable(purpose, detail=f"subject={subject!r}")
        return False

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = sender_address(env)
    msg["To"] = to_email
    msg.set_content(text_body)
    if branded:
        try:
            from src.api.email_branding import attach_branded_html

            attach_branded_html(msg, text_body)
        except Exception:  # pragma: no cover - HTML is a best-effort enhancement
            logger.debug("branded HTML alternative skipped", exc_info=True)

    user = _env(ENV_USER, "", env) or None
    password = _env(ENV_PASSWORD, "", env) or None
    try:
        with smtplib.SMTP(host, _port(env), timeout=_timeout(env)) as server:
            if _starttls(env):
                server.starttls()
            if user and password:
                server.login(user, password)
            server.send_message(msg)
    except Exception as exc:
        with _state.lock:
            _state.failed += 1
            _state.last_error = f"{type(exc).__name__}: {exc}"
            _state.last_error_utc = time.time()
        raise

    with _state.lock:
        _state.sent += 1
        _state.last_sent_utc = time.time()
        _state.last_error = None
    logger.info("email sent (%s)", purpose)
    return True


# --------------------------------------------------------------------------- #
# Boot guard
# --------------------------------------------------------------------------- #

def assert_email_delivery_configured(env: Optional[dict] = None) -> None:
    """Fail-fast in production when the verification wall has no key.

    A no-op outside production (dev/CI/tests keep working with no SMTP).

    The condition is deliberately narrow: not "SMTP must exist", but **"the
    email-verification wall must not stand while the mail that opens it cannot
    be sent"**. That is the contradiction that locked every new customer out,
    and it is the one worth refusing to deploy over.

    With the wall lowered and no SMTP, the deploy is allowed — new customers can
    still reach the product — but password reset and renewal notices are still
    dead, so that case is logged at ``ERROR`` rather than waved through.
    """
    if not _is_production(env):
        return
    if smtp_configured(env):
        return

    if not email_verification_enforced(env):
        logger.error(
            "Email delivery is NOT configured (%s unset). The verification wall is "
            "down so new accounts can still reach the product, but: %s, and %s. "
            "Configure SMTP — docs/ops/envoi-courriels.md",
            ENV_HOST,
            PURPOSES["password_reset"],
            PURPOSES["renewal_notice"],
        )
        return

    raise EmailDeliveryNotConfigured(
        "Refusing to start — the email-verification wall is up but no email can "
        "be sent.\n"
        f"  {ENV_VERIFICATION_ENFORCED} is on (it defaults to on), so "
        "access.py denies has_access to any account that has not confirmed its "
        "address.\n"
        f"  {ENV_HOST} is unset, so the confirmation code is never delivered.\n"
        "  Result: every customer who registers is locked out permanently, and "
        "the owner account — seeded verified — is the only one that still "
        "works, which is why this is easy to miss.\n"
        "Two ways out, pick one:\n"
        f"  - configure SMTP: set {ENV_HOST}, {ENV_USER}, {ENV_PASSWORD}, "
        f"{ENV_FROM} (see docs/ops/envoi-courriels.md); or\n"
        f"  - lower the wall on purpose: set {ENV_VERIFICATION_ENFORCED}=0 "
        "(new accounts then reach the product without confirming an address, "
        "and password reset stays unavailable)."
    )


__all__ = [
    "EmailDeliveryNotConfigured",
    "PURPOSES",
    "assert_email_delivery_configured",
    "email_verification_enforced",
    "health_snapshot",
    "report_undeliverable",
    "reset_state",
    "send_email",
    "sender_address",
    "smtp_configured",
]
