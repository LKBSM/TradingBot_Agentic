"""Renewal-notice job (PAY-1 / Loi 25).

Sends a courtesy notice **30 days before an ANNUAL renewal** so the customer is
never surprised by the yearly charge. Idempotent per (account, period_end): a
daily invocation that runs twice sends at most one email. Monthly cycles rely on
Stripe's own receipt emails (enabled in the Stripe dashboard) — a small monthly
charge doesn't need an advance notice under the LPC for an indefinite-duration
contract; the annual notice is the one that matters.

Intended to be called by a daily scheduler / cron. Env-gated on SMTP: with no
SMTP configured it logs and records nothing (so nothing is "silently sent").
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

ANNUAL_LEAD_DAYS = 30
NOTICE_KIND = "annual_30d"


def _app_url() -> str:
    return os.environ.get("APP_URL", "http://localhost:3000").rstrip("/")


def _annual_price_id() -> Optional[str]:
    return os.environ.get("STRIPE_PRICE_ANNUAL") or None


def _smtp_configured() -> bool:
    """SMTP-1: one definition of "can we send at all", shared with the two other
    transactional mails instead of a third private copy."""
    from src.api.mailer import smtp_configured

    return smtp_configured()


def _send_email(to_email: str, subject: str, body: str) -> bool:
    """Send one notice. SMTP-1: delivery lives in ``src/api/mailer.py`` now."""
    from src.api.mailer import send_email

    return send_email(to_email, subject, body, purpose="renewal_notice")


def _notice_body(period_end: float) -> tuple[str, str]:
    when = time.strftime("%Y-%m-%d", time.gmtime(period_end))
    subject = "Renouvellement de votre abonnement M.I.A Markets"
    body = (
        "Votre abonnement annuel M.I.A Markets se renouvellera automatiquement le "
        f"{when}.\n\n"
        "Votre abonnement est sans engagement : vous pouvez le résilier à tout "
        "moment, en un clic, depuis votre page d'abonnement — aucun prélèvement "
        "ne sera effectué après la résiliation.\n\n"
        f"Gérer mon abonnement : {_app_url()}/abonnement\n"
    )
    return subject, body


def send_due_renewal_notices(store: Any, *, now: Optional[float] = None) -> int:
    """Send the 30-day-before-annual-renewal notice to every due account.

    Returns the number of notices actually sent. No-op (returns 0) when no annual
    price id is configured or SMTP is not set up.
    """
    price_id = _annual_price_id()
    if not price_id:
        logger.info("renewal notices: STRIPE_PRICE_ANNUAL unset — skipping")
        return 0

    now = time.time() if now is None else now
    lead = ANNUAL_LEAD_DAYS * 86400.0
    due = store.renewals_due(price_id, now=now, lead_seconds=lead, kind=NOTICE_KIND)

    # SMTP-1: the SMTP check now comes AFTER the due list, and its level depends
    # on whether anyone is actually owed a notice. Nobody due → nothing is going
    # wrong today. Someone due and no way to write to them → that is a customer
    # who will be charged for a year without the advance warning this job exists
    # to give, so it is an ERROR, not a shrug. Nothing is claimed as "sent" in
    # that case: record_renewal_notice is never reached, so the notice stays due.
    if not _smtp_configured():
        if due:
            logger.error(
                "renewal notices: %d account(s) are owed a 30-day advance notice "
                "before an annual charge and email delivery is NOT configured — "
                "they will be charged without warning. See docs/ops/envoi-courriels.md",
                len(due),
            )
        else:
            logger.info("renewal notices: SMTP not configured, nobody due — skipping")
        return 0

    sent = 0
    for row in due:
        # Claim first (idempotent) so a concurrent run can't double-send.
        if not store.record_renewal_notice(
            row["account_id"], row["period_end"], NOTICE_KIND, now=now
        ):
            continue
        subject, body = _notice_body(row["period_end"])
        try:
            if _send_email(row["email"], subject, body):
                sent += 1
                logger.info("renewal notice sent for account id=%s", row["account_id"])
        except Exception:
            logger.exception(
                "renewal notice send failed for account id=%s", row["account_id"]
            )
    return sent


__all__ = ["send_due_renewal_notices", "ANNUAL_LEAD_DAYS", "NOTICE_KIND"]
