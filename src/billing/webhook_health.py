"""Webhook health — make a dead webhook impossible to miss (PAY-3, G4).

The failure this guards against
-------------------------------
``POST /api/billing/sync`` (PAY-3e) reconciles a subscription straight from the
Stripe API when no webhook arrived. It saves the customer — but it also *hides*
the breakage: from the outside everything looks fine, while the webhook has been
dead for a week and every other customer is silently locked out.

So every rescue is counted, logged at ERROR, and (when configured) pushed to an
alert channel. A rescue is not an incident by itself — the first one right after
Checkout can simply beat Stripe's delivery — but a rising counter means the
webhook is no longer doing its job.

State is in-process on purpose: this is a signal, not an audit trail. The audit
trail is ``processed_webhooks`` in the database.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Alert channel — reuses the Discord webhook the rest of the stack already uses.
ALERT_WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"

_lock = threading.Lock()
_state: Dict[str, Any] = {
    "rescues": 0,
    "last_rescue_at": None,     # unix ts
    "last_account_id": None,
}


def _post_alert(message: str) -> None:
    """Best-effort alert. Never raises, never blocks the request on a failure."""
    url = os.environ.get(ALERT_WEBHOOK_ENV)
    if not url:
        return
    try:
        import json
        import urllib.request

        payload = json.dumps({"content": message}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        urllib.request.urlopen(req, timeout=5).close()
    except Exception:
        logger.warning("webhook-health alert could not be delivered", exc_info=True)


def record_sync_rescue(
    *,
    account_id: int,
    stripe_status: str,
    stored_status: Optional[str] = None,
) -> int:
    """Record that reconciliation had to grant access the webhook never granted.

    Returns the running rescue count for this process.
    """
    with _lock:
        _state["rescues"] += 1
        _state["last_rescue_at"] = time.time()
        _state["last_account_id"] = account_id
        count = int(_state["rescues"])

    logger.error(
        "WEBHOOK MISS - /api/billing/sync had to rescue account=%s "
        "(Stripe says %s, our DB said %s). Rescue #%d in this process. "
        "If this keeps climbing the Stripe webhook endpoint is not delivering: "
        "check the endpoint URL, its signing secret, and the Stripe dashboard "
        "delivery log.",
        account_id, stripe_status, stored_status, count,
    )
    _post_alert(
        f":rotating_light: MIA billing — webhook miss #{count}: account {account_id} "
        f"was only granted access by reconciliation (Stripe: {stripe_status}, "
        f"stored: {stored_status}). The Stripe webhook may be down."
    )
    return count


def snapshot() -> Dict[str, Any]:
    """Current counters — surfaced by the health endpoint."""
    with _lock:
        return dict(_state)


def reset_for_tests() -> None:
    with _lock:
        _state["rescues"] = 0
        _state["last_rescue_at"] = None
        _state["last_account_id"] = None


__all__ = ["ALERT_WEBHOOK_ENV", "record_sync_rescue", "reset_for_tests", "snapshot"]
