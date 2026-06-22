"""Subscription gate — the SINGLE seam where a future payment wall will decide
access. NO Stripe, NO billing logic here (that is mission ②).

Today this is intentionally a pass-through: every authenticated account is
allowed, and the ``owner`` role is ALWAYS allowed (and must stay allowed even
after the gate becomes real). The one job of this module is to give the rest of
the codebase a stable, single place to import so that when the paywall lands it
is a localized change here — not a scatter of conditionals across routes.

Usage (FastAPI dependency)::

    from src.api.subscription_gate import require_active_subscription

    @router.get("/api/something-premium")
    async def premium(account = Depends(require_active_subscription)):
        ...

The dependency returns the authenticated account dict (role included). It
raises 401 when there is no authenticated account.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import Depends, HTTPException

from src.api.session_auth import require_account

logger = logging.getLogger(__name__)


def account_has_access(account: Dict[str, Any]) -> bool:
    """Return True if the account may access gated features.

    TODAY: everyone authenticated passes (free during personal-testing phase).
    The ``owner`` short-circuit is permanent — when the real paywall replaces
    the ``return True`` below, owner access MUST remain unconditional.

    FUTURE GATE (mission ②) — replace the marked line with the real check, e.g.::

        if account.get("role") == "owner":
            return True
        return billing.has_active_subscription(account["id"])
    """
    if account.get("role") == "owner":
        return True
    # ─── FUTURE PAYMENT GATE HOOKS HERE ───────────────────────────────────
    # Until mission ② wires billing, all authenticated accounts have access.
    return True


async def require_active_subscription(
    account: Dict[str, Any] = Depends(require_account),
) -> Dict[str, Any]:
    """Dependency: authenticated AND (today: always) entitled.

    Owner always passes. When the paywall is wired in :func:`account_has_access`,
    a non-entitled account gets 402 Payment Required while owner is untouched.
    """
    if not account_has_access(account):
        # 402 is the canonical "you must pay" status. Unused today because the
        # gate is open, but wired so the future change is one function away.
        raise HTTPException(
            status_code=402,
            detail="An active subscription is required for this feature.",
        )
    return account
