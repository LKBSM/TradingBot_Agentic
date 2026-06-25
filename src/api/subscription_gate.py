"""Subscription gate — the SINGLE seam where the payment wall decides access.

This is the ONE place the rest of the codebase imports to ask "may this account
use a gated feature?". Keeping it centralized means the paywall is a localized
concern here, not a scatter of conditionals across routes.

Wiring (payments mission ②)
---------------------------
Access is resolved from the account's persisted subscription state
(``AccountStore.get_subscription``) — set by Stripe webhooks, never from card
data. Two invariants hold forever:

* The ``owner`` role is ALWAYS allowed (the operator must never lock themselves
  out, including during the personal-testing phase).
* During the personal-testing phase the gate is OPEN by default. Enforcement is
  a deliberate env switch — set ``SUBSCRIPTION_GATE_ENFORCED=1`` to require a
  live subscription for non-owner accounts. This mirrors the existing
  ``SENTINEL_TESTING_MODE`` philosophy (machinery fully wired, flip when ready).

Usage (FastAPI dependency)::

    from src.api.subscription_gate import require_active_subscription

    @router.get("/api/something-premium")
    async def premium(account = Depends(require_active_subscription)):
        ...

The dependency returns the authenticated account dict (role included). It
raises 401 when there is no authenticated account, 402 when authenticated but
not entitled (and the gate is enforced).
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from fastapi import Depends, HTTPException, Request

from src.api.session_auth import require_account

logger = logging.getLogger(__name__)

# Stripe subscription statuses that grant access. ``trialing`` is included so an
# (optional) free trial counts as access while it runs.
ACTIVE_STATUSES = frozenset({"active", "trialing"})


def _gate_enforced() -> bool:
    """Whether the paywall is enforced for non-owner accounts.

    Default OFF (personal-testing phase) — only an explicit truthy env value
    turns the wall on, so wiring this in never silently locks anyone out.
    """
    raw = os.environ.get("SUBSCRIPTION_GATE_ENFORCED", "0").strip().lower()
    return raw in ("1", "true", "yes", "on")


def has_active_subscription(store: Any, account_id: int) -> bool:
    """True if the account has a non-expired active/trialing subscription.

    Reads only persisted state — no Stripe call. ``current_period_end`` is a
    safety net: even if a ``deleted`` webhook were missed, access lapses once the
    paid period ends.
    """
    if store is None:
        return False
    sub = store.get_subscription(account_id)
    if not sub:
        return False
    if sub.get("status") not in ACTIVE_STATUSES:
        return False
    period_end = sub.get("current_period_end")
    if period_end is not None and float(period_end) < time.time():
        return False
    return True


def account_has_access(account: Dict[str, Any], store: Any = None) -> bool:
    """Return True if the account may access gated features.

    * ``owner`` → always True (unconditional, permanent).
    * Gate not enforced (default, personal-testing phase) → True for any
      authenticated account.
    * Gate enforced → True only with a live subscription (see
      :func:`has_active_subscription`); requires ``store``.
    """
    if account.get("role") == "owner":
        return True
    if not _gate_enforced():
        return True
    return has_active_subscription(store, account["id"])


def _account_store(request: Request) -> Optional[Any]:
    return getattr(request.app.state.app_state, "account_store", None)


async def require_active_subscription(
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
) -> Dict[str, Any]:
    """Dependency: authenticated AND entitled (owner always passes).

    A non-entitled non-owner account gets 402 Payment Required when the gate is
    enforced; otherwise the gate is open and every authenticated account passes.
    """
    if not account_has_access(account, _account_store(request)):
        raise HTTPException(
            status_code=402,
            detail="An active subscription is required for this feature.",
        )
    return account
