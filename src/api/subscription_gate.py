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


# Grace window (days) after a FAILED renewal before access is cut. While Stripe
# runs Smart Retries the subscription sits in ``past_due``; PAY-1 requires a
# grace period, not immediate suspension. Stripe's own dunning ultimately moves
# the subscription to ``canceled``/``unpaid`` (removing access); this bound is a
# safety net so a lingering ``past_due`` can never grant access forever.
_DEFAULT_GRACE_DAYS = 7


def _grace_seconds() -> float:
    try:
        days = float(os.environ.get("SUBSCRIPTION_GRACE_DAYS", _DEFAULT_GRACE_DAYS))
    except ValueError:
        days = _DEFAULT_GRACE_DAYS
    return max(0.0, days) * 86400.0


def has_active_subscription(store: Any, account_id: int) -> bool:
    """True if the account's persisted subscription currently grants access.

    Reads only persisted state — no Stripe call. Access maps to the app-facing
    states (PAY-1):

    * ``active`` / ``trialing`` → access while the paid period has not lapsed
      (``current_period_end`` is a safety net for a missed ``deleted`` webhook).
    * ``past_due`` → access during the GRACE window after the period end (a
      failed renewal is a grace period, not an immediate cut).
    * anything else (``canceled``, ``unpaid``, ``incomplete``, ``suspended`` for
      refund/dispute, …) → NO access.
    """
    if store is None:
        return False
    sub = store.get_subscription(account_id)
    if not sub:
        return False
    status = sub.get("status")
    period_end = sub.get("current_period_end")
    now = time.time()

    if status in ACTIVE_STATUSES:
        if period_end is not None and float(period_end) < now:
            return False
        return True

    if status == "past_due":
        # Grace: keep access for a bounded window after the (failed) period end.
        if period_end is None:
            return True
        return now < float(period_end) + _grace_seconds()

    return False


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


# =============================================================================
# The single inline access guard for gated DATA routes (PAY-1: paid-only).
#
# PAY-1 removed the freemium perimeter: there is no partial/free access. A
# gated route is either fully open (gate OFF, personal-testing) or requires an
# authenticated account WITH an active subscription (owner always passes). This
# one function is the ONLY place a data route asks "may this caller see market
# data?" — the entitlements module and its instrument/timeframe/scanner/chat
# perimeter are gone, so access is decided in exactly one seam.
# =============================================================================

_LOGIN_REQUIRED = "Authentication required for this feature."
_SUBSCRIPTION_REQUIRED = "An active subscription is required for this feature."


def enforce_access(request: Request, account: Optional[Dict[str, Any]]) -> None:
    """Guard a market-data route (paid-only, all-or-nothing).

    * Gate OFF (default, personal-testing) → no-op, route stays open.
    * Gate ON, not authenticated → 401 (the caller must log in).
    * Gate ON, authenticated without access → 402 (subscribe to unlock).
    * Owner and active/trialing subscribers pass.

    No instrument/timeframe/feature distinction: an unsubscribed account sees no
    market data at all — not even a single candle (PAY-1).
    """
    if not _gate_enforced():
        return
    if account is None:
        raise HTTPException(status_code=401, detail=_LOGIN_REQUIRED)
    if not account_has_access(account, _account_store(request)):
        raise HTTPException(status_code=402, detail=_SUBSCRIPTION_REQUIRED)


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
