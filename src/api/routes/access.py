"""Access summary endpoint — whether the current account may see market data.

  GET /api/access/me   → the caller's authentication + access state

PAY-1 is paid-only: there is no free perimeter. Access is all-or-nothing, so
this endpoint reports a single boolean ``has_access`` (plus the flags the UI
needs to route: authenticated / gate_enforced / beta_lockdown / must_login /
is_owner). The webapp reads this ONCE to decide between the product and the
"subscribe" invitation — but it is only a DISPLAY hint: the server-side guard
:func:`src.api.subscription_gate.enforce_access` is the non-bypassable source of
truth on every data route.

It NEVER raises 401: an anonymous caller simply gets ``authenticated: false`` so
the frontend can route to login without treating it as an error. While the gate
is OFF (testing phase) everyone resolves to full access, exactly like the routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request

from src.api.middleware.beta_auth import beta_lockdown_enabled
from src.api.session_auth import optional_account
from src.api.subscription_gate import (
    _account_store,
    _gate_enforced,
    account_has_access,
    email_verified,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/access", tags=["access"])


@router.get("/me")
async def access_me(
    request: Request,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> Dict[str, Any]:
    store = _account_store(request)
    gate_on = _gate_enforced()
    lockdown_on = beta_lockdown_enabled()

    is_owner = bool(account and account.get("role") == "owner")
    is_verified = account is not None and email_verified(account)

    # Paid-only, all-or-nothing. While the gate is OFF every (even anonymous)
    # caller has the full product, mirroring the data routes that short-circuit
    # when not enforced. When ON, only an authenticated, email-verified account
    # with an active subscription (or the owner) has access.
    if not gate_on:
        has_access = True
    elif account is None or not is_verified:
        has_access = False
    else:
        has_access = account_has_access(account, store)

    # Verification wall comes BEFORE the subscribe wall: an authenticated but
    # unverified account is told to confirm its email first.
    verify_required = gate_on and account is not None and not is_verified

    return {
        "authenticated": account is not None,
        "gate_enforced": gate_on,
        # Private-beta wall. When on, an anonymous caller MUST be routed to login:
        # the whole product API is 401 for them (see BetaAuthMiddleware). This is
        # independent of ``gate_enforced`` (the payment wall).
        "beta_lockdown": lockdown_on,
        "must_login": lockdown_on and account is None,
        "is_owner": is_owner,
        # Mandatory email verification (PAY-1).
        "email_verified": is_verified,
        "email_verification_required": verify_required,
        # The single access boolean the UI branches on: product vs. subscribe.
        "has_access": has_access,
        # An authenticated, VERIFIED account that is logged in but not entitled —
        # the "account page + subscribe invitation" state (PAY-1).
        "subscription_required": (
            gate_on and account is not None and is_verified and not has_access
        ),
    }
