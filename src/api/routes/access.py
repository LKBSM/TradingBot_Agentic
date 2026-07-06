"""Access summary endpoint — what the current account is entitled to.

  GET /api/access/me   → the caller's tier + freemium entitlements

The webapp reads this ONCE to drive the access UI: which instruments/timeframes
are unlocked, whether the scanner is available, and the remaining daily chat
quota. It is the display-side companion to the server-side guards in
``entitlements`` — the guards are the source of truth (non-bypassable); this
endpoint only tells the UI what to show vs. lock behind an upsell.

It NEVER raises 401: an anonymous caller simply gets ``authenticated: false`` so
the frontend can route to login without treating it as an error. While the gate
is OFF (testing phase) everyone resolves to full access, exactly like the routes.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request

from src.api.entitlements import (
    Tier,
    chat_daily_limit,
    free_instruments,
    free_timeframes,
    is_full_access,
    resolve_tier,
    tier_allows_scanner,
    today_key,
)
from src.api.middleware.beta_auth import beta_lockdown_enabled
from src.api.session_auth import optional_account
from src.api.subscription_gate import _gate_enforced

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/access", tags=["access"])


def _store(request: Request) -> Any:
    return getattr(request.app.state.app_state, "account_store", None)


@router.get("/me")
async def access_me(
    request: Request,
    account: Optional[Dict[str, Any]] = Depends(optional_account),
) -> Dict[str, Any]:
    store = _store(request)
    gate_on = _gate_enforced()
    lockdown_on = beta_lockdown_enabled()
    tier = resolve_tier(account, store)
    # While the gate is OFF every (even anonymous) caller has the full product,
    # mirroring the feature routes that short-circuit when not enforced.
    full = (not gate_on) or is_full_access(tier)

    limit = None if full else chat_daily_limit(tier)
    used = (
        store.get_chat_usage(account["id"], today_key())
        if (account is not None and store is not None and limit is not None)
        else None
    )
    remaining = None if limit is None else max(0, limit - (used or 0))

    return {
        "authenticated": account is not None,
        "gate_enforced": gate_on,
        # Private-beta wall. When on, an anonymous caller MUST be routed to login:
        # the whole product API is 401 for them (see BetaAuthMiddleware). This is
        # independent of ``gate_enforced`` (the freemium/payment wall).
        "beta_lockdown": lockdown_on,
        "must_login": lockdown_on and account is None,
        "tier": tier.label,
        "is_owner": tier == Tier.OWNER,
        "has_full_access": full,
        "entitlements": {
            # None ⇒ unrestricted (full access). A list ⇒ the only allowed values.
            "instruments": None if full else sorted(free_instruments()),
            "timeframes": None if full else sorted(free_timeframes()),
            "scanner": True if full else tier_allows_scanner(tier),
            "chat": {"limit": limit, "used": used, "remaining": remaining},
        },
    }
