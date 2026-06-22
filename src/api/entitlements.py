"""Entitlements — the freemium access policy layered on the subscription gate.

This module builds on :mod:`src.api.subscription_gate` (missions ①/②). That
module answers the binary "is this account paying / the owner?". THIS module
expresses the freemium PERIMETER — what a free *Découverte* account may see
versus a paying subscriber — and enforces it **server-side**, so the wall can
never be bypassed by editing the UI. Access is filtered; the detection engine
is NEVER touched.

Tiers (ordered)::

    VISITOR < FREE < SUBSCRIBER < OWNER

* ``VISITOR``    — not authenticated. When the gate is enforced, feature routes
                   answer 401 (login required).
* ``FREE``       — authenticated, no active subscription. Limited perimeter:
                   XAU/USD M15 reading + chart + a small daily chat quota; no
                   scanner, no other markets/timeframes.
* ``SUBSCRIBER`` — active/trialing subscription → the full product.
* ``OWNER``      — role ``owner`` → everything, always (the operator can never
                   lock themselves out).

POSTURE — consistent with ``SENTINEL_TESTING_MODE`` and the subscription gate:
while ``SUBSCRIPTION_GATE_ENFORCED`` is OFF (the default, personal-testing
phase) the feature routes stay fully OPEN and these checks are no-ops, so
nothing breaks during testing. Flipping ``SUBSCRIPTION_GATE_ENFORCED=1`` makes
the freemium perimeter bite. The machinery is fully wired either way.

Every limit is env-configurable, so the perimeter is policy — not hard-coded::

    FREE_INSTRUMENTS        comma list (default "XAUUSD")
    FREE_TIMEFRAMES         comma list (default "M15")
    FREE_CHAT_DAILY_LIMIT   int        (default 5)
    FREE_SCANNER_ENABLED    bool       (default 0)
"""

from __future__ import annotations

import enum
import logging
import os
import time
from typing import Any, Dict, FrozenSet, Optional

from fastapi import HTTPException, Request

from src.api.subscription_gate import _gate_enforced, has_active_subscription

logger = logging.getLogger(__name__)

_LOGIN_REQUIRED = "Authentification requise pour cette fonctionnalité."


class Tier(enum.IntEnum):
    """Access tiers, ordered so ``tier >= Tier.SUBSCRIBER`` means full access."""

    VISITOR = 0
    FREE = 1
    SUBSCRIBER = 2
    OWNER = 3

    @property
    def label(self) -> str:
        return self.name.lower()


# =============================================================================
# Env-driven free perimeter (policy, not hard-coded)
# =============================================================================

def _csv_env(name: str, default: str) -> FrozenSet[str]:
    raw = os.environ.get(name, default)
    return frozenset(part.strip().upper() for part in raw.split(",") if part.strip())


def free_instruments() -> FrozenSet[str]:
    return _csv_env("FREE_INSTRUMENTS", "XAUUSD")


def free_timeframes() -> FrozenSet[str]:
    return _csv_env("FREE_TIMEFRAMES", "M15")


def free_chat_daily_limit() -> int:
    try:
        return max(0, int(os.environ.get("FREE_CHAT_DAILY_LIMIT", "5")))
    except ValueError:
        return 5


def free_scanner_enabled() -> bool:
    return os.environ.get("FREE_SCANNER_ENABLED", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


# =============================================================================
# Tier resolution + perimeter predicates
# =============================================================================

def resolve_tier(account: Optional[Dict[str, Any]], store: Any = None) -> Tier:
    """Resolve the access tier of an account.

    Mirrors the subscription gate: the owner is always ``OWNER``; while the gate
    is OFF every authenticated account is treated as ``SUBSCRIBER`` (testing
    phase → full access); with the gate ON a non-owner account is ``SUBSCRIBER``
    only with a live subscription, otherwise ``FREE``. ``None`` → ``VISITOR``.
    """
    if account is None:
        return Tier.VISITOR
    if account.get("role") == "owner":
        return Tier.OWNER
    if not _gate_enforced():
        return Tier.SUBSCRIBER
    if has_active_subscription(store, account["id"]):
        return Tier.SUBSCRIBER
    return Tier.FREE


def is_full_access(tier: Tier) -> bool:
    return tier >= Tier.SUBSCRIBER


def tier_allows_instrument(tier: Tier, instrument: str) -> bool:
    if tier >= Tier.SUBSCRIBER:
        return True
    if tier == Tier.FREE:
        return instrument.upper() in free_instruments()
    return False


def tier_allows_combo(tier: Tier, instrument: str, timeframe: str) -> bool:
    if tier >= Tier.SUBSCRIBER:
        return True
    if tier == Tier.FREE:
        return (
            instrument.upper() in free_instruments()
            and timeframe.upper() in free_timeframes()
        )
    return False


def tier_allows_scanner(tier: Tier) -> bool:
    if tier >= Tier.SUBSCRIBER:
        return True
    if tier == Tier.FREE:
        return free_scanner_enabled()
    return False


def chat_daily_limit(tier: Tier) -> Optional[int]:
    """Messages/day the tier may send. ``None`` means unlimited."""
    if tier >= Tier.SUBSCRIBER:
        return None
    if tier == Tier.FREE:
        return free_chat_daily_limit()
    return 0  # VISITOR


def today_key() -> str:
    """UTC day bucket for the per-account chat counter (``YYYY-MM-DD``)."""
    return time.strftime("%Y-%m-%d", time.gmtime())


# =============================================================================
# Server-side guards (used inside feature routes)
#
# Each guard is a NO-OP while the gate is OFF, so the testing phase stays fully
# open and existing endpoint tests are unaffected. When the gate is ON they
# enforce login (401) then the freemium perimeter (402) — never a raw error.
# =============================================================================

def _store(request: Request) -> Any:
    return getattr(request.app.state.app_state, "account_store", None)


def _require_login(account: Optional[Dict[str, Any]]) -> None:
    if account is None:
        raise HTTPException(status_code=401, detail=_LOGIN_REQUIRED)


def enforce_combo_access(
    request: Request,
    account: Optional[Dict[str, Any]],
    instrument: str,
    timeframe: str,
) -> None:
    """Gate a (instrument, timeframe) reading/chart route."""
    if not _gate_enforced():
        return
    _require_login(account)
    tier = resolve_tier(account, _store(request))
    if not tier_allows_combo(tier, instrument, timeframe):
        raise HTTPException(
            status_code=402,
            detail=(
                f"L'accès à {instrument} {timeframe} nécessite un abonnement. "
                "Le palier gratuit couvre XAU/USD en M15."
            ),
        )


def enforce_instrument_access(
    request: Request,
    account: Optional[Dict[str, Any]],
    instrument: str,
) -> None:
    """Gate a per-instrument route (e.g. the live-price stream — no timeframe)."""
    if not _gate_enforced():
        return
    _require_login(account)
    tier = resolve_tier(account, _store(request))
    if not tier_allows_instrument(tier, instrument):
        raise HTTPException(
            status_code=402,
            detail=(
                f"Le suivi en direct de {instrument} nécessite un abonnement. "
                "Le palier gratuit couvre XAU/USD."
            ),
        )


def enforce_scanner_access(
    request: Request, account: Optional[Dict[str, Any]]
) -> None:
    """Gate the multi-market scanner (paid feature by default)."""
    if not _gate_enforced():
        return
    _require_login(account)
    tier = resolve_tier(account, _store(request))
    if not tier_allows_scanner(tier):
        raise HTTPException(
            status_code=402,
            detail="Le scanner multi-marchés nécessite un abonnement.",
        )


def enforce_chat_access(
    request: Request, account: Optional[Dict[str, Any]]
) -> Dict[str, Optional[int]]:
    """Gate a chat turn and count it against the daily quota.

    Returns the resulting quota snapshot ``{limit, used, remaining}`` (all
    ``None`` when unlimited). Raises 401 if login is required, 402 when the free
    quota is exhausted. A no-op (unlimited) while the gate is OFF.
    """
    unlimited: Dict[str, Optional[int]] = {"limit": None, "used": None, "remaining": None}
    if not _gate_enforced():
        return unlimited
    _require_login(account)
    store = _store(request)
    tier = resolve_tier(account, store)
    limit = chat_daily_limit(tier)
    if limit is None:
        return unlimited
    if limit <= 0:
        raise HTTPException(
            status_code=402, detail="Le chat M.I.A nécessite un abonnement."
        )
    day = today_key()
    used = store.get_chat_usage(account["id"], day)
    if used >= limit:
        raise HTTPException(
            status_code=402,
            detail=(
                f"Limite quotidienne de messages atteinte ({limit}/jour). "
                "Passe à l'abonnement pour un chat illimité."
            ),
        )
    new_used = store.increment_chat_usage(account["id"], day)
    return {"limit": limit, "used": new_used, "remaining": max(0, limit - new_used)}
