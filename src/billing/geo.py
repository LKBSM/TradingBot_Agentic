"""Billing geography — the subscription is sold in Canada and the United States only.

Why this module exists
----------------------
Stripe Checkout has **no native allow-list of billing countries** for a
subscription (``shipping_address_collection.allowed_countries`` restricts a
shipping address, not the card). The restriction is therefore enforced in TWO
places, and both are needed:

1. **Stripe Radar** (dashboard, operator action) — a rule such as
   ``:card_country: not in ('CA','US')`` blocks the payment before it succeeds.
   This is the layer that prevents the charge; see ``docs/audits/AUDIT-pay-3.md``.
2. **The webhook net** (this code) — Checkout is created with
   ``billing_address_collection="required"``, so every completed session carries
   a billing country. If that country is outside the zone, the webhook cancels
   the subscription immediately and records a non-access status. This is the
   backstop for the day the Radar rule is missing, disabled, or bypassed.

Layer 2 alone would let the money land before being refunded; layer 1 alone is a
dashboard setting no test can see. Together they fail closed.

The zone is overridable via ``BILLING_ALLOWED_COUNTRIES`` (comma-separated ISO
3166-1 alpha-2) so a future expansion is a config change, not a code change. An
empty/blank value falls back to the default zone — it never means "everywhere",
because a typo in an env var must not silently open the world.
"""

from __future__ import annotations

import os
from typing import FrozenSet, Optional

# The zone the product is sold in today (PAY-3). Canada + United States.
DEFAULT_ALLOWED_COUNTRIES: FrozenSet[str] = frozenset({"CA", "US"})

ALLOWED_COUNTRIES_ENV = "BILLING_ALLOWED_COUNTRIES"

# Status persisted on a subscription that was bought from outside the zone. It is
# deliberately NOT in ``subscription_gate.ACTIVE_STATUSES``, so it grants nothing.
BLOCKED_REGION_STATUS = "blocked_region"


def allowed_countries() -> FrozenSet[str]:
    """The billing countries allowed to subscribe (upper-case ISO alpha-2)."""
    raw = os.environ.get(ALLOWED_COUNTRIES_ENV, "")
    codes = {
        part.strip().upper()
        for part in raw.split(",")
        if part.strip()
    }
    return frozenset(codes) if codes else DEFAULT_ALLOWED_COUNTRIES


def is_allowed_country(code: Optional[str]) -> bool:
    """Whether a billing country may subscribe.

    An UNKNOWN country (None/blank) returns True on purpose: we must never lock
    out a legitimate customer because Stripe didn't send an address on some event
    shape. The blocking decision is only ever taken on a country we actually read
    — combined with Radar (layer 1), which needs no address to block a card.
    """
    if not code:
        return True
    return code.strip().upper() in allowed_countries()


def billing_country_from_session(session: dict) -> Optional[str]:
    """Extract the billing country from a ``checkout.session.completed`` object.

    Stripe puts the collected address under ``customer_details.address.country``;
    older/alternate shapes use a top-level ``customer_address``. Returns None when
    no country is present (see :func:`is_allowed_country` for why that is safe).
    """
    if not isinstance(session, dict):
        return None
    for holder in (session.get("customer_details"), session):
        if not isinstance(holder, dict):
            continue
        address = holder.get("address") or holder.get("customer_address")
        if isinstance(address, dict):
            country = address.get("country")
            if country:
                return str(country).strip().upper()
    return None


__all__ = [
    "ALLOWED_COUNTRIES_ENV",
    "BLOCKED_REGION_STATUS",
    "DEFAULT_ALLOWED_COUNTRIES",
    "allowed_countries",
    "billing_country_from_session",
    "is_allowed_country",
]
