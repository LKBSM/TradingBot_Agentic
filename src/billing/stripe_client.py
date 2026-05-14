"""Stripe wrapper — Sprint INFRA-2B.3.

Thin wrapper over the official ``stripe`` SDK with:

- env-driven configuration (no creds in code),
- a deterministic webhook event parser (verifies signature),
- a graceful "Stripe not configured" fallback for dev / CI runs
  without an API key.

The handler routes the four Stripe events we care about:

  - customer.subscription.created
  - customer.subscription.updated
  - customer.subscription.deleted
  - invoice.payment_failed

Each event updates the local ``UserTierManager`` so that
``require_api_key`` sees the right tier on the next request.

This module *does not* require ``stripe`` at import time — calls into
the SDK happen lazily inside method bodies so test environments
without the package still import cleanly.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


STRIPE_API_KEY_ENV = "STRIPE_SECRET_KEY"
STRIPE_WEBHOOK_SECRET_ENV = "STRIPE_WEBHOOK_SECRET"

# Map Stripe price IDs → internal tier keys. Populated at boot from env.
def _build_price_to_tier() -> dict[str, str]:
    from src.billing.pricing import PRICING_TIERS
    out = {}
    for t in PRICING_TIERS.values():
        if t.stripe_price_id:
            out[t.stripe_price_id] = t.key
    return out


@dataclass(frozen=True)
class StripeWebhookEvent:
    event_type: str            # "customer.subscription.updated" etc.
    customer_id: str
    subscription_id: Optional[str]
    price_id: Optional[str]
    tier_key: Optional[str]    # resolved from price_id via PRICING_TIERS
    status: Optional[str]      # active / trialing / past_due / canceled
    raw: dict


def parse_webhook_event(payload: dict) -> Optional[StripeWebhookEvent]:
    """Convert a verified Stripe event payload into the internal shape.

    Returns None for events we don't care about.
    """
    event_type = payload.get("type", "")
    if event_type not in {
        "customer.subscription.created",
        "customer.subscription.updated",
        "customer.subscription.deleted",
        "invoice.payment_failed",
    }:
        return None

    data = payload.get("data", {}).get("object", {}) or {}
    customer_id = str(data.get("customer", ""))
    subscription_id = data.get("id") if event_type.startswith("customer.subscription") else data.get("subscription")
    items = (data.get("items", {}) or {}).get("data", []) if isinstance(data.get("items"), dict) else []
    price_id = None
    if items:
        price = items[0].get("price", {}) or {}
        price_id = price.get("id")
    elif "lines" in data and isinstance(data["lines"], dict):
        # invoice.payment_failed nests prices under lines.data[0].price
        lines = data["lines"].get("data", [])
        if lines:
            price_id = (lines[0].get("price") or {}).get("id")

    price_to_tier = _build_price_to_tier()
    tier_key = price_to_tier.get(price_id) if price_id else None
    status = data.get("status")

    return StripeWebhookEvent(
        event_type=event_type,
        customer_id=customer_id,
        subscription_id=subscription_id,
        price_id=price_id,
        tier_key=tier_key,
        status=status,
        raw=payload,
    )


class StripeClient:
    """Lazy-init wrapper. ``is_configured`` is False when ``STRIPE_SECRET_KEY``
    is unset — every method raises in that case so a misconfigured deploy
    fails loudly rather than silently dropping events."""

    def __init__(self, *, api_key: Optional[str] = None, webhook_secret: Optional[str] = None):
        self._api_key = api_key or os.environ.get(STRIPE_API_KEY_ENV)
        self._webhook_secret = webhook_secret or os.environ.get(STRIPE_WEBHOOK_SECRET_ENV)

    @property
    def is_configured(self) -> bool:
        return bool(self._api_key)

    def _require(self) -> Any:
        if not self.is_configured:
            raise RuntimeError(
                f"Stripe not configured — set {STRIPE_API_KEY_ENV}"
            )
        import stripe  # type: ignore
        stripe.api_key = self._api_key
        return stripe

    # ------------------------------------------------------------------
    # Customer + checkout session
    # ------------------------------------------------------------------

    def create_checkout_session(
        self,
        *,
        price_id: str,
        success_url: str,
        cancel_url: str,
        customer_email: str,
        trial_days: int = 0,
    ) -> dict:
        stripe = self._require()
        params = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "customer_email": customer_email,
            "allow_promotion_codes": True,
        }
        if trial_days > 0:
            params["subscription_data"] = {"trial_period_days": trial_days}
        return stripe.checkout.Session.create(**params)

    def cancel_subscription(self, subscription_id: str) -> dict:
        stripe = self._require()
        return stripe.Subscription.delete(subscription_id)

    # ------------------------------------------------------------------
    # Webhook verification
    # ------------------------------------------------------------------

    def verify_webhook(self, *, body: bytes, signature: str) -> dict:
        """Verify the Stripe-Signature header and return the parsed event.

        Raises ``ValueError`` on any failure — controllers should catch
        and respond with 400.
        """
        if not self._webhook_secret:
            raise RuntimeError(
                f"Stripe webhook secret not configured — set {STRIPE_WEBHOOK_SECRET_ENV}"
            )
        stripe = self._require()
        try:
            return stripe.Webhook.construct_event(
                payload=body,
                sig_header=signature,
                secret=self._webhook_secret,
            )
        except Exception as exc:
            raise ValueError(f"webhook verification failed: {exc}") from exc


__all__ = [
    "STRIPE_API_KEY_ENV",
    "STRIPE_WEBHOOK_SECRET_ENV",
    "StripeClient",
    "StripeWebhookEvent",
    "parse_webhook_event",
]
