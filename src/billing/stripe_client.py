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


# Events that drive the ACCOUNT subscription state (payments mission ②). Distinct
# from the legacy tier-keyed ``parse_webhook_event`` above.
ACCOUNT_SUBSCRIPTION_EVENTS = frozenset({
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.paid",
    "invoice.payment_succeeded",
    "invoice.payment_failed",
})


@dataclass(frozen=True)
class AccountSubscriptionEvent:
    """Account-centric projection of a verified Stripe event.

    ``account_id`` is resolved from metadata when present (None otherwise — the
    route then falls back to a customer-id lookup). ``status`` is the subscription
    status to persist; it is derived for events that don't carry one directly
    (deleted → ``canceled``, payment_failed → ``past_due``).
    """
    event_id: str
    event_type: str
    account_id: Optional[int]
    customer_id: str
    subscription_id: Optional[str]
    status: Optional[str]
    price_id: Optional[str]
    current_period_end: Optional[float]
    cancel_at_period_end: Optional[bool]
    trial_end: Optional[float]


def _coerce_account_id(meta: Any) -> Optional[int]:
    if not isinstance(meta, dict):
        return None
    raw = meta.get("account_id")
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _first_price_id(obj: dict) -> Optional[str]:
    items = obj.get("items")
    if isinstance(items, dict):
        data = items.get("data") or []
        if data:
            return ((data[0] or {}).get("price") or {}).get("id")
    lines = obj.get("lines")
    if isinstance(lines, dict):
        data = lines.get("data") or []
        if data:
            return ((data[0] or {}).get("price") or {}).get("id")
    return None


def parse_account_event(payload: dict) -> Optional[AccountSubscriptionEvent]:
    """Project a verified Stripe event onto the account subscription shape.

    Returns None for events outside :data:`ACCOUNT_SUBSCRIPTION_EVENTS`.
    """
    event_type = payload.get("type", "")
    if event_type not in ACCOUNT_SUBSCRIPTION_EVENTS:
        return None
    event_id = str(payload.get("id", ""))
    obj = (payload.get("data", {}) or {}).get("object", {}) or {}
    customer_id = str(obj.get("customer", "") or "")

    if event_type == "checkout.session.completed":
        # Linkage event: bind customer↔account; full state arrives via the
        # subscription.* events. account_id comes from client_reference_id/metadata.
        account_id = _coerce_account_id(obj.get("metadata"))
        if account_id is None and obj.get("client_reference_id"):
            try:
                account_id = int(obj["client_reference_id"])
            except (TypeError, ValueError):
                account_id = None
        return AccountSubscriptionEvent(
            event_id=event_id,
            event_type=event_type,
            account_id=account_id,
            customer_id=customer_id,
            subscription_id=str(obj.get("subscription") or "") or None,
            status=None,
            price_id=None,
            current_period_end=None,
            cancel_at_period_end=None,
            trial_end=None,
        )

    if event_type.startswith("customer.subscription."):
        status = obj.get("status")
        if event_type == "customer.subscription.deleted":
            status = "canceled"
        return AccountSubscriptionEvent(
            event_id=event_id,
            event_type=event_type,
            account_id=_coerce_account_id(obj.get("metadata")),
            customer_id=customer_id,
            subscription_id=str(obj.get("id") or "") or None,
            status=status,
            price_id=_first_price_id(obj),
            current_period_end=obj.get("current_period_end"),
            cancel_at_period_end=(
                bool(obj["cancel_at_period_end"])
                if obj.get("cancel_at_period_end") is not None
                else None
            ),
            trial_end=obj.get("trial_end"),
        )

    # invoice.* — carries customer + subscription id; status is derived.
    derived_status = "past_due" if event_type == "invoice.payment_failed" else "active"
    return AccountSubscriptionEvent(
        event_id=event_id,
        event_type=event_type,
        account_id=_coerce_account_id(obj.get("subscription_details", {}).get("metadata"))
        if isinstance(obj.get("subscription_details"), dict)
        else None,
        customer_id=customer_id,
        subscription_id=str(obj.get("subscription") or "") or None,
        status=derived_status,
        price_id=_first_price_id(obj),
        current_period_end=None,
        cancel_at_period_end=None,
        trial_end=None,
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

    def create_customer(self, *, email: str, account_id: int) -> dict:
        """Create a Stripe customer carrying the account id in metadata.

        ``metadata.account_id`` is the durable join key used by the webhook to
        map Stripe events back to a local account (more robust than email).
        """
        stripe = self._require()
        return stripe.Customer.create(
            email=email,
            metadata={"account_id": str(account_id)},
        )

    def create_checkout_session(
        self,
        *,
        price_id: str,
        success_url: str,
        cancel_url: str,
        customer_email: Optional[str] = None,
        customer: Optional[str] = None,
        account_id: Optional[int] = None,
        trial_days: int = 0,
        automatic_tax: bool = False,
    ) -> dict:
        """Create a subscription Checkout session.

        Pass EITHER an existing ``customer`` id (preferred — keeps one customer
        per account) OR a ``customer_email`` (Stripe creates the customer). When
        ``account_id`` is given it is stamped on the session AND propagated to the
        subscription metadata so webhooks can resolve the account. ``automatic_tax``
        turns on Stripe Tax (TPS/TVQ etc.) — it also requires collecting the
        customer's billing address, which Checkout does automatically when on.
        """
        stripe = self._require()
        sub_data: dict = {}
        if trial_days > 0:
            sub_data["trial_period_days"] = trial_days
        if account_id is not None:
            sub_data["metadata"] = {"account_id": str(account_id)}

        params: dict = {
            "mode": "subscription",
            "line_items": [{"price": price_id, "quantity": 1}],
            "success_url": success_url,
            "cancel_url": cancel_url,
            "allow_promotion_codes": True,
        }
        if customer:
            params["customer"] = customer
            # Let Stripe Tax save the address it collects back onto the customer.
            if automatic_tax:
                params["customer_update"] = {"address": "auto"}
        elif customer_email:
            params["customer_email"] = customer_email
        if account_id is not None:
            params["client_reference_id"] = str(account_id)
        if sub_data:
            params["subscription_data"] = sub_data
        if automatic_tax:
            params["automatic_tax"] = {"enabled": True}
        return stripe.checkout.Session.create(**params)

    def create_billing_portal_session(
        self, *, customer_id: str, return_url: str
    ) -> dict:
        """Create a Stripe Customer Portal session (hosted manage/cancel page)."""
        stripe = self._require()
        return stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )

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
    "ACCOUNT_SUBSCRIPTION_EVENTS",
    "STRIPE_API_KEY_ENV",
    "STRIPE_WEBHOOK_SECRET_ENV",
    "AccountSubscriptionEvent",
    "StripeClient",
    "StripeWebhookEvent",
    "parse_account_event",
    "parse_webhook_event",
]
