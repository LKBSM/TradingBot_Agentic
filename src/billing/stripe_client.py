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

# Map Stripe price IDs → internal plan keys. Populated at boot from env.
def _build_price_to_plan() -> dict[str, str]:
    from src.billing.pricing import list_paid_plans
    out = {}
    for p in list_paid_plans():
        if p.stripe_price_id:
            out[p.stripe_price_id] = p.key
    return out


@dataclass(frozen=True)
class StripeWebhookEvent:
    event_type: str            # "customer.subscription.updated" etc.
    customer_id: str
    subscription_id: Optional[str]
    price_id: Optional[str]
    plan_key: Optional[str]    # resolved from price_id (MONTHLY / ANNUAL)
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

    price_to_plan = _build_price_to_plan()
    plan_key = price_to_plan.get(price_id) if price_id else None
    status = data.get("status")

    return StripeWebhookEvent(
        event_type=event_type,
        customer_id=customer_id,
        subscription_id=subscription_id,
        price_id=price_id,
        plan_key=plan_key,
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
    # Refund / dispute → suspend access (PAY-1). A charge doesn't carry the
    # subscription id or account metadata; the webhook route resolves the
    # account by ``customer`` id and this sets a non-active status.
    "charge.refunded",
    "charge.dispute.created",
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
    # Stripe event ``created`` (unix ts). Stripe delivers at-least-once and OUT
    # OF ORDER, so the store applies an event only when this is >= the newest
    # already applied — a stale event never overwrites newer state.
    event_created: Optional[float] = None


def _coerce_ts(value: Any) -> Optional[float]:
    """Coerce a Stripe unix timestamp to float, or None if absent/invalid."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    event_created = _coerce_ts(payload.get("created"))
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
            event_created=event_created,
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
            current_period_end=_coerce_ts(obj.get("current_period_end")),
            cancel_at_period_end=(
                bool(obj["cancel_at_period_end"])
                if obj.get("cancel_at_period_end") is not None
                else None
            ),
            trial_end=_coerce_ts(obj.get("trial_end")),
            event_created=event_created,
        )

    if event_type in ("charge.refunded", "charge.dispute.created"):
        # Refund/dispute → suspend. A dispute always suspends; a refund suspends
        # only when the charge is FULLY refunded (a partial refund is ignored so
        # a legit partial credit doesn't cut off a paying subscriber).
        if event_type == "charge.refunded":
            amount = obj.get("amount")
            refunded = obj.get("amount_refunded")
            fully_refunded = (
                amount is not None and refunded is not None and refunded >= amount
            )
            if not fully_refunded:
                return None  # partial refund → no state change
        return AccountSubscriptionEvent(
            event_id=event_id,
            event_type=event_type,
            account_id=None,  # charges carry no account metadata → resolve by customer
            customer_id=customer_id,
            subscription_id=None,  # not on a charge; upsert preserves the stored id
            status="suspended",
            price_id=None,
            current_period_end=None,
            cancel_at_period_end=None,
            trial_end=None,
            event_created=event_created,
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
        event_created=event_created,
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

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """Convert a Stripe SDK result into a PLAIN dict.

        PAY-3d: ``stripe.Customer.create`` / ``checkout.Session.create`` / … all
        return ``StripeObject`` instances whose ``.get()`` raises
        ``AttributeError`` (only subscript works). Every caller here treats the
        result as a plain dict and calls ``.get("id")`` / ``.get("url")`` — which
        used to 500 ("Internal Server Error") on the very first click of
        "Subscribe", even with a perfectly valid API key. Convert once, at the
        boundary, so callers always get a real dict (matching the fake client).
        """
        if isinstance(obj, dict):
            return obj
        to_dict = getattr(obj, "to_dict", None)
        if callable(to_dict):
            try:
                return dict(to_dict())
            except Exception:
                pass
        try:
            return dict(obj)
        except Exception:
            # Last resort: pull the ids we actually read off the attributes.
            return {
                k: getattr(obj, k, None)
                for k in ("id", "url", "status", "customer")
            }

    def create_customer(self, *, email: str, account_id: int) -> dict:
        """Create a Stripe customer carrying the account id in metadata.

        ``metadata.account_id`` is the durable join key used by the webhook to
        map Stripe events back to a local account (more robust than email).
        """
        stripe = self._require()
        return self._to_dict(stripe.Customer.create(
            email=email,
            metadata={"account_id": str(account_id)},
        ))

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
        subscription metadata so webhooks can resolve the account.

        PRIX-1: ``automatic_tax`` stays False — the price shown is the price paid,
        no tax is ever added at Checkout.
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
            # PRIX-1: no discount surface. The price shown is the price paid — no
            # promotion-code box on the hosted Checkout, no struck-through price.
            "allow_promotion_codes": False,
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
        return self._to_dict(stripe.checkout.Session.create(**params))

    def create_billing_portal_session(
        self, *, customer_id: str, return_url: str
    ) -> dict:
        """Create a Stripe Customer Portal session (hosted manage/cancel page)."""
        stripe = self._require()
        return self._to_dict(stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        ))

    def cancel_subscription(self, subscription_id: str) -> dict:
        stripe = self._require()
        return self._to_dict(stripe.Subscription.delete(subscription_id))

    # ------------------------------------------------------------------
    # Direct reconciliation (PAY-3e — webhook-independent fallback)
    # ------------------------------------------------------------------

    def find_customer_by_email(self, email: str) -> Optional[str]:
        """Return the id of the (most recent) Stripe customer for an email, or
        None. Used to relink an account whose customer id was lost."""
        stripe = self._require()
        result = stripe.Customer.list(email=email, limit=1)
        data = list(getattr(result, "data", []) or [])
        return getattr(data[0], "id", None) if data else None

    def get_subscription_state_for_customer(
        self, customer_id: str
    ) -> Optional[dict]:
        """Ask Stripe directly for the customer's current subscription and return
        the fields we persist — the webhook-INDEPENDENT path that grants access
        even when no webhook is configured (PAY-3e). None if the customer has no
        subscription.

        Reads via ATTRIBUTES (which work on a StripeObject; only ``.get()``
        raises), defensively, so a shape change never 500s.
        """
        stripe = self._require()
        result = stripe.Subscription.list(customer=customer_id, status="all", limit=10)
        subs = list(getattr(result, "data", []) or [])
        if not subs:
            return None

        # Prefer an access-granting status, then the most recently created.
        def _rank(s: Any) -> tuple:
            status = getattr(s, "status", "") or ""
            grants = {"active": 0, "trialing": 0, "past_due": 1}.get(status, 2)
            return (grants, -(getattr(s, "created", 0) or 0))

        subs.sort(key=_rank)
        s = subs[0]

        price_id = None
        try:
            items = getattr(getattr(s, "items", None), "data", None) or []
            if items:
                price_id = getattr(getattr(items[0], "price", None), "id", None)
        except Exception:
            price_id = None

        return {
            "subscription_id": getattr(s, "id", None),
            "status": getattr(s, "status", None),
            "current_period_end": getattr(s, "current_period_end", None),
            "cancel_at_period_end": bool(getattr(s, "cancel_at_period_end", False)),
            "trial_end": getattr(s, "trial_end", None),
            "price_id": price_id,
        }

    # ------------------------------------------------------------------
    # Webhook verification
    # ------------------------------------------------------------------

    def verify_webhook(self, *, body: bytes, signature: str) -> dict:
        """Verify the Stripe-Signature header and return the event as a plain dict.

        Raises ``ValueError`` on any failure — controllers should catch
        and respond with 400.

        PAY-3: ``construct_event`` returns a ``stripe.Event`` object whose
        ``.get()`` raises ``AttributeError`` — but every downstream parser
        (``parse_account_event`` / ``parse_webhook_event``) treats the result as
        a plain dict and calls ``.get(...)``. Passing the ``Event`` object through
        would 500 on EVERY real webhook, so the subscription would never persist
        and a paying customer would be locked out ("paid but no access"). The
        fake test client returns ``json.loads(body)``; we now return the SAME
        pure-dict shape after the signature check so prod and tests agree.
        """
        if not self._webhook_secret:
            raise RuntimeError(
                f"Stripe webhook secret not configured — set {STRIPE_WEBHOOK_SECRET_ENV}"
            )
        stripe = self._require()
        try:
            # Verifies the signature (raises on mismatch); we ignore the returned
            # Event object and use the raw, already-validated JSON body.
            stripe.Webhook.construct_event(
                payload=body,
                sig_header=signature,
                secret=self._webhook_secret,
            )
        except Exception as exc:
            raise ValueError(f"webhook verification failed: {exc}") from exc
        import json as _json
        return _json.loads(body.decode("utf-8") if isinstance(body, bytes) else body)


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
