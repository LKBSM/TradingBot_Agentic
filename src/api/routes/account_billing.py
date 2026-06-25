"""Account-bound subscription billing (payments mission ②).

This is the NEW, account-centric billing surface — distinct from the legacy
tier/API-key billing at ``/api/v1/billing`` (which stays for B2B). Every route
here is tied to the authenticated ACCOUNT (session cookie) and reflects state on
the account's ``subscriptions`` row via Stripe webhooks.

  GET  /api/billing/pricing       public — configured plans (env price IDs)
  POST /api/billing/checkout      auth — create/reuse customer + Checkout session
  POST /api/billing/portal        auth — Stripe Customer Portal (manage/cancel)
  GET  /api/billing/subscription  auth — current subscription state + access
  POST /api/billing/webhook       Stripe — signed, idempotent → update account

SECURITY
--------
* Stripe keys live in the environment (``StripeClient``); none in code.
* The webhook signature is verified before ANY state change; an invalid
  signature is a hard 400.
* NO card data is ever received or stored — Checkout + Portal are hosted by
  Stripe, so card details never touch this service. We persist only Stripe IDs
  and the subscription status.
* Webhooks are idempotent: each Stripe ``event.id`` is applied at most once
  (``AccountStore.mark_webhook_processed``).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.account_store import AccountStore
from src.api.session_auth import require_account
from src.api.subscription_gate import account_has_access
from src.billing.stripe_client import parse_account_event

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/billing", tags=["billing-account"])


# =============================================================================
# Configurable plan catalogue (env-driven — the GRID is decided in Stripe)
# =============================================================================
# Plan key → env var holding its Stripe price id. Only plans whose env var is
# set are offered. Labels/amounts are intentionally NOT hard-coded here; the
# webapp can render names from Stripe or a future config without code changes.
_PLAN_ENV: Dict[str, str] = {
    "STANDARD": "STRIPE_PRICE_STANDARD",
    "PREMIUM": "STRIPE_PRICE_PREMIUM",
}


def _configured_plans() -> List[Dict[str, str]]:
    """Return the list of purchasable plans whose price id is configured."""
    plans: List[Dict[str, str]] = []
    for key, env_var in _PLAN_ENV.items():
        price_id = os.environ.get(env_var)
        if price_id:
            plans.append({"key": key, "price_id": price_id})
    return plans


def _configured_price_ids() -> set[str]:
    return {p["price_id"] for p in _configured_plans()}


def _trial_days() -> int:
    try:
        return max(0, int(os.environ.get("STRIPE_TRIAL_DAYS", "0")))
    except ValueError:
        return 0


def _tax_enabled() -> bool:
    return os.environ.get("STRIPE_TAX_ENABLED", "0").strip().lower() in (
        "1", "true", "yes", "on",
    )


def _success_url() -> str:
    return os.environ.get(
        "STRIPE_SUCCESS_URL", "http://localhost:3000/abonnement?status=success"
    )


def _cancel_url() -> str:
    return os.environ.get(
        "STRIPE_CANCEL_URL", "http://localhost:3000/abonnement?status=cancel"
    )


def _portal_return_url() -> str:
    return os.environ.get(
        "STRIPE_PORTAL_RETURN_URL", "http://localhost:3000/abonnement"
    )


# =============================================================================
# Schemas
# =============================================================================

class CheckoutBody(BaseModel):
    # Either a known plan key OR a raw price id (validated against the
    # configured set so a client can never check out an arbitrary price).
    plan_key: Optional[str] = Field(None, max_length=64)
    price_id: Optional[str] = Field(None, max_length=255)


class RedirectOut(BaseModel):
    url: str


class SubscriptionOut(BaseModel):
    status: Optional[str] = None
    price_id: Optional[str] = None
    current_period_end: Optional[float] = None
    cancel_at_period_end: bool = False
    trial_end: Optional[float] = None
    has_access: bool = False


# =============================================================================
# Helpers
# =============================================================================

def _store(request: Request) -> AccountStore:
    store = getattr(request.app.state.app_state, "account_store", None)
    if store is None:
        raise HTTPException(status_code=503, detail="Account service unavailable")
    return store


def _stripe(request: Request):
    client = getattr(request.app.state.app_state, "stripe_client", None)
    if client is None or not client.is_configured:
        raise HTTPException(status_code=503, detail="Billing not configured")
    return client


def _resolve_price_id(body: CheckoutBody) -> str:
    configured = _configured_plans()
    if not configured:
        raise HTTPException(status_code=503, detail="No purchasable plan configured")
    by_key = {p["key"]: p["price_id"] for p in configured}
    if body.plan_key:
        price_id = by_key.get(body.plan_key.upper())
        if not price_id:
            raise HTTPException(status_code=400, detail="Unknown plan")
        return price_id
    if body.price_id:
        if body.price_id not in _configured_price_ids():
            raise HTTPException(status_code=400, detail="Unknown or non-purchasable price")
        return body.price_id
    raise HTTPException(status_code=400, detail="plan_key or price_id is required")


# =============================================================================
# Routes
# =============================================================================

@router.get("/pricing")
async def pricing():
    """Public — the plans currently purchasable (price ids configured in env)."""
    return {
        "plans": _configured_plans(),
        "trial_days": _trial_days(),
        "tax_enabled": _tax_enabled(),
    }


@router.post("/checkout", response_model=RedirectOut)
async def checkout(
    body: CheckoutBody,
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
):
    """Create a Stripe Checkout session for the authenticated account.

    Reuses the account's existing Stripe customer when one is linked, else
    creates one (carrying ``account_id`` in metadata) and binds it. The hosted
    Checkout page collects payment — no card data passes through here.
    """
    store = _store(request)
    stripe = _stripe(request)
    price_id = _resolve_price_id(body)

    sub = store.get_subscription(account["id"])
    customer_id = sub.get("stripe_customer_id") if sub else None
    if not customer_id:
        customer = stripe.create_customer(
            email=account["email"], account_id=account["id"]
        )
        customer_id = customer.get("id")
        if not customer_id:
            raise HTTPException(status_code=502, detail="Stripe customer creation failed")
        store.link_stripe_customer(account["id"], customer_id)

    try:
        session = stripe.create_checkout_session(
            price_id=price_id,
            success_url=_success_url(),
            cancel_url=_cancel_url(),
            customer=customer_id,
            account_id=account["id"],
            trial_days=_trial_days(),
            automatic_tax=_tax_enabled(),
        )
    except Exception as exc:  # Stripe/network error — never leak internals
        logger.exception("checkout session creation failed for account=%s", account["id"])
        raise HTTPException(status_code=502, detail="Could not start checkout") from exc

    url = session.get("url")
    if not url:
        raise HTTPException(status_code=502, detail="Checkout session has no URL")
    return RedirectOut(url=url)


@router.post("/portal", response_model=RedirectOut)
async def portal(
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
):
    """Open the Stripe Customer Portal (hosted manage/cancel/payment-method)."""
    store = _store(request)
    stripe = _stripe(request)
    sub = store.get_subscription(account["id"])
    customer_id = sub.get("stripe_customer_id") if sub else None
    if not customer_id:
        raise HTTPException(status_code=409, detail="No billing account yet")
    try:
        session = stripe.create_billing_portal_session(
            customer_id=customer_id, return_url=_portal_return_url()
        )
    except Exception as exc:
        logger.exception("portal session creation failed for account=%s", account["id"])
        raise HTTPException(status_code=502, detail="Could not open billing portal") from exc
    url = session.get("url")
    if not url:
        raise HTTPException(status_code=502, detail="Portal session has no URL")
    return RedirectOut(url=url)


@router.get("/subscription", response_model=SubscriptionOut)
async def subscription(
    request: Request,
    account: Dict[str, Any] = Depends(require_account),
):
    """Return the account's current subscription state + resolved access."""
    store = _store(request)
    sub = store.get_subscription(account["id"]) or {}
    return SubscriptionOut(
        status=sub.get("status"),
        price_id=sub.get("price_id"),
        current_period_end=sub.get("current_period_end"),
        cancel_at_period_end=bool(sub.get("cancel_at_period_end")),
        trial_end=sub.get("trial_end"),
        has_access=account_has_access(account, store),
    )


@router.post("/webhook")
async def webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
):
    """Receive Stripe events, verify signature, reflect on the account.

    Idempotent (each event id applied once). An invalid/missing signature is a
    hard 400 — no state change. Unresolvable events (no account) are NOT marked
    processed, so a later retry can succeed once the customer is linked.
    """
    stripe = getattr(request.app.state.app_state, "stripe_client", None)
    if stripe is None or not stripe.is_configured:
        raise HTTPException(status_code=503, detail="Billing not configured")
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="Missing Stripe-Signature header")

    body = await request.body()
    try:
        verified = stripe.verify_webhook(body=body, signature=stripe_signature)
    except ValueError as exc:
        logger.warning("stripe webhook verification failed: %s", exc)
        raise HTTPException(status_code=400, detail="Invalid signature")

    event = parse_account_event(verified)
    if event is None:
        return {"received": True, "ignored": True}

    store = _store(request)

    # Resolve the account FIRST — by metadata, then by customer id.
    account_id = event.account_id
    if account_id is None and event.customer_id:
        acct = store.get_account_by_stripe_customer(event.customer_id)
        account_id = acct["id"] if acct else None
    if account_id is None:
        # Don't claim the event — allow Stripe's retry to land once linked.
        logger.warning(
            "stripe event %s (%s) could not be mapped to an account — not claimed",
            event.event_id, event.event_type,
        )
        return {"received": True, "unresolved": True}

    # Claim the event id; a duplicate delivery is a no-op (idempotency).
    if not store.mark_webhook_processed(event.event_id, event.event_type):
        return {"received": True, "duplicate": True}

    if event.event_type == "checkout.session.completed":
        # Bind customer↔account; the subscription.* events carry the full state.
        if event.customer_id:
            store.link_stripe_customer(account_id, event.customer_id)
    else:
        store.upsert_subscription(
            account_id,
            stripe_customer_id=event.customer_id or None,
            stripe_subscription_id=event.subscription_id,
            status=event.status,
            price_id=event.price_id,
            current_period_end=event.current_period_end,
            cancel_at_period_end=event.cancel_at_period_end,
            trial_end=event.trial_end,
        )

    return {"received": True, "event_type": event.event_type, "applied": True}
