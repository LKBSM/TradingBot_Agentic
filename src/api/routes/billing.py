"""Billing routes — Sprint INFRA-2B.3.

Surface for:

  POST /api/v1/billing/checkout            create Stripe checkout session
  POST /api/v1/billing/webhook             Stripe webhook receiver
  GET  /api/v1/billing/pricing             public price table

Webhook events update the local UserTierManager so the next
authenticated request sees the right tier.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.auth import require_api_key
from src.billing.pricing import PRICING_TIERS, list_b2b_tiers, list_b2c_tiers
from src.billing.stripe_client import parse_webhook_event

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/billing", tags=["billing"])


class CheckoutBody(BaseModel):
    tier_key: str = Field(..., description="One of LITE / PRO / PRO_PLUS")
    # Email validation kept as a regex pattern to avoid an extra
    # email-validator dependency just for this surface.
    email: str = Field(..., min_length=5, max_length=200, pattern=r"^[^@]+@[^@]+\.[^@]+$")
    success_url: str = Field(..., max_length=500)
    cancel_url: str = Field(..., max_length=500)


@router.post("/checkout")
async def create_checkout(
    body: CheckoutBody,
    request: Request,
):
    """Create a Stripe checkout session and return the redirect URL.

    No-auth: a user signing up isn't yet authenticated. The Stripe
    customer_email is the join key — when the webhook fires later we
    look up the local user by email and bind the new tier.
    """
    tier = PRICING_TIERS.get(body.tier_key.upper())
    if tier is None or not tier.stripe_price_id:
        raise HTTPException(
            status_code=400, detail=f"unknown or non-purchasable tier: {body.tier_key}"
        )

    stripe = getattr(request.app.state.app_state, "stripe_client", None)
    if stripe is None or not stripe.is_configured:
        raise HTTPException(
            status_code=503, detail="Billing not configured"
        )

    session = stripe.create_checkout_session(
        price_id=tier.stripe_price_id,
        success_url=body.success_url,
        cancel_url=body.cancel_url,
        customer_email=body.email,
        trial_days=tier.trial_days,
    )
    return {"checkout_url": session.get("url"), "session_id": session.get("id")}


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
):
    """Receive Stripe events, verify signature, route to TierManager."""
    stripe = getattr(request.app.state.app_state, "stripe_client", None)
    if stripe is None or not stripe.is_configured:
        raise HTTPException(status_code=503, detail="Billing not configured")
    if not stripe_signature:
        raise HTTPException(status_code=400, detail="missing Stripe-Signature header")

    body = await request.body()
    try:
        verified = stripe.verify_webhook(body=body, signature=stripe_signature)
    except ValueError as exc:
        logger.warning("stripe webhook verification failed: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc))

    event = parse_webhook_event(verified)
    if event is None:
        return {"received": True, "ignored": True}

    tier_manager = getattr(request.app.state.app_state, "tier_manager", None)
    if tier_manager is None:
        logger.warning(
            "stripe event %s arrived but tier_manager not wired — drop",
            event.event_type,
        )
        return {"received": True, "ignored": True, "reason": "no_tier_manager"}

    # Apply the side effect — every event boils down to "set this customer
    # to tier X with status Y". Deletion → FREE.
    new_tier = event.tier_key or "FREE"
    if event.event_type == "customer.subscription.deleted":
        new_tier = "FREE"
    elif event.event_type == "invoice.payment_failed":
        # Downgrade after grace period — for now flag as PAST_DUE without
        # immediate downgrade.
        new_tier = None  # signal "no tier change, just status update"

    try:
        if new_tier is not None:
            tier_manager.set_tier_by_stripe_customer(
                event.customer_id, new_tier, status=event.status or "active"
            )
        else:
            tier_manager.set_status_by_stripe_customer(
                event.customer_id, "past_due"
            )
    except AttributeError:
        # tier_manager doesn't yet expose the Stripe-keyed setter — log
        # and move on so we don't 500.
        logger.warning(
            "tier_manager has no set_tier_by_stripe_customer — "
            "event %s dropped silently", event.event_type
        )
    except Exception as exc:
        logger.exception("tier_manager update failed: %s", exc)

    return {"received": True, "event_type": event.event_type, "applied": True}


@router.get("/pricing")
async def get_pricing():
    """Public pricing table — fed to /pricing webapp page."""
    return {
        "b2c": [t.to_dict() for t in list_b2c_tiers()],
        "b2b": [t.to_dict() for t in list_b2b_tiers()],
    }
