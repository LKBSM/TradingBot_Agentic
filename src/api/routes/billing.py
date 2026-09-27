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
from src.billing.pricing import get_plan, list_plans
from src.billing.stripe_client import parse_webhook_event

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/v1/billing", tags=["billing"])


class CheckoutBody(BaseModel):
    plan_key: str = Field(..., description="One of MONTHLY / ANNUAL")
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
    look up the local user by email and bind the new plan.
    """
    plan = get_plan(body.plan_key)
    if plan is None or plan.is_free or not plan.stripe_price_id:
        raise HTTPException(
            status_code=400, detail=f"unknown or non-purchasable plan: {body.plan_key}"
        )

    stripe = getattr(request.app.state.app_state, "stripe_client", None)
    if stripe is None or not stripe.is_configured:
        raise HTTPException(
            status_code=503, detail="Billing not configured"
        )

    session = stripe.create_checkout_session(
        price_id=plan.stripe_price_id,
        success_url=body.success_url,
        cancel_url=body.cancel_url,
        customer_email=body.email,
    )
    return {"checkout_url": session.get("url"), "session_id": session.get("id")}


@router.post("/webhook")
async def stripe_webhook(
    request: Request,
    stripe_signature: Optional[str] = Header(None, alias="Stripe-Signature"),
):
    """Point d'entrée RETIRÉ : vérifie la signature, puis refuse en 410.

    Le mur d'accès du produit lit ``AccountStore.subscriptions``, alimenté par
    ``/api/billing/webhook``. Voir le refus en fin de fonction.
    """
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

    # PAY-3 — LOUD guard against the most expensive misconfiguration. This is the
    # LEGACY B2B webhook: it updates ``tier_manager``, a state the account-based
    # paywall (``subscription_gate`` → ``AccountStore.subscriptions``) NEVER reads.
    # If Stripe is pointed here instead of ``/api/billing/webhook``, a real
    # payment would grant NO product access and fail SILENTLY. Make it loud so
    # ops sees it instead of chasing a mystery conversion drop for weeks.
    logger.error(
        "LEGACY /api/v1/billing/webhook received event %s — the account paywall "
        "does NOT read this state. If this is a real customer payment, Stripe is "
        "pointed at the WRONG endpoint: it MUST target /api/billing/webhook.",
        event.event_type,
    )

    # GO-LIVE — ce point d'entrée REFUSE désormais l'événement au lieu de
    # l'absorber.
    #
    # Avant : il journalisait l'erreur ci-dessus puis renvoyait 200. Stripe
    # considérait donc la livraison RÉUSSIE, ne réessayait jamais, et n'affichait
    # aucun échec dans son tableau de bord. Un vrai paiement encaissé n'ouvrait
    # aucun accès, et le seul indice était une ligne de log que personne ne lit.
    # C'est la panne la plus coûteuse possible sur un produit payant.
    #
    # Renvoyer 410 rend l'erreur VISIBLE là où le fondateur regarde déjà : Stripe
    # marque le point d'entrée en échec, réessaie, et l'événement n'est pas perdu
    # — il suffit de reconfigurer l'URL sur /api/billing/webhook et de le rejouer.
    # Le `tier_manager` legacy n'est plus alimenté : aucune surface du produit ne
    # lit cet état (le mur d'accès lit AccountStore.subscriptions), et la webapp
    # n'appelle jamais /api/v1/billing/*.
    raise HTTPException(
        status_code=410,
        detail=(
            "Legacy endpoint retired. Stripe must target /api/billing/webhook — "
            "this one does not grant product access. Repoint the webhook URL in "
            "the Stripe dashboard and replay the event."
        ),
    )


@router.get("/pricing")
async def get_pricing():
    """Public pricing table — the single USD plan (FREE + MONTHLY + ANNUAL)."""
    return {"plans": [p.to_dict() for p in list_plans()]}
