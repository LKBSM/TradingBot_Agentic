"""Pricing grid — Sprint INFRA-2B.3.

Phase 2B 4-tier grid (eval_27 v1):

    FREE         €0/mo     — public surface + 1 chat/day
    LITE         €19/mo    — 10 chats/day + Telegram delivery
    PRO          €39/mo    — unlimited chat, multi-asset, read-only API
    PRO+         €99/mo    — full API, regime + correlations, priority support

B2B basic     €499/mo     — /enrich 1k req/mo
B2B pro       €1500/mo    — /enrich 10k req/mo
B2B enterprise €3000+     — custom invoicing, dedicated support

Trial: 14 days no card on LITE + PRO (sticky bucket per signup).

Stripe price IDs are read from env at runtime — defaults are fixture
values for dev. Production sets ``STRIPE_PRICE_LITE`` etc.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


TIER_FREE = "FREE"
TIER_LITE = "LITE"
TIER_PRO = "PRO"
TIER_PRO_PLUS = "PRO_PLUS"
TIER_B2B_BASIC = "B2B_BASIC"
TIER_B2B_PRO = "B2B_PRO"
TIER_B2B_ENTERPRISE = "B2B_ENTERPRISE"


@dataclass(frozen=True)
class PricingTier:
    key: str
    display_name: str
    monthly_price_eur: float
    stripe_price_id: Optional[str]
    chat_per_day: int        # rate-limit knob; -1 = unlimited
    enrich_per_month: int    # B2B knob; -1 = unlimited
    features: tuple[str, ...] = ()
    trial_days: int = 0
    is_b2b: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "monthly_price_eur": self.monthly_price_eur,
            "stripe_price_id": self.stripe_price_id,
            "chat_per_day": self.chat_per_day,
            "enrich_per_month": self.enrich_per_month,
            "features": list(self.features),
            "trial_days": self.trial_days,
            "is_b2b": self.is_b2b,
        }


PRICING_TIERS: dict[str, PricingTier] = {
    TIER_FREE: PricingTier(
        key=TIER_FREE,
        display_name="FREE",
        monthly_price_eur=0.0,
        stripe_price_id=None,
        chat_per_day=1,
        enrich_per_month=0,
        features=(
            "Analyses publiques",
            "Transparence en direct",
            "1 chat/jour",
        ),
    ),
    TIER_LITE: PricingTier(
        key=TIER_LITE,
        display_name="LITE",
        monthly_price_eur=19.0,
        stripe_price_id=os.environ.get("STRIPE_PRICE_LITE"),
        chat_per_day=10,
        enrich_per_month=0,
        features=(
            "10 chats/jour",
            "Notification Telegram",
            "Glossaire interactif",
        ),
        trial_days=14,
    ),
    TIER_PRO: PricingTier(
        key=TIER_PRO,
        display_name="PRO",
        monthly_price_eur=39.0,
        stripe_price_id=os.environ.get("STRIPE_PRICE_PRO"),
        chat_per_day=-1,
        enrich_per_month=100,
        features=(
            "Chat illimité",
            "Multi-asset (XAU, EURUSD, BTC)",
            "API read-only 100 req/mois",
        ),
        trial_days=14,
    ),
    TIER_PRO_PLUS: PricingTier(
        key=TIER_PRO_PLUS,
        display_name="PRO+",
        monthly_price_eur=99.0,
        stripe_price_id=os.environ.get("STRIPE_PRICE_PRO_PLUS"),
        chat_per_day=-1,
        enrich_per_month=1000,
        features=(
            "Tout PRO inclus",
            "Régime + corrélations",
            "API full 1000 req/mois",
            "Support prioritaire",
        ),
        trial_days=14,
    ),
    TIER_B2B_BASIC: PricingTier(
        key=TIER_B2B_BASIC,
        display_name="B2B Basic",
        monthly_price_eur=499.0,
        stripe_price_id=os.environ.get("STRIPE_PRICE_B2B_BASIC"),
        chat_per_day=-1,
        enrich_per_month=1000,
        features=(
            "/enrich 1 000 req/mois",
            "Audit trail B2B complet",
            "Multi-langue FR/EN/DE/ES",
        ),
        is_b2b=True,
    ),
    TIER_B2B_PRO: PricingTier(
        key=TIER_B2B_PRO,
        display_name="B2B Pro",
        monthly_price_eur=1500.0,
        stripe_price_id=os.environ.get("STRIPE_PRICE_B2B_PRO"),
        chat_per_day=-1,
        enrich_per_month=10_000,
        features=(
            "/enrich 10 000 req/mois",
            "SLA 99.9%, support dédié",
            "Webhook delivery acked",
        ),
        is_b2b=True,
    ),
    TIER_B2B_ENTERPRISE: PricingTier(
        key=TIER_B2B_ENTERPRISE,
        display_name="B2B Enterprise",
        monthly_price_eur=3000.0,
        stripe_price_id=None,  # custom invoicing
        chat_per_day=-1,
        enrich_per_month=-1,
        features=(
            "/enrich illimité",
            "Facturation personnalisée",
            "Contrat-cadre + SLA négocié",
        ),
        is_b2b=True,
    ),
}


def get_tier(key: str) -> Optional[PricingTier]:
    return PRICING_TIERS.get(key.upper())


def list_b2c_tiers() -> list[PricingTier]:
    return [t for t in PRICING_TIERS.values() if not t.is_b2b]


def list_b2b_tiers() -> list[PricingTier]:
    return [t for t in PRICING_TIERS.values() if t.is_b2b]


__all__ = [
    "PRICING_TIERS",
    "PricingTier",
    "TIER_B2B_BASIC", "TIER_B2B_ENTERPRISE", "TIER_B2B_PRO",
    "TIER_FREE", "TIER_LITE", "TIER_PRO", "TIER_PRO_PLUS",
    "get_tier", "list_b2b_tiers", "list_b2c_tiers",
]
