"""Pricing — mission PRIX-1, paid-only since PAY-2.

ONE paid plan, two billing cadences, US dollars everywhere (including Canadian
customers). PAY-2 removed the free tier entirely: paying is the condition of
entry, so the catalog holds ONLY the two purchasable cadences. The public
landing demos are the only free surface, and they are not a "plan".

    MONTHLY    $39 / month    the full tool, cancel anytime
    ANNUAL     $348 / year    the full tool, i.e. $29 / month billed yearly

The amounts live in EXACTLY ONE place — ``config/pricing.json`` — which the
frontend also consumes (via the generated ``webapp/lib/pricing.generated.ts``).
Nothing here is hard-coded; we read the JSON at import. Stripe price IDs come
from env at runtime (``STRIPE_PRICE_MONTHLY`` / ``STRIPE_PRICE_ANNUAL``); they
are NEVER committed. No tax is ever added. No discount, no struck-through price.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional


# Plan keys — used as the Stripe checkout ``plan_key`` and in webhook routing.
# PAY-2: there is no free plan. ``PLAN_FREE`` is kept ONLY as a legacy alias so
# older imports/tests don't break; it is NOT part of the catalog and can never be
# purchased or granted.
PLAN_MONTHLY = "MONTHLY"
PLAN_ANNUAL = "ANNUAL"
PLAN_FREE = "FREE"  # legacy alias — not in the catalog (PAY-2)

# Repo root: src/billing/pricing.py → parents[2].
_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "pricing.json"


@lru_cache(maxsize=1)
def _config() -> dict:
    """Load the single-source pricing config (cached for the process)."""
    with _CONFIG_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@dataclass(frozen=True)
class PricingPlan:
    key: str
    display_name: str
    cadence: str                  # "free" | "monthly" | "annual"
    amount_usd: float             # amount billed for the cadence (0 / 39 / 348)
    monthly_equivalent_usd: float # per-month equivalent (0 / 39 / 29)
    currency: str                 # ISO 4217 — always "USD"
    stripe_price_id: Optional[str]
    is_free: bool = False

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "cadence": self.cadence,
            "amount_usd": self.amount_usd,
            "monthly_equivalent_usd": self.monthly_equivalent_usd,
            "currency": self.currency,
            "stripe_price_id": self.stripe_price_id,
            "is_free": self.is_free,
        }


def _build_plans() -> "dict[str, PricingPlan]":
    cfg = _config()
    currency_code = cfg["currency"]
    monthly_amount = float(cfg["plans"]["monthly"]["amount"])
    annual_year = float(cfg["plans"]["annual"]["amountPerYear"])
    # Derived — never authored. The config keeps annual divisible by 12 so this
    # is exact (guarded in the generator too).
    annual_month = annual_year / 12.0

    return {
        PLAN_MONTHLY: PricingPlan(
            key=PLAN_MONTHLY,
            display_name="Mensuel",
            cadence="monthly",
            amount_usd=monthly_amount,
            monthly_equivalent_usd=monthly_amount,
            currency=currency_code,
            stripe_price_id=os.environ.get(cfg["plans"]["monthly"]["stripeEnvVar"]),
        ),
        PLAN_ANNUAL: PricingPlan(
            key=PLAN_ANNUAL,
            display_name="Annuel",
            cadence="annual",
            amount_usd=annual_year,
            monthly_equivalent_usd=annual_month,
            currency=currency_code,
            stripe_price_id=os.environ.get(cfg["plans"]["annual"]["stripeEnvVar"]),
        ),
    }


# Rebuilt per access so a test/monkeypatch of the Stripe env is reflected without
# reimporting the module. The amounts come from the cached config; only the env
# lookups vary.
def _plans() -> "dict[str, PricingPlan]":
    return _build_plans()


def get_plan(key: str) -> Optional[PricingPlan]:
    return _plans().get(key.upper())


def list_plans() -> "list[PricingPlan]":
    """All plans (PAY-2: paid-only — MONTHLY, ANNUAL)."""
    return list(_plans().values())


def list_paid_plans() -> "list[PricingPlan]":
    """The purchasable cadences (MONTHLY, ANNUAL). Since PAY-2 removed the free
    plan this equals :func:`list_plans`; the ``is_free`` filter is kept so a
    re-introduced non-purchasable plan would still be excluded."""
    return [p for p in _plans().values() if not p.is_free]


def currency() -> str:
    return _config()["currency"]


__all__ = [
    "PLAN_ANNUAL",
    "PLAN_FREE",
    "PLAN_MONTHLY",
    "PricingPlan",
    "currency",
    "get_plan",
    "list_paid_plans",
    "list_plans",
]
