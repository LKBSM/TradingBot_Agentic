"""Tests for the billing module — mission PRIX-1 (single USD plan).

One paid plan, two cadences (MONTHLY / ANNUAL), US dollars, plus the kept FREE
tier. Amounts come from the single source ``config/pricing.json`` — the tests
assert the module and the JSON agree, so no amount can drift.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.billing import (
    PLAN_ANNUAL,
    PLAN_FREE,
    PLAN_MONTHLY,
    PricingPlan,
    StripeClient,
    currency,
    get_plan,
    list_paid_plans,
    list_plans,
    parse_webhook_event,
)

_CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config" / "pricing.json").read_text("utf-8")
)


@pytest.fixture(autouse=True)
def _testing_mode():
    with patch("src.api.auth.TESTING_MODE", True):
        yield


# ---------------------------------------------------------------------------
# Single source of truth — the module mirrors config/pricing.json exactly
# ---------------------------------------------------------------------------


def test_currency_is_usd_everywhere():
    assert currency() == "USD"
    assert all(p.currency == "USD" for p in list_plans())


def test_amounts_come_from_the_single_source():
    monthly = get_plan(PLAN_MONTHLY)
    annual = get_plan(PLAN_ANNUAL)
    assert monthly.amount_usd == float(_CONFIG["plans"]["monthly"]["amount"])
    assert annual.amount_usd == float(_CONFIG["plans"]["annual"]["amountPerYear"])


def test_target_prices_39_and_348():
    assert get_plan(PLAN_MONTHLY).amount_usd == 39.0
    assert get_plan(PLAN_ANNUAL).amount_usd == 348.0


def test_annual_monthly_equivalent_is_exact_29():
    annual = get_plan(PLAN_ANNUAL)
    # Derived, must be whole and equal to 348/12.
    assert annual.monthly_equivalent_usd == 29.0
    assert annual.amount_usd / 12.0 == annual.monthly_equivalent_usd


def test_free_tier_kept_and_zero():
    free = get_plan(PLAN_FREE)
    assert free is not None
    assert free.is_free is True
    assert free.amount_usd == 0.0


def test_only_two_paid_plans():
    paid = list_paid_plans()
    assert {p.key for p in paid} == {PLAN_MONTHLY, PLAN_ANNUAL}
    assert all(not p.is_free for p in paid)


def test_get_plan_case_insensitive():
    assert get_plan("monthly").key == PLAN_MONTHLY
    assert get_plan("ANNUAL").key == PLAN_ANNUAL
    assert get_plan("nope") is None


def test_to_dict_serialisable():
    d = get_plan(PLAN_MONTHLY).to_dict()
    json.dumps(d)
    assert d["key"] == PLAN_MONTHLY
    assert d["amount_usd"] == 39.0
    assert d["currency"] == "USD"


def test_no_tax_field_anywhere_in_the_model():
    blob = json.dumps([p.to_dict() for p in list_plans()]).lower()
    assert "tax" not in blob
    assert "tva" not in blob and "tps" not in blob and "tvq" not in blob


# ---------------------------------------------------------------------------
# StripeClient — unconfigured behaviour
# ---------------------------------------------------------------------------


def test_unconfigured_client_reports_so():
    c = StripeClient(api_key=None)
    assert c.is_configured is False


def test_unconfigured_client_raises_on_call():
    c = StripeClient(api_key=None)
    with pytest.raises(RuntimeError, match="not configured"):
        c.create_checkout_session(
            price_id="px",
            success_url="x", cancel_url="x",
            customer_email="a@b.c",
        )


def test_configured_client_has_credentials():
    c = StripeClient(api_key="sk_test_xxx", webhook_secret="whsec_xxx")
    assert c.is_configured is True


# ---------------------------------------------------------------------------
# parse_webhook_event — resolves the plan from the price id via env
# ---------------------------------------------------------------------------


def test_parse_ignores_unrelated_event():
    out = parse_webhook_event({"type": "charge.refunded", "data": {"object": {}}})
    assert out is None


def test_parse_subscription_updated_resolves_plan_from_env(monkeypatch):
    monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_123")
    payload = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_123",
                "customer": "cus_abc",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_monthly_123"}}]},
            }
        },
    }
    out = parse_webhook_event(payload)
    assert out is not None
    assert out.customer_id == "cus_abc"
    assert out.subscription_id == "sub_123"
    assert out.price_id == "price_monthly_123"
    assert out.plan_key == PLAN_MONTHLY
    assert out.status == "active"


def test_parse_subscription_deleted():
    payload = {
        "type": "customer.subscription.deleted",
        "data": {
            "object": {
                "id": "sub_xyz",
                "customer": "cus_zzz",
                "status": "canceled",
                "items": {"data": []},
            }
        },
    }
    out = parse_webhook_event(payload)
    assert out is not None
    assert out.event_type == "customer.subscription.deleted"


# ---------------------------------------------------------------------------
# Pricing endpoint (legacy /api/v1/billing surface)
# ---------------------------------------------------------------------------


def test_pricing_endpoint_returns_plans():
    c = TestClient(create_app())
    resp = c.get("/api/v1/billing/pricing")
    assert resp.status_code == 200
    body = resp.json()
    keys = {p["key"] for p in body["plans"]}
    assert {PLAN_FREE, PLAN_MONTHLY, PLAN_ANNUAL}.issubset(keys)
    # Every advertised amount is in USD.
    assert all(p["currency"] == "USD" for p in body["plans"])


def test_checkout_503_or_400_without_stripe():
    c = TestClient(create_app())  # no stripe_client wired, no price env
    resp = c.post(
        "/api/v1/billing/checkout",
        json={
            "plan_key": "MONTHLY",
            "email": "a@b.com",
            "success_url": "https://x.com/ok",
            "cancel_url": "https://x.com/cancel",
        },
    )
    # Plan exists but price_id is None (env unset) → 400, OR no stripe → 503.
    assert resp.status_code in (400, 503)


def test_checkout_400_for_unknown_plan():
    c = TestClient(create_app())
    resp = c.post(
        "/api/v1/billing/checkout",
        json={
            "plan_key": "MEGA_ULTRA",
            "email": "a@b.com",
            "success_url": "https://x.com/ok",
            "cancel_url": "https://x.com/cancel",
        },
    )
    assert resp.status_code == 400


def test_checkout_400_for_free_plan():
    c = TestClient(create_app())
    resp = c.post(
        "/api/v1/billing/checkout",
        json={
            "plan_key": "FREE",
            "email": "a@b.com",
            "success_url": "https://x.com/ok",
            "cancel_url": "https://x.com/cancel",
        },
    )
    assert resp.status_code == 400


def test_webhook_503_without_stripe():
    c = TestClient(create_app())
    resp = c.post(
        "/api/v1/billing/webhook",
        content=b"{}",
        headers={"Stripe-Signature": "x"},
    )
    assert resp.status_code == 503
