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


def test_live_prices_are_39_99_and_359_88():
    """The amounts the LIVE Stripe prices actually charge (go-live 2026-09-27).

    price_1UKOUBFiM5Kf1kQcGwJyVv0W = 3999 cents/month,
    price_1UKOVNFiM5Kf1kQcizm0JiYB = 35988 cents/year. If this fails, the site
    quotes a price Stripe does not charge — which is the one billing bug a
    customer always notices.
    """
    assert get_plan(PLAN_MONTHLY).amount_usd == 39.99
    assert get_plan(PLAN_ANNUAL).amount_usd == 359.88


def test_the_legacy_amounts_are_gone():
    # 39 / 348 were the pre-go-live amounts. Left behind anywhere, they are a
    # price the customer is shown but not charged.
    assert get_plan(PLAN_MONTHLY).amount_usd != 39.0
    assert get_plan(PLAN_ANNUAL).amount_usd != 348.0


def test_annual_monthly_equivalent_is_exact_29_99():
    annual = get_plan(PLAN_ANNUAL)
    # Derived and EXACT: 35988 cents / 12 = 2999 cents. The page says "soit
    # 29,99 $ par mois", so it has to be the true quotient, not a rounded one.
    assert annual.monthly_equivalent_usd == 29.99
    assert round(annual.amount_usd * 100) % 12 == 0
    assert round(annual.amount_usd * 100) // 12 == round(annual.monthly_equivalent_usd * 100)


def test_amounts_are_formatted_with_two_decimals():
    from src.billing.pricing import format_amount

    # ":g" used to print "39.99" in French prose — and would print "40" the day
    # an amount lands on a round number. Two decimals, comma, always.
    assert format_amount(39.99) == "39,99"
    assert format_amount(359.88) == "359,88"
    assert format_amount(29.99) == "29,99"
    assert format_amount(120.0) == "120,00"
    assert format_amount(39.99, decimal_separator=".") == "39.99"


def test_no_free_plan_in_catalog():
    # PAY-2: paying is the condition of entry — the free plan is gone. The legacy
    # alias PLAN_FREE still exists as a name, but it is not in the catalog.
    assert get_plan(PLAN_FREE) is None
    assert all(not p.is_free for p in list_plans())


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
    assert d["amount_usd"] == 39.99
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
    # PAY-2: paid-only catalog — no free plan is ever advertised.
    assert keys == {PLAN_MONTHLY, PLAN_ANNUAL}
    assert PLAN_FREE not in keys
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


class _WebhookStripe:
    """Minimal stripe stand-in: configured, and trusts the body's signature.

    Needed because the legacy endpoint refuses in 410 only AFTER verifying the
    signature — a deliberate choice on main, so Stripe retries the event rather
    than dropping it, and the event can be replayed once the URL is repointed.
    """

    is_configured = True

    def verify_webhook(self, *, body, signature):
        return json.loads(body or b"{}")


def test_legacy_webhook_is_retired_and_answers_410():
    """The legacy webhook must REFUSE, not absorb.

    It fed ``tier_manager`` — state the account paywall never reads. While it
    answered 200, Stripe considered delivery successful, never retried, and showed
    no failure: a real payment was taken and opened NOTHING, with a log line as
    the only trace. 410 makes it visible where the founder already looks.
    """
    c = TestClient(create_app(stripe_client=_WebhookStripe()))
    resp = c.post(
        "/api/v1/billing/webhook",
        content=json.dumps({
            "id": "evt_legacy",
            "type": "customer.subscription.updated",
            "data": {"object": {
                "customer": "cus_x", "status": "active",
                "items": {"data": [{"price": {"id": "price_x"}}]},
            }},
        }),
        headers={"Stripe-Signature": "whatever"},
    )
    assert resp.status_code == 410
    # The refusal must say where to point the endpoint instead — a 410 with no
    # destination leaves the founder exactly as stuck as the silent 200 did.
    assert "/api/billing/webhook" in resp.json()["detail"]


def test_legacy_webhook_still_rejects_a_missing_signature_first():
    # An unsigned request is not "retired", it is unauthenticated: 400 before any
    # parsing, so the 410 can never be used to probe the endpoint's behaviour.
    c = TestClient(create_app(stripe_client=_WebhookStripe()))
    assert c.post("/api/v1/billing/webhook", content=b"{}").status_code == 400


def test_the_live_webhook_path_exists_and_is_the_only_receiver():
    """Guards the confusion itself: the ONE path the account paywall reads."""
    routes = {getattr(r, "path", None) for r in create_app().routes}
    assert "/api/billing/webhook" in routes
    assert "/api/v1/billing/webhook" in routes  # still routed — to a 410
