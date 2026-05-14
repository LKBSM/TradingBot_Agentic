"""Tests for the INFRA-2B.3 billing module."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.billing import (
    PRICING_TIERS,
    PricingTier,
    StripeClient,
    TIER_FREE,
    TIER_LITE,
    TIER_PRO,
    TIER_PRO_PLUS,
    get_tier,
    parse_webhook_event,
)


@pytest.fixture(autouse=True)
def _testing_mode():
    with patch("src.api.auth.TESTING_MODE", True):
        yield


# ---------------------------------------------------------------------------
# Pricing grid
# ---------------------------------------------------------------------------


def test_all_four_b2c_tiers_present():
    keys = {TIER_FREE, TIER_LITE, TIER_PRO, TIER_PRO_PLUS}
    assert keys.issubset(PRICING_TIERS.keys())


def test_free_is_zero_eur():
    assert PRICING_TIERS[TIER_FREE].monthly_price_eur == 0.0


def test_pricing_grid_monotonic():
    prices = [
        PRICING_TIERS[k].monthly_price_eur
        for k in (TIER_FREE, TIER_LITE, TIER_PRO, TIER_PRO_PLUS)
    ]
    assert prices == sorted(prices)


def test_trial_days_only_on_paid_b2c_tiers():
    assert PRICING_TIERS[TIER_FREE].trial_days == 0
    assert PRICING_TIERS[TIER_LITE].trial_days == 14
    assert PRICING_TIERS[TIER_PRO].trial_days == 14


def test_b2b_tiers_flagged():
    b2b = [t for t in PRICING_TIERS.values() if t.is_b2b]
    assert len(b2b) >= 2  # at least basic + pro
    assert all(t.is_b2b for t in b2b)


def test_get_tier_case_insensitive():
    assert get_tier("free").key == TIER_FREE
    assert get_tier("PRO_PLUS").key == TIER_PRO_PLUS
    assert get_tier("nope") is None


def test_to_dict_serialisable():
    import json

    d = PRICING_TIERS[TIER_PRO].to_dict()
    json.dumps(d)
    assert d["key"] == TIER_PRO
    assert d["monthly_price_eur"] == 39.0


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
# parse_webhook_event
# ---------------------------------------------------------------------------


def test_parse_ignores_unrelated_event():
    out = parse_webhook_event({"type": "charge.refunded", "data": {"object": {}}})
    assert out is None


def test_parse_subscription_updated_resolves_tier_from_env(monkeypatch):
    # Wire a known price ID → tier mapping via env
    monkeypatch.setenv("STRIPE_PRICE_PRO", "price_pro_123")
    # Force re-import to pick up the env
    import importlib

    import src.billing.pricing as pricing_mod

    importlib.reload(pricing_mod)
    import src.billing.stripe_client as sc_mod

    importlib.reload(sc_mod)

    payload = {
        "type": "customer.subscription.updated",
        "data": {
            "object": {
                "id": "sub_123",
                "customer": "cus_abc",
                "status": "active",
                "items": {"data": [{"price": {"id": "price_pro_123"}}]},
            }
        },
    }
    out = sc_mod.parse_webhook_event(payload)
    assert out is not None
    assert out.event_type == "customer.subscription.updated"
    assert out.customer_id == "cus_abc"
    assert out.subscription_id == "sub_123"
    assert out.price_id == "price_pro_123"
    assert out.tier_key == "PRO"
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
# Pricing endpoint
# ---------------------------------------------------------------------------


def test_pricing_endpoint_returns_table():
    c = TestClient(create_app())
    resp = c.get("/api/v1/billing/pricing")
    assert resp.status_code == 200
    body = resp.json()
    assert "b2c" in body and "b2b" in body
    b2c_keys = {t["key"] for t in body["b2c"]}
    assert {TIER_FREE, TIER_LITE, TIER_PRO, TIER_PRO_PLUS}.issubset(b2c_keys)


def test_checkout_503_without_stripe():
    c = TestClient(create_app())  # no stripe_client wired
    resp = c.post(
        "/api/v1/billing/checkout",
        json={
            "tier_key": "PRO",
            "email": "a@b.com",
            "success_url": "https://x.com/ok",
            "cancel_url": "https://x.com/cancel",
        },
    )
    # Tier exists but price_id is None (env unset) → 400
    # OR stripe_client is None → 503
    assert resp.status_code in (400, 503)


def test_checkout_400_for_unknown_tier():
    c = TestClient(create_app())
    resp = c.post(
        "/api/v1/billing/checkout",
        json={
            "tier_key": "MEGA_ULTRA",
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
