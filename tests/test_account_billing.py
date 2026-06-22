"""Tests for account-bound Stripe subscriptions (payments mission ②).

Exercises the FULL flow against a FAKE Stripe client (no network, no SDK, no
real keys) so the suite runs in any environment:

* checkout creates/reuses a customer and returns the hosted URL,
* a verified ``customer.subscription.*`` webhook flips the account to active,
* cancellation / payment failure / expiry are reflected on access,
* an invalid webhook signature is rejected (400) with NO state change,
* webhooks are idempotent — the same event id is applied at most once.

No card data is ever involved — Checkout/Portal are hosted by Stripe.
"""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app
from src.api.subscription_gate import account_has_access


# =============================================================================
# Fake Stripe client (mirrors src.billing.stripe_client.StripeClient surface)
# =============================================================================

class FakeStripeClient:
    """In-memory stand-in. ``verify_webhook`` trusts the body unless the
    signature is the sentinel ``"bad"`` (→ ValueError), which lets tests drive
    raw event payloads while still proving the bad-signature path."""

    def __init__(self):
        self.is_configured = True
        self._customer_seq = 0

    def create_customer(self, *, email, account_id):
        self._customer_seq += 1
        return {"id": f"cus_test_{account_id}_{self._customer_seq}", "email": email}

    def create_checkout_session(self, **kwargs):
        self.last_checkout_kwargs = kwargs
        return {"id": "cs_test_1", "url": "https://checkout.stripe.test/cs_test_1"}

    def create_billing_portal_session(self, *, customer_id, return_url):
        return {"id": "bps_1", "url": f"https://portal.stripe.test/{customer_id}"}

    def verify_webhook(self, *, body, signature):
        if signature == "bad":
            raise ValueError("webhook verification failed: bad signature")
        return json.loads(body)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "billing_accounts.db"))


@pytest.fixture()
def stripe():
    return FakeStripeClient()


@pytest.fixture()
def client(account_store, stripe, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    # A purchasable plan so /checkout has a valid price to resolve.
    monkeypatch.setenv("STRIPE_PRICE_STANDARD", "price_standard_test")
    app = create_app(account_store=account_store, stripe_client=stripe)
    return TestClient(app)


def _register(client, username="alice", email="alice@example.com"):
    resp = client.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": email,
            "password": "longpassword1",
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _sub_event(event_id, event_type, account_id, *, status="active",
               customer="cus_test_1", price="price_standard_test",
               current_period_end=None, cancel_at_period_end=False):
    return {
        "id": event_id,
        "type": event_type,
        "data": {"object": {
            "id": "sub_test_1",
            "customer": customer,
            "status": status,
            "current_period_end": current_period_end,
            "cancel_at_period_end": cancel_at_period_end,
            "trial_end": None,
            "metadata": {"account_id": str(account_id)},
            "items": {"data": [{"price": {"id": price}}]},
        }},
    }


def _post_webhook(client, payload, signature="good"):
    return client.post(
        "/api/billing/webhook",
        content=json.dumps(payload),
        headers={"Stripe-Signature": signature, "Content-Type": "application/json"},
    )


# =============================================================================
# Pricing + checkout + portal
# =============================================================================

class TestCheckoutFlow:
    def test_pricing_lists_configured_plan(self, client):
        resp = client.get("/api/billing/pricing")
        assert resp.status_code == 200
        keys = {p["key"] for p in resp.json()["plans"]}
        assert "STANDARD" in keys

    def test_checkout_requires_auth(self, client):
        resp = client.post("/api/billing/checkout", json={"plan_key": "STANDARD"})
        assert resp.status_code == 401

    def test_checkout_creates_customer_and_returns_url(self, client, account_store):
        acct = _register(client)
        resp = client.post("/api/billing/checkout", json={"plan_key": "STANDARD"})
        assert resp.status_code == 200, resp.text
        assert resp.json()["url"].startswith("https://checkout.stripe.test/")
        # The account is now linked to a Stripe customer.
        sub = account_store.get_subscription(acct["id"])
        assert sub is not None and sub["stripe_customer_id"]

    def test_checkout_rejects_unconfigured_price(self, client):
        _register(client)
        resp = client.post("/api/billing/checkout", json={"price_id": "price_evil"})
        assert resp.status_code == 400

    def test_portal_requires_existing_customer(self, client):
        _register(client)
        # No checkout yet → no customer linked → 409.
        resp = client.post("/api/billing/portal")
        assert resp.status_code == 409

    def test_portal_after_checkout(self, client):
        _register(client)
        client.post("/api/billing/checkout", json={"plan_key": "STANDARD"})
        resp = client.post("/api/billing/portal")
        assert resp.status_code == 200
        assert resp.json()["url"].startswith("https://portal.stripe.test/")


# =============================================================================
# Webhook → subscription state
# =============================================================================

class TestWebhookStateReflection:
    def test_subscription_created_makes_account_active(self, client, account_store):
        acct = _register(client)
        far_future = time.time() + 30 * 86400
        resp = _post_webhook(
            client,
            _sub_event("evt_1", "customer.subscription.created", acct["id"],
                       status="active", current_period_end=far_future),
        )
        assert resp.status_code == 200 and resp.json()["applied"] is True
        sub = account_store.get_subscription(acct["id"])
        assert sub["status"] == "active"
        # GET /subscription reflects it for the logged-in user.
        view = client.get("/api/billing/subscription").json()
        assert view["status"] == "active"

    def test_cancellation_reflected(self, client, account_store):
        acct = _register(client)
        _post_webhook(client, _sub_event("evt_a", "customer.subscription.created",
                                         acct["id"], status="active",
                                         current_period_end=time.time() + 86400))
        _post_webhook(client, _sub_event("evt_b", "customer.subscription.deleted",
                                         acct["id"]))
        assert account_store.get_subscription(acct["id"])["status"] == "canceled"

    def test_payment_failed_sets_past_due(self, client, account_store):
        acct = _register(client)
        # Link first so the invoice (no metadata) resolves by customer id.
        _post_webhook(client, _sub_event("evt_c", "customer.subscription.created",
                                         acct["id"], status="active",
                                         customer="cus_pay_1",
                                         current_period_end=time.time() + 86400))
        payload = {
            "id": "evt_fail",
            "type": "invoice.payment_failed",
            "data": {"object": {"customer": "cus_pay_1", "subscription": "sub_test_1"}},
        }
        _post_webhook(client, payload)
        assert account_store.get_subscription(acct["id"])["status"] == "past_due"

    def test_enforced_gate_grants_only_active(self, account_store, stripe, monkeypatch):
        monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")
        acct = account_store.create_account(
            "carol", "carol@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "1"), ("privacy", "1")],
        )
        # No subscription → denied.
        assert account_has_access(acct, account_store) is False
        # Active + future period → allowed.
        account_store.upsert_subscription(
            acct["id"], status="active",
            current_period_end=time.time() + 86400,
        )
        assert account_has_access(acct, account_store) is True
        # Expired period → denied even though status is active (safety net).
        account_store.upsert_subscription(
            acct["id"], current_period_end=time.time() - 10,
        )
        assert account_has_access(acct, account_store) is False


# =============================================================================
# Signature verification + idempotency
# =============================================================================

class TestWebhookSecurity:
    def test_invalid_signature_rejected(self, client, account_store):
        acct = _register(client)
        resp = _post_webhook(
            client,
            _sub_event("evt_x", "customer.subscription.created", acct["id"]),
            signature="bad",
        )
        assert resp.status_code == 400
        # No state was written.
        sub = account_store.get_subscription(acct["id"])
        assert sub is None or sub["status"] is None

    def test_missing_signature_rejected(self, client):
        acct = _register(client)
        resp = client.post(
            "/api/billing/webhook",
            content=json.dumps(_sub_event("evt_y", "customer.subscription.created", acct["id"])),
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 400

    def test_duplicate_event_applied_once(self, client, account_store):
        acct = _register(client)
        evt = _sub_event("evt_dup", "customer.subscription.created", acct["id"],
                         status="active", current_period_end=time.time() + 86400)
        first = _post_webhook(client, evt)
        assert first.json().get("applied") is True
        # Re-deliver the SAME event id → recognised as duplicate, not re-applied.
        second = _post_webhook(client, evt)
        assert second.json().get("duplicate") is True
        # And a subsequent state-changing event with that id is ignored even if
        # its body differs (idempotency keys on event id).
        tampered = _sub_event("evt_dup", "customer.subscription.deleted", acct["id"])
        third = _post_webhook(client, tampered)
        assert third.json().get("duplicate") is True
        assert account_store.get_subscription(acct["id"])["status"] == "active"

    def test_unresolved_event_not_claimed(self, client, account_store):
        # An event for an unknown customer with no account metadata is neither
        # applied nor claimed (so Stripe's retry can land once linked).
        payload = {
            "id": "evt_orphan",
            "type": "invoice.payment_failed",
            "data": {"object": {"customer": "cus_unknown", "subscription": "sub_z"}},
        }
        resp = _post_webhook(client, payload)
        assert resp.status_code == 200 and resp.json().get("unresolved") is True
        # Not claimed → a later delivery is still treated as new.
        assert account_store.mark_webhook_processed("evt_orphan") is True
