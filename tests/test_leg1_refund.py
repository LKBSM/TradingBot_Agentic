"""LEG-1 — the 14-day annual guarantee, honoured automatically.

The terms promise: « À l'annuel, nous offrons une garantie de 14 jours à compter
du paiement. » These tests hold the promise to its wording:

  * annual only, and a refusal always says what the customer CAN do instead;
  * the clock starts at the PAYMENT (Stripe's date), not at sign-up;
  * day 14 is still inside, day 15 is not;
  * refunding revokes access immediately — we do not wait for the webhook;
  * if the refund succeeds but the cancellation fails, the customer is still
    told it worked: the money is back, which is what they asked for.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.billing import refund_guarantee
from src.billing.refund_guarantee import GUARANTEE_SECONDS, evaluate

DAY = 24 * 60 * 60
NOW = 1_780_000_000.0

ANNUAL_PRICE = "price_annual_test"
MONTHLY_PRICE = "price_monthly_test"


# =============================================================================
# The rule, in isolation
# =============================================================================

class TestGuaranteeRule:
    def test_annual_paid_today_is_covered(self):
        d = evaluate(plan_key="ANNUAL", paid_at=NOW, now=NOW)
        assert d.eligible
        assert d.days_remaining == 14

    def test_day_14_is_still_inside(self):
        # One second before the window closes.
        d = evaluate(plan_key="ANNUAL", paid_at=NOW - GUARANTEE_SECONDS + 1, now=NOW)
        assert d.eligible
        assert d.days_remaining == 1

    def test_the_instant_the_window_closes_is_outside(self):
        d = evaluate(plan_key="ANNUAL", paid_at=NOW - GUARANTEE_SECONDS, now=NOW)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_WINDOW_ELAPSED
        assert d.days_remaining == 0

    def test_day_15_is_outside(self):
        d = evaluate(plan_key="ANNUAL", paid_at=NOW - 15 * DAY, now=NOW)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_WINDOW_ELAPSED

    def test_monthly_is_not_covered(self):
        d = evaluate(plan_key="MONTHLY", paid_at=NOW, now=NOW)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_NOT_ANNUAL

    def test_no_plan_means_no_subscription(self):
        d = evaluate(plan_key=None, paid_at=NOW, now=NOW)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_NO_SUBSCRIPTION

    def test_no_payment_date_is_refused_not_granted(self):
        d = evaluate(plan_key="ANNUAL", paid_at=None, now=NOW)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_NO_PAYMENT

    def test_already_refunded_short_circuits_everything(self):
        d = evaluate(plan_key="ANNUAL", paid_at=NOW, now=NOW, already_refunded=True)
        assert not d.eligible
        assert d.reason == refund_guarantee.REASON_ALREADY_REFUNDED

    def test_clock_skew_never_costs_the_customer_their_guarantee(self):
        # A payment dated in the future must not read as "elapsed".
        d = evaluate(plan_key="ANNUAL", paid_at=NOW + 3 * DAY, now=NOW)
        assert d.eligible

    def test_a_renewal_payment_opens_a_fresh_window(self):
        # The window is counted from the PAYMENT, so last year's charge being
        # old is irrelevant once a renewal has been paid.
        d = evaluate(plan_key="ANNUAL", paid_at=NOW - 2 * DAY, now=NOW)
        assert d.eligible

    def test_days_remaining_rounds_up(self):
        # 1 second left is still "a day left", never 0 — never tell someone the
        # window is closed while it is open.
        d = evaluate(plan_key="ANNUAL", paid_at=NOW - GUARANTEE_SECONDS + 1, now=NOW)
        assert d.days_remaining == 1


# =============================================================================
# End to end, through the API
# =============================================================================

class FakeStripe:
    """Minimal stand-in for StripeClient — records what it was asked to do."""

    def __init__(self, *, paid_at: Optional[float], fail_cancel: bool = False):
        self.is_configured = True
        self._paid_at = paid_at
        self._fail_cancel = fail_cancel
        self.refunds: list[Dict[str, Any]] = []
        self.cancelled: list[str] = []

    def get_latest_paid_invoice(self, subscription_id: str) -> Optional[dict]:
        if self._paid_at is None:
            return None
        return {
            "invoice_id": "in_test",
            "charge_id": "ch_test",
            "payment_intent_id": "pi_test",
            "paid_at": self._paid_at,
            "amount_paid": 34800,
            "currency": "usd",
        }

    def refund_payment(self, *, charge_id=None, payment_intent_id=None, reason=None):
        self.refunds.append({"charge_id": charge_id, "payment_intent_id": payment_intent_id})
        return {"id": "re_test", "status": "succeeded"}

    def cancel_subscription(self, subscription_id: str) -> dict:
        if self._fail_cancel:
            raise RuntimeError("stripe down")
        self.cancelled.append(subscription_id)
        return {"id": subscription_id, "status": "canceled"}


@pytest.fixture()
def store(tmp_path) -> AccountStore:
    return AccountStore(db_path=str(tmp_path / "accounts.db"))


def _build(store: AccountStore, monkeypatch, stripe: FakeStripe) -> TestClient:
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "leg1-refund-test-secret-value")
    monkeypatch.setenv("STRIPE_PRICE_MONTHLY", MONTHLY_PRICE)
    monkeypatch.setenv("STRIPE_PRICE_ANNUAL", ANNUAL_PRICE)
    # Production posture: without this the gate is a pass-through and
    # `has_access` stays True whatever the subscription says — the revocation
    # assertion would pass for the wrong reason.
    monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")
    # `pricing._plans()` re-reads the env on every access, so setting the price
    # ids here is enough — no cache to clear.
    from src.api.app import create_app

    app = create_app(account_store=store)
    app.state.app_state.stripe_client = stripe
    return TestClient(app)


def _subscribed(
    store: AccountStore,
    client: TestClient,
    *,
    price_id: str,
    status: str = "active",
) -> int:
    r = client.post(
        "/api/auth/register",
        json={
            "username": "buyer",
            "email": "buyer@example.com",
            "password": "correct horse battery",
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert r.status_code in (200, 201), r.text
    account_id = r.json()["id"]
    store.upsert_subscription(
        account_id,
        stripe_customer_id="cus_test",
        stripe_subscription_id="sub_test",
        status=status,
        price_id=price_id,
        current_period_end=NOW + 365 * DAY,
        cancel_at_period_end=False,
        trial_end=None,
    )
    return account_id


class TestRefundEndpoint:
    def test_requires_authentication(self, store, monkeypatch):
        client = _build(store, monkeypatch, FakeStripe(paid_at=time.time()))
        assert client.post("/api/billing/refund").status_code == 401
        assert client.get("/api/billing/refund-eligibility").status_code == 401

    def test_annual_inside_the_window_is_refunded_and_access_revoked(
        self, store, monkeypatch
    ):
        stripe = FakeStripe(paid_at=time.time() - 3 * DAY)
        client = _build(store, monkeypatch, stripe)
        account_id = _subscribed(store, client, price_id=ANNUAL_PRICE)
        assert client.get("/api/billing/subscription").json()["has_access"] is True

        r = client.post("/api/billing/refund")
        assert r.status_code == 200, r.text
        assert r.json()["refunded"] is True
        assert r.json()["currency"] == "USD"

        # The payment was actually refunded, in full, and the subscription cancelled.
        assert stripe.refunds == [{"charge_id": "ch_test", "payment_intent_id": "pi_test"}]
        assert stripe.cancelled == ["sub_test"]

        # Access is revoked NOW — not when the webhook happens to arrive.
        assert (store.get_subscription(account_id) or {})["status"] == "suspended"
        assert client.get("/api/billing/subscription").json()["has_access"] is False

    def test_eligibility_probe_reports_the_open_window(self, store, monkeypatch):
        stripe = FakeStripe(paid_at=time.time() - 3 * DAY)
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=ANNUAL_PRICE)

        body = client.get("/api/billing/refund-eligibility").json()
        assert body["eligible"] is True
        assert body["days_remaining"] == 11
        assert body["guarantee_days"] == 14

    def test_past_the_window_refuses_and_points_to_cancellation(self, store, monkeypatch):
        stripe = FakeStripe(paid_at=time.time() - 20 * DAY)
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=ANNUAL_PRICE)

        r = client.post("/api/billing/refund")
        assert r.status_code == 409
        assert "résilier" in r.json()["detail"]
        assert stripe.refunds == []

    def test_monthly_is_refused_without_calling_stripe(self, store, monkeypatch):
        stripe = FakeStripe(paid_at=time.time())
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=MONTHLY_PRICE)

        r = client.post("/api/billing/refund")
        assert r.status_code == 409
        detail = r.json()["detail"]
        assert "annuel" in detail
        assert "résilier" in detail  # never a dead end
        assert stripe.refunds == []

    def test_no_subscription_is_refused(self, store, monkeypatch):
        stripe = FakeStripe(paid_at=time.time())
        client = _build(store, monkeypatch, stripe)
        client.post(
            "/api/auth/register",
            json={
                "username": "buyer", "email": "buyer@example.com",
                "password": "correct horse battery", "age_confirmed": True,
                "accept_terms": True, "accept_privacy": True,
            },
        )
        assert client.post("/api/billing/refund").status_code == 409

    def test_a_second_refund_is_refused(self, store, monkeypatch):
        stripe = FakeStripe(paid_at=time.time() - DAY)
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=ANNUAL_PRICE)

        assert client.post("/api/billing/refund").status_code == 200
        second = client.post("/api/billing/refund")
        assert second.status_code == 409
        assert len(stripe.refunds) == 1  # the money left exactly once

    def test_refund_succeeds_even_if_cancelling_fails(self, store, monkeypatch):
        # The money is back — that is what the customer asked for. A stray
        # subscription is the operator's problem, not a failed refund.
        stripe = FakeStripe(paid_at=time.time() - DAY, fail_cancel=True)
        client = _build(store, monkeypatch, stripe)
        account_id = _subscribed(store, client, price_id=ANNUAL_PRICE)

        r = client.post("/api/billing/refund")
        assert r.status_code == 200
        assert r.json()["refunded"] is True
        assert (store.get_subscription(account_id) or {})["status"] == "suspended"

    def test_a_stripe_failure_does_not_silently_swallow_the_request(
        self, store, monkeypatch
    ):
        stripe = FakeStripe(paid_at=time.time() - DAY)

        def boom(subscription_id):
            raise RuntimeError("stripe down")

        stripe.get_latest_paid_invoice = boom  # type: ignore[assignment]
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=ANNUAL_PRICE)

        assert client.post("/api/billing/refund").status_code == 502
        # …but the read-only probe degrades to "not eligible" instead of erroring.
        assert client.get("/api/billing/refund-eligibility").json()["eligible"] is False

    def test_billing_not_configured_still_renders_the_screen(self, store, monkeypatch):
        # No Stripe key at all: the POST says so (503), but the read-only probe
        # must still answer, or the whole subscription screen breaks.
        stripe = FakeStripe(paid_at=time.time() - DAY)
        stripe.is_configured = False
        client = _build(store, monkeypatch, stripe)
        _subscribed(store, client, price_id=ANNUAL_PRICE)

        assert client.post("/api/billing/refund").status_code == 503
        probe = client.get("/api/billing/refund-eligibility")
        assert probe.status_code == 200
        assert probe.json()["eligible"] is False
