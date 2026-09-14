"""PAY-3 — selling zone, webhook health, and the legal lock on the renewal notice.

Three guards the payment journey did not have:

* **G2 — the selling zone.** The subscription is sold in Canada and the United
  States only. Stripe Checkout has no billing-country allow-list, so the defence
  is a Radar rule (dashboard) PLUS the webhook net tested here: a session that
  completes from outside the zone is cancelled, refunded, and persisted with a
  status that grants nothing. The test proves the refusal actually denies data —
  not merely that a flag was written.

* **G4 — a dead webhook cannot stay silent.** ``/api/billing/sync`` rescues a
  paying customer when no webhook arrived. That rescue must be counted and
  alerted, or the breakage hides behind its own workaround forever.

* **G5 — the renewal notice is locked.** Under the Quebec Consumer Protection
  Act the wording matters; the template does not go out until counsel approves
  it. Having SMTP configured is not approval.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app
from src.billing import renewal_notices, webhook_health
from src.billing.geo import (
    BLOCKED_REGION_STATUS,
    allowed_countries,
    billing_country_from_session,
    is_allowed_country,
)


DATA_ROUTE = "/api/market-status?instrument=XAUUSD&timeframe=M15"


# =============================================================================
# Fakes
# =============================================================================

class FakeStripeClient:
    """Mirrors the StripeClient surface the billing routes use, and RECORDS the
    refusal actions so the test can prove the money was actually given back."""

    def __init__(self):
        self.is_configured = True
        self.cancelled: List[str] = []
        self.refunded: List[str] = []
        self.remote_subscription: Optional[dict] = None
        self._seq = 0

    def create_customer(self, *, email, account_id):
        self._seq += 1
        return {"id": f"cus_{account_id}_{self._seq}", "email": email}

    def create_checkout_session(self, **kwargs):
        return {"id": "cs_1", "url": "https://checkout.stripe.test/cs_1"}

    def create_billing_portal_session(self, *, customer_id, return_url):
        return {"id": "bps_1", "url": f"https://portal.stripe.test/{customer_id}"}

    def verify_webhook(self, *, body, signature):
        if signature == "bad":
            raise ValueError("bad signature")
        return json.loads(body)

    def cancel_subscription(self, subscription_id):
        self.cancelled.append(subscription_id)
        return {"id": subscription_id, "status": "canceled"}

    def refund_subscription_payment(self, subscription_id):
        self.refunded.append(subscription_id)
        return True

    def find_customer_by_email(self, email):
        return None

    def get_subscription_state_for_customer(self, customer_id):
        return self.remote_subscription


class _StatusObj:
    def __init__(self, instrument, timeframe):
        self._d = {"instrument": instrument, "timeframe": timeframe, "state": "open"}

    def to_dict(self):
        return self._d


class FakeAssembler:
    def market_status(self, instrument, timeframe):
        return _StatusObj(instrument, timeframe)


# =============================================================================
# Fixtures — the gate is ENFORCED, as in production.
# =============================================================================

@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "pay3_zone_accounts.db"))


@pytest.fixture()
def stripe_client():
    return FakeStripeClient()


@pytest.fixture()
def client(account_store, stripe_client, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "pay3-zone-test-secret")
    monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")
    monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
    monkeypatch.delenv("BILLING_ALLOWED_COUNTRIES", raising=False)
    monkeypatch.delenv(webhook_health.ALERT_WEBHOOK_ENV, raising=False)
    webhook_health.reset_for_tests()
    app = create_app(account_store=account_store, stripe_client=stripe_client)
    app.state.app_state.market_reading_assembler = FakeAssembler()
    return TestClient(app)


def _register(client, email="zone@example.com", password="longpassword1"):
    resp = client.post(
        "/api/auth/register",
        json={
            "email": email,
            "password": password,
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert resp.status_code == 201, resp.text
    return resp.json()


def _verify_email(client, account_store, account_id):
    token = account_store.create_email_verification(account_id)
    assert client.post(
        "/api/auth/verify-email/confirm", json={"token": token}
    ).status_code == 200


def _checkout_completed(client, account_id, *, country, event_id="evt_zone_1",
                        customer="cus_zone_1", subscription="sub_zone_1"):
    """Post a REAL-shaped ``checkout.session.completed`` with a billing country."""
    session: Dict[str, Any] = {
        "id": "cs_zone_1",
        "customer": customer,
        "subscription": subscription,
        "client_reference_id": str(account_id),
        "metadata": {"account_id": str(account_id)},
    }
    if country is not None:
        session["customer_details"] = {"address": {"country": country}}
    payload = {
        "id": event_id,
        "type": "checkout.session.completed",
        "created": int(time.time()),
        "data": {"object": session},
    }
    return client.post(
        "/api/billing/webhook",
        content=json.dumps(payload),
        headers={"Stripe-Signature": "good", "Content-Type": "application/json"},
    )


# =============================================================================
# G2 — the selling zone
# =============================================================================

class TestZoneHelpers:
    def test_default_zone_is_canada_and_united_states(self):
        assert allowed_countries() == frozenset({"CA", "US"})

    @pytest.mark.parametrize("code", ["CA", "ca", "US", " us "])
    def test_in_zone(self, code):
        assert is_allowed_country(code) is True

    @pytest.mark.parametrize("code", ["BE", "FR", "GB", "MX", "de"])
    def test_out_of_zone(self, code):
        assert is_allowed_country(code) is False

    def test_unknown_country_is_never_a_lockout(self):
        # A missing address must not refuse a legitimate customer; Radar is the
        # layer that blocks a card without needing an address.
        assert is_allowed_country(None) is True
        assert is_allowed_country("") is True

    def test_blank_env_falls_back_to_the_default_zone(self, monkeypatch):
        # A typo in the env var must never silently open the world.
        monkeypatch.setenv("BILLING_ALLOWED_COUNTRIES", "   ")
        assert allowed_countries() == frozenset({"CA", "US"})

    def test_env_can_widen_the_zone(self, monkeypatch):
        monkeypatch.setenv("BILLING_ALLOWED_COUNTRIES", "ca, us ,mx")
        assert allowed_countries() == frozenset({"CA", "US", "MX"})

    def test_country_is_read_from_the_session_shape_stripe_sends(self):
        assert billing_country_from_session(
            {"customer_details": {"address": {"country": "ca"}}}
        ) == "CA"
        assert billing_country_from_session({"address": {"country": "US"}}) == "US"
        assert billing_country_from_session({}) is None


class TestCheckoutCollectsTheBillingAddress:
    """Without a MANDATORY billing address there is no country to check, so the
    whole net is blind. This pins the Checkout parameter itself."""

    def test_checkout_session_requires_a_billing_address(self, monkeypatch):
        from src.billing.stripe_client import StripeClient

        captured: Dict[str, Any] = {}

        class _Sessions:
            @staticmethod
            def create(**params):
                captured.update(params)
                return {"id": "cs_x", "url": "https://checkout.stripe.test/cs_x"}

        class _Checkout:
            Session = _Sessions

        class _FakeStripeModule:
            checkout = _Checkout

        sc = StripeClient(api_key="sk_test_x", webhook_secret="whsec_x")
        monkeypatch.setattr(sc, "_require", lambda: _FakeStripeModule)
        sc.create_checkout_session(
            price_id="price_1",
            success_url="https://app.test/ok",
            cancel_url="https://app.test/no",
            customer="cus_1",
            account_id=7,
        )
        assert captured.get("billing_address_collection") == "required", (
            "Checkout no longer collects a billing address — the CA/US zone "
            "check has nothing to read and every country would be accepted."
        )
        # PRIX-1 invariants must survive this change.
        assert captured.get("allow_promotion_codes") is False
        assert "automatic_tax" not in captured


class TestOutOfZoneIsRefused:
    def test_belgian_checkout_is_cancelled_refunded_and_denied(
        self, client, account_store, stripe_client
    ):
        acct = _register(client, email="brussels@example.com")
        _verify_email(client, account_store, acct["id"])

        resp = _checkout_completed(client, acct["id"], country="BE")
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body.get("refused") == "out_of_zone"
        assert body.get("country") == "BE"

        # The money is given back and the subscription killed — not "sorted later".
        assert stripe_client.refunded == ["sub_zone_1"]
        assert stripe_client.cancelled == ["sub_zone_1"]

        # The persisted status grants nothing…
        sub = account_store.get_subscription(acct["id"]) or {}
        assert sub.get("status") == BLOCKED_REGION_STATUS

        # …and that is what actually matters: no data, and no access.
        assert client.get(DATA_ROUTE).status_code == 402
        assert client.get("/api/billing/subscription").json()["has_access"] is False

    def test_canadian_checkout_is_linked_normally(
        self, client, account_store, stripe_client
    ):
        acct = _register(client, email="montreal@example.com")
        _verify_email(client, account_store, acct["id"])

        resp = _checkout_completed(client, acct["id"], country="CA")
        assert resp.status_code == 200, resp.text
        assert resp.json().get("refused") is None
        assert stripe_client.cancelled == []
        assert stripe_client.refunded == []

        sub = account_store.get_subscription(acct["id"]) or {}
        assert sub.get("stripe_customer_id") == "cus_zone_1"
        assert sub.get("status") != BLOCKED_REGION_STATUS

    def test_session_without_an_address_is_not_refused(
        self, client, account_store, stripe_client
    ):
        acct = _register(client, email="noaddress@example.com")
        _verify_email(client, account_store, acct["id"])

        resp = _checkout_completed(client, acct["id"], country=None)
        assert resp.status_code == 200, resp.text
        assert resp.json().get("refused") is None
        assert stripe_client.cancelled == []

    def test_refusal_survives_a_stripe_failure(
        self, client, account_store, stripe_client
    ):
        """If cancelling at Stripe fails, the refusal must STILL be recorded —
        the customer must never end up with access because cleanup errored."""
        def _boom(subscription_id):
            raise RuntimeError("stripe is down")

        stripe_client.cancel_subscription = _boom
        acct = _register(client, email="london@example.com")
        _verify_email(client, account_store, acct["id"])

        resp = _checkout_completed(client, acct["id"], country="GB")
        assert resp.status_code == 200, resp.text
        assert (account_store.get_subscription(acct["id"]) or {}).get(
            "status"
        ) == BLOCKED_REGION_STATUS
        assert client.get(DATA_ROUTE).status_code == 402


# =============================================================================
# G4 — a dead webhook cannot stay silent
# =============================================================================

class TestSyncRescueIsLoud:
    def test_rescue_is_counted_and_logged(self, client, account_store, stripe_client, caplog):
        acct = _register(client, email="rescued@example.com")
        _verify_email(client, account_store, acct["id"])
        assert client.get(DATA_ROUTE).status_code == 402

        # Stripe says "paid"; our DB knows nothing — i.e. the webhook never came.
        stripe_client.find_customer_by_email = lambda email: "cus_rescue"
        stripe_client.remote_subscription = {
            "subscription_id": "sub_rescue",
            "status": "active",
            "current_period_end": time.time() + 30 * 86400,
            "cancel_at_period_end": False,
            "trial_end": None,
            "price_id": "price_monthly_test",
        }

        with caplog.at_level("ERROR"):
            resp = client.post("/api/billing/sync")
        assert resp.status_code == 200, resp.text
        assert resp.json()["has_access"] is True  # the customer IS rescued…

        # …and the breakage is recorded rather than hidden by its own workaround.
        assert webhook_health.snapshot()["rescues"] == 1
        assert webhook_health.snapshot()["last_account_id"] == acct["id"]
        assert any("WEBHOOK MISS" in r.message for r in caplog.records), (
            "the rescue was silent — a dead webhook would go unnoticed"
        )

    def test_no_rescue_recorded_when_state_was_already_known(
        self, client, account_store, stripe_client
    ):
        acct = _register(client, email="known@example.com")
        _verify_email(client, account_store, acct["id"])
        account_store.upsert_subscription(
            acct["id"],
            stripe_customer_id="cus_known",
            stripe_subscription_id="sub_known",
            status="active",
            price_id="price_monthly_test",
            current_period_end=time.time() + 30 * 86400,
            cancel_at_period_end=False,
            trial_end=None,
        )
        stripe_client.remote_subscription = {
            "subscription_id": "sub_known",
            "status": "active",
            "current_period_end": time.time() + 30 * 86400,
            "cancel_at_period_end": False,
            "trial_end": None,
            "price_id": "price_monthly_test",
        }
        assert client.post("/api/billing/sync").status_code == 200
        assert webhook_health.snapshot()["rescues"] == 0


# =============================================================================
# G5 — the renewal notice is locked until counsel approves the text
# =============================================================================

class _FakeRenewalStore:
    def __init__(self):
        self.claimed: List[tuple] = []

    def renewals_due(self, price_id, *, now, lead_seconds, kind):
        return [{"account_id": 1, "email": "annual@example.com", "period_end": now + lead_seconds}]

    def record_renewal_notice(self, account_id, period_end, kind, *, now):
        self.claimed.append((account_id, period_end, kind))
        return True


class TestRenewalNoticeLegalLock:
    def test_nothing_is_sent_while_the_text_is_unapproved(self, monkeypatch, caplog):
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.test")  # SMTP ready…
        monkeypatch.delenv(renewal_notices.TEXT_APPROVED_ENV, raising=False)

        store = _FakeRenewalStore()
        sent_calls: List[Any] = []
        monkeypatch.setattr(
            renewal_notices, "_send_email",
            lambda *a, **k: sent_calls.append(a) or True,
        )

        with caplog.at_level("WARNING"):
            sent = renewal_notices.send_due_renewal_notices(store)

        assert sent == 0
        assert sent_calls == [], "an unapproved legal notice was sent to a customer"
        assert store.claimed == [], "a notice was claimed as sent while none went out"
        assert any(
            renewal_notices.TEXT_APPROVED_ENV in r.message for r in caplog.records
        ), "the lock is silent — nobody would know why notices stopped"

    def test_notices_flow_once_the_text_is_approved(self, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
        monkeypatch.setenv(renewal_notices.TEXT_APPROVED_ENV, "1")

        store = _FakeRenewalStore()
        sent_calls: List[Any] = []

        def _fake_send(to_email, subject, body):
            sent_calls.append((to_email, subject, body))
            return True

        monkeypatch.setattr(renewal_notices, "_send_email", _fake_send)
        sent = renewal_notices.send_due_renewal_notices(store)

        assert sent == 1
        assert sent_calls and sent_calls[0][0] == "annual@example.com"
        body = sent_calls[0][2]
        # LPC: cancelling must be presented as being as easy as subscribing.
        assert "résilier" in body and "/abonnement" in body

    def test_smtp_alone_is_not_approval(self, monkeypatch):
        """The two questions are different: 'can we send' vs 'may we send'."""
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual_test")
        monkeypatch.setenv("SMTP_HOST", "smtp.example.test")
        monkeypatch.setenv(renewal_notices.TEXT_APPROVED_ENV, "0")
        monkeypatch.setattr(renewal_notices, "_send_email", lambda *a, **k: True)
        assert renewal_notices.send_due_renewal_notices(_FakeRenewalStore()) == 0
