"""PAY-3 — the end-to-end access journey, guarded so it fails LOUDLY.

The single most expensive defect this product can ship is: the customer pays,
Stripe records it, but the app never flips their access — they hit the paywall
again and leave without complaining. It triggers no alarm; it shows up as a
conversion dip blamed on something else for weeks.

The defence is THIS test. It drives the WHOLE journey through the REAL routes,
the REAL subscription gate and the REAL account store (only the Stripe network
is faked, exactly as the hosted Checkout/Portal never touch our servers):

  register → verify email → NO access (402) → subscription webhook → ACCESS.

If any link in that chain breaks — the webhook parser, the store upsert, the
gate read — the ``has_access`` assertion goes red at ``pytest``, i.e. at every
deploy. The whole point is that "paid but no access" can never ship silently.

It also pins the other PAY-3 invariants the mission calls out:

  · the Google path grants access the same way (email pre-verified),
  · one email address on both paths yields exactly ONE account,
  · an account with no subscription gets NO market data by direct API call,
  · abandoning Checkout leaves a clean "subscribe" state, not an error,
  · and a SEPARATE test exercises the REAL Stripe signature verification
    (``stripe.Webhook.construct_event``) so a wrong webhook secret / signature
    format is caught, not just our fake — it SKIPS LOUDLY if unrunnable.
"""

from __future__ import annotations

import hmac
import json
import time
from hashlib import sha256

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountError, AccountStore
from src.api.app import create_app
from src.api.routes.legal import LAST_UPDATED as LEGAL_VERSION


# =============================================================================
# Fakes — no Stripe network, no real card. Mirrors StripeClient's surface.
# =============================================================================

class FakeStripeClient:
    def __init__(self):
        self.is_configured = True
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


class _StatusObj:
    """Minimal market-status object with a ``to_dict`` (what the route serves)."""

    def __init__(self, instrument, timeframe):
        self._d = {"instrument": instrument, "timeframe": timeframe, "state": "open"}

    def to_dict(self):
        return self._d


class FakeAssembler:
    """Just enough for /api/market-status to return 200 once the gate passes, so
    the "access granted" leg is a REAL 200 with data, not merely a non-402."""

    def market_status(self, instrument, timeframe):
        return _StatusObj(instrument, timeframe)


# =============================================================================
# Fixtures — the gate is ENFORCED (production posture), like Render.
# =============================================================================

@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "pay3_accounts.db"))


@pytest.fixture()
def client(account_store, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "pay3-test-session-secret")
    monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")  # the wall MUST bite
    monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
    app = create_app(account_store=account_store, stripe_client=FakeStripeClient())
    # Wire a tiny market-status assembler so the access-granted leg is a real 200.
    app.state.app_state.market_reading_assembler = FakeAssembler()
    return TestClient(app)


DATA_ROUTE = "/api/market-status?instrument=XAUUSD&timeframe=M15"


def _register(client, email="buyer@example.com", password="longpassword1"):
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
    """Confirm the email through the REAL endpoint (mint a token, then confirm)."""
    token = account_store.create_email_verification(account_id)
    resp = client.post("/api/auth/verify-email/confirm", json={"token": token})
    assert resp.status_code == 200, resp.text


def _sub_webhook(client, account_id, *, event_id="evt_1", status="active",
                 customer="cus_1", price="price_monthly_test", period_end=None):
    payload = {
        "id": event_id,
        "type": "customer.subscription.created",
        "created": int(time.time()),
        "data": {"object": {
            "id": "sub_1",
            "customer": customer,
            "status": status,
            "current_period_end": period_end or (time.time() + 30 * 86400),
            "cancel_at_period_end": False,
            "trial_end": None,
            "metadata": {"account_id": str(account_id)},
            "items": {"data": [{"price": {"id": price}}]},
        }},
    }
    return client.post(
        "/api/billing/webhook",
        content=json.dumps(payload),
        headers={"Stripe-Signature": "good", "Content-Type": "application/json"},
    )


# =============================================================================
# THE journey — the loud guard against "paid but no access".
# =============================================================================

class TestEmailJourneyGrantsAccess:
    def test_register_verify_pay_access(self, client, account_store):
        acct = _register(client)

        # 1) Unverified → the verification wall bites BEFORE the paywall.
        assert client.get(DATA_ROUTE).status_code == 403

        # 2) Verified but unpaid → market data is refused (402), not served.
        _verify_email(client, account_store, acct["id"])
        assert client.get(DATA_ROUTE).status_code == 402
        access = client.get("/api/access/me").json()
        assert access["has_access"] is False
        assert access["subscription_required"] is True

        # 3) The subscription webhook lands → access MUST be granted. This is the
        #    assertion whose failure means a real customer would be locked out
        #    after paying. It fails LOUDLY, at every deploy, if the wiring breaks.
        resp = _sub_webhook(client, acct["id"])
        assert resp.status_code == 200 and resp.json().get("applied") is True, resp.text

        billing = client.get("/api/billing/subscription").json()
        assert billing["has_access"] is True, (
            "PAID BUT NO ACCESS: the subscription webhook was accepted but the "
            "account still resolves to no access. This is the most expensive "
            "defect in the product — the access-granting wiring is broken."
        )
        assert client.get("/api/access/me").json()["has_access"] is True

        # 4) The gated market-data route now serves real data (a true 200).
        served = client.get(DATA_ROUTE)
        assert served.status_code == 200, served.text
        assert served.json()["state"] == "open"


class TestStripeResultsAreDicts:
    """PAY-3d — Stripe SDK results must be converted to plain dicts, or the
    checkout path 500s on ``.get()`` even with a valid key (the "Internal Server
    Error" on the first click of Subscribe)."""

    def test_to_dict_neutralises_stripe_object_get(self):
        stripe = pytest.importorskip(
            "stripe",
            reason="LOUD SKIP: the `stripe` SDK is not installed, so the real "
            "StripeObject conversion cannot be exercised.",
        )
        from src.billing.stripe_client import StripeClient

        obj = stripe.Customer.construct_from(
            {"id": "cus_x", "object": "customer"}, "sk_test"
        )
        # The failure mode we are guarding against: .get() on a StripeObject.
        with pytest.raises(AttributeError):
            obj.get("id")
        # After conversion the caller's ``.get("id")`` works.
        d = StripeClient._to_dict(obj)
        assert isinstance(d, dict)
        assert d.get("id") == "cus_x"


class TestEmailCodeVerification:
    """PAY-3c — email confirmed by a typed 6-digit CODE (not only the link)."""

    def test_code_confirms_email_and_unblocks(self, client, account_store):
        acct = _register(client, email="coder@example.com")
        # Unverified → the verification wall blocks data (403) before the paywall.
        assert client.get(DATA_ROUTE).status_code == 403
        # Issue a challenge and read its code (the emailed value).
        challenge = account_store.create_email_verification_challenge(acct["id"])
        assert challenge is not None
        _, code = challenge
        # A wrong code is refused …
        assert client.post(
            "/api/auth/verify-email/confirm-code", json={"code": "000000"}
        ).status_code == 400
        # … the right code verifies the account.
        ok = client.post("/api/auth/verify-email/confirm-code", json={"code": code})
        assert ok.status_code == 200, ok.text
        # Now verified → the wall is 402 (subscribe), no longer 403 (verify).
        assert client.get(DATA_ROUTE).status_code == 402

    def test_code_requires_a_session(self, client):
        fresh = TestClient(client.app)
        fresh.cookies.clear()
        resp = fresh.post("/api/auth/verify-email/confirm-code", json={"code": "123456"})
        assert resp.status_code == 401


class TestGoogleJourneyGrantsAccess:
    """The Google path grants access the same way — email is pre-verified, so
    there is no verification wall, but the paywall still applies until payment."""

    def _finalize_google(self, client, account_store, email):
        # Drive the REAL finalize route with a signed pending token, as the
        # callback would hand to the consent step.
        from src.api.routes import google_auth

        monkey_secret = "pay3-test-session-secret"
        signer = google_auth._signer  # uses SESSION_SECRET set in the fixture
        token = signer().dumps({"email": email, "verified": True})
        # Google must be "configured" for the finalize route to be live.
        return token

    def test_google_finalize_then_pay(self, client, account_store, monkeypatch):
        monkeypatch.setenv("GOOGLE_CLIENT_ID", "gid")
        monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "gsecret")
        email = "google.user@example.com"
        token = self._finalize_google(client, account_store, email)

        resp = client.post("/api/auth/google/finalize", json={
            "token": token,
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        })
        assert resp.status_code == 200, resp.text
        acct_id = resp.json()["id"]

        # Email pre-verified by Google → no 403; unpaid → 402.
        assert client.get(DATA_ROUTE).status_code == 402
        # Pay → access granted, same wiring as the email path.
        assert _sub_webhook(client, acct_id, event_id="evt_g",
                            customer="cus_g").status_code == 200
        assert client.get("/api/billing/subscription").json()["has_access"] is True


# =============================================================================
# Account linking — one email, one account, on BOTH paths.
# =============================================================================

class TestOneEmailOneAccount:
    def test_password_then_google_is_the_same_account(self, client, account_store):
        acct = _register(client, email="dual@example.com")

        # The Google callback resolves an EXISTING account by email BEFORE any
        # creation — so a Google sign-in with the same address logs into the very
        # same account (with its subscription), never a second empty one.
        same = account_store.get_account_by_identifier("dual@example.com")
        assert same is not None and same["id"] == acct["id"]

        # And the create path the Google finalize uses refuses a duplicate email
        # (unique constraint + pre-check), so a second account is impossible.
        with pytest.raises(AccountError) as exc:
            account_store.create_account_auto(
                "dual@example.com", "anotherlongpass1",
                age_confirmed=True,
                consents=[("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)],
            )
        assert exc.value.code in {"email_taken", "account_conflict"}

    def test_derived_usernames_are_unique(self, account_store):
        # Two different emails whose local-part collides must still yield two
        # accounts with DISTINCT usernames (the derivation retries on collision).
        a = account_store.create_account_auto(
            "sam@example.com", "longpassword1", age_confirmed=True,
            consents=[("terms", "1"), ("privacy", "1")],
        )
        b = account_store.create_account_auto(
            "sam@other.com", "longpassword1", age_confirmed=True,
            consents=[("terms", "1"), ("privacy", "1")],
        )
        assert a["username"] != b["username"]


# =============================================================================
# No subscription → no data, by DIRECT API call (not just a hidden button).
# =============================================================================

class TestNoSubscriptionNoData:
    def test_unsubscribed_account_gets_402_by_direct_call(self, client, account_store):
        acct = _register(client, email="lurker@example.com")
        _verify_email(client, account_store, acct["id"])
        # Every gated data route refuses a verified-but-unpaid account.
        for route in (
            DATA_ROUTE,
            "/api/market-reading?instrument=XAUUSD&timeframe=M15",
        ):
            assert client.get(route).status_code == 402, route

    def test_anonymous_gets_no_market_status(self, client):
        # market-status was the one data route with NO auth at all (PAY-3 hole):
        # an anonymous caller must now be refused (401 under the enforced gate).
        fresh = TestClient(client.app)
        fresh.cookies.clear()
        assert fresh.get(DATA_ROUTE).status_code == 401


# =============================================================================
# Abandoned Checkout → a clean "subscribe" state, never an error.
# =============================================================================

class TestAbandonedCheckout:
    def test_abandon_leaves_subscribe_state(self, client, account_store):
        acct = _register(client, email="quitter@example.com")
        _verify_email(client, account_store, acct["id"])
        # Start Checkout (links a customer) then walk away — NO webhook.
        assert client.post("/api/billing/checkout",
                           json={"plan_key": "MONTHLY"}).status_code == 200
        # The account exists, has a customer, but NO access. On next login the UI
        # routes it to /abonnement — driven by this exact backend signal.
        access = client.get("/api/access/me").json()
        assert access["authenticated"] is True
        assert access["has_access"] is False
        assert access["subscription_required"] is True
        assert client.get(DATA_ROUTE).status_code == 402


# =============================================================================
# The REAL Stripe signature verification — skipped LOUDLY, never silently.
# =============================================================================

class TestRealStripeSignature:
    """Exercises ``StripeClient.verify_webhook`` → ``stripe.Webhook.construct_event``
    with a genuinely-signed payload. This catches a wrong ``STRIPE_WEBHOOK_SECRET``
    or a signature-format drift that the FakeStripeClient can't. Requires the
    ``stripe`` SDK; if absent, it SKIPS with a loud reason (no silent pass)."""

    def _signed_header(self, payload: bytes, secret: str) -> str:
        ts = int(time.time())
        signed = f"{ts}.".encode() + payload
        sig = hmac.new(secret.encode(), signed, sha256).hexdigest()
        return f"t={ts},v1={sig}"

    def test_real_signature_grants_access(self, account_store, monkeypatch):
        stripe = pytest.importorskip(
            "stripe",
            reason="LOUD SKIP: the `stripe` SDK is not installed, so the REAL "
            "signature-verification path cannot run. Install it to run the true "
            "end-to-end Stripe integration test.",
        )
        from src.billing.stripe_client import StripeClient

        secret = "whsec_pay3_test_secret"
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
        monkeypatch.setenv("SESSION_SECRET", "pay3-real-sig-secret")
        monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")
        monkeypatch.setenv("STRIPE_PRICE_MONTHLY", "price_monthly_test")
        # A dummy api key makes the client "configured"; construct_event uses only
        # the webhook secret and does NOT hit the network.
        real = StripeClient(api_key="sk_test_dummy", webhook_secret=secret)
        app = create_app(account_store=account_store, stripe_client=real)
        client = TestClient(app)

        acct = _register(client, email="realsig@example.com")
        _verify_email(client, account_store, acct["id"])
        # NB: no /checkout here — that would hit the Stripe network with a dummy
        # key. The webhook carries account_id in metadata, so the access-granting
        # path (verify signature → parse → upsert → gate) is fully exercised.

        payload = json.dumps({
            "id": "evt_real_1",
            "type": "customer.subscription.created",
            "created": int(time.time()),
            "data": {"object": {
                "id": "sub_real_1",
                "customer": "cus_real_1",
                "status": "active",
                "current_period_end": time.time() + 30 * 86400,
                "cancel_at_period_end": False,
                "metadata": {"account_id": str(acct["id"])},
                "items": {"data": [{"price": {"id": "price_monthly_test"}}]},
            }},
        }).encode()

        # A CORRECTLY signed event is accepted and grants access.
        good = client.post(
            "/api/billing/webhook", content=payload,
            headers={"Stripe-Signature": self._signed_header(payload, secret),
                     "Content-Type": "application/json"},
        )
        assert good.status_code == 200 and good.json().get("applied") is True, good.text
        assert client.get("/api/billing/subscription").json()["has_access"] is True

        # A TAMPERED signature (wrong secret) is rejected — no state change.
        bad = client.post(
            "/api/billing/webhook", content=payload,
            headers={"Stripe-Signature": self._signed_header(payload, "whsec_wrong"),
                     "Content-Type": "application/json"},
        )
        assert bad.status_code == 400, bad.text
