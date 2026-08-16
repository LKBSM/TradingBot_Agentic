"""Subscription-gate paid-only tests (PAY-1).

Proves the access decision end-to-end, SERVER-SIDE (not UI masking). PAY-1 has
NO free tier: access is all-or-nothing through the single seam
``subscription_gate.enforce_access``.

* VISITOR      → 401 on every data route when the gate is enforced.
* UNSUBSCRIBED → authenticated but no active subscription → 402 on EVERY data
                 route, including XAU/USD M15 (not even one candle).
* SUBSCRIBER   → everything passes (no 402), chat unlimited.
* OWNER        → everything passes, even with NO subscription (bypass).
* EXPIRY       → an account whose paid period lapsed loses access entirely.
* GATE OFF     → the default testing posture leaves every route fully open
                 (anonymous included), so nothing breaks before launch.

"Passes the gate" is asserted as a 503 (the feature service isn't wired in these
unit tests) — i.e. the request got PAST the 401/402 wall into the route body.
Blocked requests assert the exact 401/402. The ``/api/access/me`` summary is
asserted directly.

No Stripe and no network: a subscriber is simulated by writing the subscription
row the webhook would have written (``AccountStore.upsert_subscription``).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app

PASSWORD = "longpassword1"


# =============================================================================
# Fixtures + helpers
# =============================================================================

@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "gate_accounts.db"))


@pytest.fixture()
def app(account_store, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    return create_app(account_store=account_store)


def _client(app, username, email):
    """A fresh client (own cookie jar) registered + auto-logged-in as a user."""
    c = TestClient(app)
    r = c.post(
        "/api/auth/register",
        json={
            "username": username,
            "email": email,
            "password": PASSWORD,
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert r.status_code == 201, r.text
    return c, r.json()


def _verify(account_store, account_id):
    """Mark an account email-verified via the real token flow (mandatory before
    access, PAY-1). Owner accounts are already verified."""
    token = account_store.create_email_verification(account_id)
    if token is not None:
        assert account_store.consume_email_verification(token) is True


def _unsubscribed(app, account_store):
    c, acct = _client(app, "plainuser", "plain@example.com")
    _verify(account_store, acct["id"])
    return c, acct


def _subscriber(app, account_store, *, period_offset_s=3600.0):
    c, acct = _client(app, "subuser", "sub@example.com")
    _verify(account_store, acct["id"])
    account_store.upsert_subscription(
        acct["id"],
        stripe_customer_id="cus_test_sub",
        stripe_subscription_id="sub_test",
        status="active",
        price_id="price_standard_test",
        current_period_end=time.time() + period_offset_s,
    )
    return c, acct


def _owner(app, account_store):
    c, acct = _client(app, "owneruser", "owner@example.com")
    # Promote the just-created account to owner (no subscription on purpose).
    account_store.seed_owner("owneruser", "owner@example.com", PASSWORD)
    return c, acct


def _reading(c, instrument="XAUUSD", timeframe="M15"):
    return c.get(f"/api/market-reading?instrument={instrument}&timeframe={timeframe}")


def _candles(c, instrument="XAUUSD", timeframe="M15"):
    return c.get(f"/api/candles?instrument={instrument}&timeframe={timeframe}")


def _scan(c):
    # A valid body (conditions require ≥1 item) so the access guard — not
    # request validation — is what decides the outcome.
    return c.post(
        "/api/conditions-scan",
        json={"logic": "AND", "conditions": [{"type": "bos_recent_confirmed"}]},
    )


def _chat(c):
    return c.post("/api/chatbot/message", json={"user_message": "salut", "conversation_history": []})


def _enforce(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")


# =============================================================================
# Gate OFF (default testing posture) — everything stays open
# =============================================================================

class TestGateOffOpen:
    def test_anonymous_passes_data_routes(self, app, monkeypatch):
        monkeypatch.delenv("SUBSCRIPTION_GATE_ENFORCED", raising=False)
        c = TestClient(app)  # no cookie at all
        # Open → reaches the (unwired) service: 503, never 401/402.
        assert _reading(c).status_code == 503
        assert _candles(c).status_code == 503
        assert _scan(c).status_code == 503

    def test_access_me_anonymous_has_access_when_off(self, app, monkeypatch):
        monkeypatch.delenv("SUBSCRIPTION_GATE_ENFORCED", raising=False)
        c = TestClient(app)
        body = c.get("/api/access/me").json()
        assert body["authenticated"] is False
        assert body["has_access"] is True
        assert body["subscription_required"] is False


# =============================================================================
# Gate ON — VISITOR (not authenticated)
# =============================================================================

class TestVisitorEnforced:
    def test_all_data_routes_401(self, app, monkeypatch):
        _enforce(monkeypatch)
        c = TestClient(app)  # not authenticated
        assert _reading(c).status_code == 401
        assert _candles(c).status_code == 401
        assert _scan(c).status_code == 401
        assert _chat(c).status_code == 401
        assert c.get("/api/live-price?instrument=XAUUSD").status_code == 401

    def test_access_me_visitor(self, app, monkeypatch):
        _enforce(monkeypatch)
        c = TestClient(app)
        body = c.get("/api/access/me").json()
        assert body["authenticated"] is False
        assert body["has_access"] is False
        assert body["subscription_required"] is False  # not logged in


# =============================================================================
# Gate ON — UNSUBSCRIBED: authenticated, no subscription → NO data at all
# =============================================================================

class TestUnverifiedEmail:
    """A registered but UNVERIFIED account is walled off before the subscribe
    wall — it must confirm its email first (PAY-1: mandatory before access)."""

    def test_data_routes_403_until_verified(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _client(app, "newbie", "newbie@example.com")  # not verified
        assert _reading(c, "XAUUSD", "M15").status_code == 403
        assert _candles(c, "XAUUSD", "M15").status_code == 403
        assert _scan(c).status_code == 403
        assert _chat(c).status_code == 403

    def test_access_me_reports_verification_required(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _client(app, "newbie2", "newbie2@example.com")
        body = c.get("/api/access/me").json()
        assert body["authenticated"] is True
        assert body["email_verified"] is False
        assert body["email_verification_required"] is True
        assert body["has_access"] is False
        # Not yet at the subscribe step — verification comes first.
        assert body["subscription_required"] is False


class TestUnsubscribedEnforced:
    def test_every_data_route_402_including_xau_m15(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _unsubscribed(app, account_store)
        # PAY-1: not even one candle of XAU/USD M15 without an active sub.
        assert _reading(c, "XAUUSD", "M15").status_code == 402
        assert _candles(c, "XAUUSD", "M15").status_code == 402
        assert _reading(c, "EURUSD", "H1").status_code == 402
        assert c.get("/api/live-price?instrument=XAUUSD").status_code == 402
        assert _scan(c).status_code == 402
        assert _chat(c).status_code == 402

    def test_access_me_unsubscribed(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _unsubscribed(app, account_store)
        body = c.get("/api/access/me").json()
        assert body["authenticated"] is True
        assert body["email_verified"] is True
        assert body["has_access"] is False
        assert body["is_owner"] is False
        # The "account page + subscribe invitation" state.
        assert body["subscription_required"] is True


# =============================================================================
# Gate ON — SUBSCRIBER: full product
# =============================================================================

class TestSubscriberEnforced:
    def test_all_markets_and_scanner_pass(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _subscriber(app, account_store)
        assert _reading(c, "EURUSD", "H4").status_code == 503  # allowed → past wall
        assert _candles(c, "EURUSD", "H1").status_code == 503
        assert _scan(c).status_code == 503

    def test_chat_unlimited(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _subscriber(app, account_store)
        for _ in range(6):
            assert _chat(c).status_code == 503

    def test_access_me_subscriber(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _subscriber(app, account_store)
        body = c.get("/api/access/me").json()
        assert body["has_access"] is True
        assert body["subscription_required"] is False


# =============================================================================
# Gate ON — OWNER: everything, even with NO subscription
# =============================================================================

class TestOwnerEnforced:
    def test_owner_bypasses_all(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _owner(app, account_store)
        assert _reading(c, "EURUSD", "H4").status_code == 503
        assert _scan(c).status_code == 503
        for _ in range(6):
            assert _chat(c).status_code == 503

    def test_access_me_owner(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _owner(app, account_store)
        body = c.get("/api/access/me").json()
        assert body["is_owner"] is True
        assert body["has_access"] is True


# =============================================================================
# Gate ON — EXPIRY: lapsed period loses access entirely
# =============================================================================

class TestExpiryLosesAccess:
    def test_expired_subscription_loses_all_access(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        # Period ended in the past → has_active_subscription() is False.
        c, _ = _subscriber(app, account_store, period_offset_s=-10.0)
        assert _reading(c, "XAUUSD", "M15").status_code == 402   # no freebie
        assert _reading(c, "EURUSD", "M15").status_code == 402
        body = c.get("/api/access/me").json()
        assert body["has_access"] is False
        assert body["subscription_required"] is True
