"""Subscription-gate freemium tests (mission ③).

Proves the access matrix end-to-end, SERVER-SIDE (not UI masking):

* VISITOR    → 401 on every feature route when the gate is enforced.
* FREE       → XAU/USD M15 reading/chart pass; other markets/timeframes and the
               scanner are 402; chat is capped at N/day then 402.
* SUBSCRIBER → everything passes (no 402).
* OWNER      → everything passes, even with NO subscription (bypass).
* EXPIRY     → an account whose paid period lapsed degrades to FREE cleanly.
* GATE OFF   → the default testing posture leaves every route fully open
               (anonymous included), so nothing breaks before launch.

"Passes the gate" is asserted as a 503 (the feature service isn't wired in these
unit tests) — i.e. the request got PAST the 401/402 wall into the route body.
Blocked requests assert the exact 401/402. The ``/api/access/me`` summary is
asserted directly for the precise per-tier perimeter.

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
    # Small free quota so the chat-exhaustion test is short.
    monkeypatch.setenv("FREE_CHAT_DAILY_LIMIT", "3")
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


def _free(app):
    return _client(app, "freeuser", "free@example.com")


def _subscriber(app, account_store, *, period_offset_s=3600.0):
    c, acct = _client(app, "subuser", "sub@example.com")
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
    # A valid body (conditions require ≥1 item) so the freemium guard — not
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
    def test_anonymous_passes_feature_routes(self, app, monkeypatch):
        monkeypatch.delenv("SUBSCRIPTION_GATE_ENFORCED", raising=False)
        c = TestClient(app)  # no cookie at all
        # Open → reaches the (unwired) service: 503, never 401/402.
        assert _reading(c).status_code == 503
        assert _candles(c).status_code == 503
        assert _scan(c).status_code == 503

    def test_access_me_anonymous_is_full_when_off(self, app, monkeypatch):
        monkeypatch.delenv("SUBSCRIPTION_GATE_ENFORCED", raising=False)
        c = TestClient(app)
        body = c.get("/api/access/me").json()
        assert body["authenticated"] is False
        assert body["has_full_access"] is True
        assert body["entitlements"]["instruments"] is None  # unrestricted


# =============================================================================
# Gate ON — VISITOR
# =============================================================================

class TestVisitorEnforced:
    def test_all_feature_routes_401(self, app, monkeypatch):
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
        assert body["has_full_access"] is False


# =============================================================================
# Gate ON — FREE (Découverte): XAU/USD M15 only, no scanner, capped chat
# =============================================================================

class TestFreeEnforced:
    def test_xau_m15_passes(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        assert _reading(c, "XAUUSD", "M15").status_code == 503  # past the wall
        assert _candles(c, "XAUUSD", "M15").status_code == 503

    def test_other_timeframe_blocked(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        assert _reading(c, "XAUUSD", "H1").status_code == 402
        assert _candles(c, "XAUUSD", "H4").status_code == 402

    def test_other_market_blocked(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        assert _reading(c, "EURUSD", "M15").status_code == 402
        assert c.get("/api/live-price?instrument=EURUSD").status_code == 402

    def test_scanner_blocked(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        assert _scan(c).status_code == 402

    def test_chat_quota_then_402(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        # FREE_CHAT_DAILY_LIMIT=3 → first 3 turns clear the gate (503: bot
        # unwired), the 4th is refused with an upsell 402.
        for _ in range(3):
            assert _chat(c).status_code == 503
        r = _chat(c)
        assert r.status_code == 402
        assert "abonnement" in r.json()["detail"].lower()

    def test_access_me_free_perimeter(self, app, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _free(app)
        ent = c.get("/api/access/me").json()
        assert ent["tier"] == "free"
        assert ent["has_full_access"] is False
        assert ent["entitlements"]["instruments"] == ["XAUUSD"]
        assert ent["entitlements"]["timeframes"] == ["M15"]
        assert ent["entitlements"]["scanner"] is False
        assert ent["entitlements"]["chat"]["limit"] == 3


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
        # Well beyond the free cap of 3 — never a 402.
        for _ in range(6):
            assert _chat(c).status_code == 503

    def test_access_me_subscriber(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _subscriber(app, account_store)
        ent = c.get("/api/access/me").json()
        assert ent["tier"] == "subscriber"
        assert ent["has_full_access"] is True
        assert ent["entitlements"]["instruments"] is None
        assert ent["entitlements"]["scanner"] is True
        assert ent["entitlements"]["chat"]["limit"] is None


# =============================================================================
# Gate ON — OWNER: everything, even with NO subscription
# =============================================================================

class TestOwnerEnforced:
    def test_owner_bypasses_all(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _owner(app, account_store)
        assert _reading(c, "EURUSD", "H4").status_code == 503
        assert _scan(c).status_code == 503
        for _ in range(6):  # unlimited chat
            assert _chat(c).status_code == 503

    def test_access_me_owner(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _owner(app, account_store)
        ent = c.get("/api/access/me").json()
        assert ent["tier"] == "owner"
        assert ent["is_owner"] is True
        assert ent["has_full_access"] is True


# =============================================================================
# Gate ON — EXPIRY: clean downgrade to FREE
# =============================================================================

class TestExpiryDowngrade:
    def test_expired_subscription_becomes_free(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        # Period ended in the past → has_active_subscription() is False → FREE.
        c, _ = _subscriber(app, account_store, period_offset_s=-10.0)
        # Degrades cleanly: free perimeter applies, NO raw error.
        assert _reading(c, "XAUUSD", "M15").status_code == 503   # still gets XAU M15
        assert _reading(c, "EURUSD", "M15").status_code == 402   # but not EUR
        ent = c.get("/api/access/me").json()
        assert ent["tier"] == "free"
        assert ent["has_full_access"] is False
