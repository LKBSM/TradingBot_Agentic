"""PAY-2 — access is the condition of entry, and no redirect is hard-coded.

Two guarantees, server-side (never UI masking):

1. The market-data routes that PAY-2 newly gated — the economic calendar and
   the publication measures (feeding ``/actualites``) and the persisted
   structure/coverage endpoints — now refuse a request with no active
   subscription, exactly like the reading/candles/scanner routes:
     * VISITOR (gate on)      → 401
     * UNSUBSCRIBED (verified) → 402
     * SUBSCRIBER / OWNER      → past the wall (never 401/402/403)
   Gate OFF (default testing posture) leaves them open, so nothing breaks
   before launch.

2. The outward-facing base URLs come from a single env-driven source and never
   default to localhost in production: the app REFUSES TO BOOT if
   ``APP_PUBLIC_URL``/``API_PUBLIC_URL`` are missing or localhost when
   ``ENVIRONMENT=production``, and every redirect (Stripe success/cancel/portal,
   Google OAuth, email links) resolves off those bases — no ``localhost`` leaks
   into a production redirect.

The webhook signature-rejection and idempotency guarantees are covered in
tests/test_account_billing.py (test_invalid_signature_rejected,
test_duplicate_event_applied_once); PAY-2 does not change them.
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app

PASSWORD = "longpassword1"


# =============================================================================
# Fixtures + helpers (mirror tests/test_subscription_gate_paid_only.py)
# =============================================================================

@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "pay2_accounts.db"))


@pytest.fixture()
def app(account_store, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    return create_app(account_store=account_store)


def _register(app, username, email):
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
    token = account_store.create_email_verification(account_id)
    if token is not None:
        assert account_store.consume_email_verification(token) is True


def _unsubscribed(app, account_store):
    c, acct = _register(app, "np_plain", "np_plain@example.com")
    _verify(account_store, acct["id"])
    return c, acct


def _subscriber(app, account_store, *, period_offset_s=3600.0):
    c, acct = _register(app, "np_sub", "np_sub@example.com")
    _verify(account_store, acct["id"])
    account_store.upsert_subscription(
        acct["id"],
        stripe_customer_id="cus_pay2",
        stripe_subscription_id="sub_pay2",
        status="active",
        price_id="price_test",
        current_period_end=time.time() + period_offset_s,
    )
    return c, acct


def _enforce(monkeypatch):
    monkeypatch.setenv("SUBSCRIPTION_GATE_ENFORCED", "1")


# The routes PAY-2 newly gated. Each is a (method, path) that serves market data
# / product intelligence and must sit behind the paywall.
NEWLY_GATED = [
    ("GET", "/api/calendar?lookahead_days=7&lookback_days=3"),
    ("GET", "/api/calendar/month?month=2026-08"),
    ("GET", "/api/calendar/event/bea:us_gdp:2026-08-26"),
    ("GET", "/api/publications/us_cpi/measures"),
    ("GET", "/api/structure?instrument=XAUUSD&timeframe=M15"),
    ("GET", "/api/coverage"),
]


def _call(c: TestClient, method: str, path: str):
    return c.get(path) if method == "GET" else c.post(path)


# =============================================================================
# 1. Newly-gated market-data routes
# =============================================================================

class TestNewlyGatedRoutesVisitor:
    def test_all_401_when_enforced_and_anonymous(self, app, monkeypatch):
        _enforce(monkeypatch)
        c = TestClient(app)
        for method, path in NEWLY_GATED:
            assert _call(c, method, path).status_code == 401, path


class TestNewlyGatedRoutesUnsubscribed:
    def test_all_402_when_authenticated_without_subscription(
        self, app, account_store, monkeypatch
    ):
        _enforce(monkeypatch)
        c, _ = _unsubscribed(app, account_store)
        for method, path in NEWLY_GATED:
            assert _call(c, method, path).status_code == 402, path


class TestNewlyGatedRoutesSubscriber:
    def test_subscriber_passes_the_wall(self, app, account_store, monkeypatch):
        _enforce(monkeypatch)
        c, _ = _subscriber(app, account_store)
        for method, path in NEWLY_GATED:
            r = _call(c, method, path)
            # Past the wall: whatever the service returns, it is NEVER a gate code.
            assert r.status_code not in (401, 402, 403), f"{path} -> {r.status_code}"


class TestNewlyGatedRoutesGateOff:
    def test_open_when_gate_off(self, app, monkeypatch):
        monkeypatch.delenv("SUBSCRIPTION_GATE_ENFORCED", raising=False)
        c = TestClient(app)  # anonymous
        for method, path in NEWLY_GATED:
            r = _call(c, method, path)
            assert r.status_code not in (401, 402, 403), f"{path} -> {r.status_code}"


# =============================================================================
# 2. Public URLs — single source, no localhost in production
# =============================================================================

class TestPublicUrlGuard:
    def test_noop_in_dev(self, monkeypatch):
        from src.api import public_urls

        monkeypatch.delenv("ENVIRONMENT", raising=False)
        # Must not raise even with nothing configured (localhost fallbacks apply).
        public_urls.assert_public_urls_configured()

    def test_refuses_to_boot_in_prod_without_urls(self, monkeypatch):
        from src.api import public_urls

        monkeypatch.setenv("ENVIRONMENT", "production")
        for var in ("APP_PUBLIC_URL", "APP_URL", "FRONTEND_BASE_URL",
                    "API_PUBLIC_URL", "API_BASE_URL"):
            monkeypatch.delenv(var, raising=False)
        with pytest.raises(public_urls.PublicURLMisconfigured):
            public_urls.assert_public_urls_configured()

    def test_refuses_localhost_in_prod(self, monkeypatch):
        from src.api import public_urls

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("APP_PUBLIC_URL", "http://localhost:3000")
        monkeypatch.setenv("API_PUBLIC_URL", "http://localhost:8000")
        with pytest.raises(public_urls.PublicURLMisconfigured):
            public_urls.assert_public_urls_configured()

    def test_passes_in_prod_with_real_urls(self, monkeypatch):
        from src.api import public_urls

        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.setenv("APP_PUBLIC_URL", "https://mia.markets")
        monkeypatch.setenv("API_PUBLIC_URL", "https://api.mia.markets")
        public_urls.assert_public_urls_configured()  # must not raise


class TestNoLocalhostRedirectInProduction:
    """With production public URLs set, EVERY redirect the backend builds is an
    https origin — no ``localhost`` leaks into a Stripe/Google/email redirect."""

    def test_stripe_and_email_redirects_use_public_url(self, monkeypatch):
        monkeypatch.setenv("APP_PUBLIC_URL", "https://mia.markets")
        monkeypatch.setenv("API_PUBLIC_URL", "https://api.mia.markets")
        # No per-redirect overrides → they must default THROUGH the public base.
        for var in ("STRIPE_SUCCESS_URL", "STRIPE_CANCEL_URL",
                    "STRIPE_PORTAL_RETURN_URL", "GOOGLE_REDIRECT_URI", "APP_URL"):
            monkeypatch.delenv(var, raising=False)

        from src.api.routes import account_billing, accounts, google_auth

        urls = [
            account_billing._success_url(),
            account_billing._cancel_url(),
            account_billing._portal_return_url(),
            accounts._reset_base_url(),
            google_auth._redirect_uri(),
            google_auth._app_url(),
        ]
        for u in urls:
            assert "localhost" not in u, u
            assert u.startswith("https://"), u
        assert account_billing._success_url() == "https://mia.markets/abonnement?status=success"
        # PAY-3: the Google callback now defaults to the FRONTEND origin (same
        # origin as /start, which the browser proxies to the backend) so the CSRF
        # state cookie — set for the front origin — is presented to the callback.
        # Routing it to the backend origin was the silent-failure bug.
        assert google_auth._redirect_uri() == "https://mia.markets/api/auth/google/callback"
