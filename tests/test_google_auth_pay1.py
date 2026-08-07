"""Google OAuth (PAY-1) — the parts testable without a live Google round trip:
config gating, disabled behaviour, CSRF state, and the consent-gated finalize.
The full authorize→callback exchange needs live Google credentials.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app
from src.api.routes import google_auth


@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "goog_accounts.db"))


@pytest.fixture()
def base_env(monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")


@pytest.fixture()
def app(account_store, base_env):
    return create_app(account_store=account_store)


def _configure(monkeypatch):
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "gid.apps.googleusercontent.com")
    monkeypatch.setenv("GOOGLE_CLIENT_SECRET", "gsecret")


class TestConfigGating:
    def test_config_disabled_by_default(self, app):
        assert TestClient(app).get("/api/auth/google/config").json() == {"enabled": False}

    def test_start_404_when_disabled(self, app):
        assert TestClient(app).get(
            "/api/auth/google/start", follow_redirects=False
        ).status_code == 404

    def test_config_enabled_when_configured(self, app, monkeypatch):
        _configure(monkeypatch)
        assert TestClient(app).get("/api/auth/google/config").json() == {"enabled": True}

    def test_start_redirects_to_google(self, app, monkeypatch):
        _configure(monkeypatch)
        r = TestClient(app).get("/api/auth/google/start", follow_redirects=False)
        assert r.status_code == 302
        assert r.headers["location"].startswith("https://accounts.google.com/o/oauth2/v2/auth")
        assert "g_oauth_state" in r.headers.get("set-cookie", "")


class TestCallbackCsrf:
    def test_state_mismatch_redirects_to_error(self, app, monkeypatch):
        _configure(monkeypatch)
        c = TestClient(app)
        # No matching state cookie → treated as CSRF, bounced to the error page.
        r = c.get(
            "/api/auth/google/callback?code=abc&state=evil",
            follow_redirects=False,
        )
        assert r.status_code == 302
        assert "error=google" in r.headers["location"]


class TestFinalizeConsent:
    def _pending(self, email="newgoogle@example.com"):
        return google_auth._signer().dumps({"email": email, "verified": True})

    def test_requires_configured(self, app):
        # Not configured → 404.
        r = TestClient(app).post(
            "/api/auth/google/finalize",
            json={"token": "x", "username": "guser", "age_confirmed": True,
                  "accept_terms": True, "accept_privacy": True},
        )
        assert r.status_code == 404

    def test_consent_required(self, app, monkeypatch):
        _configure(monkeypatch)
        tok = self._pending()
        r = TestClient(app).post(
            "/api/auth/google/finalize",
            json={"token": tok, "username": "guser", "age_confirmed": False,
                  "accept_terms": True, "accept_privacy": True},
        )
        assert r.status_code == 422

    def test_invalid_token_400(self, app, monkeypatch):
        _configure(monkeypatch)
        r = TestClient(app).post(
            "/api/auth/google/finalize",
            json={"token": "not-a-valid-token", "username": "guser",
                  "age_confirmed": True, "accept_terms": True, "accept_privacy": True},
        )
        assert r.status_code == 400

    def test_finalize_creates_verified_account_and_logs_in(self, app, account_store, monkeypatch):
        _configure(monkeypatch)
        tok = self._pending("newgoogle@example.com")
        c = TestClient(app)
        r = c.post(
            "/api/auth/google/finalize",
            json={"token": tok, "username": "guser", "age_confirmed": True,
                  "accept_terms": True, "accept_privacy": True},
        )
        assert r.status_code == 200, r.text
        acct_id = r.json()["id"]
        # Account exists, email pre-verified by Google, and the session works.
        stored = account_store.get_account(acct_id)
        assert stored["email"] == "newgoogle@example.com"
        assert stored["email_verified"] is True
        assert c.get("/api/auth/me").status_code == 200
