"""PR1 — session/cookie foundations hardening (audit AUTH-01/11/12/13/19).

Read-only-audit fixes:

* AUTH-01 — logout clears the cookie with the SAME attributes it was set with,
  so the browser actually deletes it.
* AUTH-12 — the cookie's browser Max-Age mirrors the DB session TTL.
* AUTH-13 — a production boot without SESSION_SECRET fails fast rather than
  silently signing cookies with a per-process ephemeral secret.
* AUTH-19 — expired session rows are purged at startup so the table can't grow
  without bound.
"""

from __future__ import annotations

import pytest
from fastapi import Response
from fastapi.testclient import TestClient

from src.api import session_auth
from src.api.account_store import AccountStore
from src.api.app import create_app


@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "sessions.db"))


@pytest.fixture()
def client(account_store, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    return TestClient(create_app(account_store=account_store))


VALID_REGISTER = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "longpassword1",
    "age_confirmed": True,
    "accept_terms": True,
    "accept_privacy": True,
}


# --------------------------------------------------------------------------- #
# AUTH-01 — clear matches set
# --------------------------------------------------------------------------- #
class TestClearCookieMatchesSet:
    def test_logout_delete_carries_httponly_samesite_path(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        r = client.post("/api/auth/logout")
        set_cookie = r.headers.get("set-cookie", "").lower()
        assert "mia_session=" in set_cookie
        # These were missing from the delete before the fix, so some browsers
        # ignored it and left a stale session cookie behind.
        assert "httponly" in set_cookie
        assert "samesite=lax" in set_cookie
        assert "path=/" in set_cookie

    def test_me_401_after_logout(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        assert client.get("/api/auth/me").status_code == 200
        client.post("/api/auth/logout")
        assert client.get("/api/auth/me").status_code == 401


# --------------------------------------------------------------------------- #
# AUTH-12 — cookie Max-Age follows the session TTL
# --------------------------------------------------------------------------- #
class TestCookieTtl:
    def test_set_session_cookie_honours_ttl_seconds(self, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "s")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
        resp = Response()
        session_auth.set_session_cookie(resp, "tok", ttl_seconds=1234)
        assert "Max-Age=1234" in resp.headers.get("set-cookie", "")

    def test_default_ttl_is_the_signature_window(self, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "s")
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
        resp = Response()
        session_auth.set_session_cookie(resp, "tok")
        assert f"Max-Age={session_auth._SIGNATURE_MAX_AGE_S}" in resp.headers.get(
            "set-cookie", ""
        )


# --------------------------------------------------------------------------- #
# AUTH-13 — production must have a stable secret
# --------------------------------------------------------------------------- #
class TestStableSecretGuard:
    def test_raises_in_production_without_secret(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "production")
        monkeypatch.delenv("SESSION_SECRET", raising=False)
        with pytest.raises(RuntimeError):
            session_auth.assert_stable_secret_configured()

    def test_noop_when_not_production(self, monkeypatch):
        monkeypatch.delenv("ENVIRONMENT", raising=False)
        monkeypatch.delenv("SESSION_SECRET", raising=False)
        session_auth.assert_stable_secret_configured()  # must not raise

    def test_ok_in_production_with_secret(self, monkeypatch):
        monkeypatch.setenv("ENVIRONMENT", "prod")
        monkeypatch.setenv("SESSION_SECRET", "stable-secret")
        session_auth.assert_stable_secret_configured()  # must not raise


# --------------------------------------------------------------------------- #
# AUTH-19 — startup purges expired sessions
# --------------------------------------------------------------------------- #
class TestStartupPurge:
    def test_boot_removes_expired_session_rows(self, account_store, monkeypatch):
        monkeypatch.setenv("SESSION_SECRET", "s")
        acc = account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        # An already-expired session row (ttl in the past).
        account_store.create_session(acc["id"], ttl_seconds=-10)

        app = create_app(account_store=account_store)
        # Entering the context manager runs the lifespan startup → boot purge.
        with TestClient(app):
            pass
        # Already swept, so a second purge finds nothing left to remove.
        assert account_store.purge_expired_sessions() == 0
