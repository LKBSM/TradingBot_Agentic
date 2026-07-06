"""Security tests for the private-beta lockdown wall (BetaAuthMiddleware).

Proves the mission-critical guarantees:

* a data / AI / scanner / view-control endpoint called WITHOUT a session → 401
  (zero data), when ``BETA_LOCKDOWN=1``;
* a seeded tester can log in and then reach the same endpoints (past the wall);
* public endpoints (login, access probe, legal, health) stay reachable anon;
* public self-registration is disabled under lockdown (403);
* with the flag OFF the middleware is a pure no-op (existing behavior intact);
* the seed script is idempotent and never stores a plaintext password.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app

# Endpoints that must NEVER answer an anonymous caller under lockdown. Covers
# both auth systems: session routes (no-op gated) AND legacy api-key routes
# (bypassed by TESTING_MODE) — the middleware fronts both.
PROTECTED_ENDPOINTS = [
    "/api/market-reading",
    "/api/candles",
    "/api/conditions-scan",
    "/api/dashboard/overview",
    "/api/v1/signals/state",      # view-control / live state
    "/api/v1/signals/current",
]

# Reachable without a session even under lockdown.
PUBLIC_ENDPOINTS = [
    "/health",
    "/api/access/me",
]


@pytest.fixture()
def store(tmp_path: Path) -> AccountStore:
    return AccountStore(db_path=str(tmp_path / "accounts.db"))


@pytest.fixture()
def tester(store: AccountStore):
    """A seeded, active tester account + its plaintext password (for login)."""
    password = "TesterStrongPass123!"
    store.create_account(
        "betatester",
        "tester@example.com",
        password,
        age_confirmed=True,
        consents=[("terms", "v1"), ("privacy", "v1")],
        role="tester",
    )
    return {"identifier": "tester@example.com", "password": password}


@pytest.fixture()
def lockdown_env(monkeypatch):
    monkeypatch.setenv("BETA_LOCKDOWN", "1")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    # Keep the legacy api-key auth bypassed (mirrors the beta deployment): the
    # lockdown middleware is the real wall in front of it.
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "1")
    yield


def test_anonymous_is_blocked_on_every_protected_endpoint(store, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)
    for path in PROTECTED_ENDPOINTS:
        resp = client.get(path)
        assert resp.status_code == 401, f"{path} should be 401 for anon, got {resp.status_code}"
        body = resp.json()
        assert body.get("error") == "authentication_required"


def test_public_endpoints_reachable_anonymously(store, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)
    for path in PUBLIC_ENDPOINTS:
        resp = client.get(path)
        assert resp.status_code == 200, f"{path} should be public, got {resp.status_code}"


def test_access_me_reports_must_login_when_anonymous(store, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)
    body = client.get("/api/access/me").json()
    assert body["beta_lockdown"] is True
    assert body["authenticated"] is False
    assert body["must_login"] is True


def test_tester_can_login_then_reach_the_product(store, tester, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)

    # Before login: blocked.
    assert client.get("/api/market-reading").status_code == 401

    # Login sets the session cookie on the client's cookie jar.
    resp = client.post("/api/auth/login", json=tester)
    assert resp.status_code == 200, resp.text

    # After login: past the wall. 401 must NOT happen; the route may 4xx/5xx for
    # its OWN reasons (missing query params / unbuilt engine) but never 401.
    after = client.get("/api/market-reading")
    assert after.status_code != 401, after.text
    assert client.get("/api/access/me").json()["authenticated"] is True


def test_registration_is_closed_under_lockdown(store, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)
    resp = client.post(
        "/api/auth/register",
        json={
            "username": "newcomer",
            "email": "newcomer@example.com",
            "password": "SomeStrongPass123!",
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert resp.status_code == 403
    # And no account was created.
    assert store.get_account_by_identifier("newcomer@example.com") is None


def test_invalid_or_expired_session_is_rejected(store, tester, lockdown_env):
    app = create_app(account_store=store)
    client = TestClient(app)
    # A junk cookie must not pass the signature/lookup.
    client.cookies.set("mia_session", "not-a-valid-signed-token")
    resp = client.get("/api/candles")
    assert resp.status_code == 401


def test_role_restriction_when_configured(store, tester, monkeypatch):
    """BETA_ALLOWED_ROLES can refuse a valid but non-listed role."""
    monkeypatch.setenv("BETA_LOCKDOWN", "1")
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("BETA_ALLOWED_ROLES", "owner")  # testers excluded
    app = create_app(account_store=store)
    client = TestClient(app)
    login = client.post("/api/auth/login", json=tester)
    assert login.status_code == 200
    # Valid session, but role 'tester' is not in the allow-list → 403.
    resp = client.get("/api/candles")
    assert resp.status_code == 403


def test_middleware_is_noop_when_flag_off(store, monkeypatch):
    """With BETA_LOCKDOWN unset, anonymous callers are NOT 401'd by the wall."""
    monkeypatch.delenv("BETA_LOCKDOWN", raising=False)
    app = create_app(account_store=store)
    client = TestClient(app)
    # Not 401 from the wall (the route answers for its own reasons instead).
    assert client.get("/api/candles").status_code != 401


# --------------------------------------------------------------------------- #
# Seed script guarantees
# --------------------------------------------------------------------------- #

def test_seed_testers_idempotent_and_hashed(tmp_path):
    from scripts.seed_testers import seed_testers

    store = AccountStore(db_path=str(tmp_path / "accounts.db"))

    first = seed_testers(store, count=10)
    created = [r for r in first if r["status"] == "created"]
    assert len(created) == 10
    assert all(r["password"] for r in created)
    assert all(r["role"] == "tester" for r in created)

    # Second run: no duplicates, no passwords re-exposed.
    second = seed_testers(store, count=10)
    assert all(r["status"] == "exists" for r in second)
    assert all(r["password"] is None for r in second)

    # Exactly 10 accounts, all 'tester'.
    import sqlite3

    conn = sqlite3.connect(str(tmp_path / "accounts.db"))
    try:
        total = conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0]
        roles = {r[0] for r in conn.execute("SELECT DISTINCT role FROM accounts")}
        # No plaintext password is ever stored — the hash must be argon2id and
        # must not equal any distributed plaintext.
        sample_pw = created[0]["password"]
        leak = conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE password_hash = ?", (sample_pw,)
        ).fetchone()[0]
        hashes = [r[0] for r in conn.execute("SELECT password_hash FROM accounts")]
    finally:
        conn.close()

    assert total == 10
    assert roles == {"tester"}
    assert leak == 0
    assert all(h.startswith("$argon2id$") for h in hashes)

    # The distributed password actually logs the tester in.
    assert store.verify_credentials(created[0]["email"], sample_pw) is not None
