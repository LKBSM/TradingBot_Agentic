"""Tests for the account/auth/legal shell.

Covers: registration (18+ + consent version/timestamp), login by username OR
email, logout, password reset, the seeded owner from env getting full access
while a normal account does not, legal document rendering, and the auth
dependencies. No payments are exercised (that is mission ②) — only the gate
seam, which is open today and always passes the owner.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountError, AccountStore
from src.api.app import create_app
from src.api.routes.legal import CONDITIONS_VERSION


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture()
def accounts_db(tmp_path):
    return str(tmp_path / "test_accounts.db")


@pytest.fixture()
def account_store(accounts_db):
    return AccountStore(db_path=accounts_db)


@pytest.fixture()
def client(account_store, monkeypatch):
    # Plain http TestClient → disable the Secure cookie flag so the session
    # cookie round-trips (production keeps it on under HTTPS).
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    app = create_app(account_store=account_store)
    return TestClient(app)


VALID_REGISTER = {
    "username": "alice",
    "email": "alice@example.com",
    "password": "longpassword1",
    "age_confirmed": True,
    "accept_terms": True,
    "accept_privacy": True,
}


# =============================================================================
# AccountStore unit tests
# =============================================================================

class TestAccountStore:
    def test_create_records_consent_version_and_timestamp(self, account_store):
        acc = account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True,
            consents=[("terms", "2026-04-28"), ("privacy", "2026-04-28")],
        )
        consents = account_store.get_consents(acc["id"])
        docs = {c["doc"]: c for c in consents}
        assert set(docs) == {"terms", "privacy"}
        assert docs["terms"]["version"] == "2026-04-28"
        assert docs["terms"]["accepted_at"]  # timestamp present
        assert docs["privacy"]["accepted_at"]

    def test_age_not_confirmed_rejected(self, account_store):
        with pytest.raises(AccountError) as ei:
            account_store.create_account(
                "bob", "bob@example.com", "longpassword1",
                age_confirmed=False,
                consents=[("terms", "v1"), ("privacy", "v1")],
            )
        assert ei.value.code == "age_not_confirmed"

    def test_missing_consent_rejected(self, account_store):
        with pytest.raises(AccountError) as ei:
            account_store.create_account(
                "bob", "bob@example.com", "longpassword1",
                age_confirmed=True,
                consents=[("terms", "v1")],  # privacy missing
            )
        assert ei.value.code == "consent_required"

    def test_weak_password_rejected(self, account_store):
        with pytest.raises(AccountError) as ei:
            account_store.create_account(
                "bob", "bob@example.com", "short",
                age_confirmed=True,
                consents=[("terms", "v1"), ("privacy", "v1")],
            )
        assert ei.value.code == "weak_password"

    def test_duplicate_username_and_email_rejected(self, account_store):
        account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        with pytest.raises(AccountError) as ei:
            account_store.create_account(
                "BOB", "other@example.com", "longpassword1",
                age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
            )
        assert ei.value.code == "username_taken"
        with pytest.raises(AccountError) as ei:
            account_store.create_account(
                "bob2", "BOB@example.com", "longpassword1",
                age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
            )
        assert ei.value.code == "email_taken"

    def test_password_is_hashed_not_plaintext(self, account_store, accounts_db):
        account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        import sqlite3
        conn = sqlite3.connect(accounts_db)
        row = conn.execute("SELECT password_hash FROM accounts").fetchone()
        conn.close()
        assert "longpassword1" not in row[0]
        assert row[0].startswith("$argon2")

    def test_verify_credentials_by_username_and_email(self, account_store):
        account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        assert account_store.verify_credentials("bob", "longpassword1") is not None
        assert account_store.verify_credentials("bob@example.com", "longpassword1") is not None
        assert account_store.verify_credentials("bob", "wrongpass12") is None
        assert account_store.verify_credentials("ghost", "longpassword1") is None

    def test_session_lifecycle(self, account_store):
        acc = account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        tok = account_store.create_session(acc["id"])
        assert account_store.resolve_session(tok)["id"] == acc["id"]
        assert account_store.delete_session(tok) is True
        assert account_store.resolve_session(tok) is None

    def test_password_reset_single_use(self, account_store):
        acc = account_store.create_account(
            "bob", "bob@example.com", "longpassword1",
            age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
        )
        tok = account_store.create_reset_token("bob@example.com")
        assert tok is not None
        assert account_store.consume_reset_token(tok, "brandnewpass1") is True
        # token cannot be reused
        assert account_store.consume_reset_token(tok, "another12345") is False
        # new password works, old does not
        assert account_store.verify_credentials("bob", "brandnewpass1") is not None
        assert account_store.verify_credentials("bob", "longpassword1") is None

    def test_reset_token_unknown_identifier_returns_none(self, account_store):
        assert account_store.create_reset_token("nobody@example.com") is None

    def test_seed_owner_idempotent_and_role(self, account_store):
        o1 = account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        assert o1["role"] == "owner"
        # second boot must not duplicate or fail
        o2 = account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        assert o2["id"] == o1["id"]
        assert o2["role"] == "owner"

    def test_seed_owner_does_not_reset_changed_password(self, account_store):
        account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        # owner changes password out-of-band
        acc = account_store.verify_credentials("owner", "ownerpass123")
        tok = account_store.create_reset_token("owner")
        account_store.consume_reset_token(tok, "newownerpass99")
        # re-seed with the ORIGINAL env password must not clobber the new one
        account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        assert account_store.verify_credentials("owner", "newownerpass99") is not None


# =============================================================================
# HTTP route tests
# =============================================================================

class TestAuthRoutes:
    def test_register_login_logout_flow(self, client):
        r = client.post("/api/auth/register", json=VALID_REGISTER)
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["role"] == "user"
        versions = {c["doc"]: c["version"] for c in body["consents"]}
        assert versions["terms"] and versions["privacy"]

        # session cookie authenticates /me
        assert client.get("/api/auth/me").status_code == 200
        assert client.post("/api/auth/logout").status_code == 200
        assert client.get("/api/auth/me").status_code == 401

    def test_login_by_username_or_email(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        client.post("/api/auth/logout")
        assert client.post("/api/auth/login", json={
            "identifier": "alice", "password": "longpassword1"}).status_code == 200
        assert client.post("/api/auth/login", json={
            "identifier": "alice@example.com", "password": "longpassword1"}).status_code == 200

    def test_login_wrong_password_401(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        r = client.post("/api/auth/login", json={
            "identifier": "alice", "password": "nopenopenope"})
        assert r.status_code == 401

    def test_register_under_18_rejected(self, client):
        payload = {**VALID_REGISTER, "age_confirmed": False}
        assert client.post("/api/auth/register", json=payload).status_code == 422

    def test_register_without_consent_rejected(self, client):
        payload = {**VALID_REGISTER, "accept_terms": False}
        assert client.post("/api/auth/register", json=payload).status_code == 422

    def test_duplicate_register_conflict(self, client):
        assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 201
        client.post("/api/auth/logout")
        assert client.post("/api/auth/register", json=VALID_REGISTER).status_code == 409

    def test_profile_update_email(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        r = client.patch("/api/auth/profile", json={"email": "alice2@example.com"})
        assert r.status_code == 200
        assert r.json()["email"] == "alice2@example.com"

    def test_password_reset_request_is_enumeration_safe(self, client):
        # Same response whether the identifier exists or not.
        r1 = client.post("/api/auth/password-reset/request", json={"identifier": "ghost"})
        client.post("/api/auth/register", json=VALID_REGISTER)
        r2 = client.post("/api/auth/password-reset/request", json={"identifier": "alice"})
        assert r1.status_code == r2.status_code == 200
        assert r1.json()["message"] == r2.json()["message"]


# =============================================================================
# Owner seeded from env → full access; normal account does not
# =============================================================================

class TestOwnerEnvSeedAndAccess:
    def _owner_client(self, accounts_db, monkeypatch):
        monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
        monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
        monkeypatch.setenv("OWNER_USERNAME", "rootowner")
        monkeypatch.setenv("OWNER_EMAIL", "root@example.com")
        monkeypatch.setenv("OWNER_PASSWORD", "ownerpass123")
        store = AccountStore(db_path=accounts_db)
        app = create_app(account_store=store)
        # Triggers the lifespan → _maybe_seed_owner.
        return TestClient(app)

    def test_owner_seeded_from_env_has_full_access(self, accounts_db, monkeypatch):
        with self._owner_client(accounts_db, monkeypatch) as client:
            r = client.post("/api/auth/login", json={
                "identifier": "rootowner", "password": "ownerpass123"})
            assert r.status_code == 200, r.text
            assert r.json()["role"] == "owner"
            # owner-only admin seam
            assert client.get("/api/auth/admin/overview").status_code == 200
            # gate probe: owner has access
            acc = client.get("/api/auth/access").json()
            assert acc["is_owner"] is True and acc["has_access"] is True

    def test_normal_account_denied_owner_surface(self, accounts_db, monkeypatch):
        with self._owner_client(accounts_db, monkeypatch) as client:
            client.post("/api/auth/register", json=VALID_REGISTER)  # role=user
            # owner-only admin surface is forbidden for a normal user
            assert client.get("/api/auth/admin/overview").status_code == 403


# =============================================================================
# Legal document rendering
# =============================================================================

class TestLegalDocuments:
    # La version vient de src/api/routes/legal.py (source unique, lockstep
    # avec l'en-tête du markdown canonique) — pas de littéral figé ici.
    def test_conditions_rendered_with_version(self, client):
        r = client.get("/api/v1/legal/conditions")
        assert r.status_code == 200
        assert r.headers["X-Document-Version"] == CONDITIONS_VERSION
        assert "Conditions Générales d'Utilisation" in r.text

    def test_conditions_doc_header_in_lockstep_with_code(self, client):
        # L'en-tête du markdown canonique doit porter la même date que
        # CONDITIONS_VERSION (horodatage de consentement Loi 25 / RGPD).
        r = client.get("/api/v1/legal/conditions")
        assert f"_Version : {CONDITIONS_VERSION}" in r.text

    def test_conditions_meta(self, client):
        r = client.get("/api/v1/legal/conditions/meta")
        assert r.status_code == 200
        assert r.json()["version"] == CONDITIONS_VERSION

    def test_legal_version_includes_conditions(self, client):
        r = client.get("/api/v1/legal/version")
        assert r.status_code == 200
        assert r.json()["conditions_version"] == CONDITIONS_VERSION
