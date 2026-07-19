"""PR2 — auth reliability & access (audit AUTH-10/14).

* AUTH-14 — changing the email revokes every existing session (account-takeover
  hardening, symmetric with a password change); the PATCH /profile route then
  re-issues a fresh cookie so the actor stays logged in.
* AUTH-10 — owner seeding never silently promotes an organic account that shares
  only the username OR the email, records the owner's implicit consent, and the
  exact-match path stays idempotent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountError, AccountStore
from src.api.app import create_app


@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "accounts.db"))


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


def _mk(store: AccountStore, username="bob", email="bob@example.com"):
    return store.create_account(
        username, email, "longpassword1",
        age_confirmed=True, consents=[("terms", "v1"), ("privacy", "v1")],
    )


# --------------------------------------------------------------------------- #
# AUTH-14
# --------------------------------------------------------------------------- #
class TestEmailChangeRevokesSessions:
    def test_update_email_kills_existing_sessions(self, account_store):
        acc = _mk(account_store)
        tok = account_store.create_session(acc["id"])
        assert account_store.resolve_session(tok) is not None
        account_store.update_email(acc["id"], "bob2@example.com")
        # The other-device session is dead.
        assert account_store.resolve_session(tok) is None

    def test_profile_route_keeps_actor_logged_in(self, client):
        client.post("/api/auth/register", json=VALID_REGISTER)
        r = client.patch("/api/auth/profile", json={"email": "alice2@example.com"})
        assert r.status_code == 200
        assert r.json()["email"] == "alice2@example.com"
        # A fresh cookie was minted → the current device is still authenticated.
        assert client.get("/api/auth/me").status_code == 200


# --------------------------------------------------------------------------- #
# AUTH-10
# --------------------------------------------------------------------------- #
class TestOwnerSeedHardening:
    def test_no_silent_promotion_on_email_collision(self, account_store):
        user = _mk(account_store, "carol", "shared@example.com")
        with pytest.raises(AccountError) as exc:
            account_store.seed_owner("owner", "shared@example.com", "ownerpass123")
        assert exc.value.code == "owner_conflict"
        # The organic account was NOT elevated.
        fresh = account_store.verify_credentials("carol", "longpassword1")
        assert fresh is not None and fresh["role"] == "user"
        assert user["role"] == "user"

    def test_no_silent_promotion_on_username_collision(self, account_store):
        _mk(account_store, "owner", "someone@example.com")
        with pytest.raises(AccountError) as exc:
            account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        assert exc.value.code == "owner_conflict"

    def test_exact_match_is_idempotent(self, account_store):
        o1 = account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        o2 = account_store.seed_owner("owner", "owner@example.com", "ownerpass123")
        assert o1["id"] == o2["id"] and o2["role"] == "owner"

    def test_records_implicit_consent(self, account_store):
        owner = account_store.seed_owner(
            "owner", "owner@example.com", "ownerpass123",
            consents=[("terms", "v9"), ("privacy", "v9")],
        )
        consents = account_store.get_consents(owner["id"])
        docs = {c["doc"] for c in consents}
        assert docs == {"terms", "privacy"}
