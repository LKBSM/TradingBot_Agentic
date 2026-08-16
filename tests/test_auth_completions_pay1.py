"""PAY-1 auth completions — email verification, password change, account deletion.

Exercises the new account endpoints against a real AccountStore (tmp DB) and a
fake Stripe client (for cancel-on-delete). No network, no SMTP: the verification
token is minted via the store (the same token the email would carry).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountStore
from src.api.app import create_app

PASSWORD = "longpassword1"


class FakeStripeClient:
    def __init__(self):
        self.is_configured = True
        self.canceled = []
        self.fail_cancel = False

    def create_customer(self, *, email, account_id):
        return {"id": f"cus_{account_id}", "email": email}

    def cancel_subscription(self, subscription_id):
        if self.fail_cancel:
            raise RuntimeError("stripe down")
        self.canceled.append(subscription_id)
        return {"id": subscription_id, "status": "canceled"}


@pytest.fixture()
def account_store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "auth_accounts.db"))


@pytest.fixture()
def stripe():
    return FakeStripeClient()


@pytest.fixture()
def app(account_store, stripe, monkeypatch):
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "test-session-secret-value")
    return create_app(account_store=account_store, stripe_client=stripe)


def _register(app, username="alice", email="alice@example.com"):
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


# =============================================================================
# Email verification
# =============================================================================

class TestEmailVerification:
    def test_new_account_is_unverified(self, app):
        _c, acct = _register(app)
        assert acct["email_verified"] is False

    def test_confirm_marks_verified(self, app, account_store):
        c, acct = _register(app)
        token = account_store.create_email_verification(acct["id"])
        assert token is not None
        r = c.post("/api/auth/verify-email/confirm", json={"token": token})
        assert r.status_code == 200, r.text
        assert c.get("/api/auth/me").json()["email_verified"] is True

    def test_confirm_invalid_token_400(self, app):
        c, _ = _register(app)
        r = c.post("/api/auth/verify-email/confirm", json={"token": "nope"})
        assert r.status_code == 400

    def test_token_is_single_use(self, app, account_store):
        c, acct = _register(app)
        token = account_store.create_email_verification(acct["id"])
        assert c.post("/api/auth/verify-email/confirm", json={"token": token}).status_code == 200
        # Re-use is rejected.
        assert c.post("/api/auth/verify-email/confirm", json={"token": token}).status_code == 400

    def test_resend_requires_auth(self, app):
        anon = TestClient(app)
        assert anon.post("/api/auth/verify-email/resend").status_code == 401

    def test_resend_ok_for_logged_in(self, app):
        c, _ = _register(app)
        r = c.post("/api/auth/verify-email/resend")
        assert r.status_code == 200 and r.json()["ok"] is True


# =============================================================================
# Password change (authenticated)
# =============================================================================

class TestPasswordChange:
    def test_requires_auth(self, app):
        anon = TestClient(app)
        r = anon.post(
            "/api/auth/password",
            json={"current_password": PASSWORD, "new_password": "brandnewpass1"},
        )
        assert r.status_code == 401

    def test_wrong_current_password_400(self, app):
        c, _ = _register(app)
        r = c.post(
            "/api/auth/password",
            json={"current_password": "wrongpassword", "new_password": "brandnewpass1"},
        )
        assert r.status_code == 400

    def test_change_and_relogin(self, app):
        c, acct = _register(app)
        r = c.post(
            "/api/auth/password",
            json={"current_password": PASSWORD, "new_password": "brandnewpass1"},
        )
        assert r.status_code == 200, r.text
        # A separate client can log in with the NEW password, not the old.
        other = TestClient(app)
        assert other.post(
            "/api/auth/login",
            json={"identifier": acct["username"], "password": PASSWORD},
        ).status_code == 401
        assert other.post(
            "/api/auth/login",
            json={"identifier": acct["username"], "password": "brandnewpass1"},
        ).status_code == 200

    def test_other_sessions_revoked(self, app):
        # Two devices logged in; changing the password on one signs the other out.
        c1, acct = _register(app)
        c2 = TestClient(app)
        assert c2.post(
            "/api/auth/login",
            json={"identifier": acct["username"], "password": PASSWORD},
        ).status_code == 200
        assert c2.get("/api/auth/me").status_code == 200
        c1.post(
            "/api/auth/password",
            json={"current_password": PASSWORD, "new_password": "brandnewpass1"},
        )
        # c2's session was revoked.
        assert c2.get("/api/auth/me").status_code == 401
        # c1 stays logged in (fresh session minted).
        assert c1.get("/api/auth/me").status_code == 200


# =============================================================================
# Account deletion (Loi 25 erasure) + cancel-on-delete
# =============================================================================

class TestAccountDeletion:
    def test_requires_auth(self, app):
        assert TestClient(app).delete("/api/auth/account").status_code == 401

    def test_delete_removes_account(self, app, account_store):
        c, acct = _register(app)
        r = c.delete("/api/auth/account")
        assert r.status_code == 200, r.text
        # Account is gone: login fails, the store has no row.
        assert account_store.get_account(acct["id"]) is None
        assert c.get("/api/auth/me").status_code == 401

    def test_delete_cancels_stripe_subscription(self, app, account_store, stripe):
        c, acct = _register(app)
        account_store.upsert_subscription(
            acct["id"], stripe_customer_id="cus_x",
            stripe_subscription_id="sub_del", status="active",
        )
        r = c.delete("/api/auth/account")
        assert r.status_code == 200
        assert "sub_del" in stripe.canceled
        assert account_store.get_account(acct["id"]) is None

    def test_delete_aborts_if_cancel_fails(self, app, account_store, stripe):
        # A failed Stripe cancel must NOT delete the account (else it keeps
        # billing a deleted user).
        stripe.fail_cancel = True
        c, acct = _register(app)
        account_store.upsert_subscription(
            acct["id"], stripe_customer_id="cus_y",
            stripe_subscription_id="sub_keep", status="active",
        )
        r = c.delete("/api/auth/account")
        assert r.status_code == 502
        assert account_store.get_account(acct["id"]) is not None
