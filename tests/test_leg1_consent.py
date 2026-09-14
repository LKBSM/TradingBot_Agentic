"""LEG-1 — consent recorded before payment.

The consent screen that precedes Stripe Checkout sends "I accept"; the SERVER
decides which version that stamps. These tests pin the three properties the
mission asks for:

  * the accepted version and the timestamp are recorded, attached to the account;
  * nothing else is recorded;
  * the version comes from the single source, never from the client.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from src.api.account_store import AccountError, AccountStore
from src.api.routes.legal import LAST_UPDATED as LEGAL_VERSION

ISO_LOCAL = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")


# =============================================================================
# Store level
# =============================================================================

@pytest.fixture()
def store(tmp_path) -> AccountStore:
    return AccountStore(db_path=str(tmp_path / "accounts.db"))


def _make_account(store: AccountStore, email: str = "a@b.co") -> int:
    account = store.create_account(
        "user1",
        email,
        "correct horse battery",
        age_confirmed=True,
        consents=[("terms", "2026-01-01"), ("privacy", "2026-01-01")],
    )
    return account["id"]


class TestRecordConsents:
    def test_records_version_and_timestamp_for_both_documents(self, store):
        account_id = _make_account(store)
        store.record_consents(
            account_id, [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
        )
        current = [c for c in store.get_consents(account_id) if c["version"] == LEGAL_VERSION]
        assert {c["doc"] for c in current} == {"terms", "privacy"}
        for consent in current:
            assert ISO_LOCAL.match(consent["accepted_at"]), consent["accepted_at"]

    def test_records_nothing_but_doc_version_and_timestamp(self, store):
        account_id = _make_account(store)
        store.record_consents(
            account_id, [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
        )
        for consent in store.get_consents(account_id):
            assert set(consent) == {"doc", "version", "accepted_at"}

    def test_accepting_the_same_version_twice_writes_once(self, store):
        account_id = _make_account(store)
        for _ in range(3):
            store.record_consents(
                account_id, [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
            )
        rows = [c for c in store.get_consents(account_id) if c["version"] == LEGAL_VERSION]
        assert len(rows) == 2

    def test_a_new_version_is_appended_not_overwritten(self, store):
        # The history of what was accepted, and when, must survive a version bump.
        account_id = _make_account(store)
        store.record_consents(account_id, [("terms", "2026-05-05"), ("privacy", "2026-05-05")])
        store.record_consents(account_id, [("terms", "2026-09-13"), ("privacy", "2026-09-13")])
        versions = {c["version"] for c in store.get_consents(account_id)}
        assert {"2026-01-01", "2026-05-05", "2026-09-13"} <= versions

    def test_both_documents_are_required(self, store):
        account_id = _make_account(store)
        with pytest.raises(AccountError) as exc:
            store.record_consents(account_id, [("terms", LEGAL_VERSION)])
        assert exc.value.code == "consent_required"

    def test_unknown_document_is_rejected(self, store):
        account_id = _make_account(store)
        with pytest.raises(AccountError):
            store.record_consents(
                account_id,
                [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION), ("cookies", "1")],
            )

    def test_blank_version_is_rejected(self, store):
        account_id = _make_account(store)
        with pytest.raises(AccountError) as exc:
            store.record_consents(account_id, [("terms", "  "), ("privacy", LEGAL_VERSION)])
        assert exc.value.code == "invalid_consent_version"

    def test_unknown_account_is_rejected(self, store):
        with pytest.raises(AccountError) as exc:
            store.record_consents(
                9999, [("terms", LEGAL_VERSION), ("privacy", LEGAL_VERSION)]
            )
        assert exc.value.code == "account_not_found"


# =============================================================================
# HTTP level
# =============================================================================

@pytest.fixture()
def client(store, monkeypatch) -> TestClient:
    # Plain http TestClient → the Secure cookie flag would stop the session
    # cookie round-tripping (production keeps it on under HTTPS).
    monkeypatch.setenv("SESSION_COOKIE_SECURE", "0")
    monkeypatch.setenv("SESSION_SECRET", "leg1-test-session-secret-value")
    from src.api.app import create_app

    return TestClient(create_app(account_store=store))


def _register(client: TestClient) -> None:
    r = client.post(
        "/api/auth/register",
        json={
            "username": "buyer",
            "email": "buyer@example.com",
            "password": "correct horse battery",
            "age_confirmed": True,
            "accept_terms": True,
            "accept_privacy": True,
        },
    )
    assert r.status_code in (200, 201), r.text


class TestConsentEndpoint:
    def test_requires_authentication(self, client):
        assert client.post("/api/auth/consents", json={"accept": True}).status_code == 401

    def test_accepting_records_the_current_version(self, client):
        _register(client)
        r = client.post("/api/auth/consents", json={"accept": True})
        assert r.status_code == 200
        consents = r.json()["consents"]
        current = [c for c in consents if c["version"] == LEGAL_VERSION]
        assert {c["doc"] for c in current} == {"terms", "privacy"}
        for consent in current:
            assert consent["accepted_at"]

    def test_refusing_is_rejected_and_records_nothing(self, client):
        _register(client)
        before = client.get("/api/auth/me").json()["consents"]
        assert client.post("/api/auth/consents", json={"accept": False}).status_code == 422
        assert client.get("/api/auth/me").json()["consents"] == before

    def test_the_client_cannot_choose_the_version(self, client):
        # A client sending its own version must not be able to stamp a consent
        # against a text the customer never saw.
        _register(client)
        r = client.post(
            "/api/auth/consents", json={"accept": True, "version": "1999-01-01"}
        )
        assert r.status_code == 200
        assert all(c["version"] != "1999-01-01" for c in r.json()["consents"])

    def test_version_matches_the_served_document(self, client):
        _register(client)
        recorded = client.post("/api/auth/consents", json={"accept": True}).json()
        served = client.get("/api/v1/legal/conditions?lang=fr")
        assert served.headers["X-Document-Version"] in {
            c["version"] for c in recorded["consents"]
        }
