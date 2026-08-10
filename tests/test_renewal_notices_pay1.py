"""PAY-1 / Loi 25 — annual renewal-notice job (30 days before renewal)."""

from __future__ import annotations

import time

import pytest

from src.api.account_store import AccountStore
from src.billing import renewal_notices as rn

ANNUAL = "price_annual_test"
MONTHLY = "price_monthly_test"
DAY = 86400.0


@pytest.fixture()
def store(tmp_path):
    return AccountStore(db_path=str(tmp_path / "renew.db"))


def _account(store, username, email):
    return store.create_account(
        username, email, "longpassword1",
        age_confirmed=True, consents=[("terms", "1"), ("privacy", "1")],
    )


def _annual_sub(store, account_id, *, days_to_renewal, price_id=ANNUAL, cancel=False):
    store.upsert_subscription(
        account_id, stripe_customer_id=f"cus_{account_id}",
        stripe_subscription_id=f"sub_{account_id}", status="active",
        price_id=price_id, current_period_end=time.time() + days_to_renewal * DAY,
        cancel_at_period_end=cancel,
    )


class TestRenewalsDueQuery:
    def test_annual_within_30_days_is_due(self, store):
        a = _account(store, "due", "due@example.com")
        _annual_sub(store, a["id"], days_to_renewal=20)
        due = store.renewals_due(ANNUAL, now=time.time(), lead_seconds=30 * DAY)
        assert [r["account_id"] for r in due] == [a["id"]]

    def test_annual_beyond_30_days_not_due(self, store):
        a = _account(store, "far", "far@example.com")
        _annual_sub(store, a["id"], days_to_renewal=40)
        assert store.renewals_due(ANNUAL, now=time.time(), lead_seconds=30 * DAY) == []

    def test_monthly_not_matched_by_annual_price(self, store):
        a = _account(store, "monthly", "monthly@example.com")
        _annual_sub(store, a["id"], days_to_renewal=10, price_id=MONTHLY)
        assert store.renewals_due(ANNUAL, now=time.time(), lead_seconds=30 * DAY) == []

    def test_cancel_at_period_end_excluded(self, store):
        a = _account(store, "canceling", "canceling@example.com")
        _annual_sub(store, a["id"], days_to_renewal=15, cancel=True)
        assert store.renewals_due(ANNUAL, now=time.time(), lead_seconds=30 * DAY) == []

    def test_recorded_notice_excludes_from_due(self, store):
        a = _account(store, "once", "once@example.com")
        _annual_sub(store, a["id"], days_to_renewal=20)
        now = time.time()
        due = store.renewals_due(ANNUAL, now=now, lead_seconds=30 * DAY)
        pe = due[0]["period_end"]
        assert store.record_renewal_notice(a["id"], pe, rn.NOTICE_KIND, now=now) is True
        # Same period is no longer due; a duplicate record is a no-op.
        assert store.renewals_due(ANNUAL, now=now, lead_seconds=30 * DAY) == []
        assert store.record_renewal_notice(a["id"], pe, rn.NOTICE_KIND, now=now) is False


class TestSendJob:
    def test_sends_once_and_is_idempotent(self, store, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", ANNUAL)
        monkeypatch.setenv("SMTP_HOST", "smtp.test")
        sent = []
        monkeypatch.setattr(rn, "_send_email", lambda to, s, b: sent.append(to) or True)

        a = _account(store, "job", "job@example.com")
        _annual_sub(store, a["id"], days_to_renewal=20)

        assert rn.send_due_renewal_notices(store) == 1
        assert sent == ["job@example.com"]
        # A second run sends nothing (already recorded).
        assert rn.send_due_renewal_notices(store) == 0
        assert sent == ["job@example.com"]

    def test_no_op_without_annual_price(self, store, monkeypatch):
        monkeypatch.delenv("STRIPE_PRICE_ANNUAL", raising=False)
        monkeypatch.setenv("SMTP_HOST", "smtp.test")
        assert rn.send_due_renewal_notices(store) == 0

    def test_no_op_without_smtp(self, store, monkeypatch):
        monkeypatch.setenv("STRIPE_PRICE_ANNUAL", ANNUAL)
        monkeypatch.delenv("SMTP_HOST", raising=False)
        assert rn.send_due_renewal_notices(store) == 0
