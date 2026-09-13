"""SMTP-1 — tests for transactional email delivery and its boot guard.

The defect being fixed was not "an email is late". ``EMAIL_VERIFICATION_ENFORCED``
defaults to on, ``access.py`` denies ``has_access`` to unverified accounts, and
the code that lifts that wall arrives by email — which was never configured in
production. Every new customer was locked out for good, and the owner account
(seeded verified, exempt) hid it.

So the tests that matter here are the ones about the *contradiction*: wall up,
no way to send.
"""

from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional

import pytest

from src.api.mailer import (
    EmailDeliveryNotConfigured,
    PURPOSES,
    assert_email_delivery_configured,
    email_verification_enforced,
    health_snapshot,
    reset_state,
    send_email,
    sender_address,
    smtp_configured,
)

PROD = {"ENVIRONMENT": "production"}
SMTP = {
    "SMTP_HOST": "smtp-relay.example.net",
    "SMTP_USER": "user",
    "SMTP_PASSWORD": "secret",
    "SMTP_FROM": "no-reply@mia.markets",
}


@pytest.fixture(autouse=True)
def _clean_counters():
    reset_state()
    yield
    reset_state()


# --------------------------------------------------------------------------- #
# The boot guard — the heart of this fix
# --------------------------------------------------------------------------- #

def test_production_refuses_to_boot_with_the_wall_up_and_no_way_to_send():
    """Wall up + no SMTP = every new customer locked out. Do not deploy that."""
    with pytest.raises(EmailDeliveryNotConfigured) as exc:
        assert_email_delivery_configured({**PROD, "EMAIL_VERIFICATION_ENFORCED": "1"})

    message = str(exc.value)
    # The error has to be actionable by someone who did not write this code.
    assert "SMTP_HOST" in message
    assert "EMAIL_VERIFICATION_ENFORCED" in message
    assert "locked out" in message
    assert "docs/ops/envoi-courriels.md" in message


def test_the_wall_is_up_by_default_which_is_why_this_bites():
    # No EMAIL_VERIFICATION_ENFORCED at all — exactly production's state before
    # this fix, since render.yaml never declared it.
    with pytest.raises(EmailDeliveryNotConfigured):
        assert_email_delivery_configured(dict(PROD))


def test_production_boots_once_smtp_is_configured():
    assert_email_delivery_configured({**PROD, **SMTP})


def test_lowering_the_wall_on_purpose_lets_the_deploy_through_but_shouts(caplog):
    """A deliberate EMAIL_VERIFICATION_ENFORCED=0 is allowed — new customers can
    still reach the product — but reset and renewal notices are still dead."""
    with caplog.at_level(logging.ERROR):
        assert_email_delivery_configured({**PROD, "EMAIL_VERIFICATION_ENFORCED": "0"})

    assert "NOT configured" in caplog.text
    assert PURPOSES["password_reset"] in caplog.text
    assert PURPOSES["renewal_notice"] in caplog.text


def test_the_guard_never_touches_dev_ci_or_tests():
    # No ENVIRONMENT → local runs and the whole test suite keep working with no
    # SMTP, which is the only reason a fail-fast is tolerable here.
    assert_email_delivery_configured({})
    assert_email_delivery_configured({"ENVIRONMENT": "development"})
    assert_email_delivery_configured({"EMAIL_VERIFICATION_ENFORCED": "1"})


def test_wall_flag_reading_matches_the_subscription_gate(monkeypatch):
    """``mailer`` re-reads EMAIL_VERIFICATION_ENFORCED instead of importing the
    gate (the boot guard must not drag FastAPI deps into startup). Pin the two
    readings together so they cannot drift — drifting is what caused the bug."""
    from src.api.subscription_gate import _email_verification_enforced

    for raw in ("1", "0", "true", "false", "on", "off", "yes", "", "nonsense"):
        monkeypatch.setenv("EMAIL_VERIFICATION_ENFORCED", raw)
        assert email_verification_enforced() == _email_verification_enforced(), raw

    monkeypatch.delenv("EMAIL_VERIFICATION_ENFORCED")
    assert email_verification_enforced() == _email_verification_enforced()


# --------------------------------------------------------------------------- #
# Sending
# --------------------------------------------------------------------------- #

class _FakeSMTP:
    """Stands in for smtplib.SMTP — records what would have gone out."""

    sent: list[EmailMessage] = []
    logins: list[tuple] = []
    started_tls = 0
    fail_with: Optional[Exception] = None

    def __init__(self, host, port, timeout=None):
        self.host, self.port, self.timeout = host, port, timeout
        if _FakeSMTP.fail_with is not None:
            raise _FakeSMTP.fail_with

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def starttls(self):
        _FakeSMTP.started_tls += 1

    def login(self, user, password):
        _FakeSMTP.logins.append((user, password))

    def send_message(self, msg):
        _FakeSMTP.sent.append(msg)


@pytest.fixture()
def fake_smtp(monkeypatch):
    _FakeSMTP.sent, _FakeSMTP.logins = [], []
    _FakeSMTP.started_tls, _FakeSMTP.fail_with = 0, None
    monkeypatch.setattr(smtplib, "SMTP", _FakeSMTP)
    return _FakeSMTP


def test_send_email_delivers_text_and_a_branded_html_alternative(fake_smtp):
    ok = send_email(
        "client@example.com",
        "Votre code",
        "Bonjour,\n\nVotre code est : 123456\n",
        purpose="verification",
        env=SMTP,
    )

    assert ok is True
    msg = fake_smtp.sent[0]
    assert msg["To"] == "client@example.com"
    assert msg["From"] == "no-reply@mia.markets"
    # Plain text is never dropped: a text-only client gets the same message.
    assert "123456" in msg.get_body(("plain",)).get_content()
    assert msg.get_body(("html",)) is not None
    assert fake_smtp.started_tls == 1
    assert fake_smtp.logins == [("user", "secret")]


def test_send_email_returns_false_without_raising_when_unconfigured(fake_smtp):
    # The contract the three call sites already relied on.
    assert send_email("a@b.c", "s", "b", purpose="verification", env={}) is False
    assert fake_smtp.sent == []


def test_an_unconfigured_send_is_logged_at_error_not_info(caplog, fake_smtp):
    """The old code logged this at INFO — indistinguishable from nothing
    happening in a production log stream, while a real person waits."""
    with caplog.at_level(logging.DEBUG):
        send_email("a@b.c", "s", "b", purpose="verification", env={})

    records = [r for r in caplog.records if "NOT DELIVERED" in r.getMessage()]
    assert records, "the failure must be logged"
    assert records[0].levelno == logging.ERROR
    # And it must say what it costs the customer, not just that it failed.
    assert PURPOSES["verification"] in records[0].getMessage()


def test_a_real_delivery_failure_still_raises(fake_smtp):
    """Callers wrap sends in try/except and log; keep that behaviour intact."""
    fake_smtp.fail_with = smtplib.SMTPAuthenticationError(535, b"bad credentials")

    with pytest.raises(smtplib.SMTPAuthenticationError):
        send_email("a@b.c", "s", "b", purpose="verification", env=SMTP)

    assert health_snapshot(SMTP)["failed"] == 1


def test_no_credential_is_ever_logged(caplog, fake_smtp):
    with caplog.at_level(logging.DEBUG):
        send_email("a@b.c", "s", "b", purpose="verification", env=SMTP)

    assert "secret" not in caplog.text


def test_sender_falls_back_through_from_then_user_then_default():
    assert sender_address({"SMTP_FROM": "a@x.io", "SMTP_USER": "b@x.io"}) == "a@x.io"
    assert sender_address({"SMTP_USER": "b@x.io"}) == "b@x.io"
    assert sender_address({}) == "no-reply@mia.markets"


def test_smtp_host_is_the_switch():
    assert smtp_configured({"SMTP_HOST": "h"}) is True
    assert smtp_configured({"SMTP_HOST": "   "}) is False
    assert smtp_configured({"SMTP_USER": "u"}) is False


# --------------------------------------------------------------------------- #
# /health
# --------------------------------------------------------------------------- #

def test_health_is_unhealthy_when_the_wall_stands_with_no_key():
    snap = health_snapshot({"EMAIL_VERIFICATION_ENFORCED": "1"})

    assert snap["configured"] is False
    assert snap["verification_wall"] is True
    assert snap["healthy"] is False


def test_health_is_fine_with_the_wall_down_or_smtp_present():
    assert health_snapshot({"EMAIL_VERIFICATION_ENFORCED": "0"})["healthy"] is True
    assert health_snapshot(SMTP)["healthy"] is True


def test_health_counts_what_happened(fake_smtp):
    send_email("a@b.c", "s", "b", purpose="verification", env=SMTP)
    send_email("a@b.c", "s", "b", purpose="password_reset", env={})

    snap = health_snapshot(SMTP)
    assert snap["sent"] == 1
    assert snap["undeliverable"] == 1


def test_health_endpoint_degrades_the_service_and_carries_the_block(monkeypatch):
    """From the outside the API looks perfectly fine while nobody new can get
    in. That is precisely what this field exists to make visible."""
    from fastapi.testclient import TestClient

    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.setenv("EMAIL_VERIFICATION_ENFORCED", "1")
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "1")
    for flag in ("BOOTSTRAP_ENABLED", "SCHEDULER_ENABLED", "CHATBOT_ENABLED"):
        monkeypatch.setenv(flag, "false")

    from src.api.app import create_app

    with TestClient(create_app()) as client:
        body = client.get("/health").json()

    assert body["email"]["healthy"] is False
    assert body["email"]["verification_wall"] is True
    assert body["status"] == "degraded"


# --------------------------------------------------------------------------- #
# The three call sites now share one sender
# --------------------------------------------------------------------------- #

def test_the_three_transactional_mails_go_through_the_one_mailer():
    """Three private copies of smtplib is how the production gap went unseen:
    a fix to one never reached the others. Keep them converged."""
    import inspect

    from src.api.routes import accounts
    from src.billing import renewal_notices

    for module in (accounts, renewal_notices):
        source = inspect.getsource(module)
        assert "import smtplib" not in source, (
            f"{module.__name__} builds its own SMTP session again — "
            "transactional mail must go through src/api/mailer.py"
        )


def test_verification_email_reports_the_stuck_account_at_error(caplog, fake_smtp):
    from src.api.routes.accounts import _send_verification_email

    with caplog.at_level(logging.ERROR):
        delivered = _send_verification_email("a@b.c", "https://x/verify", "123456")

    assert delivered is False
    assert any(r.levelno == logging.ERROR for r in caplog.records)


def test_renewal_notices_shout_only_when_someone_is_actually_due(caplog, monkeypatch):
    """Nobody due → nothing is going wrong today. Someone due and no way to
    write to them → a customer is charged for a year with no advance warning."""
    monkeypatch.setenv("STRIPE_PRICE_ANNUAL", "price_annual")
    monkeypatch.delenv("SMTP_HOST", raising=False)
    from src.billing.renewal_notices import send_due_renewal_notices

    class _Store:
        def __init__(self, due):
            self._due = due

        def renewals_due(self, *a, **k):
            return self._due

        def record_renewal_notice(self, *a, **k):  # pragma: no cover - never reached
            raise AssertionError("nothing may be claimed as sent without SMTP")

    with caplog.at_level(logging.DEBUG):
        assert send_due_renewal_notices(_Store([])) == 0
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]

    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        assert send_due_renewal_notices(
            _Store([{"account_id": 1, "email": "a@b.c", "period_end": 0.0}])
        ) == 0
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an owed notice that cannot be sent is an ERROR"
    assert "without warning" in errors[0].getMessage()
