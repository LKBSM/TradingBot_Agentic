"""Tests for the INFRA-2B.6 HMAC webhook signer."""

from __future__ import annotations

import hmac
import time

import pytest

from src.delivery.webhook_signer import (
    DEFAULT_TOLERANCE_SECONDS,
    SIGNATURE_HEADER,
    SIGNATURE_VERSION,
    generate_webhook_secret,
    parse_signature_header,
    sign_payload,
    verify_payload,
)


# ---------------------------------------------------------------------------
# Sign
# ---------------------------------------------------------------------------


def test_sign_returns_versioned_header_value():
    out = sign_payload('{"a":1}', "shhh", timestamp=1_700_000_000)
    assert out.header_value.startswith("t=1700000000,v1=")
    assert SIGNATURE_VERSION in out.header_value
    assert len(out.signature_hex) == 64  # SHA-256 hex


def test_sign_produces_deterministic_signature():
    a = sign_payload('{"a":1}', "shhh", timestamp=1)
    b = sign_payload('{"a":1}', "shhh", timestamp=1)
    assert a.signature_hex == b.signature_hex


def test_sign_changes_with_secret():
    a = sign_payload('{"a":1}', "secret-A", timestamp=1)
    b = sign_payload('{"a":1}', "secret-B", timestamp=1)
    assert a.signature_hex != b.signature_hex


def test_sign_changes_with_body():
    a = sign_payload('{"a":1}', "shhh", timestamp=1)
    b = sign_payload('{"a":2}', "shhh", timestamp=1)
    assert a.signature_hex != b.signature_hex


def test_sign_changes_with_timestamp():
    a = sign_payload('{"a":1}', "shhh", timestamp=1)
    b = sign_payload('{"a":1}', "shhh", timestamp=2)
    assert a.signature_hex != b.signature_hex


# ---------------------------------------------------------------------------
# Header parsing
# ---------------------------------------------------------------------------


def test_parse_valid_header():
    ts, sig = parse_signature_header("t=42,v1=" + "a" * 64)
    assert ts == 42
    assert sig == "a" * 64


def test_parse_handles_extra_whitespace():
    ts, sig = parse_signature_header("  t=42 , v1=abcd  ")
    assert ts == 42
    assert sig == "abcd"


def test_parse_invalid_timestamp_returns_none():
    ts, sig = parse_signature_header("t=not_a_number,v1=abcd")
    assert ts is None


def test_parse_missing_signature_returns_none():
    ts, sig = parse_signature_header("t=42")
    assert ts == 42
    assert sig is None


def test_parse_garbage_returns_none_pair():
    ts, sig = parse_signature_header("not a valid header")
    assert ts is None and sig is None


# ---------------------------------------------------------------------------
# Verify — happy path
# ---------------------------------------------------------------------------


def test_verify_round_trip_succeeds():
    body = '{"event":"insight.delivered","id":"abc"}'
    secret = "topsecret"
    signed = sign_payload(body, secret)
    res = verify_payload(body, signed.header_value, secret)
    assert res.ok
    assert not res.reason


# ---------------------------------------------------------------------------
# Verify — replay protection
# ---------------------------------------------------------------------------


def test_verify_rejects_old_timestamp():
    body = "{}"
    secret = "x"
    old_ts = int(time.time()) - DEFAULT_TOLERANCE_SECONDS - 60
    signed = sign_payload(body, secret, timestamp=old_ts)
    res = verify_payload(body, signed.header_value, secret)
    assert not res.ok
    assert "tolerance" in res.reason


def test_verify_rejects_future_timestamp():
    body = "{}"
    secret = "x"
    future = int(time.time()) + DEFAULT_TOLERANCE_SECONDS + 60
    signed = sign_payload(body, secret, timestamp=future)
    res = verify_payload(body, signed.header_value, secret)
    assert not res.ok
    assert "tolerance" in res.reason


def test_verify_custom_tolerance_window():
    body = "{}"
    secret = "x"
    signed = sign_payload(body, secret, timestamp=100)
    # now=200 ⇒ delta=100s, tolerance=200 ⇒ ok
    assert verify_payload(body, signed.header_value, secret,
                          tolerance_seconds=200, now=200).ok
    # now=400 ⇒ delta=300s, tolerance=200 ⇒ rejected
    assert not verify_payload(body, signed.header_value, secret,
                              tolerance_seconds=200, now=400).ok


# ---------------------------------------------------------------------------
# Verify — failure modes
# ---------------------------------------------------------------------------


def test_verify_rejects_tampered_body():
    body = "original"
    secret = "x"
    signed = sign_payload(body, secret)
    res = verify_payload("tampered", signed.header_value, secret, now=signed.timestamp)
    assert not res.ok
    assert "mismatch" in res.reason


def test_verify_rejects_wrong_secret():
    body = "{}"
    signed = sign_payload(body, "right-secret")
    res = verify_payload(body, signed.header_value, "wrong-secret",
                         now=signed.timestamp)
    assert not res.ok


def test_verify_rejects_malformed_header():
    res = verify_payload("{}", "garbage-header", "x")
    assert not res.ok
    assert "header" in res.reason.lower()


def test_verify_uses_constant_time_compare(monkeypatch):
    """Sanity: we route through hmac.compare_digest, not == — guards
    against timing-oracle leakage of the secret."""
    calls = {"n": 0}
    real = hmac.compare_digest

    def spy(a, b):
        calls["n"] += 1
        return real(a, b)

    monkeypatch.setattr("src.delivery.webhook_signer.hmac.compare_digest", spy)
    body = "{}"
    secret = "x"
    signed = sign_payload(body, secret)
    verify_payload(body, signed.header_value, secret, now=signed.timestamp)
    assert calls["n"] >= 1


# ---------------------------------------------------------------------------
# Secret generation
# ---------------------------------------------------------------------------


def test_generate_secret_high_entropy():
    a = generate_webhook_secret()
    b = generate_webhook_secret()
    assert a != b
    # 32 bytes ⇒ at least ~43 char URL-safe base64 string
    assert len(a) >= 40


def test_generate_secret_rejects_weak_size():
    with pytest.raises(ValueError):
        generate_webhook_secret(n_bytes=8)


# ---------------------------------------------------------------------------
# Header naming
# ---------------------------------------------------------------------------


def test_signature_header_constant_is_kebab_pascal_case():
    """We commit to ``X-Sentinel-Signature`` as the public header name —
    pin the constant so downstream subscribers don't have it move."""
    assert SIGNATURE_HEADER == "X-Sentinel-Signature"
