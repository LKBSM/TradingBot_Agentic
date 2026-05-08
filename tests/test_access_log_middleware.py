"""Tests for the OBS-2B.3 structured JSON access log middleware."""

from __future__ import annotations

import json
import logging
import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.middleware.access_log import (
    REQUEST_ID_HEADER,
    _last4,
    _new_request_id,
    _safe_inbound_request_id,
)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_safe_inbound_request_id_accepts_alnum_dash_underscore():
    assert _safe_inbound_request_id("abc-123_XYZ") == "abc-123_XYZ"


def test_safe_inbound_request_id_rejects_special_chars():
    """No dots, slashes, spaces — those would be a log-injection vector
    if we ever logged the field as a raw label."""
    for bad in ("a b", "a.b", "a/b", "<script>", "a\nb"):
        assert _safe_inbound_request_id(bad) is None


def test_safe_inbound_request_id_rejects_empty_and_long():
    assert _safe_inbound_request_id("") is None
    assert _safe_inbound_request_id(None) is None
    assert _safe_inbound_request_id("a" * 65) is None


def test_new_request_id_is_64bit_hex():
    rid = _new_request_id()
    assert len(rid) == 16
    int(rid, 16)  # parses as hex


def test_last4_short_value():
    assert _last4("ab") == "ab"  # not enough chars — returned as-is
    assert _last4("abcde") == "bcde"


def test_last4_handles_none_and_int():
    assert _last4(None) == ""
    assert _last4(12345) == "2345"


# ---------------------------------------------------------------------------
# End-to-end: middleware emits a JSON log line per request
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_testing_mode():
    with patch("src.api.auth.TESTING_MODE", True):
        yield


@pytest.fixture
def caplog_access(caplog):
    caplog.set_level(logging.INFO, logger="smart_sentinel.access")
    return caplog


def _logged_lines(caplog):
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == "smart_sentinel.access"
    ]


def test_access_log_emits_one_line_per_api_request(caplog_access):
    c = TestClient(create_app())
    resp = c.get("/api/v1/health")
    assert resp.status_code == 200

    lines = _logged_lines(caplog_access)
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["evt"] == "http_access"
    assert entry["method"] == "GET"
    assert entry["path"] == "/api/v1/health"
    assert entry["status"] == 200
    assert entry["latency_ms"] >= 0
    assert len(entry["request_id"]) == 16  # 8 random bytes hex


def test_access_log_skips_non_api_paths(caplog_access):
    """Docker /health, /api/docs, etc. shouldn't pollute the access log."""
    c = TestClient(create_app())
    c.get("/health")  # legacy docker probe — outside /api/v1
    lines = _logged_lines(caplog_access)
    assert lines == []


def test_response_carries_request_id_header(caplog_access):
    c = TestClient(create_app())
    resp = c.get("/api/v1/health")
    assert REQUEST_ID_HEADER in resp.headers
    rid = resp.headers[REQUEST_ID_HEADER]
    # Same id in the log line
    entry = json.loads(_logged_lines(caplog_access)[0])
    assert entry["request_id"] == rid


def test_inbound_safe_request_id_propagates(caplog_access):
    c = TestClient(create_app())
    resp = c.get(
        "/api/v1/health",
        headers={REQUEST_ID_HEADER: "client-supplied-12"},
    )
    assert resp.headers[REQUEST_ID_HEADER] == "client-supplied-12"
    entry = json.loads(_logged_lines(caplog_access)[0])
    assert entry["request_id"] == "client-supplied-12"


def test_inbound_unsafe_request_id_replaced(caplog_access):
    """Strange chars in the inbound header must be discarded — we mint
    a fresh id rather than logging the user-controlled string."""
    c = TestClient(create_app())
    resp = c.get(
        "/api/v1/health",
        headers={REQUEST_ID_HEADER: "x\nNEW LOG INJECTION"},
    )
    rid = resp.headers[REQUEST_ID_HEADER]
    assert rid != "x\nNEW LOG INJECTION"
    assert len(rid) == 16


def test_log_contains_status_for_404(caplog_access):
    c = TestClient(create_app())
    resp = c.get("/api/v1/this-route-does-not-exist")
    assert resp.status_code == 404
    entry = json.loads(_logged_lines(caplog_access)[-1])
    assert entry["status"] == 404


def test_user_agent_truncated(caplog_access):
    c = TestClient(create_app())
    long_ua = "A" * 500
    c.get("/api/v1/health", headers={"User-Agent": long_ua})
    entry = json.loads(_logged_lines(caplog_access)[0])
    assert len(entry["user_agent"]) <= 120


def test_log_includes_client_ip(caplog_access):
    c = TestClient(create_app())
    c.get("/api/v1/health")
    entry = json.loads(_logged_lines(caplog_access)[0])
    # TestClient sets client to ('testclient', port)
    assert entry["client_ip"] == "testclient"


def test_log_tier_is_dash_when_no_subscriber(caplog_access):
    """Routes that don't run through require_api_key (like /health) won't
    have a subscriber; the log shouldn't crash trying to extract one."""
    c = TestClient(create_app())
    c.get("/api/v1/health")
    entry = json.loads(_logged_lines(caplog_access)[0])
    assert entry["tier"] == "-"
    assert entry["api_key_id_last4"] == "-"
