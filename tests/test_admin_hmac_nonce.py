"""DG-055 — admin HMAC replay protection (nonce-based).

We avoid pytest-asyncio (extra dep): each async-bodied test is wrapped
with ``asyncio.run`` via the ``run_async`` helper so the suite stays
under the same minimal dep set as the rest of CI.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import time
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request, HTTPException

from src.api.auth import require_admin
from src.api.nonce_store import NonceStore


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def _build_request(*, path: str, hmac_manager: Any, nonce_store: Any) -> Request:
    app = FastAPI()
    app.state.app_state = type("S", (), {})()
    app.state.app_state.hmac_manager = hmac_manager
    app.state.app_state.nonce_store = nonce_store

    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "raw_path": path.encode(),
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 0),
        "app": app,
    }
    return Request(scope)


# ---------------------------------------------------------------------------
# NonceStore unit tests
# ---------------------------------------------------------------------------

def test_nonce_store_first_use_returns_true():
    s = NonceStore(ttl_seconds=60.0)
    assert s.check_and_record("abc") is True


def test_nonce_store_replay_returns_false():
    s = NonceStore(ttl_seconds=60.0)
    assert s.check_and_record("abc") is True
    assert s.check_and_record("abc") is False


def test_nonce_store_expired_entry_can_be_reused():
    s = NonceStore(ttl_seconds=60.0)
    base = 1_000_000.0
    s.check_and_record("abc", now=base)
    # After TTL window, the nonce sweeps out and a fresh insert succeeds
    assert s.check_and_record("abc", now=base + 61.0) is True


def test_nonce_store_max_entries_evicts_oldest():
    s = NonceStore(ttl_seconds=600.0, max_entries=3)
    base = 1_000.0
    for i, name in enumerate(("a", "b", "c")):
        s.check_and_record(name, now=base + i)
    assert len(s) == 3
    # Adding a 4th evicts the oldest ("a")
    s.check_and_record("d", now=base + 10)
    assert len(s) == 3


def test_nonce_store_empty_string_rejected():
    s = NonceStore()
    assert s.check_and_record("") is False


# ---------------------------------------------------------------------------
# require_admin with nonce-protected HMAC
# ---------------------------------------------------------------------------

@pytest.fixture
def hmac_secret():
    return b"unit-test-secret-key-for-admin-hmac"


@pytest.fixture
def hmac_manager(hmac_secret):
    """Minimal HMACManager stub matching the .verify() contract."""
    mgr = MagicMock()
    def _verify(data: bytes, sig: str) -> bool:
        expected = hmac.new(hmac_secret, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)
    mgr.verify.side_effect = _verify
    return mgr


@pytest.fixture
def nonce_store():
    store = NonceStore(ttl_seconds=300.0)
    yield store
    store.reset()


@pytest.fixture(autouse=True)
def _enable_nonce(monkeypatch):
    monkeypatch.setenv("ADMIN_NONCE_REQUIRED", "on")
    yield


def _sign(secret: bytes, canonical: str) -> str:
    return hmac.new(secret, canonical.encode(), hashlib.sha256).hexdigest()


def test_require_admin_accepts_fresh_signed_request(
    hmac_secret, hmac_manager, nonce_store,
):
    ts = str(time.time())
    nonce = "n-001"
    path = "/admin/rotate-key"
    canonical = f"{ts}:{nonce}:{path}"
    sig = _sign(hmac_secret, canonical)

    req = _build_request(path=path, hmac_manager=hmac_manager, nonce_store=nonce_store)

    async def scenario():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=nonce,
        )

    ok = run_async(scenario())
    assert ok is True


def test_require_admin_rejects_replay_within_window(
    hmac_secret, hmac_manager, nonce_store,
):
    ts = str(time.time())
    nonce = "n-002"
    path = "/admin/rotate-key"
    canonical = f"{ts}:{nonce}:{path}"
    sig = _sign(hmac_secret, canonical)
    req = _build_request(path=path, hmac_manager=hmac_manager, nonce_store=nonce_store)

    async def first_call():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=nonce,
        )

    async def replay_call():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=nonce,
        )

    # First call OK
    run_async(first_call())
    # Replay → 401 replay detected
    with pytest.raises(HTTPException) as ei:
        run_async(replay_call())
    assert ei.value.status_code == 401
    assert "Nonce already used" in ei.value.detail


def test_require_admin_rejects_cross_route_replay(
    hmac_secret, hmac_manager, nonce_store,
):
    """A signature minted for /admin/rotate-key must not work on /admin/revoke."""
    ts = str(time.time())
    nonce = "n-003"
    # Client signs with path=/admin/rotate-key
    canonical = f"{ts}:{nonce}:/admin/rotate-key"
    sig = _sign(hmac_secret, canonical)
    # But sends to /admin/revoke
    req = _build_request(
        path="/admin/revoke", hmac_manager=hmac_manager, nonce_store=nonce_store,
    )

    async def scenario():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=nonce,
        )

    with pytest.raises(HTTPException) as ei:
        run_async(scenario())
    assert ei.value.status_code == 401
    assert "Invalid admin signature" in ei.value.detail


def test_require_admin_rejects_missing_nonce_in_strict_mode(
    hmac_secret, hmac_manager, nonce_store, monkeypatch,
):
    monkeypatch.setenv("ADMIN_NONCE_REQUIRED", "on")
    ts = str(time.time())
    sig = _sign(hmac_secret, ts)  # legacy canonical
    req = _build_request(
        path="/admin/test", hmac_manager=hmac_manager, nonce_store=nonce_store,
    )

    async def scenario():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=None,
        )

    with pytest.raises(HTTPException) as ei:
        run_async(scenario())
    assert ei.value.status_code == 401
    assert "X-Admin-Nonce" in ei.value.detail


def test_require_admin_legacy_mode_off_allows_no_nonce(
    hmac_secret, hmac_manager, nonce_store, monkeypatch,
):
    monkeypatch.setenv("ADMIN_NONCE_REQUIRED", "off")
    ts = str(time.time())
    # Legacy path → sign just the timestamp
    sig = _sign(hmac_secret, ts)
    req = _build_request(
        path="/admin/legacy", hmac_manager=hmac_manager, nonce_store=nonce_store,
    )

    async def scenario():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=None,
        )

    ok = run_async(scenario())
    assert ok is True


def test_require_admin_rejects_stale_timestamp(
    hmac_secret, hmac_manager, nonce_store,
):
    # ts > 5 min old
    ts = str(time.time() - 600)
    nonce = "n-stale"
    canonical = f"{ts}:{nonce}:/admin/test"
    sig = _sign(hmac_secret, canonical)
    req = _build_request(
        path="/admin/test", hmac_manager=hmac_manager, nonce_store=nonce_store,
    )

    async def scenario():
        return await require_admin(
            req, x_admin_signature=sig, x_admin_timestamp=ts, x_admin_nonce=nonce,
        )

    with pytest.raises(HTTPException) as ei:
        run_async(scenario())
    assert ei.value.status_code == 401
    assert "Timestamp expired" in ei.value.detail
