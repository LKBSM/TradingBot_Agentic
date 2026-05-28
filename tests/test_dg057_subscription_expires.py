"""DG-057 — enforce subscription_expires on the auth path.

Before this fix, ``UserTierManager.users.subscription_expires`` was
written by Stripe webhook handlers but never consulted by
``require_api_key``, so lapsed subscribers retained paying-tier access
indefinitely. We now treat a past timestamp as a hard 402.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request

from src.api.auth import _subscription_is_expired, require_api_key
from src.api.tier_manager import UserTier, UserTierManager


def run_async(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Pure helper
# ---------------------------------------------------------------------------

def test_subscription_is_expired_past_naive():
    past = (datetime.utcnow() - timedelta(days=1)).isoformat()
    assert _subscription_is_expired(past) is True


def test_subscription_is_expired_future_naive():
    future = (datetime.utcnow() + timedelta(days=30)).isoformat()
    assert _subscription_is_expired(future) is False


def test_subscription_is_expired_past_aware_utc():
    past = (datetime.now(tz=timezone.utc) - timedelta(hours=1)).isoformat()
    assert _subscription_is_expired(past) is True


def test_subscription_is_expired_empty_string():
    assert _subscription_is_expired("") is False
    assert _subscription_is_expired(None) is False  # type: ignore[arg-type]


def test_subscription_is_expired_garbage_treated_active():
    # Unparseable timestamp should NOT lock the user out — operator audit.
    assert _subscription_is_expired("not-a-date") is False


# ---------------------------------------------------------------------------
# UserTierManager.set_subscription_expires
# ---------------------------------------------------------------------------

@pytest.fixture
def tier_manager():
    with tempfile.TemporaryDirectory() as td:
        yield UserTierManager(db_path=str(Path(td) / "users.db"))


def test_set_subscription_expires_writes_then_reads(tier_manager):
    u = tier_manager.create_user(email="a@x.com", api_key_id=42)
    future = (datetime.utcnow() + timedelta(days=30)).isoformat()
    assert tier_manager.set_subscription_expires(u["user_id"], future) is True
    row = tier_manager.get_user(u["user_id"])
    assert row["subscription_expires"] == future


def test_set_subscription_expires_clear(tier_manager):
    u = tier_manager.create_user(email="a@x.com", api_key_id=42)
    tier_manager.set_subscription_expires(u["user_id"], "2099-01-01T00:00:00")
    tier_manager.set_subscription_expires(u["user_id"], None)
    row = tier_manager.get_user(u["user_id"])
    assert row["subscription_expires"] is None


# ---------------------------------------------------------------------------
# require_api_key — 402 on expired subscription
# ---------------------------------------------------------------------------

def _build_request(*, app_state) -> Request:
    app = FastAPI()
    app.state.app_state = app_state
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/v1/insights/latest",
        "raw_path": b"/v1/insights/latest",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 0),
        "app": app,
    }
    return Request(scope)


def _make_key_store_stub(*, key_id: int = 42, tier: str = "STRATEGIST") -> MagicMock:
    ks = MagicMock()
    ks.verify_key.return_value = {"key_id": key_id, "label": "x", "tier": tier}
    ks.check_rate_limit.return_value = True
    return ks


def test_require_api_key_rejects_expired_subscription(monkeypatch, tier_manager):
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "0")
    # Force the module-level toggle to mirror env (it's read at import)
    import src.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "TESTING_MODE", False)

    u = tier_manager.create_user(email="lapsed@x.com", api_key_id=42, tier=UserTier.STRATEGIST)
    past = (datetime.utcnow() - timedelta(days=2)).isoformat()
    tier_manager.set_subscription_expires(u["user_id"], past)

    app_state = type("S", (), {})()
    app_state.key_store = _make_key_store_stub(key_id=42, tier="STRATEGIST")
    app_state.tier_manager = tier_manager
    req = _build_request(app_state=app_state)

    async def scenario():
        return await require_api_key(req, x_api_key="raw-key")

    with pytest.raises(HTTPException) as ei:
        run_async(scenario())
    assert ei.value.status_code == 402
    assert "expired" in ei.value.detail.lower()


def test_require_api_key_allows_active_subscription(monkeypatch, tier_manager):
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "0")
    import src.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "TESTING_MODE", False)

    u = tier_manager.create_user(email="active@x.com", api_key_id=43, tier=UserTier.STRATEGIST)
    future = (datetime.utcnow() + timedelta(days=10)).isoformat()
    tier_manager.set_subscription_expires(u["user_id"], future)

    app_state = type("S", (), {})()
    app_state.key_store = _make_key_store_stub(key_id=43, tier="STRATEGIST")
    app_state.tier_manager = tier_manager
    req = _build_request(app_state=app_state)

    async def scenario():
        return await require_api_key(req, x_api_key="raw-key")

    sub = run_async(scenario())
    assert sub["tier"] == "STRATEGIST"
    assert sub["subscription_expires"] == future


def test_require_api_key_allows_user_with_no_expires_field(monkeypatch, tier_manager):
    """A FREE user with subscription_expires=NULL must remain accessible."""
    monkeypatch.setenv("SENTINEL_TESTING_MODE", "0")
    import src.api.auth as auth_mod
    monkeypatch.setattr(auth_mod, "TESTING_MODE", False)

    u = tier_manager.create_user(email="free@x.com", api_key_id=44, tier=UserTier.FREE)
    assert tier_manager.get_user(u["user_id"])["subscription_expires"] is None

    app_state = type("S", (), {})()
    app_state.key_store = _make_key_store_stub(key_id=44, tier="FREE")
    app_state.tier_manager = tier_manager
    req = _build_request(app_state=app_state)

    async def scenario():
        return await require_api_key(req, x_api_key="raw-key")

    sub = run_async(scenario())
    assert sub["tier"] == "FREE"
