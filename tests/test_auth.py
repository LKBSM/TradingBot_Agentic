"""
Sprint 10 — API Key Authentication tests.

Covers:
  - KeyStore unit tests (create, verify, revoke, list, usage, rate limit)
  - Signal auth (401/200/429 with keys)
  - Operator auth (401/200 with keys)
  - Public endpoints (health + prometheus stay open)
  - Admin HMAC endpoints (create/revoke/list/usage, reject bad sig, replay)
"""

from __future__ import annotations

import hashlib
import hmac as hmac_mod
import time

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.auth import KeyStore
from src.api.signal_store import SignalRecord, SignalStore
from src.performance.metrics import MetricsRegistry


# =============================================================================
# HELPERS
# =============================================================================

class FakeHMACManager:
    """Minimal HMAC manager for testing admin auth."""

    def __init__(self, secret: bytes = b"test-secret"):
        self._secret = secret

    def sign(self, data: bytes):
        """Return a SignedData-like object with .signature."""
        sig = hmac_mod.new(self._secret, data, hashlib.sha256).hexdigest()

        class _Signed:
            signature = sig

        return _Signed()

    def verify(self, data: bytes, signature: str, key_version=None) -> bool:
        expected = hmac_mod.new(self._secret, data, hashlib.sha256).hexdigest()
        return hmac_mod.compare_digest(expected, signature)


def _admin_headers(hmac_mgr: FakeHMACManager) -> dict:
    """Build valid admin headers."""
    ts = str(time.time())
    sig = hmac_mgr.sign(ts.encode()).signature
    return {"X-Admin-Signature": sig, "X-Admin-Timestamp": ts}


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture()
def tmp_db(tmp_path):
    return str(tmp_path / "test_signals.db")


@pytest.fixture()
def tmp_keys_db(tmp_path):
    return str(tmp_path / "test_api_keys.db")


@pytest.fixture()
def key_store(tmp_keys_db):
    return KeyStore(db_path=tmp_keys_db)


@pytest.fixture()
def hmac_mgr():
    return FakeHMACManager()


@pytest.fixture()
def sample_signal() -> SignalRecord:
    return SignalRecord(
        signal_id="sig-001",
        action="OPEN_LONG",
        symbol="XAUUSD",
        entry_price=2350.50,
        stop_loss=2340.00,
        take_profit=2370.00,
        rr_ratio=1.86,
        created_at="2025-06-01T12:00:00",
    )


@pytest.fixture()
def authed_client(tmp_db, tmp_keys_db, hmac_mgr):
    """TestClient with KeyStore + HMACManager wired in."""
    signal_store = SignalStore(db_path=tmp_db)
    ks = KeyStore(db_path=tmp_keys_db)
    app = create_app(
        signal_store=signal_store,
        metrics_registry=MetricsRegistry(prefix="test"),
        key_store=ks,
        hmac_manager=hmac_mgr,
    )
    return TestClient(app)


@pytest.fixture()
def api_key(authed_client) -> str:
    """Pre-create a test key and return the raw key."""
    ks = authed_client.app.state.app_state.key_store
    result = ks.create_key("test-subscriber")
    return result["api_key"]


# =============================================================================
# KEYSTORE UNIT TESTS
# =============================================================================

class TestKeyStore:
    def test_create_key_returns_raw_key(self, key_store):
        result = key_store.create_key("subscriber-1")
        assert "api_key" in result
        assert "key_id" in result
        assert result["label"] == "subscriber-1"

    def test_key_prefix_sk(self, key_store):
        result = key_store.create_key("test")
        assert result["api_key"].startswith("sk_")

    def test_verify_valid_key(self, key_store):
        result = key_store.create_key("test")
        subscriber = key_store.verify_key(result["api_key"])
        assert subscriber is not None
        assert subscriber["key_id"] == result["key_id"]
        assert subscriber["label"] == "test"

    def test_verify_invalid_key(self, key_store):
        assert key_store.verify_key("sk_invalid") is None

    def test_verify_revoked_key(self, key_store):
        result = key_store.create_key("test")
        key_store.revoke_key(result["key_id"])
        assert key_store.verify_key(result["api_key"]) is None

    def test_revoke_returns_true(self, key_store):
        result = key_store.create_key("test")
        assert key_store.revoke_key(result["key_id"]) is True

    def test_revoke_nonexistent_returns_false(self, key_store):
        assert key_store.revoke_key(9999) is False

    def test_list_keys_no_hashes(self, key_store):
        key_store.create_key("sub-A")
        key_store.create_key("sub-B")
        keys = key_store.list_keys()
        assert len(keys) == 2
        for k in keys:
            assert "key_hash" not in k
            assert "api_key" not in k
            assert "key_id" in k
            assert "label" in k
            assert "is_active" in k

    def test_list_shows_revoked(self, key_store):
        result = key_store.create_key("revokable")
        key_store.revoke_key(result["key_id"])
        keys = key_store.list_keys()
        assert any(not k["is_active"] for k in keys)

    def test_usage_tracking(self, key_store):
        result = key_store.create_key("test")
        key_id = result["key_id"]
        key_store.record_usage(key_id, "/api/v1/signals/current")
        key_store.record_usage(key_id, "/api/v1/signals/current")
        key_store.record_usage(key_id, "/api/v1/signals/history")
        stats = key_store.get_usage(key_id, days=1)
        assert len(stats) == 2
        total = sum(s["count"] for s in stats)
        assert total == 3

    def test_rate_limit_under(self, key_store):
        result = key_store.create_key("test")
        assert key_store.check_rate_limit(result["key_id"]) is True

    def test_rate_limit_over(self, key_store):
        result = key_store.create_key("test")
        key_id = result["key_id"]
        # Insert 101 usage rows with current timestamps
        for _ in range(101):
            key_store.record_usage(key_id, "/test")
        assert key_store.check_rate_limit(key_id, max_per_minute=100) is False


# =============================================================================
# SIGNAL AUTH TESTS
# =============================================================================

class TestSignalAuth:
    def test_401_without_key(self, authed_client):
        resp = authed_client.get("/api/v1/signals/current")
        assert resp.status_code == 401

    def test_200_with_valid_key(self, authed_client, api_key, sample_signal):
        store = authed_client.app.state.app_state.signal_store
        store.publish(sample_signal)
        resp = authed_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200

    def test_401_with_revoked_key(self, authed_client, api_key):
        ks = authed_client.app.state.app_state.key_store
        # Find key_id from the key
        sub = ks.verify_key(api_key)
        ks.revoke_key(sub["key_id"])
        resp = authed_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 401

    def test_429_rate_limit(self, authed_client, api_key):
        ks = authed_client.app.state.app_state.key_store
        sub = ks.verify_key(api_key)
        # Flood usage to exceed rate limit
        for _ in range(101):
            ks.record_usage(sub["key_id"], "/flood")
        resp = authed_client.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 429

    def test_history_requires_key(self, authed_client):
        resp = authed_client.get("/api/v1/signals/history")
        assert resp.status_code == 401

    def test_history_with_key(self, authed_client, api_key):
        resp = authed_client.get(
            "/api/v1/signals/history",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200


# =============================================================================
# OPERATOR AUTH TESTS
# =============================================================================

class TestOperatorAuth:
    def test_401_without_key(self, authed_client):
        resp = authed_client.get("/api/v1/operator/metrics")
        assert resp.status_code == 401

    def test_200_with_key(self, authed_client, api_key):
        resp = authed_client.get(
            "/api/v1/operator/metrics",
            headers={"X-API-Key": api_key},
        )
        assert resp.status_code == 200

    def test_risk_requires_key(self, authed_client):
        resp = authed_client.get("/api/v1/operator/risk")
        assert resp.status_code == 401

    def test_kill_switch_requires_key(self, authed_client):
        resp = authed_client.get("/api/v1/operator/kill-switch")
        assert resp.status_code == 401


# =============================================================================
# PUBLIC ENDPOINTS STAY PUBLIC
# =============================================================================

class TestPublicEndpoints:
    def test_health_no_key_needed(self, authed_client):
        resp = authed_client.get("/api/v1/health")
        assert resp.status_code == 200

    def test_docker_health_no_key_needed(self, authed_client):
        resp = authed_client.get("/health")
        assert resp.status_code == 200

    def test_prometheus_no_key_needed(self, authed_client):
        resp = authed_client.get("/metrics")
        assert resp.status_code == 200


# =============================================================================
# ADMIN ENDPOINT TESTS
# =============================================================================

class TestAdminEndpoints:
    def test_create_key_via_admin(self, authed_client, hmac_mgr):
        headers = _admin_headers(hmac_mgr)
        resp = authed_client.post(
            "/api/v1/admin/keys",
            json={"label": "new-sub"},
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["label"] == "new-sub"
        assert body["api_key"].startswith("sk_")
        assert "key_id" in body

    def test_revoke_key_via_admin(self, authed_client, hmac_mgr):
        headers = _admin_headers(hmac_mgr)
        # Create a key first
        resp = authed_client.post(
            "/api/v1/admin/keys",
            json={"label": "to-revoke"},
            headers=headers,
        )
        key_id = resp.json()["key_id"]

        # Revoke it
        headers = _admin_headers(hmac_mgr)
        resp = authed_client.delete(
            f"/api/v1/admin/keys/{key_id}",
            headers=headers,
        )
        assert resp.status_code == 200
        assert resp.json()["revoked"] is True

    def test_list_keys_via_admin(self, authed_client, hmac_mgr):
        # Create two keys
        headers = _admin_headers(hmac_mgr)
        authed_client.post(
            "/api/v1/admin/keys",
            json={"label": "key-A"},
            headers=headers,
        )
        headers = _admin_headers(hmac_mgr)
        authed_client.post(
            "/api/v1/admin/keys",
            json={"label": "key-B"},
            headers=headers,
        )

        headers = _admin_headers(hmac_mgr)
        resp = authed_client.get("/api/v1/admin/keys", headers=headers)
        assert resp.status_code == 200
        # At least the 2 we just created (+ possibly fixture key)
        assert len(resp.json()["keys"]) >= 2

    def test_usage_via_admin(self, authed_client, hmac_mgr, api_key):
        # Make a request to generate usage
        authed_client.get(
            "/api/v1/signals/history",
            headers={"X-API-Key": api_key},
        )

        # Query usage
        ks = authed_client.app.state.app_state.key_store
        sub = ks.verify_key(api_key)
        headers = _admin_headers(hmac_mgr)
        resp = authed_client.get(
            f"/api/v1/admin/usage?key_id={sub['key_id']}&days=1",
            headers=headers,
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["key_id"] == sub["key_id"]
        assert isinstance(body["usage"], list)

    def test_reject_without_signature(self, authed_client):
        resp = authed_client.get("/api/v1/admin/keys")
        assert resp.status_code == 401

    def test_reject_stale_timestamp(self, authed_client, hmac_mgr):
        ts = str(time.time() - 600)  # 10 minutes ago
        sig = hmac_mgr.sign(ts.encode()).signature
        headers = {"X-Admin-Signature": sig, "X-Admin-Timestamp": ts}
        resp = authed_client.get("/api/v1/admin/keys", headers=headers)
        assert resp.status_code == 401

    def test_reject_bad_signature(self, authed_client):
        ts = str(time.time())
        headers = {"X-Admin-Signature": "bad", "X-Admin-Timestamp": ts}
        resp = authed_client.get("/api/v1/admin/keys", headers=headers)
        assert resp.status_code == 401


# =============================================================================
# FAIL-CLOSED TESTS
# =============================================================================

class TestFailClosed:
    def test_503_when_no_key_store(self, tmp_db):
        """Endpoints return 503 (not 200) when KeyStore is None."""
        app = create_app(
            signal_store=SignalStore(db_path=tmp_db),
            key_store=None,
        )
        c = TestClient(app)
        resp = c.get(
            "/api/v1/signals/current",
            headers={"X-API-Key": "sk_anything"},
        )
        assert resp.status_code == 503

    def test_503_admin_when_no_hmac(self, tmp_db, tmp_keys_db):
        """Admin endpoints return 503 when HMACManager is None."""
        app = create_app(
            signal_store=SignalStore(db_path=tmp_db),
            key_store=KeyStore(db_path=tmp_keys_db),
            hmac_manager=None,
        )
        c = TestClient(app)
        resp = c.get(
            "/api/v1/admin/keys",
            headers={"X-Admin-Signature": "x", "X-Admin-Timestamp": str(time.time())},
        )
        assert resp.status_code == 503

    @pytest.fixture()
    def tmp_keys_db(self, tmp_path):
        return str(tmp_path / "fc_keys.db")
