"""Tests for the API-2B.2 idempotency store + /enrich integration."""

from __future__ import annotations

import os

os.environ.setdefault("SENTINEL_TESTING_MODE", "1")

import pytest

from src.api.idempotency_store import (
    IdempotencyResult,
    IdempotencyStore,
    _hash_body,
)


# ---------------------------------------------------------------------------
# Body hashing
# ---------------------------------------------------------------------------


def test_hash_body_stable_under_dict_ordering():
    a = {"b": 2, "a": 1}
    b = {"a": 1, "b": 2}
    assert _hash_body(a) == _hash_body(b)


def test_hash_body_changes_with_value_change():
    a = {"a": 1, "b": 2}
    c = {"a": 1, "b": 3}
    assert _hash_body(a) != _hash_body(c)


def test_hash_body_handles_pydantic_v2_models():
    """The hasher must accept model_dump-able objects, not just dicts."""

    class _Stub:
        def model_dump(self, mode="json"):
            return {"x": 1}

    assert _hash_body(_Stub()) == _hash_body({"x": 1})


# ---------------------------------------------------------------------------
# Store — basic semantics
# ---------------------------------------------------------------------------


def test_lookup_miss_when_empty():
    s = IdempotencyStore()
    r = s.lookup("k1", "idem-1", "hash1")
    assert r.status == IdempotencyResult.MISS
    assert r.response is None


def test_store_then_lookup_hit():
    s = IdempotencyStore()
    s.store("k1", "idem-1", "hash1", {"answer": 42})
    r = s.lookup("k1", "idem-1", "hash1")
    assert r.status == IdempotencyResult.HIT
    assert r.response == {"answer": 42}


def test_lookup_clash_when_body_hash_differs():
    s = IdempotencyStore()
    s.store("k1", "idem-1", "hash1", {"answer": 42})
    r = s.lookup("k1", "idem-1", "DIFFERENT_HASH")
    assert r.status == IdempotencyResult.CLASH
    assert r.response is None


def test_distinct_api_keys_isolated():
    """Two subscribers can pick the same opaque idempotency key without
    seeing each other's responses."""
    s = IdempotencyStore()
    s.store("alice", "idem-shared", "hash_a", {"who": "alice"})
    s.store("bob", "idem-shared", "hash_b", {"who": "bob"})
    assert s.lookup("alice", "idem-shared", "hash_a").response == {"who": "alice"}
    assert s.lookup("bob", "idem-shared", "hash_b").response == {"who": "bob"}


def test_empty_idempotency_key_rejected():
    s = IdempotencyStore()
    with pytest.raises(ValueError):
        s.lookup("k1", "", "hash1")
    with pytest.raises(ValueError):
        s.store("k1", "", "hash1", {})


# ---------------------------------------------------------------------------
# Store — TTL
# ---------------------------------------------------------------------------


def test_ttl_expires_after_window():
    fake_now = [1000.0]
    s = IdempotencyStore(ttl_seconds=60.0, clock=lambda: fake_now[0])
    s.store("k1", "idem-1", "hash1", {"x": 1})

    assert s.lookup("k1", "idem-1", "hash1").status == IdempotencyResult.HIT

    # Advance past the TTL
    fake_now[0] = 1100.0
    assert s.lookup("k1", "idem-1", "hash1").status == IdempotencyResult.MISS


def test_ttl_zero_rejected():
    with pytest.raises(ValueError):
        IdempotencyStore(ttl_seconds=0)


def test_purge_returns_dropped_count():
    fake_now = [1000.0]
    s = IdempotencyStore(ttl_seconds=60.0, clock=lambda: fake_now[0])
    for i in range(5):
        s.store("k1", f"idem-{i}", f"h-{i}", {"i": i})
    fake_now[0] = 5000.0
    assert s.purge() == 5
    assert s.size == 0


# ---------------------------------------------------------------------------
# Capacity bounds
# ---------------------------------------------------------------------------


def test_capacity_evicts_oldest():
    fake_now = [1000.0]
    s = IdempotencyStore(
        ttl_seconds=86400.0, max_entries=3, clock=lambda: fake_now[0]
    )
    for i in range(5):
        fake_now[0] += 1.0
        s.store("k1", f"idem-{i}", f"h-{i}", {"i": i})
    assert s.size == 3
    # The first two should be gone
    assert s.lookup("k1", "idem-0", "h-0").status == IdempotencyResult.MISS
    assert s.lookup("k1", "idem-4", "h-4").status == IdempotencyResult.HIT


# ---------------------------------------------------------------------------
# /enrich endpoint integration
# ---------------------------------------------------------------------------


from unittest.mock import patch  # noqa: E402

from fastapi.testclient import TestClient  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.api.routes.qa import build_default_rag_pipeline  # noqa: E402


@pytest.fixture(autouse=True)
def _force_endpoint_testing_mode():
    """Endpoint-mode bypass — separate fixture so unit tests above (which
    don't need it) aren't slowed by RAG pipeline construction."""
    with (
        patch("src.api.auth.TESTING_MODE", True),
        patch("src.api.routes.enrich.TESTING_MODE", True),
    ):
        yield


@pytest.fixture(scope="module")
def populated_pipeline():
    return build_default_rag_pipeline()


@pytest.fixture
def enrich_client(populated_pipeline):
    return TestClient(
        create_app(
            rag_pipeline=populated_pipeline,
            idempotency_store=IdempotencyStore(),
        )
    )


def _enrich_payload():
    return {
        "instrument": "XAUUSD",
        "timeframe": "M15",
        "direction": "BULLISH_SETUP",
        "language": "en",
        "entry": 2400.0,
        "stop": 2390.0,
        "target_1": 2420.0,
    }


def test_enrich_first_call_no_replay_header(enrich_client):
    resp = enrich_client.post(
        "/api/v1/enrich",
        json=_enrich_payload(),
        headers={"Idempotency-Key": "test-key-1"},
    )
    assert resp.status_code == 200
    assert "idempotent-replay" not in {h.lower() for h in resp.headers}


def test_enrich_repeated_idempotency_key_replays_response(enrich_client):
    payload = _enrich_payload()
    r1 = enrich_client.post(
        "/api/v1/enrich",
        json=payload,
        headers={"Idempotency-Key": "test-key-2"},
    )
    r2 = enrich_client.post(
        "/api/v1/enrich",
        json=payload,
        headers={"Idempotency-Key": "test-key-2"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.headers.get("Idempotent-Replay") == "true"
    # Same payload → identical id (replayed)
    assert r1.json()["id"] == r2.json()["id"]


def test_enrich_clashing_body_returns_409(enrich_client):
    p1 = _enrich_payload()
    p2 = _enrich_payload()
    p2["entry"] = 2500.0  # different body, same key
    r1 = enrich_client.post(
        "/api/v1/enrich", json=p1, headers={"Idempotency-Key": "clash-key"}
    )
    r2 = enrich_client.post(
        "/api/v1/enrich", json=p2, headers={"Idempotency-Key": "clash-key"}
    )
    assert r1.status_code == 200
    assert r2.status_code == 409
    assert "different body" in r2.json()["detail"]


def test_enrich_without_idempotency_key_works_unchanged(enrich_client):
    """Backwards compat: existing clients that don't send the header
    must not see any new behaviour."""
    r1 = enrich_client.post("/api/v1/enrich", json=_enrich_payload())
    r2 = enrich_client.post("/api/v1/enrich", json=_enrich_payload())
    assert r1.status_code == 200
    assert r2.status_code == 200
    # Without idempotency, ids are independent
    assert r1.json()["id"] != r2.json()["id"]


def test_enrich_idempotency_disabled_when_store_unconfigured(populated_pipeline):
    """If no IdempotencyStore is wired in, the header is silently
    ignored (no crash, no replay)."""
    c = TestClient(create_app(rag_pipeline=populated_pipeline))
    r1 = c.post(
        "/api/v1/enrich",
        json=_enrich_payload(),
        headers={"Idempotency-Key": "x"},
    )
    r2 = c.post(
        "/api/v1/enrich",
        json=_enrich_payload(),
        headers={"Idempotency-Key": "x"},
    )
    assert r1.status_code == 200
    assert r2.status_code == 200
    # No store ⇒ no replay header even with the key
    assert "idempotent-replay" not in {h.lower() for h in r2.headers}
