"""Tests for the SECURITY-2B.2 API key rotation flow."""

from __future__ import annotations

import time

import pytest

from src.api.auth import KeyStore


@pytest.fixture
def store(tmp_path):
    return KeyStore(db_path=str(tmp_path / "keys.db"))


# ---------------------------------------------------------------------------
# Schema migration
# ---------------------------------------------------------------------------


def test_v1_db_migrates_to_v2_with_superseded_columns(tmp_path):
    """Existing v1 row keeps verifying after migration."""
    import sqlite3

    p = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(p))
    conn.executescript("""
        CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
        CREATE TABLE api_keys (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_hash TEXT NOT NULL UNIQUE,
            label TEXT NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE TABLE api_usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key_id INTEGER, endpoint TEXT, timestamp REAL
        );
        INSERT INTO schema_version VALUES (1);
        INSERT INTO api_keys (key_hash, label, created_at, is_active)
        VALUES ('deadbeef', 'legacy', '2026-01-01', 1);
    """)
    conn.commit()
    conn.close()

    store = KeyStore(db_path=str(p))
    keys = store.list_keys()
    assert len(keys) == 1
    assert keys[0]["label"] == "legacy"
    # New columns surface as None for pre-existing rows.
    assert keys[0]["superseded_at"] is None
    assert keys[0]["superseded_by"] is None


# ---------------------------------------------------------------------------
# rotate_key — basic behaviour
# ---------------------------------------------------------------------------


def test_rotate_creates_new_key_with_distinct_id(store):
    created = store.create_key("broker-1")
    res = store.rotate_key(created["key_id"])
    assert res is not None
    assert res["old_key_id"] == created["key_id"]
    assert res["new_key_id"] != created["key_id"]
    assert res["new_api_key"].startswith("sk_")
    # New label carries a "rotated" suffix
    assert "rotated" in res["label"]


def test_both_keys_verify_during_grace(store):
    created = store.create_key("broker-1")
    res = store.rotate_key(created["key_id"], grace_seconds=3600)

    # Old key still works
    sub_old = store.verify_key(created["api_key"])
    assert sub_old is not None
    assert sub_old["key_id"] == created["key_id"]
    assert sub_old["superseded_at"] is not None
    assert sub_old["superseded_by"] == res["new_key_id"]

    # New key works too
    sub_new = store.verify_key(res["new_api_key"])
    assert sub_new is not None
    assert sub_new["key_id"] == res["new_key_id"]
    assert sub_new.get("superseded_at") is None


def test_old_key_stops_verifying_after_grace(store):
    created = store.create_key("broker-1")
    res = store.rotate_key(created["key_id"], grace_seconds=0)
    # Force the verify cache to clear so we re-read from DB.
    store._cache_invalidate()
    # Sleep a tick — superseded_at is *now+0*, anything after that
    # disqualifies the old key.
    time.sleep(0.01)
    assert store.verify_key(created["api_key"]) is None
    # New key still good.
    assert store.verify_key(res["new_api_key"]) is not None


# ---------------------------------------------------------------------------
# rotate_key — guards
# ---------------------------------------------------------------------------


def test_rotate_unknown_key_returns_none(store):
    assert store.rotate_key(9999) is None


def test_rotate_revoked_key_raises(store):
    created = store.create_key("rev")
    store.revoke_key(created["key_id"])
    with pytest.raises(ValueError, match="revoked"):
        store.rotate_key(created["key_id"])


def test_double_rotate_is_rejected(store):
    created = store.create_key("once")
    store.rotate_key(created["key_id"], grace_seconds=86400)
    with pytest.raises(ValueError, match="already been rotated"):
        store.rotate_key(created["key_id"])


def test_grace_seconds_must_be_non_negative(store):
    created = store.create_key("x")
    with pytest.raises(ValueError):
        store.rotate_key(created["key_id"], grace_seconds=-1)


def test_grace_seconds_capped_at_30_days(store):
    created = store.create_key("x")
    with pytest.raises(ValueError):
        store.rotate_key(created["key_id"], grace_seconds=31 * 86400)


# ---------------------------------------------------------------------------
# list_keys after rotation surfaces both rows
# ---------------------------------------------------------------------------


def test_list_keys_shows_old_and_new_after_rotation(store):
    created = store.create_key("ledger-feed")
    res = store.rotate_key(created["key_id"], grace_seconds=120)
    rows = {r["key_id"]: r for r in store.list_keys()}
    assert created["key_id"] in rows
    assert res["new_key_id"] in rows
    old = rows[created["key_id"]]
    new = rows[res["new_key_id"]]
    assert old["superseded_at"] is not None
    assert old["superseded_by"] == res["new_key_id"]
    assert new["superseded_at"] is None
    assert new["superseded_by"] is None
