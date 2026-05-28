"""DG-056 — UNIQUE(api_key_id) constraint on users table.

Until v2 of the schema, ``users.api_key_id`` was merely indexed: two
distinct user rows could share the same key id. That means a leaked
key could be silently bound to a second user and grant access under a
different tier. Schema v2 adds a partial UNIQUE index (NULL-tolerant)
and ``link_api_key`` / ``create_user`` now refuse conflicting writes.
"""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from src.api.tier_manager import UserTier, UserTierManager


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as td:
        yield str(Path(td) / "users.db")


def _make_manager(path: str) -> UserTierManager:
    return UserTierManager(db_path=path)


# ---------------------------------------------------------------------------
# Schema state
# ---------------------------------------------------------------------------

def test_schema_v2_has_unique_index_on_api_key_id(db_path):
    _make_manager(db_path)
    conn = sqlite3.connect(db_path)
    try:
        # schema_version row
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 2
        # partial unique index exists
        idx = conn.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type='index' AND name='uq_users_api_key'"
        ).fetchone()
        assert idx is not None
        assert "UNIQUE" in idx[1].upper()
        assert "api_key_id IS NOT NULL" in idx[1]
    finally:
        conn.close()


def test_multiple_null_api_keys_allowed(db_path):
    mgr = _make_manager(db_path)
    # Two users with no key — both should be allowed
    u1 = mgr.create_user(email="a@x.com", tier=UserTier.FREE)
    u2 = mgr.create_user(email="b@x.com", tier=UserTier.FREE)
    assert u1["user_id"] != u2["user_id"]
    assert u1["api_key_id"] is None
    assert u2["api_key_id"] is None


# ---------------------------------------------------------------------------
# Uniqueness enforcement
# ---------------------------------------------------------------------------

def test_create_user_rejects_duplicate_api_key(db_path):
    mgr = _make_manager(db_path)
    mgr.create_user(email="a@x.com", api_key_id=42)
    with pytest.raises(ValueError):
        mgr.create_user(email="b@x.com", api_key_id=42)


def test_link_api_key_refuses_to_steal_existing_link(db_path):
    mgr = _make_manager(db_path)
    u1 = mgr.create_user(email="a@x.com", api_key_id=42)
    u2 = mgr.create_user(email="b@x.com")
    # Attempting to bind the same key to user 2 must fail cleanly
    ok = mgr.link_api_key(u2["user_id"], 42)
    assert ok is False
    # Original link untouched
    row = mgr.get_user_by_api_key(42)
    assert row is not None
    assert row["user_id"] == u1["user_id"]


def test_link_api_key_succeeds_for_fresh_key(db_path):
    mgr = _make_manager(db_path)
    u = mgr.create_user(email="a@x.com")
    ok = mgr.link_api_key(u["user_id"], 99)
    assert ok is True
    row = mgr.get_user_by_api_key(99)
    assert row is not None and row["user_id"] == u["user_id"]


# ---------------------------------------------------------------------------
# Migration from v1 → v2 with pre-existing duplicates
# ---------------------------------------------------------------------------

def test_migration_dedupes_existing_duplicate_links(db_path):
    """Pre-existing duplicate api_key_id rows are coalesced to the oldest."""
    # Pre-seed a v1 database with two users sharing api_key_id = 7
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript("""
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
            INSERT INTO schema_version (version) VALUES (1);
            CREATE TABLE users (
                user_id          INTEGER PRIMARY KEY AUTOINCREMENT,
                email            TEXT NOT NULL UNIQUE,
                tier             TEXT NOT NULL DEFAULT 'FREE',
                api_key_id       INTEGER,
                telegram_chat_id TEXT,
                stripe_customer_id TEXT,
                subscription_expires TEXT,
                created_at       TEXT NOT NULL,
                updated_at       TEXT NOT NULL
            );
            CREATE INDEX idx_users_api_key ON users(api_key_id);
            INSERT INTO users (email, tier, api_key_id, created_at, updated_at)
                VALUES ('older@x.com', 'FREE', 7, '2026-01-01', '2026-01-01');
            INSERT INTO users (email, tier, api_key_id, created_at, updated_at)
                VALUES ('newer@x.com', 'FREE', 7, '2026-02-01', '2026-02-01');
            CREATE TABLE usage_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id   INTEGER NOT NULL,
                endpoint  TEXT NOT NULL,
                timestamp REAL NOT NULL
            );
        """)
        conn.commit()
    finally:
        conn.close()

    # Bring up the manager — should run the v1→v2 migration
    mgr = _make_manager(db_path)
    older = mgr.get_user(1)
    newer = mgr.get_user(2)
    assert older["api_key_id"] == 7      # oldest keeps the key
    assert newer["api_key_id"] is None   # newer is cleared
    # Unique index is now in force — newer can no longer reclaim it
    assert mgr.link_api_key(newer["user_id"], 7) is False
