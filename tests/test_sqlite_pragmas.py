"""Guards for the resilient SQLite journal helper (incident 2026-08-05)."""

from __future__ import annotations

import sqlite3

import pytest

from src.persistence.sqlite_pragmas import apply_wal


def test_apply_wal_sets_wal_on_a_normal_db(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "t.db"))
    try:
        assert apply_wal(conn) == "wal"
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"
    finally:
        conn.close()


class _WalHostileConn:
    """A connection whose journal_mode=WAL raises 'disk I/O error' (as a
    WAL-unsupported disk / stale -shm does), but DELETE succeeds."""

    def __init__(self) -> None:
        self.calls = []

    def execute(self, sql: str):
        self.calls.append(sql)
        if "WAL" in sql:
            raise sqlite3.OperationalError("disk I/O error")
        return None


def test_apply_wal_falls_back_to_delete_when_wal_raises(caplog):
    conn = _WalHostileConn()
    with caplog.at_level("WARNING"):
        assert apply_wal(conn) == "delete"
    assert any("PRAGMA journal_mode=DELETE" in c for c in conn.calls)
    assert "WAL unavailable" in " ".join(r.getMessage() for r in caplog.records)


class _DiskDeadConn:
    """Every journal PRAGMA fails — a truly full/read-only disk."""

    def execute(self, sql: str):
        raise sqlite3.OperationalError("disk I/O error")


def test_apply_wal_never_raises_even_if_delete_also_fails(caplog):
    # Boot must not crash on journal configuration: worst case returns "unknown".
    with caplog.at_level("ERROR"):
        assert apply_wal(_DiskDeadConn()) == "unknown"
