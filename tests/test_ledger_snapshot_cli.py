"""Tests for the DATA-2B.9 audit ledger snapshot/restore CLI."""

from __future__ import annotations

import json

import pytest

from scripts.audit_ledger_snapshot import (
    SNAPSHOT_VERSION,
    cmd_restore,
    cmd_snapshot,
    cmd_verify,
)
from src.audit.hash_chain_ledger import HashChainLedger


@pytest.fixture
def populated_db(tmp_path):
    """A persistent SQLite ledger with three sealed insights."""
    p = tmp_path / "ledger.db"
    led = HashChainLedger(db_path=str(p))
    led.append({"id": "alpha", "v": 1})
    led.append({"id": "beta", "v": 2})
    led.append({"id": "gamma", "v": 3})
    led.close()
    return str(p)


# ---------------------------------------------------------------------------
# snapshot
# ---------------------------------------------------------------------------


def test_snapshot_writes_header_plus_one_line_per_entry(populated_db, tmp_path):
    out = tmp_path / "snap.jsonl"
    rc = cmd_snapshot(populated_db, str(out))
    assert rc == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # header + 3 entries

    header = json.loads(lines[0])
    assert "_meta" in header
    assert header["_meta"]["snapshot_v"] == SNAPSHOT_VERSION
    assert header["_meta"]["n_entries"] == 3
    assert len(header["_meta"]["head_hash"]) == 64

    seqs = [json.loads(l)["seq"] for l in lines[1:]]
    assert seqs == [1, 2, 3]


def test_snapshot_aborts_on_broken_chain(tmp_path):
    """Tamper with a row → snapshot must refuse."""
    import sqlite3

    p = tmp_path / "broken.db"
    led = HashChainLedger(db_path=str(p))
    led.append({"id": "x"})
    led.close()
    # Flip a byte in canonical_json — invalidates entry_hash.
    conn = sqlite3.connect(str(p))
    conn.execute(
        "UPDATE ledger SET canonical_json = ? WHERE seq = 1",
        ('{"id":"TAMPERED"}',),
    )
    conn.commit(); conn.close()

    rc = cmd_snapshot(str(p), str(tmp_path / "snap.jsonl"))
    assert rc == 2


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


def test_restore_recovers_bitwise_identical_chain(populated_db, tmp_path):
    snap = tmp_path / "snap.jsonl"
    cmd_snapshot(populated_db, str(snap))

    new_db = tmp_path / "restored.db"
    rc = cmd_restore(str(snap), str(new_db))
    assert rc == 0

    # Open restored DB and confirm head_hash matches.
    led_src = HashChainLedger(db_path=populated_db)
    led_dst = HashChainLedger(db_path=str(new_db))
    try:
        assert led_dst.size == led_src.size
        assert led_dst.head_hash == led_src.head_hash
        for a, b in zip(led_src.iter_entries(), led_dst.iter_entries()):
            assert a.seq == b.seq
            assert a.canonical_json == b.canonical_json
            assert a.entry_hash == b.entry_hash
        assert led_dst.verify().ok
    finally:
        led_src.close(); led_dst.close()


def test_restore_refuses_existing_target(populated_db, tmp_path):
    snap = tmp_path / "snap.jsonl"
    cmd_snapshot(populated_db, str(snap))

    rc = cmd_restore(str(snap), populated_db)  # target already exists
    assert rc == 2


def test_restore_detects_tampered_archive(populated_db, tmp_path):
    snap = tmp_path / "snap.jsonl"
    cmd_snapshot(populated_db, str(snap))
    # Flip a byte inside an entry's canonical_json field.
    raw = snap.read_text(encoding="utf-8").splitlines()
    middle = json.loads(raw[2])
    middle["canonical_json"] = '{"id":"PWNED"}'
    raw[2] = json.dumps(middle)
    snap.write_text("\n".join(raw) + "\n", encoding="utf-8")

    rc = cmd_restore(str(snap), str(tmp_path / "tampered.db"))
    assert rc == 2


def test_restore_rejects_unknown_snapshot_version(tmp_path):
    snap = tmp_path / "snap.jsonl"
    snap.write_text(
        json.dumps({"_meta": {"snapshot_v": 99}}) + "\n",
        encoding="utf-8",
    )
    rc = cmd_restore(str(snap), str(tmp_path / "x.db"))
    assert rc == 2


def test_restore_rejects_missing_header(tmp_path):
    snap = tmp_path / "snap.jsonl"
    snap.write_text("not-a-header\n", encoding="utf-8")
    rc = cmd_restore(str(snap), str(tmp_path / "x.db"))
    assert rc == 2


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------


def test_verify_passes_on_healthy_db(populated_db, capsys):
    rc = cmd_verify(populated_db)
    assert rc == 0
    out = capsys.readouterr().out
    assert "verify OK" in out
    assert "3 entries" in out


def test_verify_fails_on_tampered_db(populated_db, tmp_path):
    import sqlite3

    conn = sqlite3.connect(populated_db)
    conn.execute("UPDATE ledger SET canonical_json = ? WHERE seq = 2", ('{"x":1}',))
    conn.commit(); conn.close()

    rc = cmd_verify(populated_db)
    assert rc == 2


# ---------------------------------------------------------------------------
# Restore method directly — empty-DB guard
# ---------------------------------------------------------------------------


def test_restore_from_entries_refuses_non_empty_target(populated_db, tmp_path):
    src = HashChainLedger(db_path=populated_db)
    try:
        entries = list(src.iter_entries())
    finally:
        src.close()

    # Target already has one entry.
    target = tmp_path / "dst.db"
    dst = HashChainLedger(db_path=str(target))
    try:
        dst.append({"id": "preexisting"})
        with pytest.raises(ValueError, match="non-empty"):
            dst.restore_from_entries(iter(entries))
    finally:
        dst.close()
