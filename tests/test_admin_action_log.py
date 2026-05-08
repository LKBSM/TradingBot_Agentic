"""Tests for the SECURITY-2B.1 admin action audit log."""

from __future__ import annotations

import pytest

from src.audit import AdminActionLog, AdminActionRecord, payload_digest


# ---------------------------------------------------------------------------
# payload_digest
# ---------------------------------------------------------------------------


def test_digest_stable_under_dict_ordering():
    assert payload_digest({"a": 1, "b": 2}) == payload_digest({"b": 2, "a": 1})


def test_digest_changes_when_value_changes():
    assert payload_digest({"a": 1}) != payload_digest({"a": 2})


def test_digest_returns_dash_for_none():
    assert payload_digest(None) == "-"


def test_digest_handles_pydantic_v2():
    class _Stub:
        def model_dump(self, mode="json"):
            return {"x": 1}

    assert payload_digest(_Stub()) == payload_digest({"x": 1})


def test_digest_is_16_hex_chars():
    d = payload_digest({"foo": "bar"})
    assert len(d) == 16
    int(d, 16)


# ---------------------------------------------------------------------------
# AdminActionLog — record + query
# ---------------------------------------------------------------------------


def test_record_returns_committed_record():
    log = AdminActionLog()
    rec = log.record(
        actor="admin-9a3c",
        action="create_key",
        target="key-42",
        payload={"label": "broker"},
        result="ok",
        request_id="req-1",
    )
    assert isinstance(rec, AdminActionRecord)
    assert rec.id >= 1
    assert rec.actor == "admin-9a3c"
    assert rec.action == "create_key"
    assert rec.target == "key-42"
    assert rec.payload_digest == payload_digest({"label": "broker"})
    assert rec.ts_utc.endswith("Z")


def test_record_rejects_empty_actor_or_action():
    log = AdminActionLog()
    with pytest.raises(ValueError):
        log.record(actor="", action="x")
    with pytest.raises(ValueError):
        log.record(actor="x", action="")


def test_record_defaults_target_payload_result_request_id():
    log = AdminActionLog()
    rec = log.record(actor="a", action="ping")
    assert rec.target == "-"
    assert rec.payload_digest == "-"
    assert rec.result == "ok"
    assert rec.request_id == "-"


def test_size_grows_with_records():
    log = AdminActionLog()
    assert log.size == 0
    for i in range(3):
        log.record(actor="a", action=f"action-{i}")
    assert log.size == 3


# ---------------------------------------------------------------------------
# query() — filters + pagination
# ---------------------------------------------------------------------------


def _seed(log: AdminActionLog) -> None:
    log.record(actor="alice", action="create_key", target="k1", result="ok")
    log.record(actor="alice", action="revoke_key", target="k1", result="ok")
    log.record(actor="bob", action="create_key", target="k2", result="failed:dup")
    log.record(actor="alice", action="create_key", target="k3", result="ok")


def test_query_default_returns_newest_first():
    log = AdminActionLog()
    _seed(log)
    rows = log.query()
    # Reverse of insert order
    assert [r.target for r in rows] == ["k3", "k2", "k1", "k1"]


def test_query_filter_by_actor():
    log = AdminActionLog()
    _seed(log)
    rows = log.query(actor="alice")
    assert {r.actor for r in rows} == {"alice"}
    assert len(rows) == 3


def test_query_filter_by_action():
    log = AdminActionLog()
    _seed(log)
    rows = log.query(action="create_key")
    assert {r.action for r in rows} == {"create_key"}
    assert len(rows) == 3


def test_query_filter_combined():
    log = AdminActionLog()
    _seed(log)
    rows = log.query(actor="alice", action="create_key")
    assert len(rows) == 2
    assert all(r.actor == "alice" and r.action == "create_key" for r in rows)


def test_query_filter_by_since_iso_future_excludes_all():
    log = AdminActionLog()
    _seed(log)
    rows = log.query(since_iso="2099-01-01T00:00:00.000000Z")
    assert rows == []


def test_query_limit_cap():
    log = AdminActionLog()
    for i in range(20):
        log.record(actor="a", action=f"act-{i}")
    rows = log.query(limit=5)
    assert len(rows) == 5


def test_query_rejects_zero_limit():
    log = AdminActionLog()
    with pytest.raises(ValueError):
        log.query(limit=0)


def test_iter_records_returns_in_id_order():
    log = AdminActionLog()
    _seed(log)
    seen = [r.action for r in log.iter_records()]
    assert seen == ["create_key", "revoke_key", "create_key", "create_key"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_records_persist_across_reopen(tmp_path):
    db = tmp_path / "admin.db"
    log = AdminActionLog(db_path=db)
    log.record(actor="a", action="x", target="t", result="ok")
    log.close()

    log2 = AdminActionLog(db_path=db)
    rows = log2.query()
    assert len(rows) == 1
    assert rows[0].target == "t"
