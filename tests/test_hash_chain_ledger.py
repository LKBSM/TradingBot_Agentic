"""Tests for the DATA-2B.4 hash-chained audit ledger.

Covers:
- Canonical JSON determinism (key order / spacing)
- Genesis prev_hash sentinel
- Append → recover sequence numbers + chain linkage
- Pydantic v2 model + plain dict + insight_id override
- ``verify()`` returns ok on a fresh chain
- Tamper detection: body mutation, prev_hash mutation, missing seq
- Persistence across reopen (file-backed db)
- Thread-safety: concurrent appends maintain a coherent chain
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone

import pytest

from src.audit import HashChainLedger, LedgerEntry, canonical_json
from src.audit.hash_chain_ledger import GENESIS_PREV_HASH, _hash_entry


# ---------------------------------------------------------------------------
# canonical_json
# ---------------------------------------------------------------------------


def test_canonical_json_sorted_keys():
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b
    assert a == '{"a":1,"b":2}'


def test_canonical_json_no_whitespace():
    out = canonical_json({"x": [1, 2, 3]})
    assert " " not in out
    assert "\n" not in out


def test_canonical_json_handles_datetime_via_default():
    out = canonical_json({"ts": datetime(2026, 5, 4, tzinfo=timezone.utc)})
    assert "2026-05-04" in out


def test_canonical_json_preserves_unicode():
    out = canonical_json({"narrative": "Synthèse complète"})
    assert "Synthèse complète" in out


# ---------------------------------------------------------------------------
# Append / sequence / chain linkage
# ---------------------------------------------------------------------------


def _payload(idx: int = 1) -> dict:
    return {
        "id": f"insight-{idx}",
        "instrument": "XAUUSD",
        "direction": "BULLISH_SETUP",
        "conviction_0_100": 60 + idx,
        "narrative_short": f"signal {idx}",
    }


def test_append_first_entry_has_genesis_prev_hash():
    ledger = HashChainLedger()
    e = ledger.append(_payload(1))
    assert e.seq == 1
    assert e.prev_hash == GENESIS_PREV_HASH
    assert len(e.entry_hash) == 64
    assert ledger.size == 1


def test_append_chain_advances_correctly():
    ledger = HashChainLedger()
    e1 = ledger.append(_payload(1))
    e2 = ledger.append(_payload(2))
    e3 = ledger.append(_payload(3))
    assert e2.prev_hash == e1.entry_hash
    assert e3.prev_hash == e2.entry_hash
    assert ledger.head_hash == e3.entry_hash
    assert [e.seq for e in (e1, e2, e3)] == [1, 2, 3]


def test_append_records_canonical_body_not_raw_dict():
    ledger = HashChainLedger()
    e = ledger.append({"id": "x", "z": 9, "a": 1})
    # Sorted keys ⇒ 'a' before 'id' before 'z'
    assert e.canonical_json == '{"a":1,"id":"x","z":9}'


def test_append_explicit_insight_id_overrides_dict_id():
    ledger = HashChainLedger()
    e = ledger.append({"id": "from-dict"}, insight_id="from-arg")
    assert e.insight_id == "from-arg"


def test_append_raises_when_no_id_available():
    ledger = HashChainLedger()
    with pytest.raises(ValueError, match="insight_id"):
        ledger.append({"no_id_here": True})


def test_append_supports_pydantic_v2_model():
    """Use the actual InsightSignalV2 type as the integration probe."""
    from src.api.insight_signal_v2 import (
        ComplianceMeta,
        InsightSignalV2,
        SetupDirection,
        Timeframe,
    )

    insight = InsightSignalV2(
        id="pyd-1",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=30,
        narrative_short="neutral",
        compliance=ComplianceMeta(),
        created_at_utc=datetime(2026, 5, 4, tzinfo=timezone.utc),
    )
    ledger = HashChainLedger()
    e = ledger.append(insight)
    assert e.insight_id == "pyd-1"
    # Body must be valid JSON.
    body = json.loads(e.canonical_json)
    assert body["instrument"] == "XAUUSD"


# ---------------------------------------------------------------------------
# get / find / iter
# ---------------------------------------------------------------------------


def test_get_returns_entry_by_seq():
    ledger = HashChainLedger()
    e1 = ledger.append(_payload(1))
    fetched = ledger.get(1)
    assert fetched.entry_hash == e1.entry_hash


def test_get_unknown_seq_returns_none():
    ledger = HashChainLedger()
    assert ledger.get(99) is None


def test_find_by_insight_id_returns_all_entries():
    ledger = HashChainLedger()
    ledger.append(_payload(1))
    ledger.append(_payload(1))  # same id twice
    ledger.append(_payload(2))
    matches = ledger.find_by_insight_id("insight-1")
    assert len(matches) == 2
    assert all(m.insight_id == "insight-1" for m in matches)


def test_iter_entries_walks_in_order():
    ledger = HashChainLedger()
    for i in range(1, 6):
        ledger.append(_payload(i))
    seqs = [e.seq for e in ledger.iter_entries()]
    assert seqs == [1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


def test_verify_intact_chain():
    ledger = HashChainLedger()
    for i in range(1, 11):
        ledger.append(_payload(i))
    result = ledger.verify()
    assert result.ok
    assert result.n_entries == 10
    assert result.broken_at_seq is None


def test_verify_empty_chain_is_ok():
    ledger = HashChainLedger()
    result = ledger.verify()
    assert result.ok
    assert result.n_entries == 0


def test_verify_detects_body_tampering(tmp_path):
    db = tmp_path / "led.db"
    ledger = HashChainLedger(db)
    for i in range(1, 4):
        ledger.append(_payload(i))
    # Tamper directly on disk: change the body of seq=2 ⇒ entry_hash mismatch
    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE ledger SET canonical_json = ? WHERE seq = 2",
        (canonical_json({"id": "tampered"}),),
    )
    raw.commit()
    raw.close()

    result = ledger.verify()
    assert not result.ok
    assert result.broken_at_seq == 2
    assert "tamper" in result.reason.lower() or "mismatch" in result.reason.lower()


def test_verify_detects_prev_hash_break(tmp_path):
    db = tmp_path / "led.db"
    ledger = HashChainLedger(db)
    ledger.append(_payload(1))
    ledger.append(_payload(2))
    ledger.append(_payload(3))
    # Mutate prev_hash on seq=3 → chain link broken
    raw = sqlite3.connect(db)
    raw.execute(
        "UPDATE ledger SET prev_hash = ? WHERE seq = 3",
        ("ff" * 32,),
    )
    raw.commit()
    raw.close()
    result = ledger.verify()
    assert not result.ok
    assert result.broken_at_seq == 3


def test_verify_detects_missing_seq(tmp_path):
    db = tmp_path / "led.db"
    ledger = HashChainLedger(db)
    for i in range(1, 5):
        ledger.append(_payload(i))
    # Delete seq=2 ⇒ verify should flag non-contiguous at seq=3
    raw = sqlite3.connect(db)
    raw.execute("DELETE FROM ledger WHERE seq = 2")
    raw.commit()
    raw.close()
    result = ledger.verify()
    assert not result.ok
    assert result.broken_at_seq == 3
    assert "contiguous" in result.reason


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_ledger_survives_reopen(tmp_path):
    db = tmp_path / "led.db"
    led1 = HashChainLedger(db)
    e1 = led1.append(_payload(1))
    e2 = led1.append(_payload(2))
    led1.close()

    led2 = HashChainLedger(db)
    assert led2.size == 2
    assert led2.head_hash == e2.entry_hash
    e3 = led2.append(_payload(3))
    assert e3.prev_hash == e2.entry_hash
    assert led2.verify().ok


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


def test_concurrent_appends_maintain_chain(tmp_path):
    db = tmp_path / "led.db"
    ledger = HashChainLedger(db)
    n_threads = 8
    appends_per_thread = 25

    def worker(tid: int):
        for j in range(appends_per_thread):
            ledger.append({"id": f"t{tid}-{j}", "tid": tid, "j": j})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    expected = n_threads * appends_per_thread
    assert ledger.size == expected
    result = ledger.verify()
    assert result.ok, result.reason
    assert result.n_entries == expected


# ---------------------------------------------------------------------------
# Hash function unit
# ---------------------------------------------------------------------------


def test_hash_entry_is_deterministic():
    h1 = _hash_entry(1, "2026-05-04T00:00:00Z", '{"a":1}', GENESIS_PREV_HASH)
    h2 = _hash_entry(1, "2026-05-04T00:00:00Z", '{"a":1}', GENESIS_PREV_HASH)
    assert h1 == h2


def test_hash_entry_changes_with_any_field():
    base = (1, "2026-05-04T00:00:00Z", '{"a":1}', GENESIS_PREV_HASH)
    h0 = _hash_entry(*base)
    assert _hash_entry(2, *base[1:]) != h0
    assert _hash_entry(base[0], "2026-05-05T00:00:00Z", *base[2:]) != h0
    assert _hash_entry(*base[:2], '{"a":2}', base[3]) != h0
    assert _hash_entry(*base[:3], "ff" * 32) != h0


# ---------------------------------------------------------------------------
# VerificationResult helpers
# ---------------------------------------------------------------------------


def test_verification_result_truthy_only_when_ok():
    from src.audit import VerificationResult

    assert bool(VerificationResult(ok=True, n_entries=0))
    assert not bool(VerificationResult(ok=False, n_entries=1, broken_at_seq=1))
