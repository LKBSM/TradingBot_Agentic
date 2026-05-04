"""Append-only hash-chained ledger of delivered insights — Sprint DATA-2B.4.

Why this matters
----------------
B2B clients (brokers, prop desks, compliance teams) need to prove,
sometimes years after the fact, that:

1. they received insight X at timestamp T,
2. its body was Y exactly (no in-flight tampering),
3. the broader sequence of insights surrounding it has not been
   reordered, mutated, or had entries silently inserted.

A hash chain — a la Bitcoin block headers, but lighter — gives this
property without needing a trusted third party. Each entry's hash is
derived from the previous entry's hash plus the canonical body, so any
mutation cascades forward and is detectable in O(N) at audit time.

Schema
------
Single SQLite table ``ledger``:

  - ``seq``               INTEGER PRIMARY KEY (autoincrement, 1-based)
  - ``inserted_at_utc``   TEXT (ISO-8601 with 'Z' suffix)
  - ``insight_id``        TEXT (the InsightSignalV2.id, indexed)
  - ``canonical_json``    TEXT (deterministic body — sorted keys, no spaces)
  - ``prev_hash``         TEXT (hex SHA-256 of previous entry, or 64×'0' for genesis)
  - ``entry_hash``        TEXT (hex SHA-256 of seq | ts | body | prev_hash)

The DB is opened in WAL mode for safe reader concurrency. Writes are
serialised through a Python ``threading.Lock`` so the in-memory
``last_hash`` we use for chaining stays consistent with what's on disk.

Tamper detection
----------------
``verify()`` walks the chain from ``seq=1`` to the latest, recomputes
each ``entry_hash`` from disk fields, and returns the first ``seq`` at
which a mismatch is observed (None if all entries are intact).
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


GENESIS_PREV_HASH = "0" * 64


# ---------------------------------------------------------------------------
# Canonical JSON
# ---------------------------------------------------------------------------


def canonical_json(payload: Any) -> str:
    """Deterministic JSON serialisation used for hashing.

    Rules:
    - sort_keys=True so dict ordering can never change the hash
    - separators=(',', ':') strips all optional whitespace
    - ensure_ascii=False keeps multi-language narratives intact (UTF-8)
    - default= str() for datetimes / Decimals / unknown types

    Pydantic v2 callers should pass ``model.model_dump(mode='json')``
    so enums become strings and datetimes ISO-8601 before reaching here.
    """
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )


def _hash_entry(seq: int, ts_iso: str, body: str, prev_hash: str) -> str:
    h = hashlib.sha256()
    h.update(str(seq).encode("utf-8"))
    h.update(b"\x1f")
    h.update(ts_iso.encode("utf-8"))
    h.update(b"\x1f")
    h.update(body.encode("utf-8"))
    h.update(b"\x1f")
    h.update(prev_hash.encode("utf-8"))
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class LedgerEntry:
    seq: int
    inserted_at_utc: str
    insight_id: str
    canonical_json: str
    prev_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerificationResult:
    ok: bool
    n_entries: int
    broken_at_seq: Optional[int] = None
    reason: str = ""

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class HashChainLedger:
    """Append-only SQLite-backed hash chain.

    Use one instance per process. The internal lock serialises writes;
    readers (``get``, ``verify``) use short-lived connections so they
    don't block the writer.
    """

    SCHEMA_VERSION = 1

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._lock = threading.Lock()
        # Persistent connection only for the writer path. Readers open
        # their own short-lived connection so concurrent verification
        # doesn't lock out the appender.
        self._writer_conn = self._connect()
        self._init_schema(self._writer_conn)
        self._last_hash: str = self._compute_last_hash(self._writer_conn)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit; we manage txns explicitly
        )
        # WAL gives us reader concurrency. ":memory:" silently ignores it.
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger (
                seq             INTEGER PRIMARY KEY AUTOINCREMENT,
                inserted_at_utc TEXT    NOT NULL,
                insight_id      TEXT    NOT NULL,
                canonical_json  TEXT    NOT NULL,
                prev_hash       TEXT    NOT NULL,
                entry_hash      TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ledger_insight ON ledger(insight_id)"
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ledger_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "INSERT OR IGNORE INTO ledger_meta(key, value) VALUES (?, ?)",
            ("schema_version", str(self.SCHEMA_VERSION)),
        )

    def _compute_last_hash(self, conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT entry_hash FROM ledger ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row[0] if row else GENESIS_PREV_HASH

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        insight: Any,
        insight_id: Optional[str] = None,
    ) -> LedgerEntry:
        """Append one insight to the ledger and return its committed entry.

        ``insight`` may be a Pydantic v2 model (``model_dump(mode='json')``
        is called automatically) or any JSON-serialisable mapping.

        ``insight_id`` is required when the payload doesn't expose ``.id``
        or ``["id"]`` — callers should usually let it be auto-derived.
        """
        body_dict, derived_id = self._extract(insight)
        if insight_id is None:
            insight_id = derived_id
        if not insight_id:
            raise ValueError("insight_id is required (no .id field on payload)")

        canonical = canonical_json(body_dict)
        ts = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
            "+00:00", "Z"
        )

        with self._lock:
            cursor = self._writer_conn.execute(
                "SELECT COALESCE(MAX(seq), 0) FROM ledger"
            )
            seq = int(cursor.fetchone()[0]) + 1
            prev_hash = self._last_hash
            entry_hash = _hash_entry(seq, ts, canonical, prev_hash)

            self._writer_conn.execute(
                """
                INSERT INTO ledger
                  (seq, inserted_at_utc, insight_id, canonical_json,
                   prev_hash, entry_hash)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (seq, ts, insight_id, canonical, prev_hash, entry_hash),
            )
            self._last_hash = entry_hash

        return LedgerEntry(
            seq=seq,
            inserted_at_utc=ts,
            insight_id=insight_id,
            canonical_json=canonical,
            prev_hash=prev_hash,
            entry_hash=entry_hash,
        )

    @staticmethod
    def _extract(insight: Any) -> tuple[dict, str]:
        # Pydantic v2 model
        if hasattr(insight, "model_dump"):
            body = insight.model_dump(mode="json")
            return body, str(body.get("id", ""))
        if hasattr(insight, "dict"):  # pydantic v1 / custom
            body = insight.dict()
            return body, str(body.get("id", ""))
        if isinstance(insight, dict):
            return dict(insight), str(insight.get("id", ""))
        raise TypeError(
            f"unsupported insight type: {type(insight).__name__}; "
            "expected Pydantic v2 model or dict"
        )

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return int(
            self._writer_conn.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
        )

    @property
    def head_hash(self) -> str:
        return self._last_hash

    def get(self, seq: int) -> Optional[LedgerEntry]:
        row = self._writer_conn.execute(
            """
            SELECT seq, inserted_at_utc, insight_id, canonical_json,
                   prev_hash, entry_hash
              FROM ledger WHERE seq = ?
            """,
            (seq,),
        ).fetchone()
        if not row:
            return None
        return LedgerEntry(*row)

    def find_by_insight_id(self, insight_id: str) -> list[LedgerEntry]:
        rows = self._writer_conn.execute(
            """
            SELECT seq, inserted_at_utc, insight_id, canonical_json,
                   prev_hash, entry_hash
              FROM ledger WHERE insight_id = ?
              ORDER BY seq
            """,
            (insight_id,),
        ).fetchall()
        return [LedgerEntry(*r) for r in rows]

    def iter_entries(self) -> Iterable[LedgerEntry]:
        cursor = self._writer_conn.execute(
            """
            SELECT seq, inserted_at_utc, insight_id, canonical_json,
                   prev_hash, entry_hash
              FROM ledger ORDER BY seq
            """
        )
        for row in cursor:
            yield LedgerEntry(*row)

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify(self) -> VerificationResult:
        """Walk the chain and check every link.

        Returns ``ok=True`` only when:
        - ``seq`` is contiguous starting from 1
        - the first entry's ``prev_hash`` is the genesis sentinel
        - every entry's ``prev_hash`` equals the previous entry's
          ``entry_hash``
        - every recomputed ``entry_hash`` matches the stored value

        On the first violation, ``ok=False`` and ``broken_at_seq`` flag
        the offending row.
        """
        n = 0
        prev = GENESIS_PREV_HASH
        expected_seq = 1

        for e in self.iter_entries():
            n += 1
            if e.seq != expected_seq:
                return VerificationResult(
                    ok=False,
                    n_entries=n,
                    broken_at_seq=e.seq,
                    reason=f"non-contiguous seq: expected {expected_seq}, got {e.seq}",
                )
            if e.prev_hash != prev:
                return VerificationResult(
                    ok=False,
                    n_entries=n,
                    broken_at_seq=e.seq,
                    reason="prev_hash does not match previous entry's entry_hash",
                )
            recomputed = _hash_entry(
                e.seq, e.inserted_at_utc, e.canonical_json, e.prev_hash
            )
            if recomputed != e.entry_hash:
                return VerificationResult(
                    ok=False,
                    n_entries=n,
                    broken_at_seq=e.seq,
                    reason="entry_hash mismatch — body was tampered",
                )
            prev = e.entry_hash
            expected_seq += 1

        return VerificationResult(ok=True, n_entries=n)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._writer_conn.close()
            except Exception:  # pragma: no cover
                pass

    def __enter__(self):  # pragma: no cover
        return self

    def __exit__(self, exc_type, exc, tb):  # pragma: no cover
        self.close()
