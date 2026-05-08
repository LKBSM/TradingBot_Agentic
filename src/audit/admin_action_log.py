"""Admin action audit log — Sprint SECURITY-2B.1.

Why a dedicated log
-------------------
The DATA-2B.4 hash-chain ledger records *delivered insights* — the data
plane. The control plane (admin actions: create/revoke API key, run a
ledger verify, force a kill-switch reset, ...) needs its own append-only
record because:

- attribution: who flipped what, when?
- forensics: a tamper or unexpected outage starts with "what changed
  recently?" — the answer must be a single SQL query, not log greps;
- compliance: P29 / MiFID II finfluencer obligations require
  traceability of administrative changes to the service.

We deliberately keep this separate from the insight ledger:
- different threat model (admin = trusted but mistakes happen; insights
  = need cryptographic non-repudiation against external auditors),
- different retention (admin actions: 12 months; insights: 7 years for
  some jurisdictions),
- different read patterns (filter by actor / action; insights are
  walked sequentially).

Schema
------
Single SQLite WAL table ``admin_actions``:

  - id              INTEGER PRIMARY KEY AUTOINCREMENT
  - ts_utc          TEXT (ISO-8601 with 'Z')
  - actor           TEXT (admin identifier — e.g. last 4 of HMAC key id)
  - action          TEXT (verb: create_key, revoke_key, ...)
  - target          TEXT (object id the action operated on, or '-')
  - payload_digest  TEXT (16-hex SHA-256 prefix of the request body, or '-')
  - result          TEXT ('ok' or 'failed:<short reason>')
  - request_id      TEXT (correlates with the OBS-2B.3 access log)

We log the *digest* of the request body, never the raw body — admin
endpoints can mutate sensitive resources (rotate a webhook secret) and
storing the cleartext would leak it through the audit table itself.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def payload_digest(payload: Any) -> str:
    """16-hex SHA-256 prefix of a JSON-serialisable payload.

    Consistent with the audit-ledger ``canonical_json`` so a single
    request body produces the same digest in both layers.
    """
    if payload is None:
        return "-"
    if hasattr(payload, "model_dump"):
        payload = payload.model_dump(mode="json")
    elif hasattr(payload, "dict"):
        payload = payload.dict()
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdminActionRecord:
    id: int
    ts_utc: str
    actor: str
    action: str
    target: str
    payload_digest: str
    result: str
    request_id: str


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


class AdminActionLog:
    """Append-only SQLite store for admin actions.

    Designed for low write rate (a handful per day) — single writer
    connection, RLock around mutations, short-lived reader connections.
    """

    def __init__(self, db_path: str | Path = ":memory:"):
        self._db_path = str(db_path)
        self._lock = threading.RLock()
        self._writer_conn = self._connect()
        self._init_schema(self._writer_conn)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        try:
            conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_actions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts_utc          TEXT    NOT NULL,
                actor           TEXT    NOT NULL,
                action          TEXT    NOT NULL,
                target          TEXT    NOT NULL,
                payload_digest  TEXT    NOT NULL,
                result          TEXT    NOT NULL,
                request_id      TEXT    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_action ON admin_actions(action)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_admin_actor ON admin_actions(actor)"
        )

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def record(
        self,
        *,
        actor: str,
        action: str,
        target: str = "-",
        payload: Any = None,
        result: str = "ok",
        request_id: str = "-",
    ) -> AdminActionRecord:
        """Persist one admin action. Returns the committed record.

        ``payload`` may be a Pydantic model, dict, or None. Only its
        digest is stored; the cleartext never leaves memory.
        """
        if not actor:
            raise ValueError("actor is required")
        if not action:
            raise ValueError("action is required")
        ts = _now_iso()
        digest = payload_digest(payload)
        with self._lock:
            cur = self._writer_conn.execute(
                """
                INSERT INTO admin_actions
                  (ts_utc, actor, action, target, payload_digest, result, request_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, actor, action, target, digest, result, request_id),
            )
            row_id = int(cur.lastrowid)
        return AdminActionRecord(
            id=row_id,
            ts_utc=ts,
            actor=actor,
            action=action,
            target=target,
            payload_digest=digest,
            result=result,
            request_id=request_id,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return int(
            self._writer_conn.execute(
                "SELECT COUNT(*) FROM admin_actions"
            ).fetchone()[0]
        )

    def query(
        self,
        *,
        actor: Optional[str] = None,
        action: Optional[str] = None,
        since_iso: Optional[str] = None,
        limit: int = 100,
    ) -> list[AdminActionRecord]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        limit = min(limit, 1000)

        clauses: list[str] = []
        params: list[Any] = []
        if actor is not None:
            clauses.append("actor = ?")
            params.append(actor)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        if since_iso is not None:
            clauses.append("ts_utc >= ?")
            params.append(since_iso)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        sql = (
            "SELECT id, ts_utc, actor, action, target, payload_digest, result, "
            "request_id FROM admin_actions "
            f"{where} ORDER BY id DESC LIMIT ?"
        )
        rows = self._writer_conn.execute(sql, params).fetchall()
        return [AdminActionRecord(*r) for r in rows]

    def iter_records(self) -> Iterable[AdminActionRecord]:
        cursor = self._writer_conn.execute(
            "SELECT id, ts_utc, actor, action, target, payload_digest, result, "
            "request_id FROM admin_actions ORDER BY id"
        )
        for row in cursor:
            yield AdminActionRecord(*row)

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


__all__ = [
    "AdminActionLog",
    "AdminActionRecord",
    "payload_digest",
]
