"""Resilient SQLite journal configuration (incident 2026-08-05).

Every store opened WAL with a bare ``conn.execute("PRAGMA journal_mode=WAL")``.
WAL needs a ``-wal``/``-shm`` sidecar and shared-memory (mmap) support; on a disk
that doesn't provide it, a stale/corrupt ``-shm`` left by an unclean container
shutdown, or a full disk, that PRAGMA raises ``sqlite3.OperationalError:
disk I/O error`` — and since it ran at import/boot in every store, it crashed the
whole API on start (``Exited with status 1``).

``apply_wal`` keeps WAL's concurrency benefit when the disk supports it, but never
lets a WAL-hostile disk take the process down: on failure it falls back to the
DELETE rollback journal (no sidecar files, works on any filesystem) so the
service still boots. A genuinely failing disk (full/read-only) then surfaces on
the first real write — a localized, diagnosable error — instead of a total,
silent boot crash.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)


def apply_wal(conn: sqlite3.Connection) -> str:
    """Set WAL journal mode, falling back to DELETE if the disk can't do WAL.

    Returns the journal mode actually in effect ("wal" / "delete" / "unknown").
    Never raises: journal configuration must not be able to crash boot.
    """
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        return "wal"
    except sqlite3.OperationalError as exc:
        logger.warning(
            "SQLite WAL unavailable (%s) — falling back to DELETE journal; "
            "check the disk (full / stale .db-wal/.db-shm sidecar / no WAL support)",
            exc,
        )
        try:
            conn.execute("PRAGMA journal_mode=DELETE")
            return "delete"
        except sqlite3.OperationalError as exc2:
            logger.error(
                "SQLite journal PRAGMA failed even for DELETE (%s) — the disk is "
                "likely full or read-only; writes will fail until it is fixed",
                exc2,
            )
            return "unknown"


__all__ = ["apply_wal"]
