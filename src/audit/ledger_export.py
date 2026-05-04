"""Streaming CSV + JSONL export of the audit ledger — Sprint DATA-2B.6.

Use cases
---------
- Compliance archive: monthly export of the hash chain to immutable
  cold storage (S3 with object-lock, or paper).
- Offline reconciliation: broker downloads their slice of the ledger
  by seq range or date range and runs their own SHA-256 verification.
- Regulator request: produce a tamper-evident record of every insight
  delivered in a window without dumping the live SQLite file.

Streaming
---------
The exporters take the iterable of ``LedgerEntry`` from the live
``HashChainLedger`` and yield rows one at a time, so a 10M-row chain
exports without blowing memory.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime, timezone
from typing import Iterable, Iterator, Optional

from src.audit.hash_chain_ledger import HashChainLedger, LedgerEntry


CSV_COLUMNS = (
    "seq",
    "inserted_at_utc",
    "insight_id",
    "prev_hash",
    "entry_hash",
    "canonical_json",
)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def _filter_entries(
    entries: Iterable[LedgerEntry],
    *,
    min_seq: Optional[int] = None,
    max_seq: Optional[int] = None,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
) -> Iterator[LedgerEntry]:
    """Yield entries within the seq + timestamp window."""
    since = _parse_iso(since_iso) if since_iso else None
    until = _parse_iso(until_iso) if until_iso else None
    for e in entries:
        if min_seq is not None and e.seq < min_seq:
            continue
        if max_seq is not None and e.seq > max_seq:
            continue
        if since is not None or until is not None:
            ts = _parse_iso(e.inserted_at_utc)
            if since is not None and ts < since:
                continue
            if until is not None and ts > until:
                continue
        yield e


def _parse_iso(s: str) -> datetime:
    """Parse the ledger's ISO-8601 timestamps (with trailing 'Z' or +00:00).

    Tolerant of microsecond precision so the same parser handles both
    user-supplied filter bounds and the ledger's own stored format.
    """
    s = s.strip()
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# CSV
# ---------------------------------------------------------------------------


def to_csv(
    ledger_or_entries,
    *,
    min_seq: Optional[int] = None,
    max_seq: Optional[int] = None,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
) -> Iterator[str]:
    """Yield CSV lines (header first) for the filtered ledger range.

    Accepts either a ``HashChainLedger`` (uses ``iter_entries``) or any
    iterable of ``LedgerEntry``. Yields strings (no trailing newline
    delegated to the consumer / a writer in ``"".join`` mode).
    """
    entries = (
        ledger_or_entries.iter_entries()
        if isinstance(ledger_or_entries, HashChainLedger)
        else ledger_or_entries
    )
    filtered = _filter_entries(
        entries,
        min_seq=min_seq,
        max_seq=max_seq,
        since_iso=since_iso,
        until_iso=until_iso,
    )

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(CSV_COLUMNS)
    yield buf.getvalue()
    buf.seek(0)
    buf.truncate()

    for entry in filtered:
        writer.writerow([getattr(entry, col) for col in CSV_COLUMNS])
        yield buf.getvalue()
        buf.seek(0)
        buf.truncate()


def to_csv_string(*args, **kwargs) -> str:
    """Eager wrapper around :func:`to_csv` for callers that don't stream."""
    return "".join(to_csv(*args, **kwargs))


# ---------------------------------------------------------------------------
# JSONL
# ---------------------------------------------------------------------------


def to_jsonl(
    ledger_or_entries,
    *,
    min_seq: Optional[int] = None,
    max_seq: Optional[int] = None,
    since_iso: Optional[str] = None,
    until_iso: Optional[str] = None,
) -> Iterator[str]:
    """Yield one JSON object per line, NDJSON-compatible."""
    entries = (
        ledger_or_entries.iter_entries()
        if isinstance(ledger_or_entries, HashChainLedger)
        else ledger_or_entries
    )
    filtered = _filter_entries(
        entries,
        min_seq=min_seq,
        max_seq=max_seq,
        since_iso=since_iso,
        until_iso=until_iso,
    )
    for entry in filtered:
        yield (
            json.dumps(
                {
                    "seq": entry.seq,
                    "inserted_at_utc": entry.inserted_at_utc,
                    "insight_id": entry.insight_id,
                    "prev_hash": entry.prev_hash,
                    "entry_hash": entry.entry_hash,
                    "canonical_json": entry.canonical_json,
                },
                ensure_ascii=False,
            )
            + "\n"
        )


def to_jsonl_string(*args, **kwargs) -> str:
    return "".join(to_jsonl(*args, **kwargs))


__all__ = [
    "CSV_COLUMNS",
    "to_csv",
    "to_csv_string",
    "to_jsonl",
    "to_jsonl_string",
]
