"""Forward-test transparency commitment log — Sprint RISK-2B.1.

Sofia's discipline mechanism: every week (or every snapshot), the
paper-trading equity curve is hashed (SHA256) and the hash appended
to an append-only log. The hash log is public — anyone can verify the
curve at time T1 still matches the hash we committed to at T0.

Why hashes, not the raw curves
------------------------------
The raw curve is large and changes every bar. The hash is 64 chars,
trivially auditable, and gives the same anti-massage guarantee:
flipping a single trade's PnL post-hoc would change the hash. If
someone challenges "you doctored the equity curve", we point them at
the historical hash log and show their alleged "tampered" equity
hash doesn't match.

Append-only file ``data/risk/transparency_log.jsonl`` — one line per
commitment:
    {ts_utc, equity_curve_hash, n_trades, equity_R, prev_log_hash}

The ``prev_log_hash`` field hash-chains entries so a single deletion is
detectable (same trick as the audit ledger). Genesis prev_hash is
sixty-four zeros.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


GENESIS_PREV_HASH = "0" * 64


def _hash_equity_payload(payload: dict) -> str:
    """Canonical-JSON SHA256 of the publication payload."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TransparencyEntry:
    seq: int
    ts_utc: str
    equity_curve_hash: str
    n_trades: int
    equity_R: float
    prev_log_hash: str
    entry_hash: str

    def to_dict(self) -> dict:
        return {
            "seq": self.seq,
            "ts_utc": self.ts_utc,
            "equity_curve_hash": self.equity_curve_hash,
            "n_trades": self.n_trades,
            "equity_R": self.equity_R,
            "prev_log_hash": self.prev_log_hash,
            "entry_hash": self.entry_hash,
        }


class TransparencyLog:
    """Append-only chained log of forward-test commitment hashes."""

    def __init__(self, log_path: str | Path = "data/risk/transparency_log.jsonl"):
        self._path = Path(log_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def commit(self, publication_payload: dict) -> TransparencyEntry:
        """Append a new commitment for the current publication state.

        ``publication_payload`` is whatever the webapp shows publicly —
        usually the output of ``PaperTradingHarness.export_for_publication()``.
        We hash *that* exact object, so any future edit to the displayed
        curve is detectable.
        """
        curve_hash = _hash_equity_payload(publication_payload)
        stats = publication_payload.get("stats", {})
        n_trades = int(stats.get("n_trades", 0))
        equity_R = float(stats.get("total_R", 0.0))

        with self._lock:
            existing = self._read_all_unlocked()
            seq = (existing[-1].seq + 1) if existing else 1
            prev = existing[-1].entry_hash if existing else GENESIS_PREV_HASH
            ts = datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
                "+00:00", "Z"
            )
            entry_hash = hashlib.sha256(
                f"{seq}|{ts}|{curve_hash}|{prev}".encode("utf-8")
            ).hexdigest()

            entry = TransparencyEntry(
                seq=seq,
                ts_utc=ts,
                equity_curve_hash=curve_hash,
                n_trades=n_trades,
                equity_R=equity_R,
                prev_log_hash=prev,
                entry_hash=entry_hash,
            )
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
            return entry

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def entries(self) -> list[TransparencyEntry]:
        with self._lock:
            return self._read_all_unlocked()

    def _read_all_unlocked(self) -> list[TransparencyEntry]:
        if not self._path.is_file():
            return []
        out: list[TransparencyEntry] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                out.append(TransparencyEntry(**d))
        return out

    def verify(self) -> tuple[bool, Optional[int], str]:
        """Walk the chain and return (ok, broken_at_seq, reason)."""
        entries = self.entries()
        prev = GENESIS_PREV_HASH
        expected_seq = 1
        for e in entries:
            if e.seq != expected_seq:
                return False, e.seq, f"non-contiguous seq: expected {expected_seq}, got {e.seq}"
            if e.prev_log_hash != prev:
                return False, e.seq, "prev_log_hash does not match previous entry_hash"
            recomputed = hashlib.sha256(
                f"{e.seq}|{e.ts_utc}|{e.equity_curve_hash}|{e.prev_log_hash}".encode("utf-8")
            ).hexdigest()
            if recomputed != e.entry_hash:
                return False, e.seq, "entry_hash mismatch — tampered"
            prev = e.entry_hash
            expected_seq += 1
        return True, None, "OK"

    @property
    def size(self) -> int:
        return len(self.entries())


__all__ = [
    "GENESIS_PREV_HASH",
    "TransparencyEntry",
    "TransparencyLog",
]
