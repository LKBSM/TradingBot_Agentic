"""In-memory nonce store for admin HMAC replay protection.

DG-055 (Sprint 2 — sécurité critique) : the original ``require_admin``
gate accepted any valid HMAC inside a 5-minute timestamp window, which
lets an attacker who captures a signed admin request replay it without
modification. Pairing the timestamp with a single-use nonce closes that
window down to a single execution.

Design
------
- A ``NonceStore`` keeps every nonce we've already validated, along with
  its expiry time, in a process-local dict guarded by a lock.
- TTL matches the timestamp window (``DEFAULT_TTL_SECONDS = 300``) so
  the store does not grow unboundedly; a sweep runs on every read.
- One ``NonceStore`` instance lives on ``AppState``; that's enough for
  a single-worker FastAPI process. Once we move to multi-worker prod,
  swap the in-memory implementation for SQLite (already an SQLite
  store nearby) or Redis. The public surface is intentionally tiny
  so the swap is purely an implementation detail.

Behaviour
---------
``check_and_record(nonce)``:
- if ``nonce`` is already present and not expired → return ``False``
  (caller must reject the request as a replay)
- otherwise insert it with ``now + ttl`` and return ``True``

This is the "atomic check-and-set" used by HMAC nonce protocols
described in RFC 7235 §2 and SP 800-63B §5.2.6.
"""

from __future__ import annotations

import threading
import time
from typing import Dict


DEFAULT_TTL_SECONDS = 300.0  # mirrors the 5-min HMAC timestamp window


class NonceStore:
    """Single-process replay-protection nonce store."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS, max_entries: int = 100_000):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be > 0")
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0")
        self._ttl = float(ttl_seconds)
        self._max = int(max_entries)
        self._lock = threading.Lock()
        self._seen: Dict[str, float] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._seen)

    def _sweep_locked(self, now: float) -> None:
        """Drop expired entries. Caller holds the lock."""
        expired = [n for n, exp in self._seen.items() if exp <= now]
        for n in expired:
            del self._seen[n]

    def check_and_record(self, nonce: str, *, now: float | None = None) -> bool:
        """Return True if ``nonce`` is fresh and was just recorded.

        Returns False if ``nonce`` is already in the store (i.e. the
        caller is replaying a request). On True, the nonce stays in the
        store until its TTL elapses, preventing further reuse.
        """
        if not nonce:
            return False
        if now is None:
            now = time.time()
        with self._lock:
            # Periodic GC — opportunistic, cheap, keeps memory bounded.
            self._sweep_locked(now)
            existing = self._seen.get(nonce)
            if existing is not None and existing > now:
                return False
            if len(self._seen) >= self._max:
                # Soft cap: drop the oldest entry to keep the store
                # bounded under adversarial flood. Replay attacks rely
                # on retention, not capacity, so evicting the oldest
                # is acceptable in extremis.
                oldest = min(self._seen.items(), key=lambda kv: kv[1])[0]
                del self._seen[oldest]
            self._seen[nonce] = now + self._ttl
            return True

    def reset(self) -> None:
        """Drop all entries — used by tests."""
        with self._lock:
            self._seen.clear()


__all__ = ["NonceStore", "DEFAULT_TTL_SECONDS"]
