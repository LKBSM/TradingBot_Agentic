"""Idempotency store for B2B mutation endpoints — Sprint API-2B.2.

Background
----------
B2B brokers retry on the slightest network error. Without idempotency:

- a retry of POST /enrich creates a *second* InsightSignalV2 with a fresh
  audit-ledger seq, which invalidates the broker's "one-receipt-per-
  signal" reconciliation invariant;
- a retry of any future POST /webhook fan-out delivers the same payload
  twice to every subscriber.

Stripe-style idempotency: client sends ``Idempotency-Key: <opaque>``;
server stores ``(api_key_id, key) → (body_hash, response, ts)`` in a
24h window. Subsequent requests with the same key:

- if body_hash matches the stored one ⇒ replay the original response
  (HIT)
- if body_hash differs ⇒ 409 Conflict (CLASH) — the client reused the
  key for a different payload, which is a contract violation.

Storage
-------
In-process dict + RLock, with lazy TTL purge on read. A single deploy
sees a few thousand keys/day at most so this is fine. Production-grade
SaaS would persist to Redis; we'll swap the implementation when a
deployment actually needs the distributed property.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional


DEFAULT_TTL_SECONDS = 24 * 3600  # 24h, matches Stripe's window


@dataclass(frozen=True)
class StoredResponse:
    body_hash: str
    response: dict
    stored_at: float
    expires_at: float


class IdempotencyResult:
    """Lookup outcome — three discrete cases."""

    MISS = "miss"
    HIT = "hit"
    CLASH = "clash"


@dataclass
class IdempotencyLookup:
    status: str
    response: Optional[dict] = None
    stored_at: Optional[float] = None


def _hash_body(body: Any) -> str:
    """Stable SHA-256 over a request body for clash detection.

    Pydantic models / dataclasses get model_dump'd to a dict first so
    field ordering can't change the hash. The canonical encoding mirrors
    ``src.audit.hash_chain_ledger.canonical_json``.
    """
    if hasattr(body, "model_dump"):
        body = body.model_dump(mode="json")
    elif hasattr(body, "dict"):
        body = body.dict()
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class IdempotencyStore:
    """Thread-safe in-memory idempotency store with TTL.

    The compound key is ``(api_key_id, idempotency_key)`` so two
    different subscribers can use the same opaque key value without
    colliding.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = DEFAULT_TTL_SECONDS,
        max_entries: int = 100_000,
        clock: Any = time.time,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be positive")
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._clock = clock
        self._lock = threading.RLock()
        self._store: dict[tuple[str, str], StoredResponse] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _purge_expired(self, now: float) -> None:
        # Caller holds the lock.
        expired = [k for k, v in self._store.items() if v.expires_at <= now]
        for k in expired:
            del self._store[k]

    def _enforce_capacity(self) -> None:
        # Caller holds the lock. Naive evict-oldest when over budget.
        if len(self._store) <= self._max_entries:
            return
        # Sort by stored_at ascending and drop the oldest 10%.
        n_to_drop = max(1, len(self._store) - self._max_entries)
        ordered = sorted(self._store.items(), key=lambda kv: kv[1].stored_at)
        for key, _ in ordered[:n_to_drop]:
            del self._store[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def lookup(
        self, api_key_id: str, idempotency_key: str, body_hash: str
    ) -> IdempotencyLookup:
        """Check for an existing response under ``(api_key_id, key)``.

        Returns an ``IdempotencyLookup`` with one of:
        - ``MISS``  — no entry; caller proceeds to process the request
        - ``HIT``   — entry exists and body matches; replay
        - ``CLASH`` — entry exists but body differs; 409
        """
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            stored = self._store.get((api_key_id, idempotency_key))
            if stored is None:
                return IdempotencyLookup(status=IdempotencyResult.MISS)
            if stored.body_hash != body_hash:
                return IdempotencyLookup(
                    status=IdempotencyResult.CLASH,
                    stored_at=stored.stored_at,
                )
            return IdempotencyLookup(
                status=IdempotencyResult.HIT,
                response=dict(stored.response),
                stored_at=stored.stored_at,
            )

    def store(
        self,
        api_key_id: str,
        idempotency_key: str,
        body_hash: str,
        response: dict,
    ) -> None:
        if not idempotency_key:
            raise ValueError("idempotency_key is required")
        with self._lock:
            now = self._clock()
            self._purge_expired(now)
            self._store[(api_key_id, idempotency_key)] = StoredResponse(
                body_hash=body_hash,
                response=dict(response),  # defensive copy
                stored_at=now,
                expires_at=now + self._ttl,
            )
            self._enforce_capacity()

    @property
    def size(self) -> int:
        with self._lock:
            self._purge_expired(self._clock())
            return len(self._store)

    def purge(self) -> int:
        """Force a sweep of expired entries. Returns count removed."""
        with self._lock:
            before = len(self._store)
            self._purge_expired(self._clock())
            return before - len(self._store)


__all__ = [
    "DEFAULT_TTL_SECONDS",
    "IdempotencyLookup",
    "IdempotencyResult",
    "IdempotencyStore",
    "StoredResponse",
    "_hash_body",
]
