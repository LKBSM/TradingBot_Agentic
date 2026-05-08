"""Per-tier sliding-window rate limiter — Sprint INFRA-2B.9.

Sits alongside :class:`CostQuotaEnforcer` (USD cap) but enforces a
*request rate* cap, which is the right knob to defend the LLM API
backends against thundering-herd / scrape attacks even when each call
is below the cost cap.

Window
------
Sliding 60-second window per ``(api_key, tier)`` key. Implementation
mirrors ``src/intelligence/security.RateLimiter`` (existing per-IP
limiter) but takes the cap from a per-tier table and is keyed by the
authenticated subscriber rather than the source IP.

Why not reuse the existing per-IP limiter
-----------------------------------------
The per-IP one is shared by every endpoint and uses a single cap. Two
distinct concerns:

- *Per-IP* protects against a single network source flooding the
  service (denial-of-service shape).
- *Per-tier* protects against a legitimate API key going over its
  contractual rate (commercial-fairness shape).

A subscriber on ``INSTITUTIONAL`` may legitimately spread 1000 req/min
across several IPs (cluster, edge functions); the per-tier limiter
gives them that headroom while the per-IP one continues to catch
true abuse from a single host.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Optional


DEFAULT_PER_MINUTE_CAPS: dict[str, int] = {
    "FREE": 5,
    "ANALYST": 30,
    "STRATEGIST": 200,
    "INSTITUTIONAL": 1000,
    "unknown": 10,
}

DEFAULT_WINDOW_SECONDS = 60.0


@dataclass
class RateLimitSnapshot:
    api_key: str
    tier: str
    used: int
    cap: int
    remaining: int
    reset_in_seconds: float

    @property
    def is_throttled(self) -> bool:
        return self.remaining <= 0


class TierRateLimiter:
    """Sliding-window rate limiter keyed by (api_key, tier).

    Public surface:
    - ``allow(api_key, tier) -> bool`` : check + record atomically.
    - ``snapshot(api_key, tier)``      : read-only view for headers
      like ``X-RateLimit-Remaining``.
    - ``set_cap(tier, n)``             : runtime tuning.
    - ``reset(api_key=None)``          : test-only.
    """

    def __init__(
        self,
        caps_per_minute: Optional[dict[str, int]] = None,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
    ):
        self._caps = (
            dict(caps_per_minute)
            if caps_per_minute is not None
            else dict(DEFAULT_PER_MINUTE_CAPS)
        )
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = window_seconds
        # Per-(api_key, tier) deque of timestamps.
        self._buckets: dict[tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cap_for(self, tier: str) -> int:
        return self._caps.get(tier, self._caps.get("unknown", 0))

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _purge(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self._window
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allow(self, api_key: str, tier: str) -> bool:
        """Atomic: purge expired entries, check cap, record on success."""
        if not api_key:
            raise ValueError("api_key is required")
        cap = self._cap_for(tier)
        if cap <= 0:
            return False
        key = (api_key, tier)
        with self._lock:
            bucket = self._buckets[key]
            now = self._now()
            self._purge(bucket, now)
            if len(bucket) >= cap:
                return False
            bucket.append(now)
            return True

    def snapshot(self, api_key: str, tier: str) -> RateLimitSnapshot:
        cap = self._cap_for(tier)
        key = (api_key, tier)
        with self._lock:
            bucket = self._buckets[key]
            now = self._now()
            self._purge(bucket, now)
            used = len(bucket)
            reset_in = (
                max(0.0, bucket[0] + self._window - now) if bucket else 0.0
            )
        return RateLimitSnapshot(
            api_key=api_key,
            tier=tier,
            used=used,
            cap=cap,
            remaining=max(0, cap - used),
            reset_in_seconds=round(reset_in, 3),
        )

    def set_cap(self, tier: str, per_minute: int) -> None:
        if per_minute < 0:
            raise ValueError("per_minute must be >= 0")
        self._caps[tier] = per_minute

    def reset(self, api_key: Optional[str] = None) -> None:
        with self._lock:
            if api_key is None:
                self._buckets.clear()
                return
            for key in list(self._buckets):
                if key[0] == api_key:
                    del self._buckets[key]


__all__ = [
    "DEFAULT_PER_MINUTE_CAPS",
    "DEFAULT_WINDOW_SECONDS",
    "RateLimitSnapshot",
    "TierRateLimiter",
]
