"""Per-tier daily cost quota enforcement — Sprint INFRA-2B.7.

Why this exists
---------------
The CostTracker (LLM-2B.8) accumulates spend, but doesn't push back. A
free-tier user calling /qa in a tight loop would burn USD until manual
intervention. The quota gate sits in front of every LLM call and
short-circuits with a typed exception once the per-tier daily ceiling
is reached.

Defaults reflect the M3 unit-economics (eval 24): FREE must cost
< €0.05/day to keep marketing-acquisition cost recoverable on
conversion. INSTITUTIONAL is 1000× higher because LLM cost is
irrelevant compared to the contract value.

The quota is "rolling 24 hours" rather than a calendar day so you
can't reset by waiting until midnight UTC. Implementation: a deque of
(timestamp, usd) records per tier, anything older than 24h is dropped
on every check.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque


# Daily caps per tier — USD over a rolling 24-hour window.
# Aligned with kill_criteria_board: LLM cost / revenue < 40% green.
DEFAULT_DAILY_CAPS_USD: dict[str, float] = {
    "FREE": 0.05,
    "ANALYST": 0.50,
    "STRATEGIST": 5.00,
    "INSTITUTIONAL": 50.00,
    "unknown": 0.10,
}


WINDOW_SECONDS = 24 * 60 * 60


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class QuotaExceeded(Exception):
    """Raised when a tier's daily cost window is full."""

    def __init__(self, tier: str, used: float, cap: float):
        super().__init__(
            f"daily quota exceeded for tier {tier!r}: "
            f"${used:.4f} used vs ${cap:.4f} cap"
        )
        self.tier = tier
        self.used_usd = used
        self.cap_usd = cap


# ---------------------------------------------------------------------------
# Quota state
# ---------------------------------------------------------------------------


@dataclass
class QuotaSnapshot:
    tier: str
    used_usd: float
    cap_usd: float
    remaining_usd: float
    window_seconds: int = WINDOW_SECONDS

    @property
    def utilisation(self) -> float:
        return self.used_usd / self.cap_usd if self.cap_usd > 0 else 0.0


@dataclass
class _TierBucket:
    cap_usd: float
    spend: Deque[tuple[float, float]] = field(default_factory=deque)
    # Lock per tier — concurrent /qa from different users on the same tier
    # contend on this lock for a few microseconds. Acceptable.
    lock: threading.Lock = field(default_factory=threading.Lock)


class CostQuotaEnforcer:
    """Rolling-24h USD cap per tier.

    Check-then-record pattern:

        if not quota.allow(tier, estimated_usd):
            raise QuotaExceeded(...)
        run_llm_call()
        quota.record(tier, actual_usd)

    For most callers ``check_and_record(tier, estimated_usd)`` is the
    right one-shot helper.
    """

    def __init__(
        self,
        caps: dict[str, float] | None = None,
        *,
        window_seconds: int = WINDOW_SECONDS,
    ):
        self._caps = dict(caps) if caps is not None else dict(DEFAULT_DAILY_CAPS_USD)
        self._window = window_seconds
        self._buckets: dict[str, _TierBucket] = {}
        self._buckets_lock = threading.Lock()

    def _now(self) -> float:
        return time.monotonic()

    def _bucket(self, tier: str) -> _TierBucket:
        with self._buckets_lock:
            b = self._buckets.get(tier)
            if b is None:
                cap = self._caps.get(tier, self._caps.get("unknown", 0.0))
                b = _TierBucket(cap_usd=cap)
                self._buckets[tier] = b
            return b

    def _purge_expired(self, bucket: _TierBucket, now: float) -> None:
        cutoff = now - self._window
        while bucket.spend and bucket.spend[0][0] < cutoff:
            bucket.spend.popleft()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def used(self, tier: str) -> float:
        """Current rolling-24h USD spend for ``tier``."""
        bucket = self._bucket(tier)
        with bucket.lock:
            self._purge_expired(bucket, self._now())
            return sum(usd for _, usd in bucket.spend)

    def snapshot(self, tier: str) -> QuotaSnapshot:
        bucket = self._bucket(tier)
        with bucket.lock:
            self._purge_expired(bucket, self._now())
            used = sum(usd for _, usd in bucket.spend)
            cap = bucket.cap_usd
            return QuotaSnapshot(
                tier=tier,
                used_usd=used,
                cap_usd=cap,
                remaining_usd=max(0.0, cap - used),
                window_seconds=self._window,
            )

    def allow(self, tier: str, estimated_usd: float) -> bool:
        """Return True if a call estimated at ``estimated_usd`` would
        keep the tier under cap. Does NOT record — caller still has to
        invoke ``record`` after the actual cost is known.
        """
        bucket = self._bucket(tier)
        with bucket.lock:
            self._purge_expired(bucket, self._now())
            used = sum(usd for _, usd in bucket.spend)
            return (used + estimated_usd) <= bucket.cap_usd

    def record(self, tier: str, usd: float) -> None:
        """Add an actual spend entry. Negative values are a programming
        error and rejected (we don't refund quota mid-window — refunds
        would let abusers churn near the cap)."""
        if usd < 0:
            raise ValueError("usd must be >= 0")
        if usd == 0:
            return
        bucket = self._bucket(tier)
        with bucket.lock:
            bucket.spend.append((self._now(), usd))

    def check_and_record(self, tier: str, usd: float) -> None:
        """Atomic check + record. Raises ``QuotaExceeded`` when over.

        Pre-bills the call at ``usd``. If you don't know the exact cost
        ahead of time (typical for LLM streaming), pass an estimate;
        call ``record`` again with the delta after the actual cost is
        observed. Or just always pre-bill at a slightly conservative
        upper bound.
        """
        bucket = self._bucket(tier)
        with bucket.lock:
            self._purge_expired(bucket, self._now())
            used = sum(s for _, s in bucket.spend)
            if used + usd > bucket.cap_usd:
                raise QuotaExceeded(tier, used + usd, bucket.cap_usd)
            bucket.spend.append((self._now(), usd))

    def set_cap(self, tier: str, cap_usd: float) -> None:
        if cap_usd < 0:
            raise ValueError("cap_usd must be >= 0")
        self._caps[tier] = cap_usd
        # Re-prime the bucket so the new cap takes effect immediately.
        bucket = self._bucket(tier)
        with bucket.lock:
            bucket.cap_usd = cap_usd

    def reset(self, tier: str | None = None) -> None:
        """Clear spend for ``tier`` (or all tiers when None). Test-only."""
        if tier is None:
            with self._buckets_lock:
                for b in self._buckets.values():
                    with b.lock:
                        b.spend.clear()
            return
        bucket = self._bucket(tier)
        with bucket.lock:
            bucket.spend.clear()


__all__ = [
    "DEFAULT_DAILY_CAPS_USD",
    "WINDOW_SECONDS",
    "CostQuotaEnforcer",
    "QuotaExceeded",
    "QuotaSnapshot",
]
