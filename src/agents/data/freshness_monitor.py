"""Data-feed freshness monitor + SLA tracking — Sprint DATA-2B.5.

Tracks the wall-clock age of each data source (news, sentiment, macro,
prices) and exposes whether each source is currently inside its
published SLA. The Phase 2B CGU advertises:

- news       lag < 30  min
- sentiment  lag < 45  min
- macro      lag < 26  h
- prices     lag < 5   min (intraday) / 36 h (daily)

When any feed slips, we want (a) the SLA dashboard on the webapp to
show "amber/red" rather than silently lying, (b) Sentry to fire, and
(c) the B2B audit trail to record the breach so a broker SLA claim is
defensible against our own logs.

This module is the in-process recorder: every data provider calls
``recorder.report(source_name, ts)`` after each successful pull, and
the monitor evaluates compliance against a static SLA table.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class SLASpec:
    """Per-source SLA. ``max_lag_seconds`` is the cap before we breach."""
    max_lag_seconds: float
    description: str = ""


DEFAULT_SLAS: Dict[str, SLASpec] = {
    "news":       SLASpec(max_lag_seconds=30 * 60,       description="< 30min"),
    "sentiment":  SLASpec(max_lag_seconds=45 * 60,       description="< 45min"),
    "macro":      SLASpec(max_lag_seconds=26 * 3600,     description="< 26h"),
    "prices_m15": SLASpec(max_lag_seconds=5 * 60,        description="< 5min"),
    "prices_d1":  SLASpec(max_lag_seconds=36 * 3600,     description="< 36h"),
    "cot":        SLASpec(max_lag_seconds=8 * 86400,     description="< 8 days"),
    "fred":       SLASpec(max_lag_seconds=26 * 3600,     description="< 26h"),
}


@dataclass(frozen=True)
class FreshnessReport:
    source: str
    last_seen_ts: Optional[float]
    age_seconds: Optional[float]
    sla_seconds: float
    in_sla: bool
    description: str


class FreshnessMonitor:
    """Thread-safe in-process feed-age tracker.

    Every successful data pull should call ``report(source, ts=time.time())``.
    The webapp + Sentry hook periodically call ``snapshot()`` and a
    breach (any ``in_sla=False``) triggers an alert.
    """

    def __init__(
        self,
        slas: Optional[Dict[str, SLASpec]] = None,
        *,
        clock=time.time,
    ):
        self._slas = dict(slas) if slas is not None else dict(DEFAULT_SLAS)
        self._clock = clock
        self._lock = threading.Lock()
        self._last_seen: Dict[str, float] = {}

    def report(self, source: str, *, ts: Optional[float] = None) -> None:
        """Record that ``source`` was successfully refreshed at ``ts``.

        Unknown sources are accepted (we record their freshness) — they
        get the implicit ``unknown`` SLA when read back. That lets new
        providers light up the dashboard without code change before
        the SLA table is updated.
        """
        if not source:
            raise ValueError("source name is required")
        with self._lock:
            self._last_seen[source] = ts if ts is not None else self._clock()

    def snapshot(self, source: str) -> FreshnessReport:
        with self._lock:
            last = self._last_seen.get(source)
        sla = self._slas.get(source)
        sla_seconds = sla.max_lag_seconds if sla is not None else float("inf")
        desc = sla.description if sla is not None else "no SLA"
        if last is None:
            return FreshnessReport(
                source=source,
                last_seen_ts=None,
                age_seconds=None,
                sla_seconds=sla_seconds,
                in_sla=False,  # never seen ⇒ breach
                description=desc,
            )
        age = self._clock() - last
        return FreshnessReport(
            source=source,
            last_seen_ts=last,
            age_seconds=round(age, 2),
            sla_seconds=sla_seconds,
            in_sla=age <= sla_seconds,
            description=desc,
        )

    def snapshot_all(self) -> list[FreshnessReport]:
        with self._lock:
            sources = list(self._last_seen.keys() | self._slas.keys())
        return [self.snapshot(s) for s in sorted(sources)]

    def any_breach(self) -> bool:
        return any(not r.in_sla for r in self.snapshot_all())

    def add_sla(self, source: str, spec: SLASpec) -> None:
        with self._lock:
            self._slas[source] = spec


__all__ = [
    "DEFAULT_SLAS",
    "FreshnessMonitor",
    "FreshnessReport",
    "SLASpec",
]
