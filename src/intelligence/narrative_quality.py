"""Narrative quality tracker — Sprint LLM-2B.7.

In-process recorder for narrative-generation outcomes: faithfulness
score, hallucination flags, cost per generation, review queue
candidates. Feeds a dashboard endpoint (added in the same sprint)
plus a manual-review queue.

Rolling window: last 7 days of generations (capped at 10 000 records
to bound memory). Each record carries:

    {generation_id, ts, prompt_template_id, prompt_version,
     faithfulness, answer_relevancy, context_precision, hallucination,
     cost_usd, language, latency_ms}

The dashboard endpoint exposes:

    GET /api/v1/metrics/narrative-quality
       → { rolling_7d: {n, faithfulness_p50, hallucination_rate,
                        cost_total_usd, latency_p95_ms},
           worst_5:    [...]   # bottom-5 by faithfulness for manual review
           by_language: {...} }
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


DEFAULT_WINDOW_SECONDS = 7 * 86400  # 7 days
DEFAULT_CAPACITY = 10_000


@dataclass
class NarrativeRecord:
    generation_id: str
    ts: float
    prompt_template_id: str = "-"
    prompt_version: int = 0
    faithfulness: float = 1.0       # [0, 1], 1 = fully faithful
    answer_relevancy: float = 1.0
    context_precision: float = 1.0
    hallucination: bool = False
    cost_usd: float = 0.0
    language: str = "fr"
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "generation_id": self.generation_id,
            "ts": self.ts,
            "prompt_template_id": self.prompt_template_id,
            "prompt_version": self.prompt_version,
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
            "context_precision": round(self.context_precision, 4),
            "hallucination": self.hallucination,
            "cost_usd": round(self.cost_usd, 6),
            "language": self.language,
            "latency_ms": round(self.latency_ms, 2),
        }


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return s[0]
    if q >= 1:
        return s[-1]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


class NarrativeQualityTracker:
    """Thread-safe rolling-window recorder + summariser."""

    def __init__(
        self,
        *,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        capacity: int = DEFAULT_CAPACITY,
        clock=time.time,
    ):
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._window = window_seconds
        self._clock = clock
        self._lock = threading.Lock()
        self._records: Deque[NarrativeRecord] = deque(maxlen=capacity)

    def record(self, rec: NarrativeRecord) -> None:
        with self._lock:
            self._records.append(rec)

    def _purge(self) -> None:
        cutoff = self._clock() - self._window
        # Deque doesn't support popleft-while-condition in O(n) elegantly,
        # but since we maxlen-cap we can afford a scan.
        while self._records and self._records[0].ts < cutoff:
            self._records.popleft()

    def summary(self) -> dict:
        with self._lock:
            self._purge()
            recs = list(self._records)

        if not recs:
            return {
                "rolling_window_days": round(self._window / 86400, 2),
                "n": 0,
                "faithfulness_p50": 0.0,
                "faithfulness_p10": 0.0,
                "hallucination_rate": 0.0,
                "cost_total_usd": 0.0,
                "latency_p95_ms": 0.0,
                "worst_5": [],
                "by_language": {},
            }

        faiths = [r.faithfulness for r in recs]
        latencies = [r.latency_ms for r in recs]
        n_halluc = sum(1 for r in recs if r.hallucination)
        total_cost = sum(r.cost_usd for r in recs)

        worst_5 = sorted(recs, key=lambda r: r.faithfulness)[:5]

        by_lang: dict[str, dict] = {}
        for r in recs:
            slot = by_lang.setdefault(
                r.language, {"n": 0, "faith_sum": 0.0, "halluc": 0}
            )
            slot["n"] += 1
            slot["faith_sum"] += r.faithfulness
            slot["halluc"] += 1 if r.hallucination else 0
        by_lang_out = {
            lang: {
                "n": v["n"],
                "faithfulness_mean": round(v["faith_sum"] / v["n"], 4),
                "hallucination_rate": round(v["halluc"] / v["n"], 4),
            }
            for lang, v in by_lang.items()
        }

        return {
            "rolling_window_days": round(self._window / 86400, 2),
            "n": len(recs),
            "faithfulness_p50": round(_percentile(faiths, 0.5), 4),
            "faithfulness_p10": round(_percentile(faiths, 0.1), 4),
            "hallucination_rate": round(n_halluc / len(recs), 4),
            "cost_total_usd": round(total_cost, 4),
            "latency_p95_ms": round(_percentile(latencies, 0.95), 2),
            "worst_5": [r.to_dict() for r in worst_5],
            "by_language": by_lang_out,
        }

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)


__all__ = [
    "DEFAULT_CAPACITY",
    "DEFAULT_WINDOW_SECONDS",
    "NarrativeQualityTracker",
    "NarrativeRecord",
]
