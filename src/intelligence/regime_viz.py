"""Vol-regime visualisation data prep — Sprint REGIME-2B.2.

Generates the JSON payload the webapp's `RegimeTimeline` component
consumes. The React component itself ships in
``webapp/components/RegimeTimeline.tsx``; this module is the backend
data-prep + endpoint surface.

Input: an OHLCV DataFrame + a fitted ``RegimeClassifier`` from
REGIME-2B.1. Output: per-bar ``{ts, label, confidence}`` triples plus
a pre-aggregated count summary by label for the legend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeTimelinePoint:
    ts: int       # Unix seconds
    label: str    # low_vol_trending | low_vol_ranging | high_vol_stress
    confidence: float

    def to_dict(self) -> dict:
        return {
            "ts": self.ts,
            "label": self.label,
            "confidence": round(self.confidence, 4),
        }


def build_timeline_payload(
    df: pd.DataFrame,
    classifier: Any,
    *,
    max_points: int = 2000,
) -> dict:
    """Build a webapp-ready payload.

    ``df`` must have a DatetimeIndex (UTC) and a ``close`` column.
    ``classifier`` is a fitted RegimeClassifier (REGIME-2B.1).
    ``max_points`` caps the output by uniform sub-sampling so the
    JSON stays under ~80KB even on 6 years M15 (210k bars).
    """
    if "close" not in df.columns:
        raise ValueError("close column required")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DatetimeIndex required")

    returns = df["close"].pct_change().dropna()
    preds = classifier.predict_with_confidence(returns.values)

    # Pair returns.index with predictions.
    timeline = [
        RegimeTimelinePoint(
            ts=int(returns.index[i].timestamp()),
            label=preds[i].label,
            confidence=preds[i].confidence,
        )
        for i in range(len(returns))
    ]

    # Sub-sample uniformly to cap payload size.
    if len(timeline) > max_points:
        step = len(timeline) // max_points
        timeline = timeline[::step][:max_points]

    counts: dict[str, int] = {}
    for p in timeline:
        counts[p.label] = counts.get(p.label, 0) + 1
    total = sum(counts.values()) or 1

    return {
        "entries": [p.to_dict() for p in timeline],
        "summary": [
            {
                "label": k,
                "count": v,
                "share": round(v / total, 4),
            }
            for k, v in sorted(counts.items())
        ],
        "n_entries": len(timeline),
        "max_points": max_points,
    }


__all__ = ["RegimeTimelinePoint", "build_timeline_payload"]
