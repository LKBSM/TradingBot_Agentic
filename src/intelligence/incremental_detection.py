"""LB-1 — incremental, persisted detection driver.

Renders the detection INCREMENTAL without REWRITING a single detection rule
(LB-1 discipline): it reuses ``build_enriched_frame`` — the exact shared engine
run the readings are built from (same DataFrame, same default SMCConfig, same
``analyze`` flags) — plus ``collect_zones`` / ``collect_structure_events``, and
persists the result via :class:`StructureStore`.

  * ``refresh`` (production): on a newly-closed candle, run the engine ONCE on the
    trailing window (bounded by ``window_bars``) and persist the snapshot. Cost is
    therefore constant per refresh and independent of how deep storage goes — the
    window is fixed, not the history.
  * ``replay`` (initial fill / tests): feed candles one at a time, applying the
    same per-candle snapshot. Because each step reuses the identical engine call,
    replaying a frozen candle set to its end persists EXACTLY what a single full
    recompute of that set produces (non-regression) — the driver adds
    persistence, never a second detector.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, List, Optional, Sequence

logger = logging.getLogger(__name__)

# Bounded detection window. Matches the scanner's historical lookback so a
# refresh sees the same context the product always did — deep STORAGE never
# enlarges this, which is the whole point of separating the three depths.
DEFAULT_WINDOW_BARS = 800

# Below this the engine has too little context to detect structure; skip (the
# early bars of an initial replay contribute nothing until the window fills).
MIN_BARS = 20

# Effectively "no cap" — persistence keeps EVERY still-relevant zone/event the
# window holds, not a display top-N. The same value is used on both sides of the
# non-regression comparison, so it never introduces a discrepancy.
_UNLIMITED = 10 ** 9


def _analyze_window(candles: Sequence[Any]):
    """Run the shared product engine on a candle window → (zones, events).

    Local import keeps the heavy engine out of module import (mirrors the
    assembler). Uses ``build_enriched_frame`` verbatim, so the zones/events are
    byte-for-byte what a reading would surface for the same window.
    """
    from src.intelligence.market_reading_assembler import build_enriched_frame
    from src.intelligence.market_reading_mappers import (
        collect_structure_events,
        collect_zones,
    )

    enriched, _cfg = build_enriched_frame(candles)
    zones = collect_zones(enriched, idx=-1, max_per_type=_UNLIMITED)
    events = collect_structure_events(enriched, idx=-1, max_per_type=_UNLIMITED)
    return zones, events


class IncrementalDetector:
    def __init__(self, structure_store, *, window_bars: int = DEFAULT_WINDOW_BARS) -> None:
        self._store = structure_store
        self._window = window_bars

    def _apply(self, instrument: str, timeframe: str, window: Sequence[Any]) -> bool:
        if len(window) < MIN_BARS:
            return False
        zones, events = _analyze_window(window)
        self._store.apply_snapshot(
            instrument, timeframe, zones, events,
            window_start=getattr(window[0], "ts", None),
            processed_ts=getattr(window[-1], "ts", None),
        )
        return True

    def refresh(self, instrument: str, timeframe: str, candles: Sequence[Any]) -> bool:
        """Production entry: persist the snapshot for the latest trailing window.

        ``candles`` should be the trailing window (e.g. the store's last
        ``window_bars``); only the tail is used, so passing more is harmless and
        the work stays bounded by ``window_bars``.
        """
        window = candles[-self._window:] if len(candles) > self._window else candles
        return self._apply(instrument, timeframe, window)

    def replay(self, instrument: str, timeframe: str, candles: Sequence[Any]) -> int:
        """Feed candles one at a time (initial fill / non-regression tests).

        Returns the number of snapshots applied. Each step analyses the trailing
        window ending at that candle — identical to how ``refresh`` will run live.
        """
        applied = 0
        n = len(candles)
        for i in range(n):
            lo = max(0, i + 1 - self._window)
            if self._apply(instrument, timeframe, candles[lo:i + 1]):
                applied += 1
        return applied
