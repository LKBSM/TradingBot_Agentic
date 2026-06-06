"""Chantier 4 — condensed signal_summary provider (doc §4.4).

Produces the ``signal_summary`` that is injected into the chatbot system prompt
by default (so most questions are answered without any tool call) and is also
exposed as the ``get_signal_summary`` tool.

It condenses the 6 tracked combinations (XAUUSD/EURUSD × M15/H1/H4) into a small
dict. The result is cached for ``CACHE_TTL_SECONDS`` (60s) thread-safely: within
a short user session the summary is stable (a new candle close is the only thing
that would change it), so we avoid re-running the assembler 6× per message.

Degradation is per-combination: if one combination fails to generate, the other
five still populate the summary.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Callable, Optional, Sequence

logger = logging.getLogger(__name__)

DEFAULT_INSTRUMENTS: tuple[str, ...] = ("XAUUSD", "EURUSD")
DEFAULT_TIMEFRAMES: tuple[str, ...] = ("M15", "H1", "H4")


class SignalSummaryProvider:
    """Thread-safe, TTL-cached condenser of the tracked MarketReadings."""

    CACHE_TTL_SECONDS = 60

    def __init__(
        self,
        assembler: Any,
        instruments: Sequence[str] = DEFAULT_INSTRUMENTS,
        timeframes: Sequence[str] = DEFAULT_TIMEFRAMES,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self._assembler = assembler
        self._combinations = [(i, t) for i in instruments for t in timeframes]
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._cache: Optional[dict[str, Any]] = None
        self._cache_ts: Optional[datetime] = None
        self._lock = threading.RLock()

    def get(self) -> dict[str, Any]:
        """Return ``{"instruments_tracked": [...]}`` (cached up to TTL)."""
        with self._lock:
            now = self._clock()
            if (
                self._cache is not None
                and self._cache_ts is not None
                and (now - self._cache_ts).total_seconds() < self.CACHE_TTL_SECONDS
            ):
                return self._cache

            tracked: list[dict[str, Any]] = []
            for instrument, timeframe in self._combinations:
                try:
                    reading = self._assembler.get_or_generate(instrument, timeframe)
                    tracked.append(self._condense(reading))
                except Exception as exc:  # graceful per-combination degradation
                    logger.warning(
                        "signal_summary: %s/%s failed (%s) — skipped",
                        instrument, timeframe, exc,
                    )
                    continue

            self._cache = {"instruments_tracked": tracked}
            self._cache_ts = now
            return self._cache

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _condense(self, reading: Any) -> dict[str, Any]:
        """Condense a full MarketReading into 7 fields (doc §4.4 example)."""
        return {
            "instrument": reading.header.instrument,
            "timeframe": reading.header.timeframe,
            "trend": reading.regime.trend,
            "volatility_observed": reading.regime.volatility_observed,
            "market_phase": reading.regime.market_phase,
            "structure_summary": self._build_structure_summary(reading.structure),
            "news_upcoming_count": len(reading.events.news_upcoming),
            "last_candle_close": reading.header.candle_close_ts.isoformat(),
        }

    @staticmethod
    def _build_structure_summary(structure: Any) -> str:
        """One-line, niveau 1.5 strict structural digest (no forbidden tokens)."""
        parts: list[str] = []
        if structure.bos is not None:
            parts.append(f"BOS {structure.bos.direction} {structure.bos.validation_status}")
        if structure.choch is not None:
            parts.append(f"CHOCH {structure.choch.direction}")
        active_ob = sum(1 for ob in structure.order_blocks if ob.status == "active")
        if active_ob:
            parts.append(f"{active_ob} OB actif(s)")
        active_fvg = sum(
            1 for fvg in structure.fair_value_gaps if fvg.status == "active"
        )
        if active_fvg:
            parts.append(f"{active_fvg} FVG actif(s)")
        if structure.retest_in_progress is not None:
            parts.append("retest en cours")
        return ", ".join(parts) if parts else "aucune structure notable"


__all__ = ["DEFAULT_INSTRUMENTS", "DEFAULT_TIMEFRAMES", "SignalSummaryProvider"]
