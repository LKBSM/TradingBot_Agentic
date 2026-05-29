"""Offline news provider for historical replay.

The live ``NewsAnalysisAgent`` depends on async HTTP fetchers and sentiment
models that aren't useful in a deterministic backtest. This lightweight
provider answers the one question the backtest actually cares about:

    "Is this bar inside the blackout window of a high-impact event
     affecting the instrument being traded?"

If yes → returns a ``NewsAssessment`` with ``decision=BLOCK`` so the
``ConfluenceDetector._is_news_blocked`` gate suppresses entries.
If no  → returns ``None`` so the detector's renormalisation logic falls
back on the non-news components (see P1 fix).

The CSV format is the one produced by ``scripts/fetch_forexfactory_live.py``
and the pre-packaged ``data/economic_calendar_HIGH_IMPACT_2019_2025.csv``:

    Date, Currency, Event, Impact, Actual, Forecast, Previous

Only rows with ``Impact == 'HIGH'`` are loaded — mediums and lows do not
justify blocking a trade for XAU/USD.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


# Instrument -> currencies whose HIGH-impact events should block the trade.
# Gold is dollar-denominated and moves on Fed / US macro; FX pairs are
# classically blocked only on their two quote currencies.
DEFAULT_AFFECTING_CURRENCIES = {
    "XAUUSD": {"USD", "XAU"},
    "EURUSD": {"USD", "EUR"},
    "GBPUSD": {"USD", "GBP"},
    "USDJPY": {"USD", "JPY"},
    "BTCUSD": {"USD"},
    "US500":  {"USD"},
}


@dataclass(frozen=True)
class _BlackoutEvent:
    ts: datetime          # Event timestamp (UTC, naive)
    currency: str
    name: str


@dataclass
class BacktestNewsProvider:
    """Serve a ``NewsAssessment`` (BLOCK only) for bars inside a blackout.

    Parameters
    ----------
    events:
        High-impact events sorted by timestamp.
    block_before_min / block_after_min:
        Blackout window around each event (±).
    """
    events: List[_BlackoutEvent]
    block_before_min: int = 30
    block_after_min: int = 30

    _ts_index: List[datetime] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        self._ts_index = [e.ts for e in self.events]

    # ------------------------------------------------------------------ #
    # FACTORY
    # ------------------------------------------------------------------ #

    @classmethod
    def from_csv(
        cls,
        csv_path: str | Path,
        symbol: str = "XAUUSD",
        affecting_currencies: Optional[Iterable[str]] = None,
        block_before_min: int = 30,
        block_after_min: int = 30,
    ) -> "BacktestNewsProvider":
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Calendar CSV not found: {path}")

        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        for col in ("Date", "Currency", "Event", "Impact"):
            if col not in df.columns:
                raise ValueError(
                    f"CSV {path} missing required column '{col}' "
                    f"(got {list(df.columns)})"
                )

        df = df[df["Impact"].str.upper() == "HIGH"]
        affecting = {c.upper() for c in (
            affecting_currencies
            if affecting_currencies is not None
            else DEFAULT_AFFECTING_CURRENCIES.get(symbol.upper(), {"USD"})
        )}
        df = df[df["Currency"].str.upper().isin(affecting)]

        events: List[_BlackoutEvent] = []
        for _, row in df.iterrows():
            try:
                ts = pd.to_datetime(row["Date"])
                if ts.tz is not None:
                    ts = ts.tz_convert("UTC").tz_localize(None)
                events.append(_BlackoutEvent(
                    ts=ts.to_pydatetime(),
                    currency=row["Currency"].upper(),
                    name=row["Event"],
                ))
            except (ValueError, TypeError):
                continue
        events.sort(key=lambda e: e.ts)
        return cls(
            events=events,
            block_before_min=block_before_min,
            block_after_min=block_after_min,
        )

    # ------------------------------------------------------------------ #
    # LOOKUP
    # ------------------------------------------------------------------ #

    def at(self, bar_timestamp):
        """Return a BLOCK ``NewsAssessment`` if ``bar_timestamp`` falls
        inside the ±blackout window of any loaded event, else ``None``.

        Accepts any value pandas can coerce to a Timestamp (ISO string,
        datetime, pandas Timestamp). Returned Timestamp is treated as naive
        UTC (matching the CSV source which is UTC).
        """
        if not self.events:
            return None

        ts = pd.to_datetime(bar_timestamp)
        if ts.tz is not None:
            ts = ts.tz_convert("UTC").tz_localize(None)
        bar_dt = ts.to_pydatetime()

        # Binary search for the first event >= bar_dt - block_after_min
        # (any earlier event is already past its blackout tail).
        import bisect
        lower_bound = bar_dt - timedelta(minutes=self.block_after_min)
        idx = bisect.bisect_left(self._ts_index, lower_bound)
        if idx >= len(self.events):
            return None

        ev = self.events[idx]
        lead_min = (ev.ts - bar_dt).total_seconds() / 60.0
        if lead_min > self.block_before_min:
            return None
        if lead_min < -self.block_after_min:
            return None

        # Lazy import so ``news_analysis_agent`` isn't required at import time
        # (keeps the backtest lightweight for users who don't need news).
        from src.agents.news_analysis_agent import (
            NewsAssessment, NewsDecision,
        )
        try:
            from src.agents.news_analysis_agent import NewsImpact
        except ImportError:  # pragma: no cover — older branches
            NewsImpact = None  # type: ignore

        return NewsAssessment(
            decision=NewsDecision.BLOCK,
            current_impact_level=(
                NewsImpact.HIGH if NewsImpact is not None else "HIGH"
            ),
            sentiment_score=0.0,
            sentiment_confidence=0.0,
            blocking_events=[],
            position_multiplier=0.0,
            reasoning=[
                f"Blackout: {ev.currency} {ev.name} "
                f"at {ev.ts.isoformat()} ({lead_min:+.0f} min)"
            ],
            valid_until=bar_dt + timedelta(
                minutes=max(0, int(self.block_after_min + lead_min)),
            ),
            hours_to_next_high_impact=max(0.0, lead_min / 60.0),
        )

    # Convenience: the SignalReplay expects a plain callable.
    def __call__(self, bar_timestamp):
        return self.at(bar_timestamp)
