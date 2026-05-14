"""Multi-instrument data quality — Sprint DATA-2B.3.

Extends the data-quality probe to a 3-instrument coverage (XAU,
EURUSD, USOIL) for Phase 2B. Phase 2A targeted XAU + EURUSD; we add
USOIL daily here.

Per-instrument metrics:

- **coverage_pct**: actual_bars / expected_bars over the requested
  window.
- **max_gap_bars**: longest consecutive missing-bar run.
- **stale_bars**: count of bars where ``close == close.shift(1)``
  (likely a feed freeze masked as data).
- **last_seen_utc**: timestamp of the most recent bar.

The probe is a pure function over a Pandas DataFrame with a
DatetimeIndex (or ``ts_utc`` column) + a ``close`` column. No
network, no FS — same shape as `freshness_monitor` so callers can
compose both into the same dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SUPPORTED_INSTRUMENTS = ("XAU", "EURUSD", "USOIL")


@dataclass(frozen=True)
class InstrumentQuality:
    instrument: str
    n_bars: int
    expected_bars: int
    coverage_pct: float
    max_gap_bars: int
    stale_bars: int
    stale_pct: float
    last_seen_utc: str
    timeframe: str
    ok: bool                 # coverage > 0.95 AND stale_pct < 0.05

    def to_dict(self) -> dict:
        return {
            "instrument": self.instrument,
            "n_bars": self.n_bars,
            "expected_bars": self.expected_bars,
            "coverage_pct": round(self.coverage_pct, 4),
            "max_gap_bars": self.max_gap_bars,
            "stale_bars": self.stale_bars,
            "stale_pct": round(self.stale_pct, 4),
            "last_seen_utc": self.last_seen_utc,
            "timeframe": self.timeframe,
            "ok": self.ok,
        }


def _ensure_dt_index(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.index, pd.DatetimeIndex):
        return df
    if "ts_utc" in df.columns:
        idx = pd.DatetimeIndex(pd.to_datetime(df["ts_utc"], utc=True))
        return df.set_index(idx)
    raise ValueError("DataFrame needs DatetimeIndex or ts_utc column")


def _expected_bar_count(df: pd.DataFrame, timeframe: str) -> int:
    """How many bars *should* exist between min/max ts at this timeframe?"""
    if len(df) < 2:
        return len(df)
    span = df.index[-1] - df.index[0]
    if timeframe == "M15":
        return int(span.total_seconds() / 900) + 1
    if timeframe == "H1":
        return int(span.total_seconds() / 3600) + 1
    if timeframe == "H4":
        return int(span.total_seconds() / 14_400) + 1
    if timeframe == "D1":
        return span.days + 1
    raise ValueError(f"unsupported timeframe {timeframe}")


def check_instrument(
    df: pd.DataFrame,
    *,
    instrument: str,
    timeframe: str,
    coverage_floor: float = 0.95,
    stale_ceiling: float = 0.05,
) -> InstrumentQuality:
    if instrument not in SUPPORTED_INSTRUMENTS:
        raise ValueError(
            f"unsupported instrument {instrument}; "
            f"expected one of {SUPPORTED_INSTRUMENTS}"
        )
    if "close" not in df.columns:
        raise ValueError("DataFrame requires a 'close' column")
    if df.empty:
        return InstrumentQuality(
            instrument=instrument, n_bars=0, expected_bars=0,
            coverage_pct=0.0, max_gap_bars=0, stale_bars=0, stale_pct=0.0,
            last_seen_utc="", timeframe=timeframe, ok=False,
        )

    df = _ensure_dt_index(df).sort_index()
    n = len(df)
    expected = _expected_bar_count(df, timeframe)
    coverage = n / expected if expected > 0 else 1.0

    # Max gap — measured in *bars expected at this timeframe*.
    deltas = df.index.to_series().diff().dropna()
    if timeframe == "M15":
        unit = pd.Timedelta(minutes=15)
    elif timeframe == "H1":
        unit = pd.Timedelta(hours=1)
    elif timeframe == "H4":
        unit = pd.Timedelta(hours=4)
    elif timeframe == "D1":
        unit = pd.Timedelta(days=1)
    else:
        unit = pd.Timedelta(minutes=15)
    max_gap = int((deltas / unit).max()) if not deltas.empty else 0

    # Stale: close[t] == close[t-1] (potential frozen feed).
    diff = df["close"].diff()
    stale_bars = int((diff == 0).sum())
    stale_pct = stale_bars / n if n > 0 else 0.0

    ok = coverage >= coverage_floor and stale_pct <= stale_ceiling
    return InstrumentQuality(
        instrument=instrument,
        n_bars=n,
        expected_bars=expected,
        coverage_pct=coverage,
        max_gap_bars=max_gap,
        stale_bars=stale_bars,
        stale_pct=stale_pct,
        last_seen_utc=df.index[-1].isoformat(),
        timeframe=timeframe,
        ok=ok,
    )


def check_all(
    frames: dict[str, tuple[pd.DataFrame, str]],
) -> list[InstrumentQuality]:
    """Run check_instrument across a {instrument: (df, timeframe)} mapping.

    Returns one InstrumentQuality per entry, sorted by instrument key.
    """
    out: list[InstrumentQuality] = []
    for inst in sorted(frames.keys()):
        df, tf = frames[inst]
        out.append(check_instrument(df, instrument=inst, timeframe=tf))
    return out


__all__ = [
    "InstrumentQuality",
    "SUPPORTED_INSTRUMENTS",
    "check_all",
    "check_instrument",
]
