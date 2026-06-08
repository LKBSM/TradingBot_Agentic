"""Point-in-time FRED CSV loader.

Each CSV in ``data/macro/`` ships :
- ``date_utc``      : the observation date (the "as-of" date of the value)
- ``value``         : the data point
- ``vintage_date``  : the date the value became publicly available

A backtest at time ``t`` may only use rows where ``vintage_date <= t``
(point-in-time discipline). This loader exposes ``as_of(t)`` returning
the latest value available at time ``t`` for a given series.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FREDSeriesLoader:
    """One FRED series loaded once, queried point-in-time."""

    series_id: str          # e.g. "DGS10"
    csv_path: Path
    _df: pd.DataFrame = None  # type: ignore

    def __post_init__(self) -> None:
        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        df = pd.read_csv(self.csv_path)
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True).dt.tz_localize(None)
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True).dt.tz_localize(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("date_utc").reset_index(drop=True)
        self._df = df

    def as_of(self, ts: pd.Timestamp) -> Optional[float]:
        """Latest value where ``vintage_date <= ts``."""
        ts = pd.Timestamp(ts)
        # PIT discipline : restrict to rows available at ts
        avail = self._df[self._df["vintage_date"] <= ts]
        if avail.empty:
            return None
        # Among those, take the latest observation
        return float(avail.iloc[-1]["value"])

    def series_as_of(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        """Vectorised :meth:`as_of` for a list of bar timestamps.

        Returns a Series aligned to ``timestamps`` with the latest PIT-safe
        value per row. NaN where no data available yet.
        """
        if self._df.empty:
            return pd.Series(np.nan, index=timestamps)
        ts_arr = pd.Series(pd.to_datetime(timestamps).values)
        # asof merge requires sorted keys
        df_pit = self._df.sort_values("vintage_date").reset_index(drop=True)
        out = pd.merge_asof(
            ts_arr.to_frame("ts").assign(_k=range(len(ts_arr))),
            df_pit[["vintage_date", "value"]].rename(columns={"vintage_date": "ts"}),
            on="ts",
            direction="backward",
        )
        return pd.Series(out["value"].values, index=timestamps, name=self.series_id)


def load_all_xau_factors(macro_dir: str | Path = "data/macro") -> dict[str, FREDSeriesLoader]:
    """Load every FRED series in ``macro_dir`` keyed by series id."""
    macro_dir = Path(macro_dir)
    if not macro_dir.exists():
        raise FileNotFoundError(macro_dir)

    series_map: dict[str, FREDSeriesLoader] = {}
    for path in sorted(macro_dir.glob("fred_*.csv")):
        sid = path.stem.replace("fred_", "")
        try:
            series_map[sid] = FREDSeriesLoader(series_id=sid, csv_path=path)
            logger.info("Loaded FRED %s : %d rows", sid, len(series_map[sid]._df))
        except Exception as exc:
            logger.warning("Failed to load %s : %s", path, exc)
    return series_map


__all__ = ["FREDSeriesLoader", "load_all_xau_factors"]
