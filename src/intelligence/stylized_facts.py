"""Diurnal + calendar stylized facts — Sprint REGIME-2B.3.

Computes "what time of day / what day of week / what distance to FOMC
does XAU like to move in?" — feeds narrative ("at 14:30 UTC, XAU's
realised range over the last 6 years averages 1.4× the daily mean,
driven by the New York session open"), not a trade signal.

Three tables produced:

1. **Hourly stylized facts** — per UTC hour-of-day:
     {n, mean_abs_return, std_return, hit_rate_up, p95_abs_return}
2. **DOW stylized facts** — per day-of-week (Mon=0 … Sun=6):
     {n, mean_abs_return, std_return, hit_rate_up}
3. **FOMC distance facts** — buckets by days-to-FOMC:
     {bucket: "fomc_day" | "fomc_-1" | "fomc_-2" | "fomc_+1" | "other"}

All three are computed from a single OHLCV DataFrame with a tz-aware
``DatetimeIndex`` (or a column ``ts_utc``) and a ``close`` column.
``returns = close.pct_change()`` internally.

The output is JSON-serialisable so Aisha's RAG can ingest it as a
context document, and the webapp can render it as a heatmap.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HourlyFact:
    hour: int           # 0-23 UTC
    n: int
    mean_abs_return: float
    std_return: float
    hit_rate_up: float  # P(return > 0)
    p95_abs_return: float

    def to_dict(self) -> dict:
        return {
            "hour": self.hour,
            "n": self.n,
            "mean_abs_return": round(self.mean_abs_return, 6),
            "std_return": round(self.std_return, 6),
            "hit_rate_up": round(self.hit_rate_up, 4),
            "p95_abs_return": round(self.p95_abs_return, 6),
        }


@dataclass(frozen=True)
class DOWFact:
    dow: int  # 0=Mon, 6=Sun
    n: int
    mean_abs_return: float
    std_return: float
    hit_rate_up: float

    def to_dict(self) -> dict:
        return {
            "dow": self.dow,
            "n": self.n,
            "mean_abs_return": round(self.mean_abs_return, 6),
            "std_return": round(self.std_return, 6),
            "hit_rate_up": round(self.hit_rate_up, 4),
        }


@dataclass(frozen=True)
class FOMCBucketFact:
    bucket: str  # one of {"fomc_day", "fomc_-1", "fomc_-2", "fomc_+1", "other"}
    n: int
    mean_abs_return: float
    std_return: float

    def to_dict(self) -> dict:
        return {
            "bucket": self.bucket,
            "n": self.n,
            "mean_abs_return": round(self.mean_abs_return, 6),
            "std_return": round(self.std_return, 6),
        }


def _ensure_returns(df: pd.DataFrame) -> pd.Series:
    if "returns" in df.columns:
        return df["returns"].dropna()
    if "close" in df.columns:
        return df["close"].pct_change().dropna()
    raise ValueError("DataFrame needs either a 'returns' or 'close' column")


def _datetime_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    if isinstance(df.index, pd.DatetimeIndex):
        return df.index
    if "ts_utc" in df.columns:
        return pd.DatetimeIndex(pd.to_datetime(df["ts_utc"], utc=True))
    raise ValueError(
        "DataFrame needs a DatetimeIndex or a 'ts_utc' column"
    )


def hourly_facts(df: pd.DataFrame) -> list[HourlyFact]:
    """Per-hour-of-day stylized facts. Index must be tz-aware UTC."""
    idx = _datetime_index(df)
    returns = _ensure_returns(df)
    # align: returns is shorter by 1 if computed from close.
    common = returns.index.intersection(idx)
    r = returns.loc[common]
    h = idx[idx.isin(common)].hour

    out: list[HourlyFact] = []
    for hour in range(24):
        mask = h == hour
        slice_r = r[mask]
        if len(slice_r) == 0:
            out.append(HourlyFact(hour, 0, 0.0, 0.0, 0.0, 0.0))
            continue
        abs_r = slice_r.abs()
        out.append(
            HourlyFact(
                hour=hour,
                n=int(len(slice_r)),
                mean_abs_return=float(abs_r.mean()),
                std_return=float(slice_r.std()),
                hit_rate_up=float((slice_r > 0).mean()),
                p95_abs_return=float(abs_r.quantile(0.95)),
            )
        )
    return out


def dow_facts(df: pd.DataFrame) -> list[DOWFact]:
    """Per-day-of-week stylized facts (Mon=0 … Sun=6)."""
    idx = _datetime_index(df)
    returns = _ensure_returns(df)
    common = returns.index.intersection(idx)
    r = returns.loc[common]
    dow = idx[idx.isin(common)].dayofweek

    out: list[DOWFact] = []
    for d in range(7):
        mask = dow == d
        slice_r = r[mask]
        if len(slice_r) == 0:
            out.append(DOWFact(d, 0, 0.0, 0.0, 0.0))
            continue
        out.append(
            DOWFact(
                dow=d,
                n=int(len(slice_r)),
                mean_abs_return=float(slice_r.abs().mean()),
                std_return=float(slice_r.std()),
                hit_rate_up=float((slice_r > 0).mean()),
            )
        )
    return out


def _classify_fomc_bucket(d_to_fomc: int) -> str:
    if d_to_fomc == 0:
        return "fomc_day"
    if d_to_fomc == -1:
        return "fomc_-1"
    if d_to_fomc == -2:
        return "fomc_-2"
    if d_to_fomc == 1:
        return "fomc_+1"
    return "other"


def fomc_bucket_facts(
    df: pd.DataFrame, fomc_dates: Iterable[dt.date]
) -> list[FOMCBucketFact]:
    """Bucket each bar by days-from-nearest-FOMC, compute stats."""
    idx = _datetime_index(df)
    returns = _ensure_returns(df)
    common = returns.index.intersection(idx)
    r = returns.loc[common]
    bar_dates = idx[idx.isin(common)].date

    fomc_sorted = sorted(fomc_dates)
    if not fomc_sorted:
        return []

    # Vectorised "closest fomc date" via searchsorted.
    fomc_arr = np.array(
        [d.toordinal() for d in fomc_sorted], dtype=np.int64
    )
    bar_arr = np.array([d.toordinal() for d in bar_dates], dtype=np.int64)
    insert_idx = np.searchsorted(fomc_arr, bar_arr)
    insert_idx = np.clip(insert_idx, 0, len(fomc_arr) - 1)
    left_idx = np.clip(insert_idx - 1, 0, len(fomc_arr) - 1)

    nearest = np.where(
        np.abs(fomc_arr[insert_idx] - bar_arr)
        <= np.abs(fomc_arr[left_idx] - bar_arr),
        fomc_arr[insert_idx],
        fomc_arr[left_idx],
    )
    delta = bar_arr - nearest  # positive = after FOMC, negative = before

    buckets = np.array([_classify_fomc_bucket(int(d)) for d in delta])

    out: list[FOMCBucketFact] = []
    for b in ("fomc_-2", "fomc_-1", "fomc_day", "fomc_+1", "other"):
        mask = buckets == b
        slice_r = r[mask]
        if len(slice_r) == 0:
            out.append(FOMCBucketFact(b, 0, 0.0, 0.0))
            continue
        out.append(
            FOMCBucketFact(
                bucket=b,
                n=int(len(slice_r)),
                mean_abs_return=float(slice_r.abs().mean()),
                std_return=float(slice_r.std()),
            )
        )
    return out


def compute_stylized_facts(
    df: pd.DataFrame,
    fomc_dates: Optional[Iterable[dt.date]] = None,
) -> dict:
    """One-shot all-three-tables — JSON-serialisable output."""
    return {
        "hourly": [f.to_dict() for f in hourly_facts(df)],
        "dow": [f.to_dict() for f in dow_facts(df)],
        "fomc": [
            f.to_dict()
            for f in (fomc_bucket_facts(df, fomc_dates) if fomc_dates else [])
        ],
    }


__all__ = [
    "DOWFact",
    "FOMCBucketFact",
    "HourlyFact",
    "compute_stylized_facts",
    "dow_facts",
    "fomc_bucket_facts",
    "hourly_facts",
]
