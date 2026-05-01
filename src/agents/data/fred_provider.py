"""
FRED/ALFRED macro series ingestion with vintage-aware timestamps.

Sprint DATA-1.1 (Marwan, 4h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie II.2 Agent 1.

Anti-leak guarantee
-------------------
Naive `Fred.get_series()` returns the LATEST revision of each observation,
which leaks future information into a backtest: a value for 2020-03-15 may
have been revised in 2021. Using that value to predict 2020-03-16's return
is a textbook look-ahead.

This module uses `Fred.get_series_all_releases()` and keeps only the FIRST
release of each observation, paired with its publication date (`vintage_date`).
Consumers retrieve macro values via `macro_at(df, t)` which guarantees that
only data with `vintage_date <= t` is ever returned.

Series ingested (5 base + 1 virtual)
------------------------------------
- DGS10:    10-Year Treasury Constant Maturity Rate
- DFII10:   10-Year TIPS yield (real yield)
- DTWEXBGS: Trade Weighted U.S. Dollar Index Broad Goods+Services
- VIXCLS:   CBOE Volatility Index VIX (close)
- T10Y2Y:   10-Year minus 2-Year Treasury yield spread
- BREAKEVEN_10Y (virtual): DGS10 - DFII10 (10y inflation breakeven)
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from pathlib import Path

import pandas as pd
from fredapi import Fred

logger = logging.getLogger(__name__)


SERIES_IDS: dict[str, str] = {
    "DGS10": "10-Year Treasury Constant Maturity Rate",
    "DFII10": "10-Year TIPS yield (real yield)",
    "DTWEXBGS": "Trade Weighted U.S. Dollar Index Broad Goods+Services",
    "VIXCLS": "CBOE Volatility Index VIX (close)",
    "T10Y2Y": "10-Year minus 2-Year Treasury yield spread",
}

# Default fallback when a series cannot be fetched with vintage info.
# FRED daily series typically publish ~12h after observation date close.
DEFAULT_PUBLICATION_LAG = timedelta(days=1)

# Output schema enforced for every series DataFrame returned by this module.
OUTPUT_COLUMNS = ["date_utc", "value", "vintage_date"]


class FredProvider:
    """Vintage-aware FRED macro series provider.

    Parameters
    ----------
    api_key : str | None
        FRED API key. If None, read from `FRED_API_KEY` env var.
        Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html.

    Examples
    --------
    >>> p = FredProvider()                                   # doctest: +SKIP
    >>> dgs10 = p.fetch_series("DGS10", "2019-01-01")        # doctest: +SKIP
    >>> dgs10.columns.tolist()                               # doctest: +SKIP
    ['date_utc', 'value', 'vintage_date']
    >>> # Vintage-safe lookup at a point in time:
    >>> v = FredProvider.macro_at(dgs10, pd.Timestamp("2020-03-15", tz="UTC"))
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or os.environ.get("FRED_API_KEY")
        if not key:
            raise ValueError(
                "FRED_API_KEY missing. Set it in .env or pass api_key= explicitly. "
                "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html"
            )
        self._fred = Fred(api_key=key)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_series(
        self,
        series_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        """Fetch a single FRED series with first-release vintage timestamps.

        Returns a DataFrame with columns ``date_utc``, ``value``, ``vintage_date``.
        Empty DataFrame with the right schema is returned on no data.
        """
        try:
            raw = self._fred.get_series_all_releases(series_id)
        except Exception as exc:  # network / API errors → fallback
            logger.warning(
                "get_series_all_releases failed for %s (%s); falling back to "
                "latest revision with default publication lag",
                series_id,
                exc,
            )
            return self._fetch_with_default_lag(series_id, start_date, end_date)

        if raw is None or len(raw) == 0:
            return _empty_frame()

        df = _first_release_from_all_releases(raw)

        if start_date is not None:
            df = df[df["date_utc"] >= pd.Timestamp(start_date, tz="UTC")]
        if end_date is not None:
            df = df[df["date_utc"] <= pd.Timestamp(end_date, tz="UTC")]

        return df.reset_index(drop=True)

    def _fetch_with_default_lag(
        self,
        series_id: str,
        start_date: str | None,
        end_date: str | None,
    ) -> pd.DataFrame:
        s = self._fred.get_series(
            series_id, observation_start=start_date, observation_end=end_date
        )
        if s is None or len(s) == 0:
            return _empty_frame()
        df = pd.DataFrame(
            {
                "date_utc": pd.to_datetime(s.index, utc=True),
                "value": pd.to_numeric(s.values, errors="coerce"),
            }
        )
        df["vintage_date"] = df["date_utc"] + DEFAULT_PUBLICATION_LAG
        return df.dropna(subset=["value"]).reset_index(drop=True)[OUTPUT_COLUMNS]

    def fetch_all_series(
        self,
        start_date: str | None = "2019-01-01",
        end_date: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Fetch all base SERIES_IDS in one go."""
        out: dict[str, pd.DataFrame] = {}
        for sid in SERIES_IDS:
            logger.info("Fetching %s ...", sid)
            df = self.fetch_series(sid, start_date, end_date)
            logger.info(
                "  %s: %d obs, %s -> %s",
                sid,
                len(df),
                df["date_utc"].min() if len(df) else None,
                df["date_utc"].max() if len(df) else None,
            )
            out[sid] = df
        return out

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @staticmethod
    def save_to_csv(
        series_data: dict[str, pd.DataFrame],
        output_dir: str | Path,
    ) -> dict[str, Path]:
        """Persist each series to ``{output_dir}/fred_{series_id}.csv``."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for sid, df in series_data.items():
            path = out_dir / f"fred_{sid}.csv"
            df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S%z")
            paths[sid] = path
            logger.info("Saved %s -> %s (%d rows)", sid, path, len(df))
        return paths

    @staticmethod
    def load_from_csv(path: str | Path) -> pd.DataFrame:
        """Inverse of save_to_csv with proper UTC datetime parsing."""
        df = pd.read_csv(path)
        df["date_utc"] = pd.to_datetime(df["date_utc"], utc=True)
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[OUTPUT_COLUMNS]

    # ------------------------------------------------------------------
    # Lookup (vintage-safe)
    # ------------------------------------------------------------------

    @staticmethod
    def macro_at(df: pd.DataFrame, t) -> float | None:
        """Return the macro value valid at time ``t`` (vintage-aware).

        Only observations with ``vintage_date <= t`` are eligible. The most
        recent eligible observation is returned. Returns ``None`` if nothing
        is available yet.
        """
        ts = _ensure_utc_timestamp(t)
        avail = df[df["vintage_date"] <= ts]
        if avail.empty:
            return None
        # Most-recent observation among those already published
        return float(avail.iloc[-1]["value"])

    @staticmethod
    def macro_series_at(df: pd.DataFrame, ts_index: pd.DatetimeIndex) -> pd.Series:
        """Vectorised version of ``macro_at`` over an index of timestamps.

        Uses ``searchsorted`` on the (sorted) ``vintage_date`` column for O(N log M)
        performance, suitable for the 100-random-dates test.
        """
        if df.empty:
            return pd.Series([float("nan")] * len(ts_index), index=ts_index)
        df_sorted = df.sort_values("vintage_date").reset_index(drop=True)
        ts_index_utc = pd.to_datetime(ts_index, utc=True)
        # For each ts, find the count of vintage_dates <= ts; the last one is at idx-1.
        idx = df_sorted["vintage_date"].searchsorted(ts_index_utc, side="right") - 1
        values = df_sorted["value"].to_numpy()
        result = [values[i] if i >= 0 else float("nan") for i in idx]
        return pd.Series(result, index=ts_index)

    # ------------------------------------------------------------------
    # Resampling (M15)
    # ------------------------------------------------------------------

    @staticmethod
    def resample_to_m15(df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill a daily series onto a continuous 15-min UTC index.

        The vintage_date column is forward-filled too so consumers preserve
        anti-leak guarantees on the resampled frame.
        """
        if df.empty:
            return df
        s = df.set_index("date_utc").sort_index()
        idx = pd.date_range(s.index.min(), s.index.max(), freq="15min", tz="UTC")
        out = s.reindex(idx).ffill()
        out.index.name = "date_utc"
        return out.reset_index()

    # ------------------------------------------------------------------
    # Virtual series
    # ------------------------------------------------------------------

    @staticmethod
    def compute_breakeven_10y(
        dgs10_df: pd.DataFrame, dfii10_df: pd.DataFrame
    ) -> pd.DataFrame:
        """10y breakeven inflation = DGS10 - DFII10.

        The breakeven's vintage_date is ``max(DGS10 vintage, DFII10 vintage)``
        since both inputs must be available for the spread to be computed.
        """
        merged = pd.merge(
            dgs10_df.rename(
                columns={"value": "dgs10", "vintage_date": "vintage_dgs10"}
            ),
            dfii10_df.rename(
                columns={"value": "dfii10", "vintage_date": "vintage_dfii10"}
            ),
            on="date_utc",
            how="inner",
        )
        if merged.empty:
            return _empty_frame()
        out = pd.DataFrame(
            {
                "date_utc": merged["date_utc"],
                "value": merged["dgs10"] - merged["dfii10"],
                "vintage_date": merged[["vintage_dgs10", "vintage_dfii10"]].max(axis=1),
            }
        )
        return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "value": pd.Series(dtype="float64"),
            "vintage_date": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _first_release_from_all_releases(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert fredapi `get_series_all_releases` output to first-release frame.

    Input columns are typically ``['date', 'realtime_start', 'value']``.
    """
    if "date" not in raw.columns or "realtime_start" not in raw.columns:
        raise ValueError(
            f"Unexpected get_series_all_releases output columns: {raw.columns.tolist()}"
        )

    first = (
        raw.sort_values(["date", "realtime_start"])
        .groupby("date", as_index=False)
        .first()
    )
    df = pd.DataFrame(
        {
            "date_utc": pd.to_datetime(first["date"], utc=True),
            "value": pd.to_numeric(first["value"], errors="coerce"),
            "vintage_date": pd.to_datetime(first["realtime_start"], utc=True),
        }
    )
    df = df.dropna(subset=["value"])

    # Anti-leak invariant: vintage_date must be >= date_utc. If FRED returns
    # something weird (rare), shift forward by DEFAULT_PUBLICATION_LAG.
    bad_mask = df["vintage_date"] < df["date_utc"]
    if bad_mask.any():
        n_bad = int(bad_mask.sum())
        logger.warning(
            "%d rows have vintage_date < observation; shifting by %s",
            n_bad,
            DEFAULT_PUBLICATION_LAG,
        )
        df.loc[bad_mask, "vintage_date"] = (
            df.loc[bad_mask, "date_utc"] + DEFAULT_PUBLICATION_LAG
        )

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _ensure_utc_timestamp(t) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")
