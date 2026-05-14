"""Rolling cross-asset correlation — Sprint REGIME-2B.4.

Computes the rolling 30-day Pearson correlation of XAU returns vs.
DXY, SPX, US10Y, BTC. Output feeds narrative ("XAU is anti-correlated
with DXY at -0.78 this week") and the webapp heatmap; never a trade
signal.

Why not stationarity-tested cointegration
-----------------------------------------
Phase 2A would have run Engle-Granger / Johansen on these pairs. In
2B we don't need it — the deliverable is a context sentence Aisha
can drop into a narrative, not a trade. A simple rolling Pearson
captures "XAU is moving with/against DXY *right now*", which is
exactly what a reader wants to know.

Inputs
------
A dict of DataFrames keyed by asset symbol:
    {"XAU": df_xau, "DXY": df_dxy, "SPX": df_spx, ...}
Each frame needs a tz-aware DatetimeIndex (or ts_utc column) and a
``close`` column. The reference asset is fixed at ``"XAU"`` — the
others are correlated against it.

Output
------
``corr_table(...)`` returns a DataFrame with one column per asset and
the rolling window stride along the index; ``latest_summary(...)``
returns the most recent value per pair as a JSON-friendly dict
(``{"XAU_vs_DXY": -0.78, ...}``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


REFERENCE_ASSET = "XAU"
DEFAULT_WINDOW = 30  # ~30 trading days


@dataclass(frozen=True)
class CorrelationSummary:
    asset: str
    window_days: int
    last_value: float
    last_ts: pd.Timestamp
    n_obs: int

    def to_dict(self) -> dict:
        return {
            "asset": self.asset,
            "vs": REFERENCE_ASSET,
            "window_days": self.window_days,
            "last_value": round(self.last_value, 4),
            "last_ts": self.last_ts.isoformat() if self.last_ts is not None else None,
            "n_obs": self.n_obs,
        }


def _returns_series(df: pd.DataFrame) -> pd.Series:
    if "returns" in df.columns:
        s = df["returns"].copy()
    elif "close" in df.columns:
        s = df["close"].pct_change()
    else:
        raise ValueError("DataFrame needs 'close' or 'returns' column")
    if not isinstance(s.index, pd.DatetimeIndex):
        if "ts_utc" in df.columns:
            s.index = pd.DatetimeIndex(pd.to_datetime(df["ts_utc"], utc=True))
        else:
            raise ValueError("DataFrame needs DatetimeIndex or ts_utc column")
    return s.dropna()


def corr_table(
    prices: Mapping[str, pd.DataFrame], *, window: int = DEFAULT_WINDOW
) -> pd.DataFrame:
    """Rolling correlation of REFERENCE_ASSET vs every other asset.

    Returns a DataFrame indexed by the *intersection* of all asset
    timestamps, one column per non-reference asset.
    """
    if REFERENCE_ASSET not in prices:
        raise ValueError(f"missing reference asset {REFERENCE_ASSET}")
    if window < 2:
        raise ValueError("window must be >= 2")

    ref = _returns_series(prices[REFERENCE_ASSET])
    others = {k: _returns_series(v) for k, v in prices.items() if k != REFERENCE_ASSET}
    if not others:
        return pd.DataFrame(index=ref.index)

    # Align all series on the common index — rolling.corr needs aligned data.
    all_idx = ref.index
    for s in others.values():
        all_idx = all_idx.intersection(s.index)
    if len(all_idx) == 0:
        return pd.DataFrame()

    ref_aligned = ref.loc[all_idx]
    out = {}
    for name, s in others.items():
        s_aligned = s.loc[all_idx]
        out[name] = ref_aligned.rolling(window=window, min_periods=window).corr(
            s_aligned
        )
    return pd.DataFrame(out, index=all_idx)


def latest_summary(
    prices: Mapping[str, pd.DataFrame], *, window: int = DEFAULT_WINDOW
) -> list[CorrelationSummary]:
    """Most-recent rolling-correlation value per asset pair."""
    table = corr_table(prices, window=window)
    out: list[CorrelationSummary] = []
    if table.empty:
        return out
    for col in sorted(table.columns):
        series = table[col].dropna()
        if series.empty:
            continue
        out.append(
            CorrelationSummary(
                asset=col,
                window_days=window,
                last_value=float(series.iloc[-1]),
                last_ts=series.index[-1],
                n_obs=len(series),
            )
        )
    return out


def heatmap_payload(
    prices: Mapping[str, pd.DataFrame], *, window: int = DEFAULT_WINDOW
) -> dict:
    """Returns ``{assets, dates, matrix}`` for a webapp heatmap."""
    table = corr_table(prices, window=window).dropna(how="all")
    return {
        "reference": REFERENCE_ASSET,
        "window_days": window,
        "assets": list(table.columns),
        "dates": [t.isoformat() for t in table.index],
        "matrix": table.values.tolist(),
    }


__all__ = [
    "CorrelationSummary",
    "DEFAULT_WINDOW",
    "REFERENCE_ASSET",
    "corr_table",
    "heatmap_payload",
    "latest_summary",
]
