"""Microstructure proxies extractor."""

from __future__ import annotations

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MicrostructureExtractor:
    """Compute microstructure features from OHLCV M15 bars."""

    # UTC session boundaries (hours)
    ASIA = (0, 7)
    LONDON = (7, 13)
    NY = (13, 21)

    def extract(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        """Return microstructure features aligned to ``ohlcv.index``."""
        df = ohlcv.copy()
        # Normalize column names
        cols = {c: c.capitalize() for c in df.columns
                if c.lower() in ("open", "high", "low", "close", "volume")}
        df = df.rename(columns=cols)
        if not all(c in df.columns for c in ("Open", "High", "Low", "Close")):
            raise ValueError(f"Missing OHLC columns. Got {list(df.columns)}")

        out = pd.DataFrame(index=df.index)
        c = df["Close"]
        o = df["Open"]
        h = df["High"]
        l = df["Low"]
        rng = (h - l).replace(0, np.nan)
        body = (c - o)

        # ---- Bar imbalance ----
        out["bar_imbalance"] = (body / rng).clip(-1, 1).fillna(0)
        out["body_to_range"] = (body.abs() / rng).clip(0, 1).fillna(0)

        # ---- Roll 1984 spread estimator ----
        # 2 * sqrt(-cov(r_t, r_{t-1}))  when negative serial cov (microstructure noise)
        ret = c.pct_change()
        cov_window = 96  # 1-day rolling
        rolling_cov = ret.rolling(cov_window, min_periods=cov_window // 2).cov(ret.shift(1))
        out["roll_spread_estimate"] = (
            2.0 * np.sqrt(-rolling_cov.where(rolling_cov < 0, 0.0))
        ).fillna(0)

        # ---- Garman-Klass volatility (1-day rolling) ----
        gk = 0.5 * (np.log(h / l)) ** 2 - (2 * np.log(2) - 1) * (np.log(c / o)) ** 2
        out["gk_vol_1d"] = np.sqrt(
            gk.rolling(96, min_periods=24).sum()
        ).fillna(0)

        # ---- Realized variance by session (last 1 day) ----
        sq_ret = (ret ** 2).fillna(0)
        hour = pd.Series(df.index, index=df.index).dt.hour
        out["session_is_asia"] = ((hour >= self.ASIA[0]) & (hour < self.ASIA[1])).astype(int)
        out["session_is_london"] = ((hour >= self.LONDON[0]) & (hour < self.LONDON[1])).astype(int)
        out["session_is_ny"] = ((hour >= self.NY[0]) & (hour < self.NY[1])).astype(int)

        out["rv_1d"] = sq_ret.rolling(96, min_periods=24).sum().fillna(0)
        # Per-session decomposition (last 1 day)
        out["rv_asia_1d"] = (sq_ret * out["session_is_asia"]).rolling(96, min_periods=24).sum().fillna(0)
        out["rv_london_1d"] = (sq_ret * out["session_is_london"]).rolling(96, min_periods=24).sum().fillna(0)
        out["rv_ny_1d"] = (sq_ret * out["session_is_ny"]).rolling(96, min_periods=24).sum().fillna(0)

        # ---- Volume features (if present) ----
        if "Volume" in df.columns:
            v = df["Volume"].astype(float)
            v_ma = v.rolling(96, min_periods=24).mean().replace(0, np.nan)
            out["volume_z"] = ((v - v_ma) / v_ma.rolling(96).std(ddof=0).replace(0, np.nan)).fillna(0)
            out["volume_ratio"] = (v / v_ma).fillna(1.0)
        else:
            out["volume_z"] = 0.0
            out["volume_ratio"] = 1.0

        return out


__all__ = ["MicrostructureExtractor"]
