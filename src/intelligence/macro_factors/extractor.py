"""MacroFactorExtractor — institutional XAU/FX factors from FRED + CoT.

Computes the canonical macro features used by quant desks :

1. ``real_10y`` — Real 10Y rate = ``DGS10 - BREAKEVEN_10Y`` (the gold killer).
2. ``real_10y_z`` — Rolling 252d z-score of real_10y.
3. ``dxy_level`` — Trade-weighted USD level.
4. ``dxy_zscore`` — Rolling 60d z-score of DXY.
5. ``dxy_slope_20d`` — 20d log slope of DXY (momentum).
6. ``vix_level`` — VIX close.
7. ``vix_regime`` — Categorical : LOW (<15), NORMAL (15-25), HIGH (25-35), CRISIS (>35).
8. ``yield_curve_2s10s`` — T10Y2Y level (inversion proxy).
9. ``yield_curve_inverted`` — 1 if T10Y2Y < 0 else 0.
10. ``cot_mm_net_pct`` — Money Manager net position % of OI (positioning).
11. ``cot_mm_net_z52`` — Z-score 52w of cot_mm_net_pct.

Output : a DataFrame indexed by bar timestamps with one column per factor,
all PIT-safe (no look-ahead).

>>> ext = MacroFactorExtractor()
>>> factors_df = ext.extract(bar_index=ohlcv_df.index)
>>> factors_df.head()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.intelligence.macro_factors.fred_loader import (
    FREDSeriesLoader,
    load_all_xau_factors,
)

logger = logging.getLogger(__name__)


class MacroFactorExtractor:
    """Build a factor matrix for XAU/FX from data/macro/ CSVs (PIT-safe)."""

    def __init__(self, macro_dir: str | Path = "data/macro"):
        self.macro_dir = Path(macro_dir)
        self._fred_loaders = load_all_xau_factors(self.macro_dir)
        cot_path = self.macro_dir / "cot_gold.csv"
        self._cot_df: Optional[pd.DataFrame] = None
        if cot_path.exists():
            self._cot_df = self._load_cot(cot_path)

    @staticmethod
    def _load_cot(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["report_date"] = pd.to_datetime(df["report_date"], utc=True).dt.tz_localize(None)
        df["vintage_date"] = pd.to_datetime(df["vintage_date"], utc=True).dt.tz_localize(None)
        for col in ("mm_net", "mm_net_pct", "mm_net_pct_z52", "open_interest"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df.sort_values("vintage_date").reset_index(drop=True)

    def extract(self, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
        """Return factor DataFrame indexed by ``bar_index``."""
        bar_index = pd.DatetimeIndex(bar_index)
        out = pd.DataFrame(index=bar_index)

        # ---- FRED daily series (PIT-safe via vintage_date) ----
        dgs10 = self._series("DGS10", bar_index)
        be10y = self._series("BREAKEVEN_10Y", bar_index)
        dfii10 = self._series("DFII10", bar_index)
        dxy = self._series("DTWEXBGS", bar_index)
        vix = self._series("VIXCLS", bar_index)
        t10y2y = self._series("T10Y2Y", bar_index)

        # Real 10Y rate — prefer direct DFII10 (TIPS) if available, fallback to DGS10 - breakeven
        if dfii10 is not None and dfii10.notna().sum() > 100:
            out["real_10y"] = dfii10
        elif dgs10 is not None and be10y is not None:
            out["real_10y"] = dgs10 - be10y
        else:
            out["real_10y"] = np.nan

        out["real_10y_z"] = self._rolling_z(out["real_10y"], window_bars=252 * 96)  # 252 trading days × 96 M15 bars

        # DXY
        out["dxy_level"] = dxy if dxy is not None else np.nan
        out["dxy_z"] = self._rolling_z(out["dxy_level"], window_bars=60 * 96)
        out["dxy_slope_20d"] = self._log_slope(out["dxy_level"], window_bars=20 * 96)

        # VIX
        out["vix_level"] = vix if vix is not None else np.nan
        # Regime bins (close to industry convention)
        out["vix_regime"] = pd.cut(
            out["vix_level"],
            bins=[-np.inf, 15, 25, 35, np.inf],
            labels=["LOW", "NORMAL", "HIGH", "CRISIS"],
        ).astype(str)
        # Numerical encoding for ML
        regime_map = {"LOW": 0, "NORMAL": 1, "HIGH": 2, "CRISIS": 3, "nan": -1}
        out["vix_regime_code"] = out["vix_regime"].map(regime_map).fillna(-1).astype(int)

        # Yield curve
        out["yield_curve_2s10s"] = t10y2y if t10y2y is not None else np.nan
        out["yield_curve_inverted"] = (out["yield_curve_2s10s"] < 0).astype(int)

        # CoT positioning
        if self._cot_df is not None:
            cot_aligned = self._asof_cot(bar_index)
            out["cot_mm_net_pct"] = cot_aligned["mm_net_pct"]
            out["cot_mm_net_z52"] = cot_aligned["mm_net_pct_z52"]
        else:
            out["cot_mm_net_pct"] = np.nan
            out["cot_mm_net_z52"] = np.nan

        # Final cleanup : forward-fill NaN to handle weekends/holidays (after PIT cut)
        out = out.ffill()
        return out

    def _series(self, sid: str, bar_index: pd.DatetimeIndex) -> Optional[pd.Series]:
        loader = self._fred_loaders.get(sid)
        if loader is None:
            return None
        return loader.series_as_of(bar_index)

    @staticmethod
    def _rolling_z(s: pd.Series, window_bars: int) -> pd.Series:
        if s.isna().all():
            return s
        m = s.rolling(window_bars, min_periods=max(20, window_bars // 4)).mean()
        sd = s.rolling(window_bars, min_periods=max(20, window_bars // 4)).std(ddof=0)
        return (s - m) / sd.replace(0, np.nan)

    @staticmethod
    def _log_slope(s: pd.Series, window_bars: int) -> pd.Series:
        """Log-return over `window_bars`."""
        if s.isna().all():
            return s
        return np.log(s / s.shift(window_bars))

    def _asof_cot(self, bar_index: pd.DatetimeIndex) -> pd.DataFrame:
        if self._cot_df is None:
            return pd.DataFrame(index=bar_index)
        cot = self._cot_df.sort_values("vintage_date")
        ts_df = pd.DataFrame({"ts": pd.to_datetime(bar_index)})
        merged = pd.merge_asof(
            ts_df, cot[["vintage_date", "mm_net_pct", "mm_net_pct_z52"]],
            left_on="ts", right_on="vintage_date",
            direction="backward",
        )
        merged.index = bar_index
        return merged


__all__ = ["MacroFactorExtractor"]
