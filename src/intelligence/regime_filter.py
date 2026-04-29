"""Regime filter — gates signal publication by session and vol percentile.

Empirically validated in `reports/feature_filter_audit.md` (Chantier 1+3,
2026-04-29). On 7 years of XAU M15 replay, the combination
`skip session=NY AND skip ATR_PCTL > 0.75` lifts profit factor from
1.13 to 1.355 OOS (n_test=416, +0.22 PF). Without this gate, the NY ×
high-vol bucket alone drains −23R (PF 0.64 on 288 trades).

The filter sits between `ConfluenceDetector` and the state machine: a
score-eligible signal is dropped here if the regime is hostile, which
keeps the score itself untouched and makes the gate a swappable layer.

Design notes:
- ATR_PCTL is computed on a rolling window (default 30 days × 96 bars/day
  for M15 = 2880 bars). Below `min_periods` the filter abstains (allows).
- Session boundaries are UTC. NY = [13:00, 21:00). Adjust if you change
  data tz.
- All thresholds are tunables; see env vars below.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import datetime, time
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Session table — UTC-based. Don't mix tz here; expect upstream to give UTC.
NY_START = time(13, 0)
NY_END = time(21, 0)


@dataclass
class FilterDecision:
    allowed: bool
    reason: str


class RegimeFilter:
    """Drop signals whose regime context historically yields PF < 1.

    Two stacked rules:
      * skip_ny       — drops [13:00, 21:00) UTC entries
      * vol_pctl_max  — drops entries whose current ATR percentile (rolling
                        window) exceeds the cutoff
    """

    DEFAULT_VOL_PCTL_MAX = 0.75
    DEFAULT_VOL_WINDOW_BARS = 30 * 96  # 30 days × 96 M15 bars/day
    DEFAULT_VOL_MIN_PERIODS = 200

    def __init__(
        self,
        skip_ny: bool = True,
        vol_pctl_max: Optional[float] = DEFAULT_VOL_PCTL_MAX,
        vol_window_bars: int = DEFAULT_VOL_WINDOW_BARS,
        vol_min_periods: int = DEFAULT_VOL_MIN_PERIODS,
    ):
        self.skip_ny = skip_ny
        self.vol_pctl_max = vol_pctl_max
        self.vol_window_bars = vol_window_bars
        self.vol_min_periods = vol_min_periods
        self._dropped_ny = 0
        self._dropped_vol = 0
        self._allowed = 0

    @classmethod
    def from_env(cls) -> "RegimeFilter":
        """Build from env vars. All optional; defaults match the audit."""
        return cls(
            skip_ny=os.environ.get("REGIME_FILTER_SKIP_NY", "1") == "1",
            vol_pctl_max=(
                float(os.environ["REGIME_FILTER_VOL_PCTL_MAX"])
                if "REGIME_FILTER_VOL_PCTL_MAX" in os.environ
                else cls.DEFAULT_VOL_PCTL_MAX
            ),
        )

    def evaluate(
        self,
        bar_timestamp: Optional[str],
        atr_series: Optional[pd.Series],
    ) -> FilterDecision:
        """Return whether to allow a signal at the given bar context.

        ``atr_series`` should be the recent ATR values up to and including
        the current bar (typically `enriched["ATR"].tail(window_bars)`).
        Pass ``None`` when ATR isn't available — the vol gate then abstains.
        """
        # Session gate
        if self.skip_ny and bar_timestamp:
            ts = self._parse_ts(bar_timestamp)
            if ts is not None and self._in_ny(ts):
                self._dropped_ny += 1
                return FilterDecision(False, f"NY session ({ts.time()} UTC)")

        # Vol percentile gate
        if self.vol_pctl_max is not None and atr_series is not None and len(atr_series) >= self.vol_min_periods:
            pctl = self._rolling_pctl(atr_series)
            if pctl is not None and pctl > self.vol_pctl_max:
                self._dropped_vol += 1
                return FilterDecision(
                    False, f"high vol regime (ATR_PCTL={pctl:.2f} > {self.vol_pctl_max:.2f})",
                )

        self._allowed += 1
        return FilterDecision(True, "regime ok")

    @staticmethod
    def _parse_ts(ts: str) -> Optional[datetime]:
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except Exception:
            return None

    @staticmethod
    def _in_ny(ts: datetime) -> bool:
        t = ts.time()
        return NY_START <= t < NY_END

    def _rolling_pctl(self, atr: pd.Series) -> Optional[float]:
        """Percentile of the latest ATR within the rolling window."""
        window = atr.tail(self.vol_window_bars).dropna()
        if len(window) < self.vol_min_periods:
            return None
        latest = window.iloc[-1]
        return float((window <= latest).mean())

    def stats(self) -> dict:
        return {
            "dropped_ny": self._dropped_ny,
            "dropped_vol": self._dropped_vol,
            "allowed": self._allowed,
            "drop_rate": (
                (self._dropped_ny + self._dropped_vol) / max(1, self._dropped_ny + self._dropped_vol + self._allowed)
            ),
        }
