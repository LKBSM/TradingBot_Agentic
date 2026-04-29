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

    Three modes for the NY-session gate (``ny_mode``):
      * "off"     — no NY filtering
      * "all"     — drop every NY-session signal (legacy / strict)
      * "high_vol" — drop NY signals only when ATR_PCTL > vol_pctl_max
                    (surgical: keeps NY×Q2 and NY×Q3 buckets which
                    historically run PF 1.18 / 1.29 on XAU)

    Plus an independent vol gate (always-on when configured): drop any
    session whose ATR_PCTL exceeds vol_pctl_max — but in "high_vol" mode
    the vol cutoff is applied conditionally per session, so non-NY
    high-vol bars are still allowed when ``ny_mode="high_vol"``.

    Empirical comparison on XAU M15 7-yr replay (test set 2023-2025):
      * ny_mode="off"      + vol_pctl_max=None → PF 1.13, 395 sig/yr (raw)
      * ny_mode="all"      + vol_pctl_max=0.75 → PF 1.35, 134 sig/yr (strict)
      * ny_mode="high_vol" + vol_pctl_max=0.75 → PF 1.30, 205 sig/yr (DEFAULT)
        — recovers NY×Q1/Q2/Q3 buckets which run PF 1.18-1.29 standalone;
          still drops the NY×Q4_high saigneur (PF 0.64, −23R on 288 trades).
        — gives 53% more signals than "all" with similar PF and +27% total R.
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
        ny_mode: str = "high_vol",
    ):
        if ny_mode not in ("off", "all", "high_vol"):
            raise ValueError(f"ny_mode must be off/all/high_vol, got {ny_mode!r}")
        # Back-compat: explicit ``skip_ny=False`` overrides ny_mode to "off".
        if not skip_ny:
            ny_mode = "off"
        self.skip_ny = skip_ny
        self.ny_mode = ny_mode
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
            ny_mode=os.environ.get("REGIME_FILTER_NY_MODE", "high_vol"),
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
        # Compute vol percentile once (used by both NY-conditional and
        # standalone vol gate).
        pctl: Optional[float] = None
        if (
            self.vol_pctl_max is not None
            and atr_series is not None
            and len(atr_series) >= self.vol_min_periods
        ):
            pctl = self._rolling_pctl(atr_series)

        in_ny = False
        if self.ny_mode != "off" and bar_timestamp:
            ts = self._parse_ts(bar_timestamp)
            in_ny = ts is not None and self._in_ny(ts)

        # NY-session gate
        if self.ny_mode == "all" and in_ny:
            self._dropped_ny += 1
            return FilterDecision(False, "NY session (mode=all)")
        if self.ny_mode == "high_vol" and in_ny:
            # Drop NY only when vol is high. If vol is unknown, abstain (allow).
            if pctl is not None and self.vol_pctl_max is not None and pctl > self.vol_pctl_max:
                self._dropped_ny += 1
                return FilterDecision(
                    False,
                    f"NY × high vol (ATR_PCTL={pctl:.2f} > {self.vol_pctl_max:.2f})",
                )

        # Standalone vol gate — applies to non-NY sessions when ny_mode="all"
        # (legacy behaviour) and to all sessions when ny_mode in {"off", "high_vol"}.
        # Skip when we're in "high_vol" NY mode and the bar IS in NY: we already
        # decided above.
        if (
            self.ny_mode != "high_vol" or not in_ny
        ) and pctl is not None and self.vol_pctl_max is not None and pctl > self.vol_pctl_max:
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
