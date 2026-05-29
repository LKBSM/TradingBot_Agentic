"""Event-driven macro strategy for XAU around HIGH-impact releases.

Pilier 1 of the institutional quant transformation plan (see
``reports/institutional_quant_transformation_plan.md`` §4 Pilier 1).

Setup
-----
Around every HIGH-impact economic release (NFP, CPI, FOMC, ECB, ...), gold
experiences a documented jump component that survives transaction costs
when traded systematically (Andersen-Bollerslev-Diebold 2003 AER,
"Micro effects of macro announcements", and Faust-Wright 2018).

Because the historical ForexFactory CSV we have does NOT contain Actual /
Forecast columns (only event timestamps + impact), we cannot compute a
**surprise** score. Instead we trade the **immediate post-release momentum**:

1. **Trigger bar** : the first M15 bar whose midpoint falls within
   ``[event_time, event_time + trigger_window_min)``.
2. **Direction filter** : the trigger bar's body must exceed
   ``trigger_threshold_atr`` × ATR(14). Otherwise no trade.
3. **Direction** : LONG if close > open on the trigger bar, else SHORT.
4. **Entry** : at the close of the trigger bar.
5. **Stop-loss** : ``sl_atr`` × ATR away from entry (opposite side).
6. **Take-profit** : ``tp_atr`` × ATR away from entry (target side).
7. **Time exit** : if neither SL nor TP hit within ``max_hold_bars``, close
   at market on the next bar's open.

Returns are reported in **R-multiples** (PnL / risk), where risk is the
SL distance. R-multiples are scale-invariant — directly comparable across
instruments and time periods.

This module is a *backtest implementation*. It does not execute trades or
publish signals. The CPCV harness wraps it for cross-validated evaluation,
and the strategy_gates module decides whether to graduate it to production.

References
----------
- Andersen, T.G., Bollerslev, T., Diebold, F.X., Vega, C. (2003). *Micro
  effects of macro announcements: real-time price discovery in foreign
  exchange*. AER 93 (1), 38-62.
- Andersen, T.G., Bollerslev, T., Diebold, F.X. (2007). *Roughing it up:
  including jump components in measuring, modeling, and forecasting asset
  return volatility*. ReStud 89 (4), 701-720.
- Faust, J., Wright, J.H. (2018). *Risk Premia in the 8:30 Economic News*.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class EventStrategyConfig:
    """Hyperparameters for the event-driven strategy.

    Defaults set per ABD 2007: ~30min event window, 0.5×ATR trigger,
    2× SL / 3× TP yields RR=1.5 (acceptable for macro events).
    """

    trigger_window_min: int = 30          # First M15 bar within T..T+30min after release
    trigger_threshold_atr: float = 0.50   # Body must exceed 0.5×ATR
    sl_atr: float = 2.0                   # Stop = 2×ATR
    tp_atr: float = 3.0                   # Target = 3×ATR (RR=1.5)
    max_hold_bars: int = 8                # 8 × M15 = 2h max hold
    atr_window: int = 14
    currency_filter: Optional[str] = "USD"  # Only USD events affect XAU primarily
    impact_filter: str = "HIGH"
    # Restrict to a curated list of high-conviction event names. If None,
    # accept everything matching impact_filter.
    event_name_keywords: Optional[Sequence[str]] = field(
        default_factory=lambda: (
            "non-farm payrolls",
            "nfp",
            "cpi",
            "consumer price index",
            "fomc",
            "fed funds",
            "federal funds rate",
            "fed interest rate",
            "fed chair",
            "powell",
            "core pce",
            "pce price index",
            "unemployment rate",
            "retail sales",
            "ism manufacturing",
            "ism services",
        )
    )


# =============================================================================
# Trade record
# =============================================================================


@dataclass
class EventTrade:
    """One realised event-driven trade."""

    event_time: pd.Timestamp
    event_name: str
    direction: str             # "LONG" or "SHORT"
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str           # "TP", "SL", "TIME"
    atr_at_entry: float
    pnl_price: float           # exit_price - entry_price (sign-adjusted)
    r_multiple: float          # pnl_price / (sl_atr * atr_at_entry)

    def to_dict(self) -> dict:
        return {
            "event_time": self.event_time.isoformat(),
            "event_name": self.event_name,
            "direction": self.direction,
            "entry_time": self.entry_time.isoformat(),
            "entry_price": self.entry_price,
            "exit_time": self.exit_time.isoformat(),
            "exit_price": self.exit_price,
            "exit_reason": self.exit_reason,
            "atr_at_entry": self.atr_at_entry,
            "pnl_price": self.pnl_price,
            "r_multiple": self.r_multiple,
        }


# =============================================================================
# Data loaders & helpers
# =============================================================================


def _parse_ohlcv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names — accept Date/Open/High/Low/Close/Volume case-insensitively
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def _parse_calendar(path: str | Path, cfg: EventStrategyConfig) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" in df.columns:
        df = df.rename(columns={"date": "event_time"})
    df["event_time"] = pd.to_datetime(df["event_time"])

    # Apply filters
    if cfg.impact_filter:
        df = df[df["impact"].astype(str).str.upper() == cfg.impact_filter.upper()]
    if cfg.currency_filter:
        df = df[df["currency"].astype(str).str.upper() == cfg.currency_filter.upper()]
    if cfg.event_name_keywords:
        keywords = tuple(k.lower() for k in cfg.event_name_keywords)
        mask = df["event"].astype(str).str.lower().apply(
            lambda s: any(k in s for k in keywords)
        )
        df = df[mask]

    df = df.sort_values("event_time").reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> np.ndarray:
    """Wilder-style ATR on the OHLCV frame. Returns a numpy array aligned
    with df rows; first ``window`` entries are NaN."""
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    # Simple moving average ATR (Wilder uses smoothed but SMA is fine for
    # event-window magnitudes).
    atr = np.full_like(tr, np.nan)
    if len(tr) >= window:
        cum = np.cumsum(tr)
        atr[window - 1] = cum[window - 1] / window
        for i in range(window, len(tr)):
            atr[i] = (atr[i - 1] * (window - 1) + tr[i]) / window
    return atr


# =============================================================================
# Strategy
# =============================================================================


class EventDrivenMacroStrategy:
    """Pure backtest engine for the event-driven macro strategy.

    Stateless beyond its config: ``run`` ingests an OHLCV frame + a calendar
    frame and returns a list of ``EventTrade``.
    """

    def __init__(self, cfg: Optional[EventStrategyConfig] = None) -> None:
        self.cfg = cfg or EventStrategyConfig()

    def run(self, ohlcv: pd.DataFrame, calendar: pd.DataFrame) -> List[EventTrade]:
        """Execute the strategy over the given history.

        Parameters
        ----------
        ohlcv
            DataFrame with columns timestamp, open, high, low, close
            (lowercase), sorted ascending by timestamp.
        calendar
            DataFrame with columns event_time, event, currency, impact,
            filtered by the strategy's config.
        """
        cfg = self.cfg
        if ohlcv.empty or calendar.empty:
            return []

        bar_times = ohlcv["timestamp"].to_numpy()
        opens = ohlcv["open"].to_numpy(dtype=float)
        highs = ohlcv["high"].to_numpy(dtype=float)
        lows = ohlcv["low"].to_numpy(dtype=float)
        closes = ohlcv["close"].to_numpy(dtype=float)
        atr = compute_atr(ohlcv, window=cfg.atr_window)

        n_bars = len(ohlcv)
        trades: List[EventTrade] = []

        for _, ev in calendar.iterrows():
            event_time = pd.Timestamp(ev["event_time"])
            # Find the first bar whose timestamp >= event_time AND within
            # event_time + trigger_window_min.
            window_end = event_time + timedelta(minutes=cfg.trigger_window_min)
            mask = (bar_times >= np.datetime64(event_time)) & (
                bar_times < np.datetime64(window_end)
            )
            cand_idx = np.where(mask)[0]
            if len(cand_idx) == 0:
                continue
            trigger_idx = int(cand_idx[0])
            if trigger_idx >= n_bars - 1:
                continue
            if not np.isfinite(atr[trigger_idx]) or atr[trigger_idx] <= 0:
                continue

            body = closes[trigger_idx] - opens[trigger_idx]
            atr_t = float(atr[trigger_idx])
            if abs(body) < cfg.trigger_threshold_atr * atr_t:
                continue

            direction = "LONG" if body > 0 else "SHORT"
            entry_price = float(closes[trigger_idx])
            entry_time = pd.Timestamp(bar_times[trigger_idx])
            if direction == "LONG":
                sl_price = entry_price - cfg.sl_atr * atr_t
                tp_price = entry_price + cfg.tp_atr * atr_t
            else:
                sl_price = entry_price + cfg.sl_atr * atr_t
                tp_price = entry_price - cfg.tp_atr * atr_t

            # Walk forward up to max_hold_bars looking for SL/TP/time exit
            exit_idx: Optional[int] = None
            exit_reason = "TIME"
            exit_price = entry_price
            for i in range(trigger_idx + 1, min(trigger_idx + 1 + cfg.max_hold_bars, n_bars)):
                hi, lo = float(highs[i]), float(lows[i])
                if direction == "LONG":
                    # SL hit if low <= sl, TP if high >= tp; if both in one bar
                    # we conservatively assume SL hit first (worst case).
                    if lo <= sl_price:
                        exit_idx = i
                        exit_reason = "SL"
                        exit_price = sl_price
                        break
                    if hi >= tp_price:
                        exit_idx = i
                        exit_reason = "TP"
                        exit_price = tp_price
                        break
                else:
                    if hi >= sl_price:
                        exit_idx = i
                        exit_reason = "SL"
                        exit_price = sl_price
                        break
                    if lo <= tp_price:
                        exit_idx = i
                        exit_reason = "TP"
                        exit_price = tp_price
                        break

            if exit_idx is None:
                # Time exit on the next bar's open (or the last available bar)
                exit_idx = min(trigger_idx + cfg.max_hold_bars, n_bars - 1)
                exit_price = float(opens[exit_idx])
                exit_reason = "TIME"

            pnl = (exit_price - entry_price) if direction == "LONG" else (entry_price - exit_price)
            risk = cfg.sl_atr * atr_t
            r_mult = pnl / risk if risk > 0 else 0.0

            trades.append(
                EventTrade(
                    event_time=event_time,
                    event_name=str(ev.get("event", "")),
                    direction=direction,
                    entry_time=entry_time,
                    entry_price=entry_price,
                    exit_time=pd.Timestamp(bar_times[exit_idx]),
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    atr_at_entry=atr_t,
                    pnl_price=pnl,
                    r_multiple=r_mult,
                )
            )

        return trades


# =============================================================================
# Convenience top-level runner
# =============================================================================


def run_event_strategy_from_csv(
    ohlcv_path: str | Path,
    calendar_path: str | Path,
    cfg: Optional[EventStrategyConfig] = None,
) -> tuple[List[EventTrade], np.ndarray]:
    """Load CSVs, run strategy, return trades + R-multiple array."""
    cfg = cfg or EventStrategyConfig()
    ohlcv = _parse_ohlcv(ohlcv_path)
    calendar = _parse_calendar(calendar_path, cfg)
    strat = EventDrivenMacroStrategy(cfg)
    trades = strat.run(ohlcv, calendar)
    r_multiples = np.asarray([t.r_multiple for t in trades], dtype=float)
    return trades, r_multiples
