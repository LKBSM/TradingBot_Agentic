"""LB-1 — storage depth per (instrument, timeframe), expressed in DURATION.

Three depths that the product used to confound into one (see the LB-1 mission):

  1. STORAGE   — how many candles are kept in the DB. Deep. THIS module sizes it.
  2. DETECTION — how many candles the engine works on per pass. Bounded elsewhere.
  3. DISPLAY   — what reaches the client. Little, by default.

This module owns only #1: it reads ``config/lookback_depths.json`` and turns a
per-(instrument, timeframe) *duration* (``"6mo"``, ``"2y"`` …) into a target
candle count via a trading-hours-aware conversion. Depths are expressed in
DURATION — never as a global bar constant — because a zone that is one day old
is noise on M1 but a watched level on H4: the useful horizon differs by unit, so
the depth must too.

It is also the single source of truth for the supported instrument/timeframe
perimeter, and for the M1 gate (M1 ships off by default — its per-minute live
refresh does not fit the Twelve Data free plan; see AUDIT-lb-1-lookback-profond).
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Candle duration in minutes. Local constant (not imported from the volatility
# forecaster) to keep this module dependency-light — it is imported by the API
# perimeter and by the backfill task.
_TF_MINUTES: Dict[str, int] = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30,
    "H1": 60, "H4": 240, "D1": 1440, "W1": 10080,
}

# Minutes of trading in one full day, per instrument. FX and metals trade ~24h
# (1440 min) on a weekday. This is what turns a DURATION into a realistic bar
# count. Extend this map when a reduced-hours instrument (an index) enters the
# perimeter; a 24/7 instrument (crypto) would also raise _TRADING_DAYS_PER_WEEK.
_DEFAULT_TRADING_MINUTES_PER_DAY = 1440
_TRADING_MINUTES_PER_DAY: Dict[str, int] = {
    "XAUUSD": 1440,
    "EURUSD": 1440,
}
_TRADING_DAYS_PER_WEEK = 5.0
_MEAN_MONTH_DAYS = 30.4375   # mean Gregorian month
_MEAN_YEAR_DAYS = 365.25     # mean Gregorian year

# Small safety margin so a fetch sized from the config reliably *reaches* the
# requested duration despite holiday gaps and rounding.
_TARGET_MARGIN = 1.05

# The M1 gate. M1 stays off until the five slower units are stable and their
# consumption measured (LB-1 mission §2C). Its live refresh is fundamentally
# incompatible with the free plan (~2880 req/day vs an 800/day cap), so when it
# *is* enabled it ships as backfilled history, not per-minute live.
_M1_ENABLE_ENV = "LB1_ENABLE_M1"

_DURATION_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(d|w|mo|y)\s*$", re.IGNORECASE)
_DURATION_DAYS: Dict[str, float] = {
    "d": 1.0,
    "w": 7.0,
    "mo": 30.4375,     # mean Gregorian month
    "y": 365.25,       # mean Gregorian year
}


def _parse_token(text: str) -> Tuple[float, str]:
    m = _DURATION_RE.match(str(text))
    if not m:
        raise ValueError(
            f"Invalid lookback duration {text!r}. Use <n><unit> with unit in d|w|mo|y."
        )
    return float(m.group(1)), m.group(2).lower()


def parse_duration(text: str) -> timedelta:
    """Parse a depth grammar token (``"1d"``, ``"6mo"``, ``"2y"``) → calendar timedelta.

    Raises ValueError on anything else — a malformed depth is a config bug we
    want surfaced loudly, not silently coerced to a default.
    """
    value, unit = _parse_token(text)
    return timedelta(days=value * _DURATION_DAYS[unit])


def _token_trading_days(token: str) -> float:
    """Depth token → number of *trading* days it represents.

    The ``d`` unit denotes full recent trading days (a 1-day M1 horizon means the
    last ~24h of market, ~1440 candles — not a weekend-discounted 18h). Spans of a
    week or longer discount weekends at 5/7, so 6 months of H1 ≈ 3100 bars.
    """
    value, unit = _parse_token(token)
    if unit == "d":
        return value
    if unit == "w":
        return value * _TRADING_DAYS_PER_WEEK
    if unit == "mo":
        return value * _MEAN_MONTH_DAYS * (_TRADING_DAYS_PER_WEEK / 7.0)
    # unit == "y"
    return value * _MEAN_YEAR_DAYS * (_TRADING_DAYS_PER_WEEK / 7.0)


def _trading_minutes_per_day(instrument: str) -> int:
    return _TRADING_MINUTES_PER_DAY.get((instrument or "").upper(), _DEFAULT_TRADING_MINUTES_PER_DAY)


def _token_to_bars(instrument: str, timeframe: str, token: str) -> int:
    """Trading-hours-aware DURATION token → candle count, plus a small margin.

    trading_days × trading_minutes_per_day / tf_minutes. For D1 this collapses to
    the trading-day count. Returns at least 1 bar.
    """
    tf_min = _TF_MINUTES.get(timeframe.upper())
    if not tf_min:
        raise ValueError(f"Unknown timeframe {timeframe!r}")
    trading_minutes = _token_trading_days(token) * _trading_minutes_per_day(instrument)
    bars = math.ceil((trading_minutes / tf_min) * _TARGET_MARGIN)
    return max(1, bars)


@dataclass(frozen=True)
class LookbackDepth:
    """Resolved storage depth for one (instrument, timeframe) combo."""

    instrument: str
    timeframe: str
    duration_str: str
    duration: timedelta
    target_bars: int


# --------------------------------------------------------------------------- #
# Config loading (cached; point at a fixture via SENTINEL_LOOKBACK_DEPTHS_PATH)
# --------------------------------------------------------------------------- #
def _default_path() -> Path:
    env = os.environ.get("SENTINEL_LOOKBACK_DEPTHS_PATH")
    if env:
        return Path(env)
    # src/intelligence/lookback_config.py → repo root is two parents up.
    return Path(__file__).resolve().parents[2] / "config" / "lookback_depths.json"


@dataclass(frozen=True)
class _Config:
    # Ordered: display order (M1, M5, M15, H1, H4, D1) and perimeter order.
    timeframes: Tuple[str, ...]
    instruments: Tuple[str, ...]
    default_depths: Dict[str, str]
    overrides: Dict[str, Dict[str, str]]


_cache: Optional[_Config] = None


def _load() -> _Config:
    global _cache
    if _cache is not None:
        return _cache
    path = _default_path()
    raw = json.loads(path.read_text(encoding="utf-8"))
    default_depths: Dict[str, str] = dict(raw.get("default", {}))
    if not default_depths:
        raise ValueError(f"lookback_depths.json at {path} has no 'default' depths")
    instruments_raw: Dict[str, Dict[str, str]] = dict(raw.get("instruments", {}))
    timeframes = tuple(default_depths.keys())
    instruments = tuple(k.upper() for k in instruments_raw.keys())
    overrides = {k.upper(): dict(v or {}) for k, v in instruments_raw.items()}
    # Validate every depth token up front — fail loud on a malformed config.
    for tf, tok in default_depths.items():
        parse_duration(tok)
        if tf.upper() not in _TF_MINUTES:
            raise ValueError(f"lookback_depths.json: unknown timeframe {tf!r}")
    for inst, ov in overrides.items():
        for tf, tok in ov.items():
            parse_duration(tok)
    _cache = _Config(timeframes, instruments, default_depths, overrides)
    return _cache


def reset_cache() -> None:
    """Force a reload (tests that point at a fixture)."""
    global _cache
    _cache = None


# --------------------------------------------------------------------------- #
# Perimeter (single source of truth)
# --------------------------------------------------------------------------- #
def supported_instruments() -> Tuple[str, ...]:
    return _load().instruments


def supported_timeframes() -> Tuple[str, ...]:
    """All configured timeframes, INCLUDING M1 regardless of the gate."""
    return _load().timeframes


def is_m1_enabled() -> bool:
    return os.environ.get(_M1_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def enabled_timeframes() -> Tuple[str, ...]:
    """Configured timeframes minus M1 when the gate is off."""
    m1_on = is_m1_enabled()
    return tuple(tf for tf in _load().timeframes if tf.upper() != "M1" or m1_on)


def enabled_combos() -> Tuple[Tuple[str, str], ...]:
    """(instrument, timeframe) pairs in perimeter order, M1 gated.

    Instrument-major, timeframe order as configured — the deterministic scan
    order the scanner and scheduler consume.
    """
    tfs = enabled_timeframes()
    return tuple((inst, tf) for inst in supported_instruments() for tf in tfs)


# --------------------------------------------------------------------------- #
# Depth resolution
# --------------------------------------------------------------------------- #
def depth_for(instrument: str, timeframe: str) -> LookbackDepth:
    cfg = _load()
    inst = (instrument or "").upper()
    tf = (timeframe or "").upper()
    token = cfg.overrides.get(inst, {}).get(tf) or cfg.default_depths.get(tf)
    if token is None:
        raise ValueError(f"No lookback depth configured for timeframe {tf!r}")
    duration = parse_duration(token)
    return LookbackDepth(
        instrument=inst,
        timeframe=tf,
        duration_str=token,
        duration=duration,
        target_bars=_token_to_bars(inst, tf, token),
    )


def target_bars(instrument: str, timeframe: str) -> int:
    """Candle count to hold in storage for this combo, from its configured DURATION."""
    return depth_for(instrument, timeframe).target_bars
