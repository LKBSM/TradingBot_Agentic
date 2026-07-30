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

# Candle duration in minutes — derived from the single timeframe registry
# (TF-1), never a second copy. The registry is dependency-light, so importing it
# keeps this module importable by the API perimeter and the backfill task.
from src.intelligence import timeframe_registry as _tfreg

_TF_MINUTES: Dict[str, int] = _tfreg.minutes_map()

# A DURATION is turned into a bar count at CONTINUOUS density (1440 min/day, every
# calendar day) — the densest a series can possibly be. Real Twelve Data series
# are at most this dense (XAU/EUR intraday measured at ~24 bars/calendar-day, D1
# sparser), so a fetch sized this way always reaches AT LEAST the requested
# duration back — it never silently under-delivers the calendar span the config
# promises. Any shortfall then comes only from the provider's own history limit,
# which the audit reports per combo. (Sizing by trading-hours instead undershot
# the calendar duration; see AUDIT-lb-1-lookback-profond.)
_MINUTES_PER_CALENDAR_DAY = 1440

# A few bars of cushion so the boundary candle is comfortably included.
_TARGET_CUSHION_BARS = 4

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


def _token_to_bars(instrument: str, timeframe: str, token: str) -> int:
    """DURATION token → candle count at continuous density, so the fetch always
    reaches at least the requested calendar span back. Returns at least 1 bar.

    ``instrument`` is accepted for future per-instrument tuning (e.g. a
    reduced-hours index) but is unused today: continuous density is the safe
    ceiling for every current combo.
    """
    tf_min = _TF_MINUTES.get(timeframe.upper())
    if not tf_min:
        raise ValueError(f"Unknown timeframe {timeframe!r}")
    calendar_days = parse_duration(token).total_seconds() / 86400.0
    bars = math.ceil(calendar_days * _MINUTES_PER_CALENDAR_DAY / tf_min) + _TARGET_CUSHION_BARS
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
    """The tradeable perimeter — the single source is the timeframe registry
    (TF-1). Includes M1 regardless of the gate."""
    return _tfreg.perimeter_ids()


def is_m1_enabled() -> bool:
    return os.environ.get(_M1_ENABLE_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def enabled_timeframes() -> Tuple[str, ...]:
    """Perimeter timeframes minus M1 when the gate is off."""
    m1_on = is_m1_enabled()
    return tuple(tf for tf in supported_timeframes() if tf.upper() != "M1" or m1_on)


def enabled_combos() -> Tuple[Tuple[str, str], ...]:
    """(instrument, timeframe) pairs in perimeter order, M1 gated.

    Instrument-major, timeframe order as configured — the deterministic scan
    order the scanner and scheduler consume.
    """
    tfs = enabled_timeframes()
    return tuple((inst, tf) for inst in supported_instruments() for tf in tfs)


# Combos the LIVE scheduler actively polls (one provider request per closed
# candle). Distinct from enabled_combos: M5 polled natively costs ~288 req/day
# per symbol, which alone pushes 2 symbols to ~830/day — over the 800/day free
# cap. Until the live M5-base resample path is wired (M15/H1 derived from the M5
# feed instead of polled), M5 is excluded from live warming (its DEEP history is
# still backfilled and viewable; only its per-5-min live refresh waits). Set
# LB1_WARM_M5=1 to warm it natively once a plan/cadence allows.
_WARM_M5_ENV = "LB1_WARM_M5"


def live_warm_combos() -> Tuple[Tuple[str, str], ...]:
    """Combos safe to keep warm live within the free-plan quota (≈254 req/day)."""
    warm_m5 = os.environ.get(_WARM_M5_ENV, "").strip().lower() in ("1", "true", "yes", "on")
    return tuple(
        (inst, tf) for (inst, tf) in enabled_combos()
        if tf.upper() != "M5" or warm_m5
    )


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


# --------------------------------------------------------------------------- #
# Live analysis window (detection + trend tile + event journal), PER TIMEFRAME
# --------------------------------------------------------------------------- #
# Distinct from STORAGE depth (above) and from the deep detector window. The live
# reading used to analyse a FIXED 500-bar window on every unit — which spans ~5
# days on M15 but ~2 years on D1 (MT-D1 audit §D). Sizing it by a target CALENDAR
# span instead gives the fast units a comparable recent horizon (so M15 stops
# showing a 5-day keyhole) while the slow units keep their current depth via the
# floor. It stays ONE provider request (bounded < 5000) so the quota is unchanged
# — Twelve Data bills per request, not per bar. Overridable via env.
_ANALYSIS_TARGET_DAYS = 30.0       # ~1 month of recent context, all units
_ANALYSIS_MIN_BARS = 500           # floor: enough swings + unchanged slow-unit depth
_ANALYSIS_MAX_BARS = 3000          # ceiling: 1 request (< 5000) + bounded detection cost
_ANALYSIS_WINDOW_DAYS_ENV = "MARKET_READING_WINDOW_DAYS"


def analysis_window_bars(timeframe: str) -> int:
    """Bars of live context to analyse for a reading on ``timeframe``.

    A target calendar span (``MARKET_READING_WINDOW_DAYS`` days, default 30)
    converted to bars at continuous density, clamped to [500, 3000]. Fast units
    widen (M15 ≈ 2880 ≈ 1 month), slow units stay floored at 500 (D1 500 ≈ 2 y,
    unchanged). Raises on an unknown timeframe — a config bug we want surfaced.
    """
    tf_min = _TF_MINUTES.get(timeframe.upper())
    if not tf_min:
        raise ValueError(f"Unknown timeframe {timeframe!r}")
    try:
        days = float(os.environ.get(_ANALYSIS_WINDOW_DAYS_ENV, _ANALYSIS_TARGET_DAYS))
    except (TypeError, ValueError):
        days = _ANALYSIS_TARGET_DAYS
    bars = math.ceil(days * _MINUTES_PER_CALENDAR_DAY / tf_min)
    return max(_ANALYSIS_MIN_BARS, min(_ANALYSIS_MAX_BARS, bars))
