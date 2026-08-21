"""MKT-1 — the single source of truth for the tradeable market catalogue.

Every market enumeration that used to be recopied across the product (the
frontend perimeter, instrument labels, price decimals, the calendar market
filters, the backend supported-instruments list) derives from
``config/markets.json`` through this module. The frontend consumes the SAME file
via a generated module (``webapp/lib/markets.generated.ts``). Adding an 81st
market is one entry in the JSON — nothing else enumerates markets.

Dependency-light on purpose (json + stdlib only): it is imported by the LB-1
lookback perimeter and the API. The news-attachment rule (which currencies'
macro drives a market) is NOT stored here — it lives in
``config/event_market_map.json`` (the single consultable news rule read by
``calendar_service``); :func:`driver_currencies` references that file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_ALLOWED_TYPES = frozenset({"metal", "fx", "crypto", "index"})


@dataclass(frozen=True)
class MarketSpec:
    id: str
    label: str            # FR baseline label (i18n key = markets.<id> on the front)
    symbol: str           # data-provider (Twelve Data) symbol
    type: str             # metal | fx | crypto | index
    price_decimals: int
    glyph: str            # short mono badge for the compact market row
    timeframes: Tuple[str, ...]  # perimeter timeframe ids served for this market
    index: int            # display order (0 = first in the column)


def _default_path() -> Path:
    env = os.environ.get("SENTINEL_MARKETS_PATH")
    if env:
        return Path(env)
    # src/intelligence/market_registry.py → repo root is two parents up.
    return Path(__file__).resolve().parents[2] / "config" / "markets.json"


_cache: Optional[Tuple[MarketSpec, ...]] = None


def _load() -> Tuple[MarketSpec, ...]:
    global _cache
    if _cache is not None:
        return _cache
    raw = json.loads(_default_path().read_text(encoding="utf-8"))
    specs = []
    for i, m in enumerate(raw["markets"]):
        mtype = str(m["type"])
        if mtype not in _ALLOWED_TYPES:
            raise ValueError(
                f"markets.json: market {m.get('id')!r} has unknown type {mtype!r} "
                f"(allowed: {sorted(_ALLOWED_TYPES)})"
            )
        specs.append(MarketSpec(
            id=str(m["id"]).upper(),
            label=str(m["label"]),
            symbol=str(m["symbol"]),
            type=mtype,
            price_decimals=int(m["priceDecimals"]),
            glyph=str(m["glyph"]),
            timeframes=tuple(str(t).upper() for t in m["timeframes"]),
            index=i,
        ))
    _cache = tuple(specs)
    return _cache


def reset_cache() -> None:
    """Force a reload (tests pointing at a fixture)."""
    global _cache
    _cache = None


# --------------------------------------------------------------------------- #
# Lookups
# --------------------------------------------------------------------------- #
def all_specs() -> Tuple[MarketSpec, ...]:
    return _load()


def _by_id() -> Dict[str, MarketSpec]:
    return {s.id: s for s in _load()}


def has(market: str) -> bool:
    return (market or "").upper() in _by_id()


def spec(market: str) -> MarketSpec:
    s = _by_id().get((market or "").upper())
    if s is None:
        raise KeyError(f"Unknown market {market!r} (not in the registry)")
    return s


def all_ids() -> Tuple[str, ...]:
    """Every market id in display order — the supported-instrument perimeter."""
    return tuple(s.id for s in _load())


def label(market: str) -> str:
    return spec(market).label


def price_decimals(market: str) -> int:
    return spec(market).price_decimals


def timeframes(market: str) -> Tuple[str, ...]:
    """Perimeter timeframe ids served for this market (M1 gate applied elsewhere)."""
    return spec(market).timeframes


# --------------------------------------------------------------------------- #
# News-attachment rule — REFERENCED, never duplicated
# --------------------------------------------------------------------------- #
def _event_map_path() -> Path:
    env = os.environ.get("CALENDAR_EVENT_MAP_PATH")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "config" / "event_market_map.json"


def driver_currencies(market: str) -> Tuple[str, ...]:
    """The currencies whose macro news attach to ``market``.

    The single source is ``config/event_market_map.json`` (the consultable news
    rule) — this is a read-through reference, not a second copy. Returns an empty
    tuple if the market has no declared drivers (its calendar is then empty, per
    the news rule: an event attached to no followed market is dropped).
    """
    mid = (market or "").upper()
    try:
        raw = json.loads(_event_map_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    entry = (raw.get("markets", {}) or {}).get(mid, {}) or {}
    return tuple(str(c).upper() for c in entry.get("driver_currencies", []))
