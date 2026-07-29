"""TF-1 — the single source of truth for timeframe (unité de temps) attributes.

Every timeframe enumeration that used to be recopied across the backend (minutes
maps, provider strings, alignment key sets, per-unit relevance) derives from
``config/timeframes.json`` through this module. The frontend consumes the SAME
file via a generated module (``webapp/lib/timeframes.generated.ts``). Adding a
seventh unit is one edit to the JSON — nothing else enumerates timeframes.

Dependency-light on purpose (json + stdlib only): it is imported by the API
perimeter, the data provider, the assembler and the calendar.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

_SECONDS_PER_MINUTE = 60


@dataclass(frozen=True)
class TimeframeSpec:
    id: str
    minutes: int
    seconds: int
    provider: str          # Twelve Data interval string
    label_long: str        # FR baseline label (i18n key = timeframes.<id> on the front)
    date_format: str       # chart/axis date format hint
    perimeter: bool        # a tradeable unit (M1..D1)
    reference: bool        # served as a read-only reference series (W1)
    session_relevant: bool
    prev_levels_relevant: bool
    index: int             # ladder position (fastest = 0)


def _default_path() -> Path:
    env = os.environ.get("SENTINEL_TIMEFRAMES_PATH")
    if env:
        return Path(env)
    # src/intelligence/timeframe_registry.py → repo root is two parents up.
    return Path(__file__).resolve().parents[2] / "config" / "timeframes.json"


_cache: Optional[Tuple[TimeframeSpec, ...]] = None


def _load() -> Tuple[TimeframeSpec, ...]:
    global _cache
    if _cache is not None:
        return _cache
    raw = json.loads(_default_path().read_text(encoding="utf-8"))
    specs = []
    for i, e in enumerate(raw["timeframes"]):
        minutes = int(e["minutes"])
        specs.append(TimeframeSpec(
            id=str(e["id"]).upper(),
            minutes=minutes,
            seconds=minutes * _SECONDS_PER_MINUTE,
            provider=str(e["provider"]),
            label_long=str(e["labelLong"]),
            date_format=str(e["dateFormat"]),
            perimeter=bool(e["perimeter"]),
            reference=bool(e["reference"]),
            session_relevant=bool(e["sessionRelevant"]),
            prev_levels_relevant=bool(e["prevLevelsRelevant"]),
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
def all_specs() -> Tuple[TimeframeSpec, ...]:
    return _load()


def _by_id() -> Dict[str, TimeframeSpec]:
    return {s.id: s for s in _load()}


def has(tf: str) -> bool:
    return (tf or "").upper() in _by_id()


def spec(tf: str) -> TimeframeSpec:
    s = _by_id().get((tf or "").upper())
    if s is None:
        raise KeyError(f"Unknown timeframe {tf!r} (not in the registry)")
    return s


def minutes(tf: str) -> int:
    return spec(tf).minutes


def seconds(tf: str) -> int:
    return spec(tf).seconds


def provider_string(tf: str) -> str:
    return spec(tf).provider


def all_ids() -> Tuple[str, ...]:
    return tuple(s.id for s in _load())


def perimeter_ids() -> Tuple[str, ...]:
    return tuple(s.id for s in _load() if s.perimeter)


def reference_ids() -> Tuple[str, ...]:
    return tuple(s.id for s in _load() if s.reference)


def minutes_map() -> Dict[str, int]:
    """The canonical timeframe→minutes map (replaces the scattered copies)."""
    return {s.id: s.minutes for s in _load()}


def provider_map() -> Dict[str, str]:
    return {s.id: s.provider for s in _load()}


def upper_timeframes(tf: str) -> Tuple[str, ...]:
    """Perimeter units strictly SLOWER than ``tf`` (ascending). Empty at the top."""
    base = spec(tf)
    return tuple(s.id for s in _load() if s.perimeter and s.minutes > base.minutes)


def lower_timeframe(tf: str) -> Optional[str]:
    """The immediately faster perimeter unit, or None at the bottom."""
    base = spec(tf)
    faster = [s for s in _load() if s.perimeter and s.minutes < base.minutes]
    return faster[-1].id if faster else None


def alignment_timeframes(tf: str) -> Tuple[str, ...]:
    """Units ABOVE ``tf`` for the MTF alignment tile (TF-1 decision C).

    Perimeter units strictly above the viewed one; when the viewed unit is the
    top of the perimeter (D1) fall back to the reference unit above it (W1); above
    that, none — the tile then honestly reports "no higher unit".
    """
    above = upper_timeframes(tf)
    if above:
        return above
    base = spec(tf)
    return tuple(s.id for s in _load() if s.reference and s.minutes > base.minutes)


def is_session_relevant(tf: str) -> bool:
    return spec(tf).session_relevant


def is_prev_levels_relevant(tf: str) -> bool:
    return spec(tf).prev_levels_relevant
