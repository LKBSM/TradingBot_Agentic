"""One-time deep-history SEED for the publication measures (NW-7).

The 4 questions need MONTHS of intraday (M15) price history per measured market.
Twelve Data caps a single request at ~52 days and the free daily quota is tight,
so a fresh deployment's candles.db starts too shallow to measure. This module
loads a BUNDLED, compressed seed (``seed_data/<INSTRUMENT>_<TF>.csv.gz``, shipped
IN THE IMAGE, outside the mounted /app/data so the mount does not shadow it) into
candles.db at boot — NO Twelve Data call, NO manual trigger.

Generic and idempotent: it runs for every measured market
(``publication_measures.measured_markets()``); a market whose candles.db already
spans enough is skipped, and adding a new market's history = dropping its
``seed_data/<SYMBOL>_M15.csv.gz`` file. After the seed, ``_pick_gold_m15`` uses
the deep cache and the measures compute; the live feed keeps extending it forward.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)

# A market is considered "already deep enough" (seed skipped) once candles.db
# spans at least this many days — enough for several monthly-release windows.
_MIN_SEED_SPAN_DAYS = 200


def _seed_dir() -> Path:
    # src/intelligence/measures_history_seed.py → parents[2] == repo root (/app).
    return Path(__file__).resolve().parents[2] / "seed_data"


def _seed_path(instrument: str, timeframe: str) -> Optional[Path]:
    p = _seed_dir() / f"{instrument.upper()}_{timeframe.upper()}.csv.gz"
    return p if p.exists() else None


class _SeedCandle:
    __slots__ = ("ts", "open", "high", "low", "close", "volume")

    def __init__(self, ts, o, h, l, c, v):
        self.ts = ts
        self.open = o
        self.high = h
        self.low = l
        self.close = c
        self.volume = v


def _parse_seed(path: Path) -> List[_SeedCandle]:
    """Parse a gzipped OHLC CSV (header Date/Datetime + Open/High/Low/Close[/Volume])
    into candle records with tz-aware UTC timestamps. Never raises — returns []."""
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.reader(fh)
            header = next(reader, None)
            if not header:
                return []
            idx = {name.strip().lower(): i for i, name in enumerate(header)}
            ts_i = next((idx[k] for k in ("date", "datetime", "time", "timestamp") if k in idx), None)
            need = ("open", "high", "low", "close")
            if ts_i is None or not all(k in idx for k in need):
                logger.warning("seed %s: unexpected header %s", path.name, header)
                return []
            o_i, h_i, l_i, c_i = (idx["open"], idx["high"], idx["low"], idx["close"])
            v_i = idx.get("volume")
            out: List[_SeedCandle] = []
            for row in reader:
                if len(row) <= c_i:
                    continue
                try:
                    ts = datetime.fromisoformat(row[ts_i].strip().replace("Z", "+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                    out.append(_SeedCandle(
                        ts,
                        float(row[o_i]), float(row[h_i]),
                        float(row[l_i]), float(row[c_i]),
                        float(row[v_i]) if v_i is not None and row[v_i] not in ("", None) else 0.0,
                    ))
                except (ValueError, TypeError, IndexError):
                    continue
            return out
    except (OSError, EOFError) as exc:  # pragma: no cover - defensive
        logger.warning("seed %s unreadable: %s", path, exc)
        return []


def _span_days(cov) -> float:
    o = getattr(cov, "oldest_ts", None)
    n = getattr(cov, "newest_ts", None)
    if o is None or n is None:
        return 0.0
    try:
        return (n - o).total_seconds() / 86400.0
    except Exception:  # pragma: no cover
        return 0.0


def seed_market(candles_store, instrument: str, timeframe: str = "M15") -> dict:
    """Seed ONE (instrument, timeframe) from its bundle if candles.db is shallow.
    Idempotent (skips when already deep or no seed file). Never raises."""
    result = {"instrument": instrument, "timeframe": timeframe, "seeded": 0, "skipped": None}
    try:
        cov = candles_store.get_coverage(instrument, timeframe)
        if getattr(cov, "count", 0) and _span_days(cov) >= _MIN_SEED_SPAN_DAYS:
            result["skipped"] = "already_deep"
            return result
        path = _seed_path(instrument, timeframe)
        if path is None:
            result["skipped"] = "no_seed_file"
            return result
        candles = _parse_seed(path)
        if not candles:
            result["skipped"] = "empty_seed"
            return result
        n = candles_store.upsert_candles(instrument, timeframe, candles)
        result["seeded"] = int(n)
        logger.info(
            "seeded %s %s from %s: %d bars (%s → %s)",
            instrument, timeframe, path.name, len(candles),
            candles[0].ts.isoformat(), candles[-1].ts.isoformat(),
        )
    except Exception:  # pragma: no cover - a seed must never break boot
        logger.exception("seed failed for %s %s", instrument, timeframe)
    return result


def seed_measures_history(candles_store, markets=None, timeframe: str = "M15") -> List[dict]:
    """Seed deep history for EVERY measured market (generic). Runs at boot; each
    market is seeded once (idempotent). Opt-out with SENTINEL_SEED_MEASURES=0."""
    if os.environ.get("SENTINEL_SEED_MEASURES", "1").strip().lower() in ("0", "false", "no", "off"):
        logger.info("measures history seed disabled (SENTINEL_SEED_MEASURES=0)")
        return []
    if markets is None:
        try:
            from src.intelligence.publication_measures import measured_markets
            markets = measured_markets()
        except Exception:  # pragma: no cover
            markets = []
    return [seed_market(candles_store, m, timeframe) for m in markets]


__all__ = ["seed_measures_history", "seed_market"]
