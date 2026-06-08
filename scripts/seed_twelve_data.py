"""Seed the local candles cache from Twelve Data for XAU + EURUSD.

Usage:
    python scripts/seed_twelve_data.py                          # default 6 combos
    python scripts/seed_twelve_data.py --dry-run                # no API call
    python scripts/seed_twelve_data.py --instrument XAUUSD --timeframe M15

Requires ``TWELVE_DATA_API_KEY`` in the environment (unless ``--dry-run``).
Rate-limit compliance is delegated to ``TwelveDataProvider`` (8 req/min, 800/day).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Iterable, List, Tuple

DEFAULT_INSTRUMENTS: List[str] = ["XAUUSD", "EURUSD"]
DEFAULT_TIMEFRAMES: List[str] = ["M15", "H1", "H4"]
DEFAULT_LOOKBACK: int = 100


def _combos(
    instruments: Iterable[str], timeframes: Iterable[str]
) -> List[Tuple[str, str]]:
    return [(i, t) for i in instruments for t in timeframes]


def _seed_one(
    provider,
    store,
    instrument: str,
    timeframe: str,
    lookback: int,
    dry_run: bool,
) -> int:
    if dry_run:
        print(f"[seed] {instrument} {timeframe}: dry-run (no API call)")
        return 0
    print(f"[seed] {instrument} {timeframe}: fetching {lookback} candles...")
    candles = provider.fetch_candles(instrument, timeframe, lookback)
    affected = store.upsert_candles(instrument, timeframe, candles)
    print(
        f"[seed] {instrument} {timeframe}: "
        f"{len(candles)} candles fetched, {affected} rows affected"
    )
    return affected


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Seed candles cache from Twelve Data"
    )
    parser.add_argument("--instrument", help="Single instrument (default: all 2)")
    parser.add_argument("--timeframe", help="Single timeframe (default: all 3)")
    parser.add_argument(
        "--lookback", type=int, default=DEFAULT_LOOKBACK,
        help=f"Bars per combo (default: {DEFAULT_LOOKBACK})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Iterate over the combos without any API call",
    )
    args = parser.parse_args(argv)

    instruments = [args.instrument] if args.instrument else DEFAULT_INSTRUMENTS
    timeframes = [args.timeframe] if args.timeframe else DEFAULT_TIMEFRAMES
    combos = _combos(instruments, timeframes)

    if not args.dry_run and not os.environ.get("TWELVE_DATA_API_KEY"):
        print(
            "ERROR: TWELVE_DATA_API_KEY is not set. "
            "Add it to your .env or shell environment, then re-run.",
            file=sys.stderr,
        )
        return 1

    # Deferred imports so --dry-run can run in a stripped environment
    from src.storage.candles_cache_store import CandlesCacheStore

    if args.dry_run:
        provider = None
    else:
        from src.intelligence.data_providers import TwelveDataProvider
        provider = TwelveDataProvider()

    store = CandlesCacheStore()

    start = time.monotonic()
    total_affected = 0
    for instrument, timeframe in combos:
        total_affected += _seed_one(
            provider, store, instrument, timeframe, args.lookback, args.dry_run,
        )
    elapsed = time.monotonic() - start
    print(
        f"[seed] DONE: {len(combos)} combinations, "
        f"{total_affected} rows affected total, {elapsed:.1f}s elapsed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
