"""LB-1 — deep historical backfill to the configured storage depth.

Fills the candles cache to ``config/lookback_depths.json`` depth for every enabled
combo (M1 gated off by default). Idempotent, resumable, quota-aware (one request
per combo; the provider's rate limiter throttles). Reuses the MC-1 calendar, so a
combo already at the last expected close is not re-fetched on a weekend.

Usage:
    python -m scripts.backfill_history                     # all enabled combos
    python -m scripts.backfill_history --dry-run           # plan only, no API call
    python -m scripts.backfill_history --instrument XAUUSD --timeframe H1
    python -m scripts.backfill_history --force             # re-fetch even if complete

Requires ``TWELVE_DATA_API_KEY`` in the environment (unless ``--dry-run``).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import List, Optional, Tuple

from src.intelligence import history_backfill as hb
from src.intelligence import lookback_config


def _combos(instrument: Optional[str], timeframe: Optional[str]) -> List[Tuple[str, str]]:
    combos = list(lookback_config.enabled_combos())
    if instrument:
        combos = [c for c in combos if c[0] == instrument.upper()]
    if timeframe:
        combos = [c for c in combos if c[1] == timeframe.upper()]
    return combos


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Deep historical backfill (LB-1)")
    parser.add_argument("--instrument", help="Limit to one instrument")
    parser.add_argument("--timeframe", help="Limit to one timeframe")
    parser.add_argument("--force", action="store_true", help="Re-fetch even if a combo is complete")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without any API call")
    parser.add_argument("--no-structure", action="store_true", help="Skip seeding persisted structure after the fill")
    args = parser.parse_args(argv)

    combos = _combos(args.instrument, args.timeframe)
    if not combos:
        print("[backfill] no matching enabled combos (M1 is gated off by default)")
        return 1

    print(f"[backfill] {len(combos)} combo(s) to fill:")
    total_bars = 0
    for inst, tf in combos:
        depth = lookback_config.depth_for(inst, tf)
        total_bars += depth.target_bars
        print(f"  {inst} {tf:4} -> {depth.duration_str:5} (~{depth.target_bars} bars)")
    print(f"[backfill] estimated peak: {len(combos)} request(s) -- cap is 800/day")

    if args.dry_run:
        print("[backfill] dry-run: no request made")
        return 0

    from src.intelligence.data_providers import TwelveDataProvider
    from src.storage.candles_cache_store import CandlesCacheStore

    provider = TwelveDataProvider()
    store = CandlesCacheStore()
    now = datetime.now(timezone.utc)

    results = hb.backfill_all(provider, store, combos=combos, now=now, force=args.force)

    calls = sum(r.calls for r in results)
    filled = sum(1 for r in results if not r.skipped)
    print(f"[backfill] done: {filled} filled, {len(results) - filled} skipped, {calls} request(s) used")
    for r in results:
        if r.skipped:
            print(f"  {r.instrument} {r.timeframe:4} skipped ({r.reason})")
        else:
            print(f"  {r.instrument} {r.timeframe:4} fetched {r.fetched}, upserted {r.upserted} (target {r.target_bars})")

    if not args.no_structure:
        _seed_structure(store, combos)
    return 0


def _seed_structure(candles_store, combos) -> None:
    """Analyse the freshly-filled history ONCE and persist its current structure.

    One engine run per combo on the trailing detection window seeds the zone/event
    state; the deep event journal then accumulates going forward via live
    refreshes. No provider call (pure cache read + detection)."""
    from src.intelligence.incremental_detection import IncrementalDetector
    from src.storage.structure_store import StructureStore

    det = IncrementalDetector(StructureStore())
    print("[backfill] seeding persisted structure (one engine pass per combo)...")
    for inst, tf in combos:
        candles = candles_store.get_last_n_candles(inst, tf, det._window)
        if det.refresh(inst, tf, candles):
            print(f"  {inst} {tf:4} structure seeded ({len(candles)} bars)")
        else:
            print(f"  {inst} {tf:4} skipped (too few candles)")


if __name__ == "__main__":
    raise SystemExit(main())
