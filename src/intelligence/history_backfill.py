"""LB-1 — historical backfill to the configured storage depth, per combo.

The analysis happens ONCE over deep history; this task is what puts that deep
history in the cache. It is deliberately thin and safe:

  * DEPTH FROM CONFIG — each combo is filled to ``lookback_config.target_bars``,
    expressed in DURATION per (instrument, timeframe). Never a global constant.
  * IDEMPOTENT — writes go through ``upsert_candles`` (INSERT OR REPLACE), so a
    second run duplicates nothing.
  * RESUMABLE — a combo already deep enough AND up to date is skipped, so an
    interrupted run resumes without re-downloading finished combos.
  * QUOTA-AWARE — one ``time_series`` request per combo (every target fits in a
    single ``outputsize`` ≤ 5000 on the free plan; see the audit). Request
    throttling itself is delegated to the provider's embedded rate limiter.
  * MC-1 AWARE — freshness is measured against ``market_aware_expected_close``,
    which is frozen while the market is closed. A combo that already holds the
    last expected close is therefore not re-fetched on a weekend: zero calls for
    a period with no new candle. Twelve Data only returns real (market-open)
    candles, so we never request data for a closed period in the first place.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

from src.intelligence import lookback_config, market_calendar

logger = logging.getLogger(__name__)

# Free-plan ceiling for a single time_series request. Every LB-1 target depth is
# below this (deepest is H1 6mo ≈ 3.3k bars), so one request fills a combo. A
# future target beyond this would need date-windowed pagination — guarded below.
MAX_OUTPUTSIZE = 5000

# Freshness tolerance: a combo is "up to date" if its newest candle is within
# this many timeframe-intervals of the market-aware expected close. Generous on
# purpose — catching up the final candle or two is the live scheduler's job, not
# the backfill's; the backfill owns DEPTH.
_FRESH_TOLERANCE_BARS = 2

# Accept a small shortfall vs the target (holidays, plan gaps) as "deep enough",
# so a combo the plan simply cannot fill deeper is not re-fetched every run.
_DEEP_ENOUGH_FRACTION = 0.9


class _Provider(Protocol):
    def fetch_candles(self, symbol: str, timeframe: str, count: int): ...


class _PagingProvider(Protocol):
    def fetch_candles_until(
        self, symbol: str, timeframe: str, count: int, end_date: Optional[str] = None
    ): ...


class _Store(Protocol):
    def get_coverage(self, instrument: str, timeframe: str): ...
    def upsert_candles(self, instrument: str, timeframe: str, candles) -> int: ...


@dataclass(frozen=True)
class BackfillResult:
    instrument: str
    timeframe: str
    target_bars: int
    requested: int          # outputsize asked of the provider (0 when skipped)
    fetched: int            # candles returned
    upserted: int           # rows affected in the cache
    skipped: bool
    reason: str             # "filled" | "already_complete" | "market_closed_complete"

    @property
    def calls(self) -> int:
        return 0 if self.skipped else 1


def _to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def _is_up_to_date(
    instrument: str, timeframe: str, newest_ts: Optional[datetime], now: datetime, calendar
) -> bool:
    """True when the cache already holds (near) the last expected close.

    Uses the market-aware expected close, which is frozen while the market is
    closed — so a Friday-fresh cache reads as up to date all weekend.
    """
    if newest_ts is None:
        return False
    expected = _to_utc(calendar.market_aware_expected_close(instrument, timeframe, now))
    newest = _to_utc(newest_ts)
    tf_min = lookback_config._TF_MINUTES.get(timeframe.upper(), 1)
    tolerance_s = _FRESH_TOLERANCE_BARS * tf_min * 60
    return (expected - newest).total_seconds() <= tolerance_s


def backfill_combo(
    provider: _Provider,
    store: _Store,
    instrument: str,
    timeframe: str,
    *,
    now: Optional[datetime] = None,
    target_bars: Optional[int] = None,
    force: bool = False,
    calendar=market_calendar,
) -> BackfillResult:
    """Fill one combo to its configured depth. At most one provider request."""
    now = _to_utc(now) or datetime.now(timezone.utc)
    target = target_bars if target_bars is not None else lookback_config.target_bars(instrument, timeframe)

    coverage = store.get_coverage(instrument, timeframe)
    deep_enough = coverage.count >= int(target * _DEEP_ENOUGH_FRACTION)
    up_to_date = _is_up_to_date(instrument, timeframe, coverage.newest_ts, now, calendar)

    if not force and deep_enough and up_to_date:
        market_closed = calendar.calendar_state(instrument, now) != market_calendar.OPEN
        reason = "market_closed_complete" if market_closed else "already_complete"
        return BackfillResult(
            instrument, timeframe, target, requested=0, fetched=0, upserted=0,
            skipped=True, reason=reason,
        )

    request_size = min(target, MAX_OUTPUTSIZE)
    if target > MAX_OUTPUTSIZE:
        # No current target hits this; log rather than silently truncate depth.
        logger.warning(
            "backfill %s %s: target %d exceeds single-request cap %d — filling to "
            "%d only (date-windowed pagination not implemented).",
            instrument, timeframe, target, MAX_OUTPUTSIZE, MAX_OUTPUTSIZE,
        )

    candles = provider.fetch_candles(instrument, timeframe, request_size)
    upserted = store.upsert_candles(instrument, timeframe, candles)
    logger.info(
        "backfill %s %s: requested %d, fetched %d, upserted %d (target %d)",
        instrument, timeframe, request_size, len(candles), upserted, target,
    )
    return BackfillResult(
        instrument, timeframe, target, requested=request_size, fetched=len(candles),
        upserted=upserted, skipped=False, reason="filled",
    )


def deep_backfill_combo(
    provider: _PagingProvider,
    store: _Store,
    instrument: str,
    timeframe: str,
    *,
    target_bars: int,
    page: int = MAX_OUTPUTSIZE,
    max_pages: int = 40,
) -> dict:
    """Fill one combo DEEP by walking history BACKWARD one ``page`` (≤5000) at a
    time via ``end_date`` pagination — the single-request cap makes plain
    ``backfill_combo`` top out at ~52 days of M15, far short of the months the
    publication measures need. Idempotent (upsert), quota-aware (the provider's
    rate limiter throttles), and self-terminating: it stops when the target is
    reached, the provider returns nothing older (depth exhausted), or ``max_pages``.

    Returns a small progress dict; never raises (a failed page ends the walk)."""
    tf_min = lookback_config._TF_MINUTES.get(timeframe.upper(), 15)
    end_date: Optional[str] = None
    fetched = upserted = pages = 0
    earliest_seen: Optional[datetime] = None
    error: Optional[str] = None
    while fetched < target_bars and pages < max_pages:
        want = min(page, target_bars - fetched) or page
        try:
            candles = provider.fetch_candles_until(instrument, timeframe, want, end_date)
        except Exception as exc:  # surface the reason instead of a silent stop
            logger.exception("deep backfill page failed for %s %s", instrument, timeframe)
            error = f"{type(exc).__name__}: {exc}"[:300]
            break
        pages += 1
        if not candles:
            if pages == 1:  # empty FIRST page = a real problem, not depth exhaustion
                error = "provider returned no candles for the first page"
            break
        try:
            upserted += store.upsert_candles(instrument, timeframe, candles)
        except Exception as exc:
            logger.exception("deep backfill upsert failed for %s %s", instrument, timeframe)
            error = f"upsert: {type(exc).__name__}: {exc}"[:300]
            break
        fetched += len(candles)
        page_earliest = min(_to_utc(c.ts) for c in candles)
        if earliest_seen is not None and page_earliest >= earliest_seen:
            break  # provider returned no OLDER bar → historical depth exhausted
        earliest_seen = page_earliest
        end_date = (page_earliest - timedelta(minutes=tf_min)).strftime("%Y-%m-%d %H:%M:%S")
    result = {
        "instrument": instrument, "timeframe": timeframe, "pages": pages,
        "fetched": fetched, "upserted": upserted,
        "oldest": earliest_seen.isoformat() if earliest_seen else None,
        "error": error,
    }
    logger.info("deep backfill %s %s: %s", instrument, timeframe, result)
    return result


def backfill_all(
    provider: _Provider,
    store: _Store,
    *,
    combos: Optional[Sequence[Tuple[str, str]]] = None,
    now: Optional[datetime] = None,
    force: bool = False,
    calendar=market_calendar,
) -> List[BackfillResult]:
    """Backfill every combo (default: the enabled LB-1 perimeter, M1 gated off).

    Deterministic order. Per-request throttling is handled by the provider's rate
    limiter, so this simply walks the list; an interrupted run is resumed by
    re-invoking (finished combos skip).
    """
    todo: Iterable[Tuple[str, str]] = combos if combos is not None else lookback_config.enabled_combos()
    results: List[BackfillResult] = []
    for instrument, timeframe in todo:
        try:
            results.append(
                backfill_combo(
                    provider, store, instrument, timeframe,
                    now=now, force=force, calendar=calendar,
                )
            )
        except Exception:  # one bad combo must not abort the whole fill
            logger.exception("backfill failed for %s %s", instrument, timeframe)
    return results
