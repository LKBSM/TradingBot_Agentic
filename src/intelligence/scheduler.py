"""MarketReadingScheduler — hybrid-mode recurring regeneration (Chantier 3).

Architecture doc §3.1 ("Mode hybride") + §3.3 ("Scheduler APScheduler").

Hybrid mode has three behaviours, of which this scheduler implements the
last two (the *lazy first access* is already handled by the endpoint +
MarketReadingAssembler in Chantier 2):

  - **Lazy first access** — first user hit on (instrument, timeframe) marks
    the combination active and generates on demand. (Chantier 2.)
  - **Continu après premier accès** — a recurring tick regenerates the
    MarketReading of every *active* combination whenever a new candle has
    closed since the stored reading.
  - **Arrêt automatique 24h** — a combination not accessed for
    ``auto_stop_hours`` is simply no longer returned by
    ``get_active_combinations(since=...)``, so the tick stops regenerating it.
    No explicit teardown is needed — the access window drives everything.

An optional ``always_warm`` set is unioned into every tick on top of the
access-driven active set: those combinations (e.g. the fixed Conditions
Scanner perimeter) are regenerated whenever a new candle closes, even with
zero recent user access, so the scanner never opens onto a missing/aged
reading.

The tick is exception-isolated per combination AND globally: one failing
combination never aborts the others, and any tick-level error is swallowed
(logged) so the BackgroundScheduler thread never dies.

APScheduler is imported lazily inside ``__init__`` so importing this module
never hard-fails when the optional dependency is absent (e.g. a minimal test
env); only constructing the scheduler requires it.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, Optional, Tuple

from src.intelligence.market_reading_assembler import expected_last_candle_close

logger = logging.getLogger(__name__)

_JOB_ID = "market_reading_tick"


def _parse_iso(value: Any) -> Optional[datetime]:
    """Parse an ISO-8601 string/datetime into aware UTC, or None if unparsable."""
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class MarketReadingScheduler:
    """Recurring regenerator for active (instrument, timeframe) combinations."""

    DEFAULT_TICK_INTERVAL_SECONDS = 60
    DEFAULT_AUTO_STOP_HOURS = 24

    def __init__(
        self,
        assembler: Any,
        readings_store: Any,
        candles_store: Any = None,
        tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS,
        auto_stop_hours: int = DEFAULT_AUTO_STOP_HOURS,
        always_warm: Optional[Iterable[Tuple[str, str]]] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler

        self._assembler = assembler
        self._readings_store = readings_store
        self._candles_store = candles_store
        self._tick_interval_seconds = tick_interval_seconds
        self._auto_stop_hours = auto_stop_hours
        # Combinations kept warm regardless of recent user access — e.g. the
        # fixed perimeter the Conditions Scanner reads. Without this, a combo
        # nobody opened in the last ``auto_stop_hours`` falls out of the active
        # set and its reading goes stale (or never gets generated at all), so
        # the scanner would surface ``no_reading_yet`` / aged readings on a cold
        # open. Normalised to a de-duplicated, order-preserving tuple of
        # (instrument, timeframe) pairs.
        _seen: set = set()
        _warm: list = []
        for i, tf in (always_warm or ()):
            key = (str(i), str(tf))
            if key not in _seen:
                _seen.add(key)
                _warm.append(key)
        self._always_warm: Tuple[Tuple[str, str], ...] = tuple(_warm)
        self._clock = clock
        self._scheduler = BackgroundScheduler()

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Register the recurring tick and start the background thread."""
        from apscheduler.triggers.interval import IntervalTrigger

        self._scheduler.add_job(
            self.tick,
            IntervalTrigger(seconds=self._tick_interval_seconds),
            id=_JOB_ID,
            # Never let two ticks overlap; collapse missed runs into one.
            max_instances=1,
            coalesce=True,
            replace_existing=True,
        )
        self._scheduler.start()
        logger.info(
            "MarketReadingScheduler started (tick=%ds, auto_stop=%dh)",
            self._tick_interval_seconds, self._auto_stop_hours,
        )

    def stop(self) -> None:
        """Stop the background thread. Safe to call even if not running."""
        try:
            if self._scheduler.running:
                self._scheduler.shutdown(wait=False)
                logger.info("MarketReadingScheduler stopped")
        except Exception:  # pragma: no cover — defensive on double-shutdown
            logger.exception("MarketReadingScheduler shutdown failed")

    @property
    def running(self) -> bool:
        return bool(self._scheduler.running)

    # ------------------------------------------------------------------ #
    # Tick
    # ------------------------------------------------------------------ #
    def tick(self) -> int:
        """One scheduling pass. Returns the number of combinations regenerated.

        Safe to call directly (used by tests) — the scheduler thread calls
        the same method. Any error is contained so the recurring job survives.
        """
        regenerated = 0
        try:
            now = self._clock()
            since = now - timedelta(hours=self._auto_stop_hours)
            active = self._readings_store.get_active_combinations(since=since)
        except Exception:
            logger.exception("scheduler tick: failed to read active combinations")
            # The always-warm set must survive a failed active-set read, so the
            # scanner perimeter keeps regenerating even then.
            active = []

        # Union the access-driven active set with the always-warm perimeter,
        # preserving the active order first (deterministic), then appending any
        # always-warm combo not already queued. A combo in both runs once.
        seen: set = set()
        combos: list = []
        for i, tf in active:
            key = (str(i), str(tf))
            if key not in seen:
                seen.add(key)
                combos.append(key)
        for key in self._always_warm:
            if key not in seen:
                seen.add(key)
                combos.append(key)

        for instrument, timeframe in combos:
            try:
                if self._needs_regeneration(instrument, timeframe, now):
                    self._assembler.get_or_generate(instrument, timeframe)
                    regenerated += 1
            except Exception:
                # One combination failing must not abort the whole tick.
                logger.exception(
                    "scheduler tick: regeneration failed for %s/%s",
                    instrument, timeframe,
                )
        return regenerated

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _needs_regeneration(
        self, instrument: str, timeframe: str, now: datetime
    ) -> bool:
        """True when no current reading exists for the latest closed candle.

        Idempotence: once a tick regenerates a combination for ``expected_close``,
        a subsequent tick with no newer candle finds the stored reading current
        and skips it (no duplicate regeneration).
        """
        expected_close = expected_last_candle_close(timeframe, now)
        latest = self._readings_store.get_latest_reading(instrument, timeframe)
        if not latest:
            return True
        header = latest.get("header") if isinstance(latest, dict) else None
        stored_ts = header.get("candle_close_ts") if isinstance(header, dict) else None
        parsed = _parse_iso(stored_ts)
        if parsed is None:
            return True
        return parsed < expected_close


__all__ = ["MarketReadingScheduler"]
