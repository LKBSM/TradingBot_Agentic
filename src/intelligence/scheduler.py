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
    DATA-3 splits that in two: a combination stays KNOWN for ``auto_stop_hours``
    but stops COSTING provider credits after the shorter
    ``on_demand_active_hours``. It is never removed — it is rebuilt on the next
    request — we simply stop refreshing a screen nobody is looking at.

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

import hashlib
import logging
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Iterable, List, Optional, Tuple

from src.intelligence.market_calendar import (
    CLOSED_HOLIDAY,
    calendar_state,
    market_aware_expected_close,
)
from src.intelligence.timeframe_registry import minutes_map as _tf_minutes_map

logger = logging.getLogger(__name__)

_JOB_ID = "market_reading_tick"

# --------------------------------------------------------------------------- #
# DATA-3 — spreading (the single biggest lever on the credit PEAK)
# --------------------------------------------------------------------------- #
# Twelve Data bills per MINUTE, not per day: only the peak decides the plan. The
# tick used to emit every due combination in the same minute, so at a boundary
# where several units close together the whole catalogue fired at once — measured
# at 2 markets (8 credits in one minute at midnight), projected at 100 markets to
# 500 credits in one minute, 9x a 55/min plan, while 95 % of the day's minutes
# spent nothing at all.
#
# Each combination now gets a STABLE offset inside the window that follows its
# close, derived from a hash of (instrument, timeframe): the same combo always
# lands in the same slot — reproducible, no drift, no thundering herd after a
# restart — while the catalogue spreads evenly across the window.
#
# The price of spreading is freshness: a reading may be regenerated up to its
# offset late, which the freshness badge already states honestly. The default
# window is HALF the candle's own duration — floored at 4 minutes so a fast unit
# still has room, capped at an hour so a daily candle is never held back for
# hours, and always kept under the next close so no candle is ever skipped.
#
# Measured effect at 100 markets (tools/data_budget/simulate_credits.py, S5):
# peak 500 -> 29 credits/min, total 22 300 -> 12 700 credits/day.
_SPREAD_FRACTION_ENV = "SENTINEL_SPREAD_FRACTION"
_SPREAD_MAX_MINUTES_ENV = "SENTINEL_SPREAD_MAX_MINUTES"
_SPREAD_MIN_MINUTES_ENV = "SENTINEL_SPREAD_MIN_MINUTES"
DEFAULT_SPREAD_FRACTION = 0.5
DEFAULT_SPREAD_MAX_MINUTES = 60
DEFAULT_SPREAD_MIN_MINUTES = 4

#: Combinations regenerated concurrently per tick. 1 = the historical sequential
#: behaviour, unchanged. Above 1 the tick fans out: at ~1 s of pipeline per
#: combination (measured), a sequential tick fits only ~40-70 combinations into
#: its 60 s period, and APScheduler's ``max_instances=1`` + ``coalesce`` then
#: silently drop the overflow. The credit limiter, not the loop, is the regulator.
_WORKERS_ENV = "SENTINEL_SCHEDULER_WORKERS"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def stable_slot(seed: str, window: int) -> int:
    """Deterministic slot in ``[0, window)`` for ``seed``.

    A hash, not ``random``: the same combination always lands in the same slot, so
    a restart re-uses the spread instead of re-bunching, and a test can assert it.
    """
    if window <= 1:
        return 0
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % window


def spread_offset_minutes(instrument: str, timeframe: str, period_minutes: int) -> int:
    """Stable slot, in minutes after the candle close, for this combination.

    The window is a fraction of the candle's own duration — floored so a fast
    unit still has somewhere to spread, capped so a slow one is not held back for
    hours, and always strictly under the next close so no candle is skipped.
    Returns 0 when spreading is disabled (``SENTINEL_SPREAD_FRACTION=0``) or the
    window is under a minute.
    """
    fraction = _env_float(_SPREAD_FRACTION_ENV, DEFAULT_SPREAD_FRACTION)
    if fraction <= 0 or period_minutes <= 1:
        return 0
    cap = _env_int(_SPREAD_MAX_MINUTES_ENV, DEFAULT_SPREAD_MAX_MINUTES)
    floor = _env_int(_SPREAD_MIN_MINUTES_ENV, DEFAULT_SPREAD_MIN_MINUTES)
    # A fraction of the period, but never so narrow that a fast unit has nowhere
    # to spread: on M5 half the period is 2 minutes, which at 100 markets still
    # means 50 credits in one of them. The floor widens it to just under the next
    # close (4 minutes on M5) — costly in relative terms, negligible in absolute:
    # the reading is at most 4 minutes behind, and the badge says so.
    window = int(min(max(period_minutes * fraction, floor), period_minutes - 1, cap))
    return stable_slot(f"{instrument}|{timeframe}", window)


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
    #: How often the safety probe may fire per combo while the market is closed
    #: for a HOLIDAY (MC-1) — catches a venue that reopens before the calendar
    #: says so. Weekends/daily breaks are deterministic and never probed, so a
    #: normal weekend makes zero outbound Twelve Data calls. 0 disables it.
    DEFAULT_SAFETY_POLL_SECONDS = 1800
    #: PERF-3 — minimum delay between two regenerations of a combo that reached
    #: this tick through the ACCESS-driven active set only, i.e. one a user opened
    #: that ``live_warm_combos()`` deliberately leaves OUT of the warm perimeter.
    #:
    #: Why this exists: M5 is excluded from the warm set precisely because polling
    #: it natively costs ~288 provider requests/market/day, which alone breaks the
    #: 800/day free cap (see ``lookback_config.live_warm_combos``). But a combo a
    #: user opens ONCE is marked active and stays active for ``auto_stop_hours``,
    #: so the tick picked M5 back up at its full native cadence and re-opened the
    #: very hole the warm perimeter was closing:
    #:
    #:   warm baseline (2 markets x M15/H1/H4/D1) ......... 254 req/day
    #:   + M5 at native cadence (2 x 288) ................. 576 req/day
    #:   = 830 req/day  ->  OVER the 800/day cap
    #:
    #:   + M5 throttled to one refresh per 900 s (2 x 96) .. 192 req/day
    #:   = 446 req/day  ->  inside the cap
    #:
    #: The combo stays fully AVAILABLE and keeps advancing — just at a floor
    #: cadence instead of at every closed candle. The lag is never silent: the
    #: freshness badge is derived from ``market_status`` vs the stored
    #: ``candle_close_ts``, so a reading a few minutes behind says so on screen.
    #: 0 disables the throttle (every active combo polled at native cadence).
    DEFAULT_ON_DEMAND_MIN_INTERVAL_SECONDS = 900
    #: DATA-3 — how long a combination a user OPENED keeps being refreshed after
    #: their last visit. ``auto_stop_hours`` (24 h) governs how long it stays
    #: *known*; this shorter window governs how long we keep SPENDING CREDITS on
    #: it. One glance at M5 used to buy 24 h of refreshes (~96 requests per market
    #: per day at the 900 s floor) for a screen nobody was looking at any more.
    #: The combination is never removed: it stays fully available and is rebuilt
    #: on the next request. Set ``SENTINEL_ON_DEMAND_ACTIVE_HOURS`` to restore the
    #: previous behaviour (= ``auto_stop_hours``).
    DEFAULT_ON_DEMAND_ACTIVE_HOURS = 2

    def __init__(
        self,
        assembler: Any,
        readings_store: Any,
        candles_store: Any = None,
        tick_interval_seconds: int = DEFAULT_TICK_INTERVAL_SECONDS,
        auto_stop_hours: int = DEFAULT_AUTO_STOP_HOURS,
        always_warm: Optional[Iterable[Tuple[str, str]]] = None,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
        safety_poll_seconds: int = DEFAULT_SAFETY_POLL_SECONDS,
        on_demand_min_interval_seconds: int = DEFAULT_ON_DEMAND_MIN_INTERVAL_SECONDS,
        on_demand_active_hours: Optional[int] = None,
        workers: Optional[int] = None,
        spread: bool = True,
    ) -> None:
        from apscheduler.schedulers.background import BackgroundScheduler

        self._assembler = assembler
        self._readings_store = readings_store
        self._candles_store = candles_store
        self._tick_interval_seconds = tick_interval_seconds
        self._auto_stop_hours = auto_stop_hours
        self._safety_poll_seconds = safety_poll_seconds
        self._last_safety_probe: dict[Tuple[str, str], datetime] = {}
        self._on_demand_min_interval_seconds = on_demand_min_interval_seconds
        self._on_demand_active_hours = min(
            auto_stop_hours,
            on_demand_active_hours
            if on_demand_active_hours is not None
            else _env_int(
                "SENTINEL_ON_DEMAND_ACTIVE_HOURS", self.DEFAULT_ON_DEMAND_ACTIVE_HOURS
            ),
        )
        self._workers = workers if workers is not None else _env_int(_WORKERS_ENV, 1)
        #: Spreading is on by default. Turning it off restores the pre-DATA-3
        #: "everything fires the minute it is due" behaviour — an escape hatch,
        #: and what the tests that are ABOUT ordering ask for.
        self._spread = bool(spread)
        self._state_lock = threading.Lock()
        self._last_on_demand_regen: dict[Tuple[str, str], datetime] = {}
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
        warm_keys = set(self._always_warm)
        try:
            now = self._clock()
        except Exception:  # pragma: no cover — an injected clock that misbehaves
            logger.exception("scheduler tick: clock failed, falling back to UTC now")
            now = datetime.now(timezone.utc)
        try:
            since = now - timedelta(hours=self._auto_stop_hours)
            active = self._readings_store.get_active_combinations(since=since)
            # DATA-3: a combination nobody has opened recently stops COSTING
            # credits well before it stops being known. Warm combos are exempt —
            # they are the product's fixed perimeter, not an access artefact.
            if self._on_demand_active_hours < self._auto_stop_hours:
                recent = {
                    (str(i), str(tf))
                    for i, tf in self._readings_store.get_active_combinations(
                        since=now - timedelta(hours=self._on_demand_active_hours)
                    )
                }
                active = [
                    (i, tf)
                    for i, tf in active
                    if (str(i), str(tf)) in warm_keys or (str(i), str(tf)) in recent
                ]
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

        return self._run_combos(combos, warm_keys, now)

    def _run_combos(
        self, combos: List[Tuple[str, str]], warm_keys: set, now: datetime
    ) -> int:
        """Process one tick's combinations, sequentially or across N workers.

        Exception isolation is per combination either way: one failure never
        aborts the pass. With ``workers == 1`` this is byte-for-byte the previous
        sequential behaviour.
        """
        if self._workers <= 1 or len(combos) <= 1:
            return sum(self._process_combo(i, tf, warm_keys, now) for i, tf in combos)

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(
            max_workers=min(self._workers, len(combos)),
            thread_name_prefix="reading-tick",
        ) as pool:
            results = pool.map(
                lambda combo: self._process_combo(combo[0], combo[1], warm_keys, now),
                combos,
            )
            return sum(results)

    def _process_combo(
        self, instrument: str, timeframe: str, warm_keys: set, now: datetime
    ) -> int:
        """One combination. Returns 1 if it was regenerated, 0 otherwise."""
        regenerated = 0
        try:
            key = (instrument, timeframe)
            if self._needs_regeneration(instrument, timeframe, now):
                # PERF-3: a combo present ONLY through the access-driven active
                # set (not in the warm perimeter) is refreshed at a floor
                # cadence, not at every closed candle — otherwise one user
                # opening M5 puts the daily provider quota over its cap for the
                # next ``auto_stop_hours``. Warm combos are never throttled.
                if key not in warm_keys and not self._on_demand_due(key, now):
                    return 0
                # DATA-3: hold this combination until its own slot in the window
                # after the close, so the catalogue does not fire in one minute.
                if not self._spread_slot_reached(instrument, timeframe, now):
                    return 0
                # market-aware: while the market is closed this is False, so
                # no Twelve Data call and no re-emitted reading (MC-1 lock).
                # PERF-1: background regen is PATIENT (bound_provider=False) —
                # it must wait for the feed to actually advance candles.db so
                # the interactive read-through has fresh bars to serve.
                self._assembler.get_or_generate(
                    instrument, timeframe, bound_provider=False
                )
                if key not in warm_keys:
                    with self._state_lock:
                        self._last_on_demand_regen[key] = now
                regenerated += 1
            elif self._should_safety_probe(instrument, timeframe, now):
                # Holiday-only, low-frequency probe for an early reopen.
                if self._assembler.refresh_if_reopened(instrument, timeframe):
                    regenerated += 1
        except Exception:
            # One combination failing must not abort the whole tick.
            logger.exception(
                "scheduler tick: regeneration failed for %s/%s",
                instrument, timeframe,
            )
        return regenerated

    def _spread_slot_reached(
        self, instrument: str, timeframe: str, now: datetime
    ) -> bool:
        """True once this combination's own slot after the close has arrived.

        The slot is a stable offset in [0, window) minutes; a combination is held
        back only that long, and only when a NEW candle has just closed. While a
        market is closed the market-aware close stops advancing, so the elapsed
        time keeps growing and the gate opens immediately — closures and holiday
        probes are never delayed by spreading. A combination with nothing stored
        is spread too: a cold start then staggers itself naturally instead of
        firing the whole catalogue in one minute.
        """
        if not self._spread:
            return True
        period = _tf_minutes_map().get(str(timeframe).upper())
        if not period:
            return True
        offset = spread_offset_minutes(instrument, timeframe, int(period))
        if offset <= 0:
            return True
        expected_close = market_aware_expected_close(instrument, timeframe, now)
        elapsed_min = (now - expected_close).total_seconds() / 60.0
        return elapsed_min >= offset

    def _on_demand_due(self, key: Tuple[str, str], now: datetime) -> bool:
        """True when an access-only (non-warm) combo may be regenerated again.

        Pure predicate — it records nothing, so a combo whose regeneration is
        skipped or fails is retried on the next tick instead of being silently
        pushed a full interval away. The caller stamps ``_last_on_demand_regen``
        only after a regeneration actually happened. A combo never regenerated
        (nothing stamped) is always due, so a cold start is never delayed.
        """
        if self._on_demand_min_interval_seconds <= 0:
            return True
        last = self._last_on_demand_regen.get(key)
        if last is None:
            return True
        return (now - last).total_seconds() >= self._on_demand_min_interval_seconds

    def _should_safety_probe(
        self, instrument: str, timeframe: str, now: datetime
    ) -> bool:
        """True when a low-frequency early-reopen probe is due for this combo.

        Restricted to HOLIDAY closures: weekends and daily breaks are
        deterministic, so probing them would waste API quota on data we already
        know is frozen. Rate-limited to one probe per ``safety_poll_seconds``.

        DATA-3: the first sighting of a combination seeds its clock with a STABLE
        per-combination offset instead of the current instant. Without it every
        combination was stamped in the same tick, so they all came due again in
        the same tick 30 minutes later — measured as one probe per market landing
        in a single minute, i.e. 100-200 credits in one minute at 100 markets, on
        a day the market is closed. Seeding staggers them across the window; the
        cost is that a combination's FIRST probe waits up to one window.
        """
        if self._safety_poll_seconds <= 0:
            return False
        if calendar_state(instrument, now) != CLOSED_HOLIDAY:
            return False
        key = (str(instrument), str(timeframe))
        with self._state_lock:
            last = self._last_safety_probe.get(key)
            if last is None:
                # Spread over the WHOLE poll window, not a fraction of it: a probe
                # that arrives a few minutes later costs nothing (the market is
                # closed), so there is no freshness to trade away here.
                offset = stable_slot(
                    f"probe|{instrument}|{timeframe}", max(1, self._safety_poll_seconds)
                )
                self._last_safety_probe[key] = now - timedelta(seconds=offset)
                return False
            if (now - last).total_seconds() < self._safety_poll_seconds:
                return False
            self._last_safety_probe[key] = now
        return True

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

        MC-1: ``expected_close`` is **market-aware** — it stops advancing while
        the market is closed, so a stored Friday reading keeps matching all
        weekend and the tick makes no Twelve Data call and re-emits nothing.
        """
        expected_close = market_aware_expected_close(instrument, timeframe, now)
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
