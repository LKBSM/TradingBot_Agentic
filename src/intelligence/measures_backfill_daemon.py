"""Automatic deep-history maintainer for the publication measures (NW-7).

The measures need MONTHS of intraday (M15) price history per measured market, but
Twelve Data caps one request at ~52 days and the bundled CSV is dev-only. This
daemon fills the gap generically: every interval it advances the deep M15 history
of EVERY measured market (``publication_measures.measured_markets()``) a bounded
few pages, RESUMING from cache — so adding a market at commercialisation needs no
extra wiring, it just gets filled over time within the daily quota.

OPT-IN (``MEASURES_DEEP_BACKFILL_ENABLED=1``) and OFF by default: on a
quota-starved free plan it would only add pressure. Turn it on once the data plan
has headroom. It never raises, throttles via the provider's rate limiter, and
pauses the cycle the moment the provider reports out-of-credits (429).
"""

from __future__ import annotations

import logging
import os
import threading
import time

logger = logging.getLogger(__name__)

_ENV_ENABLED = "MEASURES_DEEP_BACKFILL_ENABLED"
_ENV_INTERVAL_S = "MEASURES_DEEP_BACKFILL_INTERVAL_S"
_ENV_PAGES = "MEASURES_DEEP_BACKFILL_PAGES"
_ENV_TIMEFRAME = "MEASURES_DEEP_BACKFILL_TIMEFRAME"

_started = False
_lock = threading.Lock()


def _enabled() -> bool:
    return os.environ.get(_ENV_ENABLED, "").strip().lower() in ("1", "true", "yes")


def start_measures_backfill_daemon(provider, store) -> bool:
    """Start the opt-in background maintainer once. Returns True if started.

    ``provider`` must expose ``fetch_candles_until`` (TwelveDataProvider does);
    ``store`` is the candles cache. Both are the same env-configured instances the
    app already built, so the daemon writes the SAME candles.db the measures read.
    """
    global _started
    if not _enabled():
        logger.info(
            "measures deep-backfill daemon OFF (set %s=1 to enable once the data "
            "plan has quota headroom)", _ENV_ENABLED,
        )
        return False
    with _lock:
        if _started:
            return False
        _started = True

    interval_s = _int_env(_ENV_INTERVAL_S, 900)      # 15 min between cycles
    pages = _int_env(_ENV_PAGES, 2)                  # ≤2 pages/market/cycle
    timeframe = os.environ.get(_ENV_TIMEFRAME, "M15") or "M15"

    def _loop() -> None:
        # Import inside the thread so a bootstrap import cycle can never block boot.
        from src.intelligence.history_backfill import maintain_deep_history
        from src.intelligence.publication_measures import measured_markets

        # A short initial delay so it never competes with the boot/warm burst.
        time.sleep(min(interval_s, 60))
        while True:
            try:
                markets = measured_markets()
                if markets:
                    results = maintain_deep_history(
                        provider, store, markets,
                        timeframe=timeframe, max_pages_per_market=pages,
                    )
                    logger.info("measures deep-backfill cycle: %s", results)
            except Exception:  # a cycle must never kill the daemon
                logger.exception("measures deep-backfill cycle failed")
            time.sleep(interval_s)

    threading.Thread(target=_loop, name="measures-backfill", daemon=True).start()
    logger.info(
        "measures deep-backfill daemon STARTED (interval=%ss, pages/market=%s, tf=%s)",
        interval_s, pages, timeframe,
    )
    return True


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


__all__ = ["start_measures_backfill_daemon"]
