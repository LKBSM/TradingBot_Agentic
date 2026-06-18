"""Twelve Data WebSocket live-tick bridge (PROTOTYPE — dev / free tier only).

A single, process-wide WebSocket connection to ``wss://ws.twelvedata.com/v1/
quotes/price`` that streams the last traded price for the V1 instruments
(XAU/USD, EUR/USD) and keeps the latest tick per instrument in memory.

WHY a *backend* bridge (not a browser WebSocket):
  - The Twelve Data **free trial allows ONE connection** (cf. audit
    PROTO_LIVE_TICK_ZONES_FAISABILITE_2026_06_17.md §1). A single shared
    backend connection multiplexes to N SSE clients; a per-browser WS would
    blow the 1-connection cap on the second tab.
  - The API key stays server-side — the front never sees it (same discipline
    as LIVE_DATA_WIRING_2026_06_08.md, "le front ne détient aucune clé").

SCOPE — strictly DESCRIPTIVE, zone-interaction only:
  - The bridge only carries the *last price*. It performs NO detection, emits
    NO structure, and never touches the candle-close path (REST/cache). The
    live tick is consumed downstream to update the INTERACTION of already-
    detected zones (FVG fill, OB touch) — never to detect new structure, and
    NEVER for BOS/CHOCH (validated at candle close only, SMC law).

COMMERCIAL NOTE: Twelve Data WebSocket is officially a Pro/Business feature;
the trial grants limited test access only. A commercial launch will require
the **Business plan**. Do not wire anything that assumes a paid plan.

Concurrency model: the async WS loop runs in a dedicated daemon thread with its
own event loop (the rest of the app already uses a background scheduler thread).
``get_latest`` is the thread-safe read surface consumed by the SSE route.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from src.intelligence.data_providers.twelve_data_provider import _SYMBOL_MAP

logger = logging.getLogger(__name__)

# App-instrument code -> Twelve Data WS symbol (reuse the REST provider's map so
# the two never drift). The reverse map turns "XAU/USD" ticks back into "XAUUSD".
_TD_TO_APP: Dict[str, str] = {td: app for app, td in _SYMBOL_MAP.items()}

_WS_URL = "wss://ws.twelvedata.com/v1/quotes/price"


@dataclass(frozen=True)
class LiveTick:
    """A single last-price tick. Descriptive only — no derived/predictive field.

    ``ts`` is the feed's UNIX epoch in seconds (Twelve Data ``timestamp``).
    ``received_monotonic`` is a local monotonic stamp used only for staleness
    checks (never surfaced to clients).
    """

    instrument: str
    price: float
    ts: int
    received_monotonic: float


class TwelveDataLiveTickBridge:
    """Single shared WS connection → latest tick per instrument (thread-safe).

    Lifecycle mirrors ``MarketReadingScheduler``: ``start()`` spins a daemon
    thread, ``stop()`` is idempotent and safe to call from the shutdown
    coordinator. Reconnects with capped exponential backoff; one bad message
    never kills the loop.

    The ``connect`` callable is injectable so the message-handling and lifecycle
    logic can be unit-tested without a network (default: ``websockets.connect``).
    """

    def __init__(
        self,
        api_key: str,
        instruments: Optional[list[str]] = None,
        *,
        connect: Optional[Callable[..., Any]] = None,
        ping_interval: float = 10.0,
        base_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        now_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not api_key:
            raise ValueError("TwelveDataLiveTickBridge requires a non-empty api_key")
        self._api_key = api_key
        # Default to the full V1 perimeter; validate each against the symbol map.
        instruments = instruments or list(_SYMBOL_MAP.keys())
        unknown = [i for i in instruments if i not in _SYMBOL_MAP]
        if unknown:
            raise ValueError(
                f"Unsupported instrument(s) {unknown}; known: {sorted(_SYMBOL_MAP)}"
            )
        self._instruments = instruments
        self._td_symbols = [_SYMBOL_MAP[i] for i in instruments]
        self._connect = connect  # resolved lazily so importing websockets is optional
        self._ping_interval = ping_interval
        self._base_backoff_s = base_backoff_s
        self._max_backoff_s = max_backoff_s
        self._now_fn = now_fn

        self._latest: Dict[str, LiveTick] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._main_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------------ #
    # Read surface (thread-safe)
    # ------------------------------------------------------------------ #
    def get_latest(self, instrument: str) -> Optional[LiveTick]:
        """Latest tick for an instrument, or None if none received yet."""
        with self._lock:
            return self._latest.get(instrument)

    @property
    def instruments(self) -> list[str]:
        return list(self._instruments)

    @property
    def running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    # ------------------------------------------------------------------ #
    # Message handling (pure — unit-tested without a network)
    # ------------------------------------------------------------------ #
    def _handle_message(self, raw: str | bytes) -> None:
        """Parse one WS frame and update the latest-tick table.

        Tolerant by design: unparsable frames, non-price events, unknown
        symbols, and malformed prices are logged-and-skipped, never raised —
        a single bad frame must not break the stream.
        """
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            logger.debug("td-ws: dropping unparsable frame")
            return
        if not isinstance(data, dict):
            return

        event = data.get("event")
        if event == "subscribe-status":
            logger.info("td-ws subscribe-status: %s", json.dumps(data)[:300])
            return
        if event != "price":
            return

        td_symbol = data.get("symbol")
        instrument = _TD_TO_APP.get(td_symbol)
        if instrument is None:
            return
        try:
            price = float(data["price"])
        except (KeyError, TypeError, ValueError):
            return
        if price <= 0:
            return
        try:
            ts = int(data.get("timestamp"))
        except (TypeError, ValueError):
            ts = 0

        tick = LiveTick(
            instrument=instrument,
            price=price,
            ts=ts,
            received_monotonic=self._now_fn(),
        )
        with self._lock:
            self._latest[instrument] = tick

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        """Spawn the daemon thread running the async WS loop. Idempotent."""
        if self.running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="td-ws-bridge", daemon=True
        )
        self._thread.start()
        logger.info(
            "TwelveDataLiveTickBridge started (symbols=%s)", ",".join(self._td_symbols)
        )

    def stop(self) -> None:
        """Signal shutdown and join the thread. Safe to call repeatedly."""
        self._stop_event.set()
        loop = self._loop
        task = self._main_task
        if loop is not None and task is not None and not loop.is_closed():
            try:
                loop.call_soon_threadsafe(task.cancel)
            except RuntimeError:  # pragma: no cover — loop already torn down
                pass
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)
        logger.info("TwelveDataLiveTickBridge stopped")

    # ------------------------------------------------------------------ #
    # Async internals
    # ------------------------------------------------------------------ #
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._main_task = loop.create_task(self._run_async())
            loop.run_until_complete(self._main_task)
        except asyncio.CancelledError:
            pass
        except Exception:  # pragma: no cover — defensive, keeps thread quiet
            logger.exception("td-ws bridge thread crashed")
        finally:
            try:
                loop.close()
            except Exception:  # pragma: no cover
                pass

    def _resolve_connect(self) -> Callable[..., Any]:
        if self._connect is not None:
            return self._connect
        import websockets  # local import: optional dep, only needed when live

        return websockets.connect

    def _url(self) -> str:
        return f"{_WS_URL}?apikey={self._api_key}"

    async def _run_async(self) -> None:
        """Connect → subscribe → consume, with reconnect/backoff until stopped."""
        connect = self._resolve_connect()
        sub_msg = json.dumps(
            {"action": "subscribe", "params": {"symbols": ",".join(self._td_symbols)}}
        )
        backoff = self._base_backoff_s
        while not self._stop_event.is_set():
            try:
                async with connect(self._url(), ping_interval=self._ping_interval) as ws:
                    await ws.send(sub_msg)
                    backoff = self._base_backoff_s  # reset after a clean connect
                    async for raw in ws:
                        if self._stop_event.is_set():
                            break
                        self._handle_message(raw)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if self._stop_event.is_set():
                    break
                logger.warning(
                    "td-ws connection error (%s); reconnecting in %.1fs",
                    type(exc).__name__, backoff,
                )
                try:
                    await asyncio.sleep(backoff)
                except asyncio.CancelledError:
                    raise
                backoff = min(backoff * 2, self._max_backoff_s)


__all__ = ["LiveTick", "TwelveDataLiveTickBridge"]
