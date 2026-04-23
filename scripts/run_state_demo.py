"""Live demo server for the signal state machine dashboard.

Runs a real :class:`SignalStateMachine` and feeds it a scripted bar sequence
every ``BAR_INTERVAL_SECONDS`` seconds. The dashboard mockup polls the same
``/api/v1/signals/state`` endpoint the production API exposes, so what you
see in the browser is exactly what a client will see once MT5 is wired.

Usage
-----
::

    python scripts/run_state_demo.py
    # then open http://localhost:8765/tradingview_dashboard_mockup.html

Environment variables
---------------------
``DEMO_PORT``              Default 8765
``DEMO_BAR_SECONDS``       Default 3.0 (seconds per simulated bar)
``DEMO_SYMBOL``            Default XAUUSD
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

import uvicorn

# Force testing mode BEFORE importing auth/app so the API is open for the demo.
os.environ.setdefault("SENTINEL_TESTING_MODE", "1")
os.environ.setdefault("CORS_ALLOWED_ORIGINS", "*")

from fastapi.staticfiles import StaticFiles  # noqa: E402

from src.api.app import create_app  # noqa: E402
from src.intelligence.signal_state_machine import (  # noqa: E402
    BarInput,
    SignalStateMachine,
    StateMachineConfig,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("state_demo")


# =============================================================================
# SCRIPTED SCENARIO
# =============================================================================

# Each tuple: (score, direction, close, high, low, vol_regime, structure_broken)
# direction: "NONE" | "LONG" | "SHORT"
# Calibrated so the default state-machine thresholds (enter=75, exit=55,
# confirm=2, cooldown=2) produce a visible full cycle — HOLD → arming
# → BUY → active → target reached → cooldown → HOLD → SELL → stop hit
# → cooldown → HOLD.
_ENTRY_PRICE_LONG = 2380.0
_SL_DIST = 6.0
_TP_DIST = 12.0
_ENTRY_PRICE_SHORT = 2395.0

_SCENARIO: List[Tuple[Optional[int], str, float, float, float, str, bool]] = [
    # Idle section — inconclusive market
    (None, "NONE", 2378.0, 2378.8, 2377.2, "normal", False),
    (50, "LONG", 2378.2, 2378.8, 2377.4, "normal", False),
    (None, "NONE", 2378.5, 2379.0, 2377.8, "normal", False),

    # Arming LONG (needs 2 bars ≥75)
    (82, "LONG", _ENTRY_PRICE_LONG, 2380.6, 2379.4, "normal", False),
    (84, "LONG", _ENTRY_PRICE_LONG, 2380.8, 2379.7, "normal", False),

    # Active LONG — price drifts up, eventually hits target
    (78, "LONG", 2381.5, 2382.3, 2380.9, "normal", False),
    (76, "LONG", 2383.0, 2384.0, 2382.3, "normal", False),
    (75, "LONG", 2385.0, 2386.1, 2384.2, "normal", False),
    (72, "LONG", 2388.0, 2389.5, 2386.8, "normal", False),
    # Hit target (TP = 2380 + 12 = 2392)
    (None, "NONE", 2391.0, 2392.8, 2390.5, "normal", False),

    # Cooldown (2 bars)
    (None, "NONE", 2391.5, 2392.0, 2390.5, "normal", False),
    (None, "NONE", 2391.0, 2391.5, 2390.2, "normal", False),

    # IDLE
    (None, "NONE", 2390.8, 2391.2, 2390.0, "normal", False),

    # Arming SHORT
    (86, "SHORT", _ENTRY_PRICE_SHORT, 2395.5, 2394.5, "normal", False),
    (83, "SHORT", _ENTRY_PRICE_SHORT, 2395.4, 2394.3, "normal", False),

    # Active SHORT — price moves against us, hits stop (SL = 2395 + 6 = 2401)
    (78, "SHORT", 2397.0, 2398.2, 2395.5, "normal", False),
    (70, "SHORT", 2399.0, 2400.0, 2397.8, "normal", False),
    (None, "NONE", 2400.5, 2401.5, 2399.5, "normal", False),

    # Cooldown
    (None, "NONE", 2401.0, 2401.8, 2400.0, "normal", False),
    (None, "NONE", 2400.2, 2401.0, 2399.5, "normal", False),

    # Idle → arming LONG but then regime shift kills it mid-signal
    (80, "LONG", 2400.0, 2400.8, 2399.4, "normal", False),
    (81, "LONG", 2400.5, 2401.2, 2399.8, "normal", False),
    (70, "LONG", 2400.8, 2401.4, 2400.1, "high", False),  # REGIME_SHIFTED exits

    # Idle after regime-shift exit
    (None, "NONE", 2400.0, 2400.5, 2399.3, "normal", False),
    (None, "NONE", 2399.5, 2400.0, 2398.8, "normal", False),
]


# =============================================================================
# FAKE CONFLUENCE SIGNAL — duck-typed, no external deps
# =============================================================================

class _DemoSignal:
    """Minimal ConfluenceSignal stand-in."""

    __slots__ = (
        "signal_id", "symbol", "signal_type", "confluence_score",
        "entry_price", "stop_loss", "take_profit", "rr_ratio",
        "atr", "vol_regime", "vol_forecast_atr",
    )

    def __init__(self, direction: str, score: float, entry: float,
                 sl_dist: float, tp_dist: float, regime: str, bar_ix: int):
        self.signal_id = f"demo-{direction[:1].lower()}{bar_ix:03d}"
        self.symbol = os.environ.get("DEMO_SYMBOL", "XAUUSD")
        self.signal_type = direction
        self.confluence_score = float(score)
        self.entry_price = float(entry)
        if direction == "LONG":
            self.stop_loss = entry - sl_dist
            self.take_profit = entry + tp_dist
        else:
            self.stop_loss = entry + sl_dist
            self.take_profit = entry - tp_dist
        self.rr_ratio = round(tp_dist / sl_dist, 2) if sl_dist > 0 else 0.0
        self.atr = round(sl_dist / 2.0, 2)
        self.vol_regime = regime
        self.vol_forecast_atr = self.atr

    def to_dict(self):
        return {k: getattr(self, k) for k in self.__slots__}


# =============================================================================
# DEMO SCANNER SHIM — exposes the state_machine to the API route
# =============================================================================

class _DemoScanner:
    def __init__(self, state_machine: SignalStateMachine, symbol: str):
        self._state_machine = state_machine
        self._symbol = symbol
        self._last_bar_ts: Optional[str] = None
        self._bars_scanned = 0
        self._signals_generated = 0
        self._start_time = time.time()

    @property
    def state_machine(self) -> SignalStateMachine:
        return self._state_machine

    def get_stats(self):
        return {
            "running": True,
            "symbol": self._symbol,
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "bars_scanned": self._bars_scanned,
            "signals_generated": self._signals_generated,
            "last_bar_ts": self._last_bar_ts,
        }


# =============================================================================
# DRIVER — feeds the scripted scenario into the state machine on a timer
# =============================================================================

class _DemoDriver:
    def __init__(self, scanner: _DemoScanner, bar_seconds: float):
        self._scanner = scanner
        self._bar_seconds = max(0.5, bar_seconds)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3)

    def _run(self):
        scenario = _SCENARIO
        bar_ix = 0
        symbol = self._scanner._symbol
        sm = self._scanner.state_machine
        while not self._stop.is_set():
            score, direction, close, high, low, regime, structure = scenario[bar_ix % len(scenario)]
            bar_ts = f"2026-04-23T{(bar_ix // 60) % 24:02d}:{bar_ix % 60:02d}:00Z"
            signal = None
            if direction != "NONE" and score is not None:
                entry = _ENTRY_PRICE_LONG if direction == "LONG" else _ENTRY_PRICE_SHORT
                signal = _DemoSignal(
                    direction=direction, score=score,
                    entry=entry, sl_dist=_SL_DIST, tp_dist=_TP_DIST,
                    regime=regime, bar_ix=bar_ix,
                )
            try:
                bar = BarInput(
                    bar_timestamp=bar_ts,
                    high=high, low=low, close=close,
                    signal=signal, vol_regime=regime,
                    structure_broken=structure,
                )
                _, transition = sm.on_bar(bar)
                self._scanner._last_bar_ts = bar_ts
                self._scanner._bars_scanned += 1
                if transition is not None:
                    if transition.to_state.value in ("BUY", "SELL"):
                        self._scanner._signals_generated += 1
                    logger.info(
                        "[%s] %s → %s (%s)",
                        symbol, transition.from_state.value,
                        transition.to_state.value, transition.reason,
                    )
            except Exception as e:  # pragma: no cover — defensive
                logger.warning("Demo bar failed: %s", e)
            bar_ix += 1
            self._stop.wait(self._bar_seconds)


# =============================================================================
# ENTRY POINT
# =============================================================================

def main() -> None:
    port = int(os.environ.get("DEMO_PORT", "8765"))
    bar_seconds = float(os.environ.get("DEMO_BAR_SECONDS", "3.0"))
    symbol = os.environ.get("DEMO_SYMBOL", "XAUUSD")

    state_machine = SignalStateMachine(StateMachineConfig(symbol=symbol))
    scanner = _DemoScanner(state_machine, symbol)
    driver = _DemoDriver(scanner, bar_seconds)

    app = create_app(scanner=scanner)

    # Same-origin static serve of the mockup directory
    mockups_dir = Path(__file__).resolve().parent.parent / "mockups"
    if mockups_dir.exists():
        app.mount(
            "/",
            StaticFiles(directory=str(mockups_dir), html=True),
            name="mockup",
        )
        logger.info("Mockups served at http://localhost:%d/", port)
    else:
        logger.warning("mockups/ directory not found at %s", mockups_dir)

    driver.start()
    logger.info(
        "Demo driver started (bar_seconds=%.1f, symbol=%s). "
        "Open http://localhost:%d/tradingview_dashboard_mockup.html",
        bar_seconds, symbol, port,
    )

    try:
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning")
    finally:
        driver.stop()
        logger.info("Demo driver stopped.")


if __name__ == "__main__":
    main()
