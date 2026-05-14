"""Forward-test paper-trading harness — Sprint INFRA-2B.2.

A *transparent* paper-trading record of every fictional signal the
system "would have" generated. Published on the webapp with the
disclaimer "Démonstration paper-trading. Smart Sentinel ne prétend
PAS posséder un edge. Cette courbe est éducative."

Why a paper harness when A1 said no edge
----------------------------------------
The pivot to Phase 2B is narrative-first; we don't claim edge. The
paper-trade equity curve isn't proof of profitability — it's proof of
*honesty*: we publish the result whether it goes up or down, and the
audit log is SHA256-hashed weekly so we can't massage history. The
disclaimer is mandatory under UE 2024/2811 for "hypothetical
performance" displays.

Public surface
--------------
- ``PaperPosition(...)`` represents one in-flight fictional trade.
- ``PaperTradingHarness.enter(...)``  open a new fictional position.
- ``PaperTradingHarness.mark(...)``   mark-to-market at a new price.
- ``PaperTradingHarness.exit(...)``   close, realise R-multiple.
- ``equity_curve()``                  list of (ts, equity) tuples.
- ``stats()``                         {n_trades, win_rate, sharpe,
                                       max_drawdown, total_R}.
- ``export_for_publication()``        JSON shape consumed by webapp.

Critical: ``enter/mark/exit`` never receive predicted/fitted values.
They consume **realised** OHLCV. The harness is deterministic given
the price feed — no random slippage, no model. The whole point is
that ops can reproduce the curve from the same CSV.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Literal, Optional


Direction = Literal["LONG", "SHORT"]


@dataclass
class PaperPosition:
    position_id: str
    direction: Direction
    entry_price: float
    stop_price: float
    target_price: float
    entry_ts: float
    rr_unit: float            # how many price units = 1R (= |entry - stop|)
    closed: bool = False
    exit_price: Optional[float] = None
    exit_ts: Optional[float] = None
    realised_r: Optional[float] = None
    extras: dict = field(default_factory=dict)

    def mark_to_market_r(self, price: float) -> float:
        if self.rr_unit <= 0:
            return 0.0
        if self.direction == "LONG":
            return (price - self.entry_price) / self.rr_unit
        return (self.entry_price - price) / self.rr_unit


@dataclass
class TradeOutcome:
    position_id: str
    direction: Direction
    realised_r: float
    held_seconds: float
    hit: Literal["STOP", "TARGET", "MANUAL"]


class PaperTradingHarness:
    """Tracks fictional positions + a running equity curve in R-multiples."""

    def __init__(self, *, starting_equity_R: float = 0.0, clock=time.time):
        self._lock = threading.Lock()
        self._clock = clock
        self._next_id = 0
        self._open: dict[str, PaperPosition] = {}
        self._closed: list[TradeOutcome] = []
        # Equity curve in R units (not USD). USD equity is presentation.
        self._equity_R = starting_equity_R
        self._equity_curve: list[tuple[float, float]] = [
            (clock(), starting_equity_R)
        ]

    # ------------------------------------------------------------------
    # Open / mark / close
    # ------------------------------------------------------------------

    def enter(
        self,
        *,
        direction: Direction,
        entry_price: float,
        stop_price: float,
        target_price: float,
        extras: Optional[dict] = None,
    ) -> PaperPosition:
        if direction not in ("LONG", "SHORT"):
            raise ValueError(f"direction must be LONG or SHORT, got {direction!r}")
        rr_unit = abs(entry_price - stop_price)
        if rr_unit <= 0:
            raise ValueError("entry and stop must differ")
        # Direction sanity
        if direction == "LONG" and (stop_price >= entry_price or target_price <= entry_price):
            raise ValueError("LONG requires stop < entry < target")
        if direction == "SHORT" and (stop_price <= entry_price or target_price >= entry_price):
            raise ValueError("SHORT requires target < entry < stop")

        with self._lock:
            self._next_id += 1
            pos = PaperPosition(
                position_id=f"pp-{self._next_id:08d}",
                direction=direction,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                entry_ts=self._clock(),
                rr_unit=rr_unit,
                extras=dict(extras or {}),
            )
            self._open[pos.position_id] = pos
            return pos

    def mark(self, position_id: str, price: float) -> Optional[TradeOutcome]:
        """Auto-closes the position if price crosses stop or target."""
        with self._lock:
            pos = self._open.get(position_id)
            if pos is None or pos.closed:
                return None

            hit: Optional[Literal["STOP", "TARGET"]] = None
            if pos.direction == "LONG":
                if price <= pos.stop_price:
                    hit = "STOP"
                elif price >= pos.target_price:
                    hit = "TARGET"
            else:  # SHORT
                if price >= pos.stop_price:
                    hit = "STOP"
                elif price <= pos.target_price:
                    hit = "TARGET"

            if hit is None:
                return None

            exit_price = pos.stop_price if hit == "STOP" else pos.target_price
            return self._close_locked(pos, exit_price, hit)

    def exit(
        self, position_id: str, price: float
    ) -> Optional[TradeOutcome]:
        """Manual close at a given price."""
        with self._lock:
            pos = self._open.get(position_id)
            if pos is None or pos.closed:
                return None
            return self._close_locked(pos, price, "MANUAL")

    def _close_locked(
        self,
        pos: PaperPosition,
        exit_price: float,
        hit: Literal["STOP", "TARGET", "MANUAL"],
    ) -> TradeOutcome:
        pos.closed = True
        pos.exit_price = exit_price
        pos.exit_ts = self._clock()
        pos.realised_r = pos.mark_to_market_r(exit_price)
        outcome = TradeOutcome(
            position_id=pos.position_id,
            direction=pos.direction,
            realised_r=pos.realised_r,
            held_seconds=pos.exit_ts - pos.entry_ts,
            hit=hit,
        )
        self._closed.append(outcome)
        self._equity_R += pos.realised_r
        self._equity_curve.append((pos.exit_ts, self._equity_R))
        del self._open[pos.position_id]
        return outcome

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def equity_curve(self) -> list[tuple[float, float]]:
        with self._lock:
            return list(self._equity_curve)

    def open_positions(self) -> list[PaperPosition]:
        with self._lock:
            return list(self._open.values())

    def closed_trades(self) -> list[TradeOutcome]:
        with self._lock:
            return list(self._closed)

    def stats(self) -> dict:
        with self._lock:
            closed = list(self._closed)
            equity = self._equity_R
            curve = list(self._equity_curve)

        n = len(closed)
        if n == 0:
            return {
                "n_trades": 0,
                "win_rate": 0.0,
                "total_R": 0.0,
                "mean_R": 0.0,
                "max_drawdown_R": 0.0,
                "sharpe_per_trade": 0.0,
            }
        wins = sum(1 for t in closed if t.realised_r > 0)
        rs = [t.realised_r for t in closed]
        mean_r = sum(rs) / n
        if n > 1:
            var = sum((r - mean_r) ** 2 for r in rs) / (n - 1)
            std = math.sqrt(var) if var > 0 else 0.0
        else:
            std = 0.0
        sharpe = (mean_r / std) if std > 0 else 0.0

        # Max drawdown in R off the equity curve.
        peak = curve[0][1]
        max_dd = 0.0
        for _, eq in curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            if dd > max_dd:
                max_dd = dd

        return {
            "n_trades": n,
            "win_rate": round(wins / n, 4),
            "total_R": round(equity, 4),
            "mean_R": round(mean_r, 4),
            "max_drawdown_R": round(max_dd, 4),
            "sharpe_per_trade": round(sharpe, 4),
        }

    def export_for_publication(self) -> dict:
        """The JSON shape the webapp's transparency page consumes."""
        return {
            "disclaimer": (
                "Démonstration paper-trading. Smart Sentinel ne prétend "
                "PAS posséder un edge. Cette courbe est éducative."
            ),
            "disclaimer_en": (
                "Paper-trading demonstration. Smart Sentinel does NOT "
                "claim a trading edge. This curve is for educational "
                "purposes only."
            ),
            "stats": self.stats(),
            "equity_curve": [
                {"ts": ts, "equity_R": round(eq, 4)} for ts, eq in self.equity_curve()
            ],
            "open_positions_count": len(self.open_positions()),
            "closed_trades_count": len(self.closed_trades()),
        }


__all__ = [
    "PaperPosition",
    "PaperTradingHarness",
    "TradeOutcome",
]
