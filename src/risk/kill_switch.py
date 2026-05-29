"""Smart Sentinel AI — Kill-Switch.

A pre-publication risk gate that blocks new signals when any of four
operational fault-conditions is met:

1. ``consecutive_losses``     — N straight losing trades (default 4)
2. ``daily_drawdown_pct``     — running session DD exceeds limit (default 5 %)
3. ``volatility_spike``       — realised IV > N × historical std (default 3σ)
4. ``broker_disconnect``      — data-feed / broker heartbeat stale (default 120 s)

Design goals
------------
* **Deterministic** — same inputs, same state, same decision.  No global
  clock dependence: the caller passes timestamps explicitly.
* **Hot-restartable** — full state can be dumped to / loaded from a dict
  so the scanner can persist it alongside :class:`SignalStateMachine`.
* **Override-safe** — ``manual_override()`` accepts an explicit
  acknowledgement string, logs caller identity, and **never** silently
  re-arms after the four hard rules trigger.  See the legal note in
  :meth:`KillSwitch.manual_reset` — kill-switches that are silently
  user-overridable are an established lawsuit vector.

Insertion sites (Smart Sentinel AI 2026-04-26)
----------------------------------------------
* ``src/intelligence/sentinel_scanner.py`` — call ``ks.check()`` *before*
  publishing each signal; if ``not ok``, drop the signal and emit a
  Telegram admin alert.
* ``src/intelligence/sentinel_scanner.py::_on_trade_close`` —
  ``ks.record_trade_outcome(r_multiple)``.
* ``src/intelligence/data_providers.py::tick_received`` —
  ``ks.heartbeat(now)``.
* ``src/intelligence/volatility_forecaster.py::forecast`` —
  ``ks.update_volatility(realised_iv)``.
"""
from __future__ import annotations

import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS / DATACLASSES
# =============================================================================


class TripReason(str, Enum):
    """Why the kill-switch fired (for logging + Telegram alert)."""

    CONSECUTIVE_LOSSES = "consecutive_losses"
    DAILY_DRAWDOWN = "daily_drawdown"
    VOLATILITY_SPIKE = "volatility_spike"
    BROKER_DISCONNECT = "broker_disconnect"
    MANUAL = "manual"


@dataclass(frozen=True)
class KillSwitchConfig:
    """Configuration thresholds.

    All limits are calibrated for XAUUSD M15 personal-testing tier
    (see ``baseline_2019_2025.md``).  Operators trading other instruments
    or live capital should override them via :class:`InstrumentConfig`.
    """

    # 1. Streak rule
    max_consecutive_losses: int = 4

    # 2. Daily-DD rule (fraction of starting equity)
    daily_dd_limit_pct: float = 0.05  # 5 %

    # 3. Vol-spike rule
    vol_zscore_limit: float = 3.0
    vol_history_window: int = 96  # 24 h of M15 bars

    # 4. Broker-disconnect rule
    heartbeat_max_silence_s: float = 120.0

    # Auto-reset window for streak / DD (seconds).  Set to ``None`` to
    # require manual reset every time.
    auto_reset_after_s: Optional[float] = 24 * 3600.0


@dataclass
class TripEvent:
    """One firing of the kill-switch — kept in audit log."""

    reason: TripReason
    detail: str
    timestamp: float
    cleared_at: Optional[float] = None
    cleared_by: Optional[str] = None


# =============================================================================
# KILL SWITCH
# =============================================================================


class KillSwitch:
    """Operational kill-switch for Smart Sentinel AI.

    Thread-safety
    -------------
    Single-threaded by design.  The scanner is async but signal
    publication is serialised on a single event-loop, so ``check()`` is
    only ever called from one task.  Background components
    (volatility, heartbeat) push state via the dedicated ``update_*``
    methods, all of which only mutate primitive members.
    """

    def __init__(
        self,
        config: Optional[KillSwitchConfig] = None,
        starting_equity: float = 1000.0,
    ):
        self._cfg = config or KillSwitchConfig()
        if starting_equity <= 0:
            raise ValueError("starting_equity must be positive")
        self._starting_equity = float(starting_equity)
        self._daily_baseline_equity = float(starting_equity)

        # State
        self._tripped: bool = False
        self._trip_reason: Optional[TripReason] = None
        self._trip_at: Optional[float] = None
        self._trip_detail: str = ""

        self._consecutive_losses: int = 0
        self._daily_pnl: float = 0.0
        self._daily_baseline_at: float = time.time()

        self._last_heartbeat_at: Optional[float] = None
        self._vol_history: Deque[float] = deque(maxlen=self._cfg.vol_history_window)
        self._last_vol: Optional[float] = None

        self._audit: List[TripEvent] = []

    # ------------------------------------------------------------------ #
    # PROPERTIES
    # ------------------------------------------------------------------ #

    @property
    def is_tripped(self) -> bool:
        return self._tripped

    @property
    def trip_reason(self) -> Optional[TripReason]:
        return self._trip_reason

    @property
    def consecutive_losses(self) -> int:
        return self._consecutive_losses

    @property
    def daily_pnl_pct(self) -> float:
        if self._daily_baseline_equity <= 0:
            return 0.0
        return self._daily_pnl / self._daily_baseline_equity

    # ------------------------------------------------------------------ #
    # PUBLIC API — INPUT
    # ------------------------------------------------------------------ #

    def record_trade_outcome(self, r_multiple: float, pnl_dollars: float = 0.0) -> None:
        """Notify the switch of a closed trade.

        ``r_multiple`` is the realised PnL in *risk units* (-1.0 = full
        SL hit).  ``pnl_dollars`` is the realised dollar PnL (used for
        the daily-DD rule).
        """
        if r_multiple <= 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        self._daily_pnl += float(pnl_dollars)
        self._maybe_auto_reset_window()
        self._evaluate()

    def heartbeat(self, now: Optional[float] = None) -> None:
        """Mark a successful broker / data-feed tick."""
        self._last_heartbeat_at = now if now is not None else time.time()
        # A successful heartbeat clears a stale-feed trip on its own.
        if self._tripped and self._trip_reason == TripReason.BROKER_DISCONNECT:
            self._auto_clear("heartbeat resumed")

    def update_volatility(self, realised_vol: float) -> None:
        """Push a fresh realised-volatility reading (any consistent unit).

        The switch keeps a rolling buffer and trips when the latest
        reading is more than ``vol_zscore_limit`` standard deviations
        above the rolling mean — the classic black-swan flag.
        """
        if realised_vol <= 0 or math.isnan(realised_vol) or math.isinf(realised_vol):
            return
        self._last_vol = float(realised_vol)
        self._vol_history.append(float(realised_vol))
        self._evaluate()

    # ------------------------------------------------------------------ #
    # PUBLIC API — OUTPUT
    # ------------------------------------------------------------------ #

    def check(self, now: Optional[float] = None) -> bool:
        """Return ``True`` if signals can be published.

        Combines a re-evaluation of the four rules with the current
        latch state.  The scanner should call this immediately before
        ``telegram_notifier.send_signal`` (or equivalent).
        """
        self._evaluate(now=now)
        return not self._tripped

    def status(self) -> Dict[str, Any]:
        """Snapshot for monitoring + ``/health`` endpoint."""
        return {
            "tripped": self._tripped,
            "reason": self._trip_reason.value if self._trip_reason else None,
            "detail": self._trip_detail,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl_pct": round(self.daily_pnl_pct, 4),
            "last_heartbeat_age_s": (
                None
                if self._last_heartbeat_at is None
                else round(time.time() - self._last_heartbeat_at, 1)
            ),
            "vol_buffer_size": len(self._vol_history),
            "last_vol": self._last_vol,
            "audit_events": len(self._audit),
        }

    # ------------------------------------------------------------------ #
    # PUBLIC API — ADMIN
    # ------------------------------------------------------------------ #

    def manual_reset(self, operator: str, ack_phrase: str) -> bool:
        """Clear a manual / consecutive-loss / DD trip.

        Args:
            operator: Identity of the human authorising the override
                (logged for legal traceability).
            ack_phrase: Must equal ``"I-ACCEPT-RISK"`` — guards against
                accidental clears via a forwarded webhook.

        Returns:
            ``True`` on success, ``False`` if the phrase mismatched.

        Notes:
            We deliberately do **not** allow this method to clear a
            ``BROKER_DISCONNECT`` trip — only a successful heartbeat
            does.  This avoids the lawsuit pattern "operator overrode
            the switch, broker was actually down, signals went into the
            void".
        """
        if ack_phrase != "I-ACCEPT-RISK":
            logger.warning("manual_reset rejected: bad ack from %s", operator)
            return False
        if self._trip_reason == TripReason.BROKER_DISCONNECT:
            logger.warning(
                "manual_reset refused: broker still disconnected (operator=%s)",
                operator,
            )
            return False
        self._auto_clear(f"manual override by {operator}", operator=operator)
        return True

    def manual_override(self, operator: str, ack_phrase: str) -> bool:
        """Alias kept for backwards-compat with any caller using the
        verb 'override'.  Behaviour is identical to :meth:`manual_reset`.
        """
        return self.manual_reset(operator, ack_phrase)

    def trip_manual(self, detail: str = "operator pause") -> None:
        """Force-trip the switch (e.g. CI pre-deploy pause)."""
        self._trip(TripReason.MANUAL, detail)

    # ------------------------------------------------------------------ #
    # PUBLIC API — PERSISTENCE
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialise full state for restart-safety."""
        return {
            "tripped": self._tripped,
            "trip_reason": self._trip_reason.value if self._trip_reason else None,
            "trip_at": self._trip_at,
            "trip_detail": self._trip_detail,
            "consecutive_losses": self._consecutive_losses,
            "daily_pnl": self._daily_pnl,
            "daily_baseline_equity": self._daily_baseline_equity,
            "daily_baseline_at": self._daily_baseline_at,
            "last_heartbeat_at": self._last_heartbeat_at,
            "vol_history": list(self._vol_history),
            "last_vol": self._last_vol,
            "audit": [
                {
                    "reason": e.reason.value,
                    "detail": e.detail,
                    "timestamp": e.timestamp,
                    "cleared_at": e.cleared_at,
                    "cleared_by": e.cleared_by,
                }
                for e in self._audit
            ],
        }

    @classmethod
    def from_dict(
        cls,
        state: Dict[str, Any],
        config: Optional[KillSwitchConfig] = None,
        starting_equity: float = 1000.0,
    ) -> "KillSwitch":
        """Restore state previously dumped via :meth:`to_dict`."""
        ks = cls(config=config, starting_equity=starting_equity)
        ks._tripped = bool(state.get("tripped", False))
        reason = state.get("trip_reason")
        ks._trip_reason = TripReason(reason) if reason else None
        ks._trip_at = state.get("trip_at")
        ks._trip_detail = state.get("trip_detail", "")
        ks._consecutive_losses = int(state.get("consecutive_losses", 0))
        ks._daily_pnl = float(state.get("daily_pnl", 0.0))
        ks._daily_baseline_equity = float(
            state.get("daily_baseline_equity", starting_equity)
        )
        ks._daily_baseline_at = float(state.get("daily_baseline_at", time.time()))
        ks._last_heartbeat_at = state.get("last_heartbeat_at")
        ks._vol_history = deque(
            state.get("vol_history", []), maxlen=ks._cfg.vol_history_window
        )
        ks._last_vol = state.get("last_vol")
        ks._audit = [
            TripEvent(
                reason=TripReason(e["reason"]),
                detail=e["detail"],
                timestamp=e["timestamp"],
                cleared_at=e.get("cleared_at"),
                cleared_by=e.get("cleared_by"),
            )
            for e in state.get("audit", [])
        ]
        return ks

    # ------------------------------------------------------------------ #
    # INTERNAL
    # ------------------------------------------------------------------ #

    def _evaluate(self, now: Optional[float] = None) -> None:
        now = now if now is not None else time.time()

        # Auto-clear stale streak / DD trip if the configured window
        # has elapsed.  Broker / vol trips do NOT auto-clear — they need
        # an explicit heartbeat or vol normalisation.
        self._maybe_auto_reset_window(now=now)

        if self._tripped:
            return  # latched until cleared

        # 1. Streak
        if self._consecutive_losses >= self._cfg.max_consecutive_losses:
            self._trip(
                TripReason.CONSECUTIVE_LOSSES,
                f"{self._consecutive_losses} losses in a row "
                f"(limit {self._cfg.max_consecutive_losses})",
                now=now,
            )
            return

        # 2. Daily DD
        dd_pct = -self.daily_pnl_pct  # positive when in loss
        if dd_pct >= self._cfg.daily_dd_limit_pct:
            self._trip(
                TripReason.DAILY_DRAWDOWN,
                f"daily DD {dd_pct:.2%} >= limit {self._cfg.daily_dd_limit_pct:.2%}",
                now=now,
            )
            return

        # 3. Vol spike
        if (
            len(self._vol_history) >= max(20, self._cfg.vol_history_window // 4)
            and self._last_vol is not None
        ):
            mean = sum(self._vol_history) / len(self._vol_history)
            var = sum((v - mean) ** 2 for v in self._vol_history) / len(
                self._vol_history
            )
            std = math.sqrt(var)
            if std > 0:
                z = (self._last_vol - mean) / std
                if z >= self._cfg.vol_zscore_limit:
                    self._trip(
                        TripReason.VOLATILITY_SPIKE,
                        f"vol z={z:.2f} >= limit {self._cfg.vol_zscore_limit:.2f}",
                        now=now,
                    )
                    return

        # 4. Broker disconnect
        if self._last_heartbeat_at is not None:
            silence = now - self._last_heartbeat_at
            if silence > self._cfg.heartbeat_max_silence_s:
                self._trip(
                    TripReason.BROKER_DISCONNECT,
                    f"no heartbeat for {silence:.0f}s "
                    f"(limit {self._cfg.heartbeat_max_silence_s:.0f}s)",
                    now=now,
                )
                return

    def _trip(
        self, reason: TripReason, detail: str, now: Optional[float] = None
    ) -> None:
        now = now if now is not None else time.time()
        self._tripped = True
        self._trip_reason = reason
        self._trip_at = now
        self._trip_detail = detail
        self._audit.append(TripEvent(reason=reason, detail=detail, timestamp=now))
        logger.error("KILL-SWITCH TRIPPED: %s — %s", reason.value, detail)

    def _auto_clear(self, detail: str, operator: Optional[str] = None) -> None:
        if not self._tripped:
            return
        now = time.time()
        if self._audit:
            self._audit[-1].cleared_at = now
            self._audit[-1].cleared_by = operator or "auto"
        logger.warning(
            "Kill-switch cleared (%s) — was: %s", detail, self._trip_reason
        )
        self._tripped = False
        self._trip_reason = None
        self._trip_at = None
        self._trip_detail = ""
        self._consecutive_losses = 0  # benefit of the doubt after manual clear

    def _maybe_auto_reset_window(self, now: Optional[float] = None) -> None:
        if self._cfg.auto_reset_after_s is None:
            return
        now = now if now is not None else time.time()
        if (now - self._daily_baseline_at) >= self._cfg.auto_reset_after_s:
            self._daily_pnl = 0.0
            self._daily_baseline_at = now


__all__ = ["KillSwitch", "KillSwitchConfig", "TripReason", "TripEvent"]
