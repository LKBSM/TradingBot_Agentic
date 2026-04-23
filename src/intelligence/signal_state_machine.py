"""Signal state machine — the trust layer for Smart Sentinel AI.

Sits between :class:`ConfluenceDetector` and the signal store. Takes a stream
of per-bar confluence scores and enforces a stable HOLD / BUY / SELL contract
for the client, preventing flicker, enforcing cooldowns, and emitting explicit
exit reasons.

Core guarantees
---------------
* **Deterministic**: state transitions are a pure function of the input bar
  sequence plus the injected :class:`StateMachineConfig`. Wall-clock time is
  never consulted inside the transition logic; only bar timestamps/indices
  drive progress. This makes the machine fully replayable from history and
  trivially testable.
* **Thread-safe**: a re-entrant lock guards every mutation. Multiple scanners
  (one per symbol) each get their own instance, and the API layer can read
  snapshots concurrently with the scanner writing them.
* **Persistence-ready**: :meth:`to_dict` / :meth:`from_dict` round-trip the
  full internal state so the machine survives process restarts without
  forgetting an active signal or cooldown window.
* **Defensive**: every input is validated. Malformed bars (NaN prices, scores
  outside [0, 100], out-of-order timestamps, replayed timestamps) are rejected
  without mutating state. The machine never raises on logic paths — it logs
  and returns the current snapshot.
* **Observable**: transition history, per-reason exit counts, confirmation
  success rate, and average signal lifetime are exposed via :meth:`get_stats`.

Five transition rules (see ``design note`` at bottom of file)
-------------------------------------------------------------
1. **Hysteresis** — arm at ``enter_threshold`` (default 75); exit at
   ``exit_threshold`` (default 55). The 20-point dead band eliminates
   oscillation around a single threshold.
2. **Confirmation window** — require ``confirm_bars`` (default 2) consecutive
   bars of same-direction ≥ enter_threshold before publishing a BUY/SELL.
   Kills single-candle ghost signals.
3. **Signal lifetime** — an active signal exits via exactly one of six
   :class:`ExitReason` values (target hit, stop hit, time expiry, score decay,
   regime shift, opposing signal).
4. **Cooldown** — after any exit, the public state is HOLD for
   ``cooldown_bars`` (default 2) before a new signal can arm.
5. **Opposing-direction lockout** — BUY cannot flip directly to SELL. The
   machine must pass through HOLD → cooldown → HOLD before arming the other
   direction. Non-negotiable for trust.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# PUBLIC ENUMS — these drive what the client sees
# =============================================================================


class PublicState(str, Enum):
    """What the client sees on their dashboard."""
    HOLD = "HOLD"
    BUY = "BUY"
    SELL = "SELL"


class Direction(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"

    @classmethod
    def from_signal_type(cls, signal_type: Any) -> "Direction":
        """Coerce from ConfluenceDetector.SignalType or a string."""
        value = getattr(signal_type, "value", signal_type)
        if value in ("LONG", "BUY"):
            return cls.LONG
        if value in ("SHORT", "SELL"):
            return cls.SHORT
        raise ValueError(f"Unknown signal direction: {signal_type!r}")

    def to_public(self) -> PublicState:
        return PublicState.BUY if self is Direction.LONG else PublicState.SELL


class ExitReason(str, Enum):
    """Why an active signal flipped back to HOLD. Shown to the client."""
    TARGET_REACHED = "target_reached"
    INVALIDATED = "invalidated"           # stop-loss hit
    TIME_EXPIRED = "time_expired"
    SCORE_DECAYED = "score_decayed"
    REGIME_SHIFTED = "regime_shifted"
    OPPOSING_SIGNAL = "opposing_signal"


class _Phase(str, Enum):
    """Internal phase. Projects down to :class:`PublicState` via :meth:`public`."""
    IDLE = "idle"                # HOLD, no pending direction
    ARMING = "arming"            # HOLD, accumulating confirmation bars
    ACTIVE_LONG = "active_long"  # BUY
    ACTIVE_SHORT = "active_short"  # SELL
    COOLDOWN = "cooldown"        # HOLD, lockout after exit

    def public(self) -> PublicState:
        if self is _Phase.ACTIVE_LONG:
            return PublicState.BUY
        if self is _Phase.ACTIVE_SHORT:
            return PublicState.SELL
        return PublicState.HOLD


# =============================================================================
# CONFIG
# =============================================================================


@dataclass(frozen=True)
class StateMachineConfig:
    """All tunable thresholds.

    Defaults are tuned for XAUUSD on 15-minute bars. All ranges are validated
    in ``__post_init__`` so an invalid config never reaches production.
    """

    # --- Hysteresis (Rule 1) --------------------------------------------- #
    enter_threshold: float = 75.0
    exit_threshold: float = 55.0

    # --- Confirmation window (Rule 2) ------------------------------------ #
    confirm_bars: int = 2

    # --- Signal lifetime (Rule 3) ---------------------------------------- #
    max_signal_age_bars: int = 12
    """Hard cap on how long a signal can stay active, regardless of score."""

    silent_bars_before_score_exit: int = 2
    """If the confluence detector returns no signal for this many bars in a
    row while we're active, we treat it as score-decay and exit. Defence-in-
    depth in case the detector's own ``min_score`` is above our exit_threshold."""

    high_vol_forces_exit: bool = True
    """Flipping to 'high' volatility regime mid-signal → exit with REGIME_SHIFTED."""

    # --- Cooldown (Rule 4) ----------------------------------------------- #
    cooldown_bars: int = 2

    # --- Observability --------------------------------------------------- #
    transition_history_max: int = 200

    # --- Symbol (for logging; doesn't affect logic) ---------------------- #
    symbol: str = "XAUUSD"

    def __post_init__(self) -> None:
        if not 0.0 <= self.exit_threshold < self.enter_threshold <= 100.0:
            raise ValueError(
                f"Require 0 <= exit_threshold ({self.exit_threshold}) "
                f"< enter_threshold ({self.enter_threshold}) <= 100"
            )
        if self.confirm_bars < 1:
            raise ValueError(f"confirm_bars must be >= 1, got {self.confirm_bars}")
        if self.cooldown_bars < 0:
            raise ValueError(f"cooldown_bars must be >= 0, got {self.cooldown_bars}")
        if self.max_signal_age_bars < 1:
            raise ValueError(f"max_signal_age_bars must be >= 1, got {self.max_signal_age_bars}")
        if self.silent_bars_before_score_exit < 1:
            raise ValueError(
                "silent_bars_before_score_exit must be >= 1, "
                f"got {self.silent_bars_before_score_exit}"
            )
        if self.transition_history_max < 1:
            raise ValueError(
                f"transition_history_max must be >= 1, got {self.transition_history_max}"
            )


# =============================================================================
# INPUT / OUTPUT STRUCTS
# =============================================================================


@dataclass(frozen=True)
class BarInput:
    """One bar of market data + optional ConfluenceSignal.

    ``signal`` is ``None`` when the ConfluenceDetector did not produce a
    signal for this bar (e.g., score below its ``min_score``, or no BOS).
    """

    bar_timestamp: str       # ISO-8601 preferred; compared lexically if orderable
    high: float
    low: float
    close: float
    signal: Optional[Any] = None        # ConfluenceSignal (duck-typed to avoid circular import)
    vol_regime: Optional[str] = None    # "low" | "normal" | "high" | None
    structure_broken: bool = False      # optional swing-break flag

    def __post_init__(self) -> None:
        for name in ("high", "low", "close"):
            val = getattr(self, name)
            if not math.isfinite(val):
                raise ValueError(f"BarInput.{name} must be finite, got {val}")
            if val <= 0:
                raise ValueError(f"BarInput.{name} must be positive, got {val}")
        if self.low > self.high:
            raise ValueError(f"BarInput.low ({self.low}) > high ({self.high})")
        if not (self.low <= self.close <= self.high):
            raise ValueError(
                f"BarInput.close ({self.close}) must lie within "
                f"[low={self.low}, high={self.high}]"
            )
        if not self.bar_timestamp:
            raise ValueError("BarInput.bar_timestamp must be non-empty")


@dataclass(frozen=True)
class StateTransition:
    """Emitted when on_bar changes phase. Callers publish this."""

    at_bar: str
    from_state: PublicState
    to_state: PublicState
    reason: str                          # human-readable summary
    exit_reason: Optional[ExitReason] = None
    direction: Optional[Direction] = None
    active_signal: Optional[Any] = None  # ConfluenceSignal on BUY/SELL entry
    entry_price: Optional[float] = None
    exit_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "at_bar": self.at_bar,
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "reason": self.reason,
            "exit_reason": self.exit_reason.value if self.exit_reason else None,
            "direction": self.direction.value if self.direction else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
        }


@dataclass(frozen=True)
class StateSnapshot:
    """Immutable view of the machine's current state — what the dashboard shows."""

    state: PublicState
    direction: Optional[Direction]
    active_signal: Optional[Any]         # ConfluenceSignal
    bars_in_state: int
    bars_remaining: Optional[int]        # countdown to time-expiry (ACTIVE only)
    cooldown_bars_remaining: Optional[int]
    confirmation_progress: Optional[Tuple[int, int]]  # (have, need) when ARMING
    entered_at_bar: Optional[str]
    entered_at_price: Optional[float]
    last_exit_reason: Optional[ExitReason]
    last_bar_processed: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        signal_dict: Optional[Dict[str, Any]] = None
        if self.active_signal is not None and hasattr(self.active_signal, "to_dict"):
            try:
                signal_dict = self.active_signal.to_dict()
            except Exception:  # pragma: no cover — defensive
                signal_dict = {"signal_id": getattr(self.active_signal, "signal_id", None)}
        return {
            "state": self.state.value,
            "direction": self.direction.value if self.direction else None,
            "active_signal": signal_dict,
            "bars_in_state": self.bars_in_state,
            "bars_remaining": self.bars_remaining,
            "cooldown_bars_remaining": self.cooldown_bars_remaining,
            "confirmation_progress": (
                list(self.confirmation_progress) if self.confirmation_progress else None
            ),
            "entered_at_bar": self.entered_at_bar,
            "entered_at_price": self.entered_at_price,
            "last_exit_reason": self.last_exit_reason.value if self.last_exit_reason else None,
            "last_bar_processed": self.last_bar_processed,
        }


# =============================================================================
# THE STATE MACHINE
# =============================================================================


class SignalStateMachine:
    """Enforces stable HOLD / BUY / SELL transitions from a stream of bars.

    Typical usage from the scanner::

        sm = SignalStateMachine(StateMachineConfig(symbol="XAUUSD"))
        for bar in bars:
            snapshot, transition = sm.on_bar(bar)
            if transition is not None:
                signal_store.publish_transition(transition, snapshot)

    Every ``on_bar`` returns ``(snapshot, transition)``. ``snapshot`` is
    always the current public view; ``transition`` is non-None only on bars
    where the state changed (HOLD→BUY, BUY→HOLD, etc.).
    """

    __slots__ = (
        "_config",
        "_lock",
        "_phase",
        "_pending_direction",
        "_pending_bars",
        "_active_signal",
        "_active_direction",
        "_active_bars",
        "_active_entry_bar",
        "_active_entry_price",
        "_silent_bars",
        "_cooldown_left",
        "_last_exit_reason",
        "_last_bar_ts",
        "_bars_since_phase_change",
        "_stats",
        "_transition_history",
    )

    def __init__(self, config: Optional[StateMachineConfig] = None):
        self._config = config or StateMachineConfig()
        self._lock = threading.RLock()

        # Core state
        self._phase: _Phase = _Phase.IDLE
        self._pending_direction: Optional[Direction] = None
        self._pending_bars: int = 0
        self._active_signal: Optional[Any] = None
        self._active_direction: Optional[Direction] = None
        self._active_bars: int = 0
        self._active_entry_bar: Optional[str] = None
        self._active_entry_price: Optional[float] = None
        self._silent_bars: int = 0
        self._cooldown_left: int = 0
        self._last_exit_reason: Optional[ExitReason] = None
        self._last_bar_ts: Optional[str] = None
        self._bars_since_phase_change: int = 0

        # Observability
        self._stats: Dict[str, Any] = {
            "bars_processed": 0,
            "bars_rejected_duplicate": 0,
            "bars_rejected_out_of_order": 0,
            "bars_rejected_invalid": 0,
            "arms_started": 0,
            "arms_confirmed": 0,
            "arms_aborted": 0,
            "signals_emitted": 0,
            "exits_by_reason": {r.value: 0 for r in ExitReason},
            "total_active_bars": 0,
        }
        max_hist = self._config.transition_history_max
        self._transition_history: Deque[Dict[str, Any]] = deque(maxlen=max_hist)

    # ------------------------------------------------------------------ #
    # PUBLIC API
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> StateMachineConfig:
        return self._config

    def snapshot(self) -> StateSnapshot:
        """Thread-safe read of current state."""
        with self._lock:
            return self._build_snapshot()

    def get_stats(self) -> Dict[str, Any]:
        """Telemetry counters. Safe to call from monitoring threads."""
        with self._lock:
            out = dict(self._stats)
            out["exits_by_reason"] = dict(out["exits_by_reason"])
            out["phase"] = self._phase.value
            out["public_state"] = self._phase.public().value
            avg_life = (
                out["total_active_bars"] / out["signals_emitted"]
                if out["signals_emitted"] > 0
                else 0.0
            )
            out["avg_signal_lifetime_bars"] = round(avg_life, 2)
            if out["arms_started"] > 0:
                out["confirmation_rate"] = round(
                    out["arms_confirmed"] / out["arms_started"], 3
                )
            else:
                out["confirmation_rate"] = None
            return out

    def transition_history(self) -> List[Dict[str, Any]]:
        """Recent transitions (oldest → newest), capped at ``transition_history_max``."""
        with self._lock:
            return list(self._transition_history)

    def on_bar(
        self, bar: BarInput
    ) -> Tuple[StateSnapshot, Optional[StateTransition]]:
        """Process one bar. Returns ``(snapshot, transition_or_none)``.

        Never raises on valid :class:`BarInput`. Malformed bars (duplicate
        timestamp, out-of-order timestamp) are logged, counted, and the
        current snapshot is returned with no transition.
        """
        with self._lock:
            self._stats["bars_processed"] += 1

            # --- Idempotency & ordering checks (security / reliability) --- #
            if self._last_bar_ts is not None:
                if bar.bar_timestamp == self._last_bar_ts:
                    self._stats["bars_rejected_duplicate"] += 1
                    logger.debug(
                        "[%s] Duplicate bar %s rejected (idempotent no-op)",
                        self._config.symbol, bar.bar_timestamp,
                    )
                    return self._build_snapshot(), None
                if bar.bar_timestamp < self._last_bar_ts:
                    self._stats["bars_rejected_out_of_order"] += 1
                    logger.warning(
                        "[%s] Out-of-order bar %s < last %s — rejected",
                        self._config.symbol, bar.bar_timestamp, self._last_bar_ts,
                    )
                    return self._build_snapshot(), None

            transition = self._process_bar(bar)
            self._last_bar_ts = bar.bar_timestamp
            self._bars_since_phase_change += 1
            # active-bar accounting (total_active_bars) is handled inside
            # _confirm_arming and _step_active so the exit bar is counted too.

            snapshot = self._build_snapshot()
            if transition is not None:
                self._transition_history.append(transition.to_dict())
            return snapshot, transition

    def reset(self) -> None:
        """Reset to IDLE. Stats and history are preserved. Use with care."""
        with self._lock:
            self._phase = _Phase.IDLE
            self._pending_direction = None
            self._pending_bars = 0
            self._active_signal = None
            self._active_direction = None
            self._active_bars = 0
            self._active_entry_bar = None
            self._active_entry_price = None
            self._silent_bars = 0
            self._cooldown_left = 0
            self._bars_since_phase_change = 0
            logger.info("[%s] State machine reset to IDLE", self._config.symbol)

    # ------------------------------------------------------------------ #
    # CORE TRANSITION LOGIC
    # ------------------------------------------------------------------ #

    def _process_bar(self, bar: BarInput) -> Optional[StateTransition]:
        """Dispatch to the phase-specific handler. Returns a transition if any."""
        score, sig_direction = self._extract_score_and_direction(bar.signal)

        # Active phases first — check hard exits (TP/SL) before anything else
        if self._phase in (_Phase.ACTIVE_LONG, _Phase.ACTIVE_SHORT):
            return self._step_active(bar, score, sig_direction)
        if self._phase is _Phase.COOLDOWN:
            return self._step_cooldown(bar, score, sig_direction)
        if self._phase is _Phase.ARMING:
            return self._step_arming(bar, score, sig_direction)
        # IDLE
        return self._step_idle(bar, score, sig_direction)

    # --- IDLE ----------------------------------------------------------- #

    def _step_idle(
        self, bar: BarInput, score: float, direction: Optional[Direction]
    ) -> Optional[StateTransition]:
        if direction is not None and score >= self._config.enter_threshold:
            self._pending_direction = direction
            self._pending_bars = 1
            self._phase = _Phase.ARMING
            self._bars_since_phase_change = 0
            self._stats["arms_started"] += 1
            logger.info(
                "[%s] ARMING %s at bar %s (score=%.1f, %d/%d confirms)",
                self._config.symbol, direction.value, bar.bar_timestamp,
                score, self._pending_bars, self._config.confirm_bars,
            )
            # If confirm_bars == 1, arm and publish in the same bar
            if self._pending_bars >= self._config.confirm_bars:
                return self._confirm_arming(bar)
        return None

    # --- ARMING --------------------------------------------------------- #

    def _step_arming(
        self, bar: BarInput, score: float, direction: Optional[Direction]
    ) -> Optional[StateTransition]:
        assert self._pending_direction is not None
        still_aligned = (
            direction is self._pending_direction
            and score >= self._config.enter_threshold
        )
        if not still_aligned:
            self._stats["arms_aborted"] += 1
            aborted = self._pending_direction
            self._pending_direction = None
            self._pending_bars = 0
            self._phase = _Phase.IDLE
            self._bars_since_phase_change = 0
            logger.info(
                "[%s] ARMING %s aborted at bar %s (score=%.1f)",
                self._config.symbol, aborted.value, bar.bar_timestamp, score,
            )
            # Could immediately arm the opposite direction if conditions met;
            # prefer conservatism — always require a fresh bar to re-arm.
            return None

        self._pending_bars += 1
        if self._pending_bars >= self._config.confirm_bars:
            return self._confirm_arming(bar)
        return None

    def _confirm_arming(self, bar: BarInput) -> Optional[StateTransition]:
        """Promote ARMING → ACTIVE_{LONG,SHORT}. Emits the entry transition."""
        direction = self._pending_direction
        signal = bar.signal
        assert direction is not None
        # Signal must be present on the confirmation bar (it's what triggered us)
        if signal is None:
            # Pathological: we armed on a bar with a signal but the confirming
            # bar has no signal object to publish. Abort safely.
            logger.warning(
                "[%s] Confirm bar %s has no signal object — aborting arm",
                self._config.symbol, bar.bar_timestamp,
            )
            self._pending_direction = None
            self._pending_bars = 0
            self._phase = _Phase.IDLE
            self._bars_since_phase_change = 0
            self._stats["arms_aborted"] += 1
            return None

        entry_price = float(getattr(signal, "entry_price", bar.close))
        self._active_signal = signal
        self._active_direction = direction
        self._active_bars = 1  # entry bar counts as the first active bar
        self._active_entry_bar = bar.bar_timestamp
        self._active_entry_price = entry_price
        self._silent_bars = 0
        self._pending_direction = None
        self._pending_bars = 0
        self._phase = (
            _Phase.ACTIVE_LONG if direction is Direction.LONG else _Phase.ACTIVE_SHORT
        )
        self._bars_since_phase_change = 0
        self._stats["arms_confirmed"] += 1
        self._stats["signals_emitted"] += 1
        self._stats["total_active_bars"] += 1  # entry bar is active for the client

        logger.info(
            "[%s] %s CONFIRMED at bar %s (entry=%.4f, score=%.1f)",
            self._config.symbol, direction.value, bar.bar_timestamp,
            entry_price, float(getattr(signal, "confluence_score", 0.0)),
        )
        return StateTransition(
            at_bar=bar.bar_timestamp,
            from_state=PublicState.HOLD,
            to_state=direction.to_public(),
            reason=f"{direction.value} confluence confirmed over {self._config.confirm_bars} bars",
            direction=direction,
            active_signal=signal,
            entry_price=entry_price,
        )

    # --- ACTIVE --------------------------------------------------------- #

    def _step_active(
        self, bar: BarInput, score: float, direction: Optional[Direction]
    ) -> Optional[StateTransition]:
        """Apply the 6 exit rules in priority order.

        _step_active is only dispatched on post-entry bars. Incrementing
        _active_bars and total_active_bars at the top ensures the exit bar
        is counted as an active bar (the client saw BUY/SELL at its start).
        """
        assert self._active_signal is not None
        assert self._active_direction is not None
        signal = self._active_signal
        active_dir = self._active_direction

        # Advance age counter FIRST so the exit-bar is counted toward lifetime.
        self._active_bars += 1
        self._stats["total_active_bars"] += 1

        tp = float(getattr(signal, "take_profit"))
        sl = float(getattr(signal, "stop_loss"))

        # 1. TARGET_REACHED / INVALIDATED — based on bar high/low touching TP/SL
        if active_dir is Direction.LONG:
            if bar.high >= tp:
                return self._exit(bar, ExitReason.TARGET_REACHED, exit_price=tp)
            if bar.low <= sl:
                return self._exit(bar, ExitReason.INVALIDATED, exit_price=sl)
        else:
            if bar.low <= tp:
                return self._exit(bar, ExitReason.TARGET_REACHED, exit_price=tp)
            if bar.high >= sl:
                return self._exit(bar, ExitReason.INVALIDATED, exit_price=sl)

        # 2. REGIME_SHIFTED — vol regime flipped to 'high' mid-signal.
        # _step_active only runs on post-entry bars, so this check never
        # fires on the entry bar itself.
        if self._config.high_vol_forces_exit and (bar.vol_regime or "").lower() == "high":
            return self._exit(bar, ExitReason.REGIME_SHIFTED, exit_price=bar.close)

        # 3. Structure broken (caller-supplied)
        if bar.structure_broken:
            return self._exit(bar, ExitReason.INVALIDATED, exit_price=bar.close)

        # 4. OPPOSING_SIGNAL — detector produced a strong opposite-direction signal
        if (
            direction is not None
            and direction is not active_dir
            and score >= self._config.enter_threshold
        ):
            return self._exit(bar, ExitReason.OPPOSING_SIGNAL, exit_price=bar.close)

        # 5. SCORE_DECAYED — same-direction score dropped below exit_threshold
        #    OR N consecutive bars of silence from the detector.
        if direction is active_dir and score < self._config.exit_threshold:
            return self._exit(bar, ExitReason.SCORE_DECAYED, exit_price=bar.close)
        if direction is None and bar.signal is None:
            self._silent_bars += 1
            if self._silent_bars >= self._config.silent_bars_before_score_exit:
                return self._exit(bar, ExitReason.SCORE_DECAYED, exit_price=bar.close)
        else:
            # Same-direction signal above exit_threshold — refresh silence counter
            self._silent_bars = 0

        # 6. TIME_EXPIRED — hard cap on signal age. _active_bars was already
        # advanced at the top of this method, so a value of N means this is
        # the Nth bar of active life (including the entry bar).
        if self._active_bars >= self._config.max_signal_age_bars:
            return self._exit(bar, ExitReason.TIME_EXPIRED, exit_price=bar.close)

        return None

    def _exit(
        self, bar: BarInput, reason: ExitReason, exit_price: float
    ) -> StateTransition:
        """Transition ACTIVE_* → COOLDOWN. Emits the exit transition."""
        from_state = self._phase.public()
        direction = self._active_direction
        signal_ref = self._active_signal
        entry_bar = self._active_entry_bar

        # Move to cooldown
        self._phase = _Phase.COOLDOWN
        self._cooldown_left = self._config.cooldown_bars
        self._active_signal = None
        self._active_direction = None
        self._active_bars = 0
        self._active_entry_bar = None
        self._active_entry_price = None
        self._silent_bars = 0
        self._last_exit_reason = reason
        self._bars_since_phase_change = 0
        self._stats["exits_by_reason"][reason.value] += 1

        logger.info(
            "[%s] EXIT %s at bar %s (reason=%s, exit=%.4f, entry_bar=%s)",
            self._config.symbol, from_state.value, bar.bar_timestamp,
            reason.value, exit_price, entry_bar,
        )
        return StateTransition(
            at_bar=bar.bar_timestamp,
            from_state=from_state,
            to_state=PublicState.HOLD,
            reason=f"{from_state.value} exited: {reason.value}",
            exit_reason=reason,
            direction=direction,
            active_signal=signal_ref,
            exit_price=exit_price,
        )

    # --- COOLDOWN ------------------------------------------------------- #

    def _step_cooldown(
        self, bar: BarInput, score: float, direction: Optional[Direction]
    ) -> Optional[StateTransition]:
        """Block new entries for ``cooldown_bars`` after an exit."""
        self._cooldown_left -= 1
        if self._cooldown_left <= 0:
            self._phase = _Phase.IDLE
            self._cooldown_left = 0
            self._bars_since_phase_change = 0
            # Don't arm on this same bar — enforce a clean HOLD window first.
            # Worst case: arming is delayed by one extra bar, which is a price
            # worth paying for preventing whipsaw re-entries.
        return None

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_score_and_direction(
        signal: Optional[Any],
    ) -> Tuple[float, Optional[Direction]]:
        """Duck-type-read score + direction from a ConfluenceSignal-like object."""
        if signal is None:
            return 0.0, None
        raw_score = getattr(signal, "confluence_score", 0.0)
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            return 0.0, None
        if not math.isfinite(score):
            return 0.0, None
        score = max(0.0, min(100.0, score))

        sig_type = getattr(signal, "signal_type", None)
        if sig_type is None:
            return score, None
        try:
            direction = Direction.from_signal_type(sig_type)
        except ValueError:
            return score, None
        return score, direction

    def _build_snapshot(self) -> StateSnapshot:
        """Construct an immutable StateSnapshot from the current internal state."""
        public = self._phase.public()
        direction: Optional[Direction] = None
        bars_remaining: Optional[int] = None
        cooldown_remaining: Optional[int] = None
        confirm_progress: Optional[Tuple[int, int]] = None
        bars_in_state = self._bars_since_phase_change

        if self._phase is _Phase.ACTIVE_LONG:
            direction = Direction.LONG
        elif self._phase is _Phase.ACTIVE_SHORT:
            direction = Direction.SHORT

        if self._phase in (_Phase.ACTIVE_LONG, _Phase.ACTIVE_SHORT):
            bars_in_state = self._active_bars
            bars_remaining = max(
                0, self._config.max_signal_age_bars - self._active_bars
            )
        elif self._phase is _Phase.ARMING and self._pending_direction is not None:
            confirm_progress = (self._pending_bars, self._config.confirm_bars)
        elif self._phase is _Phase.COOLDOWN:
            cooldown_remaining = max(0, self._cooldown_left)

        return StateSnapshot(
            state=public,
            direction=direction,
            active_signal=self._active_signal,
            bars_in_state=bars_in_state,
            bars_remaining=bars_remaining,
            cooldown_bars_remaining=cooldown_remaining,
            confirmation_progress=confirm_progress,
            entered_at_bar=self._active_entry_bar,
            entered_at_price=self._active_entry_price,
            last_exit_reason=self._last_exit_reason,
            last_bar_processed=self._last_bar_ts,
        )

    # ------------------------------------------------------------------ #
    # PERSISTENCE — survive process restart
    # ------------------------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        """Serialise enough state to reconstruct the machine.

        The active signal is serialised via its own ``to_dict()`` if available;
        otherwise we store its ``signal_id`` and the caller must re-attach the
        full object on rehydration (via :meth:`rehydrate_signal`).
        """
        with self._lock:
            sig_payload: Optional[Dict[str, Any]] = None
            if self._active_signal is not None and hasattr(self._active_signal, "to_dict"):
                try:
                    sig_payload = self._active_signal.to_dict()
                except Exception:  # pragma: no cover — defensive
                    sig_payload = {
                        "signal_id": getattr(self._active_signal, "signal_id", None)
                    }
            return {
                "schema_version": 1,
                "config": asdict(self._config),
                "phase": self._phase.value,
                "pending_direction": (
                    self._pending_direction.value if self._pending_direction else None
                ),
                "pending_bars": self._pending_bars,
                "active_signal": sig_payload,
                "active_direction": (
                    self._active_direction.value if self._active_direction else None
                ),
                "active_bars": self._active_bars,
                "active_entry_bar": self._active_entry_bar,
                "active_entry_price": self._active_entry_price,
                "silent_bars": self._silent_bars,
                "cooldown_left": self._cooldown_left,
                "last_exit_reason": (
                    self._last_exit_reason.value if self._last_exit_reason else None
                ),
                "last_bar_ts": self._last_bar_ts,
                "bars_since_phase_change": self._bars_since_phase_change,
                "stats": {**self._stats, "exits_by_reason": dict(self._stats["exits_by_reason"])},
                "transition_history": list(self._transition_history),
            }

    @classmethod
    def from_dict(
        cls,
        payload: Dict[str, Any],
        signal_rehydrator: Optional[Any] = None,
    ) -> "SignalStateMachine":
        """Reconstruct a machine from :meth:`to_dict` output.

        ``signal_rehydrator`` is an optional callable ``dict -> ConfluenceSignal``
        used to restore the active signal object. If not provided, the stored
        payload dict is kept as a placeholder (still usable for display, but
        ``.to_dict()`` methods on it may be unavailable).
        """
        if not isinstance(payload, dict):
            raise TypeError(f"payload must be dict, got {type(payload).__name__}")
        if payload.get("schema_version") != 1:
            raise ValueError(
                f"Unsupported state-machine schema_version: {payload.get('schema_version')}"
            )
        config_data = dict(payload.get("config") or {})
        machine = cls(StateMachineConfig(**config_data))
        with machine._lock:
            machine._phase = _Phase(payload["phase"])
            pd_val = payload.get("pending_direction")
            machine._pending_direction = Direction(pd_val) if pd_val else None
            machine._pending_bars = int(payload.get("pending_bars", 0))
            sig_data = payload.get("active_signal")
            if sig_data is not None and signal_rehydrator is not None:
                machine._active_signal = signal_rehydrator(sig_data)
            else:
                machine._active_signal = sig_data
            ad_val = payload.get("active_direction")
            machine._active_direction = Direction(ad_val) if ad_val else None
            machine._active_bars = int(payload.get("active_bars", 0))
            machine._active_entry_bar = payload.get("active_entry_bar")
            machine._active_entry_price = payload.get("active_entry_price")
            machine._silent_bars = int(payload.get("silent_bars", 0))
            machine._cooldown_left = int(payload.get("cooldown_left", 0))
            er_val = payload.get("last_exit_reason")
            machine._last_exit_reason = ExitReason(er_val) if er_val else None
            machine._last_bar_ts = payload.get("last_bar_ts")
            machine._bars_since_phase_change = int(
                payload.get("bars_since_phase_change", 0)
            )
            stats = payload.get("stats") or {}
            # Merge stats defensively — missing keys fall back to defaults
            for k in machine._stats:
                if k in stats:
                    machine._stats[k] = (
                        dict(stats[k]) if k == "exits_by_reason" else stats[k]
                    )
            # Ensure all ExitReason keys are present in counter map
            for r in ExitReason:
                machine._stats["exits_by_reason"].setdefault(r.value, 0)
            history = payload.get("transition_history") or []
            machine._transition_history = deque(
                history, maxlen=machine._config.transition_history_max
            )
        return machine


# =============================================================================
# design note
# =============================================================================
# The machine is intentionally pure-logic: no I/O, no clocks, no randomness.
# That lets us (1) deterministically replay historical bars to backtest any
# config, (2) write exhaustive unit tests that feed synthetic oscillating
# score series and assert no flicker, and (3) swap the config at runtime via
# A/B experiments without touching the scanner or the persistence layer.
#
# The scanner owns I/O (fetching bars, publishing signals). The state
# machine owns only "what state should the client see, and why". This
# separation is what makes the trust story testable.
