"""Live state endpoint — ``GET /api/v1/signals/state``.

Returns the current :class:`PublicStateResponse` pulled from the scanner's
signal state machine. This is the endpoint the client dashboard polls.

Design choice: HOLD is treated as a first-class signal, not a null. Every
response carries a ``headline`` + ``detail`` string crafted to explain
*why* the system is in the state it is. That transparency is the
commercial differentiator — the client should always feel informed,
never ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Request

from src.api.auth import require_api_key
from src.api.models import (
    ActiveSignalPayload,
    HoldSubState,
    PublicStateResponse,
)

router = APIRouter(prefix="/api/v1/signals", tags=["state"])


# =============================================================================
# HOLD SUB-STATE MESSAGING — what the client reads when no trade is live
# =============================================================================

_IDLE_HEADLINE = "WATCHING"
_ARMING_HEADLINE = "ARMING"
_COOLDOWN_HEADLINE = "COOLDOWN"
_POST_EXIT_HEADLINE = "RESET"

_IDLE_DETAIL = (
    "No setup meets our confluence threshold right now. The market is "
    "inconclusive — standing aside until structure clarifies."
)
_ARMING_DETAIL_TPL = (
    "Building a {direction} setup — {have} of {need} confirmation bars "
    "passed. A clean entry is forming; one more aligned bar would confirm."
)
_COOLDOWN_DETAIL_TPL = (
    "Forced cooldown after the last signal — {remaining} "
    "bar{plural} until we can arm a new one. Prevents whipsaw re-entries."
)
_POST_EXIT_DETAIL_TPL = (
    "Signal just ended: {reason}. Scanning for the next setup."
)

_EXIT_REASON_BLURB: Dict[str, str] = {
    "target_reached": "target reached",
    "invalidated": "invalidated at stop-loss",
    "time_expired": "time-expired",
    "score_decayed": "confluence faded",
    "regime_shifted": "volatility regime shifted",
    "opposing_signal": "opposite setup overrode it",
}


def _active_message(direction: str, bars_in_state: int, bars_remaining: Optional[int]) -> str:
    window = f"{bars_in_state} bar{'s' if bars_in_state != 1 else ''} active"
    countdown = ""
    if bars_remaining is not None:
        countdown = (
            f", auto-expires in {bars_remaining} bar{'s' if bars_remaining != 1 else ''}"
        )
    return f"{direction} signal live ({window}{countdown})."


# =============================================================================
# BUILDERS
# =============================================================================

def _build_active_payload(signal_obj: Any, direction: Optional[str]) -> Optional[ActiveSignalPayload]:
    """Coerce a ConfluenceSignal (or dict) into the response payload."""
    if signal_obj is None:
        return None

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(signal_obj, dict):
            return signal_obj.get(name, default)
        return getattr(signal_obj, name, default)

    sig_type = _get("signal_type")
    dir_value = direction or (
        sig_type.value if hasattr(sig_type, "value") else str(sig_type or "LONG")
    )
    try:
        return ActiveSignalPayload(
            signal_id=str(_get("signal_id", "")),
            symbol=str(_get("symbol", "")),
            direction=dir_value,
            entry_price=float(_get("entry_price", 0.0)),
            stop_loss=float(_get("stop_loss", 0.0)),
            take_profit=float(_get("take_profit", 0.0)),
            rr_ratio=float(_get("rr_ratio", 0.0)),
            confluence_score=float(_get("confluence_score", 0.0)),
            atr=_nullable_float(_get("atr")),
            vol_regime=_nullable_str(_get("vol_regime")),
            vol_forecast_atr=_nullable_float(_get("vol_forecast_atr")),
        )
    except (TypeError, ValueError):
        return None


def _nullable_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _nullable_str(v: Any) -> Optional[str]:
    if v is None:
        return None
    return str(v)


def _resolve_hold_sub_state(
    cooldown_remaining: Optional[int],
    confirmation_progress: Optional[tuple],
    last_exit_reason: Optional[str],
) -> HoldSubState:
    """Classify HOLD into one of four commercially-meaningful sub-states."""
    if cooldown_remaining is not None and cooldown_remaining > 0:
        return HoldSubState.COOLDOWN
    if confirmation_progress is not None:
        return HoldSubState.ARMING
    if last_exit_reason is not None and cooldown_remaining == 0:
        return HoldSubState.POST_EXIT
    return HoldSubState.IDLE


def _message_for_hold(
    sub_state: HoldSubState,
    confirmation_progress: Optional[tuple],
    cooldown_remaining: Optional[int],
    last_exit_reason: Optional[str],
    pending_direction: Optional[str],
) -> tuple[str, str]:
    if sub_state is HoldSubState.ARMING and confirmation_progress:
        have, need = confirmation_progress
        direction = pending_direction or "LONG"
        return _ARMING_HEADLINE, _ARMING_DETAIL_TPL.format(
            direction=direction, have=have, need=need,
        )
    if sub_state is HoldSubState.COOLDOWN and cooldown_remaining is not None:
        return _COOLDOWN_HEADLINE, _COOLDOWN_DETAIL_TPL.format(
            remaining=cooldown_remaining,
            plural="s" if cooldown_remaining != 1 else "",
        )
    if sub_state is HoldSubState.POST_EXIT and last_exit_reason:
        blurb = _EXIT_REASON_BLURB.get(last_exit_reason, last_exit_reason)
        return _POST_EXIT_HEADLINE, _POST_EXIT_DETAIL_TPL.format(reason=blurb)
    return _IDLE_HEADLINE, _IDLE_DETAIL


# =============================================================================
# ROUTE
# =============================================================================

@router.get("/state", response_model=PublicStateResponse)
async def get_current_state(
    request: Request,
    subscriber: dict = Depends(require_api_key),
):
    """Return the live public state — HOLD / BUY / SELL with full context.

    If no scanner / state machine is wired, returns an IDLE HOLD snapshot so
    the dashboard degrades gracefully instead of 500'ing.
    """
    app_state = request.app.state.app_state
    scanner = app_state.scanner
    symbol = _guess_symbol(scanner)
    now = datetime.now(timezone.utc).isoformat()

    state_machine = getattr(scanner, "state_machine", None) if scanner is not None else None
    if state_machine is None:
        return PublicStateResponse(
            symbol=symbol,
            state="HOLD",
            headline=_IDLE_HEADLINE,
            detail=_IDLE_DETAIL,
            hold_sub_state=HoldSubState.IDLE,
            bars_in_state=0,
            generated_at=now,
        )

    snap = state_machine.snapshot()
    stats = state_machine.get_stats()

    # Pull the most recent bar data — best-effort, scanner may expose it
    last_bar = getattr(scanner, "_last_bar_ts", None) if scanner else None
    confirm_prog = (
        tuple(snap.confirmation_progress)
        if snap.confirmation_progress is not None
        else None
    )

    if snap.state.value == "HOLD":
        pending_direction = None
        # When arming, the machine holds pending_direction internally;
        # exposed indirectly via confirmation_progress. We also surface any
        # direction hint from the pending direction via private state so the
        # UI can show "Arming LONG" vs "Arming SHORT".
        pending_direction_attr = getattr(state_machine, "_pending_direction", None)
        if pending_direction_attr is not None:
            pending_direction = getattr(pending_direction_attr, "value", None) or str(
                pending_direction_attr
            )
        sub_state = _resolve_hold_sub_state(
            cooldown_remaining=snap.cooldown_bars_remaining,
            confirmation_progress=confirm_prog,
            last_exit_reason=(
                snap.last_exit_reason.value if snap.last_exit_reason else None
            ),
        )
        headline, detail = _message_for_hold(
            sub_state=sub_state,
            confirmation_progress=confirm_prog,
            cooldown_remaining=snap.cooldown_bars_remaining,
            last_exit_reason=(
                snap.last_exit_reason.value if snap.last_exit_reason else None
            ),
            pending_direction=pending_direction,
        )
        return PublicStateResponse(
            symbol=symbol,
            state="HOLD",
            headline=headline,
            detail=detail,
            hold_sub_state=sub_state,
            direction=None,
            active_signal=None,
            bars_in_state=snap.bars_in_state,
            bars_remaining=None,
            cooldown_bars_remaining=snap.cooldown_bars_remaining,
            confirmation_progress=list(confirm_prog) if confirm_prog else None,
            last_exit_reason=(
                snap.last_exit_reason.value if snap.last_exit_reason else None
            ),
            last_bar_processed=snap.last_bar_processed or last_bar,
            stats=stats,
            generated_at=now,
        )

    # BUY / SELL branch
    direction_value = snap.direction.value if snap.direction else None
    active_payload = _build_active_payload(snap.active_signal, direction_value)
    headline = snap.state.value
    detail = _active_message(
        direction=direction_value or "", bars_in_state=snap.bars_in_state,
        bars_remaining=snap.bars_remaining,
    )
    return PublicStateResponse(
        symbol=symbol,
        state=snap.state.value,
        headline=headline,
        detail=detail,
        hold_sub_state=None,
        direction=direction_value,
        active_signal=active_payload,
        bars_in_state=snap.bars_in_state,
        bars_remaining=snap.bars_remaining,
        cooldown_bars_remaining=None,
        confirmation_progress=None,
        last_exit_reason=None,
        last_bar_processed=snap.last_bar_processed or last_bar,
        stats=stats,
        generated_at=now,
    )


def _guess_symbol(scanner: Any) -> str:
    """Best-effort symbol extraction from the scanner. Defaults to XAUUSD."""
    if scanner is None:
        return "XAUUSD"
    for attr in ("_symbol", "symbol"):
        val = getattr(scanner, attr, None)
        if val:
            return str(val)
    # MultiSymbolScanner case: expose first symbol
    scanners = getattr(scanner, "scanners", None)
    if scanners:
        first = next(iter(scanners), None)
        if first:
            return str(first)
    return "XAUUSD"
