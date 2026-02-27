"""Client-facing signal endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request, Response

from src.api.auth import require_api_key
from src.api.models import (
    SignalAction,
    SignalHistoryItem,
    SignalHistoryResponse,
    SignalResponse,
)

router = APIRouter(prefix="/api/v1/signals", tags=["signals"])


@router.get(
    "/current",
    response_model=SignalResponse,
    responses={204: {"description": "No signal available"}},
)
async def get_current_signal(request: Request, subscriber: dict = Depends(require_api_key)):
    """Return the latest published signal, or 204 if none exists."""
    store = request.app.state.app_state.signal_store
    record = store.get_current()
    if record is None:
        return Response(status_code=204)
    return SignalResponse(
        signal_id=record.signal_id,
        action=SignalAction(record.action),
        symbol=record.symbol,
        entry_price=record.entry_price,
        stop_loss=record.stop_loss,
        take_profit=record.take_profit,
        rr_ratio=record.rr_ratio,
        created_at=record.created_at,
    )


@router.get("/history", response_model=SignalHistoryResponse)
async def get_signal_history(
    request: Request,
    subscriber: dict = Depends(require_api_key),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
):
    """Return paginated signal history."""
    store = request.app.state.app_state.signal_store
    records, total = store.get_history(page=page, page_size=page_size)
    return SignalHistoryResponse(
        signals=[
            SignalHistoryItem(
                signal_id=r.signal_id,
                action=SignalAction(r.action),
                symbol=r.symbol,
                entry_price=r.entry_price,
                stop_loss=r.stop_loss,
                take_profit=r.take_profit,
                rr_ratio=r.rr_ratio,
                created_at=r.created_at,
                outcome=r.outcome,
                pnl_pips=r.pnl_pips,
                closed_at=r.closed_at,
            )
            for r in records
        ],
        page=page,
        page_size=page_size,
        total=total,
    )
