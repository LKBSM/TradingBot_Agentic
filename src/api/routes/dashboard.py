"""Subscriber-facing dashboard endpoints — signal performance & equity curve."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from src.api.auth import require_api_key
from src.api.models import (
    EquityCurvePoint,
    EquityCurveResponse,
    PerformanceSummaryResponse,
)

router = APIRouter(prefix="/api/v1/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=PerformanceSummaryResponse)
async def get_performance_summary(
    request: Request,
    subscriber: dict = Depends(require_api_key),
    days: int = Query(30, ge=1, le=365),
):
    """Return aggregated signal performance for the last N days."""
    tracker = request.app.state.app_state.signal_tracker
    if tracker is None:
        return PerformanceSummaryResponse(
            total_signals=0,
            winning=0,
            losing=0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_rr=0.0,
            cumulative_pnl=0.0,
            sharpe_30d=0.0,
            max_drawdown_pct=0.0,
            period_days=days,
        )

    summary = tracker.get_performance_summary(days=days)
    return PerformanceSummaryResponse(
        total_signals=summary["total"],
        winning=summary["won"],
        losing=summary["lost"],
        win_rate=summary["win_rate"],
        profit_factor=summary["profit_factor"],
        avg_rr=summary["avg_rr"],
        cumulative_pnl=summary["cumulative_pnl"],
        sharpe_30d=summary["sharpe_30d"],
        max_drawdown_pct=summary["max_drawdown_pct"],
        period_days=days,
    )


@router.get("/equity-curve", response_model=EquityCurveResponse)
async def get_equity_curve(
    request: Request,
    subscriber: dict = Depends(require_api_key),
    days: int = Query(90, ge=1, le=365),
):
    """Return the equity curve (cumulative PnL over time) for closed signals."""
    tracker = request.app.state.app_state.signal_tracker
    if tracker is None:
        return EquityCurveResponse(points=[], current_cumulative_pnl=0.0)

    curve = tracker.get_equity_curve(days=days)
    points = [
        EquityCurvePoint(
            signal_id=p["signal_id"],
            closed_at=p["closed_at"],
            pnl_pips=p["pnl_pips"],
            cumulative_pnl=p["cumulative_pnl"],
        )
        for p in curve
    ]
    current = points[-1].cumulative_pnl if points else 0.0
    return EquityCurveResponse(points=points, current_cumulative_pnl=current)
