"""Operator-only endpoints — full risk/metrics visibility."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from src.api.auth import require_api_key
from src.api.models import (
    OperatorKillSwitchResponse,
    OperatorMetricsResponse,
    OperatorRiskResponse,
)

router = APIRouter(prefix="/api/v1/operator", tags=["operator"])


@router.get("/metrics", response_model=OperatorMetricsResponse)
async def operator_metrics(request: Request, subscriber: dict = Depends(require_api_key)):
    """Full MetricsRegistry JSON dump."""
    registry = request.app.state.app_state.metrics_registry
    if registry is None:
        return OperatorMetricsResponse(metrics={})
    return OperatorMetricsResponse(metrics=registry.to_json())


@router.get("/risk", response_model=OperatorRiskResponse)
async def operator_risk(request: Request, subscriber: dict = Depends(require_api_key)):
    """VaR, drawdown, daily PnL, correlation regime."""
    app_state = request.app.state.app_state

    data: dict = {}

    # VaR engine
    var_engine = app_state.var_engine
    if var_engine is not None:
        try:
            result = var_engine.compute()
            data["var_95"] = result.var_95
            data["var_99"] = result.var_99
            data["cvar_95"] = result.cvar_95
        except Exception:
            pass

    # Kill switch tracking gives drawdown + daily PnL
    ks = app_state.kill_switch
    if ks is not None:
        try:
            status = ks.get_status()
            tracking = status.get("tracking", {})
            data["current_drawdown_pct"] = tracking.get("drawdown_pct")
            data["daily_pnl"] = tracking.get("daily_pnl")
        except Exception:
            pass

    # Correlation regime from metrics registry gauge
    registry = app_state.metrics_registry
    if registry is not None:
        try:
            regime_val = registry.gauge(
                "correlation_regime"
            ).get()
            regime_map = {0: "STABLE", 1: "ELEVATED", 2: "BREAKDOWN", 3: "DECORRELATED"}
            data["correlation_regime"] = regime_map.get(int(regime_val), "UNKNOWN")
        except Exception:
            pass

    return OperatorRiskResponse(**data)


@router.get("/kill-switch", response_model=OperatorKillSwitchResponse)
async def operator_kill_switch(request: Request, subscriber: dict = Depends(require_api_key)):
    """Full kill switch state."""
    ks = request.app.state.app_state.kill_switch
    if ks is None:
        return OperatorKillSwitchResponse(kill_switch={})
    try:
        return OperatorKillSwitchResponse(kill_switch=ks.get_status())
    except Exception:
        return OperatorKillSwitchResponse(kill_switch={})
