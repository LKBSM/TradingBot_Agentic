"""Pydantic v2 response schemas for the Signal Delivery API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class SignalAction(str, Enum):
    """Trading signal action."""
    HOLD = "HOLD"
    OPEN_LONG = "OPEN_LONG"
    CLOSE_LONG = "CLOSE_LONG"
    OPEN_SHORT = "OPEN_SHORT"
    CLOSE_SHORT = "CLOSE_SHORT"


class SystemStatus(str, Enum):
    """Overall system health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


# =============================================================================
# CLIENT MODELS (lean — no internal risk metrics)
# =============================================================================

class SignalResponse(BaseModel):
    """Current trading signal — client-facing, lean payload."""
    signal_id: str
    action: SignalAction
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float = Field(description="Reward-to-risk ratio")
    created_at: datetime

    model_config = {"from_attributes": True}


class SignalHistoryItem(BaseModel):
    """Single historical signal."""
    signal_id: str
    action: SignalAction
    symbol: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    created_at: datetime
    outcome: Optional[str] = None
    pnl_pips: Optional[float] = None
    closed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class SignalHistoryResponse(BaseModel):
    """Paginated signal history."""
    signals: List[SignalHistoryItem]
    page: int
    page_size: int
    total: int


class HealthResponse(BaseModel):
    """System health — client-facing."""
    status: SystemStatus
    uptime_seconds: float
    kill_switch_level: int = 0
    is_trading_active: bool = True


# =============================================================================
# OPERATOR MODELS (rich — internal metrics)
# =============================================================================

class OperatorMetricsResponse(BaseModel):
    """Full Prometheus metrics registry dump for operators."""
    metrics: Dict[str, Any]


class OperatorRiskResponse(BaseModel):
    """Risk subsystem snapshot for operators."""
    var_95: Optional[float] = None
    var_99: Optional[float] = None
    cvar_95: Optional[float] = None
    current_drawdown_pct: Optional[float] = None
    daily_pnl: Optional[float] = None
    correlation_regime: Optional[str] = None


class OperatorKillSwitchResponse(BaseModel):
    """Full kill switch state for operators."""
    kill_switch: Dict[str, Any]


# =============================================================================
# ERROR MODEL
# =============================================================================

class ErrorResponse(BaseModel):
    """Structured error response."""
    error: str
    detail: Optional[str] = None


# =============================================================================
# AUTH MODELS
# =============================================================================

class KeyCreateRequest(BaseModel):
    """Request body for creating a new API key."""
    label: str = Field(..., min_length=1, max_length=128)


class KeyCreateResponse(BaseModel):
    """Response after creating a key — raw key shown once."""
    key_id: int
    api_key: str
    label: str


class KeyRevokeResponse(BaseModel):
    """Response after revoking a key."""
    key_id: int
    revoked: bool


class KeyInfo(BaseModel):
    """Single key metadata (no hash exposed)."""
    key_id: int
    label: str
    created_at: str
    is_active: bool


class KeyListResponse(BaseModel):
    """List of all keys."""
    keys: List[KeyInfo]


class UsageStat(BaseModel):
    """Usage count for one endpoint."""
    endpoint: str
    count: int


class UsageResponse(BaseModel):
    """Usage stats for a key."""
    key_id: int
    days: int
    usage: List[UsageStat]


# =============================================================================
# DASHBOARD MODELS (Sprint 11)
# =============================================================================

class PerformanceSummaryResponse(BaseModel):
    """Signal performance summary over a period."""
    total_signals: int
    winning: int
    losing: int
    win_rate: float
    profit_factor: float
    avg_rr: float
    cumulative_pnl: float
    sharpe_30d: float
    max_drawdown_pct: float
    period_days: int


class EquityCurvePoint(BaseModel):
    """Single point on the equity curve."""
    signal_id: str
    closed_at: str
    pnl_pips: float
    cumulative_pnl: float


class EquityCurveResponse(BaseModel):
    """Equity curve for closed signals."""
    points: List[EquityCurvePoint]
    current_cumulative_pnl: float
