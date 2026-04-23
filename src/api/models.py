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


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    name: str
    healthy: bool


class HealthResponse(BaseModel):
    """System health — client-facing."""
    status: SystemStatus
    uptime_seconds: float
    kill_switch_level: int = 0
    is_trading_active: bool = True
    testing_mode: bool = False
    components: List[ComponentHealth] = Field(default_factory=list)
    scanner_running: bool = False
    signals_generated: int = 0


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


# =============================================================================
# NARRATIVE MODELS (Smart Sentinel AI)
# =============================================================================

class NarrativeResponse(BaseModel):
    """Tier-gated narrative for a signal."""
    signal_id: str
    symbol: str
    action: str
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confluence_score: Optional[float] = None
    market_context: Optional[str] = None
    validation_reason: Optional[str] = None       # ANALYST+
    full_narrative: Optional[str] = None           # STRATEGIST+
    key_confluences: Optional[str] = None          # STRATEGIST+
    risk_warnings: Optional[str] = None            # STRATEGIST+
    # Volatility forecast fields (Sprint 2)
    vol_forecast_atr: Optional[float] = None
    vol_regime: Optional[str] = None
    vol_confidence_lower: Optional[float] = None
    vol_confidence_upper: Optional[float] = None


class ChatRequest(BaseModel):
    """Request body for signal chat (Institutional only)."""
    signal_id: str
    question: str = Field(..., min_length=5, max_length=1000)


class ChatResponse(BaseModel):
    """Response from signal chat."""
    signal_id: str
    question: str
    answer: str
    cost_usd: float = 0.0


class ScannerStatusResponse(BaseModel):
    """Scanner health and stats."""
    running: bool = False
    uptime_seconds: float = 0.0
    bars_scanned: int = 0
    signals_generated: int = 0
    cache_hits: int = 0
    llm_calls: int = 0
    errors: int = 0
    last_bar_ts: Optional[str] = None


# =============================================================================
# PUBLIC STATE RESPONSE — what the dashboard polls
# =============================================================================

class HoldSubState(str, Enum):
    """Granular HOLD reason so the client UI can tell why we're standing aside."""
    IDLE = "idle"              # no pending, no recent exit — market inconclusive
    ARMING = "arming"           # building confirmation bars for a direction
    COOLDOWN = "cooldown"       # recently exited, forced-wait window
    POST_EXIT = "post_exit"     # cooldown=0 config; just-exited, ready next bar


class ActiveSignalPayload(BaseModel):
    """Lean signal view for the dashboard during BUY/SELL state."""
    signal_id: str
    symbol: str
    direction: str               # "LONG" | "SHORT"
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    confluence_score: float
    atr: Optional[float] = None
    vol_regime: Optional[str] = None
    vol_forecast_atr: Optional[float] = None


class PublicStateResponse(BaseModel):
    """Live state snapshot consumed by the client dashboard.

    HOLD carries as much information as BUY/SELL — the client always knows
    *why* no trade is recommended, not just that none is. This is the
    transparency contract that justifies the price anchor.
    """
    symbol: str
    state: str                              # "HOLD" | "BUY" | "SELL"
    headline: str                           # short banner text
    detail: str                             # one-sentence explanation
    hold_sub_state: Optional[HoldSubState] = None
    direction: Optional[str] = None         # "LONG" | "SHORT" when active
    active_signal: Optional[ActiveSignalPayload] = None
    bars_in_state: int = 0
    bars_remaining: Optional[int] = None
    cooldown_bars_remaining: Optional[int] = None
    confirmation_progress: Optional[List[int]] = None   # [have, need]
    last_exit_reason: Optional[str] = None
    last_bar_processed: Optional[str] = None
    last_bar_high: Optional[float] = None
    last_bar_low: Optional[float] = None
    last_bar_close: Optional[float] = None
    stats: Optional[Dict[str, Any]] = None
    generated_at: str                       # ISO timestamp
