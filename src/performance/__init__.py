# =============================================================================
# PERFORMANCE MODULE - Sprint 2: Performance, Observability & Latency
# =============================================================================

from .async_audit_logger import AsyncAuditLogger, AuditLogConfig
from .mt5_connection_pool import MT5ConnectionPool, MT5PoolConfig
from .vectorized_risk import VectorizedRiskCalculator

from .metrics import (
    Counter,
    Gauge,
    Histogram,
    MetricsRegistry,
    get_registry,
    create_trading_metrics,
    TRADE_PNL_BUCKETS,
    LATENCY_BUCKETS,
)
from .logging_config import (
    setup_structured_logging,
    get_trading_logger,
    LogContext,
    TimingContext,
)
from .health import (
    HealthCheck,
    HealthStatus,
    HealthResult,
    AggregateHealth,
    HealthMonitor,
    create_agent_health_check,
    create_kill_switch_health_check,
    create_memory_health_check,
)

__all__ = [
    # Existing
    'AsyncAuditLogger',
    'AuditLogConfig',
    'MT5ConnectionPool',
    'MT5PoolConfig',
    'VectorizedRiskCalculator',
    # Metrics
    'Counter',
    'Gauge',
    'Histogram',
    'MetricsRegistry',
    'get_registry',
    'create_trading_metrics',
    'TRADE_PNL_BUCKETS',
    'LATENCY_BUCKETS',
    # Logging
    'setup_structured_logging',
    'get_trading_logger',
    'LogContext',
    'TimingContext',
    # Health
    'HealthCheck',
    'HealthStatus',
    'HealthResult',
    'AggregateHealth',
    'HealthMonitor',
    'create_agent_health_check',
    'create_kill_switch_health_check',
    'create_memory_health_check',
]
