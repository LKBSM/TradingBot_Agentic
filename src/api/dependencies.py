"""Dependency-injection container for the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.api.signal_store import SignalStore


@dataclass
class AppState:
    """Holds all subsystem references injected into the FastAPI app."""
    signal_store: SignalStore
    metrics_registry: Any = None          # MetricsRegistry
    health_monitor: Any = None            # HealthMonitor
    kill_switch: Optional[Any] = None     # KillSwitch (legacy, src/agents/kill_switch.py)
    # New operational kill-switch (src/risk/kill_switch.py) used by the
    # SentinelScanner. Surfaced via /health and /admin/operational-resume.
    # Kept alongside the legacy one to avoid breaking operator.py callers.
    operational_kill_switch: Optional[Any] = None
    var_engine: Optional[Any] = None      # VaREngine
    live_risk_manager: Optional[Any] = None  # LiveRiskManager
    key_store: Optional[Any] = None       # KeyStore
    hmac_manager: Optional[Any] = None    # HMACKeyManager
    signal_tracker: Optional[Any] = None  # SignalTracker
    # Smart Sentinel AI
    tier_manager: Optional[Any] = None    # UserTierManager
    llm_engine: Optional[Any] = None      # LLMNarrativeEngine
    scanner: Optional[Any] = None         # SentinelScanner
    vol_forecaster: Optional[Any] = None  # VolatilityForecaster
    # Reliability
    circuit_breakers: Dict[str, Any] = field(default_factory=dict)  # {name: CircuitBreaker}
    health_checker: Optional[Any] = None  # HealthChecker
    rate_limiter: Optional[Any] = None    # RateLimiter
    # RAG (LLM-2B.5)
    rag_pipeline: Optional[Any] = None    # src.intelligence.rag.RAGPipeline
    rag_llm: Optional[Any] = None         # callable(system, user) -> str
    # Audit ledger (DATA-2B.4) — optional. /enrich appends each delivered
    # InsightSignalV2 to the chain when this is wired in.
    audit_ledger: Optional[Any] = None    # src.audit.HashChainLedger
    # Phase 2B production guards — surfaced in the /health/deep probe
    # (OBS-2B.2) so ops can see quota/limit/queue state at a glance.
    cost_quota: Optional[Any] = None        # src.intelligence.rag.cost_quota.CostQuotaEnforcer
    tier_rate_limiter: Optional[Any] = None # src.intelligence.rag.tier_rate_limiter.TierRateLimiter
    webhook_queue: Optional[Any] = None     # src.delivery.webhook_queue.WebhookDeliveryQueue
    embedder: Optional[Any] = None          # src.intelligence.rag.embedders.Embedder (DATA-2B.7)
    idempotency_store: Optional[Any] = None # src.api.idempotency_store.IdempotencyStore (API-2B.2)
    admin_action_log: Optional[Any] = None  # src.audit.AdminActionLog (SECURITY-2B.1)
    latency_tracker: Optional[Any] = None   # src.api.latency_tracker.LatencyTracker (OBS-2B.4)
    shutdown_coordinator: Optional[Any] = None  # src.api.shutdown.GracefulShutdownCoordinator (INFRA-2B.11)
    error_budget_watcher: Optional[Any] = None   # src.api.error_budget_watcher.ErrorBudgetWatcher (OBS-2B.5)
    webhook_drain_worker: Optional[Any] = None   # src.delivery.webhook_drain_worker.WebhookDrainWorker (OBS-2B.6)
