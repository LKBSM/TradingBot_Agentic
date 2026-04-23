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
    kill_switch: Optional[Any] = None     # KillSwitch
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
