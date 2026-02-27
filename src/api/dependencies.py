"""Dependency-injection container for the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

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
