# =============================================================================
# SECURITY MODULE - Sprint 1: Critical Security Infrastructure
# =============================================================================
# Production-grade security components for institutional trading systems.
#
# Components:
#   - SecretManager: Vault integration with secure fallback
#   - HMACKeyManager: Persistent HMAC key storage
#   - AlertManager: Multi-channel alerting (PagerDuty, Slack, Email, SMS)
#   - DeadManSwitch: External heartbeat monitoring
#   - SIEMIntegration: Security event logging
#
# =============================================================================

from .secrets_manager import SecretManager, SecretManagerConfig
from .hmac_manager import HMACKeyManager, HMACKeyConfig
from .alert_manager import AlertManager, AlertConfig, AlertSeverity, AlertChannel
from .dead_man_switch import DeadManSwitch, DeadManSwitchConfig, HeartbeatStatus
from .siem_integration import (
    SIEMClient, SIEMConfig, SecurityEvent,
    EventCategory, EventSeverity, EventOutcome
)
from .security_orchestrator import (
    SecurityOrchestrator, SecurityOrchestratorConfig,
    get_security, init_security
)

__all__ = [
    # Secrets
    'SecretManager',
    'SecretManagerConfig',
    # HMAC
    'HMACKeyManager',
    'HMACKeyConfig',
    # Alerts
    'AlertManager',
    'AlertConfig',
    'AlertSeverity',
    'AlertChannel',
    # Dead Man Switch
    'DeadManSwitch',
    'DeadManSwitchConfig',
    'HeartbeatStatus',
    # SIEM
    'SIEMClient',
    'SIEMConfig',
    'SecurityEvent',
    'EventCategory',
    'EventSeverity',
    'EventOutcome',
    # Orchestrator
    'SecurityOrchestrator',
    'SecurityOrchestratorConfig',
    'get_security',
    'init_security',
]
