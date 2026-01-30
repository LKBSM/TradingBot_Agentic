# =============================================================================
# SECURITY ORCHESTRATOR - Unified Security Infrastructure
# =============================================================================
# Integrates all Sprint 1 security components into a single cohesive system.
#
# Components Managed:
#   - SecretManager: Credential management
#   - HMACKeyManager: Audit log integrity
#   - AlertManager: Multi-channel alerting
#   - DeadManSwitch: Crash detection
#   - SIEMClient: Security event logging
#
# Usage:
#   security = SecurityOrchestrator.from_environment()
#   security.start()
#
#   # In trading loop
#   security.heartbeat(positions_count=5)
#
#   # On critical event
#   security.alert_critical("Kill Switch Activated", details={...})
#
#   # On shutdown
#   security.shutdown()
#
# =============================================================================

import os
import logging
import atexit
from datetime import datetime
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass

from .secrets_manager import SecretManager, SecretManagerConfig
from .hmac_manager import HMACKeyManager, HMACKeyConfig
from .alert_manager import AlertManager, AlertConfig, AlertSeverity
from .dead_man_switch import DeadManSwitch, DeadManSwitchConfig
from .siem_integration import SIEMClient, SIEMConfig, SecurityEvent, EventCategory, EventSeverity, EventOutcome


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class SecurityOrchestratorConfig:
    """Configuration for the security orchestrator."""
    # Component configs
    secrets_config: SecretManagerConfig = None
    hmac_config: HMACKeyConfig = None
    alert_config: AlertConfig = None
    deadman_config: DeadManSwitchConfig = None
    siem_config: SIEMConfig = None

    # Feature flags
    enable_secrets_manager: bool = True
    enable_hmac_manager: bool = True
    enable_alert_manager: bool = True
    enable_dead_man_switch: bool = True
    enable_siem: bool = True

    # Behavior
    auto_start_deadman: bool = True
    register_atexit: bool = True

    @classmethod
    def from_environment(cls) -> 'SecurityOrchestratorConfig':
        """Create config from environment variables."""
        return cls(
            secrets_config=SecretManagerConfig.from_environment(),
            hmac_config=HMACKeyConfig.from_environment(),
            alert_config=AlertConfig.from_environment(),
            deadman_config=DeadManSwitchConfig.from_environment(),
            siem_config=SIEMConfig.from_environment(),
            enable_secrets_manager=os.getenv('ENABLE_SECRETS_MANAGER', 'true').lower() == 'true',
            enable_hmac_manager=os.getenv('ENABLE_HMAC_MANAGER', 'true').lower() == 'true',
            enable_alert_manager=os.getenv('ENABLE_ALERT_MANAGER', 'true').lower() == 'true',
            enable_dead_man_switch=os.getenv('ENABLE_DEAD_MAN_SWITCH', 'true').lower() == 'true',
            enable_siem=os.getenv('ENABLE_SIEM', 'true').lower() == 'true',
        )


# =============================================================================
# SECURITY ORCHESTRATOR
# =============================================================================

class SecurityOrchestrator:
    """
    Unified security infrastructure for the trading bot.

    This class integrates all Sprint 1 security components:
    - Secrets management (Vault/encrypted file)
    - HMAC key persistence for audit integrity
    - Multi-channel alerting (PagerDuty, Slack, SMS, Email)
    - Dead Man's Switch for crash detection
    - SIEM integration for security logging

    Architecture:
    ```
    ┌─────────────────────────────────────────────────────────────────┐
    │                   SECURITY ORCHESTRATOR                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                 │
    │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
    │   │ SecretManager│  │ HMACManager  │  │ AlertManager │         │
    │   │              │  │              │  │              │         │
    │   │ - Vault      │  │ - Persistent │  │ - PagerDuty  │         │
    │   │ - Encrypted  │  │   keys       │  │ - Slack      │         │
    │   │   files      │  │ - Rotation   │  │ - Email/SMS  │         │
    │   └──────────────┘  └──────────────┘  └──────────────┘         │
    │                                                                 │
    │   ┌──────────────┐  ┌──────────────┐                           │
    │   │ DeadManSwitch│  │  SIEMClient  │                           │
    │   │              │  │              │                           │
    │   │ - Heartbeat  │  │ - Splunk     │                           │
    │   │ - Crash      │  │ - ELK        │                           │
    │   │   detection  │  │ - CloudWatch │                           │
    │   └──────────────┘  └──────────────┘                           │
    │                                                                 │
    └─────────────────────────────────────────────────────────────────┘
    ```

    Example:
        # Initialize from environment
        security = SecurityOrchestrator.from_environment()
        security.start()

        # Get credentials
        mt5_creds = security.get_mt5_credentials()

        # In trading loop
        while trading:
            security.heartbeat(positions_count=len(positions))

            # Log trades
            security.log_trade("order_executed", {"symbol": "EURUSD", ...})

            # On risk event
            if risk_breach:
                security.alert_critical("Risk Breach", details={...})

        # Graceful shutdown
        security.shutdown()
    """

    _instance: Optional['SecurityOrchestrator'] = None

    def __init__(self, config: SecurityOrchestratorConfig):
        """
        Initialize the security orchestrator.

        Args:
            config: SecurityOrchestratorConfig
        """
        self.config = config
        self._logger = logging.getLogger("security.orchestrator")
        self._started = False

        # Initialize components
        self.secrets: Optional[SecretManager] = None
        self.hmac: Optional[HMACKeyManager] = None
        self.alerts: Optional[AlertManager] = None
        self.deadman: Optional[DeadManSwitch] = None
        self.siem: Optional[SIEMClient] = None

        self._init_components()

        # Register atexit handler for graceful shutdown
        if config.register_atexit:
            atexit.register(self.shutdown)

        # Set singleton instance
        SecurityOrchestrator._instance = self

        self._logger.info("SecurityOrchestrator initialized")

    @classmethod
    def from_environment(cls) -> 'SecurityOrchestrator':
        """Create orchestrator from environment variables."""
        config = SecurityOrchestratorConfig.from_environment()
        return cls(config)

    @classmethod
    def get_instance(cls) -> Optional['SecurityOrchestrator']:
        """Get the singleton instance."""
        return cls._instance

    # =========================================================================
    # INITIALIZATION
    # =========================================================================

    def _init_components(self) -> None:
        """Initialize all security components."""
        # 1. Secrets Manager (needed by others)
        if self.config.enable_secrets_manager and self.config.secrets_config:
            try:
                self.secrets = SecretManager(self.config.secrets_config)
                self._logger.info("SecretManager initialized")
            except Exception as e:
                self._logger.error(f"Failed to initialize SecretManager: {e}")

        # 2. HMAC Key Manager
        if self.config.enable_hmac_manager and self.config.hmac_config:
            try:
                self.hmac = HMACKeyManager(
                    self.config.hmac_config,
                    secret_manager=self.secrets
                )
                self._logger.info("HMACKeyManager initialized")
            except Exception as e:
                self._logger.error(f"Failed to initialize HMACKeyManager: {e}")

        # 3. Alert Manager
        if self.config.enable_alert_manager and self.config.alert_config:
            try:
                self.alerts = AlertManager(self.config.alert_config)
                self._logger.info("AlertManager initialized")
            except Exception as e:
                self._logger.error(f"Failed to initialize AlertManager: {e}")

        # 4. SIEM Client
        if self.config.enable_siem and self.config.siem_config:
            try:
                self.siem = SIEMClient(self.config.siem_config)
                self._logger.info("SIEMClient initialized")
            except Exception as e:
                self._logger.error(f"Failed to initialize SIEMClient: {e}")

        # 5. Dead Man's Switch (last, uses AlertManager)
        if self.config.enable_dead_man_switch and self.config.deadman_config:
            try:
                self.deadman = DeadManSwitch(
                    self.config.deadman_config,
                    alert_manager=self.alerts,
                    on_failure_callback=self._on_deadman_failure
                )
                self._logger.info("DeadManSwitch initialized")
            except Exception as e:
                self._logger.error(f"Failed to initialize DeadManSwitch: {e}")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    def start(self) -> None:
        """
        Start the security orchestrator.

        This starts the Dead Man's Switch heartbeat and logs startup.
        """
        if self._started:
            self._logger.warning("SecurityOrchestrator already started")
            return

        self._started = True

        # Start Dead Man's Switch
        if self.deadman and self.config.auto_start_deadman:
            self.deadman.start()
            self._logger.info("DeadManSwitch started")

        # Log startup event
        if self.siem:
            self.siem.log_system(
                "startup",
                "Security orchestrator started",
                details={
                    'components': {
                        'secrets': self.secrets is not None,
                        'hmac': self.hmac is not None,
                        'alerts': self.alerts is not None,
                        'deadman': self.deadman is not None,
                        'siem': self.siem is not None,
                    }
                }
            )

        # Send startup notification
        if self.alerts:
            self.alerts.info(
                "Trading Bot Started",
                message="Security orchestrator initialized successfully",
                source="security_orchestrator"
            )

        self._logger.info("SecurityOrchestrator started")

    def shutdown(self, graceful: bool = True) -> None:
        """
        Shutdown the security orchestrator.

        Args:
            graceful: If True, send shutdown notifications
        """
        if not self._started:
            return

        self._logger.info("SecurityOrchestrator shutting down...")

        # Log shutdown event
        if self.siem:
            self.siem.log_system(
                "shutdown",
                "Security orchestrator shutting down",
                details={'graceful': graceful}
            )
            self.siem.flush()
            self.siem.shutdown()

        # Stop Dead Man's Switch
        if self.deadman:
            self.deadman.stop(graceful=graceful)

        # Send shutdown notification
        if self.alerts and graceful:
            self.alerts.info(
                "Trading Bot Stopped",
                message="Graceful shutdown completed",
                source="security_orchestrator"
            )
            self.alerts.shutdown()

        self._started = False
        self._logger.info("SecurityOrchestrator shutdown complete")

    # =========================================================================
    # CREDENTIALS
    # =========================================================================

    def get_mt5_credentials(self) -> Dict[str, Any]:
        """
        Get MT5 trading credentials.

        Returns:
            Dict with 'account', 'password', 'server' keys
        """
        if self.secrets:
            return self.secrets.get_mt5_credentials()

        # Fallback to environment
        return {
            'account': int(os.getenv('MT5_LOGIN', '0')),
            'password': os.getenv('MT5_PASSWORD', ''),
            'server': os.getenv('MT5_SERVER', ''),
        }

    def get_api_key(self, service: str) -> str:
        """Get API key for a service."""
        if self.secrets:
            return self.secrets.get_api_key(service)
        return os.getenv(f'{service.upper()}_API_KEY', '')

    def get_secret(self, path: str) -> Dict[str, Any]:
        """Get a secret by path."""
        if self.secrets:
            return self.secrets.get_secret(path)
        raise RuntimeError("SecretManager not available")

    # =========================================================================
    # HMAC SIGNING
    # =========================================================================

    def sign_data(self, data: Dict[str, Any]) -> str:
        """
        Sign data with HMAC for integrity verification.

        Args:
            data: Dictionary to sign

        Returns:
            HMAC signature string
        """
        if self.hmac:
            return self.hmac.sign_dict(data)
        raise RuntimeError("HMACKeyManager not available")

    def verify_signature(
        self,
        data: Dict[str, Any],
        signature: str,
        key_version: Optional[int] = None
    ) -> bool:
        """
        Verify HMAC signature.

        Args:
            data: Original dictionary
            signature: Signature to verify
            key_version: Optional key version

        Returns:
            True if signature is valid
        """
        if self.hmac:
            return self.hmac.verify_dict(data, signature, key_version)
        raise RuntimeError("HMACKeyManager not available")

    def get_hmac_key_info(self) -> Dict[str, Any]:
        """Get information about HMAC keys."""
        if self.hmac:
            return self.hmac.get_key_info()
        return {}

    def rotate_hmac_key(self, reason: str = "Manual rotation") -> int:
        """Rotate HMAC key and return new version."""
        if self.hmac:
            version = self.hmac.rotate_key(reason)

            # Log rotation event
            if self.siem:
                self.siem.log_security(
                    "hmac_key_rotation",
                    f"HMAC key rotated: {reason}",
                    details={'new_version': version, 'reason': reason},
                    severity=EventSeverity.MEDIUM
                )

            return version
        raise RuntimeError("HMACKeyManager not available")

    # =========================================================================
    # ALERTING
    # =========================================================================

    def alert_info(self, title: str, message: str = "", details: Optional[Dict] = None):
        """Send INFO level alert."""
        if self.alerts:
            self.alerts.info(title, message, details)

    def alert_warning(self, title: str, message: str = "", details: Optional[Dict] = None):
        """Send WARNING level alert."""
        if self.alerts:
            self.alerts.warning(title, message, details)

        if self.siem:
            self.siem.log_security(
                "alert_warning",
                title,
                details=details or {},
                severity=EventSeverity.MEDIUM
            )

    def alert_error(self, title: str, message: str = "", details: Optional[Dict] = None):
        """Send ERROR level alert."""
        if self.alerts:
            self.alerts.error(title, message, details)

        if self.siem:
            self.siem.log_security(
                "alert_error",
                title,
                details=details or {},
                severity=EventSeverity.HIGH
            )

    def alert_critical(self, title: str, message: str = "", details: Optional[Dict] = None):
        """Send CRITICAL level alert."""
        if self.alerts:
            self.alerts.critical(title, message, details)

        if self.siem:
            self.siem.log_security(
                "alert_critical",
                title,
                details=details or {},
                severity=EventSeverity.VERY_HIGH
            )

    # =========================================================================
    # HEARTBEAT
    # =========================================================================

    def heartbeat(
        self,
        positions_count: int = 0,
        last_trade_at: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update heartbeat context.

        Call this regularly from your trading loop.

        Args:
            positions_count: Current number of open positions
            last_trade_at: Timestamp of last trade
            metadata: Additional context data
        """
        if self.deadman:
            self.deadman.heartbeat(
                positions_count=positions_count,
                last_trade_at=last_trade_at,
                metadata=metadata
            )

    def get_heartbeat_status(self) -> Dict[str, Any]:
        """Get Dead Man's Switch status."""
        if self.deadman:
            return self.deadman.get_status()
        return {'status': 'disabled'}

    # =========================================================================
    # SIEM LOGGING
    # =========================================================================

    def log_trade(self, event_type: str, details: Dict[str, Any], message: str = "") -> None:
        """Log a trading event to SIEM."""
        if self.siem:
            self.siem.log_trade(
                event_type,
                message or f"Trade event: {event_type}",
                details
            )

    def log_risk(self, event_type: str, details: Dict[str, Any], message: str = "") -> None:
        """Log a risk event to SIEM."""
        if self.siem:
            self.siem.log_risk(
                event_type,
                message or f"Risk event: {event_type}",
                details
            )

    def log_security_event(self, event: SecurityEvent) -> None:
        """Log a custom security event."""
        if self.siem:
            self.siem.log_event(event)

    # =========================================================================
    # STATUS & DIAGNOSTICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all security components."""
        return {
            'started': self._started,
            'timestamp': datetime.utcnow().isoformat(),
            'components': {
                'secrets_manager': {
                    'enabled': self.secrets is not None,
                    'status': 'ok' if self.secrets else 'disabled',
                },
                'hmac_manager': {
                    'enabled': self.hmac is not None,
                    'status': 'ok' if self.hmac else 'disabled',
                    'key_info': self.hmac.get_key_info() if self.hmac else None,
                },
                'alert_manager': {
                    'enabled': self.alerts is not None,
                    'status': 'ok' if self.alerts else 'disabled',
                    'stats': self.alerts.get_stats() if self.alerts else None,
                },
                'dead_man_switch': {
                    'enabled': self.deadman is not None,
                    'status': self.deadman.get_status() if self.deadman else {'status': 'disabled'},
                },
                'siem_client': {
                    'enabled': self.siem is not None,
                    'status': 'ok' if self.siem else 'disabled',
                    'stats': self.siem.get_stats() if self.siem else None,
                },
            }
        }

    def health_check(self) -> bool:
        """
        Perform health check on all components.

        Returns:
            True if all enabled components are healthy
        """
        issues = []

        # Check HMAC key rotation
        if self.hmac:
            needs_rotation, reason = self.hmac.check_rotation_needed()
            if needs_rotation:
                issues.append(f"HMAC key needs rotation: {reason}")

        # Check Dead Man's Switch
        if self.deadman:
            status = self.deadman.get_status()
            if status['status'] not in ['healthy', 'stopped']:
                issues.append(f"Dead Man's Switch status: {status['status']}")

        if issues:
            self._logger.warning(f"Health check issues: {issues}")
            return False

        return True

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _on_deadman_failure(self) -> None:
        """Called when Dead Man's Switch detects repeated failures."""
        self._logger.critical("Dead Man's Switch failure detected!")

        if self.siem:
            self.siem.log_security(
                "deadman_failure",
                "Dead Man's Switch reporting failures",
                details={'action': 'investigate_connectivity'},
                severity=EventSeverity.VERY_HIGH
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_security() -> Optional[SecurityOrchestrator]:
    """Get the global SecurityOrchestrator instance."""
    return SecurityOrchestrator.get_instance()


def init_security() -> SecurityOrchestrator:
    """Initialize and return SecurityOrchestrator from environment."""
    return SecurityOrchestrator.from_environment()
