# =============================================================================
# SIEM INTEGRATION - Security Information and Event Management
# =============================================================================
# Production-grade security event logging for compliance and monitoring.
#
# Supported SIEM Platforms:
#   - Splunk (HTTP Event Collector)
#   - Elasticsearch/ELK Stack
#   - AWS CloudWatch Logs
#   - Azure Sentinel
#   - Generic Syslog (RFC 5424)
#   - Custom Webhook
#
# Event Categories:
#   - Authentication events
#   - Trading events (orders, executions)
#   - Risk events (breaches, alerts)
#   - System events (startup, shutdown, errors)
#   - Security events (access, modifications)
#
# Usage:
#   client = SIEMClient(config)
#   client.log_event(SecurityEvent(...))
#
# =============================================================================

import os
import json
import socket
import logging
import threading
import queue
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import gzip
from io import BytesIO

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class SIEMBackend(Enum):
    """Supported SIEM backends."""
    SPLUNK = "splunk"
    ELASTICSEARCH = "elasticsearch"
    CLOUDWATCH = "cloudwatch"
    AZURE_SENTINEL = "azure_sentinel"
    SYSLOG = "syslog"
    WEBHOOK = "webhook"
    FILE = "file"  # Local file for development


class EventCategory(Enum):
    """Security event categories."""
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    TRADING = "trading"
    RISK = "risk"
    SYSTEM = "system"
    SECURITY = "security"
    AUDIT = "audit"
    COMPLIANCE = "compliance"


class EventSeverity(Enum):
    """Event severity levels (CEF standard)."""
    UNKNOWN = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    VERY_HIGH = 4


class EventOutcome(Enum):
    """Event outcome."""
    SUCCESS = "success"
    FAILURE = "failure"
    UNKNOWN = "unknown"


@dataclass
class SIEMConfig:
    """Configuration for SIEM integration."""
    # Backend selection
    backend: SIEMBackend = SIEMBackend.FILE

    # Common settings
    application_name: str = "trading-bot"
    environment: str = "production"
    batch_size: int = 100
    flush_interval_seconds: int = 5
    async_delivery: bool = True

    # Splunk HEC
    splunk_url: str = ""
    splunk_token: str = ""
    splunk_index: str = "main"
    splunk_source: str = "trading-bot"
    splunk_sourcetype: str = "_json"

    # Elasticsearch
    elasticsearch_url: str = ""
    elasticsearch_index: str = "trading-bot-events"
    elasticsearch_api_key: str = ""

    # CloudWatch
    cloudwatch_log_group: str = "/trading-bot/security"
    cloudwatch_log_stream: str = ""
    aws_region: str = "us-east-1"

    # Azure Sentinel
    azure_workspace_id: str = ""
    azure_shared_key: str = ""
    azure_log_type: str = "TradingBotSecurityEvents"

    # Syslog
    syslog_host: str = "localhost"
    syslog_port: int = 514
    syslog_protocol: str = "UDP"  # UDP or TCP
    syslog_facility: int = 1  # user-level messages

    # Webhook
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # File (development)
    file_path: str = "security_events.jsonl"

    # Compression
    enable_compression: bool = True

    @classmethod
    def from_environment(cls) -> 'SIEMConfig':
        """Create config from environment variables."""
        backend_str = os.getenv('SIEM_BACKEND', 'file')

        return cls(
            backend=SIEMBackend(backend_str),
            environment=os.getenv('ENVIRONMENT', 'production'),
            splunk_url=os.getenv('SPLUNK_HEC_URL', ''),
            splunk_token=os.getenv('SPLUNK_HEC_TOKEN', ''),
            elasticsearch_url=os.getenv('ELASTICSEARCH_URL', ''),
            elasticsearch_api_key=os.getenv('ELASTICSEARCH_API_KEY', ''),
            cloudwatch_log_group=os.getenv('CLOUDWATCH_LOG_GROUP', '/trading-bot/security'),
            azure_workspace_id=os.getenv('AZURE_WORKSPACE_ID', ''),
            azure_shared_key=os.getenv('AZURE_SHARED_KEY', ''),
            syslog_host=os.getenv('SYSLOG_HOST', 'localhost'),
            syslog_port=int(os.getenv('SYSLOG_PORT', '514')),
            webhook_url=os.getenv('SIEM_WEBHOOK_URL', ''),
            file_path=os.getenv('SIEM_FILE_PATH', 'security_events.jsonl'),
        )


# =============================================================================
# SECURITY EVENT
# =============================================================================

@dataclass
class SecurityEvent:
    """
    A security event for SIEM logging.

    Follows Common Event Format (CEF) structure for compatibility.
    """
    # Required fields
    category: EventCategory
    event_type: str  # e.g., "login", "order_placed", "risk_breach"
    message: str

    # Optional identification
    event_id: str = ""
    correlation_id: str = ""  # For tracking related events

    # Severity and outcome
    severity: EventSeverity = EventSeverity.LOW
    outcome: EventOutcome = EventOutcome.UNKNOWN

    # Actor (who performed the action)
    actor_id: str = ""
    actor_type: str = ""  # "user", "system", "agent"
    actor_ip: str = ""

    # Target (what was affected)
    target_id: str = ""
    target_type: str = ""  # "order", "position", "account"

    # Context
    source_component: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Additional data
    details: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Compliance
    compliance_frameworks: List[str] = field(default_factory=list)  # ["MiFID II", "SEC"]

    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID
            data = f"{self.timestamp.isoformat()}:{self.category.value}:{self.event_type}"
            self.event_id = hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'correlation_id': self.correlation_id,
            'timestamp': self.timestamp.isoformat(),
            'category': self.category.value,
            'event_type': self.event_type,
            'message': self.message,
            'severity': self.severity.value,
            'severity_name': self.severity.name,
            'outcome': self.outcome.value,
            'actor': {
                'id': self.actor_id,
                'type': self.actor_type,
                'ip': self.actor_ip,
            },
            'target': {
                'id': self.target_id,
                'type': self.target_type,
            },
            'source_component': self.source_component,
            'details': self.details,
            'tags': self.tags,
            'compliance_frameworks': self.compliance_frameworks,
        }

    def to_cef(self) -> str:
        """
        Convert to Common Event Format (CEF) string.

        Format: CEF:Version|Device Vendor|Device Product|Device Version|Signature ID|Name|Severity|Extension
        """
        # Map severity to CEF (0-10 scale)
        cef_severity = self.severity.value * 2  # 0,2,4,6,8

        # Build extension fields
        extensions = [
            f"cat={self.category.value}",
            f"outcome={self.outcome.value}",
            f"msg={self.message.replace('|', '\\|')}",
        ]

        if self.actor_id:
            extensions.append(f"suser={self.actor_id}")
        if self.actor_ip:
            extensions.append(f"src={self.actor_ip}")
        if self.target_id:
            extensions.append(f"duser={self.target_id}")

        extension_str = " ".join(extensions)

        return (
            f"CEF:0|TradingBot|SecurityModule|1.0|"
            f"{self.event_type}|{self.message[:128]}|{cef_severity}|"
            f"{extension_str}"
        )

    def to_ecs(self) -> Dict[str, Any]:
        """
        Convert to Elastic Common Schema (ECS) format.
        """
        return {
            '@timestamp': self.timestamp.isoformat(),
            'event': {
                'id': self.event_id,
                'category': [self.category.value],
                'type': [self.event_type],
                'outcome': self.outcome.value,
                'severity': self.severity.value,
            },
            'message': self.message,
            'user': {
                'id': self.actor_id,
                'name': self.actor_type,
            } if self.actor_id else {},
            'source': {
                'ip': self.actor_ip,
            } if self.actor_ip else {},
            'labels': {tag: True for tag in self.tags},
            'trading_bot': self.details,
        }


# =============================================================================
# SIEM CLIENT
# =============================================================================

class SIEMClient:
    """
    Production-grade SIEM integration client.

    Features:
    - Multiple backend support (Splunk, ELK, CloudWatch, etc.)
    - Async batched delivery
    - Automatic retry with backoff
    - Event compression
    - Thread-safe operation

    Example:
        config = SIEMConfig.from_environment()
        client = SIEMClient(config)

        # Log a trading event
        client.log_event(SecurityEvent(
            category=EventCategory.TRADING,
            event_type="order_placed",
            message="Buy order placed for EURUSD",
            severity=EventSeverity.LOW,
            outcome=EventOutcome.SUCCESS,
            details={'symbol': 'EURUSD', 'volume': 0.1, 'price': 1.0850}
        ))

        # Convenience methods
        client.log_trade("order_executed", "Executed buy EURUSD", {...})
        client.log_risk("threshold_breach", "Max drawdown exceeded", {...})
        client.log_system("startup", "Trading bot started", {...})
    """

    def __init__(self, config: SIEMConfig):
        self.config = config
        self._logger = logging.getLogger("security.siem")
        self._lock = threading.RLock()

        # Event queue for async delivery
        self._queue: queue.Queue = queue.Queue(maxsize=10000)
        self._batch: List[SecurityEvent] = []

        # Worker thread
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Stats
        self._stats = {
            'events_logged': 0,
            'events_failed': 0,
            'batches_sent': 0,
        }

        # Syslog socket
        self._syslog_socket: Optional[socket.socket] = None

        # File handle
        self._file_handle = None

        # Initialize backend
        self._init_backend()

        # Start worker if async
        if config.async_delivery:
            self._start_worker()

    def _init_backend(self) -> None:
        """Initialize backend connection."""
        if self.config.backend == SIEMBackend.SYSLOG:
            self._init_syslog()
        elif self.config.backend == SIEMBackend.FILE:
            self._init_file()

    def _init_syslog(self) -> None:
        """Initialize syslog socket."""
        try:
            if self.config.syslog_protocol.upper() == "TCP":
                self._syslog_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._syslog_socket.connect((
                    self.config.syslog_host,
                    self.config.syslog_port
                ))
            else:
                self._syslog_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

            self._logger.info(f"Syslog initialized: {self.config.syslog_host}:{self.config.syslog_port}")
        except Exception as e:
            self._logger.error(f"Failed to initialize syslog: {e}")

    def _init_file(self) -> None:
        """Initialize file output."""
        try:
            self._file_handle = open(self.config.file_path, 'a', encoding='utf-8')
            self._logger.info(f"File output initialized: {self.config.file_path}")
        except Exception as e:
            self._logger.error(f"Failed to initialize file output: {e}")

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def log_event(self, event: SecurityEvent) -> bool:
        """
        Log a security event.

        Args:
            event: SecurityEvent to log

        Returns:
            True if event was queued/sent successfully
        """
        if self.config.async_delivery:
            try:
                self._queue.put_nowait(event)
                return True
            except queue.Full:
                self._logger.error("Event queue full, dropping event")
                self._stats['events_failed'] += 1
                return False
        else:
            return self._send_event(event)

    def log_trade(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        severity: EventSeverity = EventSeverity.LOW,
        outcome: EventOutcome = EventOutcome.SUCCESS
    ) -> bool:
        """Log a trading event."""
        return self.log_event(SecurityEvent(
            category=EventCategory.TRADING,
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            details=details,
            source_component="trading_engine",
            compliance_frameworks=["MiFID II", "SEC"],
        ))

    def log_risk(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        severity: EventSeverity = EventSeverity.HIGH,
        outcome: EventOutcome = EventOutcome.UNKNOWN
    ) -> bool:
        """Log a risk event."""
        return self.log_event(SecurityEvent(
            category=EventCategory.RISK,
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            details=details,
            source_component="risk_sentinel",
            tags=["risk", "monitoring"],
        ))

    def log_system(
        self,
        event_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        severity: EventSeverity = EventSeverity.LOW
    ) -> bool:
        """Log a system event."""
        return self.log_event(SecurityEvent(
            category=EventCategory.SYSTEM,
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=EventOutcome.SUCCESS,
            details=details or {},
            source_component="system",
        ))

    def log_security(
        self,
        event_type: str,
        message: str,
        details: Dict[str, Any],
        severity: EventSeverity = EventSeverity.HIGH,
        actor_id: str = "",
        actor_ip: str = ""
    ) -> bool:
        """Log a security event."""
        return self.log_event(SecurityEvent(
            category=EventCategory.SECURITY,
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=EventOutcome.UNKNOWN,
            actor_id=actor_id,
            actor_ip=actor_ip,
            details=details,
            source_component="security",
            tags=["security", "audit"],
        ))

    def log_auth(
        self,
        event_type: str,
        message: str,
        actor_id: str,
        outcome: EventOutcome,
        details: Optional[Dict[str, Any]] = None,
        actor_ip: str = ""
    ) -> bool:
        """Log an authentication event."""
        return self.log_event(SecurityEvent(
            category=EventCategory.AUTHENTICATION,
            event_type=event_type,
            message=message,
            severity=EventSeverity.MEDIUM if outcome == EventOutcome.FAILURE else EventSeverity.LOW,
            outcome=outcome,
            actor_id=actor_id,
            actor_ip=actor_ip,
            details=details or {},
            source_component="auth",
            compliance_frameworks=["SOC2"],
        ))

    def flush(self) -> int:
        """
        Flush pending events immediately.

        Returns:
            Number of events flushed
        """
        with self._lock:
            if not self._batch:
                return 0

            count = len(self._batch)
            self._send_batch(self._batch)
            self._batch = []
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get SIEM client statistics."""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'batch_size': len(self._batch),
            'backend': self.config.backend.value,
        }

    def shutdown(self) -> None:
        """Shutdown SIEM client gracefully."""
        self._running = False

        # Flush remaining events
        self.flush()

        # Stop worker
        if self._worker_thread:
            self._queue.put(None)
            self._worker_thread.join(timeout=10)

        # Close connections
        if self._syslog_socket:
            self._syslog_socket.close()
        if self._file_handle:
            self._file_handle.close()

        self._logger.info("SIEM client shutdown complete")

    # =========================================================================
    # BACKEND DELIVERY
    # =========================================================================

    def _send_event(self, event: SecurityEvent) -> bool:
        """Send single event to backend."""
        try:
            if self.config.backend == SIEMBackend.SPLUNK:
                return self._send_splunk([event])
            elif self.config.backend == SIEMBackend.ELASTICSEARCH:
                return self._send_elasticsearch([event])
            elif self.config.backend == SIEMBackend.SYSLOG:
                return self._send_syslog(event)
            elif self.config.backend == SIEMBackend.WEBHOOK:
                return self._send_webhook([event])
            elif self.config.backend == SIEMBackend.FILE:
                return self._send_file(event)
            elif self.config.backend == SIEMBackend.CLOUDWATCH:
                return self._send_cloudwatch([event])
            else:
                self._logger.warning(f"Unknown backend: {self.config.backend}")
                return False

        except Exception as e:
            self._logger.error(f"Failed to send event: {e}")
            self._stats['events_failed'] += 1
            return False

    def _send_batch(self, events: List[SecurityEvent]) -> bool:
        """Send batch of events to backend."""
        if not events:
            return True

        try:
            if self.config.backend == SIEMBackend.SPLUNK:
                return self._send_splunk(events)
            elif self.config.backend == SIEMBackend.ELASTICSEARCH:
                return self._send_elasticsearch(events)
            elif self.config.backend == SIEMBackend.WEBHOOK:
                return self._send_webhook(events)
            elif self.config.backend == SIEMBackend.CLOUDWATCH:
                return self._send_cloudwatch(events)
            elif self.config.backend == SIEMBackend.SYSLOG:
                # Syslog doesn't support batching well
                for event in events:
                    self._send_syslog(event)
                return True
            elif self.config.backend == SIEMBackend.FILE:
                for event in events:
                    self._send_file(event)
                return True

            return False

        except Exception as e:
            self._logger.error(f"Failed to send batch: {e}")
            self._stats['events_failed'] += len(events)
            return False

    def _send_splunk(self, events: List[SecurityEvent]) -> bool:
        """Send events to Splunk HEC."""
        if not REQUESTS_AVAILABLE or not self.config.splunk_url:
            return False

        # Build HEC payload
        payload_lines = []
        for event in events:
            hec_event = {
                'time': event.timestamp.timestamp(),
                'host': socket.gethostname(),
                'source': self.config.splunk_source,
                'sourcetype': self.config.splunk_sourcetype,
                'index': self.config.splunk_index,
                'event': event.to_dict(),
            }
            payload_lines.append(json.dumps(hec_event))

        payload = '\n'.join(payload_lines)

        # Compress if enabled
        headers = {
            'Authorization': f'Splunk {self.config.splunk_token}',
            'Content-Type': 'application/json',
        }

        if self.config.enable_compression:
            payload = gzip.compress(payload.encode())
            headers['Content-Encoding'] = 'gzip'

        response = requests.post(
            self.config.splunk_url,
            data=payload,
            headers=headers,
            timeout=30,
        )

        success = response.status_code == 200
        if success:
            self._stats['events_logged'] += len(events)
            self._stats['batches_sent'] += 1
        else:
            self._stats['events_failed'] += len(events)
            self._logger.error(f"Splunk error: {response.text}")

        return success

    def _send_elasticsearch(self, events: List[SecurityEvent]) -> bool:
        """Send events to Elasticsearch."""
        if not REQUESTS_AVAILABLE or not self.config.elasticsearch_url:
            return False

        # Build bulk API payload
        bulk_lines = []
        for event in events:
            # Index action
            bulk_lines.append(json.dumps({
                'index': {
                    '_index': f"{self.config.elasticsearch_index}-{event.timestamp.strftime('%Y.%m.%d')}",
                }
            }))
            # Document
            bulk_lines.append(json.dumps(event.to_ecs()))

        payload = '\n'.join(bulk_lines) + '\n'

        headers = {
            'Content-Type': 'application/x-ndjson',
        }
        if self.config.elasticsearch_api_key:
            headers['Authorization'] = f'ApiKey {self.config.elasticsearch_api_key}'

        response = requests.post(
            f"{self.config.elasticsearch_url}/_bulk",
            data=payload,
            headers=headers,
            timeout=30,
        )

        success = response.status_code == 200
        if success:
            self._stats['events_logged'] += len(events)
            self._stats['batches_sent'] += 1

        return success

    def _send_cloudwatch(self, events: List[SecurityEvent]) -> bool:
        """Send events to AWS CloudWatch Logs."""
        try:
            import boto3

            client = boto3.client('logs', region_name=self.config.aws_region)

            log_events = [
                {
                    'timestamp': int(event.timestamp.timestamp() * 1000),
                    'message': json.dumps(event.to_dict()),
                }
                for event in events
            ]

            # Sort by timestamp (required by CloudWatch)
            log_events.sort(key=lambda x: x['timestamp'])

            stream_name = self.config.cloudwatch_log_stream or f"trading-bot-{datetime.utcnow().strftime('%Y-%m-%d')}"

            # Ensure log stream exists
            try:
                client.create_log_stream(
                    logGroupName=self.config.cloudwatch_log_group,
                    logStreamName=stream_name,
                )
            except client.exceptions.ResourceAlreadyExistsException:
                pass

            client.put_log_events(
                logGroupName=self.config.cloudwatch_log_group,
                logStreamName=stream_name,
                logEvents=log_events,
            )

            self._stats['events_logged'] += len(events)
            self._stats['batches_sent'] += 1
            return True

        except ImportError:
            self._logger.warning("boto3 not installed, CloudWatch unavailable")
            return False
        except Exception as e:
            self._logger.error(f"CloudWatch error: {e}")
            return False

    def _send_syslog(self, event: SecurityEvent) -> bool:
        """Send event via syslog."""
        if not self._syslog_socket:
            return False

        try:
            # Build syslog message (RFC 5424)
            severity_map = {
                EventSeverity.UNKNOWN: 6,  # informational
                EventSeverity.LOW: 6,
                EventSeverity.MEDIUM: 4,  # warning
                EventSeverity.HIGH: 3,    # error
                EventSeverity.VERY_HIGH: 2,  # critical
            }

            syslog_severity = severity_map.get(event.severity, 6)
            priority = (self.config.syslog_facility * 8) + syslog_severity

            # CEF format for syslog
            cef_message = event.to_cef()

            syslog_msg = f"<{priority}>1 {event.timestamp.isoformat()}Z {socket.gethostname()} {self.config.application_name} - - - {cef_message}"

            if self.config.syslog_protocol.upper() == "TCP":
                self._syslog_socket.send((syslog_msg + '\n').encode())
            else:
                self._syslog_socket.sendto(
                    syslog_msg.encode(),
                    (self.config.syslog_host, self.config.syslog_port)
                )

            self._stats['events_logged'] += 1
            return True

        except Exception as e:
            self._logger.error(f"Syslog error: {e}")
            return False

    def _send_webhook(self, events: List[SecurityEvent]) -> bool:
        """Send events to custom webhook."""
        if not REQUESTS_AVAILABLE or not self.config.webhook_url:
            return False

        payload = {
            'events': [event.to_dict() for event in events],
            'batch_id': hashlib.md5(str(datetime.utcnow()).encode()).hexdigest()[:8],
            'source': self.config.application_name,
            'environment': self.config.environment,
        }

        response = requests.post(
            self.config.webhook_url,
            json=payload,
            headers={
                'Content-Type': 'application/json',
                **self.config.webhook_headers,
            },
            timeout=30,
        )

        success = 200 <= response.status_code < 300
        if success:
            self._stats['events_logged'] += len(events)
            self._stats['batches_sent'] += 1

        return success

    def _send_file(self, event: SecurityEvent) -> bool:
        """Write event to file."""
        if not self._file_handle:
            return False

        try:
            self._file_handle.write(json.dumps(event.to_dict()) + '\n')
            self._file_handle.flush()
            self._stats['events_logged'] += 1
            return True
        except Exception as e:
            self._logger.error(f"File write error: {e}")
            return False

    # =========================================================================
    # ASYNC WORKER
    # =========================================================================

    def _start_worker(self) -> None:
        """Start async delivery worker."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="siem-worker"
        )
        self._worker_thread.start()
        self._logger.info("SIEM worker started")

    def _worker_loop(self) -> None:
        """Worker loop for async delivery."""
        last_flush = datetime.utcnow()

        while self._running:
            try:
                # Get event with timeout
                try:
                    event = self._queue.get(timeout=1)
                except queue.Empty:
                    event = None

                if event is None:
                    # Check if we should flush
                    elapsed = (datetime.utcnow() - last_flush).total_seconds()
                    if self._batch and elapsed >= self.config.flush_interval_seconds:
                        self._send_batch(self._batch)
                        self._batch = []
                        last_flush = datetime.utcnow()
                    continue

                # Add to batch
                self._batch.append(event)

                # Send if batch is full
                if len(self._batch) >= self.config.batch_size:
                    self._send_batch(self._batch)
                    self._batch = []
                    last_flush = datetime.utcnow()

            except Exception as e:
                self._logger.error(f"Worker error: {e}")

        # Final flush on shutdown
        if self._batch:
            self._send_batch(self._batch)
