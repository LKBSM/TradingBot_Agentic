# =============================================================================
# ALERT MANAGER - Multi-Channel Alerting System
# =============================================================================
# Production-grade alerting for critical trading events.
#
# Channels:
#   - PagerDuty: On-call escalation
#   - Slack: Team notifications
#   - Email: Formal notifications
#   - SMS (Twilio): Critical alerts
#   - Webhook: Custom integrations
#
# Usage:
#   manager = AlertManager(config)
#   manager.critical("Kill Switch Activated", details={...})
#
# =============================================================================

import os
import json
import logging
import smtplib
import asyncio
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import hashlib

# Optional imports
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False
    TwilioClient = None


# =============================================================================
# ENUMS & CONFIGURATION
# =============================================================================

class AlertSeverity(IntEnum):
    """Alert severity levels (maps to PagerDuty severity)."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


class AlertChannel(Enum):
    """Available alert channels."""
    PAGERDUTY = "pagerduty"
    SLACK = "slack"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    CONSOLE = "console"  # For development


@dataclass
class AlertConfig:
    """Configuration for AlertManager."""
    # Enabled channels
    enabled_channels: List[AlertChannel] = field(
        default_factory=lambda: [AlertChannel.CONSOLE]
    )

    # PagerDuty
    pagerduty_routing_key: str = ""
    pagerduty_api_url: str = "https://events.pagerduty.com/v2/enqueue"

    # Slack
    slack_webhook_url: str = ""
    slack_channel: str = "#trading-alerts"

    # Email
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: List[str] = field(default_factory=list)

    # SMS (Twilio)
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    sms_to_numbers: List[str] = field(default_factory=list)

    # Custom webhook
    webhook_url: str = ""
    webhook_headers: Dict[str, str] = field(default_factory=dict)

    # Behavior
    rate_limit_seconds: int = 60  # Min seconds between same alerts
    async_delivery: bool = True
    max_queue_size: int = 1000
    retry_attempts: int = 3
    retry_delay_seconds: float = 5.0

    # Severity routing (which channels for which severity)
    severity_routing: Dict[AlertSeverity, List[AlertChannel]] = field(
        default_factory=lambda: {
            AlertSeverity.INFO: [AlertChannel.SLACK, AlertChannel.CONSOLE],
            AlertSeverity.WARNING: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.CONSOLE],
            AlertSeverity.ERROR: [AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.PAGERDUTY, AlertChannel.CONSOLE],
            AlertSeverity.CRITICAL: [AlertChannel.PAGERDUTY, AlertChannel.SMS, AlertChannel.SLACK, AlertChannel.EMAIL, AlertChannel.CONSOLE],
        }
    )

    @classmethod
    def from_environment(cls) -> 'AlertConfig':
        """Create config from environment variables."""
        channels = []

        # Auto-detect available channels
        if os.getenv('PAGERDUTY_ROUTING_KEY'):
            channels.append(AlertChannel.PAGERDUTY)
        if os.getenv('SLACK_WEBHOOK_URL'):
            channels.append(AlertChannel.SLACK)
        if os.getenv('SMTP_HOST'):
            channels.append(AlertChannel.EMAIL)
        if os.getenv('TWILIO_ACCOUNT_SID'):
            channels.append(AlertChannel.SMS)

        # Always include console for dev visibility
        channels.append(AlertChannel.CONSOLE)

        return cls(
            enabled_channels=channels,
            pagerduty_routing_key=os.getenv('PAGERDUTY_ROUTING_KEY', ''),
            slack_webhook_url=os.getenv('SLACK_WEBHOOK_URL', ''),
            smtp_host=os.getenv('SMTP_HOST', ''),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            smtp_username=os.getenv('SMTP_USERNAME', ''),
            smtp_password=os.getenv('SMTP_PASSWORD', ''),
            email_from=os.getenv('ALERT_EMAIL_FROM', ''),
            email_to=os.getenv('ALERT_EMAIL_TO', '').split(',') if os.getenv('ALERT_EMAIL_TO') else [],
            twilio_account_sid=os.getenv('TWILIO_ACCOUNT_SID', ''),
            twilio_auth_token=os.getenv('TWILIO_AUTH_TOKEN', ''),
            twilio_from_number=os.getenv('TWILIO_FROM_NUMBER', ''),
            sms_to_numbers=os.getenv('SMS_TO_NUMBERS', '').split(',') if os.getenv('SMS_TO_NUMBERS') else [],
        )


# =============================================================================
# ALERT DATA STRUCTURES
# =============================================================================

@dataclass
class Alert:
    """An alert to be sent."""
    severity: AlertSeverity
    title: str
    message: str
    source: str = "trading-bot"
    details: Dict[str, Any] = field(default_factory=dict)
    dedup_key: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    channels: Optional[List[AlertChannel]] = None

    def __post_init__(self):
        # Generate dedup key if not provided
        if not self.dedup_key:
            key_data = f"{self.severity}:{self.title}:{self.source}"
            self.dedup_key = hashlib.md5(key_data.encode()).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'severity': self.severity.name,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'details': self.details,
            'dedup_key': self.dedup_key,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class AlertResult:
    """Result of sending an alert."""
    alert: Alert
    channel: AlertChannel
    success: bool
    error_message: Optional[str] = None
    sent_at: datetime = field(default_factory=datetime.utcnow)
    response: Optional[Dict[str, Any]] = None


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """
    Production-grade multi-channel alerting system.

    Features:
    - Multiple channels (PagerDuty, Slack, Email, SMS)
    - Severity-based routing
    - Rate limiting to prevent alert storms
    - Async delivery with retry
    - Alert deduplication
    - Audit trail

    Example:
        config = AlertConfig.from_environment()
        manager = AlertManager(config)

        # Send critical alert
        manager.critical(
            "Kill Switch Activated",
            message="Emergency halt triggered due to max drawdown",
            details={
                'drawdown': -15.5,
                'threshold': -10.0,
                'positions_closed': 3
            }
        )

        # Send warning
        manager.warning(
            "High Volatility Detected",
            message="VIX above threshold",
            details={'vix': 35.2}
        )
    """

    def __init__(self, config: AlertConfig):
        self.config = config
        self._logger = logging.getLogger("security.alerts")
        self._lock = threading.RLock()

        # Rate limiting
        self._last_sent: Dict[str, datetime] = {}

        # Async delivery
        self._queue: queue.Queue = queue.Queue(maxsize=config.max_queue_size)
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="alert-")
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Stats
        self._stats = {
            'alerts_sent': 0,
            'alerts_failed': 0,
            'alerts_rate_limited': 0,
        }

        # Delivery history
        self._history: List[AlertResult] = []
        self._max_history = 1000

        # Initialize Twilio client if available
        self._twilio_client = None
        if TWILIO_AVAILABLE and config.twilio_account_sid:
            try:
                self._twilio_client = TwilioClient(
                    config.twilio_account_sid,
                    config.twilio_auth_token
                )
            except Exception as e:
                self._logger.warning(f"Failed to initialize Twilio: {e}")

        # Start async worker if enabled
        if config.async_delivery:
            self._start_worker()

    # =========================================================================
    # PUBLIC API - Convenience Methods
    # =========================================================================

    def info(
        self,
        title: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        source: str = "trading-bot"
    ) -> List[AlertResult]:
        """Send an INFO level alert."""
        return self.send(Alert(
            severity=AlertSeverity.INFO,
            title=title,
            message=message,
            source=source,
            details=details or {},
        ))

    def warning(
        self,
        title: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        source: str = "trading-bot"
    ) -> List[AlertResult]:
        """Send a WARNING level alert."""
        return self.send(Alert(
            severity=AlertSeverity.WARNING,
            title=title,
            message=message,
            source=source,
            details=details or {},
        ))

    def error(
        self,
        title: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        source: str = "trading-bot"
    ) -> List[AlertResult]:
        """Send an ERROR level alert."""
        return self.send(Alert(
            severity=AlertSeverity.ERROR,
            title=title,
            message=message,
            source=source,
            details=details or {},
        ))

    def critical(
        self,
        title: str,
        message: str = "",
        details: Optional[Dict[str, Any]] = None,
        source: str = "trading-bot"
    ) -> List[AlertResult]:
        """Send a CRITICAL level alert."""
        return self.send(Alert(
            severity=AlertSeverity.CRITICAL,
            title=title,
            message=message,
            source=source,
            details=details or {},
        ))

    def send(self, alert: Alert) -> List[AlertResult]:
        """
        Send an alert through configured channels.

        Args:
            alert: Alert to send

        Returns:
            List of AlertResult for each channel
        """
        # Determine channels
        if alert.channels:
            channels = alert.channels
        else:
            channels = self.config.severity_routing.get(
                alert.severity,
                [AlertChannel.CONSOLE]
            )

        # Filter to enabled channels
        channels = [c for c in channels if c in self.config.enabled_channels]

        # Check rate limiting
        if self._is_rate_limited(alert):
            self._stats['alerts_rate_limited'] += 1
            self._logger.debug(f"Rate limited: {alert.title}")
            return []

        # Send to each channel
        results = []
        for channel in channels:
            if self.config.async_delivery:
                # Queue for async delivery
                try:
                    self._queue.put_nowait((alert, channel))
                    results.append(AlertResult(
                        alert=alert,
                        channel=channel,
                        success=True,  # Queued successfully
                    ))
                except queue.Full:
                    self._logger.error("Alert queue full, dropping alert")
                    results.append(AlertResult(
                        alert=alert,
                        channel=channel,
                        success=False,
                        error_message="Queue full",
                    ))
            else:
                # Synchronous delivery
                result = self._deliver_to_channel(alert, channel)
                results.append(result)
                with self._lock:
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]

        # Update rate limit tracking
        self._last_sent[alert.dedup_key] = datetime.utcnow()

        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'enabled_channels': [c.value for c in self.config.enabled_channels],
            'history_size': len(self._history),
        }

    def get_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent alert history."""
        with self._lock:
            history = self._history[-limit:]
            return [
                {
                    'alert': r.alert.to_dict(),
                    'channel': r.channel.value,
                    'success': r.success,
                    'error': r.error_message,
                    'sent_at': r.sent_at.isoformat(),
                }
                for r in history
            ]

    def shutdown(self) -> None:
        """Shutdown alert manager gracefully."""
        self._running = False
        if self._worker_thread:
            self._queue.put(None)  # Signal worker to stop
            self._worker_thread.join(timeout=10)
        self._executor.shutdown(wait=True)
        self._logger.info("AlertManager shutdown complete")

    # =========================================================================
    # CHANNEL DELIVERY
    # =========================================================================

    def _deliver_to_channel(
        self,
        alert: Alert,
        channel: AlertChannel
    ) -> AlertResult:
        """Deliver alert to specific channel."""
        try:
            if channel == AlertChannel.PAGERDUTY:
                return self._send_pagerduty(alert)
            elif channel == AlertChannel.SLACK:
                return self._send_slack(alert)
            elif channel == AlertChannel.EMAIL:
                return self._send_email(alert)
            elif channel == AlertChannel.SMS:
                return self._send_sms(alert)
            elif channel == AlertChannel.WEBHOOK:
                return self._send_webhook(alert)
            elif channel == AlertChannel.CONSOLE:
                return self._send_console(alert)
            else:
                return AlertResult(
                    alert=alert,
                    channel=channel,
                    success=False,
                    error_message=f"Unknown channel: {channel}",
                )

        except Exception as e:
            self._logger.error(f"Failed to send to {channel.value}: {e}")
            return AlertResult(
                alert=alert,
                channel=channel,
                success=False,
                error_message=str(e),
            )

    def _send_pagerduty(self, alert: Alert) -> AlertResult:
        """Send alert to PagerDuty."""
        if not REQUESTS_AVAILABLE:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.PAGERDUTY,
                success=False,
                error_message="requests package not available",
            )

        if not self.config.pagerduty_routing_key:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.PAGERDUTY,
                success=False,
                error_message="PagerDuty routing key not configured",
            )

        # Map severity
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        payload = {
            "routing_key": self.config.pagerduty_routing_key,
            "event_action": "trigger",
            "dedup_key": alert.dedup_key,
            "payload": {
                "summary": f"[{alert.source}] {alert.title}",
                "source": alert.source,
                "severity": severity_map.get(alert.severity, "error"),
                "timestamp": alert.created_at.isoformat(),
                "custom_details": {
                    "message": alert.message,
                    **alert.details,
                },
            },
        }

        response = requests.post(
            self.config.pagerduty_api_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        success = response.status_code == 202

        if success:
            self._stats['alerts_sent'] += 1
        else:
            self._stats['alerts_failed'] += 1

        return AlertResult(
            alert=alert,
            channel=AlertChannel.PAGERDUTY,
            success=success,
            error_message=None if success else response.text,
            response=response.json() if success else None,
        )

    def _send_slack(self, alert: Alert) -> AlertResult:
        """Send alert to Slack."""
        if not REQUESTS_AVAILABLE:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.SLACK,
                success=False,
                error_message="requests package not available",
            )

        if not self.config.slack_webhook_url:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.SLACK,
                success=False,
                error_message="Slack webhook URL not configured",
            )

        # Emoji based on severity
        emoji_map = {
            AlertSeverity.INFO: ":information_source:",
            AlertSeverity.WARNING: ":warning:",
            AlertSeverity.ERROR: ":x:",
            AlertSeverity.CRITICAL: ":rotating_light:",
        }

        # Color based on severity
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffcc00",
            AlertSeverity.ERROR: "#ff6600",
            AlertSeverity.CRITICAL: "#ff0000",
        }

        # Build message blocks
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{emoji_map.get(alert.severity, '')} {alert.title}",
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": alert.message or "_No message_",
                }
            },
        ]

        # Add details if present
        if alert.details:
            detail_text = "\n".join(
                f"*{k}:* {v}" for k, v in alert.details.items()
            )
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": detail_text,
                }
            })

        # Add timestamp
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Source: {alert.source} | {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
                }
            ]
        })

        payload = {
            "channel": self.config.slack_channel,
            "username": "Trading Bot Alert",
            "icon_emoji": emoji_map.get(alert.severity, ":robot_face:"),
            "attachments": [
                {
                    "color": color_map.get(alert.severity, "#cccccc"),
                    "blocks": blocks,
                }
            ],
        }

        response = requests.post(
            self.config.slack_webhook_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        success = response.status_code == 200

        if success:
            self._stats['alerts_sent'] += 1
        else:
            self._stats['alerts_failed'] += 1

        return AlertResult(
            alert=alert,
            channel=AlertChannel.SLACK,
            success=success,
            error_message=None if success else response.text,
        )

    def _send_email(self, alert: Alert) -> AlertResult:
        """Send alert via email."""
        if not self.config.smtp_host or not self.config.email_to:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.EMAIL,
                success=False,
                error_message="Email not configured",
            )

        try:
            # Build email
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.severity.name}] {alert.title}"
            msg['From'] = self.config.email_from
            msg['To'] = ", ".join(self.config.email_to)

            # Plain text version
            text_body = f"""
Trading Bot Alert

Severity: {alert.severity.name}
Title: {alert.title}
Source: {alert.source}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Message:
{alert.message}

Details:
{json.dumps(alert.details, indent=2)}
            """

            # HTML version
            details_html = "<br>".join(
                f"<strong>{k}:</strong> {v}"
                for k, v in alert.details.items()
            )

            html_body = f"""
<html>
<body>
<h2 style="color: {'red' if alert.severity >= AlertSeverity.ERROR else 'orange'}">
    [{alert.severity.name}] {alert.title}
</h2>
<p><strong>Source:</strong> {alert.source}</p>
<p><strong>Time:</strong> {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
<hr>
<p>{alert.message}</p>
<hr>
<h3>Details:</h3>
<p>{details_html}</p>
</body>
</html>
            """

            msg.attach(MIMEText(text_body, 'plain'))
            msg.attach(MIMEText(html_body, 'html'))

            # Send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                if self.config.smtp_username:
                    server.login(self.config.smtp_username, self.config.smtp_password)
                server.sendmail(
                    self.config.email_from,
                    self.config.email_to,
                    msg.as_string()
                )

            self._stats['alerts_sent'] += 1
            return AlertResult(
                alert=alert,
                channel=AlertChannel.EMAIL,
                success=True,
            )

        except Exception as e:
            self._stats['alerts_failed'] += 1
            return AlertResult(
                alert=alert,
                channel=AlertChannel.EMAIL,
                success=False,
                error_message=str(e),
            )

    def _send_sms(self, alert: Alert) -> AlertResult:
        """Send alert via SMS (Twilio)."""
        if not self._twilio_client:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.SMS,
                success=False,
                error_message="Twilio client not available",
            )

        if not self.config.sms_to_numbers:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.SMS,
                success=False,
                error_message="No SMS recipients configured",
            )

        # Build SMS message (160 char limit)
        body = f"[{alert.severity.name}] {alert.title}"
        if alert.message:
            remaining = 160 - len(body) - 3
            if remaining > 10:
                body += f" - {alert.message[:remaining]}"

        errors = []
        for number in self.config.sms_to_numbers:
            try:
                self._twilio_client.messages.create(
                    body=body,
                    from_=self.config.twilio_from_number,
                    to=number.strip(),
                )
            except Exception as e:
                errors.append(f"{number}: {e}")

        if errors:
            self._stats['alerts_failed'] += 1
            return AlertResult(
                alert=alert,
                channel=AlertChannel.SMS,
                success=False,
                error_message="; ".join(errors),
            )

        self._stats['alerts_sent'] += 1
        return AlertResult(
            alert=alert,
            channel=AlertChannel.SMS,
            success=True,
        )

    def _send_webhook(self, alert: Alert) -> AlertResult:
        """Send alert to custom webhook."""
        if not REQUESTS_AVAILABLE:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.WEBHOOK,
                success=False,
                error_message="requests package not available",
            )

        if not self.config.webhook_url:
            return AlertResult(
                alert=alert,
                channel=AlertChannel.WEBHOOK,
                success=False,
                error_message="Webhook URL not configured",
            )

        response = requests.post(
            self.config.webhook_url,
            json=alert.to_dict(),
            headers={
                "Content-Type": "application/json",
                **self.config.webhook_headers,
            },
            timeout=30,
        )

        success = 200 <= response.status_code < 300

        if success:
            self._stats['alerts_sent'] += 1
        else:
            self._stats['alerts_failed'] += 1

        return AlertResult(
            alert=alert,
            channel=AlertChannel.WEBHOOK,
            success=success,
            error_message=None if success else response.text,
        )

    def _send_console(self, alert: Alert) -> AlertResult:
        """Print alert to console (for development)."""
        severity_colors = {
            AlertSeverity.INFO: "\033[94m",     # Blue
            AlertSeverity.WARNING: "\033[93m",  # Yellow
            AlertSeverity.ERROR: "\033[91m",    # Red
            AlertSeverity.CRITICAL: "\033[95m", # Magenta
        }
        reset = "\033[0m"
        color = severity_colors.get(alert.severity, "")

        print(f"\n{color}{'='*60}")
        print(f"ALERT [{alert.severity.name}]: {alert.title}")
        print(f"{'='*60}{reset}")
        print(f"Source: {alert.source}")
        print(f"Time: {alert.created_at}")
        if alert.message:
            print(f"Message: {alert.message}")
        if alert.details:
            print(f"Details: {json.dumps(alert.details, indent=2)}")
        print(f"{color}{'='*60}{reset}\n")

        self._stats['alerts_sent'] += 1
        return AlertResult(
            alert=alert,
            channel=AlertChannel.CONSOLE,
            success=True,
        )

    # =========================================================================
    # ASYNC WORKER
    # =========================================================================

    def _start_worker(self) -> None:
        """Start async delivery worker."""
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="alert-worker"
        )
        self._worker_thread.start()
        self._logger.info("Alert worker started")

    def _worker_loop(self) -> None:
        """Worker loop for async delivery."""
        while self._running:
            try:
                item = self._queue.get(timeout=1)

                if item is None:  # Shutdown signal
                    break

                alert, channel = item
                result = self._deliver_to_channel(alert, channel)

                # Store in history
                with self._lock:
                    self._history.append(result)
                    if len(self._history) > self._max_history:
                        self._history = self._history[-self._max_history:]

            except queue.Empty:
                continue
            except Exception as e:
                self._logger.error(f"Worker error: {e}")

    # =========================================================================
    # RATE LIMITING
    # =========================================================================

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        if not alert.dedup_key:
            return False

        last_sent = self._last_sent.get(alert.dedup_key)
        if not last_sent:
            return False

        elapsed = (datetime.utcnow() - last_sent).total_seconds()
        return elapsed < self.config.rate_limit_seconds
