# =============================================================================
# ALERTING SYSTEM - Multi-Channel Notifications
# =============================================================================
# Production alerting system for trading notifications.
#
# Channels supported:
#   - Telegram: Instant messaging via bot
#   - Discord: Webhook notifications
#   - Email: SMTP email alerts
#   - SMS: Via Twilio (optional)
#
# Alert levels:
#   - INFO: Trade executed, position update
#   - WARNING: Risk threshold approaching, slippage high
#   - CRITICAL: Max drawdown, connection lost, kill switch
#
# =============================================================================

import os
import json
import logging
import smtplib
import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from threading import Thread, Lock
from queue import Queue, Empty
import time

# Try to import optional dependencies
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import telegram
    from telegram import Bot
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    Bot = None


# =============================================================================
# ENUMS
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4

    def get_emoji(self) -> str:
        """Get emoji for alert level."""
        emojis = {
            AlertLevel.DEBUG: "🔍",
            AlertLevel.INFO: "ℹ️",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ERROR: "❌",
            AlertLevel.CRITICAL: "🚨",
        }
        return emojis.get(self, "📢")

    def get_color(self) -> int:
        """Get Discord color for alert level."""
        colors = {
            AlertLevel.DEBUG: 0x808080,    # Gray
            AlertLevel.INFO: 0x00FF00,     # Green
            AlertLevel.WARNING: 0xFFFF00,  # Yellow
            AlertLevel.ERROR: 0xFF0000,    # Red
            AlertLevel.CRITICAL: 0xFF0000, # Red
        }
        return colors.get(self, 0xFFFFFF)


class AlertChannel(Enum):
    """Available alert channels."""
    TELEGRAM = "telegram"
    DISCORD = "discord"
    EMAIL = "email"
    SMS = "sms"
    CONSOLE = "console"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Alert:
    """Alert message container."""
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    data: Dict[str, Any] = field(default_factory=dict)
    channels: List[AlertChannel] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'level': self.level.name,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'data': self.data,
        }

    def format_text(self) -> str:
        """Format as plain text."""
        emoji = self.level.get_emoji()
        lines = [
            f"{emoji} {self.level.name}: {self.title}",
            f"Time: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            self.message,
        ]

        if self.data:
            lines.append("")
            lines.append("Details:")
            for key, value in self.data.items():
                lines.append(f"  • {key}: {value}")

        return "\n".join(lines)

    def format_html(self) -> str:
        """Format as HTML."""
        emoji = self.level.get_emoji()
        html = f"""
        <div style="font-family: Arial, sans-serif; padding: 15px; border-radius: 8px;
                    background-color: {'#ffeeee' if self.level == AlertLevel.CRITICAL else '#f5f5f5'};">
            <h2 style="margin: 0 0 10px 0;">{emoji} {self.title}</h2>
            <p style="color: #666; margin: 0 0 10px 0;">
                {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {self.level.name}
            </p>
            <p style="margin: 0 0 10px 0;">{self.message}</p>
        """

        if self.data:
            html += "<ul style='margin: 0; padding-left: 20px;'>"
            for key, value in self.data.items():
                html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"

        html += "</div>"
        return html


@dataclass
class AlertConfig:
    """Configuration for alert channels."""
    # Telegram
    telegram_bot_token: Optional[str] = None
    telegram_chat_id: Optional[str] = None

    # Discord
    discord_webhook_url: Optional[str] = None

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: Optional[str] = None
    smtp_password: Optional[str] = None
    email_from: Optional[str] = None
    email_to: List[str] = field(default_factory=list)

    # SMS (Twilio)
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    twilio_from_number: Optional[str] = None
    sms_to_numbers: List[str] = field(default_factory=list)

    # Rate limiting
    min_alert_interval_seconds: int = 60  # Minimum time between same alerts
    max_alerts_per_hour: int = 30         # Rate limit

    # Filtering
    min_level: AlertLevel = AlertLevel.INFO  # Minimum level to send

    @classmethod
    def from_env(cls) -> 'AlertConfig':
        """Load configuration from environment variables."""
        return cls(
            telegram_bot_token=os.getenv('TELEGRAM_BOT_TOKEN'),
            telegram_chat_id=os.getenv('TELEGRAM_CHAT_ID'),
            discord_webhook_url=os.getenv('DISCORD_WEBHOOK_URL'),
            smtp_host=os.getenv('SMTP_HOST', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            smtp_user=os.getenv('SMTP_USER'),
            smtp_password=os.getenv('SMTP_PASSWORD'),
            email_from=os.getenv('ALERT_EMAIL_FROM'),
            email_to=os.getenv('ALERT_EMAIL_TO', '').split(',') if os.getenv('ALERT_EMAIL_TO') else [],
        )


# =============================================================================
# ALERT CHANNEL IMPLEMENTATIONS
# =============================================================================

class AlertChannelBase(ABC):
    """Base class for alert channels."""

    def __init__(self, config: AlertConfig):
        self.config = config
        self._logger = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def send(self, alert: Alert) -> bool:
        """Send alert through this channel."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if this channel is properly configured."""
        pass


class TelegramAlert(AlertChannelBase):
    """Telegram alert channel."""

    def __init__(self, config: AlertConfig):
        super().__init__(config)
        self._bot = None

    def is_configured(self) -> bool:
        return (
            self.config.telegram_bot_token is not None and
            self.config.telegram_chat_id is not None
        )

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            self._logger.warning("Telegram not configured")
            return False

        if not REQUESTS_AVAILABLE:
            self._logger.error("requests library not available")
            return False

        try:
            # Format message
            message = alert.format_text()

            # Telegram API endpoint
            url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"

            payload = {
                "chat_id": self.config.telegram_chat_id,
                "text": message,
                "parse_mode": "HTML" if "<" in message else None,
            }

            response = requests.post(url, json=payload, timeout=10)
            response.raise_for_status()

            self._logger.debug(f"Telegram alert sent: {alert.title}")
            return True

        except Exception as e:
            self._logger.error(f"Telegram alert failed: {e}")
            return False


class DiscordAlert(AlertChannelBase):
    """Discord webhook alert channel."""

    def is_configured(self) -> bool:
        return self.config.discord_webhook_url is not None

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            self._logger.warning("Discord not configured")
            return False

        if not REQUESTS_AVAILABLE:
            self._logger.error("requests library not available")
            return False

        try:
            # Build Discord embed
            embed = {
                "title": f"{alert.level.get_emoji()} {alert.title}",
                "description": alert.message,
                "color": alert.level.get_color(),
                "timestamp": alert.timestamp.isoformat(),
                "footer": {"text": "Trading Bot Alert"},
            }

            # Add fields for data
            if alert.data:
                embed["fields"] = [
                    {"name": k, "value": str(v), "inline": True}
                    for k, v in alert.data.items()
                ]

            payload = {
                "embeds": [embed],
                "username": "Trading Bot",
            }

            response = requests.post(
                self.config.discord_webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()

            self._logger.debug(f"Discord alert sent: {alert.title}")
            return True

        except Exception as e:
            self._logger.error(f"Discord alert failed: {e}")
            return False


class EmailAlert(AlertChannelBase):
    """Email alert channel via SMTP."""

    def is_configured(self) -> bool:
        return (
            self.config.smtp_user is not None and
            self.config.smtp_password is not None and
            self.config.email_from is not None and
            len(self.config.email_to) > 0
        )

    def send(self, alert: Alert) -> bool:
        if not self.is_configured():
            self._logger.warning("Email not configured")
            return False

        try:
            # Create message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert.level.name}] Trading Alert: {alert.title}"
            msg['From'] = self.config.email_from
            msg['To'] = ', '.join(self.config.email_to)

            # Plain text version
            text_part = MIMEText(alert.format_text(), 'plain')
            msg.attach(text_part)

            # HTML version
            html_part = MIMEText(alert.format_html(), 'html')
            msg.attach(html_part)

            # Send
            with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
                server.starttls()
                server.login(self.config.smtp_user, self.config.smtp_password)
                server.send_message(msg)

            self._logger.debug(f"Email alert sent: {alert.title}")
            return True

        except Exception as e:
            self._logger.error(f"Email alert failed: {e}")
            return False


class ConsoleAlert(AlertChannelBase):
    """Console output alert channel (for debugging)."""

    def is_configured(self) -> bool:
        return True

    def send(self, alert: Alert) -> bool:
        print(f"\n{'='*60}")
        print(alert.format_text())
        print(f"{'='*60}\n")
        return True


# =============================================================================
# ALERT MANAGER
# =============================================================================

class AlertManager:
    """
    Central alert management system.

    Handles:
    - Multi-channel delivery
    - Rate limiting
    - Alert queuing
    - Deduplication
    - Async delivery

    Usage:
        manager = AlertManager(config)
        manager.start()

        # Send alert
        manager.alert(
            level=AlertLevel.WARNING,
            title="High Drawdown",
            message="Portfolio drawdown reached 8%",
            data={'drawdown': 8.5, 'equity': 9150}
        )

        # Or use convenience methods
        manager.info("Trade Executed", "Bought 0.1 XAUUSD @ 2025.50")
        manager.warning("Risk Alert", "Approaching max drawdown")
        manager.critical("Kill Switch", "Max drawdown exceeded!")
    """

    def __init__(self, config: AlertConfig = None):
        """
        Initialize AlertManager.

        Args:
            config: AlertConfig with channel credentials
        """
        self.config = config or AlertConfig.from_env()
        self._logger = logging.getLogger("AlertManager")

        # Initialize channels
        self._channels: Dict[AlertChannel, AlertChannelBase] = {}
        self._setup_channels()

        # Queue for async delivery
        self._queue: Queue = Queue()
        self._running = False
        self._worker_thread: Optional[Thread] = None

        # Rate limiting
        self._alert_history: List[Alert] = []
        self._lock = Lock()

        # Statistics
        self._stats = {
            'total_sent': 0,
            'by_level': {level.name: 0 for level in AlertLevel},
            'by_channel': {channel.name: 0 for channel in AlertChannel},
            'rate_limited': 0,
        }

    def _setup_channels(self):
        """Initialize configured alert channels."""
        channel_classes = {
            AlertChannel.TELEGRAM: TelegramAlert,
            AlertChannel.DISCORD: DiscordAlert,
            AlertChannel.EMAIL: EmailAlert,
            AlertChannel.CONSOLE: ConsoleAlert,
        }

        for channel_type, channel_class in channel_classes.items():
            try:
                channel = channel_class(self.config)
                if channel.is_configured():
                    self._channels[channel_type] = channel
                    self._logger.info(f"Alert channel configured: {channel_type.name}")
            except Exception as e:
                self._logger.warning(f"Failed to setup {channel_type.name}: {e}")

    def start(self):
        """Start the alert manager (async delivery)."""
        if self._running:
            return

        self._running = True
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()
        self._logger.info("AlertManager started")

    def stop(self):
        """Stop the alert manager."""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join(timeout=5)
        self._logger.info("AlertManager stopped")

    def _worker_loop(self):
        """Background worker for async alert delivery."""
        while self._running:
            try:
                alert = self._queue.get(timeout=1)
                self._deliver_alert(alert)
            except Empty:
                continue
            except Exception as e:
                self._logger.error(f"Worker error: {e}")

    def _deliver_alert(self, alert: Alert):
        """Deliver alert to all specified channels."""
        channels_to_use = alert.channels if alert.channels else list(self._channels.keys())

        for channel_type in channels_to_use:
            if channel_type not in self._channels:
                continue

            try:
                channel = self._channels[channel_type]
                if channel.send(alert):
                    self._stats['by_channel'][channel_type.name] += 1
            except Exception as e:
                self._logger.error(f"Delivery failed for {channel_type.name}: {e}")

        self._stats['total_sent'] += 1
        self._stats['by_level'][alert.level.name] += 1

    def _is_rate_limited(self, alert: Alert) -> bool:
        """Check if alert should be rate limited."""
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(hours=1)

            # Clean old alerts
            self._alert_history = [a for a in self._alert_history if a.timestamp > cutoff]

            # Check hourly limit
            if len(self._alert_history) >= self.config.max_alerts_per_hour:
                return True

            # Check duplicate within interval
            min_interval = timedelta(seconds=self.config.min_alert_interval_seconds)
            for recent in self._alert_history:
                if (alert.title == recent.title and
                    alert.level == recent.level and
                    now - recent.timestamp < min_interval):
                    return True

            return False

    def alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        data: Dict[str, Any] = None,
        channels: List[AlertChannel] = None,
        force: bool = False
    ) -> bool:
        """
        Send an alert.

        Args:
            level: Alert severity level
            title: Alert title
            message: Alert message
            data: Additional data to include
            channels: Specific channels (None = all)
            force: Bypass rate limiting

        Returns:
            True if alert was queued
        """
        # Check minimum level
        if level.value < self.config.min_level.value:
            return False

        alert = Alert(
            level=level,
            title=title,
            message=message,
            data=data or {},
            channels=channels or [],
        )

        # Rate limiting
        if not force and self._is_rate_limited(alert):
            self._stats['rate_limited'] += 1
            self._logger.debug(f"Alert rate limited: {title}")
            return False

        # Record in history
        with self._lock:
            self._alert_history.append(alert)

        # Queue for delivery
        if self._running:
            self._queue.put(alert)
        else:
            # Sync delivery if not started
            self._deliver_alert(alert)

        return True

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def debug(self, title: str, message: str, **data):
        """Send DEBUG level alert."""
        return self.alert(AlertLevel.DEBUG, title, message, data)

    def info(self, title: str, message: str, **data):
        """Send INFO level alert."""
        return self.alert(AlertLevel.INFO, title, message, data)

    def warning(self, title: str, message: str, **data):
        """Send WARNING level alert."""
        return self.alert(AlertLevel.WARNING, title, message, data)

    def error(self, title: str, message: str, **data):
        """Send ERROR level alert."""
        return self.alert(AlertLevel.ERROR, title, message, data)

    def critical(self, title: str, message: str, **data):
        """Send CRITICAL level alert (always delivered)."""
        return self.alert(AlertLevel.CRITICAL, title, message, data, force=True)

    # =========================================================================
    # TRADING-SPECIFIC ALERTS
    # =========================================================================

    def trade_opened(
        self,
        symbol: str,
        direction: str,
        volume: float,
        price: float,
        sl: float = None,
        tp: float = None,
        ticket: int = None
    ):
        """Alert for trade opened."""
        self.info(
            "Trade Opened",
            f"{direction} {volume} lots {symbol} @ {price:.2f}",
            symbol=symbol,
            direction=direction,
            volume=volume,
            price=price,
            sl=sl,
            tp=tp,
            ticket=ticket,
        )

    def trade_closed(
        self,
        symbol: str,
        direction: str,
        volume: float,
        profit: float,
        pips: float = None,
        ticket: int = None
    ):
        """Alert for trade closed."""
        level = AlertLevel.INFO if profit >= 0 else AlertLevel.WARNING
        self.alert(
            level,
            "Trade Closed",
            f"Closed {direction} {symbol}: {'+'if profit >= 0 else ''}{profit:.2f}",
            symbol=symbol,
            direction=direction,
            volume=volume,
            profit=profit,
            pips=pips,
            ticket=ticket,
        )

    def drawdown_warning(self, current_dd: float, max_dd: float):
        """Alert for drawdown approaching limit."""
        pct = (current_dd / max_dd) * 100

        if pct >= 90:
            level = AlertLevel.CRITICAL
        elif pct >= 75:
            level = AlertLevel.ERROR
        else:
            level = AlertLevel.WARNING

        self.alert(
            level,
            "Drawdown Warning",
            f"Current drawdown: {current_dd:.1%} ({pct:.0f}% of limit)",
            current_dd=f"{current_dd:.2%}",
            max_dd=f"{max_dd:.2%}",
            utilization=f"{pct:.1f}%",
        )

    def kill_switch_triggered(self, reason: str, equity: float = None):
        """Alert for kill switch activation."""
        self.critical(
            "KILL SWITCH ACTIVATED",
            f"Trading halted: {reason}",
            reason=reason,
            equity=equity,
            action="All new trades blocked",
        )

    def connection_lost(self, service: str):
        """Alert for lost connection."""
        self.error(
            "Connection Lost",
            f"Lost connection to {service}",
            service=service,
            action="Attempting to reconnect...",
        )

    def connection_restored(self, service: str):
        """Alert for restored connection."""
        self.info(
            "Connection Restored",
            f"Connection to {service} restored",
            service=service,
        )

    def daily_summary(
        self,
        trades: int,
        profit: float,
        win_rate: float,
        max_dd: float
    ):
        """Send daily trading summary."""
        self.info(
            "Daily Summary",
            f"Trades: {trades} | P&L: ${profit:+.2f} | Win Rate: {win_rate:.1%}",
            trades=trades,
            profit=f"${profit:+.2f}",
            win_rate=f"{win_rate:.1%}",
            max_drawdown=f"{max_dd:.2%}",
        )

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            **self._stats,
            'queue_size': self._queue.qsize(),
            'active_channels': list(self._channels.keys()),
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_alert_manager(
    telegram_token: str = None,
    telegram_chat_id: str = None,
    discord_webhook: str = None,
    email_config: Dict = None,
    auto_start: bool = True
) -> AlertManager:
    """
    Factory function to create AlertManager.

    Args:
        telegram_token: Telegram bot token
        telegram_chat_id: Telegram chat ID
        discord_webhook: Discord webhook URL
        email_config: Email configuration dict
        auto_start: Start worker thread immediately

    Returns:
        Configured AlertManager
    """
    config = AlertConfig(
        telegram_bot_token=telegram_token,
        telegram_chat_id=telegram_chat_id,
        discord_webhook_url=discord_webhook,
    )

    if email_config:
        config.smtp_host = email_config.get('host', 'smtp.gmail.com')
        config.smtp_port = email_config.get('port', 587)
        config.smtp_user = email_config.get('user')
        config.smtp_password = email_config.get('password')
        config.email_from = email_config.get('from')
        config.email_to = email_config.get('to', [])

    manager = AlertManager(config)

    if auto_start:
        manager.start()

    return manager
