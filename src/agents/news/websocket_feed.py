# =============================================================================
# WEBSOCKET NEWS FEED - Real-time News via WebSocket
# =============================================================================
"""
WebSocket client for receiving real-time news updates.

Sprint 3 Feature: Reduces news latency from 1-5 minutes (polling) to < 1 second.

Supports:
- Automatic reconnection with exponential backoff
- Message deduplication
- Multiple provider connections
- Heartbeat monitoring

Usage:
    feed = WebSocketNewsFeed(url="wss://news.example.com/ws")
    async for article in feed.stream():
        process_article(article)
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Callable, AsyncIterator
from enum import Enum, auto
import logging
import asyncio
import json

logger = logging.getLogger(__name__)

# Try to import websockets (optional dependency)
try:
    import websockets
    HAS_WEBSOCKETS = True
except ImportError:
    HAS_WEBSOCKETS = False
    logger.warning("websockets not installed, WebSocket feed will be limited")

from .sources.base_adapter import NewsArticle, ArticleSource, ArticleCategory


class ConnectionState(Enum):
    """WebSocket connection state."""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    CLOSED = auto()


@dataclass
class WebSocketConfig:
    """Configuration for WebSocket news feed."""
    url: str
    name: str = "WebSocket"

    # Connection settings
    connect_timeout_sec: float = 30.0
    ping_interval_sec: float = 30.0
    ping_timeout_sec: float = 10.0

    # Reconnection
    auto_reconnect: bool = True
    max_reconnect_attempts: int = 10
    reconnect_base_delay_sec: float = 1.0
    reconnect_max_delay_sec: float = 60.0

    # Authentication
    auth_token: Optional[str] = None
    auth_header: str = "Authorization"

    # Message handling
    max_queue_size: int = 1000
    dedupe_window_sec: int = 60


class WebSocketNewsFeed:
    """
    WebSocket client for real-time news feeds.

    Provides async streaming of news articles with automatic
    reconnection and error handling.
    """

    def __init__(self, config: WebSocketConfig):
        """
        Initialize WebSocket feed.

        Args:
            config: WebSocket configuration
        """
        self.config = config
        self._logger = logging.getLogger(f"news.ws.{config.name}")

        # Connection state
        self._state = ConnectionState.DISCONNECTED
        self._websocket: Optional[Any] = None
        self._reconnect_count: int = 0
        self._last_message_time: Optional[datetime] = None

        # Message queue for buffering
        self._message_queue: asyncio.Queue = asyncio.Queue(maxsize=config.max_queue_size)

        # Deduplication
        self._seen_ids: Dict[str, datetime] = {}

        # Callbacks
        self._on_connect_callbacks: List[Callable] = []
        self._on_disconnect_callbacks: List[Callable] = []
        self._on_error_callbacks: List[Callable] = []

        # Statistics
        self._messages_received: int = 0
        self._messages_processed: int = 0
        self._connect_time: Optional[datetime] = None

    @property
    def state(self) -> ConnectionState:
        """Get current connection state."""
        return self._state

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._state == ConnectionState.CONNECTED

    async def connect(self) -> bool:
        """
        Establish WebSocket connection.

        Returns:
            True if connection successful
        """
        if not HAS_WEBSOCKETS:
            self._logger.error("websockets library not installed")
            return False

        if self._state == ConnectionState.CONNECTED:
            return True

        self._state = ConnectionState.CONNECTING
        self._logger.info(f"Connecting to {self.config.url}")

        try:
            # Build connection headers
            headers = {}
            if self.config.auth_token:
                headers[self.config.auth_header] = f"Bearer {self.config.auth_token}"

            # Connect
            self._websocket = await asyncio.wait_for(
                websockets.connect(
                    self.config.url,
                    extra_headers=headers,
                    ping_interval=self.config.ping_interval_sec,
                    ping_timeout=self.config.ping_timeout_sec
                ),
                timeout=self.config.connect_timeout_sec
            )

            self._state = ConnectionState.CONNECTED
            self._reconnect_count = 0
            self._connect_time = datetime.now()

            self._logger.info("Connected successfully")

            # Notify callbacks
            for callback in self._on_connect_callbacks:
                try:
                    await callback() if asyncio.iscoroutinefunction(callback) else callback()
                except Exception as e:
                    self._logger.error(f"Connect callback error: {e}")

            return True

        except asyncio.TimeoutError:
            self._logger.error("Connection timeout")
            self._state = ConnectionState.DISCONNECTED
            return False

        except Exception as e:
            self._logger.error(f"Connection error: {e}")
            self._state = ConnectionState.DISCONNECTED
            return False

    async def disconnect(self) -> None:
        """Close WebSocket connection."""
        self._state = ConnectionState.CLOSED

        if self._websocket:
            try:
                await self._websocket.close()
            except Exception as e:
                self._logger.debug(f"Error closing websocket: {e}")

            self._websocket = None

        self._logger.info("Disconnected")

        # Notify callbacks
        for callback in self._on_disconnect_callbacks:
            try:
                await callback() if asyncio.iscoroutinefunction(callback) else callback()
            except Exception as e:
                self._logger.error(f"Disconnect callback error: {e}")

    async def stream(self) -> AsyncIterator[NewsArticle]:
        """
        Stream news articles as they arrive.

        Yields:
            NewsArticle objects from the WebSocket feed
        """
        while self._state != ConnectionState.CLOSED:
            # Ensure connected
            if not self.is_connected:
                if self.config.auto_reconnect:
                    success = await self._reconnect()
                    if not success:
                        await asyncio.sleep(1)
                        continue
                else:
                    break

            try:
                # Receive message
                message = await asyncio.wait_for(
                    self._websocket.recv(),
                    timeout=self.config.ping_interval_sec + 5
                )

                self._messages_received += 1
                self._last_message_time = datetime.now()

                # Parse and yield article
                article = self._parse_message(message)
                if article and not self._is_duplicate(article):
                    self._mark_seen(article)
                    self._messages_processed += 1
                    yield article

            except asyncio.TimeoutError:
                # No message received, send ping
                self._logger.debug("Receive timeout, checking connection")
                continue

            except websockets.exceptions.ConnectionClosed as e:
                self._logger.warning(f"Connection closed: {e}")
                self._state = ConnectionState.DISCONNECTED
                if not self.config.auto_reconnect:
                    break

            except Exception as e:
                self._logger.error(f"Stream error: {e}")
                self._state = ConnectionState.DISCONNECTED

                # Notify error callbacks
                for callback in self._on_error_callbacks:
                    try:
                        await callback(e) if asyncio.iscoroutinefunction(callback) else callback(e)
                    except Exception:
                        pass

    async def _reconnect(self) -> bool:
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_count >= self.config.max_reconnect_attempts:
            self._logger.error("Max reconnection attempts reached")
            self._state = ConnectionState.CLOSED
            return False

        self._state = ConnectionState.RECONNECTING
        self._reconnect_count += 1

        # Calculate delay with exponential backoff
        delay = min(
            self.config.reconnect_base_delay_sec * (2 ** (self._reconnect_count - 1)),
            self.config.reconnect_max_delay_sec
        )

        self._logger.info(f"Reconnecting in {delay:.1f}s (attempt {self._reconnect_count})")
        await asyncio.sleep(delay)

        return await self.connect()

    def _parse_message(self, message: str) -> Optional[NewsArticle]:
        """
        Parse WebSocket message into NewsArticle.

        Override this method for provider-specific message formats.
        """
        try:
            data = json.loads(message)

            # Handle different message types
            msg_type = data.get('type', 'article')

            if msg_type == 'heartbeat':
                return None

            if msg_type == 'error':
                self._logger.warning(f"Server error: {data.get('message')}")
                return None

            # Parse article
            return NewsArticle(
                article_id=data.get('id', NewsArticle.generate_id(
                    self.config.name,
                    data.get('title', ''),
                    data.get('url', '')
                )),
                source_name=data.get('source', self.config.name),
                source_type=ArticleSource.WEBSOCKET,
                title=data.get('title', ''),
                content=data.get('content', ''),
                summary=data.get('summary', ''),
                url=data.get('url', ''),
                published_at=datetime.fromisoformat(data['published_at']) if 'published_at' in data else datetime.now(),
                category=ArticleCategory[data.get('category', 'UNKNOWN').upper()] if 'category' in data else ArticleCategory.UNKNOWN,
                assets=data.get('assets', []),
                keywords=data.get('keywords', []),
                importance=data.get('importance', 'MEDIUM')
            )

        except json.JSONDecodeError as e:
            self._logger.debug(f"Invalid JSON message: {e}")
            return None
        except Exception as e:
            self._logger.debug(f"Error parsing message: {e}")
            return None

    def _is_duplicate(self, article: NewsArticle) -> bool:
        """Check if article was already seen."""
        return article.article_id in self._seen_ids

    def _mark_seen(self, article: NewsArticle) -> None:
        """Mark article as seen."""
        self._seen_ids[article.article_id] = datetime.now()

        # Clean old entries
        cutoff = datetime.now() - timedelta(seconds=self.config.dedupe_window_sec)
        self._seen_ids = {
            k: v for k, v in self._seen_ids.items()
            if v > cutoff
        }

    def on_connect(self, callback: Callable) -> None:
        """Register callback for connection events."""
        self._on_connect_callbacks.append(callback)

    def on_disconnect(self, callback: Callable) -> None:
        """Register callback for disconnection events."""
        self._on_disconnect_callbacks.append(callback)

    def on_error(self, callback: Callable) -> None:
        """Register callback for error events."""
        self._on_error_callbacks.append(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get feed status."""
        return {
            'name': self.config.name,
            'url': self.config.url,
            'state': self._state.name,
            'is_connected': self.is_connected,
            'reconnect_count': self._reconnect_count,
            'messages_received': self._messages_received,
            'messages_processed': self._messages_processed,
            'last_message': self._last_message_time.isoformat() if self._last_message_time else None,
            'uptime_sec': (
                (datetime.now() - self._connect_time).total_seconds()
                if self._connect_time and self.is_connected else 0
            )
        }


async def create_websocket_feed(
    url: str,
    name: str = "Default",
    auth_token: Optional[str] = None
) -> WebSocketNewsFeed:
    """
    Factory function to create a WebSocket news feed.

    Args:
        url: WebSocket URL
        name: Feed name
        auth_token: Optional authentication token

    Returns:
        Connected WebSocketNewsFeed instance
    """
    config = WebSocketConfig(
        url=url,
        name=name,
        auth_token=auth_token
    )

    feed = WebSocketNewsFeed(config)
    await feed.connect()

    return feed
