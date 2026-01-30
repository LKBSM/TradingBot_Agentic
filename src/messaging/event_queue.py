# =============================================================================
# EVENT QUEUE
# =============================================================================
# Async event queue for inter-agent communication.
#
# Features:
# - Priority-based message ordering
# - Async/await compatible
# - Dead letter queue for failed messages
# - Timeout support
# - Pub/sub event bus pattern
#
# Usage:
#   # Basic queue
#   queue = EventQueue()
#   await queue.put(Event(EventType.TRADE_PROPOSAL, data={'action': 1}))
#   event = await queue.get(timeout=5.0)
#
#   # Event bus (pub/sub)
#   bus = EventBus()
#   bus.subscribe(EventType.TRADE_EXECUTED, handle_trade)
#   await bus.publish(Event(EventType.TRADE_EXECUTED, data=trade_result))
#
# =============================================================================

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Coroutine, Dict, List, Optional, Set
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# EVENT TYPES
# =============================================================================

class EventType(Enum):
    """Types of events in the trading system."""

    # Agent lifecycle
    AGENT_STARTED = auto()
    AGENT_STOPPED = auto()
    AGENT_ERROR = auto()
    AGENT_HEARTBEAT = auto()

    # Trade flow
    TRADE_PROPOSAL = auto()
    TRADE_APPROVED = auto()
    TRADE_REJECTED = auto()
    TRADE_MODIFIED = auto()
    TRADE_EXECUTED = auto()
    TRADE_FAILED = auto()
    TRADE_CLOSED = auto()

    # Risk events
    RISK_LIMIT_BREACH = auto()
    RISK_WARNING = auto()
    DRAWDOWN_ALERT = auto()
    KILL_SWITCH_TRIGGERED = auto()

    # Market events
    MARKET_DATA_UPDATE = auto()
    NEWS_ALERT = auto()
    REGIME_CHANGE = auto()
    VOLATILITY_SPIKE = auto()

    # System events
    SYSTEM_STARTUP = auto()
    SYSTEM_SHUTDOWN = auto()
    CONFIG_RELOAD = auto()
    HEALTH_CHECK = auto()

    # Custom/generic
    CUSTOM = auto()


class EventPriority(Enum):
    """Priority levels for events."""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3
    EMERGENCY = 4

    def __lt__(self, other):
        if isinstance(other, EventPriority):
            return self.value < other.value
        return NotImplemented


# =============================================================================
# EVENT DATA STRUCTURES
# =============================================================================

@dataclass(order=True)
class Event:
    """
    Event message for inter-agent communication.

    Events are ordered by priority (higher first), then by timestamp (older first).
    """
    # For priority queue ordering
    sort_index: tuple = field(init=False, repr=False)

    # Event metadata
    event_type: EventType = field(compare=False)
    priority: EventPriority = field(compare=False, default=EventPriority.NORMAL)
    event_id: str = field(compare=False, default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(compare=False, default_factory=datetime.utcnow)

    # Event content
    source: str = field(compare=False, default="")
    target: str = field(compare=False, default="")  # Empty = broadcast
    data: Dict[str, Any] = field(compare=False, default_factory=dict)

    # Retry tracking
    retry_count: int = field(compare=False, default=0)
    max_retries: int = field(compare=False, default=3)
    last_error: Optional[str] = field(compare=False, default=None)

    def __post_init__(self):
        # Sort by priority (descending) then timestamp (ascending)
        # Negate priority so higher priority comes first
        self.sort_index = (-self.priority.value, self.timestamp.timestamp())

    def can_retry(self) -> bool:
        """Check if event can be retried."""
        return self.retry_count < self.max_retries

    def mark_retry(self, error: str) -> None:
        """Mark event for retry."""
        self.retry_count += 1
        self.last_error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.name,
            'priority': self.priority.name,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'target': self.target,
            'data': self.data,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries,
            'last_error': self.last_error,
        }


@dataclass
class DeadLetterRecord:
    """Record of a failed event in the dead letter queue."""
    event: Event
    failure_reason: str
    failure_time: datetime = field(default_factory=datetime.utcnow)
    original_source: str = ""
    handler_name: str = ""


# =============================================================================
# EVENT QUEUE
# =============================================================================

class EventQueue:
    """
    Async priority queue for events.

    Uses asyncio.PriorityQueue for ordering events by priority and timestamp.
    Supports timeout, bounded capacity, and metrics.
    """

    def __init__(
        self,
        maxsize: int = 0,
        name: str = "default"
    ):
        """
        Initialize event queue.

        Args:
            maxsize: Maximum queue size (0 = unlimited)
            name: Queue name for logging
        """
        self._queue: asyncio.PriorityQueue = asyncio.PriorityQueue(maxsize=maxsize)
        self._name = name
        self._closed = False

        # Metrics
        self._total_put = 0
        self._total_get = 0
        self._total_timeout = 0

        self._logger = logging.getLogger(f"event_queue.{name}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def qsize(self) -> int:
        """Get current queue size."""
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()

    @property
    def full(self) -> bool:
        """Check if queue is full."""
        return self._queue.full()

    async def put(
        self,
        event: Event,
        timeout: Optional[float] = None
    ) -> bool:
        """
        Put an event into the queue.

        Args:
            event: Event to queue
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            True if event was queued

        Raises:
            asyncio.TimeoutError: If timeout exceeded
            RuntimeError: If queue is closed
        """
        if self._closed:
            raise RuntimeError(f"Queue '{self._name}' is closed")

        try:
            if timeout is not None:
                await asyncio.wait_for(
                    self._queue.put(event),
                    timeout=timeout
                )
            else:
                await self._queue.put(event)

            self._total_put += 1
            self._logger.debug(
                f"Event queued: {event.event_type.name} "
                f"(priority={event.priority.name}, id={event.event_id[:8]})"
            )
            return True

        except asyncio.TimeoutError:
            self._total_timeout += 1
            self._logger.warning(
                f"Timeout putting event: {event.event_type.name}"
            )
            raise

    async def get(
        self,
        timeout: Optional[float] = None
    ) -> Optional[Event]:
        """
        Get an event from the queue.

        Args:
            timeout: Timeout in seconds (None = wait forever)

        Returns:
            Event or None if timeout

        Raises:
            RuntimeError: If queue is closed and empty
        """
        if self._closed and self._queue.empty():
            raise RuntimeError(f"Queue '{self._name}' is closed and empty")

        try:
            if timeout is not None:
                event = await asyncio.wait_for(
                    self._queue.get(),
                    timeout=timeout
                )
            else:
                event = await self._queue.get()

            self._total_get += 1
            self._logger.debug(
                f"Event dequeued: {event.event_type.name} "
                f"(id={event.event_id[:8]})"
            )
            return event

        except asyncio.TimeoutError:
            self._total_timeout += 1
            return None

    def get_nowait(self) -> Optional[Event]:
        """Get an event without waiting."""
        try:
            return self._queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def task_done(self) -> None:
        """Mark current task as done."""
        self._queue.task_done()

    async def join(self) -> None:
        """Wait for all events to be processed."""
        await self._queue.join()

    def close(self) -> None:
        """Close the queue (no new events accepted)."""
        self._closed = True
        self._logger.info(f"Queue '{self._name}' closed")

    def get_metrics(self) -> Dict[str, Any]:
        """Get queue metrics."""
        return {
            'name': self._name,
            'current_size': self.qsize,
            'is_closed': self._closed,
            'total_put': self._total_put,
            'total_get': self._total_get,
            'total_timeout': self._total_timeout,
        }


# =============================================================================
# DEAD LETTER QUEUE
# =============================================================================

class DeadLetterQueue:
    """
    Queue for failed events that couldn't be processed.

    Failed events are stored for later analysis or manual intervention.
    """

    def __init__(
        self,
        maxsize: int = 1000,
        name: str = "dlq"
    ):
        """
        Initialize dead letter queue.

        Args:
            maxsize: Maximum number of failed events to keep
            name: Queue name for logging
        """
        self._records: List[DeadLetterRecord] = []
        self._maxsize = maxsize
        self._name = name
        self._lock = asyncio.Lock()

        self._logger = logging.getLogger(f"dead_letter_queue.{name}")

    @property
    def size(self) -> int:
        """Get current DLQ size."""
        return len(self._records)

    async def add(
        self,
        event: Event,
        reason: str,
        handler_name: str = ""
    ) -> None:
        """
        Add a failed event to the dead letter queue.

        Args:
            event: Failed event
            reason: Failure reason
            handler_name: Name of handler that failed
        """
        async with self._lock:
            record = DeadLetterRecord(
                event=event,
                failure_reason=reason,
                original_source=event.source,
                handler_name=handler_name,
            )

            self._records.append(record)

            # Trim if over capacity
            if len(self._records) > self._maxsize:
                removed = self._records.pop(0)
                self._logger.warning(
                    f"DLQ overflow, removed oldest: {removed.event.event_id}"
                )

            self._logger.error(
                f"Event added to DLQ: {event.event_type.name} "
                f"(id={event.event_id[:8]}, reason={reason})"
            )

    async def get_all(self) -> List[DeadLetterRecord]:
        """Get all dead letter records."""
        async with self._lock:
            return list(self._records)

    async def get_by_type(
        self,
        event_type: EventType
    ) -> List[DeadLetterRecord]:
        """Get dead letter records by event type."""
        async with self._lock:
            return [
                r for r in self._records
                if r.event.event_type == event_type
            ]

    async def clear(self) -> int:
        """Clear all dead letter records."""
        async with self._lock:
            count = len(self._records)
            self._records.clear()
            self._logger.info(f"Cleared {count} records from DLQ")
            return count

    async def retry(
        self,
        event_id: str,
        target_queue: EventQueue
    ) -> bool:
        """
        Retry a failed event.

        Args:
            event_id: ID of event to retry
            target_queue: Queue to put retried event into

        Returns:
            True if event was found and requeued
        """
        async with self._lock:
            for i, record in enumerate(self._records):
                if record.event.event_id == event_id:
                    event = record.event
                    if event.can_retry():
                        event.mark_retry(record.failure_reason)
                        await target_queue.put(event)
                        self._records.pop(i)
                        self._logger.info(
                            f"Retrying event: {event.event_id[:8]} "
                            f"(attempt {event.retry_count}/{event.max_retries})"
                        )
                        return True
                    else:
                        self._logger.warning(
                            f"Event {event_id} exceeded max retries"
                        )
                        return False

            return False

    def get_metrics(self) -> Dict[str, Any]:
        """Get DLQ metrics."""
        return {
            'name': self._name,
            'current_size': self.size,
            'max_size': self._maxsize,
            'by_type': dict(
                (t.name, sum(1 for r in self._records if r.event.event_type == t))
                for t in EventType
                if any(r.event.event_type == t for r in self._records)
            ),
        }


# =============================================================================
# EVENT BUS (PUB/SUB)
# =============================================================================

# Type alias for event handlers
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Pub/sub event bus for agent communication.

    Allows agents to subscribe to event types and receive notifications
    when those events are published.
    """

    def __init__(
        self,
        name: str = "main",
        dead_letter_queue: Optional[DeadLetterQueue] = None
    ):
        """
        Initialize event bus.

        Args:
            name: Bus name for logging
            dead_letter_queue: DLQ for failed deliveries
        """
        self._name = name
        self._subscribers: Dict[EventType, List[tuple]] = defaultdict(list)
        self._global_subscribers: List[tuple] = []
        self._dlq = dead_letter_queue or DeadLetterQueue(name=f"{name}_dlq")
        self._lock = asyncio.Lock()

        # Metrics
        self._total_published = 0
        self._total_delivered = 0
        self._total_failed = 0

        self._logger = logging.getLogger(f"event_bus.{name}")

    def subscribe(
        self,
        event_type: EventType,
        handler: EventHandler,
        handler_name: str = ""
    ) -> None:
        """
        Subscribe to events of a specific type.

        Args:
            event_type: Type of events to receive
            handler: Async function to handle events
            handler_name: Name for logging
        """
        name = handler_name or handler.__name__
        self._subscribers[event_type].append((handler, name))
        self._logger.debug(f"Subscribed '{name}' to {event_type.name}")

    def subscribe_all(
        self,
        handler: EventHandler,
        handler_name: str = ""
    ) -> None:
        """
        Subscribe to all event types.

        Args:
            handler: Async function to handle events
            handler_name: Name for logging
        """
        name = handler_name or handler.__name__
        self._global_subscribers.append((handler, name))
        self._logger.debug(f"Subscribed '{name}' to ALL events")

    def unsubscribe(
        self,
        event_type: EventType,
        handler: EventHandler
    ) -> bool:
        """
        Unsubscribe from events.

        Args:
            event_type: Type of events
            handler: Handler to remove

        Returns:
            True if handler was found and removed
        """
        subscribers = self._subscribers[event_type]
        for i, (h, name) in enumerate(subscribers):
            if h == handler:
                subscribers.pop(i)
                self._logger.debug(f"Unsubscribed '{name}' from {event_type.name}")
                return True
        return False

    async def publish(
        self,
        event: Event,
        timeout: float = 5.0
    ) -> int:
        """
        Publish an event to all subscribers.

        Args:
            event: Event to publish
            timeout: Timeout for each handler

        Returns:
            Number of successful deliveries
        """
        self._total_published += 1

        # Get handlers for this event type
        handlers = list(self._subscribers.get(event.event_type, []))
        handlers.extend(self._global_subscribers)

        if not handlers:
            self._logger.debug(
                f"No subscribers for {event.event_type.name}"
            )
            return 0

        # Deliver to all handlers
        delivered = 0
        for handler, handler_name in handlers:
            try:
                await asyncio.wait_for(
                    handler(event),
                    timeout=timeout
                )
                delivered += 1
                self._total_delivered += 1

            except asyncio.TimeoutError:
                self._total_failed += 1
                self._logger.error(
                    f"Timeout delivering to '{handler_name}': "
                    f"{event.event_type.name}"
                )
                await self._dlq.add(
                    event,
                    f"Handler timeout: {handler_name}",
                    handler_name
                )

            except Exception as e:
                self._total_failed += 1
                self._logger.error(
                    f"Error delivering to '{handler_name}': {e}"
                )
                await self._dlq.add(
                    event,
                    f"Handler error: {e}",
                    handler_name
                )

        self._logger.debug(
            f"Published {event.event_type.name}: "
            f"{delivered}/{len(handlers)} delivered"
        )

        return delivered

    async def publish_and_wait(
        self,
        event: Event,
        timeout: float = 30.0
    ) -> List[Any]:
        """
        Publish event and collect responses from handlers.

        Handlers should return a value, which will be collected.

        Args:
            event: Event to publish
            timeout: Total timeout for all handlers

        Returns:
            List of responses from handlers
        """
        handlers = list(self._subscribers.get(event.event_type, []))
        handlers.extend(self._global_subscribers)

        if not handlers:
            return []

        # Create tasks for all handlers
        async def call_handler(handler, name):
            try:
                return await asyncio.wait_for(handler(event), timeout=timeout)
            except Exception as e:
                self._logger.error(f"Error in handler '{name}': {e}")
                return None

        tasks = [
            call_handler(handler, name)
            for handler, name in handlers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None and exceptions
        return [r for r in results if r is not None and not isinstance(r, Exception)]

    def get_subscribers(
        self,
        event_type: EventType
    ) -> List[str]:
        """Get names of subscribers for an event type."""
        handlers = self._subscribers.get(event_type, [])
        return [name for _, name in handlers]

    def get_metrics(self) -> Dict[str, Any]:
        """Get event bus metrics."""
        return {
            'name': self._name,
            'total_published': self._total_published,
            'total_delivered': self._total_delivered,
            'total_failed': self._total_failed,
            'delivery_rate': (
                self._total_delivered / max(1, self._total_published)
            ),
            'subscriber_counts': {
                et.name: len(handlers)
                for et, handlers in self._subscribers.items()
                if handlers
            },
            'global_subscribers': len(self._global_subscribers),
            'dlq_metrics': self._dlq.get_metrics(),
        }

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """Access the dead letter queue."""
        return self._dlq


# =============================================================================
# QUEUE PROCESSOR
# =============================================================================

class QueueProcessor:
    """
    Background processor for event queues.

    Continuously processes events from a queue and dispatches them
    to the event bus.
    """

    def __init__(
        self,
        queue: EventQueue,
        bus: EventBus,
        name: str = "processor"
    ):
        """
        Initialize queue processor.

        Args:
            queue: Event queue to process
            bus: Event bus to publish to
            name: Processor name
        """
        self._queue = queue
        self._bus = bus
        self._name = name
        self._running = False
        self._task: Optional[asyncio.Task] = None

        self._logger = logging.getLogger(f"queue_processor.{name}")

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self) -> None:
        """Start processing events."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._process_loop())
        self._logger.info(f"Queue processor '{self._name}' started")

    async def stop(self, timeout: float = 5.0) -> None:
        """Stop processing events."""
        if not self._running:
            return

        self._running = False
        self._queue.close()

        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=timeout)
            except asyncio.TimeoutError:
                self._task.cancel()

        self._logger.info(f"Queue processor '{self._name}' stopped")

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                event = await self._queue.get(timeout=1.0)
                if event:
                    await self._bus.publish(event)
                    self._queue.task_done()

            except Exception as e:
                self._logger.error(f"Error processing event: {e}")

    def get_metrics(self) -> Dict[str, Any]:
        """Get processor metrics."""
        return {
            'name': self._name,
            'is_running': self._running,
            'queue_metrics': self._queue.get_metrics(),
            'bus_metrics': self._bus.get_metrics(),
        }
