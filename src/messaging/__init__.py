# =============================================================================
# MESSAGING MODULE
# =============================================================================
# Async messaging infrastructure for agent communication.
#
# Components:
# - EventQueue: Async event queue with priorities
# - DeadLetterQueue: Failed message handling
# - EventBus: Pub/sub for agent events
#
# =============================================================================

from src.messaging.event_queue import (
    Event,
    EventType,
    EventPriority,
    EventQueue,
    DeadLetterQueue,
    EventBus,
    EventHandler,
)

__all__ = [
    'Event',
    'EventType',
    'EventPriority',
    'EventQueue',
    'DeadLetterQueue',
    'EventBus',
    'EventHandler',
]
