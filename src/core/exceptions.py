# =============================================================================
# CUSTOM EXCEPTIONS
# =============================================================================
# Hierarchical exception system for intelligent error handling.
#
# Exception Hierarchy:
#
#   TradingError (base)
#   ├── TransientError (retry possible)
#   │   ├── DataFeedError
#   │   ├── BrokerConnectionError
#   │   └── TimeoutError
#   └── PermanentError (don't retry)
#       ├── ConfigurationError
#       ├── ValidationError
#       ├── RiskLimitError
#       ├── ExecutionError
#       └── AgentError
#
# Usage:
#   try:
#       result = broker.execute_trade(order)
#   except TransientError as e:
#       # Retry with backoff
#       result = retry_with_backoff(broker.execute_trade, order)
#   except RiskLimitError as e:
#       # Don't retry - this is expected behavior
#       logger.info(f"Trade blocked by risk limit: {e}")
#   except PermanentError as e:
#       # Alert operator - something is wrong
#       alert_manager.send_critical(f"Permanent error: {e}")
#
# =============================================================================

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# =============================================================================
# BASE EXCEPTIONS
# =============================================================================

@dataclass
class ErrorContext:
    """Context information for errors."""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    component: str = ""
    operation: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class TradingError(Exception):
    """
    Base exception for all trading-related errors.

    All custom exceptions inherit from this class, allowing
    catch-all handling when needed.
    """

    def __init__(
        self,
        message: str,
        context: Optional[ErrorContext] = None,
        cause: Optional[Exception] = None
    ):
        super().__init__(message)
        self.message = message
        self.context = context or ErrorContext()
        self.cause = cause
        self.timestamp = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/serialization."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'component': self.context.component,
            'operation': self.context.operation,
            'details': self.context.details,
            'cause': str(self.cause) if self.cause else None,
        }


class TransientError(TradingError):
    """
    Transient error - retry is possible and recommended.

    These errors are typically caused by temporary conditions
    like network issues, service unavailability, or rate limits.

    Attributes:
        retry_after_seconds: Suggested wait time before retry
        max_retries: Suggested maximum retry attempts
    """

    retry_after_seconds: int = 5
    max_retries: int = 3

    def __init__(
        self,
        message: str,
        retry_after: int = None,
        max_retries: int = None,
        **kwargs
    ):
        super().__init__(message, **kwargs)
        if retry_after is not None:
            self.retry_after_seconds = retry_after
        if max_retries is not None:
            self.max_retries = max_retries


class PermanentError(TradingError):
    """
    Permanent error - do not retry.

    These errors indicate a condition that won't be resolved
    by retrying, such as invalid configuration, validation
    failures, or intentional blocks (risk limits).
    """

    def __init__(self, message: str, recoverable: bool = False, **kwargs):
        super().__init__(message, **kwargs)
        self.recoverable = recoverable


# =============================================================================
# TRANSIENT ERRORS (retry possible)
# =============================================================================

class DataFeedError(TransientError):
    """
    Error fetching market data.

    Typically caused by:
    - API rate limits
    - Network timeouts
    - Service unavailability
    """

    retry_after_seconds = 5
    max_retries = 5

    def __init__(self, message: str, source: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.source = source
        self.context.component = "data_feed"
        self.context.details['source'] = source


class BrokerConnectionError(TransientError):
    """
    Error connecting to broker.

    Typically caused by:
    - Network issues
    - Broker server maintenance
    - Authentication token expiry
    """

    retry_after_seconds = 30
    max_retries = 3

    def __init__(self, message: str, broker: str = "", **kwargs):
        super().__init__(message, **kwargs)
        self.broker = broker
        self.context.component = "broker"
        self.context.details['broker'] = broker


class TimeoutError(TransientError):
    """
    Operation timed out.

    The operation took longer than the allowed time limit.
    May succeed on retry.
    """

    retry_after_seconds = 10
    max_retries = 2

    def __init__(
        self,
        message: str,
        timeout_seconds: float = 0,
        operation: str = "",
        **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_seconds = timeout_seconds
        self.context.operation = operation
        self.context.details['timeout_seconds'] = timeout_seconds


class RateLimitError(TransientError):
    """
    Rate limit exceeded.

    The API or service has rate-limited requests.
    Wait before retrying.
    """

    def __init__(
        self,
        message: str,
        retry_after: int = 60,
        limit: int = 0,
        **kwargs
    ):
        super().__init__(message, retry_after=retry_after, **kwargs)
        self.limit = limit
        self.context.details['limit'] = limit


# =============================================================================
# PERMANENT ERRORS (don't retry)
# =============================================================================

class ConfigurationError(PermanentError):
    """
    Invalid configuration.

    The system configuration is invalid or incomplete.
    Requires manual intervention to fix.
    """

    def __init__(self, message: str, config_key: str = "", **kwargs):
        super().__init__(message, recoverable=False, **kwargs)
        self.config_key = config_key
        self.context.component = "configuration"
        self.context.details['config_key'] = config_key


class ValidationError(PermanentError):
    """
    Input validation failed.

    The input data failed validation checks.
    The request should not be retried with the same data.
    """

    def __init__(
        self,
        message: str,
        field: str = "",
        value: Any = None,
        **kwargs
    ):
        super().__init__(message, recoverable=False, **kwargs)
        self.field = field
        self.value = value
        self.context.component = "validation"
        self.context.details['field'] = field
        self.context.details['value'] = str(value)


class RiskLimitError(PermanentError):
    """
    Risk limit exceeded.

    The proposed action would exceed configured risk limits.
    This is expected behavior, not a system error.

    Note: This should typically be logged at INFO level,
    not ERROR, as it indicates the risk system working correctly.
    """

    def __init__(
        self,
        message: str,
        limit_name: str = "",
        limit_value: float = 0,
        current_value: float = 0,
        **kwargs
    ):
        super().__init__(message, recoverable=True, **kwargs)
        self.limit_name = limit_name
        self.limit_value = limit_value
        self.current_value = current_value
        self.context.component = "risk_management"
        self.context.details.update({
            'limit_name': limit_name,
            'limit_value': limit_value,
            'current_value': current_value,
        })


class ExecutionError(PermanentError):
    """
    Trade execution failed.

    The trade could not be executed due to:
    - Insufficient funds
    - Market closed
    - Invalid order parameters
    - Broker rejection
    """

    def __init__(
        self,
        message: str,
        order_id: str = "",
        rejection_reason: str = "",
        **kwargs
    ):
        super().__init__(message, recoverable=False, **kwargs)
        self.order_id = order_id
        self.rejection_reason = rejection_reason
        self.context.component = "execution"
        self.context.details.update({
            'order_id': order_id,
            'rejection_reason': rejection_reason,
        })


class AgentError(PermanentError):
    """
    Agent encountered an unrecoverable error.

    The agent has failed in a way that requires attention.
    The orchestrator should apply graceful degradation.
    """

    def __init__(
        self,
        message: str,
        agent_name: str = "",
        agent_state: str = "",
        **kwargs
    ):
        super().__init__(message, recoverable=True, **kwargs)
        self.agent_name = agent_name
        self.agent_state = agent_state
        self.context.component = f"agent.{agent_name}"
        self.context.details.update({
            'agent_name': agent_name,
            'agent_state': agent_state,
        })


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def is_retryable(error: Exception) -> bool:
    """
    Check if an error is retryable.

    Args:
        error: Exception to check

    Returns:
        True if the error is a TransientError
    """
    return isinstance(error, TransientError)


def get_retry_delay(error: Exception) -> int:
    """
    Get recommended retry delay for an error.

    Args:
        error: Exception to get delay for

    Returns:
        Retry delay in seconds (0 if not retryable)
    """
    if isinstance(error, TransientError):
        return error.retry_after_seconds
    return 0


def wrap_exception(
    error: Exception,
    context: ErrorContext = None
) -> TradingError:
    """
    Wrap a standard exception in a TradingError.

    Args:
        error: Exception to wrap
        context: Optional context information

    Returns:
        TradingError wrapping the original exception
    """
    if isinstance(error, TradingError):
        # Already a TradingError, just update context if provided
        if context:
            error.context = context
        return error

    # Determine error type based on message/type
    message = str(error)
    lower_msg = message.lower()

    if any(x in lower_msg for x in ['timeout', 'timed out']):
        return TimeoutError(message, cause=error, context=context)
    elif any(x in lower_msg for x in ['connection', 'network', 'socket']):
        return BrokerConnectionError(message, cause=error, context=context)
    elif any(x in lower_msg for x in ['rate limit', 'too many requests']):
        return RateLimitError(message, cause=error, context=context)
    else:
        # Default to TransientError for unknown errors
        return TransientError(message, cause=error, context=context)
