# =============================================================================
# RETRY MODULE
# =============================================================================
# Retry patterns with exponential backoff and circuit breaker.
#
# Features:
# - Exponential backoff with jitter
# - Configurable retry conditions
# - Circuit breaker pattern
# - Async support
#
# Usage:
#   # Simple retry
#   result = await retry_with_backoff(fetch_data, url, max_retries=3)
#
#   # With circuit breaker
#   breaker = CircuitBreaker(failure_threshold=5)
#   result = await breaker.call(fetch_data, url)
#
# =============================================================================

import asyncio
import functools
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Optional, Set, Type, TypeVar, Union

from src.core.exceptions import (
    TradingError,
    TransientError,
    PermanentError,
    is_retryable,
    get_retry_delay,
)


logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# RETRY CONFIGURATION
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""

    # Maximum number of retry attempts
    max_retries: int = 3

    # Base delay between retries (seconds)
    base_delay: float = 1.0

    # Maximum delay between retries (seconds)
    max_delay: float = 60.0

    # Exponential backoff multiplier
    backoff_multiplier: float = 2.0

    # Add random jitter to prevent thundering herd
    jitter: bool = True

    # Maximum jitter range (0.0 to 1.0)
    jitter_range: float = 0.25

    # Exception types to retry on (empty = retry TransientError only)
    retry_on: Set[Type[Exception]] = field(default_factory=set)

    # Exception types to never retry on
    no_retry_on: Set[Type[Exception]] = field(default_factory=set)

    # Callback for retry events (for logging/metrics)
    on_retry: Optional[Callable[[int, Exception, float], None]] = None


# Default configuration
DEFAULT_RETRY_CONFIG = RetryConfig()


# =============================================================================
# RETRY LOGIC
# =============================================================================

def calculate_delay(
    attempt: int,
    config: RetryConfig,
    error: Optional[Exception] = None
) -> float:
    """
    Calculate delay before next retry attempt.

    Uses exponential backoff with optional jitter.

    Args:
        attempt: Current attempt number (0-indexed)
        config: Retry configuration
        error: The error that triggered the retry

    Returns:
        Delay in seconds
    """
    # Check if error specifies a retry delay
    if error and hasattr(error, 'retry_after_seconds'):
        base = error.retry_after_seconds
    else:
        base = config.base_delay

    # Exponential backoff
    delay = base * (config.backoff_multiplier ** attempt)

    # Cap at maximum delay
    delay = min(delay, config.max_delay)

    # Add jitter
    if config.jitter:
        jitter_amount = delay * config.jitter_range
        delay += random.uniform(-jitter_amount, jitter_amount)

    return max(0, delay)


def should_retry(
    error: Exception,
    attempt: int,
    config: RetryConfig
) -> bool:
    """
    Determine if an operation should be retried.

    Args:
        error: The error that occurred
        attempt: Current attempt number (0-indexed)
        config: Retry configuration

    Returns:
        True if should retry
    """
    # Check max retries
    if attempt >= config.max_retries:
        return False

    # Check no_retry_on list
    for exc_type in config.no_retry_on:
        if isinstance(error, exc_type):
            return False

    # Check retry_on list (if specified)
    if config.retry_on:
        for exc_type in config.retry_on:
            if isinstance(error, exc_type):
                return True
        return False

    # Default: retry on TransientError
    return is_retryable(error)


async def retry_with_backoff(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """
    Execute a function with retry and exponential backoff.

    Args:
        func: Function to execute (sync or async)
        *args: Positional arguments for func
        config: Retry configuration (uses default if not provided)
        **kwargs: Keyword arguments for func

    Returns:
        Result of the function

    Raises:
        The last exception if all retries fail
    """
    config = config or DEFAULT_RETRY_CONFIG
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)
            return result

        except Exception as e:
            last_error = e

            # Check if we should retry
            if not should_retry(e, attempt, config):
                logger.warning(
                    f"Not retrying {func.__name__}: {e.__class__.__name__} "
                    f"(attempt {attempt + 1}/{config.max_retries + 1})"
                )
                raise

            # Calculate delay
            delay = calculate_delay(attempt, config, e)

            # Log retry
            logger.warning(
                f"Retrying {func.__name__} after {delay:.2f}s: "
                f"{e.__class__.__name__}: {e} "
                f"(attempt {attempt + 1}/{config.max_retries + 1})"
            )

            # Callback
            if config.on_retry:
                try:
                    config.on_retry(attempt, e, delay)
                except Exception as cb_err:
                    logger.warning(f"Retry callback failed: {cb_err}")

            # Wait before retry
            await asyncio.sleep(delay)

    # All retries exhausted
    raise last_error


def retry_sync(
    func: Callable[..., T],
    *args,
    config: RetryConfig = None,
    **kwargs
) -> T:
    """
    Synchronous version of retry_with_backoff.

    For use in non-async contexts.
    """
    config = config or DEFAULT_RETRY_CONFIG
    last_error: Optional[Exception] = None

    for attempt in range(config.max_retries + 1):
        try:
            result = func(*args, **kwargs)
            return result

        except Exception as e:
            last_error = e

            if not should_retry(e, attempt, config):
                raise

            delay = calculate_delay(attempt, config, e)

            logger.warning(
                f"Retrying {func.__name__} after {delay:.2f}s: "
                f"{e.__class__.__name__}: {e} "
                f"(attempt {attempt + 1}/{config.max_retries + 1})"
            )

            if config.on_retry:
                try:
                    config.on_retry(attempt, e, delay)
                except Exception as cb_err:
                    logger.warning(f"Retry callback failed: {cb_err}")

            time.sleep(delay)

    raise last_error


# =============================================================================
# RETRY DECORATOR
# =============================================================================

def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    backoff_multiplier: float = 2.0,
    retry_on: Set[Type[Exception]] = None,
):
    """
    Decorator to add retry behavior to a function.

    Args:
        max_retries: Maximum retry attempts
        base_delay: Initial delay between retries
        backoff_multiplier: Delay multiplier for each retry
        retry_on: Exception types to retry on

    Example:
        @with_retry(max_retries=3, base_delay=1.0)
        async def fetch_data(url):
            ...
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        backoff_multiplier=backoff_multiplier,
        retry_on=retry_on or set(),
    )

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs) -> T:
                return await retry_with_backoff(func, *args, config=config, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs) -> T:
                return retry_sync(func, *args, config=config, **kwargs)
            return sync_wrapper

    return decorator


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """State of a circuit breaker."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""

    # Number of failures before opening circuit
    failure_threshold: int = 5

    # Time to wait before testing recovery (seconds)
    recovery_timeout: float = 30.0

    # Number of successes needed to close circuit
    success_threshold: int = 2

    # Exceptions that count as failures
    failure_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {Exception}
    )

    # Exceptions that don't count as failures (allow through)
    excluded_exceptions: Set[Type[Exception]] = field(
        default_factory=lambda: {PermanentError}
    )


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.

    Prevents cascading failures by failing fast when a service
    is experiencing problems.

    States:
    - CLOSED: Normal operation, calls go through
    - OPEN: Service is failing, calls rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5)

        try:
            result = await breaker.call(external_api.fetch, params)
        except CircuitOpenError:
            # Service is down, use fallback
            result = fallback_value
    """

    def __init__(self, config: CircuitBreakerConfig = None, name: str = ""):
        self.config = config or CircuitBreakerConfig()
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._last_state_change: datetime = datetime.utcnow()

        self._logger = logging.getLogger(f"circuit_breaker.{name}" if name else "circuit_breaker")

    @property
    def state(self) -> CircuitState:
        """Get current circuit state."""
        self._check_recovery()
        return self._state

    @property
    def is_open(self) -> bool:
        """Check if circuit is open (rejecting calls)."""
        return self.state == CircuitState.OPEN

    def _check_recovery(self) -> None:
        """Check if circuit should transition to half-open."""
        if self._state != CircuitState.OPEN:
            return

        if self._last_failure_time is None:
            return

        time_since_failure = (datetime.utcnow() - self._last_failure_time).total_seconds()
        if time_since_failure >= self.config.recovery_timeout:
            self._transition_to(CircuitState.HALF_OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state
        self._last_state_change = datetime.utcnow()

        self._logger.info(
            f"Circuit breaker '{self.name}' state change: "
            f"{old_state.value} -> {new_state.value}"
        )

        if new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

    def _record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED)
                self._failure_count = 0

        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def _record_failure(self, error: Exception) -> None:
        """Record a failed call."""
        # Check if this exception is excluded
        for exc_type in self.config.excluded_exceptions:
            if isinstance(error, exc_type):
                return

        # Check if this exception counts as a failure
        is_failure = False
        for exc_type in self.config.failure_exceptions:
            if isinstance(error, exc_type):
                is_failure = True
                break

        if not is_failure:
            return

        self._failure_count += 1
        self._last_failure_time = datetime.utcnow()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open goes back to open
            self._transition_to(CircuitState.OPEN)

        elif self._state == CircuitState.CLOSED:
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN)

    async def call(
        self,
        func: Callable[..., T],
        *args,
        fallback: Optional[Callable[..., T]] = None,
        **kwargs
    ) -> T:
        """
        Execute a function through the circuit breaker.

        Args:
            func: Function to execute
            *args: Positional arguments
            fallback: Optional fallback function if circuit is open
            **kwargs: Keyword arguments

        Returns:
            Result of func or fallback

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided
        """
        # Check circuit state
        state = self.state

        if state == CircuitState.OPEN:
            if fallback:
                self._logger.warning(
                    f"Circuit '{self.name}' is open, using fallback for {func.__name__}"
                )
                if asyncio.iscoroutinefunction(fallback):
                    return await fallback(*args, **kwargs)
                return fallback(*args, **kwargs)
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is open",
                breaker_name=self.name
            )

        # Execute function
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self._record_success()
            return result

        except Exception as e:
            self._record_failure(e)
            raise

    def get_status(self) -> dict:
        """Get circuit breaker status."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure': (
                self._last_failure_time.isoformat()
                if self._last_failure_time else None
            ),
            'last_state_change': self._last_state_change.isoformat(),
        }

    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._logger.info(f"Circuit breaker '{self.name}' manually reset")


class CircuitOpenError(TradingError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str, breaker_name: str = ""):
        super().__init__(message)
        self.breaker_name = breaker_name
