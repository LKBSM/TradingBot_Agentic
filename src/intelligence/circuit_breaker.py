"""Circuit breaker pattern for external service calls.

Protects against cascading failures when services (LLM API, Telegram, MT5)
are unavailable. Three states: CLOSED → OPEN → HALF_OPEN → CLOSED.

Usage:
    breaker = CircuitBreaker(name="llm_api", failure_threshold=3, recovery_timeout=60)

    try:
        breaker.call(lambda: api_call())
    except CircuitOpenError:
        # Service circuit is open — use fallback
        ...
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitState(str, Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Blocking calls
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open and call is blocked."""
    def __init__(self, name: str, retry_after: float):
        self.name = name
        self.retry_after = retry_after
        super().__init__(
            f"Circuit '{name}' is OPEN. Retry after {retry_after:.1f}s."
        )


@dataclass
class CircuitBreaker:
    """Thread-safe circuit breaker for external service protection.

    Args:
        name: Human-readable name for logging.
        failure_threshold: Consecutive failures before opening circuit.
        recovery_timeout: Seconds before attempting recovery (half-open).
        success_threshold: Consecutive successes in half-open to close circuit.
        max_history: Max events in sliding window (bounded memory).
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 2
    max_history: int = 100

    # Internal state (not exposed in __init__)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False, repr=False)
    _consecutive_failures: int = field(default=0, init=False, repr=False)
    _consecutive_successes: int = field(default=0, init=False, repr=False)
    _last_failure_time: float = field(default=0.0, init=False, repr=False)
    _total_calls: int = field(default=0, init=False, repr=False)
    _total_failures: int = field(default=0, init=False, repr=False)
    _total_successes: int = field(default=0, init=False, repr=False)
    _history: deque = field(default_factory=lambda: deque(maxlen=100), init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def __post_init__(self):
        self._history = deque(maxlen=self.max_history)

    @property
    def state(self) -> CircuitState:
        with self._lock:
            return self._state

    def call(self, func: Callable[[], T], fallback: Optional[Callable[[], T]] = None) -> T:
        """Execute a function through the circuit breaker.

        Args:
            func: The function to call.
            fallback: Optional fallback function if circuit is open.

        Returns:
            Result of func() or fallback().

        Raises:
            CircuitOpenError: If circuit is open and no fallback provided.
        """
        with self._lock:
            self._total_calls += 1

            if self._state == CircuitState.OPEN:
                if self._should_attempt_recovery():
                    self._state = CircuitState.HALF_OPEN
                    logger.info("Circuit '%s' → HALF_OPEN (testing recovery)", self.name)
                else:
                    retry_after = self.recovery_timeout - (time.time() - self._last_failure_time)
                    if fallback is not None:
                        return fallback()
                    raise CircuitOpenError(self.name, max(0, retry_after))

        # Execute outside lock
        try:
            result = func()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure(e)
            raise

    def _on_success(self) -> None:
        with self._lock:
            self._consecutive_failures = 0
            self._consecutive_successes += 1
            self._total_successes += 1
            self._history.append(("success", time.time()))

            if self._state == CircuitState.HALF_OPEN:
                if self._consecutive_successes >= self.success_threshold:
                    self._state = CircuitState.CLOSED
                    self._consecutive_successes = 0
                    logger.info("Circuit '%s' → CLOSED (recovered)", self.name)

    def _on_failure(self, error: Exception) -> None:
        with self._lock:
            self._consecutive_failures += 1
            self._consecutive_successes = 0
            self._total_failures += 1
            self._last_failure_time = time.time()
            self._history.append(("failure", time.time(), str(error)))

            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.OPEN
                logger.warning(
                    "Circuit '%s' → OPEN (recovery failed: %s)", self.name, error
                )
            elif self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self.failure_threshold:
                    self._state = CircuitState.OPEN
                    logger.warning(
                        "Circuit '%s' → OPEN (%d consecutive failures)",
                        self.name, self._consecutive_failures,
                    )

    def _should_attempt_recovery(self) -> bool:
        return (time.time() - self._last_failure_time) >= self.recovery_timeout

    def reset(self) -> None:
        """Manually reset circuit to CLOSED state."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._consecutive_successes = 0
            logger.info("Circuit '%s' manually reset to CLOSED", self.name)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "consecutive_failures": self._consecutive_failures,
                "total_calls": self._total_calls,
                "total_failures": self._total_failures,
                "total_successes": self._total_successes,
                "failure_rate": (
                    self._total_failures / max(self._total_calls, 1) * 100
                ),
                "last_failure_age": (
                    round(time.time() - self._last_failure_time, 1)
                    if self._last_failure_time > 0 else None
                ),
            }


# =============================================================================
# HEALTH CHECK
# =============================================================================

@dataclass
class HealthStatus:
    """Aggregated system health status."""
    healthy: bool
    checks: Dict[str, Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "healthy": self.healthy,
            "checks": self.checks,
        }


class HealthChecker:
    """Aggregated health checker for all subsystems.

    Usage:
        checker = HealthChecker()
        checker.register("llm_api", lambda: circuit.get_stats()["state"] == "closed")
        checker.register("data_provider", lambda: provider.available_symbols() != [])
        status = checker.check()
    """

    def __init__(self):
        self._checks: Dict[str, Callable[[], bool]] = {}

    def register(self, name: str, check_fn: Callable[[], bool]) -> None:
        """Register a named health check function."""
        self._checks[name] = check_fn

    def check(self) -> HealthStatus:
        """Run all health checks and return aggregated status."""
        results: Dict[str, Dict[str, Any]] = {}
        all_healthy = True

        for name, check_fn in self._checks.items():
            try:
                healthy = check_fn()
                results[name] = {"healthy": healthy}
                if not healthy:
                    all_healthy = False
            except Exception as e:
                results[name] = {"healthy": False, "error": str(e)}
                all_healthy = False

        return HealthStatus(healthy=all_healthy, checks=results)
