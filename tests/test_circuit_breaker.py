"""Tests for Sprint 9: Reliability & Circuit Breakers.

Tests cover:
  - Circuit breaker state transitions (CLOSED → OPEN → HALF_OPEN → CLOSED)
  - Failure threshold triggers OPEN
  - Recovery timeout triggers HALF_OPEN
  - Success threshold in HALF_OPEN closes circuit
  - Fallback execution when OPEN
  - CircuitOpenError raised when OPEN without fallback
  - Thread safety
  - Manual reset
  - Stats tracking
  - HealthChecker aggregation
"""

import threading
import time
from unittest.mock import MagicMock

import pytest

from src.intelligence.circuit_breaker import (
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    HealthChecker,
    HealthStatus,
)


# =============================================================================
# CIRCUIT BREAKER STATE TRANSITIONS
# =============================================================================

class TestCircuitBreakerStates:
    def test_initial_state_closed(self):
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED

    def test_success_keeps_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        cb.call(lambda: "ok")
        assert cb.state == CircuitState.CLOSED

    def test_failure_below_threshold_stays_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.CLOSED

    def test_failure_at_threshold_opens(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(3):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN

    def test_open_blocks_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=60)
        # Trigger OPEN
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN

        # Next call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError) as exc_info:
            cb.call(lambda: "should not run")
        assert exc_info.value.name == "test"
        assert exc_info.value.retry_after > 0

    def test_recovery_timeout_transitions_to_half_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=0.1)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN

        # Wait for recovery
        time.sleep(0.15)

        # Next call should transition to HALF_OPEN and execute
        result = cb.call(lambda: "recovered")
        assert result == "recovered"
        # After one success, still HALF_OPEN (need success_threshold=2)

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=0.1, success_threshold=2,
        )
        # Open circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.15)

        # Two successes should close
        cb.call(lambda: "ok1")
        cb.call(lambda: "ok2")
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(
            name="test", failure_threshold=2,
            recovery_timeout=0.1, success_threshold=2,
        )
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        time.sleep(0.15)

        # Failure in half-open → back to OPEN
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail again")))
        assert cb.state == CircuitState.OPEN


# =============================================================================
# FALLBACK
# =============================================================================

class TestCircuitBreakerFallback:
    def test_fallback_when_open(self):
        cb = CircuitBreaker(name="test", failure_threshold=2, recovery_timeout=60)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        result = cb.call(
            lambda: "should not run",
            fallback=lambda: "fallback_value",
        )
        assert result == "fallback_value"

    def test_fallback_not_used_when_closed(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        result = cb.call(
            lambda: "primary",
            fallback=lambda: "fallback",
        )
        assert result == "primary"


# =============================================================================
# MANUAL RESET
# =============================================================================

class TestManualReset:
    def test_reset_closes_circuit(self):
        cb = CircuitBreaker(name="test", failure_threshold=2)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.OPEN

        cb.reset()
        assert cb.state == CircuitState.CLOSED

    def test_reset_clears_failure_count(self):
        cb = CircuitBreaker(name="test", failure_threshold=3)
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        cb.reset()
        # Should need 3 more failures to open again
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
        assert cb.state == CircuitState.CLOSED


# =============================================================================
# STATS
# =============================================================================

class TestCircuitBreakerStats:
    def test_stats_initial(self):
        cb = CircuitBreaker(name="test_circuit")
        stats = cb.get_stats()
        assert stats["name"] == "test_circuit"
        assert stats["state"] == "closed"
        assert stats["total_calls"] == 0
        assert stats["total_failures"] == 0

    def test_stats_after_calls(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        cb.call(lambda: "ok")
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats["total_calls"] == 2
        assert stats["total_successes"] == 1
        assert stats["total_failures"] == 1
        assert stats["failure_rate"] == 50.0

    def test_stats_last_failure_age(self):
        cb = CircuitBreaker(name="test", failure_threshold=5)
        with pytest.raises(ValueError):
            cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        stats = cb.get_stats()
        assert stats["last_failure_age"] is not None
        assert stats["last_failure_age"] < 1.0  # Just happened


# =============================================================================
# THREAD SAFETY
# =============================================================================

class TestThreadSafety:
    def test_concurrent_calls(self):
        cb = CircuitBreaker(name="thread_test", failure_threshold=100)
        results = []
        errors = []

        def worker():
            try:
                r = cb.call(lambda: "ok")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 20
        assert cb.get_stats()["total_calls"] == 20

    def test_concurrent_failures(self):
        cb = CircuitBreaker(name="fail_test", failure_threshold=3)
        errors_caught = []

        def worker():
            try:
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))
            except (ValueError, CircuitOpenError) as e:
                errors_caught.append(type(e).__name__)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors_caught) == 10
        # After 3+ failures, circuit should be OPEN
        assert cb.state == CircuitState.OPEN


# =============================================================================
# HEALTH CHECKER
# =============================================================================

class TestHealthChecker:
    def test_all_healthy(self):
        hc = HealthChecker()
        hc.register("service_a", lambda: True)
        hc.register("service_b", lambda: True)

        status = hc.check()
        assert status.healthy is True
        assert len(status.checks) == 2

    def test_one_unhealthy(self):
        hc = HealthChecker()
        hc.register("service_a", lambda: True)
        hc.register("service_b", lambda: False)

        status = hc.check()
        assert status.healthy is False
        assert status.checks["service_a"]["healthy"] is True
        assert status.checks["service_b"]["healthy"] is False

    def test_exception_in_check(self):
        def bad_check():
            raise RuntimeError("check failed")

        hc = HealthChecker()
        hc.register("bad_service", bad_check)
        status = hc.check()

        assert status.healthy is False
        assert "error" in status.checks["bad_service"]

    def test_empty_checker_healthy(self):
        hc = HealthChecker()
        status = hc.check()
        assert status.healthy is True
        assert len(status.checks) == 0

    def test_to_dict(self):
        hc = HealthChecker()
        hc.register("test", lambda: True)
        status = hc.check()
        d = status.to_dict()

        assert isinstance(d, dict)
        assert "healthy" in d
        assert "checks" in d

    def test_circuit_breaker_integration(self):
        """Health check should reflect circuit breaker state."""
        cb = CircuitBreaker(name="llm", failure_threshold=2)
        hc = HealthChecker()
        hc.register("llm_api", lambda: cb.state != CircuitState.OPEN)

        # Initially healthy
        assert hc.check().healthy is True

        # Trip the circuit
        for _ in range(2):
            with pytest.raises(ValueError):
                cb.call(lambda: (_ for _ in ()).throw(ValueError("fail")))

        # Now unhealthy
        assert hc.check().healthy is False

        # Reset
        cb.reset()
        assert hc.check().healthy is True
