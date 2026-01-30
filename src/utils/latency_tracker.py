# =============================================================================
# LATENCY TRACKER - Performance Monitoring for Trading Operations
# =============================================================================
# High-precision latency measurement for all trading operations.
#
# Features:
#   - Nanosecond precision timing
#   - Percentile calculations (p50, p95, p99)
#   - Per-operation tracking
#   - Thread-safe operation
#   - Memory-bounded history
#   - Prometheus/StatsD export ready
#
# Usage:
#   tracker = LatencyTracker()
#
#   with tracker.track("mt5_order"):
#       result = mt5.order_send(request)
#
#   stats = tracker.get_stats("mt5_order")
#   print(f"p99: {stats.p99_ms}ms")
#
# =============================================================================

import time
import threading
import numpy as np
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from contextlib import contextmanager
from collections import defaultdict
from enum import Enum


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LatencyStats:
    """Statistics for a tracked operation."""
    operation: str
    count: int = 0
    total_ms: float = 0.0
    min_ms: float = float('inf')
    max_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p50_ms: float = 0.0
    p75_ms: float = 0.0
    p90_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    last_ms: float = 0.0
    samples: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'operation': self.operation,
            'count': self.count,
            'total_ms': round(self.total_ms, 3),
            'min_ms': round(self.min_ms, 3) if self.min_ms != float('inf') else None,
            'max_ms': round(self.max_ms, 3),
            'mean_ms': round(self.mean_ms, 3),
            'std_ms': round(self.std_ms, 3),
            'p50_ms': round(self.p50_ms, 3),
            'p75_ms': round(self.p75_ms, 3),
            'p90_ms': round(self.p90_ms, 3),
            'p95_ms': round(self.p95_ms, 3),
            'p99_ms': round(self.p99_ms, 3),
            'last_ms': round(self.last_ms, 3),
        }


class LatencyLevel(Enum):
    """Latency threshold levels."""
    EXCELLENT = "excellent"  # < 10ms
    GOOD = "good"            # 10-50ms
    ACCEPTABLE = "acceptable" # 50-100ms
    SLOW = "slow"            # 100-500ms
    CRITICAL = "critical"    # > 500ms

    @classmethod
    def from_latency(cls, latency_ms: float) -> 'LatencyLevel':
        if latency_ms < 10:
            return cls.EXCELLENT
        elif latency_ms < 50:
            return cls.GOOD
        elif latency_ms < 100:
            return cls.ACCEPTABLE
        elif latency_ms < 500:
            return cls.SLOW
        else:
            return cls.CRITICAL


# =============================================================================
# LATENCY TRACKER
# =============================================================================

class LatencyTracker:
    """
    High-precision latency tracking for trading operations.

    Tracks execution time for different operations and provides
    detailed statistics including percentiles.

    Example:
        tracker = LatencyTracker()

        # Context manager usage
        with tracker.track("inference"):
            action = model.predict(obs)

        with tracker.track("risk_check"):
            approved = risk_sentinel.evaluate(action)

        with tracker.track("execution"):
            result = mt5.order_send(request)

        # Get statistics
        stats = tracker.get_all_stats()
        for op, stat in stats.items():
            print(f"{op}: p99={stat.p99_ms:.1f}ms")

        # Check for slow operations
        alerts = tracker.check_thresholds({
            "execution": 100.0,  # Alert if execution > 100ms
            "inference": 10.0,   # Alert if inference > 10ms
        })
    """

    def __init__(
        self,
        max_samples: int = 10000,
        enable_detailed_tracking: bool = True
    ):
        """
        Initialize latency tracker.

        Args:
            max_samples: Maximum samples to keep per operation
            enable_detailed_tracking: Store individual samples for percentiles
        """
        self.max_samples = max_samples
        self.enable_detailed_tracking = enable_detailed_tracking
        self._lock = threading.RLock()

        # Per-operation data
        self._samples: Dict[str, List[float]] = defaultdict(list)
        self._counts: Dict[str, int] = defaultdict(int)
        self._totals: Dict[str, float] = defaultdict(float)
        self._mins: Dict[str, float] = defaultdict(lambda: float('inf'))
        self._maxs: Dict[str, float] = defaultdict(float)
        self._lasts: Dict[str, float] = defaultdict(float)

        # Callbacks for threshold alerts
        self._threshold_callbacks: List[Callable[[str, float, float], None]] = []

    # =========================================================================
    # TRACKING API
    # =========================================================================

    @contextmanager
    def track(self, operation: str):
        """
        Context manager to track operation latency.

        Args:
            operation: Name of the operation being tracked

        Example:
            with tracker.track("mt5_order"):
                result = mt5.order_send(request)
        """
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_ns = time.perf_counter_ns() - start
            elapsed_ms = elapsed_ns / 1_000_000
            self._record(operation, elapsed_ms)

    def record(self, operation: str, latency_ms: float) -> None:
        """
        Manually record a latency measurement.

        Args:
            operation: Operation name
            latency_ms: Latency in milliseconds
        """
        self._record(operation, latency_ms)

    def _record(self, operation: str, latency_ms: float) -> None:
        """Internal recording method."""
        with self._lock:
            self._counts[operation] += 1
            self._totals[operation] += latency_ms
            self._mins[operation] = min(self._mins[operation], latency_ms)
            self._maxs[operation] = max(self._maxs[operation], latency_ms)
            self._lasts[operation] = latency_ms

            if self.enable_detailed_tracking:
                samples = self._samples[operation]
                samples.append(latency_ms)

                # Trim to max_samples
                if len(samples) > self.max_samples:
                    self._samples[operation] = samples[-self.max_samples:]

    def start_timer(self, operation: str) -> 'Timer':
        """
        Start a timer for manual control.

        Returns:
            Timer object with stop() method

        Example:
            timer = tracker.start_timer("complex_operation")
            # ... do work ...
            timer.stop()
        """
        return Timer(self, operation)

    # =========================================================================
    # STATISTICS API
    # =========================================================================

    def get_stats(self, operation: str) -> LatencyStats:
        """
        Get statistics for a specific operation.

        Args:
            operation: Operation name

        Returns:
            LatencyStats with all metrics
        """
        with self._lock:
            count = self._counts.get(operation, 0)
            if count == 0:
                return LatencyStats(operation=operation)

            total = self._totals[operation]
            samples = self._samples.get(operation, [])

            stats = LatencyStats(
                operation=operation,
                count=count,
                total_ms=total,
                min_ms=self._mins[operation],
                max_ms=self._maxs[operation],
                mean_ms=total / count,
                last_ms=self._lasts[operation],
                samples=len(samples),
            )

            if samples:
                arr = np.array(samples)
                stats.std_ms = float(np.std(arr))
                stats.p50_ms = float(np.percentile(arr, 50))
                stats.p75_ms = float(np.percentile(arr, 75))
                stats.p90_ms = float(np.percentile(arr, 90))
                stats.p95_ms = float(np.percentile(arr, 95))
                stats.p99_ms = float(np.percentile(arr, 99))

            return stats

    def get_all_stats(self) -> Dict[str, LatencyStats]:
        """Get statistics for all tracked operations."""
        with self._lock:
            operations = set(self._counts.keys())

        return {op: self.get_stats(op) for op in operations}

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all operations."""
        all_stats = self.get_all_stats()

        summary = {
            'total_operations': len(all_stats),
            'total_measurements': sum(s.count for s in all_stats.values()),
            'operations': {}
        }

        for op, stats in all_stats.items():
            level = LatencyLevel.from_latency(stats.p99_ms)
            summary['operations'][op] = {
                'count': stats.count,
                'p50_ms': stats.p50_ms,
                'p99_ms': stats.p99_ms,
                'level': level.value,
            }

        return summary

    # =========================================================================
    # THRESHOLD MONITORING
    # =========================================================================

    def check_thresholds(
        self,
        thresholds: Dict[str, float],
        percentile: str = 'p99_ms'
    ) -> List[Dict[str, Any]]:
        """
        Check if any operations exceed thresholds.

        Args:
            thresholds: Dict of operation -> max acceptable latency (ms)
            percentile: Which percentile to check (p50_ms, p95_ms, p99_ms)

        Returns:
            List of threshold violations
        """
        violations = []
        all_stats = self.get_all_stats()

        for operation, threshold in thresholds.items():
            if operation in all_stats:
                stats = all_stats[operation]
                actual = getattr(stats, percentile, 0.0)

                if actual > threshold:
                    violations.append({
                        'operation': operation,
                        'threshold_ms': threshold,
                        'actual_ms': actual,
                        'percentile': percentile,
                        'exceeded_by_ms': actual - threshold,
                        'level': LatencyLevel.from_latency(actual).value,
                    })

        return violations

    def register_threshold_callback(
        self,
        callback: Callable[[str, float, float], None]
    ) -> None:
        """
        Register callback for threshold violations.

        Callback receives: (operation, threshold_ms, actual_ms)
        """
        self._threshold_callbacks.append(callback)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def reset(self, operation: Optional[str] = None) -> None:
        """
        Reset statistics.

        Args:
            operation: Specific operation to reset (None = all)
        """
        with self._lock:
            if operation:
                self._samples.pop(operation, None)
                self._counts.pop(operation, None)
                self._totals.pop(operation, None)
                self._mins.pop(operation, None)
                self._maxs.pop(operation, None)
                self._lasts.pop(operation, None)
            else:
                self._samples.clear()
                self._counts.clear()
                self._totals.clear()
                self._mins.clear()
                self._maxs.clear()
                self._lasts.clear()

    def get_operations(self) -> List[str]:
        """Get list of tracked operations."""
        with self._lock:
            return list(self._counts.keys())

    # =========================================================================
    # EXPORT FORMATS
    # =========================================================================

    def to_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        all_stats = self.get_all_stats()

        for op, stats in all_stats.items():
            safe_op = op.replace('.', '_').replace('-', '_')

            lines.append(f'trading_latency_count{{operation="{op}"}} {stats.count}')
            lines.append(f'trading_latency_sum_ms{{operation="{op}"}} {stats.total_ms:.3f}')
            lines.append(f'trading_latency_p50_ms{{operation="{op}"}} {stats.p50_ms:.3f}')
            lines.append(f'trading_latency_p95_ms{{operation="{op}"}} {stats.p95_ms:.3f}')
            lines.append(f'trading_latency_p99_ms{{operation="{op}"}} {stats.p99_ms:.3f}')

        return '\n'.join(lines)

    def to_statsd(self) -> List[str]:
        """Export metrics in StatsD format."""
        lines = []
        all_stats = self.get_all_stats()

        for op, stats in all_stats.items():
            prefix = f"trading.latency.{op}"
            lines.append(f"{prefix}.count:{stats.count}|c")
            lines.append(f"{prefix}.p50:{stats.p50_ms:.3f}|ms")
            lines.append(f"{prefix}.p95:{stats.p95_ms:.3f}|ms")
            lines.append(f"{prefix}.p99:{stats.p99_ms:.3f}|ms")

        return lines


# =============================================================================
# TIMER HELPER
# =============================================================================

class Timer:
    """Manual timer for tracking latency."""

    def __init__(self, tracker: LatencyTracker, operation: str):
        self.tracker = tracker
        self.operation = operation
        self.start_time = time.perf_counter_ns()
        self._stopped = False

    def stop(self) -> float:
        """Stop timer and record latency. Returns latency in ms."""
        if self._stopped:
            return 0.0

        elapsed_ns = time.perf_counter_ns() - self.start_time
        elapsed_ms = elapsed_ns / 1_000_000
        self.tracker.record(self.operation, elapsed_ms)
        self._stopped = True
        return elapsed_ms

    def elapsed_ms(self) -> float:
        """Get elapsed time without stopping."""
        elapsed_ns = time.perf_counter_ns() - self.start_time
        return elapsed_ns / 1_000_000


# =============================================================================
# GLOBAL TRACKER INSTANCE
# =============================================================================

_global_tracker: Optional[LatencyTracker] = None


def get_tracker() -> LatencyTracker:
    """Get or create global latency tracker."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = LatencyTracker()
    return _global_tracker


def track(operation: str):
    """Convenience decorator/context manager using global tracker."""
    return get_tracker().track(operation)


# =============================================================================
# DECORATOR
# =============================================================================

def track_latency(operation: Optional[str] = None):
    """
    Decorator to track function latency.

    Args:
        operation: Operation name (defaults to function name)

    Example:
        @track_latency("model_inference")
        def predict(self, obs):
            return self.model.predict(obs)
    """
    def decorator(func):
        op_name = operation or func.__name__

        def wrapper(*args, **kwargs):
            with get_tracker().track(op_name):
                return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
