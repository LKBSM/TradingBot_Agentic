# =============================================================================
# HEALTH CHECKS - System Health Monitoring
# =============================================================================
# Health check infrastructure for the trading system.
#
# Features:
# - Pluggable health checks (database, agents, broker, data feed)
# - Liveness and readiness probes (Kubernetes-compatible)
# - Aggregated health status
# - Health history tracking
# - Automatic degradation detection
#
# Usage:
#   monitor = HealthMonitor()
#
#   # Register checks
#   monitor.register("database", check_database)
#   monitor.register("broker", check_broker, critical=True)
#   monitor.register("agents", check_agents)
#
#   # Run all checks
#   status = await monitor.check_all()
#   print(status.is_healthy)  # True/False
#
#   # Liveness (is the system alive?)
#   live = await monitor.liveness()
#
#   # Readiness (is the system ready to trade?)
#   ready = await monitor.readiness()
#
# =============================================================================

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import (
    Any, Callable, Coroutine, Dict, List, Optional, Tuple
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str = ""
    latency_ms: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    is_critical: bool = False

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_degraded(self) -> bool:
        return self.status == HealthStatus.DEGRADED

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'status': self.status.value,
            'message': self.message,
            'latency_ms': round(self.latency_ms, 2),
            'details': self.details,
            'timestamp': self.timestamp.isoformat(),
            'is_critical': self.is_critical,
        }


@dataclass
class AggregateHealth:
    """Aggregated health status from all checks."""
    status: HealthStatus
    checks: List[HealthResult]
    total_latency_ms: float = 0.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    @property
    def is_healthy(self) -> bool:
        return self.status == HealthStatus.HEALTHY

    @property
    def is_ready(self) -> bool:
        """System is ready if no critical checks are unhealthy."""
        return not any(
            c.is_critical and not c.is_healthy
            for c in self.checks
        )

    @property
    def healthy_count(self) -> int:
        return sum(1 for c in self.checks if c.is_healthy)

    @property
    def unhealthy_count(self) -> int:
        return sum(
            1 for c in self.checks
            if c.status == HealthStatus.UNHEALTHY
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            'status': self.status.value,
            'is_ready': self.is_ready,
            'total_latency_ms': round(self.total_latency_ms, 2),
            'healthy': self.healthy_count,
            'total': len(self.checks),
            'checks': [c.to_dict() for c in self.checks],
            'timestamp': self.timestamp.isoformat(),
        }


# =============================================================================
# TYPE ALIAS
# =============================================================================

# A health check is an async function returning HealthResult
HealthCheck = Callable[[], Coroutine[Any, Any, HealthResult]]


# =============================================================================
# HEALTH MONITOR
# =============================================================================

class HealthMonitor:
    """
    Central health monitoring system.

    Manages registered health checks and provides aggregated status.
    """

    def __init__(
        self,
        name: str = "trading_system",
        check_timeout: float = 10.0,
        history_size: int = 100,
    ):
        """
        Initialize health monitor.

        Args:
            name: System name
            check_timeout: Timeout for each health check (seconds)
            history_size: Number of past results to keep
        """
        self._name = name
        self._check_timeout = check_timeout
        self._history_size = history_size

        # Registered checks
        self._checks: Dict[str, Tuple[HealthCheck, bool]] = {}

        # History
        self._history: List[AggregateHealth] = []

        # State
        self._last_result: Optional[AggregateHealth] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False

        self._logger = logging.getLogger(f"health_monitor.{name}")

    # =========================================================================
    # REGISTRATION
    # =========================================================================

    def register(
        self,
        name: str,
        check: HealthCheck,
        critical: bool = False,
    ) -> None:
        """
        Register a health check.

        Args:
            name: Check name
            check: Async function returning HealthResult
            critical: If True, failure means system is not ready
        """
        self._checks[name] = (check, critical)
        self._logger.debug(
            f"Registered health check: {name} "
            f"(critical={critical})"
        )

    def unregister(self, name: str) -> None:
        """Unregister a health check."""
        self._checks.pop(name, None)

    # =========================================================================
    # CHECK EXECUTION
    # =========================================================================

    async def check_all(self) -> AggregateHealth:
        """
        Run all registered health checks.

        Returns:
            AggregateHealth with all results
        """
        start = time.monotonic()
        results: List[HealthResult] = []

        # Run all checks concurrently
        tasks = {}
        for name, (check, critical) in self._checks.items():
            tasks[name] = asyncio.create_task(
                self._run_check(name, check, critical)
            )

        for name, task in tasks.items():
            try:
                result = await asyncio.wait_for(
                    task, timeout=self._check_timeout
                )
                results.append(result)
            except asyncio.TimeoutError:
                _, critical = self._checks[name]
                results.append(HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out ({self._check_timeout}s)",
                    latency_ms=self._check_timeout * 1000,
                    is_critical=critical,
                ))
            except Exception as e:
                _, critical = self._checks[name]
                results.append(HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check error: {e}",
                    is_critical=critical,
                ))

        # Determine aggregate status
        total_ms = (time.monotonic() - start) * 1000

        aggregate_status = self._compute_aggregate_status(results)

        aggregate = AggregateHealth(
            status=aggregate_status,
            checks=results,
            total_latency_ms=total_ms,
        )

        # Store result
        self._last_result = aggregate
        self._history.append(aggregate)
        if len(self._history) > self._history_size:
            self._history.pop(0)

        return aggregate

    async def _run_check(
        self,
        name: str,
        check: HealthCheck,
        critical: bool
    ) -> HealthResult:
        """Run a single health check with timing."""
        start = time.monotonic()
        try:
            result = await check()
            result.latency_ms = (time.monotonic() - start) * 1000
            result.is_critical = critical
            return result
        except Exception as e:
            elapsed = (time.monotonic() - start) * 1000
            return HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
                latency_ms=elapsed,
                is_critical=critical,
            )

    def _compute_aggregate_status(
        self,
        results: List[HealthResult]
    ) -> HealthStatus:
        """Compute overall health status from individual results."""
        if not results:
            return HealthStatus.UNKNOWN

        # Any critical unhealthy -> UNHEALTHY
        if any(
            r.status == HealthStatus.UNHEALTHY and r.is_critical
            for r in results
        ):
            return HealthStatus.UNHEALTHY

        # Any unhealthy (non-critical) -> DEGRADED
        if any(r.status == HealthStatus.UNHEALTHY for r in results):
            return HealthStatus.DEGRADED

        # Any degraded -> DEGRADED
        if any(r.status == HealthStatus.DEGRADED for r in results):
            return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    # =========================================================================
    # KUBERNETES-COMPATIBLE PROBES
    # =========================================================================

    async def liveness(self) -> bool:
        """
        Liveness probe: Is the system alive?

        Returns True if the process is running and responsive.
        Does NOT check external dependencies.
        """
        return True  # If we can execute this, we're alive

    async def readiness(self) -> bool:
        """
        Readiness probe: Is the system ready to accept trades?

        Runs all critical health checks and returns True
        only if all critical checks pass.
        """
        result = await self.check_all()
        return result.is_ready

    async def startup(self) -> bool:
        """
        Startup probe: Has the system finished initializing?

        Returns True if all checks have been run at least once.
        """
        return self._last_result is not None

    # =========================================================================
    # CONTINUOUS MONITORING
    # =========================================================================

    async def start_monitoring(
        self,
        interval: float = 60.0,
        on_degraded: Optional[Callable] = None,
        on_unhealthy: Optional[Callable] = None,
    ) -> None:
        """
        Start continuous health monitoring.

        Args:
            interval: Check interval in seconds
            on_degraded: Callback when system becomes degraded
            on_unhealthy: Callback when system becomes unhealthy
        """
        self._running = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(interval, on_degraded, on_unhealthy)
        )
        self._logger.info(
            f"Health monitoring started (interval={interval}s)"
        )

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        self._logger.info("Health monitoring stopped")

    async def _monitor_loop(
        self,
        interval: float,
        on_degraded: Optional[Callable],
        on_unhealthy: Optional[Callable],
    ) -> None:
        """Background monitoring loop."""
        prev_status = HealthStatus.UNKNOWN

        while self._running:
            try:
                result = await self.check_all()

                # Detect status transitions
                if result.status != prev_status:
                    self._logger.info(
                        f"Health status changed: {prev_status.value} -> "
                        f"{result.status.value}"
                    )

                    if result.status == HealthStatus.DEGRADED and on_degraded:
                        try:
                            await on_degraded(result)
                        except Exception as e:
                            self._logger.error(
                                f"Degraded callback error: {e}"
                            )

                    if result.status == HealthStatus.UNHEALTHY and on_unhealthy:
                        try:
                            await on_unhealthy(result)
                        except Exception as e:
                            self._logger.error(
                                f"Unhealthy callback error: {e}"
                            )

                prev_status = result.status

            except Exception as e:
                self._logger.error(f"Monitor loop error: {e}")

            await asyncio.sleep(interval)

    # =========================================================================
    # STATUS API
    # =========================================================================

    @property
    def last_result(self) -> Optional[AggregateHealth]:
        """Get the most recent health check result."""
        return self._last_result

    def get_history(
        self,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get recent health check history."""
        return [
            h.to_dict() for h in self._history[-limit:]
        ]

    def get_dashboard(self) -> str:
        """Get text-based health dashboard."""
        if not self._last_result:
            return "No health checks have been run yet."

        result = self._last_result
        status_icon = {
            HealthStatus.HEALTHY: "[OK]",
            HealthStatus.DEGRADED: "[!!]",
            HealthStatus.UNHEALTHY: "[XX]",
            HealthStatus.UNKNOWN: "[??]",
        }

        lines = [
            "=" * 60,
            f"  HEALTH STATUS: {result.status.value.upper()}",
            f"  Time: {result.timestamp.isoformat()}",
            f"  Latency: {result.total_latency_ms:.1f}ms",
            f"  Ready: {'YES' if result.is_ready else 'NO'}",
            "=" * 60,
        ]

        for check in result.checks:
            icon = status_icon.get(check.status, "[??]")
            crit = "*" if check.is_critical else " "
            lines.append(
                f"  {icon}{crit} {check.name:25} "
                f"{check.latency_ms:7.1f}ms  {check.message}"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# PRE-BUILT HEALTH CHECKS
# =============================================================================

def create_agent_health_check(
    orchestrator: Any,
    name: str = "agents"
) -> HealthCheck:
    """
    Create a health check for the agent orchestrator.

    Args:
        orchestrator: TradingOrchestrator instance
        name: Check name

    Returns:
        Async health check function
    """
    async def check() -> HealthResult:
        try:
            status = orchestrator.get_status()
            failed = status.get('failed_agents', [])
            total = status.get('total_agents', 0)
            running = total - len(failed)

            if failed:
                return HealthResult(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    message=f"{len(failed)}/{total} agents failed: {failed}",
                    details={
                        'total': total,
                        'running': running,
                        'failed': failed,
                    }
                )

            return HealthResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message=f"All {total} agents running",
                details={'total': total, 'running': running}
            )
        except Exception as e:
            return HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


def create_kill_switch_health_check(
    kill_switch: Any,
    name: str = "kill_switch"
) -> HealthCheck:
    """
    Create a health check for the kill switch.

    Args:
        kill_switch: KillSwitch instance
        name: Check name

    Returns:
        Async health check function
    """
    async def check() -> HealthResult:
        try:
            is_halted = kill_switch.is_halted
            details = {'is_halted': is_halted}

            if is_halted:
                reason = getattr(kill_switch, 'halt_reason', 'Unknown')
                return HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Kill switch triggered: {reason}",
                    details=details,
                )

            return HealthResult(
                name=name,
                status=HealthStatus.HEALTHY,
                message="Kill switch not triggered",
                details=details,
            )
        except Exception as e:
            return HealthResult(
                name=name,
                status=HealthStatus.UNHEALTHY,
                message=str(e),
            )

    return check


def create_memory_health_check(
    warn_threshold_mb: float = 500,
    critical_threshold_mb: float = 1000,
    name: str = "memory"
) -> HealthCheck:
    """
    Create a health check for memory usage.

    Args:
        warn_threshold_mb: Warning threshold in MB
        critical_threshold_mb: Critical threshold in MB
        name: Check name

    Returns:
        Async health check function
    """
    async def check() -> HealthResult:
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)

            if memory_mb > critical_threshold_mb:
                return HealthResult(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Memory critical: {memory_mb:.0f}MB",
                    details={'memory_mb': memory_mb},
                )
            elif memory_mb > warn_threshold_mb:
                return HealthResult(
                    name=name,
                    status=HealthStatus.DEGRADED,
                    message=f"Memory high: {memory_mb:.0f}MB",
                    details={'memory_mb': memory_mb},
                )
            else:
                return HealthResult(
                    name=name,
                    status=HealthStatus.HEALTHY,
                    message=f"Memory OK: {memory_mb:.0f}MB",
                    details={'memory_mb': memory_mb},
                )
        except ImportError:
            return HealthResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message="psutil not installed",
            )
        except Exception as e:
            return HealthResult(
                name=name,
                status=HealthStatus.UNKNOWN,
                message=str(e),
            )

    return check
