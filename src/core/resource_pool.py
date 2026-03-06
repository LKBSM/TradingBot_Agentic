# =============================================================================
# RESOURCE POOL
# =============================================================================
# Connection and resource pooling for the trading system.
#
# Features:
# - Generic resource pool with configurable size
# - Automatic health checking and recycling
# - Context manager support for safe resource usage
# - Metrics for monitoring pool utilization
#
# Usage:
#   pool = ResourcePool(factory=create_connection, max_size=5)
#
#   async with pool.acquire() as conn:
#       result = conn.query("SELECT 1")
#
#   pool.get_metrics()
#
# =============================================================================

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Callable, Coroutine, Generic, Optional, TypeVar,
    Dict, List
)
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# POOL CONFIGURATION
# =============================================================================

@dataclass
class PoolConfig:
    """Configuration for resource pool."""

    # Pool sizing
    min_size: int = 1
    max_size: int = 10

    # Timeouts
    acquire_timeout: float = 10.0       # Max wait to acquire a resource
    idle_timeout: float = 300.0         # Recycle idle resources after N seconds
    max_lifetime: float = 3600.0        # Recycle resources after N seconds total

    # Health checking
    health_check_interval: float = 60.0  # Check health every N seconds
    validate_on_acquire: bool = True     # Validate resource before returning

    # Retry
    max_create_retries: int = 3
    create_retry_delay: float = 1.0


# =============================================================================
# POOLED RESOURCE WRAPPER
# =============================================================================

@dataclass
class PooledResource(Generic[T]):
    """Wrapper around a pooled resource with lifecycle tracking."""

    resource: T
    created_at: float = field(default_factory=time.monotonic)
    last_used_at: float = field(default_factory=time.monotonic)
    last_validated_at: float = field(default_factory=time.monotonic)
    use_count: int = 0
    is_valid: bool = True
    resource_id: int = 0

    def mark_used(self) -> None:
        """Mark resource as used."""
        self.last_used_at = time.monotonic()
        self.use_count += 1

    def is_expired(self, max_lifetime: float) -> bool:
        """Check if resource has exceeded max lifetime."""
        return (time.monotonic() - self.created_at) > max_lifetime

    def is_idle(self, idle_timeout: float) -> bool:
        """Check if resource has been idle too long."""
        return (time.monotonic() - self.last_used_at) > idle_timeout

    def needs_validation(self, interval: float) -> bool:
        """Check if resource needs health check."""
        return (time.monotonic() - self.last_validated_at) > interval


# =============================================================================
# RESOURCE POOL
# =============================================================================

class ResourcePool(Generic[T]):
    """
    Generic async resource pool.

    Manages a pool of reusable resources (connections, clients, etc.)
    with automatic health checking, lifecycle management, and metrics.
    """

    def __init__(
        self,
        factory: Callable[[], Coroutine[Any, Any, T]],
        config: Optional[PoolConfig] = None,
        validator: Optional[Callable[[T], Coroutine[Any, Any, bool]]] = None,
        destructor: Optional[Callable[[T], Coroutine[Any, Any, None]]] = None,
        name: str = "pool"
    ):
        """
        Initialize resource pool.

        Args:
            factory: Async function that creates a new resource
            config: Pool configuration
            validator: Async function to validate a resource is healthy
            destructor: Async function to clean up a resource
            name: Pool name for logging/metrics
        """
        self._factory = factory
        self._config = config or PoolConfig()
        self._validator = validator
        self._destructor = destructor
        self._name = name

        # Pool storage
        self._available: asyncio.Queue = asyncio.Queue(
            maxsize=self._config.max_size
        )
        self._in_use: Dict[int, PooledResource[T]] = {}
        self._all_resources: Dict[int, PooledResource[T]] = {}
        self._resource_counter = 0

        # State
        self._lock = asyncio.Lock()
        self._closed = False
        self._health_task: Optional[asyncio.Task] = None

        # Metrics
        self._total_acquired = 0
        self._total_released = 0
        self._total_created = 0
        self._total_destroyed = 0
        self._total_failed_validations = 0
        self._total_acquire_timeouts = 0
        self._peak_in_use = 0

        self._logger = logging.getLogger(f"resource_pool.{name}")

    @property
    def size(self) -> int:
        """Total resources (available + in use)."""
        return len(self._all_resources)

    @property
    def available(self) -> int:
        """Number of available resources."""
        return self._available.qsize()

    @property
    def in_use(self) -> int:
        """Number of resources currently in use."""
        return len(self._in_use)

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """Start the pool and pre-create minimum resources."""
        if self._closed:
            raise RuntimeError(f"Pool '{self._name}' is closed")

        # Pre-create minimum resources
        for _ in range(self._config.min_size):
            try:
                resource = await self._create_resource()
                if resource:
                    await self._available.put(resource)
            except Exception as e:
                self._logger.error(f"Failed to pre-create resource: {e}")

        # Start health check loop
        self._health_task = asyncio.create_task(self._health_check_loop())

        self._logger.info(
            f"Pool '{self._name}' started with {self.size} resources"
        )

    async def close(self) -> None:
        """Close the pool and destroy all resources."""
        self._closed = True

        # Stop health check
        if self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        # Destroy all resources
        async with self._lock:
            for resource_id, pooled in list(self._all_resources.items()):
                await self._destroy_resource(pooled)

            self._all_resources.clear()
            self._in_use.clear()

        self._logger.info(f"Pool '{self._name}' closed")

    # =========================================================================
    # ACQUIRE / RELEASE
    # =========================================================================

    @asynccontextmanager
    async def acquire(self):
        """
        Acquire a resource from the pool (context manager).

        Usage:
            async with pool.acquire() as resource:
                result = resource.do_something()

        Yields:
            The underlying resource

        Raises:
            asyncio.TimeoutError: If acquire times out
            RuntimeError: If pool is closed
        """
        resource = await self.acquire_resource()
        try:
            yield resource.resource
        finally:
            await self.release_resource(resource)

    async def acquire_resource(self) -> PooledResource[T]:
        """
        Acquire a resource from the pool.

        Returns:
            PooledResource wrapper

        Raises:
            asyncio.TimeoutError: If acquire times out
            RuntimeError: If pool is closed
        """
        if self._closed:
            raise RuntimeError(f"Pool '{self._name}' is closed")

        start = time.monotonic()

        while True:
            elapsed = time.monotonic() - start
            remaining = self._config.acquire_timeout - elapsed

            if remaining <= 0:
                self._total_acquire_timeouts += 1
                raise asyncio.TimeoutError(
                    f"Pool '{self._name}' acquire timeout "
                    f"({self._config.acquire_timeout}s)"
                )

            # Try to get from available pool
            try:
                pooled = await asyncio.wait_for(
                    self._available.get(),
                    timeout=min(remaining, 1.0)
                )

                # Validate resource
                if self._config.validate_on_acquire:
                    if await self._validate_resource(pooled):
                        pooled.mark_used()
                        async with self._lock:
                            self._in_use[pooled.resource_id] = pooled
                            self._total_acquired += 1
                            self._peak_in_use = max(
                                self._peak_in_use, len(self._in_use)
                            )
                        return pooled
                    else:
                        # Resource invalid, destroy and try again
                        await self._destroy_resource(pooled)
                        continue
                else:
                    pooled.mark_used()
                    async with self._lock:
                        self._in_use[pooled.resource_id] = pooled
                        self._total_acquired += 1
                        self._peak_in_use = max(
                            self._peak_in_use, len(self._in_use)
                        )
                    return pooled

            except asyncio.TimeoutError:
                # No available resources, try to create one
                if self.size < self._config.max_size:
                    pooled = await self._create_resource()
                    if pooled:
                        pooled.mark_used()
                        async with self._lock:
                            self._in_use[pooled.resource_id] = pooled
                            self._total_acquired += 1
                            self._peak_in_use = max(
                                self._peak_in_use, len(self._in_use)
                            )
                        return pooled
                # Otherwise loop and wait

    async def release_resource(self, pooled: PooledResource[T]) -> None:
        """
        Release a resource back to the pool.

        Args:
            pooled: Resource to release
        """
        async with self._lock:
            self._in_use.pop(pooled.resource_id, None)
            self._total_released += 1

        # Check if resource should be recycled
        if pooled.is_expired(self._config.max_lifetime):
            await self._destroy_resource(pooled)
            return

        if not pooled.is_valid:
            await self._destroy_resource(pooled)
            return

        # Return to available pool
        try:
            self._available.put_nowait(pooled)
        except asyncio.QueueFull:
            self._logger.warning(
                f"Pool '{self._name}' queue full, "
                f"destroying excess resource #{pooled.resource_id}"
            )
            await self._destroy_resource(pooled)

    # =========================================================================
    # RESOURCE MANAGEMENT
    # =========================================================================

    async def _create_resource(self) -> Optional[PooledResource[T]]:
        """Create a new resource with retry."""
        for attempt in range(self._config.max_create_retries):
            try:
                resource = await self._factory()

                async with self._lock:
                    self._resource_counter += 1
                    pooled = PooledResource(
                        resource=resource,
                        resource_id=self._resource_counter
                    )
                    self._all_resources[pooled.resource_id] = pooled
                    self._total_created += 1

                self._logger.debug(
                    f"Created resource #{pooled.resource_id}"
                )
                return pooled

            except Exception as e:
                self._logger.warning(
                    f"Failed to create resource "
                    f"(attempt {attempt + 1}/{self._config.max_create_retries}): {e}"
                )
                if attempt < self._config.max_create_retries - 1:
                    await asyncio.sleep(self._config.create_retry_delay)

        self._logger.error("Failed to create resource after all retries")
        return None

    async def _destroy_resource(self, pooled: PooledResource[T]) -> None:
        """Destroy a resource and clean up."""
        pooled.is_valid = False

        try:
            if self._destructor:
                await self._destructor(pooled.resource)
        except Exception as e:
            self._logger.warning(
                f"Error destroying resource #{pooled.resource_id}: {e}"
            )

        async with self._lock:
            self._all_resources.pop(pooled.resource_id, None)
            self._total_destroyed += 1

        self._logger.debug(f"Destroyed resource #{pooled.resource_id}")

    async def _validate_resource(self, pooled: PooledResource[T]) -> bool:
        """Validate a resource is healthy."""
        # Check expiry
        if pooled.is_expired(self._config.max_lifetime):
            self._logger.debug(
                f"Resource #{pooled.resource_id} expired"
            )
            self._total_failed_validations += 1
            return False

        # Check idle timeout
        if pooled.is_idle(self._config.idle_timeout):
            self._logger.debug(
                f"Resource #{pooled.resource_id} idle too long"
            )
            self._total_failed_validations += 1
            return False

        # Custom validation
        if self._validator:
            try:
                valid = await self._validator(pooled.resource)
                if not valid:
                    self._total_failed_validations += 1
                pooled.last_validated_at = time.monotonic()
                return valid
            except Exception as e:
                self._logger.warning(
                    f"Validation failed for #{pooled.resource_id}: {e}"
                )
                self._total_failed_validations += 1
                return False

        return True

    # =========================================================================
    # HEALTH CHECK
    # =========================================================================

    async def _health_check_loop(self) -> None:
        """Background health check loop."""
        while not self._closed:
            try:
                await asyncio.sleep(self._config.health_check_interval)
                await asyncio.wait_for(
                    self._run_health_check(),
                    timeout=self._config.health_check_interval / 2
                )
            except asyncio.CancelledError:
                break
            except asyncio.TimeoutError:
                self._logger.error(
                    f"Health check timeout for pool '{self._name}'"
                )
            except Exception as e:
                self._logger.error(f"Health check error: {e}")

    async def _run_health_check(self) -> None:
        """Run health check on all available resources."""
        checked = 0
        removed = 0

        # Check available resources
        temp_resources: List[PooledResource[T]] = []
        while not self._available.empty():
            try:
                pooled = self._available.get_nowait()
                temp_resources.append(pooled)
            except asyncio.QueueEmpty:
                break

        for pooled in temp_resources:
            checked += 1
            if await self._validate_resource(pooled):
                try:
                    self._available.put_nowait(pooled)
                except asyncio.QueueFull:
                    await self._destroy_resource(pooled)
                    removed += 1
            else:
                await self._destroy_resource(pooled)
                removed += 1

        # Ensure minimum pool size
        while self.size < self._config.min_size:
            resource = await self._create_resource()
            if resource:
                try:
                    self._available.put_nowait(resource)
                except asyncio.QueueFull:
                    await self._destroy_resource(resource)
                    break
            else:
                break

        if removed > 0:
            self._logger.info(
                f"Health check: {checked} checked, "
                f"{removed} removed, {self.size} total"
            )

    # =========================================================================
    # METRICS
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """Get pool metrics."""
        return {
            'name': self._name,
            'size': self.size,
            'available': self.available,
            'in_use': self.in_use,
            'peak_in_use': self._peak_in_use,
            'total_created': self._total_created,
            'total_destroyed': self._total_destroyed,
            'total_acquired': self._total_acquired,
            'total_released': self._total_released,
            'total_acquire_timeouts': self._total_acquire_timeouts,
            'total_failed_validations': self._total_failed_validations,
            'utilization': (
                self.in_use / max(1, self.size)
            ),
            'is_closed': self._closed,
        }
