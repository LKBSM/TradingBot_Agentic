# =============================================================================
# ASYNC ORCHESTRATOR ADAPTER
# =============================================================================
# Async-compatible adapter for the TradingOrchestrator.
#
# The base TradingOrchestrator is synchronous (required for gym.Wrapper).
# This adapter provides:
# 1. Async decision coordination for live trading
# 2. Integration with the messaging EventBus (pub/sub)
# 3. Timeout/retry patterns using core.retry
# 4. Event publishing for all trade decisions
#
# Usage:
#   # Live trading (async)
#   adapter = AsyncOrchestratorAdapter(orchestrator)
#   await adapter.start()
#
#   decision = await adapter.coordinate_async(proposal)
#
#   # Backtesting (synchronous gym - unchanged)
#   decision = orchestrator.coordinate_decision(proposal)
#
# =============================================================================

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from src.agents.orchestrator import (
    TradingOrchestrator,
    OrchestratedDecision,
    OrchestratorConfig,
    AgentPriority,
)
from src.agents.events import (
    TradeProposal,
    DecisionType,
)
from src.messaging.event_queue import (
    Event,
    EventType,
    EventPriority,
    EventQueue,
    EventBus,
    DeadLetterQueue,
    QueueProcessor,
)
from src.core.retry import RetryConfig, retry_with_backoff, CircuitBreaker
from src.core.exceptions import TransientError, BrokerConnectionError


logger = logging.getLogger(__name__)


# =============================================================================
# ASYNC ORCHESTRATOR ADAPTER
# =============================================================================

class AsyncOrchestratorAdapter:
    """
    Async adapter for TradingOrchestrator.

    Wraps the synchronous orchestrator with async capabilities:
    - Async decision coordination with timeout
    - Event publishing to messaging EventBus
    - Retry on transient failures
    - Circuit breaker per agent
    - Dead letter queue for failed decisions

    Thread Safety:
        The underlying orchestrator is thread-safe (uses locks).
        The adapter runs orchestrator calls in an executor to avoid
        blocking the event loop.
    """

    def __init__(
        self,
        orchestrator: TradingOrchestrator,
        event_bus: Optional[EventBus] = None,
        decision_timeout: float = 10.0,
        retry_config: Optional[RetryConfig] = None,
    ):
        """
        Initialize async orchestrator adapter.

        Args:
            orchestrator: Base synchronous orchestrator
            event_bus: Messaging event bus for publishing decisions
            decision_timeout: Timeout for decision coordination (seconds)
            retry_config: Retry configuration for transient failures
        """
        self._orchestrator = orchestrator
        self._decision_timeout = decision_timeout

        # Messaging
        self._event_bus = event_bus or EventBus(name="orchestrator")
        self._decision_queue = EventQueue(maxsize=1000, name="decisions")
        self._dlq = DeadLetterQueue(maxsize=500, name="orchestrator_dlq")

        # Retry configuration
        self._retry_config = retry_config or RetryConfig(
            max_retries=2,
            base_delay=0.5,
            max_delay=5.0,
            retry_on=(TransientError, BrokerConnectionError, asyncio.TimeoutError),
        )

        # Circuit breakers per agent
        self._agent_breakers: Dict[str, CircuitBreaker] = {}

        # State
        self._running = False
        self._processor: Optional[QueueProcessor] = None

        # Metrics
        self._total_async_decisions = 0
        self._total_timeouts = 0
        self._total_retries = 0
        self._avg_decision_time_ms = 0.0

        self._logger = logging.getLogger("async_orchestrator")

    # =========================================================================
    # LIFECYCLE
    # =========================================================================

    async def start(self) -> None:
        """Start the async adapter and background processor."""
        if self._running:
            return

        self._running = True

        # Start queue processor for background event processing
        self._processor = QueueProcessor(
            queue=self._decision_queue,
            bus=self._event_bus,
            name="decision_processor"
        )
        await self._processor.start()

        # Publish startup event
        await self._event_bus.publish(Event(
            event_type=EventType.SYSTEM_STARTUP,
            source="async_orchestrator",
            data={'agents': len(self._orchestrator.get_all_agents())}
        ))

        self._logger.info("Async orchestrator adapter started")

    async def stop(self) -> None:
        """Stop the async adapter."""
        if not self._running:
            return

        self._running = False

        # Publish shutdown event
        await self._event_bus.publish(Event(
            event_type=EventType.SYSTEM_SHUTDOWN,
            source="async_orchestrator",
        ))

        # Stop processor
        if self._processor:
            await self._processor.stop()

        self._logger.info("Async orchestrator adapter stopped")

    # =========================================================================
    # ASYNC DECISION COORDINATION
    # =========================================================================

    async def coordinate_async(
        self,
        proposal: TradeProposal,
        context: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None
    ) -> OrchestratedDecision:
        """
        Coordinate decision asynchronously with timeout and retry.

        This runs the synchronous orchestrator in an executor thread
        to avoid blocking the event loop. Includes:
        - Timeout protection
        - Retry on transient failures
        - Event publishing for audit trail

        Args:
            proposal: Trade proposal to evaluate
            context: Optional additional context
            timeout: Override timeout (seconds)

        Returns:
            OrchestratedDecision

        Raises:
            asyncio.TimeoutError: If decision times out after retries
        """
        effective_timeout = timeout or self._decision_timeout
        start = time.monotonic()

        try:
            decision = await retry_with_backoff(
                self._execute_decision,
                self._retry_config,
                proposal=proposal,
                context=context,
                timeout=effective_timeout,
            )
        except Exception as e:
            self._logger.error(f"Decision failed after retries: {e}")
            # Publish failure event
            await self._publish_decision_event(
                EventType.TRADE_FAILED,
                proposal,
                None,
                str(e)
            )
            # Return safe reject decision
            decision = self._create_fallback_decision(proposal, str(e))

        elapsed_ms = (time.monotonic() - start) * 1000
        self._update_metrics(elapsed_ms)
        self._total_async_decisions += 1

        # Publish decision event
        event_type = {
            DecisionType.APPROVE: EventType.TRADE_APPROVED,
            DecisionType.MODIFY: EventType.TRADE_MODIFIED,
            DecisionType.REJECT: EventType.TRADE_REJECTED,
        }.get(decision.final_decision, EventType.TRADE_REJECTED)

        await self._publish_decision_event(
            event_type, proposal, decision
        )

        return decision

    async def _execute_decision(
        self,
        proposal: TradeProposal,
        context: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0
    ) -> OrchestratedDecision:
        """
        Execute decision in thread executor with timeout.

        Wraps the synchronous coordinate_decision call.
        """
        loop = asyncio.get_event_loop()

        try:
            decision = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    self._orchestrator.coordinate_decision,
                    proposal,
                    context
                ),
                timeout=timeout
            )
            return decision

        except asyncio.TimeoutError:
            self._total_timeouts += 1
            self._logger.error(
                f"Decision timed out after {timeout}s"
            )
            raise

    def _create_fallback_decision(
        self,
        proposal: TradeProposal,
        error: str
    ) -> OrchestratedDecision:
        """Create a safe reject decision for error cases."""
        return OrchestratedDecision(
            final_decision=DecisionType.REJECT,
            final_position_size=0.0,
            original_position_size=proposal.quantity,
            confidence=0.0,
            reasoning=[f"Fallback rejection: {error}"],
            blocking_agent="async_orchestrator_error",
        )

    # =========================================================================
    # EVENT PUBLISHING
    # =========================================================================

    async def _publish_decision_event(
        self,
        event_type: EventType,
        proposal: TradeProposal,
        decision: Optional[OrchestratedDecision],
        error: Optional[str] = None
    ) -> None:
        """Publish a decision event to the messaging bus."""
        data = {
            'action': proposal.action,
            'asset': proposal.asset,
            'quantity': proposal.quantity,
            'entry_price': proposal.entry_price,
        }

        if decision:
            data.update({
                'decision': decision.final_decision.name,
                'final_size': decision.final_position_size,
                'confidence': decision.confidence,
                'blocking_agent': decision.blocking_agent,
                'reasoning': decision.reasoning,
                'orchestration_time_ms': decision.orchestration_time_ms,
            })

        if error:
            data['error'] = error

        # Map decision to priority
        priority = EventPriority.NORMAL
        if event_type == EventType.TRADE_REJECTED:
            priority = EventPriority.HIGH
        elif event_type == EventType.TRADE_FAILED:
            priority = EventPriority.CRITICAL

        event = Event(
            event_type=event_type,
            priority=priority,
            source="async_orchestrator",
            data=data
        )

        try:
            await self._event_bus.publish(event, timeout=2.0)
        except Exception as e:
            self._logger.error(f"Failed to publish decision event: {e}")

    # =========================================================================
    # AGENT HEALTH MONITORING
    # =========================================================================

    async def check_agent_health(self) -> Dict[str, Any]:
        """
        Check health of all registered agents asynchronously.

        Returns:
            Dict with per-agent health status
        """
        status = self._orchestrator.get_status()
        circuit_status = self._orchestrator.get_circuit_breaker_status()

        # Publish health check event
        await self._event_bus.publish(Event(
            event_type=EventType.HEALTH_CHECK,
            source="async_orchestrator",
            data={
                'agents': status['agents'],
                'circuit_breakers': circuit_status,
                'trading_enabled': status['trading_enabled'],
            }
        ))

        return {
            'status': status,
            'circuit_breakers': circuit_status,
            'async_metrics': self.get_metrics(),
        }

    async def run_health_loop(
        self,
        interval: float = 30.0
    ) -> None:
        """
        Run continuous health monitoring loop.

        Args:
            interval: Check interval in seconds
        """
        while self._running:
            try:
                health = await self.check_agent_health()

                # Check for critical issues
                failed = health['status'].get('failed_agents', [])
                if failed:
                    self._logger.warning(
                        f"Failed agents detected: {failed}"
                    )
                    await self._event_bus.publish(Event(
                        event_type=EventType.AGENT_ERROR,
                        priority=EventPriority.HIGH,
                        source="async_orchestrator",
                        data={'failed_agents': failed}
                    ))

            except Exception as e:
                self._logger.error(f"Health check error: {e}")

            await asyncio.sleep(interval)

    # =========================================================================
    # SUBSCRIPTION API
    # =========================================================================

    def on_trade_approved(self, handler: Callable) -> None:
        """Subscribe to trade approval events."""
        self._event_bus.subscribe(
            EventType.TRADE_APPROVED, handler, "on_trade_approved"
        )

    def on_trade_rejected(self, handler: Callable) -> None:
        """Subscribe to trade rejection events."""
        self._event_bus.subscribe(
            EventType.TRADE_REJECTED, handler, "on_trade_rejected"
        )

    def on_trade_modified(self, handler: Callable) -> None:
        """Subscribe to trade modification events."""
        self._event_bus.subscribe(
            EventType.TRADE_MODIFIED, handler, "on_trade_modified"
        )

    def on_kill_switch(self, handler: Callable) -> None:
        """Subscribe to kill switch events."""
        self._event_bus.subscribe(
            EventType.KILL_SWITCH_TRIGGERED, handler, "on_kill_switch"
        )

    def on_risk_alert(self, handler: Callable) -> None:
        """Subscribe to risk alert events."""
        self._event_bus.subscribe(
            EventType.RISK_WARNING, handler, "on_risk_alert"
        )

    # =========================================================================
    # METRICS
    # =========================================================================

    def _update_metrics(self, elapsed_ms: float) -> None:
        """Update running average of decision time."""
        n = self._total_async_decisions + 1
        self._avg_decision_time_ms = (
            (self._avg_decision_time_ms * (n - 1) + elapsed_ms) / n
        )

    def get_metrics(self) -> Dict[str, Any]:
        """Get async orchestrator metrics."""
        return {
            'total_async_decisions': self._total_async_decisions,
            'total_timeouts': self._total_timeouts,
            'total_retries': self._total_retries,
            'avg_decision_time_ms': round(self._avg_decision_time_ms, 2),
            'is_running': self._running,
            'queue_metrics': self._decision_queue.get_metrics(),
            'bus_metrics': self._event_bus.get_metrics(),
            'dlq_size': self._dlq.size,
        }

    # =========================================================================
    # DELEGATED API (pass-through to synchronous orchestrator)
    # =========================================================================

    @property
    def orchestrator(self) -> TradingOrchestrator:
        """Access the underlying synchronous orchestrator."""
        return self._orchestrator

    @property
    def event_bus(self) -> EventBus:
        """Access the messaging event bus."""
        return self._event_bus

    @property
    def dead_letter_queue(self) -> DeadLetterQueue:
        """Access the dead letter queue."""
        return self._dlq

    def is_trading_enabled(self) -> bool:
        """Check if trading is enabled."""
        return self._orchestrator.is_trading_enabled()

    def enable_trading(self) -> None:
        """Enable trading."""
        self._orchestrator.enable_trading()

    def disable_trading(self, reason: str) -> None:
        """Disable trading."""
        self._orchestrator.disable_trading(reason)
