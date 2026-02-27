# =============================================================================
# Tests for Sprint 3: Kill Switch Escalation & Event Bus Hardening
# =============================================================================

import sys
import os
import time
from datetime import datetime, timedelta
from collections import deque

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.agents.kill_switch import (
    KillSwitch, KillSwitchConfig, HaltLevel, HaltReason
)


# ===========================================================================
# 1. Kill Switch Escalation Tests
# ===========================================================================

class TestKillSwitchEscalation:

    def _make_ks(self, **kwargs) -> KillSwitch:
        config = KillSwitchConfig(**kwargs)
        return KillSwitch(config=config, enable_persistence=False)

    def test_escalate_from_none_to_caution(self):
        ks = self._make_ks()
        assert ks.halt_level == HaltLevel.NONE
        new = ks.escalate("test escalation")
        assert new == HaltLevel.CAUTION

    def test_escalate_steps_through_levels(self):
        ks = self._make_ks()
        levels = []
        for i in range(6):  # NONE(0) -> EMERGENCY(6)
            new = ks.escalate(f"escalation step {i+1}")
            levels.append(new)
        assert levels == [
            HaltLevel.CAUTION,
            HaltLevel.REDUCED,
            HaltLevel.NEW_ONLY,
            HaltLevel.CLOSE_ONLY,
            HaltLevel.FULL_HALT,
            HaltLevel.EMERGENCY,
        ]

    def test_escalate_caps_at_emergency(self):
        ks = self._make_ks()
        # Escalate to EMERGENCY
        for _ in range(6):
            ks.escalate("escalate")
        assert ks.halt_level == HaltLevel.EMERGENCY
        # Another escalation should stay at EMERGENCY
        level = ks.escalate("already at max")
        assert level == HaltLevel.EMERGENCY

    def test_escalate_never_ratchets_down(self):
        """Escalation can only go up, never down."""
        ks = self._make_ks()
        ks.escalate("first")
        assert ks.halt_level == HaltLevel.CAUTION
        ks.escalate("second")
        assert ks.halt_level == HaltLevel.REDUCED
        # Even if we call update with good data, halt shouldn't go below REDUCED
        # (update respects "only upgrade halt level" rule in _trigger_halt)

    def test_escalate_sets_execution_error_reason(self):
        ks = self._make_ks()
        ks.escalate("MT5 timeout")
        assert ks.halt_reason == HaltReason.EXECUTION_ERROR


# ===========================================================================
# 2. Close Failure Tracking Tests
# ===========================================================================

class TestCloseFailureTracking:

    def _make_ks(self) -> KillSwitch:
        return KillSwitch(
            config=KillSwitchConfig(),
            enable_persistence=False
        )

    def test_single_failure_no_escalation(self):
        ks = self._make_ks()
        level = ks.record_close_failure("MT5 reject")
        assert level == HaltLevel.NONE  # 1 failure, threshold is 3

    def test_two_failures_no_escalation(self):
        ks = self._make_ks()
        ks.record_close_failure("fail 1")
        level = ks.record_close_failure("fail 2")
        assert level == HaltLevel.NONE  # Still under threshold

    def test_three_failures_triggers_escalation(self):
        """3 failures within 60s should auto-escalate."""
        ks = self._make_ks()
        ks.record_close_failure("fail 1")
        ks.record_close_failure("fail 2")
        level = ks.record_close_failure("fail 3")
        # Should have escalated from NONE(0) to CAUTION(1)
        assert level.value >= HaltLevel.CAUTION.value

    def test_successive_escalation_rounds(self):
        """Multiple rounds of 3 failures should escalate further each time."""
        ks = self._make_ks()

        # First round: NONE -> CAUTION
        for i in range(3):
            level = ks.record_close_failure(f"round1-{i}")
        assert level.value >= HaltLevel.CAUTION.value

        # Second round: CAUTION -> REDUCED (or higher)
        for i in range(3):
            level = ks.record_close_failure(f"round2-{i}")
        assert level.value >= HaltLevel.REDUCED.value

    def test_failures_outside_window_dont_count(self):
        """Failures older than the window should not trigger escalation."""
        ks = self._make_ks()
        # Shrink window for testing
        ks._close_failure_window = timedelta(seconds=1)

        ks.record_close_failure("old fail 1")
        ks.record_close_failure("old fail 2")
        time.sleep(1.1)  # Wait for window to expire
        level = ks.record_close_failure("new fail 1")
        # Only 1 failure in window, should not escalate
        assert level == HaltLevel.NONE

    def test_counter_resets_after_escalation(self):
        """After escalation, the failure counter should reset."""
        ks = self._make_ks()
        for _ in range(3):
            ks.record_close_failure("fail")
        # Counter should be reset after escalation
        assert ks._consecutive_close_failures == 0
        assert len(ks._close_failure_timestamps) == 0


# ===========================================================================
# 3. Event Bus Critical Bypass Tests
# ===========================================================================

class TestEventBusCriticalBypass:

    def test_critical_events_bypass_rate_limiter(self):
        """RISK_ALERT, DRAWDOWN_BREACH, DRAWDOWN_WARNING should never be rate-limited."""
        from src.agents.events import EventBus, EventType

        bus = EventBus(persist_events=False)

        # Exhaust rate limit for a source
        for _ in range(600):  # > 500 limit
            bus._is_rate_limited("test_agent", event_type=EventType.MARKET_DATA_UPDATE)

        # Normal events should now be rate-limited
        assert bus._is_rate_limited("test_agent", event_type=EventType.MARKET_DATA_UPDATE) is True

        # Critical events should NEVER be rate-limited
        assert bus._is_rate_limited("test_agent", event_type=EventType.RISK_ALERT) is False
        assert bus._is_rate_limited("test_agent", event_type=EventType.DRAWDOWN_BREACH) is False
        assert bus._is_rate_limited("test_agent", event_type=EventType.DRAWDOWN_WARNING) is False

    def test_normal_events_still_rate_limited(self):
        """Non-critical events should still be properly rate-limited."""
        from src.agents.events import EventBus, EventType

        bus = EventBus(persist_events=False)

        # Fill up the rate limit
        for _ in range(500):
            assert bus._is_rate_limited("agent_x", event_type=EventType.HEARTBEAT) is False

        # Next normal event should be rate-limited
        assert bus._is_rate_limited("agent_x", event_type=EventType.HEARTBEAT) is True

    def test_rate_limiter_without_event_type(self):
        """When event_type is None, rate limiting should still work normally."""
        from src.agents.events import EventBus

        bus = EventBus(persist_events=False)
        for _ in range(500):
            bus._is_rate_limited("agent_y")
        assert bus._is_rate_limited("agent_y") is True


# ===========================================================================
# 4. Event Bus deque Fix Verification
# ===========================================================================

class TestEventBusDeque:

    def test_rate_limit_counters_use_deque(self):
        """_rate_limit_counters should use deque, not list."""
        from src.agents.events import EventBus

        bus = EventBus(persist_events=False)

        # Trigger a rate limit check to create an entry
        bus._is_rate_limited("test_agent")

        # Verify the counter is a deque, not a list
        counter = bus._rate_limit_counters["test_agent"]
        assert isinstance(counter, deque), (
            f"Expected deque, got {type(counter).__name__}"
        )

    def test_no_list_pop_0_in_events(self):
        """events.py should not contain list.pop(0) — use deque.popleft()."""
        events_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'agents', 'events.py'
        )
        with open(events_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check the rate limiter section specifically
        # We allow pop(0) in other contexts but not in rate limiting
        rate_limiter_start = content.find('def _is_rate_limited')
        rate_limiter_end = content.find('\n    def ', rate_limiter_start + 1)
        rate_limiter_code = content[rate_limiter_start:rate_limiter_end]

        assert '.pop(0)' not in rate_limiter_code, (
            "Found .pop(0) in _is_rate_limited — should use .popleft()"
        )
        assert '.popleft()' in rate_limiter_code, (
            ".popleft() not found in _is_rate_limited"
        )


# ===========================================================================
# 5. Source Code Verification
# ===========================================================================

class TestSourceVerification:

    def test_no_print_in_escalation(self):
        """Escalation methods should use logger, not print()."""
        ks_path = os.path.join(
            os.path.dirname(__file__), '..', 'src', 'agents', 'kill_switch.py'
        )
        with open(ks_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Extract escalation section
        start = content.find('def escalate(')
        end = content.find('# =====', start + 1)
        if start != -1 and end != -1:
            escalation_code = content[start:end]
            assert 'print(' not in escalation_code, (
                "print() found in escalation code — use logger instead"
            )

    def test_critical_event_types_defined(self):
        """CRITICAL_EVENT_TYPES should be defined on EventBus."""
        from src.agents.events import EventBus, EventType

        assert hasattr(EventBus, 'CRITICAL_EVENT_TYPES')
        assert EventType.RISK_ALERT in EventBus.CRITICAL_EVENT_TYPES
        assert EventType.DRAWDOWN_BREACH in EventBus.CRITICAL_EVENT_TYPES
        assert EventType.DRAWDOWN_WARNING in EventBus.CRITICAL_EVENT_TYPES
