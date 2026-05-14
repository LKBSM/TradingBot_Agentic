"""Tests for the OBS-2B.5 error-budget watcher."""

from __future__ import annotations

import asyncio

import pytest

from src.api.error_budget_watcher import (
    AlertEvent,
    ErrorBudgetWatcher,
    SLOSpec,
    UNKNOWN_SLO,
)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Stub LatencyTracker
# ---------------------------------------------------------------------------


class _StubSnap:
    def __init__(self, path, count, p95_ms, error_rate):
        self.path = path
        self.count = count
        self.p95_ms = p95_ms
        self.error_rate = error_rate


class _StubTracker:
    """A tracker stand-in that returns whatever snapshots() the test pushes."""

    def __init__(self):
        self.snaps: list[_StubSnap] = []

    def push(self, path, count, p95_ms=10.0, error_rate=0.0):
        self.snaps.append(_StubSnap(path, count, p95_ms, error_rate))

    def clear(self):
        self.snaps.clear()

    def snapshot_all(self):
        return list(self.snaps)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_constructor_rejects_none_tracker():
    with pytest.raises(ValueError):
        ErrorBudgetWatcher(None)


def test_constructor_rejects_zero_threshold():
    with pytest.raises(ValueError):
        ErrorBudgetWatcher(_StubTracker(), threshold_count=0)


def test_constructor_rejects_non_positive_poll():
    with pytest.raises(ValueError):
        ErrorBudgetWatcher(_StubTracker(), poll_seconds=0)


# ---------------------------------------------------------------------------
# Hysteresis: N consecutive breaches required to fire
# ---------------------------------------------------------------------------


def test_single_breach_does_not_fire():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=3,
    )
    tr.push("/x", count=10, p95_ms=500)  # breach
    events = w.evaluate_once()
    assert events == []
    assert w.firing_routes() == []  # not firing yet


def test_three_consecutive_breaches_fire_once():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=3,
    )
    fired = []
    for _ in range(3):
        tr.clear(); tr.push("/x", count=10, p95_ms=500)
        fired.extend(w.evaluate_once())
    # Exactly ONE fire event after the third breach.
    assert [(e.evt, e.metric) for e in fired] == [("alert_fire", "p95_ms")]
    assert w.firing_routes()[0]["route"] == "/x"


def test_fourth_breach_does_not_re_fire():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=2,
    )
    for _ in range(5):
        tr.clear(); tr.push("/x", count=10, p95_ms=500)
        w.evaluate_once()
    # Only the first fire — subsequent breaches don't spam alerts.
    fires = [e for e in w._fired_events if e.evt == "alert_fire"]
    assert len(fires) == 1


def test_alert_clears_on_first_satisfying_snapshot():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
    )
    tr.clear(); tr.push("/x", count=10, p95_ms=500)
    w.evaluate_once()  # fires
    tr.clear(); tr.push("/x", count=10, p95_ms=50)
    events = w.evaluate_once()
    assert len(events) == 1
    assert events[0].evt == "alert_clear"
    assert w.firing_routes() == []


# ---------------------------------------------------------------------------
# Error-rate metric is independent of p95
# ---------------------------------------------------------------------------


def test_error_rate_breach_fires_independently():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=1000, error_rate_warn=0.01)},
        threshold_count=2,
    )
    for _ in range(2):
        tr.clear(); tr.push("/x", count=10, p95_ms=50, error_rate=0.5)
        w.evaluate_once()
    metrics = sorted(e.metric for e in w._fired_events if e.evt == "alert_fire")
    assert metrics == ["error_rate"]


def test_both_metrics_can_fire_simultaneously():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
    )
    tr.push("/x", count=10, p95_ms=500, error_rate=0.5)
    events = w.evaluate_once()
    metrics = sorted(e.metric for e in events if e.evt == "alert_fire")
    assert metrics == ["error_rate", "p95_ms"]


# ---------------------------------------------------------------------------
# Empty buckets are no-ops
# ---------------------------------------------------------------------------


def test_zero_count_snapshot_does_not_breach_or_satisfy():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
    )
    # First, push real breaches so the alert fires.
    tr.push("/x", count=10, p95_ms=500)
    w.evaluate_once()
    assert w.firing_routes()
    # Now a snapshot with count=0 — must NOT count as clearing.
    tr.clear(); tr.push("/x", count=0, p95_ms=0)
    events = w.evaluate_once()
    assert events == []
    assert w.firing_routes()  # still firing


# ---------------------------------------------------------------------------
# Unknown route uses UNKNOWN_SLO
# ---------------------------------------------------------------------------


def test_unknown_route_uses_default_unknown_slo():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(tr, threshold_count=1)
    tr.push("/api/v1/wat", count=10, p95_ms=UNKNOWN_SLO.p95_ms_warn + 1)
    events = w.evaluate_once()
    assert len(events) == 1
    assert events[0].route == "/api/v1/wat"


# ---------------------------------------------------------------------------
# Sink callable
# ---------------------------------------------------------------------------


def test_sink_callable_receives_each_event():
    received: list[AlertEvent] = []
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr,
        slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
        sink=received.append,
    )
    tr.push("/x", count=10, p95_ms=500)
    w.evaluate_once()
    assert len(received) == 1
    assert received[0].route == "/x"


def test_sink_exception_does_not_break_watcher():
    def boom(_ev):
        raise RuntimeError("sink down")

    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr,
        slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
        sink=boom,
    )
    tr.push("/x", count=10, p95_ms=500)
    # Must not raise.
    w.evaluate_once()
    assert w.firing_routes()


# ---------------------------------------------------------------------------
# Async lifecycle
# ---------------------------------------------------------------------------


def test_start_then_stop_cleanly():
    async def scenario():
        tr = _StubTracker()
        w = ErrorBudgetWatcher(tr, poll_seconds=0.05)
        await w.start()
        assert w.is_running
        await asyncio.sleep(0.1)  # let one poll cycle pass
        await w.stop()
        assert not w.is_running

    _run(scenario())


# ---------------------------------------------------------------------------
# Bounded event history
# ---------------------------------------------------------------------------


def test_recent_events_cap_at_100():
    tr = _StubTracker()
    w = ErrorBudgetWatcher(
        tr, slos={"/x": SLOSpec(p95_ms_warn=100, error_rate_warn=0.01)},
        threshold_count=1,
    )
    # Generate 200 fire/clear pairs.
    for _ in range(200):
        tr.clear(); tr.push("/x", count=10, p95_ms=500); w.evaluate_once()
        tr.clear(); tr.push("/x", count=10, p95_ms=50); w.evaluate_once()
    assert len(w._fired_events) == 100
    assert len(w.recent_events(limit=50)) == 50
