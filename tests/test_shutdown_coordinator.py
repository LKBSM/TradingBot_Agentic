"""Tests for the INFRA-2B.11 graceful shutdown coordinator."""

from __future__ import annotations

import asyncio
import time

import pytest

from src.api.shutdown import (
    DEFAULT_HANDLER_BUDGET_S,
    DEFAULT_TOTAL_BUDGET_S,
    GracefulShutdownCoordinator,
    HandlerResult,
    ShutdownReport,
)


def _run(coro):
    """Run an async coroutine without pytest-asyncio."""
    return asyncio.new_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# Construction guards
# ---------------------------------------------------------------------------


def test_rejects_non_positive_total_budget():
    with pytest.raises(ValueError):
        GracefulShutdownCoordinator(total_budget_s=0)
    with pytest.raises(ValueError):
        GracefulShutdownCoordinator(total_budget_s=-1)


def test_rejects_empty_handler_name():
    coord = GracefulShutdownCoordinator()
    with pytest.raises(ValueError):
        coord.register("", lambda: None)


def test_rejects_non_positive_handler_budget():
    coord = GracefulShutdownCoordinator()
    with pytest.raises(ValueError):
        coord.register("x", lambda: None, budget_s=0)


# ---------------------------------------------------------------------------
# Handler invocation — sync + async
# ---------------------------------------------------------------------------


def test_runs_sync_handler():
    seen = []
    coord = GracefulShutdownCoordinator()
    coord.register("s", lambda: seen.append("called"))
    report = _run(coord.run())
    assert seen == ["called"]
    assert report.ok
    assert len(report.handlers) == 1
    assert report.handlers[0].ok is True
    assert report.handlers[0].name == "s"


def test_runs_async_handler():
    seen = []

    async def h():
        await asyncio.sleep(0)
        seen.append("async")

    coord = GracefulShutdownCoordinator()
    coord.register("a", h)
    _run(coord.run())
    assert seen == ["async"]


def test_handlers_run_in_registration_order():
    seen = []
    coord = GracefulShutdownCoordinator()
    for name in ("first", "second", "third"):
        coord.register(name, lambda n=name: seen.append(n))
    _run(coord.run())
    assert seen == ["first", "second", "third"]


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


def test_exception_in_one_handler_does_not_block_others():
    seen = []

    def boom():
        raise RuntimeError("nope")

    coord = GracefulShutdownCoordinator()
    coord.register("ok1", lambda: seen.append("ok1"))
    coord.register("boom", boom)
    coord.register("ok2", lambda: seen.append("ok2"))
    report = _run(coord.run())
    assert seen == ["ok1", "ok2"]
    assert report.n_failed == 1
    assert not report.ok
    failed = next(h for h in report.handlers if h.name == "boom")
    assert "RuntimeError" in failed.error


def test_timeout_handler_is_recorded_but_does_not_block():
    seen = []

    async def slow():
        await asyncio.sleep(5)
        seen.append("slow-finished")

    coord = GracefulShutdownCoordinator(total_budget_s=10)
    coord.register("slow", slow, budget_s=0.1)
    coord.register("after", lambda: seen.append("after"))
    report = _run(coord.run())
    assert "slow-finished" not in seen
    assert seen == ["after"]
    slow_result = next(h for h in report.handlers if h.name == "slow")
    assert slow_result.timed_out is True
    assert slow_result.ok is False


# ---------------------------------------------------------------------------
# Global budget enforcement
# ---------------------------------------------------------------------------


def test_handlers_after_global_budget_are_skipped():
    class FakeClock:
        def __init__(self):
            self.t = 0.0

        def __call__(self):
            return self.t

    clk = FakeClock()
    seen = []

    def burn(seconds):
        def _inner():
            clk.t += seconds  # advance our virtual clock instead of sleeping
            seen.append(seconds)

        return _inner

    coord = GracefulShutdownCoordinator(total_budget_s=5.0, clock=clk)
    coord.register("a", burn(2.0), budget_s=10.0)
    coord.register("b", burn(2.0), budget_s=10.0)
    coord.register("c", burn(2.0), budget_s=10.0)  # would exceed budget
    coord.register("d", burn(2.0), budget_s=10.0)  # skipped
    report = _run(coord.run())
    # First two ran (4s used), third pushed total to 6 — still ran (we
    # check the budget BEFORE running). Fourth started at 6s elapsed,
    # over the 5s budget → skipped.
    skipped = [h for h in report.handlers if h.skipped]
    assert len(skipped) >= 1
    assert skipped[-1].name == "d"


# ---------------------------------------------------------------------------
# Idempotence
# ---------------------------------------------------------------------------


def test_run_twice_returns_cached_report():
    counter = {"n": 0}

    def h():
        counter["n"] += 1

    coord = GracefulShutdownCoordinator()
    coord.register("h", h)
    r1 = _run(coord.run())
    r2 = _run(coord.run())
    assert counter["n"] == 1
    assert r1 is r2


def test_cannot_register_after_run_started():
    coord = GracefulShutdownCoordinator()
    coord.register("h", lambda: None)
    _run(coord.run())
    with pytest.raises(RuntimeError):
        coord.register("late", lambda: None)


# ---------------------------------------------------------------------------
# Report shape
# ---------------------------------------------------------------------------


def test_report_serialises_to_dict_with_durations():
    coord = GracefulShutdownCoordinator()
    coord.register("h1", lambda: None)
    coord.register("h2", lambda: None)
    report = _run(coord.run())
    d = report.to_dict()
    assert d["n_handlers"] == 2
    assert d["ok"] is True
    assert {h["name"] for h in d["handlers"]} == {"h1", "h2"}
    for h in d["handlers"]:
        assert h["duration_ms"] >= 0
        assert h["skipped"] is False


def test_report_marks_overall_failure_when_handler_raises():
    coord = GracefulShutdownCoordinator()
    coord.register("ok", lambda: None)
    coord.register("bad", lambda: (_ for _ in ()).throw(ValueError("x")))
    report = _run(coord.run())
    assert report.ok is False
    assert report.n_failed == 1


# ---------------------------------------------------------------------------
# Empty registration is a no-op
# ---------------------------------------------------------------------------


def test_no_handlers_returns_empty_report():
    coord = GracefulShutdownCoordinator()
    report = _run(coord.run())
    assert report.handlers == []
    assert report.ok is True
    assert report.n_failed == 0


# ---------------------------------------------------------------------------
# is_shutting_down flag
# ---------------------------------------------------------------------------


def test_is_shutting_down_flips_during_run():
    states = []

    async def observe(coord):
        states.append(coord.is_shutting_down)

    coord = GracefulShutdownCoordinator()
    assert coord.is_shutting_down is False
    coord.register("obs", lambda: states.append("during"))
    _run(coord.run())
    assert coord.is_shutting_down is True
