"""Tests for the OBS-2B.4 per-endpoint latency tracker."""

from __future__ import annotations

import pytest

from src.api.latency_tracker import (
    DEFAULT_MAX_ROUTES,
    DEFAULT_PER_ROUTE_CAP,
    LatencySnapshot,
    LatencyTracker,
)


class _FakeClock:
    """Monotonic-style clock you can advance manually."""

    def __init__(self, t0: float = 0.0):
        self._t = t0

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


# ---------------------------------------------------------------------------
# Construction / argument validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_non_positive_window():
    with pytest.raises(ValueError):
        LatencyTracker(window_seconds=0)
    with pytest.raises(ValueError):
        LatencyTracker(window_seconds=-5)


def test_constructor_rejects_zero_per_route_cap():
    with pytest.raises(ValueError):
        LatencyTracker(per_route_cap=0)


def test_default_caps_match_module_constants():
    t = LatencyTracker()
    assert t._max_routes == DEFAULT_MAX_ROUTES
    assert t._per_route_cap == DEFAULT_PER_ROUTE_CAP


# ---------------------------------------------------------------------------
# record() + snapshot() basics
# ---------------------------------------------------------------------------


def test_snapshot_unknown_path_returns_zeroed():
    t = LatencyTracker(clock=_FakeClock())
    snap = t.snapshot("/api/v1/never-seen")
    assert isinstance(snap, LatencySnapshot)
    assert snap.count == 0
    assert snap.p50_ms == 0.0
    assert snap.error_rate == 0.0
    assert snap.count_total == 0


def test_record_then_snapshot_yields_percentiles():
    clk = _FakeClock()
    t = LatencyTracker(clock=clk)
    # Insert 1..100 ms — easy percentiles to assert.
    for ms in range(1, 101):
        t.record("/api/v1/x", float(ms), 200)
    snap = t.snapshot("/api/v1/x")
    assert snap.count == 100
    # Linear interp at q=0.5 over a 0-indexed length-100 sorted list:
    # idx = 0.5 * 99 = 49.5 → between values[49]=50 and values[50]=51 → 50.5.
    assert snap.p50_ms == pytest.approx(50.5)
    # idx = 0.95 * 99 = 94.05 → between values[94]=95 and values[95]=96 → 95.05
    assert snap.p95_ms == pytest.approx(95.05)
    assert snap.p99_ms == pytest.approx(99.01)
    assert snap.max_ms == 100.0
    assert snap.error_rate == 0.0


def test_negative_latency_is_dropped():
    t = LatencyTracker(clock=_FakeClock())
    t.record("/x", -1.0, 200)
    assert t.snapshot("/x").count == 0


def test_empty_path_is_dropped():
    t = LatencyTracker(clock=_FakeClock())
    t.record("", 12.0, 200)
    assert t.routes_tracked == 0


# ---------------------------------------------------------------------------
# Error rate
# ---------------------------------------------------------------------------


def test_error_rate_uses_4xx_and_5xx():
    t = LatencyTracker(clock=_FakeClock())
    for _ in range(8):
        t.record("/api/v1/x", 5.0, 200)
    t.record("/api/v1/x", 5.0, 404)
    t.record("/api/v1/x", 5.0, 500)
    snap = t.snapshot("/api/v1/x")
    assert snap.error_count == 2
    assert snap.error_rate == 0.2


def test_error_count_total_survives_eviction():
    clk = _FakeClock()
    t = LatencyTracker(window_seconds=10, clock=clk)
    t.record("/x", 1.0, 500)
    t.record("/x", 1.0, 200)
    clk.advance(60)  # purge everything
    snap = t.snapshot("/x")
    assert snap.count == 0
    assert snap.error_rate == 0.0
    # Lifetime counters survive eviction so SLO dashboards stay coherent.
    assert snap.count_total == 2
    assert snap.error_count_total == 1


# ---------------------------------------------------------------------------
# Window eviction
# ---------------------------------------------------------------------------


def test_old_observations_purged_on_read():
    clk = _FakeClock()
    t = LatencyTracker(window_seconds=10, clock=clk)
    t.record("/x", 1.0, 200)
    t.record("/x", 2.0, 200)
    clk.advance(11)
    # Both samples older than window → purged when we read.
    snap = t.snapshot("/x")
    assert snap.count == 0


def test_partial_window_eviction():
    clk = _FakeClock()
    t = LatencyTracker(window_seconds=10, clock=clk)
    t.record("/x", 5.0, 200)
    clk.advance(8)
    t.record("/x", 6.0, 200)
    clk.advance(5)
    # Now older sample is 13s old (out), newer is 5s old (in).
    snap = t.snapshot("/x")
    assert snap.count == 1
    assert snap.max_ms == 6.0


# ---------------------------------------------------------------------------
# Cardinality cap
# ---------------------------------------------------------------------------


def test_new_routes_dropped_when_at_cardinality_cap():
    t = LatencyTracker(max_routes=2, clock=_FakeClock())
    t.record("/a", 1.0, 200)
    t.record("/b", 1.0, 200)
    t.record("/c", 1.0, 200)  # rejected
    assert t.routes_tracked == 2
    assert t.snapshot("/c").count == 0
    # Existing routes keep recording after the cap is hit.
    t.record("/a", 2.0, 200)
    assert t.snapshot("/a").count == 2


def test_per_route_cap_bounds_buffer():
    t = LatencyTracker(per_route_cap=10, clock=_FakeClock())
    for ms in range(50):
        t.record("/x", float(ms), 200)
    snap = t.snapshot("/x")
    # Only the last 10 samples kept (ms 40..49).
    assert snap.count == 10
    assert snap.max_ms == 49.0
    # Lifetime counter still sees the full 50.
    assert snap.count_total == 50


# ---------------------------------------------------------------------------
# snapshot_all + reset
# ---------------------------------------------------------------------------


def test_snapshot_all_returns_sorted_by_path():
    t = LatencyTracker(clock=_FakeClock())
    for path in ["/zeta", "/alpha", "/mu"]:
        t.record(path, 5.0, 200)
    snaps = t.snapshot_all()
    assert [s.path for s in snaps] == ["/alpha", "/mu", "/zeta"]


def test_reset_clears_state_including_lifetime():
    t = LatencyTracker(clock=_FakeClock())
    t.record("/x", 5.0, 200)
    t.reset()
    snap = t.snapshot("/x")
    assert snap.count == 0
    assert snap.count_total == 0
    assert t.routes_tracked == 0


# ---------------------------------------------------------------------------
# Percentile helper edge cases
# ---------------------------------------------------------------------------


def test_percentile_on_empty_list_returns_zero():
    assert LatencyTracker._percentile([], 0.5) == 0.0


def test_percentile_boundary_values():
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert LatencyTracker._percentile(vals, 0.0) == 1.0
    assert LatencyTracker._percentile(vals, 1.0) == 5.0


def test_single_observation_all_percentiles_equal():
    t = LatencyTracker(clock=_FakeClock())
    t.record("/x", 42.0, 200)
    snap = t.snapshot("/x")
    assert snap.p50_ms == 42.0
    assert snap.p95_ms == 42.0
    assert snap.p99_ms == 42.0
    assert snap.max_ms == 42.0
