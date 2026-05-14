"""Tests for the DATA-2B.5 freshness monitor."""

from __future__ import annotations

import pytest

from src.agents.data.freshness_monitor import (
    DEFAULT_SLAS,
    FreshnessMonitor,
    SLASpec,
)


class _Clock:
    def __init__(self, t0=0.0):
        self.t = t0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def test_never_seen_source_is_a_breach():
    clk = _Clock(1000.0)
    m = FreshnessMonitor(clock=clk)
    snap = m.snapshot("news")
    assert snap.in_sla is False
    assert snap.last_seen_ts is None
    assert snap.age_seconds is None


def test_report_then_in_sla_inside_window():
    clk = _Clock(1000.0)
    m = FreshnessMonitor(clock=clk)
    m.report("news")
    clk.advance(60)  # 1 minute later
    snap = m.snapshot("news")
    assert snap.in_sla is True
    assert snap.age_seconds == 60


def test_breach_when_age_exceeds_sla():
    clk = _Clock(1000.0)
    m = FreshnessMonitor(clock=clk)
    m.report("news")
    clk.advance(31 * 60)  # 31 min, SLA is 30 min
    snap = m.snapshot("news")
    assert snap.in_sla is False
    assert snap.age_seconds > snap.sla_seconds


def test_unknown_source_has_infinite_sla():
    clk = _Clock(1000.0)
    m = FreshnessMonitor(clock=clk)
    m.report("custom_feed")
    snap = m.snapshot("custom_feed")
    assert snap.sla_seconds == float("inf")
    assert snap.in_sla is True  # any age satisfies infinite SLA


def test_report_empty_source_raises():
    m = FreshnessMonitor()
    with pytest.raises(ValueError):
        m.report("")


def test_any_breach_aggregates():
    clk = _Clock(1000.0)
    # Custom SLA set with only news + macro so we control the state space.
    slas = {
        "news":  SLASpec(max_lag_seconds=30 * 60),
        "macro": SLASpec(max_lag_seconds=26 * 3600),
    }
    m = FreshnessMonitor(slas=slas, clock=clk)
    m.report("news"); m.report("macro")
    clk.advance(60)
    assert m.any_breach() is False
    clk.advance(30 * 3600)  # news + macro both stale
    assert m.any_breach() is True


def test_snapshot_all_returns_sorted():
    m = FreshnessMonitor(clock=_Clock())
    m.report("z")
    m.report("a")
    paths = [r.source for r in m.snapshot_all()]
    # includes default SLA sources too — just check ordering
    assert paths == sorted(paths)


def test_default_slas_cover_phase_2b_feeds():
    for key in ("news", "sentiment", "macro", "prices_m15", "cot", "fred"):
        assert key in DEFAULT_SLAS


def test_add_sla_at_runtime():
    m = FreshnessMonitor(clock=_Clock())
    m.add_sla("custom", SLASpec(max_lag_seconds=10, description="< 10s"))
    m.report("custom")
    snap = m.snapshot("custom")
    assert snap.sla_seconds == 10
    assert snap.description == "< 10s"


def test_report_with_explicit_ts():
    clk = _Clock(2000.0)
    m = FreshnessMonitor(clock=clk)
    m.report("news", ts=1000.0)  # 1000 seconds ago vs clock=2000
    snap = m.snapshot("news")
    assert snap.age_seconds == 1000.0
    # 1000s > 1800s SLA? No, 1000 < 1800 → in SLA
    assert snap.in_sla is True
