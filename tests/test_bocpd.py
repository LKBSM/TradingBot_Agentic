"""Tests for src/intelligence/bocpd.py (REGIME-1.2).

Per DoD: 1 test pytest synthetic step-change. Plus performance + invariants.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src.intelligence.bocpd import (
    DEFAULT_HAZARD_INV,
    BOCPDState,
    bocpd_run,
    bocpd_step,
    expected_run_length,
)


# ---------------------------------------------------------------------------
# DoD: synthetic step-change detection
# ---------------------------------------------------------------------------


def test_bocpd_detects_drastic_mean_shift_at_known_index():
    """**REGIME-1.2 DoD**: with a clear step-change at t=100 (mean 0 → 5),
    cp_prob must spike at the changepoint and exceed the steady-state
    baseline by at least an order of magnitude."""
    rng = np.random.default_rng(seed=42)
    pre = rng.normal(0.0, 0.5, 100)
    post = rng.normal(5.0, 0.5, 100)
    data = np.concatenate([pre, post])

    cp = bocpd_run(data, hazard_inv=240)

    # Steady-state baseline (well into pre-segment, away from start jitter)
    baseline = float(np.median(cp[40:90]))
    # Changepoint detection window — allow ±5 bars of latency
    peak_window = cp[95:115]
    peak = float(peak_window.max())

    assert peak > 10 * max(baseline, 0.001), (
        f"changepoint peak {peak:.4f} should be >>10× baseline {baseline:.4f}"
    )
    # Peak should land in the immediate vicinity of t=100
    assert 95 <= 95 + int(peak_window.argmax()) <= 105


# ---------------------------------------------------------------------------
# Variance-shift detection (subtler, slower)
# ---------------------------------------------------------------------------


def test_bocpd_detects_variance_shift_eventually():
    """A smaller variance shift should also produce an elevated cp_prob,
    even if not as sharp as a mean shift."""
    rng = np.random.default_rng(seed=7)
    pre = rng.normal(0.0, 0.3, 200)
    post = rng.normal(0.0, 1.5, 200)  # 5× variance
    data = np.concatenate([pre, post])

    cp = bocpd_run(data, hazard_inv=240)

    baseline = float(np.median(cp[50:150]))
    # Variance shift detection can lag — search a wider window
    peak = float(cp[195:230].max())
    assert peak > 5 * max(baseline, 0.001)


# ---------------------------------------------------------------------------
# No false positive on steady noise
# ---------------------------------------------------------------------------


def test_bocpd_steady_noise_keeps_cp_prob_low():
    """Pure noise with no changepoint should produce only sporadic, modest
    cp_prob spikes (none above 0.5 in expectation for 1k steps with hazard 240)."""
    rng = np.random.default_rng(seed=11)
    data = rng.normal(0.0, 1.0, 1000)
    cp = bocpd_run(data, hazard_inv=240)

    # Drop the first few burn-in steps where prior dominates
    p99 = float(np.quantile(cp[20:], 0.99))
    assert p99 < 0.5, f"99th percentile cp_prob {p99:.4f} too high on pure noise"


# ---------------------------------------------------------------------------
# Performance gate (REGIME-1.2 KPI: <100ms per bar)
# ---------------------------------------------------------------------------


def test_bocpd_step_latency_under_100ms():
    rng = np.random.default_rng(seed=99)
    state = BOCPDState(max_run_length=300, hazard_inv=DEFAULT_HAZARD_INV)
    # Warm up
    for _ in range(10):
        bocpd_step(state, float(rng.normal()))

    n = 500
    t0 = time.perf_counter()
    for _ in range(n):
        bocpd_step(state, float(rng.normal()))
    elapsed = time.perf_counter() - t0
    per_step_ms = (elapsed / n) * 1000
    # KPI is 100ms, but the implementation is comfortably <5ms — fail loudly
    # if a future change pushes it 10×.
    assert per_step_ms < 50.0, (
        f"BOCPD step latency {per_step_ms:.2f}ms is over 50ms (regression!)"
    )


# ---------------------------------------------------------------------------
# State invariants
# ---------------------------------------------------------------------------


def test_run_length_probs_sum_to_one_after_each_step():
    rng = np.random.default_rng(seed=3)
    state = BOCPDState(max_run_length=200, hazard_inv=240)
    for _ in range(100):
        bocpd_step(state, float(rng.normal()))
        s = float(state.run_length_probs.sum())
        assert abs(s - 1.0) < 1e-6


def test_first_step_returns_cp_prob_equal_to_hazard():
    """At t=0 the only existing run is r=0 with posterior = prior. The
    posterior-predictive equals the prior-predictive, so growth and cp
    factors cancel out and cp_prob collapses to the bare hazard.
    This is the correct mathematical behaviour, not a bug."""
    state = BOCPDState(max_run_length=200, hazard_inv=240)
    cp_first = bocpd_step(state, 0.0)
    assert cp_first == pytest.approx(state.hazard, abs=1e-6)


def test_expected_run_length_grows_during_steady_segment():
    rng = np.random.default_rng(seed=15)
    state = BOCPDState(max_run_length=300, hazard_inv=240)
    elens = []
    for i in range(150):
        bocpd_step(state, float(rng.normal()))
        if i in (10, 50, 100, 149):
            elens.append(expected_run_length(state))
    # Run length grows monotonically during a steady segment
    assert elens[0] < elens[1] < elens[2] < elens[3]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_bocpd_run_deterministic_for_same_input():
    rng = np.random.default_rng(seed=21)
    data = rng.normal(0, 1, 100)
    cp1 = bocpd_run(data, hazard_inv=240)
    cp2 = bocpd_run(data, hazard_inv=240)
    np.testing.assert_array_equal(cp1, cp2)
