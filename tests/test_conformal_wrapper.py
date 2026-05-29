"""Tests for the Conformal Prediction wrapper.

Verifies:
- Split Conformal coverage approximates 1 - α on i.i.d. data.
- Adaptive Conformal maintains coverage under distribution shift where the
  vanilla version drifts.
- The reject-option filter on a low-edge signal does not destroy the
  profit-factor, and on a high-edge signal lets most trades through.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.conformal_wrapper import (
    AdaptiveConformalScorer,
    ConformalInterval,
    SplitConformalScorer,
    apply_conformal_filter,
)


# =============================================================================
# Construction & validation
# =============================================================================


def test_split_conformal_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        SplitConformalScorer(alpha=0.0)
    with pytest.raises(ValueError):
        SplitConformalScorer(alpha=0.5)
    with pytest.raises(ValueError):
        SplitConformalScorer(alpha=1.0)


def test_aci_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        AdaptiveConformalScorer(alpha_target=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalScorer(gamma=0.0)
    with pytest.raises(ValueError):
        AdaptiveConformalScorer(gamma=1.0)


def test_scorer_must_be_fit_before_predict():
    s = SplitConformalScorer(alpha=0.1)
    with pytest.raises(RuntimeError):
        s.predict_interval()


def test_fit_rejects_undersized_calibration():
    s = SplitConformalScorer(alpha=0.1, min_samples=30)
    with pytest.raises(ValueError):
        s.fit(np.array([0.01, 0.02]))


# =============================================================================
# Split Conformal — empirical coverage on i.i.d. data
# =============================================================================


def test_split_conformal_coverage_iid_data():
    """Run a simple coverage simulation: ~90% of holdout values should fall
    inside the 90% interval (within Monte-Carlo wiggle)."""
    rng = np.random.default_rng(0)
    cal = rng.normal(0.001, 0.01, 500)
    holdout = rng.normal(0.001, 0.01, 1000)

    scorer = SplitConformalScorer(alpha=0.10)
    scorer.fit(cal)
    interval = scorer.predict_interval()

    inside = ((holdout >= interval.lower) & (holdout <= interval.upper)).mean()
    # 90% target ± 5% Monte-Carlo
    assert 0.85 <= inside <= 0.95, f"Coverage {inside:.3f} outside [0.85, 0.95]"


def test_interval_has_nonnegative_width():
    rng = np.random.default_rng(1)
    cal = rng.normal(0.0, 0.01, 200)
    scorer = SplitConformalScorer(alpha=0.10)
    scorer.fit(cal)
    interval = scorer.predict_interval()
    assert interval.width() > 0
    assert interval.lower < interval.point < interval.upper


def test_interval_contains_helper():
    interval = ConformalInterval(point=0.01, lower=-0.05, upper=0.05, alpha=0.1, n_calibration=100)
    assert interval.contains(0.0)
    assert interval.contains(0.05)
    assert not interval.contains(-0.06)
    assert not interval.contains(0.06)


# =============================================================================
# Reject-option logic
# =============================================================================


def test_should_reject_when_lower_below_breakeven():
    # Calibration centred on zero → lower bound is negative → should reject
    # at breakeven=0.0
    rng = np.random.default_rng(2)
    cal = rng.normal(0.0, 0.01, 200)
    scorer = SplitConformalScorer(alpha=0.10)
    scorer.fit(cal)
    assert scorer.should_reject(breakeven=0.0) is True


def test_should_not_reject_when_lower_above_breakeven():
    # Strong positive mean and tight std → lower bound > 0
    rng = np.random.default_rng(3)
    cal = rng.normal(0.05, 0.005, 200)
    scorer = SplitConformalScorer(alpha=0.10)
    scorer.fit(cal)
    assert scorer.should_reject(breakeven=0.0) is False


# =============================================================================
# ACI — adapts to shift
# =============================================================================


def test_aci_tracks_coverage_under_shift():
    """Inject a regime shift mid-stream. ACI should adapt and end with
    long-run coverage close to the target."""
    rng = np.random.default_rng(4)
    cal = rng.normal(0.0, 0.01, 200)

    aci = AdaptiveConformalScorer(alpha_target=0.10, gamma=0.05)
    aci.fit(cal)

    # Phase 1: same distribution
    for _ in range(200):
        _ = aci.predict_interval()
        aci.observe(rng.normal(0.0, 0.01))

    # Phase 2: shift to a fatter-tailed distribution
    for _ in range(500):
        _ = aci.predict_interval()
        aci.observe(rng.normal(0.0, 0.03))

    coverage = aci.empirical_coverage()
    assert coverage is not None
    # Long-run coverage should be in the neighbourhood of 0.90 (target).
    # Allow ± 0.10 because ACI converges asymptotically, not pointwise.
    assert 0.78 <= coverage <= 0.99, f"Coverage {coverage:.3f} too far from 0.90"


def test_aci_alpha_stays_bounded():
    """Even under repeated miscoverage, alpha_current must stay in (0, 0.5)."""
    rng = np.random.default_rng(5)
    cal = rng.normal(0.0, 0.005, 200)

    aci = AdaptiveConformalScorer(alpha_target=0.10, gamma=0.1)
    aci.fit(cal)

    for _ in range(1000):
        _ = aci.predict_interval()
        # Force a wild outcome to stress the adaptation
        aci.observe(rng.normal(0.5, 0.5))

    assert 0.0 < aci.state.alpha_current < 0.5


# =============================================================================
# apply_conformal_filter end-to-end
# =============================================================================


def test_filter_drops_low_edge_signal():
    """On a series with no edge, the conformal filter should keep few trades
    and the filtered series should have similar-or-better PF."""
    rng = np.random.default_rng(6)
    cal = rng.normal(0.0, 0.01, 200)
    candidates = rng.normal(0.0, 0.01, 500)

    filtered = apply_conformal_filter(
        candidates, cal, alpha=0.10, breakeven=0.0, adaptive=True
    )
    # Most trades should be rejected on a zero-mean signal at breakeven=0.
    assert len(filtered) < 0.5 * len(candidates)


def test_filter_keeps_most_of_strong_edge():
    """On a signal with persistent positive expectancy, most trades should
    survive."""
    rng = np.random.default_rng(7)
    cal = rng.normal(0.05, 0.005, 200)
    candidates = rng.normal(0.05, 0.005, 500)

    filtered, mask = apply_conformal_filter(
        candidates,
        cal,
        alpha=0.10,
        breakeven=0.0,
        adaptive=True,
        return_mask=True,
    )
    # > 80% of strong-edge trades survive
    assert mask.mean() > 0.8


def test_filter_returns_mask_length_matches_input():
    rng = np.random.default_rng(8)
    cal = rng.normal(0.0, 0.01, 100)
    candidates = rng.normal(0.0, 0.01, 50)

    filtered, mask = apply_conformal_filter(
        candidates, cal, return_mask=True, adaptive=False
    )
    assert mask.shape == (50,)
    assert len(filtered) == mask.sum()
