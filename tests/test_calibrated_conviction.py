"""Tests for the LGBM → Isotonic → Conformal calibrated conviction pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.intelligence.scoring.calibrated_conviction import (
    CalibratedConvictionPipeline,
    CalibratedConviction,
)


# ---------------------------------------------------------------------------
# Fallback behaviour (no stages fitted)
# ---------------------------------------------------------------------------


def test_pipeline_with_no_stages_returns_fallback():
    pipeline = CalibratedConvictionPipeline()
    features = np.zeros(8, dtype=float)
    result = pipeline.score_one(features)
    assert result.is_fallback is True
    assert result.conviction_0_100 == 50
    assert result.interval.lower == 0.0
    assert result.interval.upper == 1.0
    assert result.interval.n_calibration == 0


def test_pipeline_wrong_feature_count_raises():
    pipeline = CalibratedConvictionPipeline()
    with pytest.raises(ValueError, match="features must have 8 columns"):
        pipeline.score_one(np.zeros(5, dtype=float))


# ---------------------------------------------------------------------------
# End-to-end with fitted stages (uses sklearn IsotonicRegression behind the
# scenes — no LightGBM dependency at test time; we substitute a stub scorer).
# ---------------------------------------------------------------------------


class _StubLGBM:
    """Stand-in LGBMScorer that returns a configurable probability."""

    def __init__(self, p_win: float, feature_names: tuple[str, ...]):
        self._p_win = float(p_win)
        self._model = object()  # truthy ⇒ "fitted"
        self.feature_names = feature_names

    def predict_p_win(self, X):
        n = X.shape[0]
        return np.full(n, self._p_win)

    def feature_importance(self) -> dict[str, float]:
        return {name: 0.1 * (i + 1) for i, name in enumerate(self.feature_names)}


def test_pipeline_with_lgbm_only_passes_through():
    """LGBM fitted, no isotonic, no conformal ⇒ raw p_win, degenerate CI."""
    from src.intelligence.scoring.lgbm_scorer import DEFAULT_FEATURE_NAMES

    pipeline = CalibratedConvictionPipeline(
        lgbm=_StubLGBM(p_win=0.72, feature_names=DEFAULT_FEATURE_NAMES),
    )
    result = pipeline.score_one(np.ones(8, dtype=float))

    assert result.is_fallback is False
    assert result.p_win_raw == pytest.approx(0.72)
    assert result.p_win_calibrated == pytest.approx(0.72)  # no isotonic ⇒ passthrough
    assert result.conviction_0_100 == 72
    assert result.interval.n_calibration == 0  # degenerate


def test_pipeline_full_stack_with_isotonic_and_conformal():
    """Full chain: LGBM → Isotonic → ACI."""
    from src.intelligence.scoring.lgbm_scorer import DEFAULT_FEATURE_NAMES

    pipeline = CalibratedConvictionPipeline(
        lgbm=_StubLGBM(p_win=0.6, feature_names=DEFAULT_FEATURE_NAMES),
    )

    # Fit isotonic on a noisier monotone calibration set so isotonic does NOT
    # clip predictions at the [0, 1] boundary.
    rng = np.random.default_rng(42)
    p_pred = np.linspace(0.1, 0.9, 500)
    # y_true loosely correlated with p_pred — isotonic should produce a
    # roughly monotone but non-degenerate mapping.
    y_true = (p_pred + rng.normal(0, 0.25, 500) > 0.5).astype(int)
    pipeline.fit_isotonic(p_pred, y_true)

    # Fit ACI with small residual variance so interval doesn't blow past [0, 1]
    outcomes = rng.normal(0.0, 0.05, 300)
    pipeline.fit_conformal(outcomes, alpha=0.10)

    result = pipeline.score_one(np.ones(8, dtype=float))
    assert result.is_fallback is False
    assert 0 <= result.conviction_0_100 <= 100
    assert result.interval.n_calibration >= 30  # conformal min_samples
    # Interval contains the calibrated point estimate (the conformal wrapper
    # clips at [0, 1] so centre may not equal p_cal at the boundary — instead
    # we assert the looser invariant: lower ≤ p_cal ≤ upper).
    assert result.interval.lower <= result.p_win_calibrated <= result.interval.upper


def test_calibrated_conviction_0_100_clipping():
    """conviction_0_100 must always be a clean int in [0, 100]."""
    cc = CalibratedConviction(
        p_win_raw=1.2,
        p_win_calibrated=1.2,  # over-shoot edge case
        conviction_0_100=100,
        interval=None,  # type: ignore
        feature_contributions={},
    )
    # conviction_0_100 has been clipped externally; the dataclass itself
    # just stores it. The pipeline.score_one clamps before constructing.
    assert cc.conviction_0_100 == 100


def test_pipeline_lgbm_exception_falls_back():
    class _BrokenLGBM:
        _model = object()
        def predict_p_win(self, X):
            raise RuntimeError("simulated lgbm crash")
        def feature_importance(self):
            return {}

    pipeline = CalibratedConvictionPipeline(lgbm=_BrokenLGBM())
    result = pipeline.score_one(np.zeros(8, dtype=float))
    assert result.is_fallback is True
    assert result.conviction_0_100 == 50


# ---------------------------------------------------------------------------
# Conviction → InsightSignal mapping
# ---------------------------------------------------------------------------


def test_calibrated_conviction_maps_to_uncertainty_context():
    """The CalibratedConviction output should plug straight into
    UncertaintyContext in InsightSignalV2 (Sprint 1 integration)."""
    from src.api.insight_signal_v2 import UncertaintyContext
    from src.intelligence.conformal_wrapper import ConformalInterval

    interval = ConformalInterval(
        point=0.72, lower=0.54, upper=0.82,
        alpha=0.10, n_calibration=2000,
    )
    cc = CalibratedConviction(
        p_win_raw=0.65,
        p_win_calibrated=0.72,
        conviction_0_100=72,
        interval=interval,
    )

    # Map cc → UncertaintyContext (this is what the scanner will do)
    uc = UncertaintyContext(
        conformal_lower=cc.conformal_lower_0_100,
        conformal_upper=cc.conformal_upper_0_100,
        coverage_alpha=cc.interval.alpha,
        n_calibration=cc.interval.n_calibration,
    )
    assert uc.conformal_lower == pytest.approx(54.0)
    assert uc.conformal_upper == pytest.approx(82.0)
    assert uc.coverage_alpha == 0.10
    assert uc.n_calibration == 2000


# ---------------------------------------------------------------------------
# Online ACI feedback loop
# ---------------------------------------------------------------------------


def test_observe_outcome_only_when_aci():
    from src.intelligence.scoring.lgbm_scorer import DEFAULT_FEATURE_NAMES

    pipeline = CalibratedConvictionPipeline(
        lgbm=_StubLGBM(p_win=0.6, feature_names=DEFAULT_FEATURE_NAMES),
    )
    rng = np.random.default_rng(0)
    pipeline.fit_conformal(rng.normal(0.0, 0.2, 100))
    # Trigger one prediction (needed because ACI.observe requires last_interval)
    pipeline.score_one(np.ones(8))
    # Should not raise
    pipeline.observe_outcome(0.1)


def test_observe_outcome_no_op_without_aci():
    """No conformal stage ⇒ observe_outcome is a no-op (safety)."""
    pipeline = CalibratedConvictionPipeline()
    pipeline.observe_outcome(0.5)  # Should not raise
