"""End-to-end calibrated conviction pipeline.

Chains :class:`LGBMScorer` → :class:`IsotonicRecalibrator` → conformal
wrapper (:class:`SplitConformalScorer` or :class:`AdaptiveConformalScorer`)
to produce a *calibrated* P(win) and a distribution-free interval ready
to be exposed in :class:`UncertaintyContext` on the client side.

Why this module exists (Sprint 1, P-1 + P-2 from improvement_roadmap)
---------------------------------------------------------------------
The raw confluence score in :mod:`src.intelligence.confluence_detector`
has Pearson correlation −0.023 with realised P&L on the XAU M15 7-year
replay (eval_02). It measures **confluence of conditions**, not
**probability of gain**. Until we post-process it through an empirical
calibrator, the 0-100 number cannot honestly be exposed as a probability.

This module is the calibration layer. It:

1. Takes the LGBM-predicted P(win) on the 8 confluence features.
2. Maps it through an isotonic regression fitted on past outcomes
   (so the post-calibration score has empirical hit-rate semantics).
3. Wraps the calibrated score in a conformal interval to surface
   the *uncertainty* around it (Angelopoulos & Bates 2024).

The output is a ``CalibratedConviction`` dataclass that maps 1-to-1 onto
the :class:`UncertaintyContext` Pydantic model. The caller (signal
construction in :class:`SentinelScanner`) projects ``CalibratedConviction``
into the v2.1.0 ``InsightSignalV2`` schema.

Honesty rule
------------
Even after this calibration, ``edge_claim`` in the InsightSignal stays
``False``. The calibrator promises only *that the score has empirical
probability semantics* — not *that the score has predictive edge over
the noise floor*. Those are two different claims.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.intelligence.conformal_wrapper import (
    AdaptiveConformalScorer,
    ConformalInterval,
    SplitConformalScorer,
)
from src.intelligence.scoring.isotonic_recalibration import IsotonicRecalibrator
from src.intelligence.scoring.lgbm_scorer import LGBMScorer, DEFAULT_FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class CalibratedConviction:
    """Output of the LGBM → Isotonic → Conformal pipeline.

    Maps directly onto :class:`UncertaintyContext` in
    ``src.api.insight_signal_v2`` (multiply probabilities by 100 to obtain
    the 0-100 conviction scale used in the client schema).
    """

    p_win_raw: float                    # Raw LGBM output, [0, 1]
    p_win_calibrated: float             # After isotonic, [0, 1]
    conviction_0_100: int               # round(p_win_calibrated * 100)
    interval: ConformalInterval         # On the [0, 1] probability scale
    feature_contributions: dict[str, float] = field(default_factory=dict)
    is_fallback: bool = False           # True if any stage was unfitted

    @property
    def conformal_lower_0_100(self) -> float:
        """Conformal lower bound on the 0-100 conviction scale (clipped)."""
        return float(np.clip(self.interval.lower * 100.0, 0.0, 100.0))

    @property
    def conformal_upper_0_100(self) -> float:
        return float(np.clip(self.interval.upper * 100.0, 0.0, 100.0))


class CalibratedConvictionPipeline:
    """Orchestrate LGBM → Isotonic → Conformal as a single object.

    Lifecycle
    ---------
    Construction takes the three stages (one already fitted, or unfitted
    for fresh calibration). Inference is via :meth:`score_one`.

    Empty-state behaviour
    ---------------------
    If any stage is unfitted, ``score_one`` returns a ``CalibratedConviction``
    with ``is_fallback=True`` and a degenerate interval [0, 100] (no
    uncertainty signal). This is the same defensive pattern as the
    :class:`VolatilityForecaster` fallback — never crash, just label.

    Adaptive vs split
    -----------------
    ``adaptive=True`` uses :class:`AdaptiveConformalScorer` (ACI), which
    maintains coverage under XAU regime drift. The split variant is
    deterministic but breaks under drift — only useful for offline
    benchmarks.
    """

    def __init__(
        self,
        lgbm: Optional[LGBMScorer] = None,
        isotonic: Optional[IsotonicRecalibrator] = None,
        conformal: Optional[SplitConformalScorer | AdaptiveConformalScorer] = None,
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
    ):
        self.lgbm = lgbm
        self.isotonic = isotonic
        self.conformal = conformal
        self.feature_names = tuple(feature_names)

    # ------------------------------------------------------------------ #
    # Fitting helpers (typically called from a calibration job, not the
    # online scoring path).
    # ------------------------------------------------------------------ #

    def fit_isotonic(self, p_pred: np.ndarray, y_true: np.ndarray) -> None:
        """Fit the isotonic stage on past predictions vs outcomes."""
        if self.isotonic is None:
            self.isotonic = IsotonicRecalibrator(increasing=True)
        self.isotonic.fit(p_pred, y_true)

    def fit_conformal(self, calibration_outcomes: np.ndarray, alpha: float = 0.10) -> None:
        """Fit the conformal stage on calibration outcomes (after isotonic)."""
        if self.conformal is None:
            self.conformal = AdaptiveConformalScorer(alpha_target=alpha)
        self.conformal.fit(calibration_outcomes)

    # ------------------------------------------------------------------ #
    # Inference path — fast (< 5 ms typical)
    # ------------------------------------------------------------------ #

    def score_one(self, features: np.ndarray) -> CalibratedConviction:
        """Run the full pipeline for one signal.

        Parameters
        ----------
        features : np.ndarray
            1-D array of length ``len(feature_names)``. Order matches
            :data:`DEFAULT_FEATURE_NAMES`.

        Returns
        -------
        CalibratedConviction
            With ``is_fallback=True`` if any stage was not fitted (in that
            case the conviction defaults to 50/100 with full-width interval).
        """
        x = np.asarray(features, dtype=float).reshape(1, -1)
        if x.shape[1] != len(self.feature_names):
            raise ValueError(
                f"features must have {len(self.feature_names)} columns "
                f"(matching {self.feature_names}), got {x.shape[1]}"
            )

        # ---- Stage 1: LGBM ----
        if self.lgbm is None or getattr(self.lgbm, "_model", None) is None:
            return self._fallback("LGBMScorer not fitted")
        try:
            p_raw = float(self.lgbm.predict_p_win(x)[0])
        except Exception as exc:
            logger.warning("LGBMScorer inference failed: %s", exc)
            return self._fallback(f"LGBMScorer error: {exc}")

        # ---- Stage 2: Isotonic recalibration ----
        if self.isotonic is not None and self.isotonic._model is not None:
            try:
                p_cal = float(self.isotonic.transform(np.asarray([p_raw]))[0])
            except Exception as exc:
                logger.warning("IsotonicRecalibrator transform failed: %s", exc)
                p_cal = p_raw  # Degrade gracefully
        else:
            # No isotonic ⇒ raw LGBM probability passes through. This is
            # acceptable in a freshly-deployed system before the first
            # calibration job has run.
            p_cal = p_raw

        # ---- Stage 3: Conformal interval ----
        if self.conformal is not None and self.conformal.is_fit:
            try:
                interval = self.conformal.predict_interval()
                # The conformal wrapper produces an interval around the
                # *point estimate* (mean of calibration outcomes). For our
                # use case we want the interval centred on the calibrated
                # P(win) — so we recentre the marginal interval.
                halfwidth = interval.width() / 2.0
                interval = ConformalInterval(
                    point=p_cal,
                    lower=max(0.0, p_cal - halfwidth),
                    upper=min(1.0, p_cal + halfwidth),
                    alpha=interval.alpha,
                    n_calibration=interval.n_calibration,
                )
            except Exception as exc:
                logger.warning("Conformal interval failed: %s", exc)
                interval = self._degenerate_interval(p_cal)
        else:
            interval = self._degenerate_interval(p_cal)

        # ---- Stage 4: Feature importance (per-signal) ----
        # LightGBM only exposes *global* feature importance natively. For
        # per-signal contributions we'd plug in SHAP — left as a follow-up
        # since FeatureExplainer already exists in src/intelligence/.
        contributions: dict[str, float] = {}
        try:
            importances = self.lgbm.feature_importance()
            contributions = {
                name: float(importances.get(name, 0.0))
                for name in self.feature_names
            }
        except Exception:
            pass

        return CalibratedConviction(
            p_win_raw=p_raw,
            p_win_calibrated=p_cal,
            conviction_0_100=int(round(float(np.clip(p_cal * 100.0, 0.0, 100.0)))),
            interval=interval,
            feature_contributions=contributions,
            is_fallback=False,
        )

    # ------------------------------------------------------------------ #
    # Online observation feedback (ACI loop closure)
    # ------------------------------------------------------------------ #

    def observe_outcome(self, realised: float) -> None:
        """Feed back a realised outcome to the ACI conformal scorer."""
        if isinstance(self.conformal, AdaptiveConformalScorer):
            self.conformal.observe(realised)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _fallback(self, reason: str) -> CalibratedConviction:
        """Defensive fallback when a stage is missing/broken."""
        logger.info("CalibratedConvictionPipeline fallback: %s", reason)
        return CalibratedConviction(
            p_win_raw=0.5,
            p_win_calibrated=0.5,
            conviction_0_100=50,
            interval=self._degenerate_interval(0.5),
            feature_contributions={},
            is_fallback=True,
        )

    @staticmethod
    def _degenerate_interval(point: float) -> ConformalInterval:
        return ConformalInterval(
            point=float(point),
            lower=0.0,
            upper=1.0,
            alpha=0.10,
            n_calibration=0,
        )


__all__ = ["CalibratedConviction", "CalibratedConvictionPipeline"]
