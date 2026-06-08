"""Conformal Prediction wrapper for the confluence signal pipeline.

Pilier 2 of the institutional quant transformation plan (see
``reports/institutional_quant_transformation_plan.md`` §4 Pilier 2).

Purpose
-------
The current confluence pipeline emits a 0-100 score per candidate trade. The
state machine then gates on the score (enter >= 75, exit <= 55). What it
*cannot* do today is tell the client "how confident are we that this score
is right?" — and more importantly, it cannot reject low-confidence trades.

Conformal Prediction (Angelopoulos & Bates 2024, *Foundations of Conformal
Prediction*, arXiv:2411.11824) gives us **distribution-free intervals with
guaranteed marginal coverage** : if we calibrate the wrapper on past signals
and observed outcomes, then for any new score `s`, the wrapper returns an
interval ``[lo, hi]`` such that the *true* expected outcome lies inside the
interval with probability ≥ 1 − α, regardless of the underlying distribution.

We expose two variants:

1. **Split Conformal** (vanilla, exchangeable assumption)
2. **ACI** — Adaptive Conformal Inference (Gibbs & Candès 2021,
   arXiv:2106.00170). Tracks the empirical miscoverage online and adjusts
   the nominal α to maintain the target coverage even under distribution
   drift. Important for time series where exchangeability is violated.

How it plugs into the existing pipeline
----------------------------------------
Given an already-built model that maps `(features) → confluence_score` and a
ground-truth `outcome` (in our case the realised return-on-risk after
``signal_lifetime_bars``), the wrapper computes nonconformity scores on a
held-out calibration set, then for any new (score, features) tuple it
returns ``(lo, hi)``. The pipeline only trades when ``lo`` exceeds the
"break-even" threshold (default 0.0 = "non-negative expected return").

This is a **reject-option** filter — not a new signal. Trades that pass the
existing state machine but whose conformal lower bound is below threshold
are dropped. Empirically this converts a low-PF SMC strategy (PF 0.94 in the
state-machine replay) into a selective one with higher PF but fewer trades.

Reference
---------
- Angelopoulos, A.N. & Bates, S. (2024). *Foundations of Conformal
  Prediction*. arXiv:2411.11824.
- Gibbs, I. & Candès, E.J. (2021). *Adaptive Conformal Inference Under
  Distribution Shift*. NeurIPS 2021. arXiv:2106.00170.
- Kato, M. (2024). *Conformal Predictive Portfolio Selection*.
  arXiv:2410.16333.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Result containers
# =============================================================================


@dataclass
class ConformalInterval:
    """A conformal prediction interval for a single query."""

    point: float          # Point prediction (e.g. recent calibration mean)
    lower: float          # 1 − α lower bound
    upper: float          # 1 − α upper bound
    alpha: float          # Nominal miscoverage rate
    n_calibration: int    # Calibration set size used

    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, x: float) -> bool:
        return self.lower <= x <= self.upper


# =============================================================================
# Split Conformal — vanilla, exchangeable assumption
# =============================================================================


class SplitConformalScorer:
    """Split Conformal prediction over a residual buffer.

    The nonconformity score is the *absolute residual* between the
    point-predicted outcome (here: the mean of calibration outcomes, since
    we treat the score itself as the predictor and want a band around it)
    and the realised outcome. This gives a symmetric ± half-width interval.

    More sophisticated variants (e.g. CQR — Romano et al. 2019) regress
    residuals on features; for the M15 confluence pipeline the marginal
    interval is sufficient and computationally trivial.

    Usage
    -----
    >>> scorer = SplitConformalScorer(alpha=0.05)
    >>> scorer.fit(outcomes_history)        # 1-D array of past R-multiples
    >>> interval = scorer.predict_interval(current_score)
    >>> assert interval.lower > 0.0         # only trade if positive expectancy
    """

    def __init__(self, alpha: float = 0.10, min_samples: int = 30) -> None:
        if not 0.0 < alpha < 0.5:
            raise ValueError(f"alpha must be in (0, 0.5), got {alpha}")
        self.alpha = float(alpha)
        self.min_samples = int(min_samples)
        self._calibration: np.ndarray = np.empty(0)
        self._point_estimate: float = 0.0
        self._q_hat: float = float("inf")  # Unbounded until fit

    @property
    def is_fit(self) -> bool:
        return len(self._calibration) >= self.min_samples

    def fit(self, outcomes: np.ndarray, point_estimator: Optional[float] = None) -> None:
        """Calibrate from a history of realised outcomes.

        outcomes
            1-D array of realised R-multiples (or any return-like metric
            on a consistent scale). NaNs are dropped.
        point_estimator
            Optional override for the point prediction; defaults to the
            mean of the calibration outcomes.
        """
        out = np.asarray(outcomes, dtype=float)
        out = out[np.isfinite(out)]
        if len(out) < self.min_samples:
            raise ValueError(
                f"need at least {self.min_samples} calibration samples, "
                f"got {len(out)}"
            )
        self._calibration = out
        self._point_estimate = (
            float(point_estimator) if point_estimator is not None else float(out.mean())
        )
        # Nonconformity scores: |outcome - point|
        scores = np.abs(out - self._point_estimate)
        # Quantile with finite-sample correction (Angelopoulos & Bates §3.2)
        n = len(scores)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        self._q_hat = float(np.quantile(scores, q_level))

    def predict_interval(self, _query_features: Optional[np.ndarray] = None) -> ConformalInterval:
        """Return a marginal conformal interval around the point prediction.

        The current implementation is feature-independent (marginal CP). The
        query argument is accepted for API forward-compatibility with CQR or
        feature-conditional variants.
        """
        if not self.is_fit:
            raise RuntimeError("SplitConformalScorer must be fit before predict_interval")
        return ConformalInterval(
            point=self._point_estimate,
            lower=self._point_estimate - self._q_hat,
            upper=self._point_estimate + self._q_hat,
            alpha=self.alpha,
            n_calibration=len(self._calibration),
        )

    def should_reject(self, breakeven: float = 0.0) -> bool:
        """Return True if the conformal lower bound is below ``breakeven``.

        Use this as a reject-option gate: when True, the candidate trade
        should be dropped because the conformal lower bound on its expected
        outcome does not clear the minimum acceptable threshold.
        """
        interval = self.predict_interval()
        return interval.lower <= breakeven


# =============================================================================
# Adaptive Conformal Inference (Gibbs & Candès 2021)
# =============================================================================


@dataclass
class ACIState:
    """Online state of the Adaptive Conformal Inference adjustment."""

    alpha_target: float           # Nominal coverage target (e.g. 0.10)
    alpha_current: float          # Current adjusted alpha used for quantile
    gamma: float                  # Step size (Gibbs-Candès default 0.05)
    miscoverage_history: list = field(default_factory=list)


class AdaptiveConformalScorer:
    """Online-adaptive conformal prediction tracking miscoverage.

    Each time the wrapper observes an outcome, it checks whether the realised
    outcome fell inside the most-recent interval. If it did **not**, the
    adjustment increases the conformal radius (lowers alpha_current); if it
    did, the radius shrinks. Long-run miscoverage converges to ``alpha_target``
    even under distribution drift.

    Crucial for our use case because XAU regime shifts (FOMC pivots, geopolitical
    shocks) violate the exchangeability assumption.

    Reference: Gibbs & Candès (2021), Algorithm 1.
    """

    def __init__(
        self,
        alpha_target: float = 0.10,
        gamma: float = 0.05,
        min_samples: int = 30,
        buffer_size: int = 500,
    ) -> None:
        if not 0.0 < alpha_target < 0.5:
            raise ValueError(f"alpha_target must be in (0, 0.5), got {alpha_target}")
        if not 0.0 < gamma < 1.0:
            raise ValueError(f"gamma must be in (0, 1), got {gamma}")

        self.state = ACIState(
            alpha_target=float(alpha_target),
            alpha_current=float(alpha_target),
            gamma=float(gamma),
        )
        self.min_samples = int(min_samples)
        self.buffer_size = int(buffer_size)
        self._calibration_buffer: list = []
        self._point_estimate: float = 0.0
        self._last_interval: Optional[ConformalInterval] = None

    @property
    def is_fit(self) -> bool:
        return len(self._calibration_buffer) >= self.min_samples

    def fit(self, outcomes: np.ndarray) -> None:
        out = np.asarray(outcomes, dtype=float)
        out = out[np.isfinite(out)]
        if len(out) < self.min_samples:
            raise ValueError(
                f"need at least {self.min_samples} samples, got {len(out)}"
            )
        # Seed the buffer with the most recent `buffer_size` observations
        if len(out) > self.buffer_size:
            out = out[-self.buffer_size :]
        self._calibration_buffer = list(out)
        self._point_estimate = float(np.mean(out))

    def predict_interval(self) -> ConformalInterval:
        if not self.is_fit:
            raise RuntimeError("AdaptiveConformalScorer must be fit before predict_interval")
        cal = np.asarray(self._calibration_buffer)
        residuals = np.abs(cal - self._point_estimate)
        n = len(residuals)
        q_level = min(1.0, np.ceil((n + 1) * (1 - self.state.alpha_current)) / n)
        q_hat = float(np.quantile(residuals, q_level))
        interval = ConformalInterval(
            point=self._point_estimate,
            lower=self._point_estimate - q_hat,
            upper=self._point_estimate + q_hat,
            alpha=self.state.alpha_current,
            n_calibration=n,
        )
        self._last_interval = interval
        return interval

    def observe(self, realised_outcome: float) -> None:
        """Update the running miscoverage tally and adapt alpha.

        Must be called *after* ``predict_interval`` for the same query, with
        the realised outcome ground truth. Sliding-window calibration buffer
        is also updated.
        """
        if self._last_interval is None:
            logger.debug("ACI.observe called before any predict_interval; skipping")
        else:
            miscovered = not self._last_interval.contains(realised_outcome)
            self.state.miscoverage_history.append(int(miscovered))
            # Gibbs-Candès update: alpha ← alpha + γ * (α_target - miscovered)
            err = int(miscovered)
            self.state.alpha_current = float(
                np.clip(
                    self.state.alpha_current
                    + self.state.gamma * (self.state.alpha_target - err),
                    1e-4,
                    0.5,
                )
            )

        # Update the calibration buffer with the new observation
        self._calibration_buffer.append(float(realised_outcome))
        if len(self._calibration_buffer) > self.buffer_size:
            # Drop oldest
            self._calibration_buffer = self._calibration_buffer[-self.buffer_size :]
        # Refresh the point estimate (rolling mean)
        self._point_estimate = float(np.mean(self._calibration_buffer))

    def empirical_coverage(self) -> Optional[float]:
        """Fraction of past intervals that contained their realised outcome."""
        h = self.state.miscoverage_history
        if not h:
            return None
        return 1.0 - float(np.mean(h))

    def should_reject(self, breakeven: float = 0.0) -> bool:
        interval = self.predict_interval()
        return interval.lower <= breakeven


# =============================================================================
# Reject-option helper for backtest replay
# =============================================================================


def apply_conformal_filter(
    candidate_returns: np.ndarray,
    calibration_returns: np.ndarray,
    alpha: float = 0.10,
    breakeven: float = 0.0,
    adaptive: bool = True,
    return_mask: bool = False,
):
    """Apply a conformal reject-option filter to a candidate trade sequence.

    Walks the candidate returns chronologically. For each candidate, fits the
    conformal scorer on the trailing calibration set (= the most recent
    ``len(calibration_returns)`` realised outcomes including past candidates
    that *did* trade) and keeps the trade only if the conformal lower bound
    > ``breakeven``.

    Parameters
    ----------
    candidate_returns
        1-D returns of trades that the upstream pipeline would have taken
        (in chronological order).
    calibration_returns
        Initial calibration outcomes (must have >= 30 samples).
    alpha
        Nominal miscoverage. 0.10 → 90% intervals.
    breakeven
        Reject when conformal lower bound is ≤ this. 0.0 = positive expectancy.
    adaptive
        Use ACI (recommended for time series) vs SplitConformal.
    return_mask
        If True, also return a boolean mask of which trades were kept.

    Returns
    -------
    filtered_returns : np.ndarray
        Returns of the trades that passed the conformal filter, in order.
    mask : np.ndarray (only if return_mask=True)
        Boolean array same length as candidate_returns.
    """
    cal = np.asarray(calibration_returns, dtype=float)
    cal = cal[np.isfinite(cal)]
    cand = np.asarray(candidate_returns, dtype=float)

    if adaptive:
        scorer = AdaptiveConformalScorer(alpha_target=alpha)
    else:
        scorer = SplitConformalScorer(alpha=alpha)
    scorer.fit(cal)

    kept = []
    mask = np.zeros(len(cand), dtype=bool)
    for i, r in enumerate(cand):
        if not np.isfinite(r):
            continue
        if not scorer.should_reject(breakeven=breakeven):
            kept.append(r)
            mask[i] = True
            # For adaptive scorers, feed back the realised outcome
            if adaptive:
                scorer.observe(r)
        elif adaptive:
            # Even when rejected, ACI needs the observed outcome to adapt
            # alpha (counterfactual: had we taken it, what would the
            # coverage have been?). This is the Gibbs-Candès recommendation.
            scorer.observe(r)

    out = np.asarray(kept, dtype=float)
    if return_mask:
        return out, mask
    return out
