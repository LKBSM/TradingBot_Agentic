"""Mondrian conformal prediction — Sprint 4 batch 4.1.

Stratifies the conformal calibration by **regime** so that exchangeability
holds within each stratum (low / normal / high vol — or BOCPD CP state).

Audit reference
---------------
- ``audits/2026-Q2/section_3_6_conformal.md`` P1-9 :
  "Pas de variante Mondrian (conformal stratifié par régime). Avec 2
  régimes vol distincts, exchangeabilité est plus crédible intra-régime."
- ``audits/2026-Q2/section_3_4_volatility.md`` P0-20 :
  "PICP empirique 43.6% pour cible 80% sur XAU 2024 → bandes non fiables."

Why Mondrian
------------
Split conformal assumes exchangeability of the calibration and test sets.
In time series, this is violated across regime changes (low vol cal vs
crisis vol test). Mondrian partitions the calibration set by a discrete
*taxonomy variable* (here : regime label) and computes per-stratum
quantiles. At query time the right stratum's quantile is used.

This restores the **conditional coverage** guarantee (PICP nominal within
each regime) at the cost of larger intervals during regime tails.

References
----------
- Boström et al. (2017). *Mondrian Conformal Regressors*. Proc. COPA.
- Angelopoulos & Bates (2024). §3.4 stratified conformal.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


class MondrianConformal:
    """Mondrian conformal regressor over a discrete regime taxonomy.

    Parameters
    ----------
    alpha
        Target miscoverage (e.g. ``0.05`` → 95 % intervals).
    min_per_stratum
        Minimum calibration size per stratum to publish an interval.
        Below threshold, the global quantile is used as a fallback.
    """

    def __init__(self, alpha: float = 0.05, min_per_stratum: int = 30):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = float(alpha)
        self.min_per_stratum = int(min_per_stratum)
        self._quantiles_per_stratum: dict[Any, tuple[float, float]] = {}
        self._global_quantiles: tuple[float, float] = (0.0, 0.0)
        self._fitted = False

    def fit(
        self,
        residuals: np.ndarray,
        regime_labels: np.ndarray,
    ) -> "MondrianConformal":
        """Fit per-stratum + global empirical quantiles."""
        residuals = np.asarray(residuals, dtype=float)
        labels = np.asarray(regime_labels)
        if residuals.shape != labels.shape:
            raise ValueError(
                f"residuals shape {residuals.shape} != labels shape {labels.shape}"
            )
        if residuals.size == 0:
            raise ValueError("Empty residuals")

        # Global fallback quantiles
        q_low = float(np.quantile(residuals, self.alpha / 2.0))
        q_high = float(np.quantile(residuals, 1.0 - self.alpha / 2.0))
        self._global_quantiles = (q_low, q_high)

        # Per-stratum
        by_stratum: dict[Any, list[float]] = defaultdict(list)
        for r, lbl in zip(residuals, labels):
            by_stratum[lbl].append(float(r))

        self._quantiles_per_stratum = {}
        for lbl, vals in by_stratum.items():
            if len(vals) < self.min_per_stratum:
                logger.warning(
                    "Stratum %r has %d residuals (< min %d) — will fall back to global",
                    lbl, len(vals), self.min_per_stratum,
                )
                continue
            arr = np.asarray(vals, dtype=float)
            ql = float(np.quantile(arr, self.alpha / 2.0))
            qh = float(np.quantile(arr, 1.0 - self.alpha / 2.0))
            self._quantiles_per_stratum[lbl] = (ql, qh)

        self._fitted = True
        logger.info(
            "MondrianConformal fitted on %d residuals across %d strata "
            "(%d with sufficient samples, alpha=%.3f)",
            len(residuals), len(by_stratum), len(self._quantiles_per_stratum),
            self.alpha,
        )
        return self

    def interval(
        self,
        point_prediction: float,
        regime_label: Any,
    ) -> tuple[float, float]:
        """Return ``(lower, upper)`` for a single query."""
        if not self._fitted:
            raise RuntimeError("MondrianConformal not fitted")
        ql, qh = self._quantiles_per_stratum.get(regime_label, self._global_quantiles)
        return (float(point_prediction + ql), float(point_prediction + qh))

    def coverage_per_stratum(
        self,
        residuals_test: np.ndarray,
        labels_test: np.ndarray,
    ) -> dict[Any, float]:
        """Empirical PICP per stratum on a held-out test set."""
        if not self._fitted:
            raise RuntimeError("MondrianConformal not fitted")
        residuals_test = np.asarray(residuals_test, dtype=float)
        labels_test = np.asarray(labels_test)
        out: dict[Any, float] = {}
        for lbl in np.unique(labels_test):
            mask = labels_test == lbl
            if not mask.any():
                continue
            ql, qh = self._quantiles_per_stratum.get(lbl, self._global_quantiles)
            covered = (residuals_test[mask] >= ql) & (residuals_test[mask] <= qh)
            out[lbl] = float(covered.mean())
        return out

    def is_fitted(self) -> bool:
        return self._fitted


__all__ = ["MondrianConformal"]
