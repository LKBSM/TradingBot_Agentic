"""Logistic regression L1 scorer — Sprint 4 batch 4.2.

Replaces the additive ``ConfluenceDetector`` score (Pearson −0.008, Brier
skill −0.022, audit 3.3) with a multi-feature logistic regression on the
8 ``weighted_scores`` components.

Why logistic L1 (not isotonic)
------------------------------
Audit 3.3 found :

- Pearson(additive_score, R) = −0.008 → no linear signal.
- Spearman = −0.019 → no rank signal either.
- Isotonic regression preserves rank order → if Spearman ≈ 0, isotonic
  cannot recover a calibration (no monotonic signal to align).

A multi-feature classifier (logistic L1) projects the 8 components into
a new coordinate where some linear combination *may* be predictive — and
L1 sparsity simultaneously drops components without information
(audit 3.3 P1-5 : ``OB ↔ Retest`` Cramér's V = 0.489, news quasi-saturated
99.8 % activation).

Public API
----------
>>> scorer = LogisticL1Scorer()
>>> scorer.fit(X, y)         # X: (n, 8) components, y: realised R > 0
>>> p_win = scorer.predict_proba(x_new)[:, 1]
>>> # then apply isotonic for final calibration

Inputs / outputs
----------------
- X : np.ndarray (n_trades, 8) — the 8 weighted_scores at signal-time.
- y : np.ndarray (n_trades,)   — binary (R > 0 = win) or float (R) for
                                  regression variant.
- predict_proba : (n_new, 2)   — P(loss), P(win) per signal.

Calibration loop
----------------
This module exposes the *static* fit/predict interface. The walk-forward
calibration loop (refit every K trades, drift detection) lives in
``src.intelligence.scoring.calibration_loop`` (Sprint 4 batch 4.3).

Status (Sprint 4 prep) : **scaffold + sklearn-backed implementation**. The
class is wireable but **requires** the 8 per-component scores to be
persisted at signal-time (currently only the final additive score is
stored).

Reference
---------
- ``audits/2026-Q2/section_3_3_confluence.md`` — P0 findings + reco.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# 8 components per ConfluenceDetector contract
DEFAULT_COMPONENT_NAMES = (
    "smc_structure",   # BOS / CHOCH
    "order_blocks",
    "fvg",
    "retest",
    "regime",
    "vol_forecast",
    "news",
    "momentum_rsi_div",
)


class LogisticL1Scorer:
    """Sparse logistic regression scorer over the 8 confluence components.

    Wraps :class:`sklearn.linear_model.LogisticRegression` with
    ``penalty='l1', solver='saga'`` and a configurable inverse-regularisation
    strength ``C``. Returns calibrated probabilities via the model's
    built-in sigmoid.

    Notes
    -----
    The default ``C=1.0`` should be cross-validated per actif. Sprint 4
    batch 4.2 will run a 5-fold time-series CV grid.
    """

    def __init__(
        self,
        C: float = 1.0,
        component_names: tuple[str, ...] = DEFAULT_COMPONENT_NAMES,
        random_state: int = 42,
        class_weight: Optional[str] = "balanced",
    ):
        self.C = float(C)
        self.component_names = tuple(component_names)
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self._model = None  # populated by fit()

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticL1Scorer":
        """Fit the L1 logistic regression on (n, n_components) features."""
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n, n_components), got shape {X.shape}")
        if X.shape[1] != len(self.component_names):
            logger.warning(
                "X has %d columns but %d component names — alignment may be off",
                X.shape[1], len(self.component_names),
            )

        self._model = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=2000,
            n_jobs=1,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LogisticL1Scorer not fitted yet — call fit() first.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._model.predict_proba(X)

    def predict_p_win(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def coefficients(self) -> dict[str, float]:
        """Return non-zero coefficients per component (sparse L1)."""
        if self._model is None:
            raise RuntimeError("LogisticL1Scorer not fitted yet")
        coefs = self._model.coef_[0]
        return dict(zip(self.component_names, coefs.tolist()))

    def non_zero_components(self) -> list[str]:
        """Return the names of components kept by L1 sparsity."""
        return [n for n, c in self.coefficients().items() if abs(c) > 1e-8]


__all__ = ["LogisticL1Scorer", "DEFAULT_COMPONENT_NAMES"]
