"""Isotonic post-hoc recalibration — Sprint 4 batch 4.2 follow-up.

After the :class:`LogisticL1Scorer` finds a coordinate where the
probability ordering carries signal (non-zero Spearman vs realised R),
isotonic regression aligns the predicted ``P(win)`` to the empirical
bucket win rate.
"""

from __future__ import annotations

import logging
import numpy as np

logger = logging.getLogger(__name__)


class IsotonicRecalibrator:
    """Wrap :class:`sklearn.isotonic.IsotonicRegression`."""

    def __init__(self, increasing: bool = True):
        self.increasing = bool(increasing)
        self._model = None

    def fit(self, p_pred: np.ndarray, y_true: np.ndarray) -> "IsotonicRecalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(p_pred, dtype=float)
        y = np.asarray(y_true, dtype=float)
        if p.ndim != 1:
            raise ValueError(f"p_pred must be 1D, got shape {p.shape}")
        self._model = IsotonicRegression(increasing=self.increasing, out_of_bounds="clip")
        self._model.fit(p, y)
        return self

    def transform(self, p_pred: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("IsotonicRecalibrator not fitted yet")
        return self._model.transform(np.asarray(p_pred, dtype=float))


__all__ = ["IsotonicRecalibrator"]
