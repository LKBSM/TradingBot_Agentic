"""3-state HMM regime classifier — Sprint REGIME-2B.1.

Discriminates three market regimes from realised returns:

- 0 — **low-vol trending**: small variance, drift in one direction.
- 1 — **low-vol ranging**:  small variance, near-zero drift.
- 2 — **high-vol stress**:  large variance, no consistent drift.

The classification feeds narrative ("XAU is in a high-vol stress
regime") rather than a trade signal — Phase 2B is narrative-first.
No edge claim is attached.

Why HMM not Jump-Model
----------------------
Phase 2A would have used Statistical Jump Models (Bemporad 2018). In
2B we don't need SOTA — HMM(GaussianHMM 3-state) is interpretable,
trains in seconds on 6 years of M15 returns, and is enough to label
"trending vs ranging vs stress" for context. Calling out an analyst
on a quant choice this far down the stack would burn time we don't
have.

The model is intentionally tiny:
- 2 features: |return| and signed return (proxies for vol + drift)
- 3 hidden states
- 100 EM iterations, fixed seed for reproducibility across re-trains

Public surface
--------------
- ``RegimeClassifier()`` constructs an unfit instance.
- ``fit(returns)`` trains on a numpy array of returns (1D float).
- ``predict(returns)`` returns the per-bar state.
- ``predict_with_confidence(returns)`` returns (state, posterior_prob).
- ``state_labels()`` returns the deterministic ``{0: "low_vol_trending",
  1: "low_vol_ranging", 2: "high_vol_stress"}`` mapping — labels are
  assigned post-fit by inspecting the per-state variance + drift, so
  state ordering doesn't depend on EM random init.

The label assignment is itself the only non-trivial bit:
1. sort states by variance ascending → ``low_vol_state``, ``mid``, ``high_vol_state``
2. of the two low-variance states, the one with smaller |mean| is
   "ranging", the one with larger |mean| is "trending"

Persistence is via joblib (or fallback pickle) on ``save()/load()``.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


LABEL_LOW_VOL_TRENDING = "low_vol_trending"
LABEL_LOW_VOL_RANGING = "low_vol_ranging"
LABEL_HIGH_VOL_STRESS = "high_vol_stress"


@dataclass(frozen=True)
class RegimePrediction:
    state: int
    label: str
    confidence: float  # posterior probability of the chosen state, [0, 1]


def _features(returns: np.ndarray) -> np.ndarray:
    """Two-column feature matrix: (|r|, r)."""
    r = np.asarray(returns, dtype=float).reshape(-1, 1)
    abs_r = np.abs(r)
    return np.hstack([abs_r, r])


class RegimeClassifier:
    """3-state GaussianHMM with deterministic label assignment."""

    N_STATES = 3
    DEFAULT_RANDOM_STATE = 42

    def __init__(self, *, random_state: int = DEFAULT_RANDOM_STATE):
        self._random_state = random_state
        self._model = None
        self._state_to_label: dict[int, str] = {}

    # ------------------------------------------------------------------
    # Fit / predict
    # ------------------------------------------------------------------

    def fit(self, returns: np.ndarray) -> "RegimeClassifier":
        from hmmlearn import hmm

        if len(returns) < 30:
            raise ValueError(
                f"need >= 30 returns to fit a 3-state HMM, got {len(returns)}"
            )
        X = _features(returns)
        self._model = hmm.GaussianHMM(
            n_components=self.N_STATES,
            covariance_type="diag",
            n_iter=100,
            random_state=self._random_state,
            tol=1e-3,
        )
        self._model.fit(X)
        self._assign_labels()
        return self

    def predict(self, returns: np.ndarray) -> np.ndarray:
        self._require_fit()
        return self._model.predict(_features(returns))

    def predict_with_confidence(self, returns: np.ndarray) -> list[RegimePrediction]:
        self._require_fit()
        X = _features(returns)
        states = self._model.predict(X)
        posteriors = self._model.predict_proba(X)
        out = []
        for i, s in enumerate(states):
            out.append(
                RegimePrediction(
                    state=int(s),
                    label=self._state_to_label[int(s)],
                    confidence=float(posteriors[i, s]),
                )
            )
        return out

    def state_labels(self) -> dict[int, str]:
        self._require_fit()
        return dict(self._state_to_label)

    # ------------------------------------------------------------------
    # Label assignment — deterministic post-fit
    # ------------------------------------------------------------------

    def _assign_labels(self) -> None:
        """Inspect fitted state means/variances, assign semantic labels.

        Without this, hidden state 0 might be "stress" on one EM run
        and "ranging" on the next — labels are arbitrary integers
        until we anchor them to interpretable parameters.
        """
        # means_ shape = (n_components, n_features); we used [|r|, r]
        means = self._model.means_       # state x [mean_abs_r, mean_r]
        covars = self._model.covars_      # state x feature x feature

        # Variance of returns (col 1) — high variance = stress
        vars_signed = np.array([c[1, 1] for c in covars])
        order_by_var = np.argsort(vars_signed)  # ascending

        low_var_states = order_by_var[:2].tolist()
        high_vol_state = int(order_by_var[2])

        # Of the two low-var states, the one with smaller |mean_r| is
        # ranging, the larger is trending.
        a, b = low_var_states
        abs_mean_a = abs(means[a, 1])
        abs_mean_b = abs(means[b, 1])
        if abs_mean_a <= abs_mean_b:
            ranging = int(a); trending = int(b)
        else:
            ranging = int(b); trending = int(a)

        self._state_to_label = {
            trending: LABEL_LOW_VOL_TRENDING,
            ranging: LABEL_LOW_VOL_RANGING,
            high_vol_state: LABEL_HIGH_VOL_STRESS,
        }

    def _require_fit(self) -> None:
        if self._model is None:
            raise RuntimeError("RegimeClassifier must be fit() before predict")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        self._require_fit()
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model": self._model,
                    "state_to_label": self._state_to_label,
                    "random_state": self._random_state,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "RegimeClassifier":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(random_state=data["random_state"])
        obj._model = data["model"]
        obj._state_to_label = data["state_to_label"]
        return obj


__all__ = [
    "LABEL_HIGH_VOL_STRESS",
    "LABEL_LOW_VOL_RANGING",
    "LABEL_LOW_VOL_TRENDING",
    "RegimeClassifier",
    "RegimePrediction",
]
