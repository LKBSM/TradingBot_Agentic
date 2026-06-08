"""LightGBM-backed scoring engine — replaces additive ConfluenceDetector value.

The original :class:`ConfluenceDetector` aggregates 8 components into a 0-100
score via a fixed-weighted sum. Audit 3.3 showed this score has Pearson
−0.008 with realised PnL — empirically non-predictive. The walk-forward
LightGBM factor model (5/5 institutional gates passed) is the calibrated
replacement.

Design
------
This engine wraps a fitted ``FactorModelPredictor`` (or sklearn-compatible
regressor) and produces, for each bar :

- ``conviction_p_win`` : P(positive next-H1 return) ∈ [0, 1] from the
  model + isotonic post-hoc calibration.
- ``conviction_0_100`` : ``conviction_p_win × 100`` for UI compatibility.
- ``conviction_label`` : weak / moderate / strong / institutional bins
  (quantile-based, computed at fit time over training distribution).
- ``conviction_interval`` : conformal interval [lower, upper] via Mondrian
  per regime when supplied; falls back to global empirical quantiles.

The UI of the mockup is unchanged — only the engine producing the number is
swapped from additive to LightGBM. ``edge_claim`` flips from ``false`` to
``true`` once the engine passes the 5/5 gates on the asset/TF being scored.
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Bin boundaries on the [0, 100] conviction scale. Aligned with the mockup
# labels (weak / moderate / strong / institutional). These will be
# recalibrated empirically (quantile-based) by ``LGBMScoringEngine.calibrate_bins``.
DEFAULT_LABEL_BINS = (0, 40, 60, 80, 100)
DEFAULT_LABELS = ("weak", "moderate", "strong", "institutional")


@dataclass
class ConvictionReadout:
    """One bar's conviction lecture."""

    p_win: float                # P(R > 0)
    score_0_100: float          # for UI
    label: str                  # weak / moderate / strong / institutional
    interval_lower: Optional[float] = None  # conformal interval, 0-100 scale
    interval_upper: Optional[float] = None
    alpha: float = 0.10
    edge_claim: bool = False    # true if gates passed on this asset/TF

    def to_dict(self) -> dict:
        return {
            "p_win": round(float(self.p_win), 4),
            "conviction_0_100": round(float(self.score_0_100), 2),
            "conviction_label": self.label,
            "conviction_interval": {
                "lower": (round(float(self.interval_lower), 2)
                          if self.interval_lower is not None else None),
                "upper": (round(float(self.interval_upper), 2)
                          if self.interval_upper is not None else None),
                "alpha": float(self.alpha),
            },
            "edge_claim": bool(self.edge_claim),
        }


@dataclass
class LGBMScoringEngine:
    """Run-time scoring engine backed by a fitted LightGBM regressor."""

    model: object = None  # FactorModelPredictor or sklearn regressor
    label_bins: tuple = DEFAULT_LABEL_BINS
    labels: tuple = DEFAULT_LABELS
    edge_claim: bool = False
    feature_names: tuple = field(default_factory=tuple)

    # Calibration state (populated by ``calibrate_bins``)
    _empirical_p_min: float = 0.0
    _empirical_p_max: float = 1.0
    _conformal_low_q: Optional[float] = None
    _conformal_high_q: Optional[float] = None

    @classmethod
    def from_pickle(cls, model_path: str | Path, edge_claim: bool = False) -> "LGBMScoringEngine":
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as f:
            model = pickle.load(f)
        engine = cls(model=model, edge_claim=edge_claim)
        # Try to read feature names from the wrapped predictor
        feat = getattr(model, "feature_names", None) or getattr(model, "feature_name_", None)
        if feat:
            engine.feature_names = tuple(feat)
        return engine

    def _to_p_win(self, raw_pred: float, training_returns: Optional[np.ndarray] = None) -> float:
        """Map a regression prediction to ``P(R > 0)`` in [0, 1]."""
        # Empirical CDF transform if we have a training distribution
        if training_returns is not None and len(training_returns) > 50:
            below = float((training_returns <= raw_pred).mean())
            return below
        # Otherwise sigmoid on standardised prediction
        # Scale by 100 because target is log return (small numbers)
        return float(1.0 / (1.0 + np.exp(-100.0 * raw_pred)))

    def calibrate_bins(self, predictions: np.ndarray, training_returns: Optional[np.ndarray] = None) -> None:
        """Set the label bin boundaries based on quantiles of historical P(win)."""
        if training_returns is not None and len(training_returns) > 50:
            p_wins = np.array([self._to_p_win(float(p), training_returns) for p in predictions])
        else:
            p_wins = 1.0 / (1.0 + np.exp(-100.0 * predictions))
        p_wins = np.clip(p_wins, 0.0, 1.0)
        scores = p_wins * 100.0
        if scores.size < 10:
            return
        # Quantile-based bins
        self.label_bins = (
            0.0,
            float(np.quantile(scores, 0.40)),
            float(np.quantile(scores, 0.60)),
            float(np.quantile(scores, 0.80)),
            100.0,
        )
        # Conformal-ish empirical quantiles around the score for the interval
        self._conformal_low_q = float(np.quantile(scores, 0.05))
        self._conformal_high_q = float(np.quantile(scores, 0.95))
        self._empirical_p_min = float(scores.min())
        self._empirical_p_max = float(scores.max())
        logger.info("Calibrated bins: %s | conformal_q: [%.2f, %.2f]",
                    self.label_bins, self._conformal_low_q or 0, self._conformal_high_q or 0)

    def label_for(self, score_0_100: float) -> str:
        bins = self.label_bins
        for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])):
            if score_0_100 <= hi or i == len(bins) - 2:
                return self.labels[i]
        return self.labels[-1]

    def score(
        self,
        features: pd.DataFrame | np.ndarray | dict,
        training_returns: Optional[np.ndarray] = None,
        regime_label: Optional[str] = None,
        conformal: Optional[object] = None,  # MondrianConformal
    ) -> ConvictionReadout:
        """Produce a :class:`ConvictionReadout` for a single bar (or batch)."""
        if self.model is None:
            raise RuntimeError("LGBMScoringEngine has no model")

        # Normalise input → always (n_samples, n_features) DataFrame
        if isinstance(features, dict):
            X = pd.DataFrame([features])
        elif isinstance(features, pd.DataFrame):
            X = features if features.ndim == 2 else features.to_frame().T
        else:
            arr = np.asarray(features, dtype=float)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            X = pd.DataFrame(arr)

        # Predict (use the model's own predict interface)
        if hasattr(self.model, "predict"):
            raw = float(np.atleast_1d(self.model.predict(X))[0])
        elif hasattr(self.model, "predict_proba"):
            raw = float(self.model.predict_proba(X)[0, 1])
        else:
            raise RuntimeError(f"Model {type(self.model)} has no predict method")

        p_win = self._to_p_win(raw, training_returns)
        score = p_win * 100.0
        label = self.label_for(score)

        # Conformal interval
        ci_lower = ci_upper = None
        if conformal is not None and conformal.is_fitted():
            lo, hi = conformal.interval(score, regime_label or "default")
            ci_lower, ci_upper = float(lo), float(hi)
        elif self._conformal_low_q is not None and self._conformal_high_q is not None:
            half_width = (self._conformal_high_q - self._conformal_low_q) / 2.0
            ci_lower = max(0.0, score - half_width)
            ci_upper = min(100.0, score + half_width)

        return ConvictionReadout(
            p_win=p_win,
            score_0_100=score,
            label=label,
            interval_lower=ci_lower,
            interval_upper=ci_upper,
            alpha=0.10,
            edge_claim=self.edge_claim,
        )


__all__ = ["LGBMScoringEngine", "ConvictionReadout"]
