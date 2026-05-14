"""Per-signal feature importance — Sprint QUANT-2B.4.

LightGBM has a built-in `pred_contrib=True` mode that returns the SHAP
values directly, without requiring the `shap` package as a separate
dependency. (SHAP and LightGBM's pred_contrib both implement the
TreeSHAP algorithm — Lundberg 2018 — so the numbers are equivalent.)

This wraps that mode in an ergonomic surface for Aisha's narrative
engine:

    explainer = FeatureExplainer(booster, feature_names=[...])
    top = explainer.top_drivers(X, k=3)
    # → [{'feature': 'rsi_14', 'value': 78.2, 'contribution': 0.42,
    #     'direction': 'up'}, ...]

The narrative then renders this as
"top facteurs poussant le score : rsi_14 (sur-acheté), atr_normalised
(volatilité élevée), bos_recent (changement de structure)".

This is *not* a model explanation — it's a per-row attribution that
says "here's how each input feature contributed to this specific
score". For Phase 2B narrative-first, that's the right level of
abstraction: educate the reader about which factors are driving the
algo's view, without claiming any predictive accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class FeatureDriver:
    feature: str
    value: float
    contribution: float  # SHAP value — pushes score up (>0) or down (<0)
    direction: str       # "up" if contribution >= 0, "down" otherwise

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": round(self.value, 6),
            "contribution": round(self.contribution, 6),
            "direction": self.direction,
        }


class FeatureExplainer:
    """Wraps a LightGBM Booster to expose per-row top-K feature drivers.

    Works with both ``lightgbm.Booster`` and ``lightgbm.LGBMRegressor /
    LGBMClassifier`` (sklearn wrappers) — auto-detects which one was
    passed and routes to the right ``predict(..., pred_contrib=True)``.
    """

    def __init__(self, booster: Any, *, feature_names: Optional[Sequence[str]] = None):
        if booster is None:
            raise ValueError("booster is required")
        self._booster = booster
        self._feature_names = (
            list(feature_names)
            if feature_names is not None
            else self._derive_feature_names(booster)
        )

    @staticmethod
    def _derive_feature_names(booster: Any) -> list[str]:
        # sklearn wrapper
        if hasattr(booster, "feature_name_") and booster.feature_name_ is not None:
            return list(booster.feature_name_)
        # raw Booster
        if hasattr(booster, "feature_name"):
            try:
                return list(booster.feature_name())
            except Exception:
                pass
        raise ValueError(
            "could not derive feature_names from booster; pass them explicitly"
        )

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def contributions(self, X: np.ndarray) -> np.ndarray:
        """SHAP contributions per (row, feature) — last column is the bias."""
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        # sklearn wrapper vs raw Booster — both accept pred_contrib via predict.
        if hasattr(self._booster, "predict"):
            try:
                return np.asarray(self._booster.predict(X, pred_contrib=True))
            except TypeError:
                # Sklearn wrapper exposes booster_ for raw access
                if hasattr(self._booster, "booster_"):
                    return np.asarray(
                        self._booster.booster_.predict(X, pred_contrib=True)
                    )
                raise
        raise TypeError(f"booster type {type(self._booster).__name__} not supported")

    def top_drivers(
        self, X: np.ndarray, *, k: int = 3, row: int = 0
    ) -> list[FeatureDriver]:
        """Top-K features by absolute SHAP contribution for a single row.

        ``X`` may be 1D (one row) or 2D (in which case ``row`` selects).
        """
        contribs = self.contributions(X)
        if contribs.ndim == 1:
            contribs = contribs.reshape(1, -1)
        row_contribs = contribs[row]
        # Drop the bias column (last in LightGBM's pred_contrib output).
        feature_contribs = row_contribs[:-1]
        feature_values = (
            np.asarray(X)[row]
            if np.asarray(X).ndim > 1
            else np.asarray(X)
        )

        if len(feature_contribs) != len(self._feature_names):
            raise ValueError(
                f"contribution shape {len(feature_contribs)} doesn't match "
                f"feature_names {len(self._feature_names)}"
            )

        sorted_idx = np.argsort(-np.abs(feature_contribs))[:k]
        out: list[FeatureDriver] = []
        for i in sorted_idx:
            c = float(feature_contribs[i])
            out.append(
                FeatureDriver(
                    feature=self._feature_names[i],
                    value=float(feature_values[i]),
                    contribution=c,
                    direction="up" if c >= 0 else "down",
                )
            )
        return out

    def explain_signal(
        self, X: np.ndarray, *, k: int = 3, row: int = 0
    ) -> dict:
        """JSON-friendly summary consumable by the LLM narrative engine."""
        drivers = self.top_drivers(X, k=k, row=row)
        contribs = self.contributions(X)
        if contribs.ndim == 1:
            contribs = contribs.reshape(1, -1)
        bias = float(contribs[row, -1])
        return {
            "drivers": [d.to_dict() for d in drivers],
            "bias": round(bias, 6),
            "feature_count": len(self._feature_names),
            "k": k,
        }


__all__ = [
    "FeatureDriver",
    "FeatureExplainer",
]
