"""LightGBM scorer — alternative plus puissante au LogisticL1Scorer.

Pourquoi LightGBM (et pas LLM)
------------------------------
Pour transformer N features tabulaires en P(win), LightGBM est :
- **Calibrable** : `predict_proba` + isotonic post-hoc → vraie probabilité.
- **Rapide** : < 1 ms / inference (vs 500-2000 ms pour un LLM).
- **Déterministe** : seed fixé → bit-by-bit reproductible (gate Sprint 6).
- **Capable de non-linéarité** : capture interactions OB×retest×regime, ce
  que L1 (linéaire) ne peut pas.
- **Sparsité naturelle** : importance des features pour audit/explainability.
- **Quasi-gratuit** : pas de coût marginal par inference (vs $0.003/LLM).

L'audit 3.3 recommandait L1 comme baseline simple. LightGBM est l'option
"un cran plus puissant" à essayer si L1 ne tient pas son Brier skill OOS.

Architecture
------------
Wraps :class:`lightgbm.LGBMClassifier` (déjà dans requirements.txt — utilisé
par le VolatilityForecaster aussi).

>>> scorer = LGBMScorer(num_leaves=31, learning_rate=0.05, n_estimators=200)
>>> scorer.fit(X, y)
>>> p_win = scorer.predict_p_win(X_test)
>>> # then post-hoc isotonic calibration if needed
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


DEFAULT_FEATURE_NAMES = (
    "smc_structure", "order_blocks", "fvg", "retest",
    "regime", "vol_forecast", "news", "momentum_rsi_div",
)


class LGBMScorer:
    """Gradient-boosted scorer using LightGBM."""

    def __init__(
        self,
        num_leaves: int = 31,
        learning_rate: float = 0.05,
        n_estimators: int = 200,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,     # L1 on weights (sparsity)
        reg_lambda: float = 0.0,    # L2 on weights
        feature_names: tuple[str, ...] = DEFAULT_FEATURE_NAMES,
        random_state: int = 42,
        class_weight: str = "balanced",
    ):
        self.num_leaves = int(num_leaves)
        self.learning_rate = float(learning_rate)
        self.n_estimators = int(n_estimators)
        self.min_child_samples = int(min_child_samples)
        self.reg_alpha = float(reg_alpha)
        self.reg_lambda = float(reg_lambda)
        self.feature_names = tuple(feature_names)
        self.random_state = int(random_state)
        self.class_weight = class_weight
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMScorer":
        import lightgbm as lgb
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).astype(int)
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got {X.shape}")
        self._model = lgb.LGBMClassifier(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            class_weight=self.class_weight,
            random_state=self.random_state,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        self._model.fit(X, y, feature_name=list(self.feature_names))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("LGBMScorer not fitted")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self._model.predict_proba(X)

    def predict_p_win(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)[:, 1]

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("LGBMScorer not fitted")
        importances = self._model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))


__all__ = ["LGBMScorer", "DEFAULT_FEATURE_NAMES"]
