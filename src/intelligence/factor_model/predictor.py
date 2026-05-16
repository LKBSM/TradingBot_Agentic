"""FactorModelPredictor — LightGBM regressor on macro + microstructure features.

Target = next-H1 log return of XAU (i.e. next 4 M15 bars).

Output = expected_return (regression) → signal via threshold × ATR.

Walk-forward training preferred (refit monthly). For Sprint 4 we ship
a static fit/predict scaffold + a static training script.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FactorModelPredictor:
    """Predict next-N-bar log return from concatenated features."""

    horizon_bars: int = 4               # 4 M15 bars = next 1 hour
    n_estimators: int = 400
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 50
    reg_alpha: float = 0.1
    reg_lambda: float = 0.0
    random_state: int = 42
    feature_names: tuple[str, ...] = field(default_factory=tuple)
    _model: object = None

    def build_target(self, close: pd.Series) -> pd.Series:
        """Target = log(close.shift(-H) / close)."""
        return np.log(close.shift(-self.horizon_bars) / close)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FactorModelPredictor":
        import lightgbm as lgb
        mask = y.notna() & X.notna().all(axis=1)
        X_clean = X.loc[mask].to_numpy(dtype=float)
        y_clean = y.loc[mask].to_numpy(dtype=float)
        if len(X_clean) < 200:
            raise ValueError(f"Not enough clean rows to fit : {len(X_clean)}")
        self.feature_names = tuple(X.columns)
        self._model = lgb.LGBMRegressor(
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            n_estimators=self.n_estimators,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
            deterministic=True,
            force_row_wise=True,
        )
        self._model.fit(X_clean, y_clean, feature_name=list(self.feature_names))
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("FactorModelPredictor not fitted")
        X_array = X.fillna(0).to_numpy(dtype=float)
        return pd.Series(self._model.predict(X_array), index=X.index, name="expected_return")

    def feature_importance(self) -> dict[str, float]:
        if self._model is None:
            raise RuntimeError("FactorModelPredictor not fitted")
        importance = self._model.feature_importances_
        return dict(zip(self.feature_names, importance.tolist()))

    def directional_accuracy(self, X: pd.DataFrame, y: pd.Series) -> float:
        """Fraction of OOS rows where sign(predict) == sign(actual)."""
        if self._model is None:
            raise RuntimeError("FactorModelPredictor not fitted")
        mask = y.notna() & X.notna().all(axis=1)
        X_c = X.loc[mask]
        y_c = y.loc[mask]
        if len(X_c) < 10:
            return float("nan")
        pred = self.predict(X_c)
        same_sign = (np.sign(pred) == np.sign(y_c))
        return float(same_sign.mean())


__all__ = ["FactorModelPredictor"]
