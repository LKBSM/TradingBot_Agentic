# =============================================================================
# FEATURE REDUCER - Observation Space Dimensionality Reduction (Sprint 6)
# =============================================================================
# Reduces multicollinear 303-dim observation space to ~50-80 decorrelated dims.
#
# Pipeline:
#   1. Replace raw OHLCV with decorrelated features (4 → 3)
#   2. Compute VIF to flag remaining multicollinearity
#   3. Apply IncrementalPCA to compress to 95% explained variance
#   4. Save/load transformer for inference consistency
#
# =============================================================================

import logging
import pickle
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DECORRELATED FEATURE TRANSFORMS
# =============================================================================

def compute_decorrelated_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace correlated OHLC with 3 decorrelated features.

    Input columns required: Open, High, Low, Close, ATR
    Output columns added: log_return, hl_range, close_position
    Output columns removed: Open, High, Low, Close

    Args:
        df: DataFrame with OHLCV columns.

    Returns:
        DataFrame with OHLC replaced by decorrelated features.
    """
    out = df.copy()

    # 1. log_return = log(Close / Close[-1]) — replaces Close
    close = out["Close"].values.astype(np.float64)
    log_ret = np.zeros_like(close)
    log_ret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-10))
    out["log_return"] = log_ret

    # 2. hl_range = (High - Low) / ATR — normalized range, replaces High and Low
    hl = (out["High"] - out["Low"]).values.astype(np.float64)
    atr = out["ATR"].values.astype(np.float64) if "ATR" in out.columns else np.ones_like(hl)
    out["hl_range"] = hl / np.maximum(atr, 1e-10)

    # 3. close_position = (Close - Open) / (High - Low + 1e-8) — candle body, replaces Open
    out["close_position"] = (
        (out["Close"] - out["Open"]).values.astype(np.float64)
        / np.maximum(hl, 1e-8)
    )

    # NOTE: We do NOT drop OHLC from the DataFrame.
    # The environment still needs Close, High, Low for:
    #   - Trade execution (current_market_price = Close)
    #   - SL/TP checking (High, Low for intra-bar checks)
    #   - Portfolio valuation
    # The FEATURES list in config.py controls which columns enter the obs space.
    # OHLC is excluded from FEATURES when USE_DECORRELATED_FEATURES=True,
    # so the decorrelated features replace OHLC in the observation only.

    return out


# =============================================================================
# VIF CALCULATION
# =============================================================================

def compute_vif(data: np.ndarray, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.

    VIF > 10 indicates severe multicollinearity.
    VIF > 5 indicates moderate multicollinearity.

    Uses OLS-free formula: VIF_j = 1 / (1 - R²_j)
    where R²_j is from regressing feature j on all others.

    Args:
        data: 2D array (n_samples, n_features).
        feature_names: Optional list of feature names.

    Returns:
        DataFrame with columns ['feature', 'VIF'], sorted by VIF descending.
    """
    n_samples, n_features = data.shape
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Remove constant/zero-variance columns
    stds = np.std(data, axis=0)
    valid_mask = stds > 1e-10
    valid_data = data[:, valid_mask]
    valid_names = [feature_names[i] for i in range(n_features) if valid_mask[i]]

    if valid_data.shape[1] < 2:
        return pd.DataFrame({"feature": valid_names, "VIF": [1.0] * len(valid_names)})

    # Standardize for numerical stability
    means = np.mean(valid_data, axis=0)
    stds_valid = np.std(valid_data, axis=0)
    X = (valid_data - means) / np.maximum(stds_valid, 1e-10)

    # Correlation matrix → VIF
    corr = np.corrcoef(X, rowvar=False)
    try:
        corr_inv = np.linalg.inv(corr)
        vif_values = np.diag(corr_inv)
    except np.linalg.LinAlgError:
        # Singular matrix — use pseudo-inverse
        corr_inv = np.linalg.pinv(corr)
        vif_values = np.diag(corr_inv)

    # Clamp negative VIF (numerical artifact) to 1.0
    vif_values = np.maximum(vif_values, 1.0)

    result = pd.DataFrame({"feature": valid_names, "VIF": vif_values})
    return result.sort_values("VIF", ascending=False).reset_index(drop=True)


def flag_high_vif(data: np.ndarray, feature_names: Optional[List[str]] = None,
                  threshold: float = 10.0) -> pd.DataFrame:
    """Return only features with VIF above threshold."""
    vif_df = compute_vif(data, feature_names)
    return vif_df[vif_df["VIF"] > threshold].reset_index(drop=True)


# =============================================================================
# FEATURE REDUCER (PCA WRAPPER)
# =============================================================================

class FeatureReducer:
    """
    Dimensionality reduction wrapper using IncrementalPCA.

    Designed for the trading environment observation space:
    - Fits on training data (flattened lookback windows)
    - Transforms individual observations during training and inference
    - Saves/loads alongside the model for consistency

    Usage:
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(training_observations)  # (N, 303) array
        reduced_obs = reducer.transform(obs)  # (303,) → (~60,)
        reducer.save("path/to/pca_transformer.pkl")
        reducer = FeatureReducer.load("path/to/pca_transformer.pkl")
    """

    def __init__(
        self,
        variance_threshold: float = 0.95,
        max_components: Optional[int] = None,
    ):
        """
        Args:
            variance_threshold: Minimum cumulative explained variance to retain.
            max_components: Hard cap on number of components. None = auto.
        """
        self.variance_threshold = variance_threshold
        self.max_components = max_components
        self._pca = None
        self._n_components: int = 0
        self._input_dim: int = 0
        self._fitted: bool = False
        self._explained_variance_ratio: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def n_components(self) -> int:
        return self._n_components

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        return self._explained_variance_ratio

    @property
    def cumulative_variance(self) -> float:
        """Total explained variance retained."""
        if self._explained_variance_ratio is not None:
            return float(np.sum(self._explained_variance_ratio))
        return 0.0

    def fit(self, data: np.ndarray) -> "FeatureReducer":
        """
        Fit PCA on training data.

        Args:
            data: 2D array (n_samples, n_features). Should be the full
                  flattened observation vectors from training episodes.

        Returns:
            self
        """
        from sklearn.decomposition import PCA

        n_samples, n_features = data.shape
        self._input_dim = n_features

        # First fit full PCA to determine component count
        max_comp = min(n_samples, n_features)
        if self.max_components:
            max_comp = min(max_comp, self.max_components)

        full_pca = PCA(n_components=max_comp)
        full_pca.fit(data)

        # Find number of components for variance threshold
        cumvar = np.cumsum(full_pca.explained_variance_ratio_)
        n_components = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        n_components = min(n_components, max_comp)
        n_components = max(n_components, 1)

        # Refit with exact component count
        self._pca = PCA(n_components=n_components)
        self._pca.fit(data)
        self._n_components = n_components
        self._explained_variance_ratio = self._pca.explained_variance_ratio_
        self._fitted = True

        logger.info(
            "FeatureReducer fitted: %d → %d dimensions (%.1f%% variance retained)",
            n_features, n_components,
            self.cumulative_variance * 100,
        )

        return self

    def transform(self, observation: np.ndarray) -> np.ndarray:
        """
        Transform a single observation or batch.

        Args:
            observation: 1D array (n_features,) or 2D (n_samples, n_features).

        Returns:
            Reduced observation(s).
        """
        if not self._fitted:
            raise RuntimeError("FeatureReducer not fitted. Call fit() first.")

        if observation.ndim == 1:
            return self._pca.transform(observation.reshape(1, -1)).flatten()
        return self._pca.transform(observation)

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, reduced: np.ndarray) -> np.ndarray:
        """Reconstruct approximate original from reduced dimensions."""
        if not self._fitted:
            raise RuntimeError("FeatureReducer not fitted.")
        if reduced.ndim == 1:
            return self._pca.inverse_transform(reduced.reshape(1, -1)).flatten()
        return self._pca.inverse_transform(reduced)

    # -------------------------------------------------------------------------
    # PERSISTENCE
    # -------------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save fitted reducer to disk."""
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted FeatureReducer.")
        state = {
            "pca": self._pca,
            "n_components": self._n_components,
            "input_dim": self._input_dim,
            "variance_threshold": self.variance_threshold,
            "max_components": self.max_components,
            "explained_variance_ratio": self._explained_variance_ratio,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(state, f)
        logger.info("FeatureReducer saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "FeatureReducer":
        """Load a fitted reducer from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        reducer = cls(
            variance_threshold=state["variance_threshold"],
            max_components=state["max_components"],
        )
        reducer._pca = state["pca"]
        reducer._n_components = state["n_components"]
        reducer._input_dim = state["input_dim"]
        reducer._explained_variance_ratio = state["explained_variance_ratio"]
        reducer._fitted = True
        logger.info(
            "FeatureReducer loaded: %d → %d dims (%.1f%% variance)",
            reducer._input_dim, reducer._n_components,
            reducer.cumulative_variance * 100,
        )
        return reducer

    def to_dict(self) -> Dict[str, Any]:
        """Get reducer stats as dict."""
        return {
            "fitted": self._fitted,
            "input_dim": self._input_dim,
            "output_dim": self._n_components,
            "variance_threshold": self.variance_threshold,
            "cumulative_variance": self.cumulative_variance,
            "max_components": self.max_components,
        }
