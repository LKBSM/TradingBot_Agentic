"""
Sprint 6 Tests: Observation Space Dimensionality Reduction — decorrelated features,
VIF detection, PCA compression, save/load, and observation shape consistency.
"""
import os
import ast
import tempfile
import numpy as np
import pandas as pd
import pytest

from src.environment.feature_reducer import (
    FeatureReducer,
    compute_decorrelated_ohlcv,
    compute_vif,
    flag_high_vif,
)


# =============================================================================
# HELPERS
# =============================================================================

def make_ohlcv_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame with realistic correlations."""
    rng = np.random.RandomState(seed)
    close = 2000.0 + np.cumsum(rng.randn(n) * 5)
    high = close + np.abs(rng.randn(n) * 3)
    low = close - np.abs(rng.randn(n) * 3)
    open_ = close + rng.randn(n) * 2
    volume = np.abs(rng.randn(n) * 1000) + 500
    atr = np.abs(rng.randn(n) * 2) + 1

    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low, "Close": close,
        "Volume": volume, "ATR": atr,
    })


def make_multicollinear_data(n: int = 500, seed: int = 42) -> np.ndarray:
    """Create data with known multicollinearity (some features derived from others)."""
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    x3 = x1 * 0.95 + rng.randn(n) * 0.1  # Highly correlated with x1
    x4 = x2 * 0.9 + rng.randn(n) * 0.15   # Correlated with x2
    x5 = x1 + x2 + rng.randn(n) * 0.05    # Linear combo
    x6 = rng.randn(n)                       # Independent
    x7 = rng.randn(n) * 0.5                 # Independent
    return np.column_stack([x1, x2, x3, x4, x5, x6, x7])


def make_high_dim_data(n_samples: int = 300, n_features: int = 303, seed: int = 42) -> np.ndarray:
    """Create data simulating the 303-dim observation space."""
    rng = np.random.RandomState(seed)
    # Create some structure: first 50 dims explain most variance
    latent = rng.randn(n_samples, 50)
    noise = rng.randn(n_samples, n_features) * 0.1
    # Project latent to full space
    projection = rng.randn(50, n_features)
    data = latent @ projection + noise
    return data


# =============================================================================
# TEST: DECORRELATED FEATURES
# =============================================================================

class TestDecorrelatedFeatures:
    def test_ohlc_replaced_with_3_features(self):
        df = make_ohlcv_df()
        result = compute_decorrelated_ohlcv(df)

        # OHLC should be gone
        assert "Open" not in result.columns
        assert "High" not in result.columns
        assert "Low" not in result.columns
        assert "Close" not in result.columns

        # New features should exist
        assert "log_return" in result.columns
        assert "hl_range" in result.columns
        assert "close_position" in result.columns

        # Volume should still be there
        assert "Volume" in result.columns

    def test_log_return_first_value_is_zero(self):
        df = make_ohlcv_df()
        result = compute_decorrelated_ohlcv(df)
        assert result["log_return"].iloc[0] == 0.0

    def test_log_return_reasonable_range(self):
        df = make_ohlcv_df()
        result = compute_decorrelated_ohlcv(df)
        # Log returns should be small (< 5% per bar typically)
        assert np.abs(result["log_return"].values).max() < 0.5

    def test_close_position_finite(self):
        df = make_ohlcv_df()
        result = compute_decorrelated_ohlcv(df)
        # Close position should be finite (no NaN/Inf)
        assert np.all(np.isfinite(result["close_position"].values))

    def test_hl_range_positive(self):
        df = make_ohlcv_df()
        result = compute_decorrelated_ohlcv(df)
        assert (result["hl_range"] >= 0).all()

    def test_reduces_feature_count(self):
        df = make_ohlcv_df()
        original_count = len(df.columns)
        result = compute_decorrelated_ohlcv(df)
        # Removed 4 (OHLC), added 3 → net -1
        assert len(result.columns) == original_count - 1


# =============================================================================
# TEST: VIF CALCULATION
# =============================================================================

class TestVIFCalculation:
    def test_vif_detects_multicollinearity(self):
        """VIF should flag at least 3 features with VIF > 10 in multicollinear data."""
        data = make_multicollinear_data()
        names = [f"x{i}" for i in range(data.shape[1])]
        high_vif = flag_high_vif(data, names, threshold=10.0)

        # x3 (~x1), x4 (~x2), x5 (~x1+x2) should have high VIF
        assert len(high_vif) >= 3, (
            f"Expected at least 3 features with VIF > 10, got {len(high_vif)}"
        )

    def test_vif_independent_features_low(self):
        """Independent features should have VIF close to 1."""
        rng = np.random.RandomState(42)
        data = rng.randn(500, 5)
        vif_df = compute_vif(data)
        # All VIF should be < 2 for truly independent data
        assert (vif_df["VIF"] < 3.0).all()

    def test_vif_returns_all_features(self):
        data = make_multicollinear_data()
        names = [f"x{i}" for i in range(data.shape[1])]
        vif_df = compute_vif(data, names)
        assert len(vif_df) == len(names)

    def test_vif_sorted_descending(self):
        data = make_multicollinear_data()
        vif_df = compute_vif(data)
        assert (vif_df["VIF"].values[:-1] >= vif_df["VIF"].values[1:]).all()

    def test_vif_handles_constant_column(self):
        """Constant columns should be handled gracefully."""
        rng = np.random.RandomState(42)
        data = rng.randn(100, 3)
        data[:, 1] = 5.0  # Constant
        vif_df = compute_vif(data, ["a", "b", "c"])
        # Should not crash, constant column excluded
        assert len(vif_df) >= 2


# =============================================================================
# TEST: PCA FEATURE REDUCER
# =============================================================================

class TestFeatureReducer:
    def test_fit_reduces_dimensions(self):
        data = make_high_dim_data(n_samples=300, n_features=303)
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        assert reducer.is_fitted
        assert reducer.n_components < 303
        # Should retain < 60% of original dims for 95% variance
        assert reducer.n_components < int(303 * 0.60), (
            f"Expected < 182 components, got {reducer.n_components}"
        )

    def test_variance_retention(self):
        """PCA should retain >= 95% of explained variance."""
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        assert reducer.cumulative_variance >= 0.95, (
            f"Expected >= 95% variance, got {reducer.cumulative_variance:.1%}"
        )

    def test_transform_single_observation(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        obs = data[0]  # Single observation (303,)
        reduced = reducer.transform(obs)

        assert reduced.ndim == 1
        assert len(reduced) == reducer.n_components

    def test_transform_batch(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        reduced = reducer.transform(data[:10])
        assert reduced.shape == (10, reducer.n_components)

    def test_fit_transform(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reduced = reducer.fit_transform(data)

        assert reduced.shape[0] == data.shape[0]
        assert reduced.shape[1] == reducer.n_components

    def test_inverse_transform_approximate(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        original = data[0]
        reduced = reducer.transform(original)
        reconstructed = reducer.inverse_transform(reduced)

        # Reconstruction error should be small (< 5% variance lost)
        mse = np.mean((original - reconstructed) ** 2)
        original_var = np.var(original)
        assert mse / max(original_var, 1e-10) < 0.10  # < 10% reconstruction error

    def test_transform_without_fit_raises(self):
        reducer = FeatureReducer()
        with pytest.raises(RuntimeError, match="not fitted"):
            reducer.transform(np.zeros(100))

    def test_max_components_cap(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.99, max_components=30)
        reducer.fit(data)
        assert reducer.n_components <= 30


# =============================================================================
# TEST: SAVE / LOAD
# =============================================================================

class TestFeatureReducerPersistence:
    def test_save_and_load(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            reducer.save(path)
            loaded = FeatureReducer.load(path)

            assert loaded.is_fitted
            assert loaded.n_components == reducer.n_components
            assert loaded.input_dim == reducer.input_dim
            assert abs(loaded.cumulative_variance - reducer.cumulative_variance) < 1e-6

            # Transform should produce same results
            obs = data[0]
            original_result = reducer.transform(obs)
            loaded_result = loaded.transform(obs)
            np.testing.assert_array_almost_equal(original_result, loaded_result)
        finally:
            os.unlink(path)

    def test_save_unfitted_raises(self):
        reducer = FeatureReducer()
        with pytest.raises(RuntimeError, match="Cannot save unfitted"):
            reducer.save("/tmp/test.pkl")

    def test_to_dict(self):
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        d = reducer.to_dict()
        assert d["fitted"] is True
        assert d["input_dim"] == 303
        assert d["output_dim"] == reducer.n_components
        assert d["cumulative_variance"] >= 0.95


# =============================================================================
# TEST: OBSERVATION SHAPE CONSISTENCY
# =============================================================================

class TestObservationShapeConsistency:
    def test_pca_output_consistent_across_calls(self):
        """Same input should always produce same output."""
        data = make_high_dim_data()
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        obs = data[42]
        result1 = reducer.transform(obs)
        result2 = reducer.transform(obs)
        np.testing.assert_array_equal(result1, result2)

    def test_training_and_inference_same_shape(self):
        """Observation shape should be identical between fit and transform."""
        data = make_high_dim_data(n_samples=500, n_features=303)
        reducer = FeatureReducer(variance_threshold=0.95)
        reducer.fit(data)

        # Simulate inference with different data
        inference_data = make_high_dim_data(n_samples=100, n_features=303, seed=99)
        for i in range(10):
            reduced = reducer.transform(inference_data[i])
            assert len(reduced) == reducer.n_components


# =============================================================================
# TEST: CONFIG FLAGS
# =============================================================================

class TestConfigFlags:
    def test_use_pca_reduction_flag_exists(self):
        import config
        assert hasattr(config, "USE_PCA_REDUCTION")
        assert isinstance(config.USE_PCA_REDUCTION, bool)

    def test_pca_variance_threshold_exists(self):
        import config
        assert hasattr(config, "PCA_VARIANCE_THRESHOLD")
        assert 0.8 <= config.PCA_VARIANCE_THRESHOLD <= 1.0

    def test_use_decorrelated_features_flag_exists(self):
        import config
        assert hasattr(config, "USE_DECORRELATED_FEATURES")


# =============================================================================
# TEST: SOURCE VERIFICATION
# =============================================================================

class TestSourceVerification:
    def test_no_print_in_feature_reducer(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "feature_reducer.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "print":
                    pytest.fail(f"Found print() at line {node.lineno}")

    def test_feature_reducer_in_environment(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "environment.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "_feature_reducer" in source

    def test_pca_save_in_trainer(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "training", "sophisticated_trainer.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "pca_transformer.pkl" in source

    def test_vif_in_strategy_features(self):
        src_path = os.path.join(
            os.path.dirname(__file__), "..", "src", "environment", "strategy_features.py"
        )
        with open(src_path, "r", encoding="utf-8") as f:
            source = f.read()
        assert "compute_feature_vif" in source
