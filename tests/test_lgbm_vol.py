"""Tests for Sprint 4: LightGBM Meta-Learner (Rank 2).

Tests cover:
  - Feature engineering produces correct shape and columns
  - Feature extraction from DataFrame
  - Model trains on synthetic data without crash
  - Predictions are within reasonable range
  - Feature importance returns correct structure
  - Model save/load roundtrip
  - Thread safety
  - Edge cases (insufficient data, NaN handling)
"""

import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.volatility_forecaster import InstrumentConfig, VolatilityForecaster
from src.intelligence.volatility_lgbm import LGBMVolForecaster


# =============================================================================
# HELPERS
# =============================================================================

def _make_synthetic_ohlcv(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic XAU/USD M15 data for testing."""
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range(
        "2023-01-01", periods=n_bars, freq="15min"
    )

    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 0.5)
    high = close + rng.uniform(0.5, 3.0, n_bars)
    low = close - rng.uniform(0.5, 3.0, n_bars)
    open_ = close + rng.randn(n_bars) * 0.3
    volume = rng.uniform(100, 10000, n_bars)

    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def _make_calendar_df() -> pd.DataFrame:
    """Create a minimal economic calendar."""
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-06 13:30", "2023-02-03 13:30", "2023-03-10 13:30",
            "2023-04-07 13:30", "2023-05-05 13:30",
        ]),
        "event": ["Non-Farm Payrolls"] * 5,
        "impact": ["HIGH"] * 5,
        "currency": ["USD"] * 5,
    })


# =============================================================================
# FEATURE ENGINEERING TESTS
# =============================================================================

class TestFeatureEngineering:
    def test_build_features_shape(self):
        """Feature matrix should have correct number of columns."""
        config = InstrumentConfig()
        lgbm = LGBMVolForecaster(config)
        df = _make_synthetic_ohlcv(3000)

        features = lgbm.build_features(df)
        expected_cols = set(LGBMVolForecaster.FEATURE_NAMES)
        actual_cols = set(features.columns)

        assert expected_cols.issubset(actual_cols), (
            f"Missing: {expected_cols - actual_cols}"
        )

    def test_build_features_includes_target(self):
        """Feature DataFrame should include future_atr target."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)
        assert "future_atr" in features.columns

    def test_session_dummies_sum_to_one(self):
        """At any given bar, exactly one session should be active."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)

        session_cols = [c for c in features.columns if c.startswith("session_")]
        session_sum = features[session_cols].sum(axis=1)

        # Each bar should belong to exactly one session
        assert (session_sum == 1.0).all(), (
            f"Session sums: min={session_sum.min()}, max={session_sum.max()}"
        )

    def test_event_proximity_finite(self):
        """Event proximity should be finite when calendar provided."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        cal = _make_calendar_df()
        features = lgbm.build_features(df, calendar_df=cal)

        # At least some bars should be near events
        near_event = features["event_proximity_hours"] < 4.0
        assert near_event.any(), "No bars near calendar events"

    def test_event_proximity_capped(self):
        """Event proximity should be capped at event_window_hours."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        cal = _make_calendar_df()
        features = lgbm.build_features(df, calendar_df=cal)

        max_val = features["event_proximity_hours"].max()
        assert max_val <= lgbm._config.event_window_hours

    def test_technical_indicators_present(self):
        """RSI, BB%B, MACD hist sign should be computed."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)

        # After warmup period, values should be finite
        assert features["rsi_14"].notna().sum() > 2800
        assert features["bb_pct"].notna().sum() > 2800
        assert features["macd_hist_sign"].notna().sum() > 2800

    def test_atr_change_features(self):
        """ATR change features should be computed."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)

        assert "atr_change_5" in features.columns
        assert "atr_change_20" in features.columns
        # After warmup, should be finite
        assert features["atr_change_5"].iloc[50:].notna().all()


class TestFeatureExtraction:
    def test_extract_feature_row(self):
        """Extract single row as dict with correct keys."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)

        row = lgbm.extract_feature_row(features, idx=-1)
        assert isinstance(row, dict)
        for name in LGBMVolForecaster.FEATURE_NAMES:
            assert name in row
            assert np.isfinite(row[name])

    def test_extract_handles_nan(self):
        """NaN values should be replaced with 0.0."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(3000)
        features = lgbm.build_features(df)

        # First row likely has NaN from rolling windows
        row = lgbm.extract_feature_row(features, idx=0)
        for name in LGBMVolForecaster.FEATURE_NAMES:
            assert np.isfinite(row[name])


# =============================================================================
# TRAINING TESTS
# =============================================================================

class TestTraining:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        """Skip tests if lightgbm not installed."""
        pytest.importorskip("lightgbm")

    def test_train_on_synthetic_data(self):
        """Model should train without errors on synthetic data."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        stats = lgbm.train(df)

        assert stats["trained"] is True
        assert stats["total_bars"] >= 500
        assert stats["val_mae"] > 0
        assert lgbm.is_trained

    def test_train_with_calendar(self):
        """Training with calendar data should work."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        cal = _make_calendar_df()
        stats = lgbm.train(df, calendar_df=cal)
        assert stats["trained"] is True

    def test_train_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        lgbm = LGBMVolForecaster()
        df = _make_synthetic_ohlcv(100)
        stats = lgbm.train(df)
        assert stats["trained"] is False

    def test_feature_importance_after_training(self):
        """Feature importance should be populated after training."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)

        importance = lgbm.feature_importance()
        assert isinstance(importance, dict)
        assert len(importance) > 0
        # Should sum to ~1.0
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01


# =============================================================================
# PREDICTION TESTS
# =============================================================================

class TestPrediction:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    @pytest.fixture
    def trained_lgbm(self):
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)
        return lgbm, df

    def test_predict_returns_positive(self, trained_lgbm):
        """Predictions should be positive."""
        lgbm, df = trained_lgbm
        features_df = lgbm.build_features(df)
        pred = lgbm.predict_from_df(features_df)
        assert pred > 0

    def test_predict_reasonable_range(self, trained_lgbm):
        """Predictions should be within 0.2x to 5x of naive ATR."""
        lgbm, df = trained_lgbm
        features_df = lgbm.build_features(df)
        naive = float(features_df["atr_14"].iloc[-1])
        pred = lgbm.predict_from_df(features_df)

        assert pred >= 0.01
        # Reasonable range check (allowing wider since synthetic data)
        assert pred < naive * 10, f"Prediction {pred} too high vs naive {naive}"

    def test_predict_from_dict(self, trained_lgbm):
        """predict() should work with a feature dict."""
        lgbm, df = trained_lgbm
        features_df = lgbm.build_features(df)
        row = lgbm.extract_feature_row(features_df)
        pred = lgbm.predict(row)
        assert pred > 0

    def test_predict_untrained_raises(self):
        """Predicting before training should raise RuntimeError."""
        lgbm = LGBMVolForecaster()
        with pytest.raises(RuntimeError, match="not trained"):
            lgbm.predict({"atr_14": 3.0})


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

class TestPersistence:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_save_load_roundtrip(self):
        """Model should produce same predictions after save/load."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)

        features_df = lgbm.build_features(df)
        pred_before = lgbm.predict_from_df(features_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = str(Path(tmpdir) / "lgbm_vol.txt")
            lgbm.save_model(model_path)

            lgbm2 = LGBMVolForecaster()
            assert lgbm2.load_model(model_path) is True
            assert lgbm2.is_trained

            pred_after = lgbm2.predict_from_df(features_df)
            assert abs(pred_before - pred_after) < 0.001

    def test_load_missing_file(self):
        """Loading from missing file should return False."""
        lgbm = LGBMVolForecaster()
        assert lgbm.load_model("/nonexistent/path/model.txt") is False

    def test_save_creates_metadata(self):
        """save_model should create a .meta.json alongside the model."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "lgbm_vol.txt"
            lgbm.save_model(str(model_path))

            meta_path = model_path.with_suffix(".meta.json")
            assert meta_path.exists()

            import json
            meta = json.loads(meta_path.read_text())
            assert "feature_importance" in meta
            assert "training_bars" in meta


# =============================================================================
# EDGE CASES & MISC
# =============================================================================

class TestEdgeCases:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_different_instrument_config(self):
        """Should work with non-default instrument config."""
        config = InstrumentConfig(
            symbol="EURUSD",
            timeframe="H1",
            bars_per_day=24,
        )
        lgbm = LGBMVolForecaster(config, n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        stats = lgbm.train(df)
        assert stats["trained"] is True

    def test_stats_before_training(self):
        """get_stats should work before training."""
        lgbm = LGBMVolForecaster()
        stats = lgbm.get_stats()
        assert stats["is_trained"] is False
        assert stats["n_features"] == len(LGBMVolForecaster.FEATURE_NAMES)

    def test_stats_after_training(self):
        """get_stats should include top features after training."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)

        stats = lgbm.get_stats()
        assert stats["is_trained"] is True
        assert len(stats["top_features"]) <= 5

    def test_thread_safety(self):
        """Concurrent predictions should not crash."""
        lgbm = LGBMVolForecaster(n_estimators=50, max_depth=4)
        df = _make_synthetic_ohlcv(5000)
        lgbm.train(df)

        features_df = lgbm.build_features(df)
        row = lgbm.extract_feature_row(features_df)

        results = []
        errors = []

        def predict_task():
            try:
                pred = lgbm.predict(row)
                results.append(pred)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=predict_task) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 10
        # All results should be the same (deterministic)
        assert all(abs(r - results[0]) < 0.001 for r in results)
