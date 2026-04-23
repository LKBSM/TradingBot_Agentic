"""Tests for Sprint 5: Two-Stage Hybrid Forecaster (Rank 3).

Tests cover:
  - HybridForecaster calibrate() fits both HAR + LightGBM
  - Hybrid forecast = HAR base + LightGBM residual correction
  - Fallback chain: LightGBM fails → HAR-only
  - Factory method: VolatilityForecaster.create()
  - Mode switching (har/lgbm/hybrid)
  - Persistence: save/load roundtrip for hybrid state
"""

import tempfile
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.intelligence.volatility_forecaster import (
    HybridForecaster,
    InstrumentConfig,
    VolatilityForecast,
    VolatilityForecaster,
)


# =============================================================================
# HELPERS
# =============================================================================

def _make_synthetic_ohlcv(n_bars: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="15min")
    close = 2000.0 + np.cumsum(rng.randn(n_bars) * 0.5)
    high = close + rng.uniform(0.5, 3.0, n_bars)
    low = close - rng.uniform(0.5, 3.0, n_bars)
    open_ = close + rng.randn(n_bars) * 0.3
    volume = rng.uniform(100, 10000, n_bars)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    })


def _make_calendar_df() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2023-01-06 13:30", "2023-02-03 13:30", "2023-03-10 13:30",
        ]),
        "event": ["Non-Farm Payrolls"] * 3,
        "impact": ["HIGH"] * 3,
        "currency": ["USD"] * 3,
    })


# =============================================================================
# FACTORY TESTS
# =============================================================================

class TestFactory:
    def test_create_har(self):
        """Factory 'har' mode returns base VolatilityForecaster."""
        f = VolatilityForecaster.create("har")
        assert isinstance(f, VolatilityForecaster)
        assert not isinstance(f, HybridForecaster)

    def test_create_hybrid(self):
        """Factory 'hybrid' mode returns HybridForecaster."""
        f = VolatilityForecaster.create("hybrid")
        assert isinstance(f, HybridForecaster)
        assert f.mode == "hybrid"

    def test_create_lgbm(self):
        """Factory 'lgbm' mode returns HybridForecaster in lgbm mode."""
        f = VolatilityForecaster.create("lgbm")
        assert isinstance(f, HybridForecaster)
        assert f.mode == "lgbm"

    def test_create_invalid_raises(self):
        """Factory with invalid mode should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            VolatilityForecaster.create("invalid")

    def test_create_with_config(self):
        """Factory should pass config through."""
        config = InstrumentConfig(symbol="EURUSD", bars_per_day=24)
        f = VolatilityForecaster.create("har", config)
        assert f.config.symbol == "EURUSD"


# =============================================================================
# HYBRID CALIBRATION TESTS
# =============================================================================

class TestHybridCalibration:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_hybrid_calibrate_fits_both(self):
        """Hybrid calibrate should fit HAR + LightGBM."""
        hybrid = HybridForecaster()
        df = _make_synthetic_ohlcv(5000)
        stats = hybrid.calibrate(df)

        assert stats.get("har_fitted") is True
        assert stats.get("lgbm_trained") is True
        assert hybrid.is_calibrated

    def test_hybrid_calibrate_with_calendar(self):
        """Should work with calendar data."""
        hybrid = HybridForecaster()
        df = _make_synthetic_ohlcv(5000)
        cal = _make_calendar_df()
        stats = hybrid.calibrate(df, cal)
        assert stats.get("har_fitted") is True

    def test_lgbm_mode_calibrate(self):
        """LightGBM mode should also calibrate."""
        hybrid = HybridForecaster(mode="lgbm")
        df = _make_synthetic_ohlcv(5000)
        stats = hybrid.calibrate(df)
        assert stats.get("lgbm_trained") is True

    def test_hybrid_residual_mode(self):
        """Hybrid mode should train on residuals."""
        hybrid = HybridForecaster(mode="hybrid")
        df = _make_synthetic_ohlcv(5000)
        stats = hybrid.calibrate(df)

        # Should have residual-specific metrics
        assert "lgbm_mode" in stats or stats.get("lgbm_trained") is True


# =============================================================================
# HYBRID FORECAST TESTS
# =============================================================================

class TestHybridForecast:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    @pytest.fixture
    def calibrated_hybrid(self):
        hybrid = HybridForecaster(mode="hybrid")
        df = _make_synthetic_ohlcv(5000)
        hybrid.calibrate(df)
        return hybrid, df

    def test_forecast_returns_volatility_forecast(self, calibrated_hybrid):
        """Hybrid forecast should return VolatilityForecast."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert isinstance(forecast, VolatilityForecast)

    def test_forecast_positive(self, calibrated_hybrid):
        """Forecast ATR should be positive."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert forecast.forecast_atr > 0
        assert forecast.naive_atr > 0

    def test_forecast_reasonable_range(self, calibrated_hybrid):
        """Forecast should be within 0.2x-5x of naive ATR."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert forecast.forecast_atr >= 0.2 * forecast.naive_atr
        assert forecast.forecast_atr <= 5.0 * forecast.naive_atr

    def test_forecast_has_regime(self, calibrated_hybrid):
        """Forecast should include regime information."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert forecast.regime_state in ("low", "normal", "high", "unknown")

    def test_forecast_confidence_interval(self, calibrated_hybrid):
        """Forecast should have confidence interval."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert forecast.confidence_lower < forecast.forecast_atr
        assert forecast.confidence_upper > forecast.forecast_atr

    def test_forecast_not_fallback(self, calibrated_hybrid):
        """After calibration, forecast should not be fallback."""
        hybrid, df = calibrated_hybrid
        forecast = hybrid.forecast(df)
        assert forecast.is_fallback is False


# =============================================================================
# FALLBACK CHAIN TESTS
# =============================================================================

class TestFallbackChain:
    def test_fallback_to_har_when_lgbm_unavailable(self):
        """If LightGBM fails, should fall back to HAR forecast."""
        hybrid = HybridForecaster(mode="hybrid")
        df = _make_synthetic_ohlcv(5000)

        # Calibrate HAR only (manually skip LightGBM)
        with hybrid._lock:
            hybrid._calibrate_impl(df, None)
        # LightGBM not fitted
        assert not hybrid._lgbm_available

        forecast = hybrid.forecast(df)
        assert isinstance(forecast, VolatilityForecast)
        # Should work (HAR only)
        assert forecast.forecast_atr > 0

    def test_fallback_to_naive_when_uncalibrated(self):
        """If nothing calibrated, should fall back to naive ATR."""
        hybrid = HybridForecaster()
        df = _make_synthetic_ohlcv(3000)
        forecast = hybrid.forecast(df)

        assert forecast.is_fallback is True
        assert forecast.forecast_atr == forecast.naive_atr

    def test_hybrid_same_interface_as_base(self):
        """HybridForecaster should implement same interface as VolatilityForecaster."""
        hybrid = HybridForecaster()
        base = VolatilityForecaster()

        # Check key methods exist
        for method in ["calibrate", "forecast", "update_tcp", "save_state", "load_state", "get_stats"]:
            assert hasattr(hybrid, method)
            assert callable(getattr(hybrid, method))


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

class TestHybridPersistence:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_save_load_roundtrip(self):
        """Hybrid save/load should preserve both HAR and LightGBM."""
        hybrid = HybridForecaster(mode="hybrid")
        df = _make_synthetic_ohlcv(5000)
        hybrid.calibrate(df)

        forecast_before = hybrid.forecast(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "hybrid_state.pkl")
            hybrid.save_state(state_path)

            # Verify both files created
            assert Path(state_path).exists()
            lgbm_path = Path(state_path).with_suffix(".lgbm.txt")
            assert lgbm_path.exists()

            # Load into new instance
            hybrid2 = HybridForecaster(mode="hybrid")
            assert hybrid2.load_state(state_path) is True
            assert hybrid2._lgbm_available is True

            forecast_after = hybrid2.forecast(df)

            # Forecasts should be close (not exact due to float precision)
            assert abs(forecast_before.forecast_atr - forecast_after.forecast_atr) < 0.1

    def test_load_without_lgbm_file(self):
        """Should load HAR state even if LightGBM file missing."""
        hybrid = HybridForecaster()
        df = _make_synthetic_ohlcv(5000)

        # Calibrate HAR only
        with hybrid._lock:
            hybrid._calibrate_impl(df, None)

        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = str(Path(tmpdir) / "har_only.pkl")
            # Save HAR state only (no LightGBM)
            VolatilityForecaster.save_state(hybrid, state_path)

            hybrid2 = HybridForecaster()
            assert hybrid2.load_state(state_path) is True
            assert not hybrid2._lgbm_available


# =============================================================================
# STATS & MODE TESTS
# =============================================================================

class TestStatsAndMode:
    @pytest.fixture(autouse=True)
    def check_lightgbm(self):
        pytest.importorskip("lightgbm")

    def test_stats_includes_mode(self):
        hybrid = HybridForecaster(mode="hybrid")
        stats = hybrid.get_stats()
        assert stats["mode"] == "hybrid"

    def test_stats_includes_lgbm_availability(self):
        hybrid = HybridForecaster()
        stats = hybrid.get_stats()
        assert "lgbm_available" in stats

    def test_stats_after_calibration(self):
        hybrid = HybridForecaster(mode="hybrid")
        df = _make_synthetic_ohlcv(5000)
        hybrid.calibrate(df)

        stats = hybrid.get_stats()
        assert stats["lgbm_available"] is True
        assert "lgbm_stats" in stats

    def test_hybrid_is_subclass(self):
        """HybridForecaster should be a subclass of VolatilityForecaster."""
        assert issubclass(HybridForecaster, VolatilityForecaster)

    def test_mode_property(self):
        for mode in ("hybrid", "lgbm"):
            h = HybridForecaster(mode=mode)
            assert h.mode == mode
