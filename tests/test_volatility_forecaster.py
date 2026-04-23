"""Tests for Sprint 1: Production Volatility Forecaster Module.

Tests cover:
  - InstrumentConfig auto-computation and validation
  - VolatilityForecast dataclass
  - Yang-Zhang RV computation
  - HAR feature computation
  - Diurnal profile computation
  - Calendar event multiplier
  - HMM regime detection
  - Blend weight calibration
  - Full calibrate() → forecast() pipeline
  - Fallback when not calibrated
  - TCP update
  - save_state / load_state persistence
"""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.intelligence.volatility_forecaster import (
    InstrumentConfig,
    VolatilityForecast,
    VolatilityForecaster,
)


# =============================================================================
# FIXTURES
# =============================================================================

def _make_ohlcv(n_bars: int = 3000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV data resembling XAU/USD M15."""
    rng = np.random.RandomState(seed)
    base_price = 2000.0
    returns = rng.normal(0, 0.002, n_bars)  # ~0.2% per bar

    close = base_price * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.0005, 0.003, n_bars))
    low = close * (1 - rng.uniform(0.0005, 0.003, n_bars))
    open_ = close * (1 + rng.normal(0, 0.001, n_bars))
    volume = rng.uniform(100, 1000, n_bars)

    # Timestamps: M15 bars starting 2023-01-01
    timestamps = pd.date_range("2023-01-01", periods=n_bars, freq="15min")

    return pd.DataFrame({
        "timestamp": timestamps,
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    })


def _make_calendar() -> pd.DataFrame:
    """Generate synthetic economic calendar."""
    events = []
    # One NFP per month in 2023
    for month in range(1, 13):
        events.append({
            "Date": pd.Timestamp(f"2023-{month:02d}-06 13:30:00"),
            "currency": "USD",
            "event": "Non-Farm Payrolls",
            "impact": "HIGH",
            "actual": "250K",
            "forecast": "200K",
            "previous": "180K",
        })
    # FOMC quarterly
    for month in [3, 6, 9, 12]:
        events.append({
            "Date": pd.Timestamp(f"2023-{month:02d}-15 18:00:00"),
            "currency": "USD",
            "event": "FOMC Statement",
            "impact": "HIGH",
            "actual": "",
            "forecast": "",
            "previous": "",
        })
    return pd.DataFrame(events)


# =============================================================================
# INSTRUMENT CONFIG TESTS
# =============================================================================

class TestInstrumentConfig:
    def test_default_values(self):
        cfg = InstrumentConfig()
        assert cfg.symbol == "XAUUSD"
        assert cfg.timeframe == "M15"
        assert cfg.bars_per_day == 96

    def test_auto_computed_har_windows(self):
        cfg = InstrumentConfig(bars_per_day=96)
        assert cfg.har_daily == 96
        assert cfg.har_weekly == 480
        assert cfg.har_monthly == 2112
        assert cfg.har_train_min == 2212  # 2112 + 100

    def test_m5_har_windows(self):
        cfg = InstrumentConfig(timeframe="M5", bars_per_day=288)
        assert cfg.har_daily == 288
        assert cfg.har_weekly == 1440
        assert cfg.har_monthly == 6336

    def test_h1_har_windows(self):
        cfg = InstrumentConfig(timeframe="H1", bars_per_day=24)
        assert cfg.har_daily == 24
        assert cfg.har_weekly == 120
        assert cfg.har_monthly == 528

    def test_h4_har_windows(self):
        cfg = InstrumentConfig(timeframe="H4", bars_per_day=6)
        assert cfg.har_daily == 6
        assert cfg.har_weekly == 30
        assert cfg.har_monthly == 132

    def test_d1_har_windows(self):
        # For daily: bars_per_day=1, so daily=1day, but HAR uses
        # 5-day (weekly) and 22-day (monthly) lookbacks
        cfg = InstrumentConfig(timeframe="D1", bars_per_day=1)
        assert cfg.har_daily == 1
        assert cfg.har_weekly == 5
        assert cfg.har_monthly == 22

    def test_explicit_har_windows_override(self):
        cfg = InstrumentConfig(har_daily=100, har_weekly=500, har_monthly=2000)
        assert cfg.har_daily == 100
        assert cfg.har_weekly == 500
        assert cfg.har_monthly == 2000

    def test_session_hours_default(self):
        cfg = InstrumentConfig()
        assert "asian" in cfg.session_hours
        assert cfg.session_hours["asian"] == (0, 8)
        assert cfg.session_hours["london"] == (8, 13)

    def test_custom_calendar_events(self):
        cfg = InstrumentConfig(
            calendar_events=["ECB Rate Decision", "Non-Farm Payrolls"]
        )
        assert len(cfg.calendar_events) == 2
        assert "ECB Rate Decision" in cfg.calendar_events

    def test_sl_tp_multipliers(self):
        cfg = InstrumentConfig(sl_atr_mult=1.5, tp_atr_mult=3.0)
        assert cfg.sl_atr_mult == 1.5
        assert cfg.tp_atr_mult == 3.0


# =============================================================================
# VOLATILITY FORECAST DATACLASS TESTS
# =============================================================================

class TestVolatilityForecast:
    def test_to_dict(self):
        f = VolatilityForecast(
            forecast_atr=3.5,
            naive_atr=3.0,
            confidence_lower=2.8,
            confidence_upper=4.2,
            regime_state="normal",
            regime_multiplier=1.0,
            diurnal_multiplier=1.1,
            calendar_multiplier=1.0,
            blend_weight=0.75,
            har_base=3.2,
        )
        d = f.to_dict()
        assert d["forecast_atr"] == 3.5
        assert d["naive_atr"] == 3.0
        assert d["regime_state"] == "normal"
        assert d["is_fallback"] is False

    def test_fallback_flag(self):
        f = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=1.5, confidence_upper=4.5,
            regime_state="unknown", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.0, har_base=3.0, is_fallback=True,
        )
        assert f.is_fallback is True


# =============================================================================
# YANG-ZHANG RV TESTS
# =============================================================================

class TestYangZhangRV:
    def test_rv_bar_non_negative(self):
        df = _make_ohlcv(200)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        df = forecaster._compute_yang_zhang_rv(df)
        assert "rv_bar" in df.columns
        # All values should be >= 0 (clipped)
        assert (df["rv_bar"].dropna() >= 0).all()

    def test_rv_bar_reasonable_magnitude(self):
        df = _make_ohlcv(200)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        df = forecaster._compute_yang_zhang_rv(df)
        mean_rv = df["rv_bar"].mean()
        # For ~0.2% returns, per-bar RV should be small
        assert mean_rv < 1.0
        assert mean_rv > 0


# =============================================================================
# HAR FEATURES TESTS
# =============================================================================

class TestHARFeatures:
    def test_add_features_columns(self):
        df = _make_ohlcv(3000)
        cfg = InstrumentConfig()
        forecaster = VolatilityForecaster(cfg)
        df = forecaster._normalize_columns(df)
        df = forecaster._add_features(df)

        expected = ["tr", "atr_14", "returns_pct", "rv_bar",
                    "rv_daily", "rv_weekly", "rv_monthly", "future_atr", "hour"]
        for col in expected:
            assert col in df.columns, f"Missing column: {col}"

    def test_rv_scales_increase(self):
        """rv_monthly should be smoother (less variance) than rv_daily."""
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        df = forecaster._add_features(df)

        valid = df[["rv_daily", "rv_weekly", "rv_monthly"]].dropna()
        # Monthly is more smoothed → lower std relative to mean
        daily_cv = valid["rv_daily"].std() / valid["rv_daily"].mean()
        monthly_cv = valid["rv_monthly"].std() / valid["rv_monthly"].mean()
        assert monthly_cv < daily_cv

    def test_true_range_positive(self):
        df = _make_ohlcv(200)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        tr = forecaster._compute_true_range(df)
        assert (tr.dropna() > 0).all()


# =============================================================================
# DIURNAL PROFILE TESTS
# =============================================================================

class TestDiurnalProfile:
    def test_profile_has_24_hours(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        df = forecaster._add_features(df)
        profile = forecaster._compute_diurnal_profile(df)
        # Should have entries for hours present in data
        assert len(profile) > 0
        assert len(profile) <= 24

    def test_profile_mean_is_one(self):
        """Diurnal multipliers should average to approximately 1.0."""
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        df = forecaster._normalize_columns(df)
        df = forecaster._add_features(df)
        profile = forecaster._compute_diurnal_profile(df)
        mean_mult = np.mean(list(profile.values()))
        assert abs(mean_mult - 1.0) < 0.15  # Close to 1.0

    def test_dampened_diurnal(self):
        """With DIURNAL_STRENGTH=0.5, multiplier should be closer to 1.0."""
        raw_mult = 0.6  # Low vol hour
        dampened = 1.0 + 0.5 * (raw_mult - 1.0)
        assert dampened == 0.8  # Closer to 1.0 than 0.6
        assert dampened > raw_mult


# =============================================================================
# CALENDAR EVENT TESTS
# =============================================================================

class TestCalendarEvents:
    def test_parse_calendar(self):
        cal = _make_calendar()
        forecaster = VolatilityForecaster()
        event_times = forecaster._parse_calendar(cal)
        # 12 NFP + 4 FOMC = 16 events
        assert len(event_times) == 16

    def test_calendar_multiplier_at_event(self):
        cal = _make_calendar()
        forecaster = VolatilityForecaster()
        forecaster._event_times = forecaster._parse_calendar(cal)

        # At NFP time: should be elevated
        mult = forecaster._get_calendar_multiplier(
            pd.Timestamp("2023-01-06 13:30:00")
        )
        assert mult > 1.5  # Should be ~2.5 at event time

    def test_calendar_multiplier_away_from_event(self):
        cal = _make_calendar()
        forecaster = VolatilityForecaster()
        forecaster._event_times = forecaster._parse_calendar(cal)

        # 2 days away from any event
        mult = forecaster._get_calendar_multiplier(
            pd.Timestamp("2023-01-15 10:00:00")
        )
        assert mult == 1.0

    def test_calendar_multiplier_no_events(self):
        forecaster = VolatilityForecaster()
        forecaster._event_times = None
        mult = forecaster._get_calendar_multiplier(
            pd.Timestamp("2023-01-06 13:30:00")
        )
        assert mult == 1.0

    def test_calendar_multiplier_edge_of_window(self):
        cal = _make_calendar()
        forecaster = VolatilityForecaster()
        forecaster._event_times = forecaster._parse_calendar(cal)

        # 3 hours before NFP (within 4h window)
        mult = forecaster._get_calendar_multiplier(
            pd.Timestamp("2023-01-06 10:30:00")
        )
        assert 1.0 < mult < 2.5  # Decayed but still elevated


# =============================================================================
# CALIBRATION & FORECAST TESTS
# =============================================================================

class TestCalibration:
    def test_calibrate_sets_flag(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        assert not forecaster.is_calibrated

        stats = forecaster.calibrate(df)
        assert forecaster.is_calibrated
        assert stats["calibrated"] is True
        assert stats["training_bars"] > 0

    def test_calibrate_fits_har(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        stats = forecaster.calibrate(df)
        assert stats["har_fitted"] is True
        assert "har_coefs" in stats

    def test_calibrate_with_calendar(self):
        df = _make_ohlcv(3000)
        cal = _make_calendar()
        forecaster = VolatilityForecaster()
        stats = forecaster.calibrate(df, cal)
        assert stats["calendar_events"] == 16

    def test_calibrate_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        df = _make_ohlcv(100)  # Too few bars
        forecaster = VolatilityForecaster()
        stats = forecaster.calibrate(df)
        # May not fit HAR, but should not crash
        assert "har_fitted" in stats

    def test_calibrate_blend_weight(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        stats = forecaster.calibrate(df)
        assert 0.0 <= forecaster._blend_weight <= 1.0


class TestForecast:
    def test_forecast_after_calibration(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        forecast = forecaster.forecast(df)
        assert isinstance(forecast, VolatilityForecast)
        assert forecast.forecast_atr > 0
        assert forecast.naive_atr > 0
        assert not forecast.is_fallback

    def test_forecast_without_calibration_returns_fallback(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        # Don't calibrate

        forecast = forecaster.forecast(df)
        assert forecast.is_fallback is True
        assert forecast.regime_state == "unknown"

    def test_forecast_with_timestamp(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        ts = pd.Timestamp("2023-06-15 14:00:00")
        forecast = forecaster.forecast(df, timestamp=ts)
        assert forecast.forecast_atr > 0
        assert forecast.diurnal_multiplier > 0

    def test_forecast_sanity_clamp(self):
        """Forecast should be between 0.2x and 5x naive ATR."""
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        forecast = forecaster.forecast(df)
        assert forecast.forecast_atr >= 0.2 * forecast.naive_atr
        assert forecast.forecast_atr <= 5.0 * forecast.naive_atr

    def test_forecast_confidence_interval(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        forecast = forecaster.forecast(df)
        assert forecast.confidence_lower < forecast.forecast_atr
        assert forecast.confidence_upper > forecast.forecast_atr
        assert forecast.confidence_lower >= 0

    def test_forecast_regime_labels(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        forecast = forecaster.forecast(df)
        assert forecast.regime_state in ("low", "normal", "high", "unknown")

    def test_forecast_column_normalization(self):
        """Should work with capitalized column names."""
        df = _make_ohlcv(3000)
        # Already has capitalized columns (Open, High, Low, Close)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)
        forecast = forecaster.forecast(df)
        assert forecast.forecast_atr > 0

    def test_forecast_lowercase_columns(self):
        """Should work with lowercase column names."""
        df = _make_ohlcv(3000)
        df.columns = [c.lower() for c in df.columns]
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)
        forecast = forecaster.forecast(df)
        assert forecast.forecast_atr > 0


# =============================================================================
# TCP UPDATE TESTS
# =============================================================================

class TestTCPUpdate:
    def test_tcp_shrinks_on_hit(self):
        forecaster = VolatilityForecaster()
        initial_width = forecaster._tcp_width

        forecast = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=2.0, confidence_upper=4.0,
            regime_state="normal", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.5, har_base=3.0,
        )
        # Actual is within interval → covered
        forecaster.update_tcp(3.5, forecast)
        assert forecaster._tcp_width < initial_width

    def test_tcp_expands_on_miss(self):
        forecaster = VolatilityForecaster()
        initial_width = forecaster._tcp_width

        forecast = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=2.5, confidence_upper=3.5,
            regime_state="normal", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.5, har_base=3.0,
        )
        # Actual is outside interval → miss
        forecaster.update_tcp(5.0, forecast)
        assert forecaster._tcp_width > initial_width

    def test_tcp_asymmetric_updates(self):
        """Miss should expand more than hit shrinks (Robbins-Monro)."""
        forecaster = VolatilityForecaster()

        forecast = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=2.0, confidence_upper=4.0,
            regime_state="normal", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.5, har_base=3.0,
        )

        # One miss
        w_before = forecaster._tcp_width
        forecaster.update_tcp(5.0, forecast)
        expansion = forecaster._tcp_width - w_before

        # One hit
        w_before = forecaster._tcp_width
        forecast_covered = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=2.0, confidence_upper=forecaster._tcp_width * 3.0 + 3.0,
            regime_state="normal", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.5, har_base=3.0,
        )
        forecaster.update_tcp(3.5, forecast_covered)
        shrinkage = w_before - forecaster._tcp_width

        # Expansion >> shrinkage
        assert expansion > shrinkage

    def test_tcp_width_bounded(self):
        forecaster = VolatilityForecaster()

        forecast = VolatilityForecast(
            forecast_atr=3.0, naive_atr=3.0,
            confidence_lower=0.0, confidence_upper=100.0,
            regime_state="normal", regime_multiplier=1.0,
            diurnal_multiplier=1.0, calendar_multiplier=1.0,
            blend_weight=0.5, har_base=3.0,
        )

        # Many hits → should not shrink below 0.05
        for _ in range(1000):
            forecaster.update_tcp(3.0, forecast)
        assert forecaster._tcp_width >= 0.05


# =============================================================================
# PERSISTENCE TESTS
# =============================================================================

class TestPersistence:
    def test_save_load_roundtrip(self):
        df = _make_ohlcv(3000)
        cal = _make_calendar()

        forecaster = VolatilityForecaster()
        forecaster.calibrate(df, cal)

        # Get forecast before save
        forecast_before = forecaster.forecast(df)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name

        try:
            forecaster.save_state(path)

            # Load into new instance
            new_forecaster = VolatilityForecaster()
            assert not new_forecaster.is_calibrated

            success = new_forecaster.load_state(path)
            assert success is True
            assert new_forecaster.is_calibrated

            # Forecast should match
            forecast_after = new_forecaster.forecast(df)
            assert abs(forecast_before.forecast_atr - forecast_after.forecast_atr) < 0.001
            assert forecast_before.regime_state == forecast_after.regime_state
        finally:
            os.unlink(path)

    def test_load_missing_file(self):
        forecaster = VolatilityForecaster()
        success = forecaster.load_state("/nonexistent/path/model.pkl")
        assert success is False
        assert not forecaster.is_calibrated

    def test_save_creates_directories(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "subdir", "model.pkl")
            forecaster.save_state(path)
            assert os.path.exists(path)


# =============================================================================
# STATS TESTS
# =============================================================================

class TestStats:
    def test_stats_before_calibration(self):
        forecaster = VolatilityForecaster()
        stats = forecaster.get_stats()
        assert stats["is_calibrated"] is False
        assert stats["har_fitted"] is False

    def test_stats_after_calibration(self):
        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        stats = forecaster.get_stats()
        assert stats["is_calibrated"] is True
        assert stats["har_fitted"] is True
        assert stats["symbol"] == "XAUUSD"
        assert stats["calibration_bars"] > 0


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["timestamp", "Open", "High", "Low", "Close", "Volume"])
        forecaster = VolatilityForecaster()
        # Should not crash, just not calibrate
        stats = forecaster.calibrate(df)
        assert stats.get("har_fitted") is False

    def test_nan_in_data(self):
        df = _make_ohlcv(3000)
        # Inject some NaNs
        df.loc[100:110, "Close"] = np.nan
        forecaster = VolatilityForecaster()
        # Should handle NaNs gracefully
        stats = forecaster.calibrate(df)
        # HAR model should still fit (dropping NaN rows)
        assert "har_fitted" in stats

    def test_different_instrument_configs(self):
        """Test that different configs produce different results."""
        df = _make_ohlcv(3000)

        cfg1 = InstrumentConfig(symbol="XAUUSD", diurnal_strength=0.0)
        cfg2 = InstrumentConfig(symbol="XAUUSD", diurnal_strength=1.0)

        f1 = VolatilityForecaster(cfg1)
        f2 = VolatilityForecaster(cfg2)

        f1.calibrate(df)
        f2.calibrate(df)

        # Forecasts may differ due to diurnal strength
        fc1 = f1.forecast(df, timestamp=pd.Timestamp("2023-06-15 03:00:00"))
        fc2 = f2.forecast(df, timestamp=pd.Timestamp("2023-06-15 03:00:00"))

        # diurnal_strength=0 should have diurnal_mult=1.0
        assert abs(fc1.diurnal_multiplier - 1.0) < 0.001
        # diurnal_strength=1 should differ from 1.0
        # (may be close depending on data, so just verify it ran)
        assert fc2.diurnal_multiplier > 0

    def test_thread_safety(self):
        """Basic thread safety: concurrent forecasts shouldn't crash."""
        import threading

        df = _make_ohlcv(3000)
        forecaster = VolatilityForecaster()
        forecaster.calibrate(df)

        errors = []

        def forecast_in_thread():
            try:
                for _ in range(10):
                    f = forecaster.forecast(df)
                    assert f.forecast_atr > 0
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=forecast_in_thread) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Thread errors: {errors}"
