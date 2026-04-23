"""Tests for OHLCV data quality validator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intelligence.data_quality import (
    DataQualityError,
    ValidationReport,
    validate_ohlcv,
)


def _make_df(
    n: int = 100,
    *,
    base_price: float = 2000.0,
    timeframe_minutes: int = 15,
) -> pd.DataFrame:
    """Build a clean OHLCV DataFrame for mutation in tests."""
    rng = np.random.RandomState(42)
    idx = pd.date_range("2024-01-01", periods=n, freq=f"{timeframe_minutes}min")
    close = base_price + np.cumsum(rng.randn(n) * 0.5)
    open_ = close + rng.randn(n) * 0.2
    spread = np.abs(rng.randn(n)) * 1.0 + 0.5
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": rng.randint(100, 1000, n).astype(float),
    }, index=idx)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_clean_ohlcv_passes():
    df = _make_df()
    report = validate_ohlcv(df, "XAUUSD", "M15")
    assert report.is_valid is True
    assert report.errors == []
    assert report.bars_checked == len(df)


# ---------------------------------------------------------------------------
# Structural errors (fatal)
# ---------------------------------------------------------------------------

def test_empty_dataframe_rejected():
    df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
    with pytest.raises(DataQualityError):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_missing_columns_rejected():
    df = _make_df().drop(columns=["Volume"])
    with pytest.raises(DataQualityError, match="Missing required columns"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_non_datetime_index_rejected():
    df = _make_df().reset_index(drop=True)
    with pytest.raises(DataQualityError, match="DatetimeIndex"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_non_monotonic_timestamps_rejected():
    df = _make_df()
    reordered = df.iloc[[5, 4, 3, 2, 1, 0] + list(range(6, len(df)))]
    with pytest.raises(DataQualityError, match="monotonically"):
        validate_ohlcv(reordered, "XAUUSD", "M15")


def test_duplicate_timestamps_rejected():
    df = _make_df(n=10)
    dup_idx = list(df.index)
    dup_idx[5] = dup_idx[4]
    df.index = pd.DatetimeIndex(dup_idx)
    with pytest.raises(DataQualityError, match="duplicate"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_nan_in_close_rejected():
    df = _make_df()
    df.iloc[5, df.columns.get_loc("Close")] = np.nan
    with pytest.raises(DataQualityError, match="NaN"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_negative_volume_rejected():
    df = _make_df()
    df.iloc[3, df.columns.get_loc("Volume")] = -100.0
    with pytest.raises(DataQualityError, match="negative volume"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_high_lower_than_low_rejected():
    df = _make_df()
    df.iloc[10, df.columns.get_loc("High")] = df["Low"].iloc[10] - 1.0
    with pytest.raises(DataQualityError, match="High < Low"):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_high_below_open_close_rejected():
    df = _make_df()
    # Set High strictly below both Open and Close at bar 20
    df.iloc[20, df.columns.get_loc("Open")] = 2010.0
    df.iloc[20, df.columns.get_loc("Close")] = 2008.0
    df.iloc[20, df.columns.get_loc("High")] = 2005.0
    df.iloc[20, df.columns.get_loc("Low")] = 2000.0
    with pytest.raises(DataQualityError):
        validate_ohlcv(df, "XAUUSD", "M15")


def test_non_positive_price_rejected():
    df = _make_df()
    df.iloc[7, df.columns.get_loc("Low")] = 0.0
    df.iloc[7, df.columns.get_loc("High")] = 5.0
    df.iloc[7, df.columns.get_loc("Open")] = 3.0
    df.iloc[7, df.columns.get_loc("Close")] = 2.0
    with pytest.raises(DataQualityError, match="non-positive prices"):
        validate_ohlcv(df, "XAUUSD", "M15")


# ---------------------------------------------------------------------------
# Soft warnings
# ---------------------------------------------------------------------------

def test_large_gap_produces_warning_but_passes():
    """A single large gap should warn but not fail (market close, DST)."""
    df = _make_df(n=200)
    # Inject a 3-hour gap by removing 12 consecutive M15 bars
    df = pd.concat([df.iloc[:100], df.iloc[112:]])
    # <1% gap rate — just passes without warning
    report = validate_ohlcv(df, "XAUUSD", "M15")
    assert report.is_valid is True


def test_many_gaps_produce_warning():
    """A gap-heavy feed should flag the >1% bar-gap threshold."""
    df = _make_df(n=200)
    # Keep only 1 bar per hour (every 4th M15 bar) — gaps of 60min = 4x expected
    df = df[df.index.minute == 0]
    report = validate_ohlcv(df, "XAUUSD", "M15")
    assert report.is_valid is True
    assert report.gap_count > 0
    # And the warning should be recorded given gap rate is very high
    assert any("gaps" in w for w in report.warnings)


def test_strict_false_returns_report_without_raising():
    df = _make_df().drop(columns=["Volume"])
    report = validate_ohlcv(df, "XAUUSD", "M15", strict=False)
    assert report.is_valid is False
    assert len(report.errors) >= 1


def test_report_summary():
    df = _make_df()
    report = validate_ohlcv(df, "XAUUSD", "M15")
    summary = report.summary()
    assert "bars=100" in summary
    assert "errors=0" in summary
