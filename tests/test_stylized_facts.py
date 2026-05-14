"""Tests for the REGIME-2B.3 diurnal/calendar stylized facts."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.intelligence.stylized_facts import (
    compute_stylized_facts,
    dow_facts,
    fomc_bucket_facts,
    hourly_facts,
)


@pytest.fixture
def synthetic_df():
    """500 hours of synthetic price data starting 2024-01-01 UTC."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=500, freq="h", tz="UTC")
    # Add a deliberate spike at hour 14 each day so we can verify the
    # hourly aggregation catches it.
    base = rng.normal(0, 0.001, len(idx))
    is_14_utc = idx.hour == 14
    base[is_14_utc] += 0.005  # bigger moves at 14h UTC
    close = (1.0 + pd.Series(base, index=idx)).cumprod()
    return pd.DataFrame({"close": close})


# ---------------------------------------------------------------------------
# hourly_facts
# ---------------------------------------------------------------------------


def test_hourly_returns_24_rows(synthetic_df):
    rows = hourly_facts(synthetic_df)
    assert len(rows) == 24
    assert [r.hour for r in rows] == list(range(24))


def test_hour_14_has_higher_volatility_than_other_hours(synthetic_df):
    rows = hourly_facts(synthetic_df)
    h14 = next(r for r in rows if r.hour == 14)
    other_means = [r.mean_abs_return for r in rows if r.hour != 14 and r.n > 0]
    assert h14.mean_abs_return > np.mean(other_means)


def test_hourly_hit_rate_is_probability(synthetic_df):
    for r in hourly_facts(synthetic_df):
        if r.n > 0:
            assert 0.0 <= r.hit_rate_up <= 1.0


# ---------------------------------------------------------------------------
# dow_facts
# ---------------------------------------------------------------------------


def test_dow_returns_seven_rows(synthetic_df):
    rows = dow_facts(synthetic_df)
    assert [r.dow for r in rows] == list(range(7))


def test_dow_n_sums_close_to_total_returns(synthetic_df):
    # 500 bars → 499 returns (pct_change drops the first)
    total = sum(r.n for r in dow_facts(synthetic_df))
    assert total == 499


# ---------------------------------------------------------------------------
# fomc_bucket_facts
# ---------------------------------------------------------------------------


def test_fomc_bucket_assignment(synthetic_df):
    # 500 hours starting 2024-01-01 → covers ~20 days. Pretend FOMC was
    # 2024-01-10 (middle of the window).
    fomc = [dt.date(2024, 1, 10)]
    rows = fomc_bucket_facts(synthetic_df, fomc)
    buckets = {r.bucket: r.n for r in rows}
    assert "fomc_day" in buckets
    assert buckets["fomc_day"] > 0
    assert "other" in buckets
    assert buckets["other"] > 0


def test_fomc_no_dates_returns_empty(synthetic_df):
    assert fomc_bucket_facts(synthetic_df, []) == []


# ---------------------------------------------------------------------------
# compute_stylized_facts — JSON serialisable
# ---------------------------------------------------------------------------


def test_compute_returns_all_three_keys(synthetic_df):
    out = compute_stylized_facts(synthetic_df, fomc_dates=[dt.date(2024, 1, 10)])
    assert set(out.keys()) == {"hourly", "dow", "fomc"}
    assert len(out["hourly"]) == 24
    assert len(out["dow"]) == 7
    assert len(out["fomc"]) > 0


def test_compute_output_is_json_serialisable(synthetic_df):
    import json

    out = compute_stylized_facts(synthetic_df, fomc_dates=[dt.date(2024, 1, 10)])
    s = json.dumps(out)  # must not raise
    assert "hourly" in s


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_close_and_returns_raises():
    df = pd.DataFrame({"x": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3, freq="h", tz="UTC"))
    with pytest.raises(ValueError, match="returns.*close"):
        hourly_facts(df)


def test_non_datetime_index_raises():
    df = pd.DataFrame({"close": [1, 2, 3]})  # default RangeIndex
    with pytest.raises(ValueError, match="DatetimeIndex"):
        hourly_facts(df)


def test_ts_utc_column_works(synthetic_df):
    df = synthetic_df.reset_index().rename(columns={"index": "ts_utc"})
    rows = hourly_facts(df)
    assert len(rows) == 24
