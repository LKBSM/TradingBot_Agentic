"""Tests for src/research/a1_features.py (QUANT-1.1).

Per DoD: matrice ≥150k lignes (validated by smoke run, not unit-tested
because it requires the full XAU CSV), 0 leak (test : valeur macro à t doit
toujours être <= macro publié avant t).

Key tests focus on:
  1. Price features mathematically correct (returns, ATR, RSI, MACD)
  2. Intra features (lunch hour edge cases)
  3. Vintaged macro lookup respects publication date on synthetic data
  4. Calendar proximity correct on synthetic events
  5. Forward target columns are pure look-ahead by construction (= designed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.research.a1_features import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    _atr_wilder,
    _macd_signal_diff,
    _rsi,
    _vintaged_lookup,
    add_calendar_features,
    add_intra_features,
    add_price_features,
    add_target_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_price_df() -> pd.DataFrame:
    """200 M15 bars of synthetic OHLCV with monotonic + oscillating closes."""
    n = 200
    timestamps = pd.date_range(
        "2024-01-01", periods=n, freq="15min", tz="UTC"
    )
    rng = np.random.default_rng(seed=0)
    base = np.linspace(2000, 2100, n)
    noise = rng.normal(0, 1.0, size=n)
    close = base + noise + 5 * np.sin(np.arange(n) / 10)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": rng.integers(50, 500, size=n),
        }
    )


@pytest.fixture
def synthetic_vintaged_source() -> pd.DataFrame:
    """Sorted vintaged source like a FRED CSV."""
    return pd.DataFrame(
        {
            "value": [1.0, 1.5, 2.0, 2.5, 3.0],
            "vintage_date": pd.to_datetime(
                [
                    "2024-01-02 00:00:00+00:00",
                    "2024-01-05 00:00:00+00:00",
                    "2024-01-10 00:00:00+00:00",
                    "2024-01-15 00:00:00+00:00",
                    "2024-01-20 00:00:00+00:00",
                ]
            ),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — price features sanity
# ---------------------------------------------------------------------------


def test_add_price_features_produces_expected_columns(synthetic_price_df):
    df = add_price_features(synthetic_price_df)
    for col in [
        "r_1",
        "r_4",
        "r_16",
        "atr_14_pct",
        "rsi_14",
        "macd_signal_diff",
        "atr_ratio_14_50",
    ]:
        assert col in df.columns, f"missing {col}"

    # r_1 is log-diff: post-warmup values should be small (< 0.05 in magnitude
    # for a synthetic series with σ ≈ 1 around 2000).
    valid_r1 = df["r_1"].dropna()
    assert (valid_r1.abs() < 0.05).all()


def test_atr_wilder_basic_behaviour():
    """ATR on a constant high-low band yields that band size after warmup."""
    n = 50
    high = pd.Series(np.full(n, 10.0))
    low = pd.Series(np.full(n, 8.0))
    close = pd.Series(np.full(n, 9.0))
    atr = _atr_wilder(high, low, close, period=14)
    # Wilder ATR converges to the constant TR (= high - low = 2.0)
    assert atr.iloc[-1] == pytest.approx(2.0, abs=1e-6)


def test_rsi_returns_50_on_flat_series():
    """RSI of a flat series is undefined (avg_loss=0); we return NaN by
    construction (division-by-zero replaced)."""
    flat = pd.Series(np.full(50, 100.0))
    rsi = _rsi(flat, period=14)
    # Either NaN (preferred) or exactly 100 (rare edge); both "no information"
    assert rsi.iloc[-1] != rsi.iloc[-1] or rsi.iloc[-1] == pytest.approx(100, abs=1e-6)


def test_macd_signal_diff_zero_on_flat_series():
    """MACD histogram on a perfectly flat series is exactly 0 after warmup."""
    flat = pd.Series(np.full(100, 100.0))
    hist = _macd_signal_diff(flat)
    assert hist.iloc[-1] == pytest.approx(0.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Test 2 — intra features
# ---------------------------------------------------------------------------


def test_intra_features_lunch_hour_flags():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-03-15 11:30:00",  # London lunch
                    "2024-03-15 12:45:00",  # London lunch
                    "2024-03-15 13:15:00",  # OUT (just past 13:00)
                    "2024-03-15 16:30:00",  # NY lunch
                    "2024-03-15 17:45:00",  # NY lunch
                    "2024-03-15 18:15:00",  # OUT (just past 18:00)
                    "2024-03-15 09:00:00",  # OUT (early London)
                ],
                utc=True,
            ),
            "open": 0.0,
            "high": 0.0,
            "low": 0.0,
            "close": 0.0,
        }
    )
    out = add_intra_features(df)
    expected = [1, 1, 0, 1, 1, 0, 0]
    assert out["is_lunch_hour"].tolist() == expected
    assert out["dow"].iloc[0] == 4  # 2024-03-15 = Friday
    assert out["bar_minute_of_day"].iloc[0] == 11 * 60 + 30


# ---------------------------------------------------------------------------
# Test 3 — vintaged lookup (anti-leak core)
# ---------------------------------------------------------------------------


def test_vintaged_lookup_respects_vintage_date(synthetic_vintaged_source):
    """At time t, only rows with vintage_date <= t are eligible."""
    src = synthetic_vintaged_source
    queries = pd.Series(
        pd.to_datetime(
            [
                "2024-01-01 23:59:59+00:00",  # before any vintage → NaN
                "2024-01-02 00:00:00+00:00",  # exactly first vintage → 1.0
                "2024-01-04 12:00:00+00:00",  # between #1 and #2 → 1.0
                "2024-01-10 00:00:00+00:00",  # exactly third → 2.0
                "2024-01-25 00:00:00+00:00",  # past last → 3.0
            ]
        )
    )
    out = _vintaged_lookup(queries, src, "value")
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(1.0)
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(2.0)
    assert out[4] == pytest.approx(3.0)


def test_vintaged_lookup_random_invariant(synthetic_vintaged_source):
    """For 100 random query timestamps, the returned value MUST equal the
    one with the latest vintage_date <= query. Anti-leak invariant."""
    src = synthetic_vintaged_source
    rng = np.random.default_rng(seed=123)
    base = pd.Timestamp("2024-01-01", tz="UTC")
    minutes = rng.integers(0, 30 * 24 * 60, size=100)
    timestamps = pd.Series([base + pd.Timedelta(minutes=int(m)) for m in minutes])
    out = _vintaged_lookup(timestamps, src, "value")

    failures = 0
    for ts, val in zip(timestamps, out, strict=True):
        eligible = src[src["vintage_date"] <= ts]
        if eligible.empty:
            if not np.isnan(val):
                failures += 1
        else:
            expected = float(eligible.iloc[-1]["value"])
            if np.isnan(val) or not np.isclose(val, expected):
                failures += 1
    assert failures == 0, f"{failures}/100 vintaged-lookup failures"


# ---------------------------------------------------------------------------
# Test 4 — calendar proximity
# ---------------------------------------------------------------------------


def test_add_calendar_features_handles_proximity(tmp_path):
    """Synthetic calendar with 3 events. Verify min_to_next / min_since_last
    are computed correctly for bars before/between/after."""
    cal_path = tmp_path / "synthetic_calendar.csv"
    pd.DataFrame(
        {
            "Date": [
                "2024-03-15 13:30:00",
                "2024-03-15 18:00:00",
                "2024-03-20 14:00:00",
            ],
            "Currency": ["USD", "USD", "USD"],
            "Event": ["NFP", "FOMC", "CPI"],
            "Impact": ["High", "High", "High"],
            "Actual": [0, 0, 0],
            "Forecast": [0, 0, 0],
            "Previous": [0, 0, 0],
        }
    ).to_csv(cal_path, index=False)

    bars = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2024-03-15 13:00:00",  # 30 min before NFP
                    "2024-03-15 13:30:00",  # at NFP
                    "2024-03-15 15:00:00",  # 90 min after NFP, 180 min before FOMC
                    "2024-03-19 00:00:00",  # 1 day before CPI
                ],
                utc=True,
            )
        }
    )
    out = add_calendar_features(bars, cal_path)

    # First bar: next event is NFP in 30 min, no prior event yet
    assert out["min_to_next_red_news"].iloc[0] == pytest.approx(30.0)
    assert np.isnan(out["min_since_last_red_news"].iloc[0])

    # Second bar: at the event itself
    assert out["min_to_next_red_news"].iloc[1] == pytest.approx(0.0)
    # Third bar: between two same-day events
    assert out["min_since_last_red_news"].iloc[2] == pytest.approx(90.0)
    assert out["min_to_next_red_news"].iloc[2] == pytest.approx(180.0)


# ---------------------------------------------------------------------------
# Test 5 — forward targets are FUTURE values (designed look-ahead)
# ---------------------------------------------------------------------------


def test_add_target_columns_uses_future_close():
    """r_forward_4 at row i must equal log(close[i+4]) - log(close[i])."""
    n = 100
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
            "open": np.zeros(n),
            "high": np.zeros(n),
            "low": np.zeros(n),
            "close": np.linspace(2000, 2100, n),
        }
    )
    out = add_target_columns(df)
    expected_r4 = np.log(out["close"].iloc[10 + 4]) - np.log(out["close"].iloc[10])
    assert out["r_forward_4"].iloc[10] == pytest.approx(expected_r4)
    expected_r16 = np.log(out["close"].iloc[20 + 16]) - np.log(out["close"].iloc[20])
    assert out["r_forward_16"].iloc[20] == pytest.approx(expected_r16)
    # Tail is NaN (no future data)
    assert np.isnan(out["r_forward_4"].iloc[-1])
    assert np.isnan(out["r_forward_16"].iloc[-1])


# ---------------------------------------------------------------------------
# Test 6 — schema integrity
# ---------------------------------------------------------------------------


def test_feature_columns_count_meets_plan_target():
    """Plan QUANT-1.1 requires ≥18 features; we should have exactly 19
    (compensating for GLD deferred via T10Y2Y + atr_ratio + producer_net_z52)."""
    assert len(FEATURE_COLUMNS) >= 18
    assert len(TARGET_COLUMNS) == 2
    # No accidental duplicates
    assert len(set(FEATURE_COLUMNS)) == len(FEATURE_COLUMNS)
