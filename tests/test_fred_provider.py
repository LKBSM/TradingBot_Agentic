"""Tests for src/agents/data/fred_provider.py (DATA-1.1).

Five required tests per DoD (`PLAN_12_MOIS.md` Partie II.2 Agent 1):
  1. fetch_series returns DataFrame with the right schema
  2. vintage_date is present and >= observation_date for all rows
  3. macro_at respects publication lag on 100 random dates
  4. resample_to_m15 produces no NaN after forward-fill
  5. save_to_csv produces the 5 expected CSV files

Plus one optional `@pytest.mark.live` smoke test that hits the real FRED API
when `FRED_API_KEY` is set in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.agents.data.fred_provider import (
    OUTPUT_COLUMNS,
    SERIES_IDS,
    FredProvider,
    _empty_frame,
    _first_release_from_all_releases,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_all_releases() -> pd.DataFrame:
    """Mock of `Fred.get_series_all_releases()` output.

    Schema: date, realtime_start, value. Two observations, the first revised
    once (so its first release should be picked).
    """
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2020-01-01",
                    "2020-01-01",  # revision of the same observation
                    "2020-01-02",
                    "2020-01-03",
                    "2020-01-06",  # weekend gap
                    "2020-01-07",
                ]
            ),
            "realtime_start": pd.to_datetime(
                [
                    "2020-01-02",  # original release
                    "2020-06-15",  # later revision (must be ignored)
                    "2020-01-03",
                    "2020-01-04",
                    "2020-01-07",
                    "2020-01-08",
                ]
            ),
            "value": [1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
        }
    )


@pytest.fixture
def synthetic_first_release_df(synthetic_all_releases) -> pd.DataFrame:
    """First-release frame produced by `_first_release_from_all_releases`."""
    return _first_release_from_all_releases(synthetic_all_releases)


@pytest.fixture
def fred_provider_mocked(monkeypatch, synthetic_all_releases) -> FredProvider:
    """`FredProvider` whose underlying `Fred` client returns synthetic data.

    Avoids hitting the real FRED API and the need for an API key in CI.
    """
    monkeypatch.setenv("FRED_API_KEY", "fake-key-for-testing")

    class _StubFred:
        def __init__(self, *_, **__):
            pass

        def get_series_all_releases(self, _series_id):  # noqa: ARG002
            return synthetic_all_releases.copy()

        def get_series(self, _series_id, **__):  # fallback path
            df = synthetic_all_releases.copy()
            df = df.sort_values(["date", "realtime_start"]).groupby("date").first()
            return pd.Series(df["value"].values, index=df.index)

    monkeypatch.setattr("src.agents.data.fred_provider.Fred", _StubFred)
    return FredProvider()


# ---------------------------------------------------------------------------
# Test 1 — schema
# ---------------------------------------------------------------------------


def test_fetch_series_returns_dataframe_with_correct_schema(fred_provider_mocked):
    """fetch_series() must return columns ['date_utc', 'value', 'vintage_date']
    with the documented dtypes (UTC timezone-aware datetimes, float values)."""
    df = fred_provider_mocked.fetch_series("DGS10")

    assert isinstance(df, pd.DataFrame)
    assert df.columns.tolist() == OUTPUT_COLUMNS

    assert pd.api.types.is_datetime64_any_dtype(df["date_utc"])
    assert df["date_utc"].dt.tz is not None, "date_utc must be tz-aware"
    assert str(df["date_utc"].dt.tz) == "UTC"

    assert pd.api.types.is_float_dtype(df["value"])

    assert pd.api.types.is_datetime64_any_dtype(df["vintage_date"])
    assert df["vintage_date"].dt.tz is not None, "vintage_date must be tz-aware"
    assert str(df["vintage_date"].dt.tz) == "UTC"

    # Empty frame with same schema is also valid.
    empty = _empty_frame()
    assert empty.columns.tolist() == OUTPUT_COLUMNS


# ---------------------------------------------------------------------------
# Test 2 — vintage timestamps present and >= observation date
# ---------------------------------------------------------------------------


def test_vintage_dates_present_and_after_observation_dates(
    fred_provider_mocked, synthetic_first_release_df
):
    """For every row, vintage_date must be present and >= date_utc.

    A vintage strictly before the observation would mean the data was
    "published before it happened" — a leak by definition.
    Also verify that the synthetic DGS10 fixture picks the FIRST release
    (1.5 for 2020-01-01) and not the later revision (1.6).
    """
    df = fred_provider_mocked.fetch_series("DGS10")

    assert df["vintage_date"].notna().all(), "no vintage_date may be NaT"
    assert (df["vintage_date"] >= df["date_utc"]).all(), (
        "vintage_date must always be >= observation date"
    )

    # First release of 2020-01-01 must be 1.5 (the 1.6 revision is later).
    jan_1 = df[df["date_utc"] == pd.Timestamp("2020-01-01", tz="UTC")].iloc[0]
    assert jan_1["value"] == pytest.approx(1.5)
    assert jan_1["vintage_date"] == pd.Timestamp("2020-01-02", tz="UTC")

    # synthetic_first_release_df must agree (used as direct sanity).
    assert (
        synthetic_first_release_df["vintage_date"]
        >= synthetic_first_release_df["date_utc"]
    ).all()


# ---------------------------------------------------------------------------
# Test 3 — macro_at respects publication lag on 100 random dates
# ---------------------------------------------------------------------------


def test_macro_at_respects_publication_lag_100_random_dates(
    fred_provider_mocked,
):
    """For 100 random query timestamps t, macro_at(df, t) must only consider
    rows whose vintage_date <= t. The returned value must equal the most
    recent eligible observation; otherwise it would constitute a look-ahead.
    """
    df = fred_provider_mocked.fetch_series("DGS10")

    rng = np.random.default_rng(seed=42)
    # Generate 100 random query timestamps spanning the data range plus
    # a buffer on either side (to test "before any release" and "after all").
    base = pd.Timestamp("2019-01-01", tz="UTC")
    span_days = 730
    deltas_min = rng.integers(0, span_days * 24 * 60, size=100)
    timestamps = [base + pd.Timedelta(minutes=int(d)) for d in deltas_min]

    failures = []
    for t in timestamps:
        eligible = df[df["vintage_date"] <= t]
        expected = (
            float(eligible.iloc[-1]["value"]) if not eligible.empty else None
        )
        actual = FredProvider.macro_at(df, t)

        if expected is None:
            if actual is not None:
                failures.append((t, expected, actual))
        else:
            if actual is None or not np.isclose(actual, expected):
                failures.append((t, expected, actual))

    assert not failures, (
        f"macro_at violated publication-lag guarantee on "
        f"{len(failures)}/100 timestamps; first 3: {failures[:3]}"
    )


# ---------------------------------------------------------------------------
# Test 4 — resample_to_m15 + ffill produces no NaN
# ---------------------------------------------------------------------------


def test_resample_to_m15_no_nan_after_ffill(fred_provider_mocked):
    """Resampling daily series to M15 with forward-fill must produce
    a contiguous frame without NaN values inside the data range."""
    df = fred_provider_mocked.fetch_series("DGS10")
    m15 = FredProvider.resample_to_m15(df)

    # M15 frame must cover the full date range from first to last observation.
    expected_start = df["date_utc"].min()
    expected_end = df["date_utc"].max()
    assert m15["date_utc"].min() == expected_start
    assert m15["date_utc"].max() == expected_end

    # 0 NaN in value/vintage_date columns after ffill (the requirement).
    assert m15["value"].isna().sum() == 0
    assert m15["vintage_date"].isna().sum() == 0

    # M15 grid is exactly 15 minutes apart.
    deltas = m15["date_utc"].diff().dropna().unique()
    assert len(deltas) == 1
    assert deltas[0] == pd.Timedelta(minutes=15)


# ---------------------------------------------------------------------------
# Test 5 — save_to_csv produces 5 expected files
# ---------------------------------------------------------------------------


def test_save_to_csv_produces_expected_files(tmp_path, fred_provider_mocked):
    """save_to_csv must produce one CSV per series at
    `{output_dir}/fred_{series_id}.csv`. The files must round-trip via
    load_from_csv to identical data (modulo dtype quirks)."""
    series_data = fred_provider_mocked.fetch_all_series()
    paths = FredProvider.save_to_csv(series_data, tmp_path)

    expected_files = {
        sid: tmp_path / f"fred_{sid}.csv" for sid in SERIES_IDS
    }

    assert set(paths.keys()) == set(SERIES_IDS.keys())
    for sid, path in paths.items():
        assert path == expected_files[sid]
        assert path.exists()
        assert path.stat().st_size > 0

    # Round-trip a series and verify the data matches.
    sid = "DGS10"
    loaded = FredProvider.load_from_csv(paths[sid])
    original = series_data[sid].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        loaded.reset_index(drop=True),
        original,
        check_exact=False,
        rtol=1e-9,
    )


# ---------------------------------------------------------------------------
# Test 6 — virtual breakeven_10y vintage = max of inputs (anti-leak)
# ---------------------------------------------------------------------------


def test_compute_breakeven_10y_takes_max_vintage():
    """The breakeven 10y = DGS10 - DFII10 cannot be computed before BOTH
    inputs are published, so its vintage_date must be the MAX of the two."""
    dgs10 = pd.DataFrame(
        {
            "date_utc": pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
            "value": [2.0, 2.1],
            "vintage_date": pd.to_datetime(
                ["2020-01-03", "2020-01-04"], utc=True
            ),
        }
    )
    dfii10 = pd.DataFrame(
        {
            "date_utc": pd.to_datetime(["2020-01-02", "2020-01-03"], utc=True),
            "value": [0.5, 0.4],
            "vintage_date": pd.to_datetime(
                ["2020-01-04", "2020-01-04"], utc=True
            ),
        }
    )

    bk = FredProvider.compute_breakeven_10y(dgs10, dfii10)
    assert bk["value"].tolist() == [pytest.approx(1.5), pytest.approx(1.7)]
    # First row: max(2020-01-03, 2020-01-04) == 2020-01-04
    assert bk["vintage_date"].iloc[0] == pd.Timestamp("2020-01-04", tz="UTC")


# ---------------------------------------------------------------------------
# Optional live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    not os.environ.get("FRED_API_KEY"),
    reason="FRED_API_KEY not set; live smoke test skipped",
)
def test_live_fetch_dgs10_smoke():
    """End-to-end live test: fetch DGS10 from real FRED API.

    Marked `live` so it is excluded from the default test run.
    Run with:  pytest tests/test_fred_provider.py -m live -v
    """
    p = FredProvider()
    df = p.fetch_series("DGS10", start_date="2024-01-01", end_date="2024-01-31")
    assert not df.empty
    assert df.columns.tolist() == OUTPUT_COLUMNS
    assert df["value"].between(0, 8).all(), "DGS10 yields outside sane range"
    assert (df["vintage_date"] >= df["date_utc"]).all()
