"""Tests for NW-3 publication measures — PURE, synthetic candles, no CSV/network.

Builds a ~90-day synthetic M15 gold-like series with KNOWN per-bar ranges and a
synthetic list of "CPI" releases, then asserts the three engine-measured facts
carry honest provenance, honest denominators, whole-minute durations, and are
omitted (None) when too few releases can be measured.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.intelligence.publication_measures import compute_publication_measures
from src.intelligence.publication_measures_schema import PublicationMeasures


BAR = timedelta(minutes=15)
START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)


def _make_candles(days: int = 90, base: float = 2000.0) -> pd.DataFrame:
    """Synthetic M15 OHLCV with a small, deterministic per-bar range everywhere
    except a spike injected AT and just after each release hour (added by the
    caller's release list). Here we build the calm baseline; the release spikes
    are painted on top in the fixture below."""
    n = days * 24 * 4  # 96 bars/day
    idx = pd.date_range(START, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(42)
    # Calm baseline: per-bar range ~ 1.0 USD, slow drift.
    drift = np.cumsum(rng.normal(0.0, 0.05, n))
    close = base + drift
    half = 0.5  # → per-bar high-low ~ 1.0
    df = pd.DataFrame(
        {
            "open": close,
            "high": close + half,
            "low": close - half,
            "close": close,
            "volume": 100.0,
        },
        index=idx,
    )
    return df


def _inject_release_volatility(df: pd.DataFrame, releases, spike: float = 15.0,
                               n_spike_bars: int = 8) -> pd.DataFrame:
    """Widen bars from each release T for a few bars to simulate the reaction,
    so #4 (return_to_calm) measures a real settle time and #1 sees a busy hour
    if we also widen the PRE hour for some releases."""
    df = df.copy()
    for t in releases:
        # widen n_spike_bars starting AT T (the reaction)
        mask = (df.index >= t) & (df.index < t + n_spike_bars * BAR)
        df.loc[mask, "high"] = df.loc[mask, "close"] + spike
        df.loc[mask, "low"] = df.loc[mask, "close"] - spike
    return df


@pytest.fixture
def synthetic():
    df = _make_candles(days=90)
    # 12 releases, all at 13:00 UTC (a fixed publication hour), every ~7 days.
    releases = [START + timedelta(days=3 + 7 * i, hours=13) for i in range(12)]
    # keep only those inside coverage
    releases = [r for r in releases if r <= df.index[-1]]
    df = _inject_release_volatility(df, releases)
    return df, releases


def test_all_measures_have_provenance(synthetic):
    df, releases = synthetic
    m = compute_publication_measures("us_cpi", "XAUUSD", releases, df)
    assert isinstance(m, PublicationMeasures)
    assert m.event_key == "us_cpi"
    assert m.market == "XAUUSD"

    for measure in (m.calm_before, m.return_to_calm, m.structure_state):
        assert measure is not None, "expected the measure to be computable"
        prov = measure.provenance
        assert prov.sample_size > 0
        assert prov.market == "XAUUSD"
        assert prov.period_start <= prov.period_end


def test_calm_before_counts_and_extremes(synthetic):
    df, releases = synthetic
    m = compute_publication_measures("us_cpi", "XAUUSD", releases, df)
    cb = m.calm_before
    assert cb is not None
    # counts partition the sample
    assert cb.calmer_count + cb.busier_count == cb.provenance.sample_size
    assert cb.reference_amount >= 0
    assert cb.provenance.quote_unit == "USD"
    assert cb.provenance.reference_days == 60
    # extremes carry an observed_at and an amount (not minutes)
    assert cb.calmest.observed_at is not None
    assert cb.calmest.amount is not None
    assert cb.busiest.amount is not None
    assert cb.calmest.amount <= cb.busiest.amount


def test_return_to_calm_tranches_sum_to_sample(synthetic):
    df, releases = synthetic
    m = compute_publication_measures("us_cpi", "XAUUSD", releases, df)
    rtc = m.return_to_calm
    assert rtc is not None
    tranche_total = sum(t.count for t in rtc.tranches)
    # tranche counts + never_settled == sample_size
    assert tranche_total + rtc.never_settled_count == rtc.provenance.sample_size
    # durations are whole minutes (ints)
    assert isinstance(rtc.fastest.minutes, int)
    assert isinstance(rtc.slowest.minutes, int)
    assert rtc.fastest.minutes <= rtc.slowest.minutes
    for t in rtc.tranches:
        assert isinstance(t.lower_minutes, int)
        assert t.upper_minutes is None or isinstance(t.upper_minutes, int)
    # extremes carry observed_at
    assert rtc.fastest.observed_at is not None
    assert rtc.slowest.observed_at is not None


def test_structure_state_counts_partition(synthetic):
    df, releases = synthetic
    m = compute_publication_measures("us_cpi", "XAUUSD", releases, df)
    ss = m.structure_state
    assert ss is not None
    band_total = ss.range_lower_count + ss.range_middle_count + ss.range_upper_count
    assert band_total == ss.provenance.sample_size
    assert 0 <= ss.inside_zone_count <= ss.provenance.sample_size
    assert 0 <= ss.intact_pocket_count <= ss.provenance.sample_size
    # no baseline used → reference_days None, quote_unit None
    assert ss.provenance.reference_days is None
    # now_* either populated or None (never fabricated), and bucket is valid
    if ss.now_range_position is not None:
        assert ss.now_range_position in ("lower", "middle", "upper")


def test_too_few_releases_yields_none(synthetic):
    df, releases = synthetic
    # only 3 releases → below the reliability floor (4) → measures omitted
    m = compute_publication_measures("us_cpi", "XAUUSD", releases[:3], df)
    assert m.calm_before is None
    assert m.return_to_calm is None
    assert m.structure_state is None
    assert m.has_any() is False


def test_empty_inputs_are_safe():
    df = _make_candles(days=5)
    m = compute_publication_measures("us_cpi", "XAUUSD", [], df)
    assert m.has_any() is False
    m2 = compute_publication_measures("us_cpi", "XAUUSD",
                                      [START + timedelta(hours=13)],
                                      df.iloc[0:0])
    assert m2.has_any() is False


def test_naive_release_datetimes_are_treated_as_utc(synthetic):
    df, releases = synthetic
    naive = [r.replace(tzinfo=None) for r in releases]
    m = compute_publication_measures("us_cpi", "XAUUSD", naive, df)
    # naive == UTC assumption → same sample as tz-aware releases
    assert m.calm_before is not None
    assert m.calm_before.provenance.sample_size == len(releases)
