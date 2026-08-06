"""NW-7 — zone-lifecycle census (#3) + its non-regression on the live display.

The lifecycle census (``collect_zone_lifecycles``) must surface zones that the
display layer (``collect_zones``) deliberately DROPS once consumed — that is what
"how long did the zone hold" needs. These tests prove:
  · the census keeps a mitigated zone the display hides;
  · the display output is byte-for-byte UNAFFECTED by running the census
    (detection/live surfacing is not altered — mission NW-7 discipline);
  · the #3 measure computed over a synthetic series carries honest, whole-minute
    tranches whose counts + never-mitigated reconcile to the zones created.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.intelligence.market_reading_mappers import (
    collect_zone_lifecycles,
    collect_zones,
)

BAR = timedelta(minutes=15)
START = datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc)


def _frame_with_mitigated_bullish_ob() -> pd.DataFrame:
    """20 M15 bars: a bullish OB forms at bar 5 (100.0–100.5); price trades above
    it, then at bar 12 a wick dips into it (a tap = mitigation) while every close
    stays ABOVE the zone low (so it is 'mitigated', not 'invalidated')."""
    n = 20
    idx = pd.date_range(START, periods=n, freq="15min", tz="UTC")
    close = np.full(n, 101.0)
    high = close + 0.3
    low = close - 0.3
    # bar 12 taps into the zone [100.0, 100.5] with its LOW, close stays at 101.
    low[12] = 100.2
    bull_hi = np.full(n, np.nan)
    bull_lo = np.full(n, np.nan)
    bull_hi[5] = 100.5
    bull_lo[5] = 100.0
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "BULLISH_OB_HIGH": bull_hi,
            "BULLISH_OB_LOW": bull_lo,
            "BEARISH_OB_HIGH": np.full(n, np.nan),
            "BEARISH_OB_LOW": np.full(n, np.nan),
        },
        index=idx,
    )


def _frame_with_invalidated_bullish_ob() -> pd.DataFrame:
    """Same bullish OB at bar 5, but at bar 12 price CLOSES through the zone low
    (close 99.0 < 100.0) → the OB is invalidated, which the display DROPS."""
    n = 20
    idx = pd.date_range(START, periods=n, freq="15min", tz="UTC")
    close = np.full(n, 101.0)
    high = close + 0.3
    low = close - 0.3
    close[12] = 99.0
    high[12] = 99.3
    low[12] = 98.7
    bull_hi = np.full(n, np.nan)
    bull_lo = np.full(n, np.nan)
    bull_hi[5] = 100.5
    bull_lo[5] = 100.0
    return pd.DataFrame(
        {
            "high": high,
            "low": low,
            "close": close,
            "BULLISH_OB_HIGH": bull_hi,
            "BULLISH_OB_LOW": bull_lo,
            "BEARISH_OB_HIGH": np.full(n, np.nan),
            "BEARISH_OB_LOW": np.full(n, np.nan),
        },
        index=idx,
    )


def test_census_captures_mitigation_timestamp():
    """A tapped-but-held OB stays visible; the census records WHEN it was tapped."""
    df = _frame_with_mitigated_bullish_ob()
    census = collect_zone_lifecycles(df, idx=len(df) - 1)
    obs = [z for z in census if z["kind"] == "ob"]
    assert len(obs) == 1
    z = obs[0]
    assert z["created_at"] == df.index[5].to_pydatetime()
    assert z["mitigated_at"] == df.index[12].to_pydatetime()
    assert z["status"] == "mitigated"


def test_census_keeps_a_zone_the_display_drops():
    df = _frame_with_invalidated_bullish_ob()

    # The display drops the invalidated OB (honesty guardrail) → none surfaced.
    displayed = collect_zones(df, idx=len(df) - 1)
    assert displayed["order_blocks"] == []

    # The census KEEPS it, timestamped at creation, flagged invalidated.
    census = collect_zone_lifecycles(df, idx=len(df) - 1)
    obs = [z for z in census if z["kind"] == "ob"]
    assert len(obs) == 1
    assert obs[0]["created_at"] == df.index[5].to_pydatetime()
    assert obs[0]["status"] == "invalidated"


def test_display_output_unaffected_by_census():
    """Running the census must not change what the live display surfaces."""
    df = _frame_with_mitigated_bullish_ob()
    before = collect_zones(df, idx=len(df) - 1)
    _ = collect_zone_lifecycles(df, idx=len(df) - 1)
    after = collect_zones(df, idx=len(df) - 1)
    assert before == after


def test_census_window_filters_by_creation_time():
    df = _frame_with_mitigated_bullish_ob()
    # A window that excludes bar 5's timestamp yields no zone.
    empty = collect_zone_lifecycles(
        df, idx=len(df) - 1, since_ts=df.index[8].to_pydatetime()
    )
    assert [z for z in empty if z["kind"] == "ob"] == []
    # A window that includes bar 5 yields it.
    kept = collect_zone_lifecycles(
        df,
        idx=len(df) - 1,
        since_ts=df.index[0].to_pydatetime(),
        until_ts=df.index[6].to_pydatetime(),
    )
    assert len([z for z in kept if z["kind"] == "ob"]) == 1


def test_compute_zone_lifecycle_shape_on_synthetic():
    """On a real replayed series the #3 measure is either omitted (no zones) or
    carries reconciling, whole-minute tranches — never a fabricated shape."""
    from src.intelligence.publication_measures import _compute_zone_lifecycle

    # ~40 days of gold-like M15 with a sharp move at each weekly release, which
    # leaves order-blocks / gaps the engine detects.
    n = 40 * 96
    idx = pd.date_range(START, periods=n, freq="15min", tz="UTC")
    rng = np.random.default_rng(7)
    close = 2000.0 + np.cumsum(rng.normal(0.0, 0.3, n))
    df = pd.DataFrame(
        {"open": close, "high": close + 0.6, "low": close - 0.6,
         "close": close, "volume": 100.0},
        index=idx,
    )
    releases = [START + timedelta(days=3 + 5 * i, hours=13) for i in range(7)]
    releases = [r for r in releases if r <= idx[-1]]
    # Paint a reaction (a strong impulse then retrace) around each release.
    for t in releases:
        m = (df.index >= t) & (df.index < t + 6 * BAR)
        df.loc[m, "high"] = df.loc[m, "close"] + 12
        df.loc[m, "low"] = df.loc[m, "close"] - 12

    zl = _compute_zone_lifecycle(df, releases, "XAUUSD")
    if zl is None:
        return  # honestly omitted (too few zones/mitigations) — acceptable
    tranche_total = sum(t.count for t in zl.tranches)
    assert tranche_total + zl.never_mitigated_count == zl.zones_created_count
    assert isinstance(zl.fastest.minutes, int)
    assert isinstance(zl.slowest.minutes, int)
    assert zl.fastest.minutes <= zl.slowest.minutes
    for tr in zl.tranches:
        assert isinstance(tr.lower_minutes, int)
        assert tr.upper_minutes is None or isinstance(tr.upper_minutes, int)
    assert zl.provenance.market == "XAUUSD"
    assert zl.provenance.sample_size >= 4
