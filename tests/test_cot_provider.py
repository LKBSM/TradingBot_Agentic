"""Tests for src/agents/data/cot_provider.py (DATA-1.2).

Three required tests per DoD (`PLAN_12_MOIS.md` Partie II.2 Agent 1):
  1. fetched DataFrame has the right schema and types
  2. cot_at(t) at Friday 14:30 ET returns previous week's report;
     cot_at(t) at Friday 16:00 ET returns the current week's report (after
     publication at 15:30 ET) — the headline DoD fixture
  3. cot_at respects vintage_date <= t on 100 random timestamps (seed=42)

Plus a `@pytest.mark.live` test that hits the real CFTC URL when network is
available; auto-skipped offline.
"""

from __future__ import annotations

import os
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from src.agents.data.cot_provider import (
    GOLD_CFTC_CODE,
    OUTPUT_COLUMNS,
    CotProvider,
    _empty_frame,
    _vintage_date_for_report,
)


ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Synthetic raw frame factory (mirrors CFTC text format columns)
# ---------------------------------------------------------------------------


def _make_synthetic_raw(num_weeks: int = 120) -> pd.DataFrame:
    """Generate `num_weeks` of synthetic Tuesday COT rows for Gold 088691.

    Tuesdays start 2023-01-03 and step by 7 days. Default 120 weeks covers
    January 2023 through April 2025 — enough margin for any DoD test fixture.
    mm_long oscillates with a small trend so 52-week z-scores are non-trivial.
    """
    start = pd.Timestamp("2023-01-03")  # Tuesday
    rows = []
    for i in range(num_weeks):
        report_dt = start + pd.Timedelta(days=7 * i)
        rows.append(
            {
                "Report_Date_as_YYYY-MM-DD": report_dt.strftime("%Y-%m-%d"),
                "CFTC_Contract_Market_Code": GOLD_CFTC_CODE,
                "Open_Interest_All": 500_000 + i * 100,
                "Prod_Merc_Positions_Long_All": 100_000 + i * 50,
                "Prod_Merc_Positions_Short_All": 200_000 + i * 70,
                "M_Money_Positions_Long_All": int(
                    150_000 + i * 1_000 + 5_000 * np.sin(i / 5)
                ),
                "M_Money_Positions_Short_All": 50_000 + i * 200,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1 — schema
# ---------------------------------------------------------------------------


def test_compute_features_schema_and_types():
    """compute_features must return columns OUTPUT_COLUMNS with the right
    dtypes (UTC-aware timestamps, Int64 counts, float ratios) and the
    z52 column NaN before 26 weeks of data accumulate."""
    raw = _make_synthetic_raw(num_weeks=120)
    df = CotProvider.compute_features(raw)

    assert df.columns.tolist() == OUTPUT_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(df["report_date"])
    assert df["report_date"].dt.tz is not None
    assert pd.api.types.is_datetime64_any_dtype(df["vintage_date"])
    assert df["vintage_date"].dt.tz is not None

    for col in [
        "mm_long",
        "mm_short",
        "mm_net",
        "open_interest",
        "producer_long",
        "producer_short",
        "producer_net",
    ]:
        # Int64 is the nullable extension dtype; "Int64" string check.
        assert str(df[col].dtype) == "Int64", f"{col} dtype is {df[col].dtype}"

    assert df["mm_net_pct"].dtype == np.float64
    assert df["mm_net_pct_z52"].dtype == np.float64

    # Z52 should be NaN before min_periods=26, populated after.
    early = df.iloc[:25]["mm_net_pct_z52"]
    later = df.iloc[26:]["mm_net_pct_z52"]
    assert early.isna().all()
    assert not later.isna().all()

    # vintage_date is always strictly after report_date by 3+ days at 15:30 ET.
    assert (df["vintage_date"] > df["report_date"]).all()
    deltas = df["vintage_date"] - df["report_date"]
    # +3 days + (UTC equivalent of 15:30 ET) → between 3d 19h and 3d 20h.
    assert (deltas >= pd.Timedelta(days=3, hours=19)).all()
    assert (deltas <= pd.Timedelta(days=3, hours=21)).all()

    # Empty input → empty frame with correct schema.
    empty = CotProvider.compute_features(pd.DataFrame())
    assert empty.columns.tolist() == OUTPUT_COLUMNS
    assert len(empty) == 0


# ---------------------------------------------------------------------------
# Test 2 — DoD headline: Fri 14:30 ET vs Fri 16:00 ET
# ---------------------------------------------------------------------------


def test_cot_at_publication_lag_friday_1430_vs_1600():
    """**The DoD fixture from `PLAN_12_MOIS.md`**.

    At 14:30 ET Friday (before publication), `cot_at` must return the
    *previous* week's COT (this Tuesday's report not yet released).
    At 16:00 ET Friday (after the 15:30 release), `cot_at` must return
    the current week's COT.
    """
    raw = _make_synthetic_raw(num_weeks=120)
    df = CotProvider.compute_features(raw)

    # Pick Tuesday 2024-03-26 ; publication Friday 2024-03-29 15:30 ET.
    fri = "2024-03-29"
    fri_1430_et = pd.Timestamp(f"{fri} 14:30", tz=ET)
    fri_1600_et = pd.Timestamp(f"{fri} 16:00", tz=ET)

    pre = CotProvider.cot_at(df, fri_1430_et)
    post = CotProvider.cot_at(df, fri_1600_et)

    assert pre is not None and post is not None
    assert pre["report_date"] == pd.Timestamp("2024-03-19", tz="UTC")
    assert post["report_date"] == pd.Timestamp("2024-03-26", tz="UTC")

    # Exactly one week apart by construction.
    assert (post["report_date"] - pre["report_date"]) == pd.Timedelta(days=7)


# ---------------------------------------------------------------------------
# Test 3 — 100 random timestamps respect the vintage invariant
# ---------------------------------------------------------------------------


def test_cot_at_respects_vintage_invariant_on_100_random_dates():
    """For 100 random UTC timestamps, the row returned by `cot_at` must
    satisfy `vintage_date <= t`, and there must be no later (more recent)
    row that also satisfies it. Any violation is a look-ahead leak.
    """
    raw = _make_synthetic_raw(num_weeks=120)
    df = CotProvider.compute_features(raw)

    rng = np.random.default_rng(seed=42)
    base = pd.Timestamp("2023-01-01", tz="UTC")
    span_min = 365 * 2 * 24 * 60  # 2 years in minutes
    deltas = rng.integers(0, span_min, size=100)
    timestamps = [base + pd.Timedelta(minutes=int(d)) for d in deltas]

    failures = []
    for t in timestamps:
        actual = CotProvider.cot_at(df, t)
        eligible = df[df["vintage_date"] <= t]

        if eligible.empty:
            if actual is not None:
                failures.append(("no rows expected", t, actual))
            continue

        expected = eligible.sort_values("vintage_date").iloc[-1]
        if actual is None:
            failures.append(("got None expected row", t, expected))
            continue
        if actual["report_date"] != expected["report_date"]:
            failures.append(
                ("wrong row", t, expected["report_date"], actual["report_date"])
            )
        if actual["vintage_date"] > t:
            failures.append(("vintage > t (leak!)", t, actual["vintage_date"]))

    assert not failures, (
        f"cot_at violated vintage invariant on {len(failures)}/100 timestamps; "
        f"first 3: {failures[:3]}"
    )


# ---------------------------------------------------------------------------
# Bonus — vintage helper at boundary times
# ---------------------------------------------------------------------------


def test_vintage_date_helper_handles_dst_correctly():
    """Sanity: a March Tuesday (post-US-DST start) maps to Friday 15:30 EDT
    = 19:30 UTC, while a January Tuesday maps to Friday 15:30 EST = 20:30 UTC."""
    summer_tue = pd.Timestamp("2024-07-02", tz="UTC")  # EDT (UTC-4)
    winter_tue = pd.Timestamp("2024-01-02", tz="UTC")  # EST (UTC-5)

    summer_v = _vintage_date_for_report(summer_tue)
    winter_v = _vintage_date_for_report(winter_tue)

    # Summer: Fri 15:30 EDT (UTC-4) → 19:30 UTC
    assert summer_v == pd.Timestamp("2024-07-05 19:30:00+00:00")
    # Winter: Fri 15:30 EST (UTC-5) → 20:30 UTC
    assert winter_v == pd.Timestamp("2024-01-05 20:30:00+00:00")


# ---------------------------------------------------------------------------
# Live smoke test
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("SKIP_LIVE_NETWORK_TESTS") == "1",
    reason="SKIP_LIVE_NETWORK_TESTS=1 set; live CFTC test skipped",
)
def test_live_fetch_2024_smoke(tmp_path):
    """End-to-end live test: download CFTC zip for 2024, parse, compute,
    verify ~52 weekly Gold rows. Skipped via env var or marker selection.
    """
    p = CotProvider(cache_dir=tmp_path)
    raw = p.fetch_year(2024)
    df = CotProvider.compute_features(raw)
    assert 50 <= len(df) <= 54, f"unexpected number of weekly rows: {len(df)}"
    assert df.columns.tolist() == OUTPUT_COLUMNS
    assert (df["vintage_date"] > df["report_date"]).all()
    assert (df["mm_long"] > 0).all()
    assert (df["open_interest"] > 0).all()
