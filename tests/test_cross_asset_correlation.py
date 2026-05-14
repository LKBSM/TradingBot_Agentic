"""Tests for the REGIME-2B.4 cross-asset correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.intelligence.cross_asset_correlation import (
    DEFAULT_WINDOW,
    corr_table,
    heatmap_payload,
    latest_summary,
)


@pytest.fixture
def anticorrelated_prices():
    """XAU + DXY moving inversely; SPX uncorrelated."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    common_factor = rng.normal(0, 0.01, len(idx))
    xau = pd.Series(np.cumsum(common_factor) + 100, index=idx)
    dxy = pd.Series(np.cumsum(-common_factor) + 100, index=idx)  # anti
    spx = pd.Series(np.cumsum(rng.normal(0, 0.01, len(idx))) + 100, index=idx)
    return {
        "XAU": pd.DataFrame({"close": xau}),
        "DXY": pd.DataFrame({"close": dxy}),
        "SPX": pd.DataFrame({"close": spx}),
    }


def test_corr_table_dxy_is_negative(anticorrelated_prices):
    t = corr_table(anticorrelated_prices, window=30)
    final_dxy = t["DXY"].dropna().iloc[-1]
    final_spx = t["SPX"].dropna().iloc[-1]
    assert final_dxy < -0.5
    assert abs(final_spx) < 0.5  # uncorrelated → near zero


def test_latest_summary_returns_one_per_non_ref_asset(anticorrelated_prices):
    summary = latest_summary(anticorrelated_prices)
    assets = sorted(s.asset for s in summary)
    assert assets == ["DXY", "SPX"]


def test_missing_reference_raises(anticorrelated_prices):
    del anticorrelated_prices["XAU"]
    with pytest.raises(ValueError, match="reference"):
        corr_table(anticorrelated_prices)


def test_invalid_window_raises():
    with pytest.raises(ValueError):
        corr_table({"XAU": pd.DataFrame()}, window=1)


def test_heatmap_payload_shape(anticorrelated_prices):
    payload = heatmap_payload(anticorrelated_prices, window=30)
    assert payload["reference"] == "XAU"
    assert set(payload["assets"]) == {"DXY", "SPX"}
    # matrix is len(dates) × len(assets)
    assert len(payload["matrix"]) == len(payload["dates"])
    if payload["matrix"]:
        assert len(payload["matrix"][0]) == len(payload["assets"])


def test_summary_to_dict_carries_vs_field(anticorrelated_prices):
    summary = latest_summary(anticorrelated_prices)
    d = summary[0].to_dict()
    assert d["vs"] == "XAU"
    assert "last_value" in d


def test_no_intersection_returns_empty():
    """Disjoint timestamps → empty correlation table, no crash."""
    a = pd.DataFrame(
        {"close": [1, 2]},
        index=pd.DatetimeIndex(pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True)),
    )
    b = pd.DataFrame(
        {"close": [3, 4]},
        index=pd.DatetimeIndex(pd.to_datetime(["2025-01-01", "2025-01-02"], utc=True)),
    )
    t = corr_table({"XAU": a, "DXY": b})
    assert t.empty


def test_only_reference_returns_empty_table(anticorrelated_prices):
    only_xau = {"XAU": anticorrelated_prices["XAU"]}
    t = corr_table(only_xau)
    # No other assets → DataFrame with reference's index but no columns.
    assert list(t.columns) == []
