"""Tests for RegimeFilter — empirical filter that drops PF<1 buckets."""
from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.intelligence.regime_filter import RegimeFilter


@pytest.fixture
def asian_ts() -> str:
    return "2024-06-03 03:15:00"


@pytest.fixture
def london_ts() -> str:
    return "2024-06-03 09:15:00"


@pytest.fixture
def ny_ts() -> str:
    return "2024-06-03 15:30:00"


@pytest.fixture
def offhours_ts() -> str:
    return "2024-06-03 22:00:00"


def _atr_series(values, n_extra: int = 250) -> pd.Series:
    """Build a long ATR series ending in `values` so rolling pctl is defined."""
    base = list(np.linspace(1.0, 2.0, n_extra)) + list(values)
    return pd.Series(base)


# ---------- Session gate ----------

def test_skip_ny_drops_signal_in_ny_window(ny_ts):
    """Legacy strict mode: ny_mode='all' drops every NY signal."""
    f = RegimeFilter(ny_mode="all", vol_pctl_max=None)
    decision = f.evaluate(ny_ts, atr_series=None)
    assert decision.allowed is False
    assert "NY" in decision.reason


def test_skip_ny_allows_london(london_ts):
    f = RegimeFilter(skip_ny=True, vol_pctl_max=None)
    assert f.evaluate(london_ts, atr_series=None).allowed is True


def test_skip_ny_allows_asian(asian_ts):
    f = RegimeFilter(skip_ny=True, vol_pctl_max=None)
    assert f.evaluate(asian_ts, atr_series=None).allowed is True


def test_skip_ny_allows_offhours(offhours_ts):
    f = RegimeFilter(skip_ny=True, vol_pctl_max=None)
    assert f.evaluate(offhours_ts, atr_series=None).allowed is True


def test_skip_ny_disabled_allows_ny(ny_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=None)
    assert f.evaluate(ny_ts, atr_series=None).allowed is True


def test_ny_boundaries():
    """NY window is [13:00, 21:00) — boundary semantics matter (strict mode)."""
    f = RegimeFilter(ny_mode="all", vol_pctl_max=None)
    assert f.evaluate("2024-06-03 12:59:00", None).allowed is True   # just before
    assert f.evaluate("2024-06-03 13:00:00", None).allowed is False  # at open
    assert f.evaluate("2024-06-03 20:59:59", None).allowed is False  # just before close
    assert f.evaluate("2024-06-03 21:00:00", None).allowed is True   # at close (excluded)


# ---------- Vol percentile gate ----------

def test_vol_gate_drops_top_quartile(london_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=0.75)
    # Make latest ATR the maximum → percentile = 1.0
    atr = _atr_series([1.5, 1.8, 2.0, 5.0])
    decision = f.evaluate(london_ts, atr)
    assert decision.allowed is False
    assert "high vol" in decision.reason


def test_vol_gate_allows_low_vol(london_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=0.75)
    # Latest ATR is the minimum → percentile near 0
    atr = _atr_series([2.0, 1.5, 0.5, 0.1])
    assert f.evaluate(london_ts, atr).allowed is True


def test_vol_gate_abstains_when_history_too_short(london_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=0.75, vol_min_periods=200)
    short = pd.Series([1.0, 2.0, 5.0])  # below min_periods
    # Should allow (gate abstains, not blocks)
    assert f.evaluate(london_ts, short).allowed is True


def test_vol_gate_abstains_when_atr_none(london_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=0.75)
    assert f.evaluate(london_ts, None).allowed is True


def test_vol_gate_disabled_allows_high_vol(london_ts):
    f = RegimeFilter(skip_ny=False, vol_pctl_max=None)
    atr = _atr_series([1.0] * 10 + [99.0])
    assert f.evaluate(london_ts, atr).allowed is True


# ---------- Combined ----------

def test_ny_drops_first_even_if_vol_low(ny_ts):
    """In ny_mode='all', NY gate fires regardless of vol."""
    f = RegimeFilter(ny_mode="all", vol_pctl_max=0.75)
    atr = _atr_series([2.0] * 10 + [0.1])
    decision = f.evaluate(ny_ts, atr)
    assert decision.allowed is False
    assert "NY" in decision.reason


def test_london_high_vol_dropped_by_vol_gate(london_ts):
    f = RegimeFilter(skip_ny=True, vol_pctl_max=0.75)
    atr = _atr_series([1.0] * 10 + [99.0])
    decision = f.evaluate(london_ts, atr)
    assert decision.allowed is False
    assert "high vol" in decision.reason


def test_london_low_vol_allowed(london_ts):
    f = RegimeFilter(skip_ny=True, vol_pctl_max=0.75)
    atr = _atr_series([5.0] * 10 + [0.1])
    assert f.evaluate(london_ts, atr).allowed is True


# ---------- Stats ----------

def test_stats_track_drops(london_ts, ny_ts):
    f = RegimeFilter(ny_mode="all", vol_pctl_max=0.75)
    f.evaluate(london_ts, None)               # allowed (no atr → vol gate abstains)
    f.evaluate(ny_ts, None)                   # dropped (NY all-mode)
    f.evaluate(london_ts, _atr_series([1.0]*10 + [99]))  # dropped (vol)
    s = f.stats()
    assert s["allowed"] == 1
    assert s["dropped_ny"] == 1
    assert s["dropped_vol"] == 1
    assert 0 < s["drop_rate"] < 1


def test_bad_timestamp_does_not_crash():
    f = RegimeFilter(skip_ny=True, vol_pctl_max=None)
    decision = f.evaluate("not-a-timestamp", None)
    assert decision.allowed is True


# ---------- Env factory ----------

def test_from_env_defaults():
    with patch.dict(os.environ, {}, clear=False):
        for k in ("REGIME_FILTER_SKIP_NY", "REGIME_FILTER_VOL_PCTL_MAX", "REGIME_FILTER_NY_MODE"):
            os.environ.pop(k, None)
        f = RegimeFilter.from_env()
        assert f.skip_ny is True
        assert f.ny_mode == "high_vol"  # surgical default
        assert f.vol_pctl_max == RegimeFilter.DEFAULT_VOL_PCTL_MAX


def test_from_env_overrides():
    with patch.dict(os.environ, {
        "REGIME_FILTER_SKIP_NY": "0",
        "REGIME_FILTER_VOL_PCTL_MAX": "0.9",
        "REGIME_FILTER_NY_MODE": "all",
    }):
        f = RegimeFilter.from_env()
        assert f.skip_ny is False
        # skip_ny=False forces ny_mode="off" (back-compat)
        assert f.ny_mode == "off"
        assert f.vol_pctl_max == 0.9


# ---------- Surgical NY mode (ny_mode="high_vol") ----------

def test_ny_mode_high_vol_drops_ny_when_high_vol(ny_ts):
    """In surgical mode, NY signals are dropped only when ATR_PCTL > vol_pctl_max."""
    f = RegimeFilter(ny_mode="high_vol", vol_pctl_max=0.75)
    high_vol_atr = _atr_series([1.0] * 10 + [99.0])
    decision = f.evaluate(ny_ts, high_vol_atr)
    assert decision.allowed is False
    assert "NY" in decision.reason and "high vol" in decision.reason


def test_ny_mode_high_vol_allows_ny_when_low_vol(ny_ts):
    """In surgical mode, NY signals are allowed when vol is in the safe range."""
    f = RegimeFilter(ny_mode="high_vol", vol_pctl_max=0.75)
    low_vol_atr = _atr_series([5.0] * 10 + [0.5])
    assert f.evaluate(ny_ts, low_vol_atr).allowed is True


def test_ny_mode_high_vol_abstains_when_atr_unknown(ny_ts):
    """In surgical mode with unknown ATR, the NY rule abstains (allows)."""
    f = RegimeFilter(ny_mode="high_vol", vol_pctl_max=0.75)
    assert f.evaluate(ny_ts, atr_series=None).allowed is True


def test_ny_mode_high_vol_still_drops_non_ny_when_high_vol(london_ts):
    """In surgical mode, the standalone vol gate still applies outside NY."""
    f = RegimeFilter(ny_mode="high_vol", vol_pctl_max=0.75)
    high_vol_atr = _atr_series([1.0] * 10 + [99.0])
    decision = f.evaluate(london_ts, high_vol_atr)
    assert decision.allowed is False
    assert "high vol" in decision.reason and "NY" not in decision.reason


def test_ny_mode_off_allows_everything(ny_ts):
    f = RegimeFilter(ny_mode="off", vol_pctl_max=None)
    assert f.evaluate(ny_ts, None).allowed is True


def test_ny_mode_invalid_raises():
    with pytest.raises(ValueError):
        RegimeFilter(ny_mode="bogus")


def test_skip_ny_false_overrides_ny_mode():
    """Back-compat: skip_ny=False must turn the NY rule off entirely."""
    f = RegimeFilter(skip_ny=False, ny_mode="all")
    assert f.ny_mode == "off"
