"""LB-1 — lookback depth config (DURATION per instrument×timeframe, M1 gate).

Verifies that depths are driven by the config file (never a global constant),
that the trading-hours-aware DURATION→bars conversion lands near the mission's
reference horizons, that the M1 gate is off by default, and that a malformed
config fails loud rather than silently defaulting.
"""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from src.intelligence import lookback_config as lb


@pytest.fixture(autouse=True)
def _reset():
    lb.reset_cache()
    yield
    lb.reset_cache()


@pytest.fixture
def _default_env(monkeypatch):
    """Ensure M1 gate is off and the real repo config is used."""
    monkeypatch.delenv("LB1_ENABLE_M1", raising=False)
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    lb.reset_cache()


# --------------------------------------------------------------------------- #
# Duration grammar
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tok,days",
    [("1d", 1.0), ("1w", 7.0), ("1mo", 30.4375), ("6mo", 182.625), ("2y", 730.5), ("5y", 1826.25)],
)
def test_parse_duration(tok, days):
    assert lb.parse_duration(tok) == timedelta(days=days)


@pytest.mark.parametrize("bad", ["", "1", "1x", "mo", "-1d", "1 month", "abc"])
def test_parse_duration_rejects_garbage(bad):
    with pytest.raises(ValueError):
        lb.parse_duration(bad)


# --------------------------------------------------------------------------- #
# DURATION → bars lands near the mission reference horizons
# --------------------------------------------------------------------------- #
_TF_MIN = {"M1": 1, "M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


# Combos intentionally deeper than one free-plan request (≤5000): they are filled
# by the PAGINATED deep_backfill_combo (NW-7d), not a single call. XAUUSD M15 was
# deepened to 24mo so the publication measures have months of intraday history.
_PAGINATED_DEEP_COMBOS = {("XAUUSD", "M15")}


@pytest.mark.parametrize("tf", ["M1", "M5", "M15", "H1", "H4", "D1"])
def test_target_bars_cover_the_duration_within_one_request(_default_env, tf):
    """The bar count must SPAN at least the configured calendar duration (so the
    stated depth is never silently under-delivered). Every combo fits a single
    free-plan request (≤ 5000) — EXCEPT the deep paginated combos, which are
    intentionally larger and filled page-by-page by deep_backfill_combo."""
    depth = lb.depth_for("XAUUSD", tf)
    covered_minutes = depth.target_bars * _TF_MIN[tf]
    duration_minutes = depth.duration.total_seconds() / 60.0
    assert covered_minutes >= duration_minutes, f"{tf}: {covered_minutes} < {duration_minutes}"
    if ("XAUUSD", tf) in _PAGINATED_DEEP_COMBOS:
        assert depth.target_bars > 5000, f"{tf}: expected a deep paginated combo"
    else:
        assert depth.target_bars <= 5000, f"{tf}: {depth.target_bars} exceeds a single request"


def test_depth_is_config_driven_not_constant(monkeypatch, tmp_path):
    """Changing the config file changes the depth — no hardcoded fallback wins."""
    cfg = {
        "default": {"H1": "6mo"},
        "instruments": {"XAUUSD": {"H1": "1y"}, "EURUSD": {}},
    }
    p = tmp_path / "depths.json"
    p.write_text(json.dumps(cfg), encoding="utf-8")
    monkeypatch.setenv("SENTINEL_LOOKBACK_DEPTHS_PATH", str(p))
    lb.reset_cache()
    # XAUUSD overrides to 1y, EURUSD falls back to the 6mo default → deeper for XAU.
    assert lb.target_bars("XAUUSD", "H1") > lb.target_bars("EURUSD", "H1")
    assert lb.depth_for("XAUUSD", "H1").duration_str == "1y"
    assert lb.depth_for("EURUSD", "H1").duration_str == "6mo"


# --------------------------------------------------------------------------- #
# Perimeter + M1 gate
# --------------------------------------------------------------------------- #
def test_supported_perimeter(_default_env):
    assert set(lb.supported_instruments()) == {"XAUUSD", "EURUSD"}
    assert lb.supported_timeframes() == ("M1", "M5", "M15", "H1", "H4", "D1")


def test_m1_off_by_default(_default_env):
    assert lb.is_m1_enabled() is False
    assert "M1" not in lb.enabled_timeframes()
    assert lb.enabled_timeframes() == ("M5", "M15", "H1", "H4", "D1")
    combos = lb.enabled_combos()
    assert ("XAUUSD", "M1") not in combos
    assert len(combos) == 10  # 2 instruments × 5 enabled timeframes


def test_m1_gate_enables(monkeypatch):
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    monkeypatch.setenv("LB1_ENABLE_M1", "1")
    lb.reset_cache()
    assert lb.is_m1_enabled() is True
    assert lb.enabled_timeframes()[0] == "M1"
    combos = lb.enabled_combos()
    assert ("XAUUSD", "M1") in combos and ("EURUSD", "M1") in combos
    assert len(combos) == 12  # 2 × 6


def test_live_warm_excludes_m5_by_default(_default_env):
    """M5 is too costly to poll natively on the free plan → not live-warmed."""
    warm = lb.live_warm_combos()
    assert all(tf != "M5" for (_i, tf) in warm)
    assert len(warm) == 8  # 2 instruments × {M15,H1,H4,D1}
    assert ("XAUUSD", "M15") in warm and ("EURUSD", "D1") in warm


def test_live_warm_includes_m5_when_flagged(monkeypatch):
    monkeypatch.delenv("SENTINEL_LOOKBACK_DEPTHS_PATH", raising=False)
    monkeypatch.delenv("LB1_ENABLE_M1", raising=False)
    monkeypatch.setenv("LB1_WARM_M5", "1")
    lb.reset_cache()
    warm = lb.live_warm_combos()
    assert ("XAUUSD", "M5") in warm and len(warm) == 10


def test_live_warm_instruments_allowlist_shrinks_perimeter(_default_env, monkeypatch):
    """LIVE_WARM_INSTRUMENTS restricts warming to the listed markets (others stay
    available on demand) — the config lever to cut Twelve Data REST usage."""
    monkeypatch.setenv("LIVE_WARM_INSTRUMENTS", "xauusd")  # case-insensitive
    warm = lb.live_warm_combos()
    assert warm == (("XAUUSD", "M15"), ("XAUUSD", "H1"), ("XAUUSD", "H4"), ("XAUUSD", "D1"))
    assert all(inst == "XAUUSD" for (inst, _tf) in warm)


def test_live_warm_timeframes_allowlist_shrinks_perimeter(_default_env, monkeypatch):
    monkeypatch.setenv("LIVE_WARM_INSTRUMENTS", "XAUUSD")
    monkeypatch.setenv("LIVE_WARM_TIMEFRAMES", "M15,H1")
    warm = lb.live_warm_combos()
    assert warm == (("XAUUSD", "M15"), ("XAUUSD", "H1"))


def test_enabled_combos_are_instrument_major_and_ordered(_default_env):
    combos = lb.enabled_combos()
    # All XAUUSD combos precede all EURUSD combos; timeframe order preserved.
    xau = [tf for (inst, tf) in combos if inst == "XAUUSD"]
    assert xau == ["M5", "M15", "H1", "H4", "D1"]
    assert combos[0][0] == "XAUUSD" and combos[-1][0] == "EURUSD"


# --------------------------------------------------------------------------- #
# Fail loud on malformed config
# --------------------------------------------------------------------------- #
def test_malformed_depth_raises(monkeypatch, tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"default": {"H1": "banana"}}), encoding="utf-8")
    monkeypatch.setenv("SENTINEL_LOOKBACK_DEPTHS_PATH", str(p))
    lb.reset_cache()
    # The perimeter now comes from the registry; depth-config validation fires
    # when a depth is actually read.
    with pytest.raises(ValueError):
        lb.depth_for("XAUUSD", "H1")


def test_empty_default_raises(monkeypatch, tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"default": {}}), encoding="utf-8")
    monkeypatch.setenv("SENTINEL_LOOKBACK_DEPTHS_PATH", str(p))
    lb.reset_cache()
    with pytest.raises(ValueError):
        lb.depth_for("XAUUSD", "H1")


# --------------------------------------------------------------------------- #
# MT-D1 — per-timeframe LIVE analysis window (detection + trend + journal)
# --------------------------------------------------------------------------- #
def test_analysis_window_widens_fast_units_floors_slow(_default_env):
    # Fast units get a wider recent horizon than the legacy fixed 500; slow units
    # stay floored at 500 (their 500-bar span is already deep — D1 ≈ 2 y).
    assert lb.analysis_window_bars("M15") > 500          # ~1 month, was ~5 days
    assert lb.analysis_window_bars("H1") > 500
    assert lb.analysis_window_bars("H4") == 500          # floor
    assert lb.analysis_window_bars("D1") == 500          # floor
    assert lb.analysis_window_bars("W1") == 500          # floor


def test_analysis_window_bounded_to_single_request(_default_env):
    # Never exceed the ceiling (bounded detection cost + one provider request).
    for tf in ("M5", "M15", "H1", "H4", "D1", "W1"):
        assert 500 <= lb.analysis_window_bars(tf) <= 3000


def test_analysis_window_env_override(monkeypatch):
    # A larger target span widens fast units further; slow units still floor.
    monkeypatch.setenv("MARKET_READING_WINDOW_DAYS", "60")
    assert lb.analysis_window_bars("M15") > lb.analysis_window_bars("H1")
    assert lb.analysis_window_bars("D1") == 500
    monkeypatch.delenv("MARKET_READING_WINDOW_DAYS", raising=False)


def test_analysis_window_unknown_tf_raises(_default_env):
    with pytest.raises(ValueError):
        lb.analysis_window_bars("ZZ9")
