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
@pytest.mark.parametrize(
    "tf,expected_ref",
    [("M1", 1440), ("M5", 1440), ("M15", 2000), ("H1", 3100), ("H4", 3100), ("D1", 1300)],
)
def test_target_bars_near_reference(_default_env, tf, expected_ref):
    bars = lb.target_bars("XAUUSD", tf)
    # Within 15% of the mission's dimensioning figure — the reference exists only
    # to prove the load stays bounded, not as a hard target.
    assert expected_ref * 0.85 <= bars <= expected_ref * 1.20, f"{tf}: {bars} vs ~{expected_ref}"


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
    with pytest.raises(ValueError):
        lb.supported_timeframes()


def test_empty_default_raises(monkeypatch, tmp_path):
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"default": {}}), encoding="utf-8")
    monkeypatch.setenv("SENTINEL_LOOKBACK_DEPTHS_PATH", str(p))
    lb.reset_cache()
    with pytest.raises(ValueError):
        lb.supported_timeframes()
