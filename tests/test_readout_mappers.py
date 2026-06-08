"""Tests for readout_mappers.py — internal dataclasses → v2.1.0 readouts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.intelligence.readout_mappers import (
    map_breakdown_components,
    map_event_readout,
    map_historical_stats,
    map_regime_readout,
    map_structure_readout,
    map_uncertainty_context,
    map_volatility_readout,
)


# ---------------------------------------------------------------------------
# map_structure_readout
# ---------------------------------------------------------------------------


def test_structure_readout_both_inputs_none_returns_none():
    assert map_structure_readout(None, None) is None


def test_structure_readout_minimal_smc():
    """Just a BOS_SIGNAL — produces a non-empty readout with retest=None."""
    sig = SimpleNamespace(signal_type=SimpleNamespace(value="LONG"), atr=8.0)
    smc = {"BOS_SIGNAL": 1.0, "BOS_BREAK_LEVEL": 2391.50}
    r = map_structure_readout(sig, smc)
    assert r is not None
    assert r.bos_level == 2391.50
    assert r.choch_present is False
    assert r.fvg_zone is None


def test_structure_readout_full_bullish():
    sig = SimpleNamespace(signal_type=SimpleNamespace(value="LONG"))
    smc = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_BREAK_LEVEL": 2391.50,
        "CHOCH_SIGNAL": 0.0,
        "FVG_SIGNAL": 1.0,
        "FVG_SIZE_NORM": 0.42,
        "FVG_LOW": 2378.0,
        "FVG_HIGH": 2381.0,
        "BULLISH_OB_LOW": 2375.0,
        "BULLISH_OB_HIGH": 2378.0,
        "OB_STRENGTH_NORM": 0.73,
        "BOS_RETEST_ARMED": 1.0,
    }
    r = map_structure_readout(sig, smc)
    assert r.bos_level == 2391.50
    assert r.fvg_zone == [2378.0, 2381.0]
    assert r.fvg_size_atr == pytest.approx(0.42)
    assert r.ob_zone == [2375.0, 2378.0]
    assert r.ob_strength == pytest.approx(0.73)
    assert r.retest_state == "armed"
    # Bullish ⇒ invalidation = FVG floor
    assert r.structural_invalidation == 2378.0


def test_structure_readout_bearish_ob_polarity():
    sig = SimpleNamespace(signal_type=SimpleNamespace(value="SHORT"))
    smc = {
        "BOS_SIGNAL": -1.0,
        "BEARISH_OB_LOW": 2401.0,
        "BEARISH_OB_HIGH": 2404.0,
        "OB_STRENGTH_NORM": 0.65,
        "FVG_SIGNAL": -1.0,
        "FVG_LOW": 2399.0, "FVG_HIGH": 2402.0,
    }
    r = map_structure_readout(sig, smc)
    assert r.ob_zone == [2401.0, 2404.0]
    # Bearish ⇒ invalidation = FVG ceiling
    assert r.structural_invalidation == 2402.0


def test_structure_readout_clamps_ob_strength():
    sig = SimpleNamespace(signal_type=SimpleNamespace(value="LONG"))
    smc = {"BOS_SIGNAL": 1.0, "OB_STRENGTH_NORM": 1.5}  # >1.0 input
    r = map_structure_readout(sig, smc)
    assert 0.0 <= r.ob_strength <= 1.0


def test_structure_readout_handles_nan_inputs():
    sig = SimpleNamespace(signal_type=SimpleNamespace(value="LONG"))
    smc = {
        "BOS_SIGNAL": 1.0,
        "BOS_BREAK_LEVEL": float("nan"),
        "FVG_SIGNAL": 1.0,
        "FVG_SIZE_NORM": float("inf"),
    }
    r = map_structure_readout(sig, smc)
    assert r.bos_level is None
    assert r.fvg_size_atr is None


# ---------------------------------------------------------------------------
# map_regime_readout
# ---------------------------------------------------------------------------


def test_regime_readout_both_none():
    assert map_regime_readout(None, None) is None


def test_regime_readout_full():
    analysis = SimpleNamespace(
        regime=SimpleNamespace(value="uptrend"),
        confidence=0.71,
    )
    gate = SimpleNamespace(
        decision=SimpleNamespace(value="TRADE"),
        cp_prob=0.03,
        jump_ratio=0.12,
        expected_run_length=180.0,
    )
    r = map_regime_readout(analysis, gate, direction_hint="LONG")
    assert r.hmm_label == "trend_bullish"
    assert r.hmm_posterior == 0.71
    assert r.bocpd_changepoint_prob == 0.03
    assert r.jump_ratio == 0.12
    assert r.expected_run_length == 180.0
    assert r.regime_gate_decision == "TRADE"


def test_regime_readout_short_direction_flips_label():
    analysis = SimpleNamespace(
        regime=SimpleNamespace(value="downtrend"),
        confidence=0.66,
    )
    r = map_regime_readout(analysis, None, direction_hint="SHORT")
    assert r.hmm_label == "trend_bearish"


def test_regime_readout_clamps_probabilities():
    gate = SimpleNamespace(
        decision=SimpleNamespace(value="REDUCE"),
        cp_prob=1.5,  # over 1.0 ⇒ clamped
        jump_ratio=-0.1,  # under 0 ⇒ clamped
    )
    r = map_regime_readout(None, gate)
    assert r.bocpd_changepoint_prob == 1.0
    assert r.jump_ratio == 0.0


def test_regime_readout_rejects_bad_gate_decision():
    gate = SimpleNamespace(
        decision=SimpleNamespace(value="OPEN_LONG"),  # forbidden
        cp_prob=0.0, jump_ratio=0.0,
    )
    r = map_regime_readout(None, gate)
    # gate_decision should be None (invalid wording dropped, not raised)
    assert r.regime_gate_decision is None


# ---------------------------------------------------------------------------
# map_volatility_readout
# ---------------------------------------------------------------------------


def test_volatility_readout_none_in_none_out():
    assert map_volatility_readout(None) is None


def test_volatility_readout_full():
    forecast = SimpleNamespace(
        forecast_atr=8.7,
        naive_atr=7.9,
        confidence_lower=7.2,
        confidence_upper=10.4,
        regime_state="normal",
        is_fallback=False,
    )
    r = map_volatility_readout(forecast)
    assert r.regime == "normal"
    assert r.forecast_atr_pips == 8.7
    assert r.naive_atr_pips == 7.9
    assert r.forecast_vs_naive_pct == pytest.approx(10.13, abs=0.1)
    assert r.confidence_interval_pips == [7.2, 10.4]
    assert r.is_fallback is False


def test_volatility_readout_fallback():
    forecast = SimpleNamespace(
        forecast_atr=8.0, naive_atr=8.0,
        confidence_lower=None, confidence_upper=None,
        regime_state="normal", is_fallback=True,
    )
    r = map_volatility_readout(forecast)
    assert r.is_fallback is True
    assert r.confidence_interval_pips is None


def test_volatility_readout_invalid_regime_drops_field():
    forecast = SimpleNamespace(
        forecast_atr=8.0, naive_atr=8.0,
        confidence_lower=7.0, confidence_upper=9.0,
        regime_state="something_else",
        is_fallback=False,
    )
    r = map_volatility_readout(forecast)
    assert r.regime is None


def test_volatility_readout_inverted_ci_dropped():
    forecast = SimpleNamespace(
        forecast_atr=8.0, naive_atr=8.0,
        confidence_lower=10.0, confidence_upper=7.0,  # inverted
        regime_state="normal", is_fallback=False,
    )
    r = map_volatility_readout(forecast)
    # The mapper drops the inverted CI rather than raising
    assert r.confidence_interval_pips is None


# ---------------------------------------------------------------------------
# map_event_readout
# ---------------------------------------------------------------------------


def test_event_readout_both_none_returns_none():
    assert map_event_readout(None, None) is None


def test_event_readout_with_session_only():
    r = map_event_readout(None, session="new_york")
    assert r.session == "new_york"
    assert r.news_blackout_active is False
    assert r.next_event_label is None


def test_event_readout_blackout_active():
    assessment = SimpleNamespace(
        decision=SimpleNamespace(value="block"),
        sentiment_score=-0.2,
        sentiment_confidence=0.8,
        blocking_events=[SimpleNamespace(event_name="FOMC Minutes")],
        hours_to_next_high_impact=0.5,
    )
    r = map_event_readout(assessment, session="ny_overlap")
    assert r.news_blackout_active is True
    assert r.next_event_label == "FOMC Minutes"
    assert r.next_event_in_minutes == 30
    assert r.sentiment_score == pytest.approx(-0.2)
    assert r.session == "ny_overlap"


def test_event_readout_invalid_session_dropped():
    r = map_event_readout(None, session="european_overlap_extended")
    assert r.session is None


# ---------------------------------------------------------------------------
# map_breakdown_components
# ---------------------------------------------------------------------------


def test_breakdown_components_empty_input():
    assert map_breakdown_components(None) == []


def test_breakdown_components_full():
    sig = SimpleNamespace(components=[
        SimpleNamespace(name="BOS", weighted_score=13.5, weight=15.0, reasoning="BOS retest"),
        SimpleNamespace(name="FVG", weighted_score=12.8, weight=15.0, reasoning="FVG retest"),
        SimpleNamespace(name="OrderBlock", weighted_score=7.5, weight=10.0, reasoning="OB aligned"),
        SimpleNamespace(name="Regime", weighted_score=18.2, weight=25.0, reasoning="HMM trend_bullish"),
        SimpleNamespace(name="News", weighted_score=0.0, weight=20.0, reasoning="no blackout"),
        SimpleNamespace(name="Volume", weighted_score=7.0, weight=10.0, reasoning="1.4x avg"),
        SimpleNamespace(name="Momentum", weighted_score=2.4, weight=3.0, reasoning="RSI 42→58"),
        SimpleNamespace(name="RSI_Divergence", weighted_score=0.6, weight=2.0, reasoning="hidden bull"),
    ])
    rows = map_breakdown_components(sig)
    assert len(rows) == 8
    assert rows[0].name == "bos"  # name normalized to lowercase canonical
    assert rows[0].contribution == 13.5
    assert rows[0].weight_max == 15.0
    assert rows[0].score_pct == 90.0
    assert rows[2].name == "order_block"
    assert rows[7].name == "rsi_divergence"


def test_breakdown_components_redacted_weights():
    """B2C surface: expose_weights=False redacts weight_max."""
    sig = SimpleNamespace(components=[
        SimpleNamespace(name="BOS", weighted_score=13.5, weight=15.0, reasoning="x"),
    ])
    rows = map_breakdown_components(sig, expose_weights=False)
    assert rows[0].weight_max is None
    assert rows[0].score_pct is None  # because weight_max is None
    assert rows[0].contribution == 13.5  # contribution is always exposed


def test_breakdown_components_skips_missing_names():
    sig = SimpleNamespace(components=[
        SimpleNamespace(name=None, weighted_score=10.0, weight=15.0, reasoning="x"),
        SimpleNamespace(name="BOS", weighted_score=13.5, weight=15.0, reasoning="ok"),
    ])
    rows = map_breakdown_components(sig)
    assert len(rows) == 1
    assert rows[0].name == "bos"


# ---------------------------------------------------------------------------
# map_uncertainty_context
# ---------------------------------------------------------------------------


def test_uncertainty_context_from_calibrated_conviction():
    from src.intelligence.scoring.calibrated_conviction import CalibratedConviction
    from src.intelligence.conformal_wrapper import ConformalInterval

    cc = CalibratedConviction(
        p_win_raw=0.65,
        p_win_calibrated=0.72,
        conviction_0_100=72,
        interval=ConformalInterval(point=0.72, lower=0.54, upper=0.82, alpha=0.10, n_calibration=2000),
    )
    uc = map_uncertainty_context(cc)
    assert uc is not None
    assert uc.conformal_lower == pytest.approx(54.0)
    assert uc.conformal_upper == pytest.approx(82.0)
    assert uc.coverage_alpha == 0.10
    assert uc.n_calibration == 2000


def test_uncertainty_context_none_input():
    assert map_uncertainty_context(None) is None


# ---------------------------------------------------------------------------
# map_historical_stats
# ---------------------------------------------------------------------------


def test_historical_stats_all_none_returns_none():
    assert map_historical_stats() is None


def test_historical_stats_full():
    h = map_historical_stats(
        similar_setups_n=329,
        hit_rate=0.319,
        profit_factor=1.30,
        profit_factor_ci95=[1.12, 1.49],
        empirical_coverage=0.91,
        backtest_window="XAU M15 2019-2025 walk-forward",
    )
    assert h.similar_setups_n == 329
    assert h.profit_factor == 1.30
    assert h.profit_factor_ci95 == [1.12, 1.49]
    assert h.backtest_window.startswith("XAU")


def test_historical_stats_pf_ci_sorted_automatically():
    h = map_historical_stats(profit_factor=1.30, profit_factor_ci95=[1.49, 1.12])  # inverted
    assert h.profit_factor_ci95 == [1.12, 1.49]
