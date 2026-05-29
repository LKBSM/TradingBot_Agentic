"""Tests for InsightSignalV2 2.1.0 enrichment (Sprint 1).

Covers the 7 new descriptive sub-models added to expose buried alpha:
  - UncertaintyContext (conformal interval around conviction)
  - StructureReadout (BOS, FVG, OB, retest, invalidation zones)
  - RegimeReadout (HMM + BOCPD + jump ratio + gate decision)
  - VolatilityReadout (forecast + naïve + conformal CI)
  - EventReadout (news, calendar, session)
  - ComponentBreakdown (8-component decomposition)
  - HistoricalStats (similar-setup statistics)

Tests ensure:
  1. Each sub-model validates its invariants (zone ordering, bounds, enums).
  2. The top-level InsightSignalV2 accepts None for every new field (backward
     compat with 2.0.0 consumers that don't populate them).
  3. Round-trip JSON serialisation preserves all new sub-models.
  4. The new schema_version is exposed correctly (2.1.0).
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.api.insight_signal_v2 import (
    SCHEMA_VERSION,
    ComponentBreakdown,
    EventReadout,
    HistoricalStats,
    InsightSignalV2,
    RegimeReadout,
    SetupDirection,
    SignalLevels,
    StructureReadout,
    Timeframe,
    UncertaintyContext,
    VolatilityReadout,
    to_audit_row,
    to_b2b_dict,
)


# ---------------------------------------------------------------------------
# UncertaintyContext
# ---------------------------------------------------------------------------


def test_uncertainty_context_basic():
    u = UncertaintyContext(
        conformal_lower=54.0,
        conformal_upper=82.0,
        coverage_alpha=0.10,
        n_calibration=2000,
        empirical_coverage=0.91,
    )
    assert u.width() == pytest.approx(28.0)


def test_uncertainty_upper_must_be_ge_lower():
    with pytest.raises(ValueError, match="upper.*≥.*lower"):
        UncertaintyContext(
            conformal_lower=80.0,
            conformal_upper=70.0,  # inverted
            coverage_alpha=0.10,
            n_calibration=2000,
        )


def test_uncertainty_alpha_must_be_in_open_unit_interval():
    with pytest.raises(ValueError):
        UncertaintyContext(
            conformal_lower=50.0, conformal_upper=70.0,
            coverage_alpha=0.0,  # excluded boundary
            n_calibration=100,
        )
    with pytest.raises(ValueError):
        UncertaintyContext(
            conformal_lower=50.0, conformal_upper=70.0,
            coverage_alpha=0.5,  # at boundary not allowed
            n_calibration=100,
        )


def test_uncertainty_empirical_coverage_optional():
    """ACI hasn't accumulated enough observations yet ⇒ empirical_coverage=None."""
    u = UncertaintyContext(
        conformal_lower=40.0, conformal_upper=80.0,
        coverage_alpha=0.10, n_calibration=30,
    )
    assert u.empirical_coverage is None


# ---------------------------------------------------------------------------
# StructureReadout
# ---------------------------------------------------------------------------


def test_structure_readout_full_bullish():
    s = StructureReadout(
        bos_level=2391.50,
        bos_event_age_bars=2,
        choch_present=False,
        fvg_zone=[2378.00, 2381.00],
        fvg_size_atr=0.42,
        ob_zone=[2375.00, 2378.00],
        ob_strength=0.73,
        retest_state="armed",
        structural_invalidation=2374.50,
        liquidity_zone_upper=[2398.50, 2401.20],
    )
    assert s.fvg_zone == [2378.0, 2381.0]
    assert s.retest_state == "armed"


def test_structure_readout_zone_must_be_2_elements():
    with pytest.raises(ValueError, match="must have exactly 2 elements"):
        StructureReadout(fvg_zone=[2378.0, 2381.0, 2400.0])


def test_structure_readout_zone_must_be_ordered():
    with pytest.raises(ValueError, match="must be ≤"):
        StructureReadout(fvg_zone=[2381.0, 2378.0])  # inverted


def test_structure_readout_all_optional():
    """All fields default to None — useful for NEUTRAL setups with no structure."""
    s = StructureReadout()
    assert s.bos_level is None
    assert s.choch_present is False
    assert s.fvg_zone is None


def test_structure_readout_ob_strength_bounded():
    with pytest.raises(ValueError):
        StructureReadout(ob_strength=1.5)  # > 1.0


# ---------------------------------------------------------------------------
# RegimeReadout
# ---------------------------------------------------------------------------


def test_regime_readout_full():
    r = RegimeReadout(
        hmm_label="trend_bullish",
        hmm_posterior=0.71,
        bocpd_changepoint_prob=0.03,
        expected_run_length=180.0,
        jump_ratio=0.12,
        regime_gate_decision="TRADE",
    )
    assert r.regime_gate_decision == "TRADE"


@pytest.mark.parametrize("decision", ["TRADE", "REDUCE", "BLOCK"])
def test_regime_gate_decision_accepted(decision):
    r = RegimeReadout(regime_gate_decision=decision)
    assert r.regime_gate_decision == decision


def test_regime_gate_decision_rejected():
    with pytest.raises(ValueError, match="TRADE/REDUCE/BLOCK"):
        RegimeReadout(regime_gate_decision="OPEN_LONG")  # nope, never


def test_regime_readout_probabilities_bounded():
    with pytest.raises(ValueError):
        RegimeReadout(hmm_posterior=1.5)
    with pytest.raises(ValueError):
        RegimeReadout(jump_ratio=-0.1)


# ---------------------------------------------------------------------------
# VolatilityReadout
# ---------------------------------------------------------------------------


def test_volatility_readout_full():
    v = VolatilityReadout(
        regime="normal",
        forecast_atr_pips=8.7,
        naive_atr_pips=7.9,
        forecast_vs_naive_pct=10.13,
        confidence_interval_pips=[7.2, 10.4],
        is_fallback=False,
    )
    assert v.confidence_interval_pips == [7.2, 10.4]


def test_volatility_ci_must_be_ordered():
    with pytest.raises(ValueError, match="\\[0\\] must be ≤"):
        VolatilityReadout(confidence_interval_pips=[10.4, 7.2])  # inverted


def test_volatility_ci_must_be_2_elements():
    with pytest.raises(ValueError, match="must have 2 elements"):
        VolatilityReadout(confidence_interval_pips=[7.2])


def test_volatility_fallback_flag():
    v = VolatilityReadout(is_fallback=True, regime="normal", forecast_atr_pips=8.0, naive_atr_pips=8.0)
    assert v.is_fallback is True


# ---------------------------------------------------------------------------
# EventReadout
# ---------------------------------------------------------------------------


def test_event_readout_blackout_inactive():
    e = EventReadout(
        news_blackout_active=False,
        next_event_label="FOMC Minutes",
        next_event_in_minutes=1083,
        sentiment_score=0.3,
        sentiment_confidence=0.7,
        session="new_york",
    )
    assert e.session == "new_york"


def test_event_readout_sentiment_bounded():
    with pytest.raises(ValueError):
        EventReadout(sentiment_score=1.5)
    with pytest.raises(ValueError):
        EventReadout(sentiment_score=-2.0)


# ---------------------------------------------------------------------------
# ComponentBreakdown
# ---------------------------------------------------------------------------


def test_component_breakdown_full():
    c = ComponentBreakdown(
        name="bos",
        contribution=13.5,
        weight_max=15.0,
        reasoning="Bullish BOS retest confirmed",
    )
    assert c.score_pct == 90.0


def test_component_breakdown_redacted_weight():
    """B2C surface may redact weight_max ⇒ score_pct returns None."""
    c = ComponentBreakdown(
        name="bos",
        contribution=13.5,
        weight_max=None,
        reasoning="Bullish BOS retest confirmed",
    )
    assert c.score_pct is None


def test_component_breakdown_zero_weight():
    c = ComponentBreakdown(name="x", contribution=0.0, weight_max=0.0, reasoning="n/a")
    assert c.score_pct is None  # division by zero ⇒ None


# ---------------------------------------------------------------------------
# HistoricalStats
# ---------------------------------------------------------------------------


def test_historical_stats_full():
    h = HistoricalStats(
        similar_setups_n=329,
        hit_rate_observed=0.319,
        profit_factor=1.30,
        profit_factor_ci95=[1.12, 1.49],
        empirical_coverage=0.91,
        backtest_window="XAU M15 2019-2025 walk-forward",
    )
    assert h.profit_factor == 1.30


def test_historical_stats_pf_ci_ordered():
    with pytest.raises(ValueError):
        HistoricalStats(profit_factor_ci95=[1.49, 1.12])  # inverted


def test_historical_stats_pf_ci_2_elements():
    with pytest.raises(ValueError, match="2 elements"):
        HistoricalStats(profit_factor_ci95=[1.30])


def test_historical_stats_hit_rate_bounded():
    with pytest.raises(ValueError):
        HistoricalStats(hit_rate_observed=1.5)


# ---------------------------------------------------------------------------
# InsightSignalV2 top-level integration
# ---------------------------------------------------------------------------


def _make_enriched_signal() -> InsightSignalV2:
    """Build a fully-populated v2.1.0 signal — used across tests."""
    return InsightSignalV2(
        id="enriched_001",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.BULLISH_SETUP,
        conviction_0_100=72,
        levels=SignalLevels(entry=2350, stop=2340, target_1=2370),
        uncertainty=UncertaintyContext(
            conformal_lower=54.0, conformal_upper=82.0,
            coverage_alpha=0.10, n_calibration=2000,
            empirical_coverage=0.91,
        ),
        structure_readout=StructureReadout(
            bos_level=2391.50, bos_event_age_bars=2,
            fvg_zone=[2378.0, 2381.0], fvg_size_atr=0.42,
            ob_zone=[2375.0, 2378.0], ob_strength=0.73,
            retest_state="armed", structural_invalidation=2374.50,
            liquidity_zone_upper=[2398.50, 2401.20],
        ),
        regime_readout=RegimeReadout(
            hmm_label="trend_bullish", hmm_posterior=0.71,
            bocpd_changepoint_prob=0.03, expected_run_length=180.0,
            jump_ratio=0.12, regime_gate_decision="TRADE",
        ),
        volatility_readout=VolatilityReadout(
            regime="normal", forecast_atr_pips=8.7,
            naive_atr_pips=7.9, forecast_vs_naive_pct=10.13,
            confidence_interval_pips=[7.2, 10.4],
        ),
        event_readout=EventReadout(
            news_blackout_active=False,
            next_event_label="FOMC Minutes",
            next_event_in_minutes=1083,
            sentiment_score=0.3, sentiment_confidence=0.7,
            session="new_york",
        ),
        breakdown_components=[
            ComponentBreakdown(name="bos", contribution=13.5, weight_max=15.0, reasoning="BOS retest"),
            ComponentBreakdown(name="fvg", contribution=12.8, weight_max=15.0, reasoning="FVG retest"),
            ComponentBreakdown(name="order_block", contribution=7.5, weight_max=10.0, reasoning="OB aligned"),
            ComponentBreakdown(name="regime", contribution=18.2, weight_max=25.0, reasoning="HMM trend_bullish"),
            ComponentBreakdown(name="news", contribution=0.0, weight_max=20.0, reasoning="no blackout"),
            ComponentBreakdown(name="volume", contribution=7.0, weight_max=10.0, reasoning="1.4x avg"),
            ComponentBreakdown(name="momentum", contribution=2.4, weight_max=3.0, reasoning="RSI 42→58"),
            ComponentBreakdown(name="rsi_divergence", contribution=0.6, weight_max=2.0, reasoning="hidden bull div"),
        ],
        historical_stats=HistoricalStats(
            similar_setups_n=329, hit_rate_observed=0.319,
            profit_factor=1.30, profit_factor_ci95=[1.12, 1.49],
            empirical_coverage=0.91,
            backtest_window="XAU M15 2019-2025 walk-forward",
        ),
        narrative_short="Bullish XAU M15: BOS+retest+regime+vol normal.",
        created_at_utc=datetime(2026, 5, 1, 11, 47, tzinfo=timezone.utc),
    )


def test_signal_v2_accepts_all_new_subfields():
    sig = _make_enriched_signal()
    assert sig.schema_version == "2.2.0"
    assert sig.uncertainty.conformal_lower == 54.0
    assert sig.structure_readout.retest_state == "armed"
    assert sig.regime_readout.hmm_label == "trend_bullish"
    assert sig.volatility_readout.forecast_atr_pips == 8.7
    assert sig.event_readout.session == "new_york"
    assert len(sig.breakdown_components) == 8
    assert sig.historical_stats.similar_setups_n == 329


def test_signal_v2_new_fields_all_optional():
    """A minimal 2.0.0-style signal still validates under 2.1.0 (backward compat)."""
    sig = InsightSignalV2(
        id="minimal_001",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=30,
        narrative_short="No setup.",
        created_at_utc=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )
    assert sig.uncertainty is None
    assert sig.structure_readout is None
    assert sig.regime_readout is None
    assert sig.volatility_readout is None
    assert sig.event_readout is None
    assert sig.breakdown_components == []
    assert sig.historical_stats is None


def test_enriched_signal_round_trip_json():
    """Full 2.1.0 enrichment round-trips through JSON without loss."""
    original = _make_enriched_signal()
    payload = original.model_dump_json()
    parsed = InsightSignalV2.model_validate_json(payload)

    assert parsed.uncertainty.conformal_lower == original.uncertainty.conformal_lower
    assert parsed.uncertainty.empirical_coverage == original.uncertainty.empirical_coverage
    assert parsed.structure_readout.fvg_zone == original.structure_readout.fvg_zone
    assert parsed.regime_readout.bocpd_changepoint_prob == original.regime_readout.bocpd_changepoint_prob
    assert parsed.volatility_readout.confidence_interval_pips == original.volatility_readout.confidence_interval_pips
    assert parsed.event_readout.next_event_in_minutes == original.event_readout.next_event_in_minutes
    assert len(parsed.breakdown_components) == 8
    assert parsed.breakdown_components[0].name == "bos"
    assert parsed.historical_stats.profit_factor_ci95 == [1.12, 1.49]


def test_enriched_signal_b2b_dict_includes_all_readouts():
    sig = _make_enriched_signal()
    d = to_b2b_dict(sig)
    assert d["schema_version"] == "2.2.0"
    assert d["uncertainty"]["conformal_lower"] == 54.0
    assert d["structure_readout"]["bos_level"] == 2391.50
    assert d["regime_readout"]["regime_gate_decision"] == "TRADE"
    assert d["volatility_readout"]["regime"] == "normal"
    assert d["event_readout"]["news_blackout_active"] is False
    assert len(d["breakdown_components"]) == 8
    assert d["historical_stats"]["similar_setups_n"] == 329


def test_audit_row_remains_minimal_after_enrichment():
    """to_audit_row is a deliberate compaction — should ignore the 2.1.0 enrichment."""
    sig = _make_enriched_signal()
    row = to_audit_row(sig)
    # Same 12 keys as 2.0.0 — readouts are NOT in the audit row (hashed
    # separately upstream). This is intentional: the audit chain must remain
    # compact and deterministic.
    expected_keys = {
        "signal_id", "schema_version", "instrument", "timeframe",
        "direction", "conviction_0_100", "entry", "stop", "target_1",
        "edge_claim", "is_paper_demo", "created_at_utc",
    }
    assert set(row.keys()) == expected_keys
    assert row["schema_version"] == "2.2.0"


# ---------------------------------------------------------------------------
# Compliance: descriptive-only invariants on the new readouts
# ---------------------------------------------------------------------------


def test_regime_gate_never_contains_buy_sell_verbs():
    """Sanity check: the regime gate enum is descriptive, never prescriptive."""
    for decision in ("TRADE", "REDUCE", "BLOCK"):
        r = RegimeReadout(regime_gate_decision=decision)
        # No verb form — these are *system categorical assessments*, not user
        # instructions. The trader still composes the trade themselves.
        assert decision in ("TRADE", "REDUCE", "BLOCK")
    # And the forbidden ones really are forbidden:
    for forbidden in ("BUY", "SELL", "OPEN_LONG", "OPEN_SHORT", "CLOSE"):
        with pytest.raises(ValueError):
            RegimeReadout(regime_gate_decision=forbidden)


def test_structure_readout_invalidation_is_not_a_stop():
    """The field is named 'structural_invalidation', NOT 'stop_loss' — the
    semantic is 'where the SMC read breaks', not 'place your stop here'."""
    s = StructureReadout(structural_invalidation=2374.50)
    # If anyone renames or adds a 'stop_loss' field to StructureReadout in the
    # future, this assertion will fail and force review.
    assert not hasattr(s, "stop_loss")
    assert not hasattr(s, "stop")
    assert hasattr(s, "structural_invalidation")
