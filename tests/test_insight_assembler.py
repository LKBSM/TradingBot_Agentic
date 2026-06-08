"""Tests for InsightAssembler — pipeline → InsightSignalV2 (v2.1.0)."""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from src.api.insight_signal_v2 import (
    InsightSignalV2,
    NarrativeLanguage,
    SetupDirection,
    Timeframe,
)
from src.intelligence.insight_assembler import AssemblerDefaults, InsightAssembler
from src.intelligence.scoring.calibrated_conviction import CalibratedConvictionPipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_confluence_signal(signal_type="LONG", score=72.0):
    return SimpleNamespace(
        signal_id="abc123def456",
        symbol="XAUUSD",
        signal_type=SimpleNamespace(value=signal_type),
        confluence_score=score,
        entry_price=2350.0,
        stop_loss=2340.0,
        take_profit=2370.0,
        atr=8.0,
        bar_timestamp="2026-05-01T11:47:00Z",
        components=[
            SimpleNamespace(name="BOS", weighted_score=13.5, weight=15.0, reasoning="BOS retest"),
            SimpleNamespace(name="FVG", weighted_score=12.8, weight=15.0, reasoning="FVG retest"),
        ],
    )


def _make_smc_features():
    return {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_BREAK_LEVEL": 2391.50,
        "FVG_SIGNAL": 1.0,
        "FVG_SIZE_NORM": 0.42,
        "FVG_LOW": 2378.0,
        "FVG_HIGH": 2381.0,
        "BULLISH_OB_LOW": 2375.0,
        "BULLISH_OB_HIGH": 2378.0,
        "OB_STRENGTH_NORM": 0.73,
        "BOS_RETEST_ARMED": 1.0,
    }


def _make_volatility_forecast():
    return SimpleNamespace(
        forecast_atr=8.7, naive_atr=7.9,
        confidence_lower=7.2, confidence_upper=10.4,
        regime_state="normal", is_fallback=False,
    )


def _make_regime_gate_output(decision="TRADE"):
    return SimpleNamespace(
        decision=SimpleNamespace(value=decision),
        cp_prob=0.03, jump_ratio=0.12, expected_run_length=180.0,
    )


def _make_news_assessment(blocked=False):
    return SimpleNamespace(
        decision=SimpleNamespace(value="block" if blocked else "allow"),
        sentiment_score=0.3,
        sentiment_confidence=0.7,
        blocking_events=[SimpleNamespace(event_name="FOMC Minutes")] if blocked else [],
        hours_to_next_high_impact=18.05,
    )


# ---------------------------------------------------------------------------
# Direction derivation
# ---------------------------------------------------------------------------


def test_assembler_long_signal_is_bullish_setup():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal("LONG"),
    )
    assert sig.direction == SetupDirection.BULLISH_SETUP


def test_assembler_short_signal_is_bearish_setup():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal("SHORT"),
    )
    assert sig.direction == SetupDirection.BEARISH_SETUP


def test_assembler_none_signal_is_neutral():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=None,
    )
    assert sig.direction == SetupDirection.NEUTRAL
    # NEUTRAL must not carry levels
    assert sig.levels.entry is None


# ---------------------------------------------------------------------------
# Readouts integration
# ---------------------------------------------------------------------------


def test_assembler_populates_all_readouts():
    a = InsightAssembler()
    cs = _make_confluence_signal()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=cs,
        smc_features=_make_smc_features(),
        volatility_forecast=_make_volatility_forecast(),
        regime_analysis=SimpleNamespace(regime=SimpleNamespace(value="uptrend"), confidence=0.71),
        regime_gate_output=_make_regime_gate_output("TRADE"),
        news_assessment=_make_news_assessment(blocked=False),
        session_label="new_york",
        narrative_short="Lecture haussière XAU M15.",
    )
    assert sig.structure_readout is not None
    assert sig.structure_readout.bos_level == 2391.50
    assert sig.regime_readout is not None
    assert sig.regime_readout.hmm_label == "trend_bullish"
    assert sig.regime_readout.regime_gate_decision == "TRADE"
    assert sig.volatility_readout is not None
    assert sig.volatility_readout.regime == "normal"
    assert sig.event_readout is not None
    assert sig.event_readout.session == "new_york"
    assert sig.event_readout.news_blackout_active is False
    assert len(sig.breakdown_components) == 2  # only 2 components in fixture


# ---------------------------------------------------------------------------
# Conviction sourcing
# ---------------------------------------------------------------------------


def test_assembler_uses_raw_score_when_no_calibrated_pipeline():
    a = InsightAssembler(calibrated_pipeline=None)
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(score=68.4),
    )
    assert sig.conviction_0_100 == 68
    assert sig.uncertainty is None  # no calibrated pipeline ⇒ no uncertainty


def test_assembler_uses_calibrated_pipeline_when_available():
    class _StubLGBM:
        from src.intelligence.scoring.lgbm_scorer import DEFAULT_FEATURE_NAMES
        feature_names = DEFAULT_FEATURE_NAMES
        _model = object()

        def predict_p_win(self, X):
            return np.full(X.shape[0], 0.72)

        def feature_importance(self):
            return {n: 0.1 for n in self.feature_names}

    pipe = CalibratedConvictionPipeline(lgbm=_StubLGBM())
    a = InsightAssembler(calibrated_pipeline=pipe)
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(score=68.4),
        feature_vector=np.ones(8),
    )
    # Calibrated wins over raw — 0.72 * 100 = 72
    assert sig.conviction_0_100 == 72
    assert sig.uncertainty is not None


def test_assembler_defaults_to_50_on_neutral():
    a = InsightAssembler()
    sig = a.assemble(instrument="XAUUSD", timeframe="M15", confluence_signal=None)
    assert sig.conviction_0_100 == 50


# ---------------------------------------------------------------------------
# Levels handling — indicator stance
# ---------------------------------------------------------------------------


def test_assembler_excludes_levels_by_default():
    """include_levels=False (default) ⇒ entry/stop/target are all None."""
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    assert sig.levels.entry is None
    assert sig.levels.stop is None
    assert sig.levels.target_1 is None


def test_assembler_includes_levels_when_requested():
    """include_levels=True ⇒ backward compat for v2.0.0 consumers."""
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
        include_levels=True,
    )
    assert sig.levels.entry == 2350.0
    assert sig.levels.stop == 2340.0
    assert sig.levels.target_1 == 2370.0


# ---------------------------------------------------------------------------
# Compliance defaults
# ---------------------------------------------------------------------------


def test_assembler_default_compliance_is_conservative():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    assert sig.compliance.edge_claim is False
    assert sig.compliance.is_paper_demo is True


def test_assembler_compliance_can_be_overridden():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
        is_paper_demo=False,
    )
    assert sig.compliance.is_paper_demo is False


# ---------------------------------------------------------------------------
# Lifecycle / created_at
# ---------------------------------------------------------------------------


def test_assembler_uses_bar_timestamp_from_signal():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    assert sig.created_at_utc.year == 2026
    assert sig.created_at_utc.month == 5
    assert sig.created_at_utc.tzinfo is not None


def test_assembler_valid_until_4_hours_after_creation():
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    delta = sig.valid_until_utc - sig.created_at_utc
    assert delta.total_seconds() == 4 * 3600


# ---------------------------------------------------------------------------
# Historical stats callback
# ---------------------------------------------------------------------------


def test_historical_stats_callback_populates_field():
    def stats_fn(symbol):
        return {
            "similar_setups_n": 329,
            "hit_rate": 0.319,
            "profit_factor": 1.30,
            "profit_factor_ci95": [1.12, 1.49],
            "empirical_coverage": 0.91,
        }

    a = InsightAssembler(historical_stats_fn=stats_fn)
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    assert sig.historical_stats is not None
    assert sig.historical_stats.similar_setups_n == 329


def test_historical_stats_callback_exception_does_not_crash():
    def stats_fn(symbol):
        raise RuntimeError("DB down")

    a = InsightAssembler(historical_stats_fn=stats_fn)
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    assert sig.historical_stats is None  # gracefully omitted


# ---------------------------------------------------------------------------
# End-to-end smoke
# ---------------------------------------------------------------------------


def test_full_e2e_serializes_to_json():
    """Full pipeline → assemble → JSON → reparse — no data loss."""
    a = InsightAssembler()
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
        smc_features=_make_smc_features(),
        volatility_forecast=_make_volatility_forecast(),
        regime_analysis=SimpleNamespace(regime=SimpleNamespace(value="uptrend"), confidence=0.71),
        regime_gate_output=_make_regime_gate_output(),
        news_assessment=_make_news_assessment(),
        session_label="new_york",
    )
    payload = sig.model_dump_json()
    parsed = InsightSignalV2.model_validate_json(payload)
    assert parsed.structure_readout.fvg_zone == [2378.0, 2381.0]
    assert parsed.regime_readout.regime_gate_decision == "TRADE"


def test_assembler_b2c_redacts_weights():
    """expose_component_weights=False → weight_max is None on every row."""
    a = InsightAssembler(
        defaults=AssemblerDefaults(expose_component_weights=False),
    )
    sig = a.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_make_confluence_signal(),
    )
    for row in sig.breakdown_components:
        assert row.weight_max is None
        assert row.score_pct is None
        # But contribution is still exposed
        assert row.contribution is not None


# ---------------------------------------------------------------------------
# observe_outcome
# ---------------------------------------------------------------------------


def test_observe_outcome_no_pipeline_is_safe():
    a = InsightAssembler()
    a.observe_outcome(0.5)  # Should not raise


def test_observe_outcome_with_pipeline_calls_through():
    pipeline = CalibratedConvictionPipeline()
    called = []
    original = pipeline.observe_outcome
    pipeline.observe_outcome = lambda r: called.append(r)
    a = InsightAssembler(calibrated_pipeline=pipeline)
    a.observe_outcome(1.2)
    assert called == [1.2]
