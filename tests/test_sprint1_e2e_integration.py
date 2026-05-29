"""End-to-end integration test for Sprint 1 deliverables.

This test exercises the full Sprint 1 stack:
  1. Train a calibrated pipeline on synthetic data (J1-2)
  2. Save and reload (J1-2)
  3. Mappers project pipeline outputs onto readouts (J5-6)
  4. Assembler composes InsightSignalV2 from pipeline objects (J3-4)
  5. Renderers produce Telegram + B2B payloads (J7)
  6. Round-trip through JSON survives all readouts

If this test passes, Sprint 1 is wired end-to-end and ready for the
scanner integration (which is a one-line call to InsightAssembler.assemble).
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_calibrated_conviction as train_mod  # noqa: E402

from src.api.insight_signal_v2 import (  # noqa: E402
    InsightSignalV2,
    SetupDirection,
    to_b2b_dict,
    to_telegram_b2c,
)
from src.intelligence.insight_assembler import InsightAssembler  # noqa: E402
from src.intelligence.scoring.calibrated_conviction import (  # noqa: E402
    CalibratedConvictionPipeline,
)


@pytest.fixture(scope="module")
def trained_pipeline(tmp_path_factory) -> CalibratedConvictionPipeline:
    """Train a small calibrated pipeline on synthetic data, save+reload."""
    tmp = tmp_path_factory.mktemp("sprint1")
    df = train_mod.synthetic_replay(n_signals=400, seed=2026)
    pipeline = train_mod.train(df, val_fraction=0.30, alpha=0.10,
                                lgbm_kwargs={"n_estimators": 50})
    out = tmp / "pipeline.pkl"
    train_mod.save_pipeline(pipeline, out)
    return train_mod.load_calibrated_pipeline(out)


def _make_pipeline_outputs():
    """Build a synthetic set of pipeline outputs as the scanner would assemble."""
    confluence_signal = SimpleNamespace(
        signal_id="e2e_test_signal_001",
        symbol="XAUUSD",
        signal_type=SimpleNamespace(value="LONG"),
        confluence_score=72.4,
        entry_price=2350.0,
        stop_loss=2340.0,
        take_profit=2370.0,
        atr=8.0,
        bar_timestamp="2026-05-16T11:47:00Z",
        components=[
            SimpleNamespace(name="BOS", weighted_score=13.5, weight=15.0, reasoning="BOS retest"),
            SimpleNamespace(name="FVG", weighted_score=12.8, weight=15.0, reasoning="FVG fresh + retest"),
            SimpleNamespace(name="OrderBlock", weighted_score=7.5, weight=10.0, reasoning="OB aligned"),
            SimpleNamespace(name="Regime", weighted_score=18.2, weight=25.0, reasoning="HMM trend"),
            SimpleNamespace(name="News", weighted_score=0.0, weight=20.0, reasoning="no blackout"),
            SimpleNamespace(name="Volume", weighted_score=7.0, weight=10.0, reasoning="1.4× avg"),
            SimpleNamespace(name="Momentum", weighted_score=2.4, weight=3.0, reasoning="RSI 42→58"),
            SimpleNamespace(name="RSI_Divergence", weighted_score=0.6, weight=2.0, reasoning="hidden bull div"),
        ],
    )
    smc_features = {
        "BOS_SIGNAL": 1.0, "BOS_EVENT": 0.0,
        "BOS_BREAK_LEVEL": 2391.50,
        "FVG_SIGNAL": 1.0, "FVG_SIZE_NORM": 0.42,
        "FVG_LOW": 2378.0, "FVG_HIGH": 2381.0,
        "BULLISH_OB_LOW": 2375.0, "BULLISH_OB_HIGH": 2378.0,
        "OB_STRENGTH_NORM": 0.73,
        "BOS_RETEST_ARMED": 1.0,
    }
    vol_forecast = SimpleNamespace(
        forecast_atr=8.7, naive_atr=7.9,
        confidence_lower=7.2, confidence_upper=10.4,
        regime_state="normal", is_fallback=False,
    )
    regime_analysis = SimpleNamespace(
        regime=SimpleNamespace(value="uptrend"),
        confidence=0.71,
    )
    regime_gate = SimpleNamespace(
        decision=SimpleNamespace(value="TRADE"),
        cp_prob=0.03, jump_ratio=0.12, expected_run_length=180.0,
    )
    news = SimpleNamespace(
        decision=SimpleNamespace(value="allow"),
        sentiment_score=0.3, sentiment_confidence=0.7,
        blocking_events=[],
        hours_to_next_high_impact=18.05,
    )
    return {
        "confluence_signal": confluence_signal,
        "smc_features": smc_features,
        "volatility_forecast": vol_forecast,
        "regime_analysis": regime_analysis,
        "regime_gate_output": regime_gate,
        "news_assessment": news,
    }


# ---------------------------------------------------------------------------
# E2E flow — the canonical scanner integration call
# ---------------------------------------------------------------------------


def test_e2e_assemble_and_render(trained_pipeline):
    """The single integration point for the scanner: build → render."""
    assembler = InsightAssembler(
        calibrated_pipeline=trained_pipeline,
        historical_stats_fn=lambda symbol: {
            "similar_setups_n": 329,
            "hit_rate": 0.319,
            "profit_factor": 1.30,
            "profit_factor_ci95": [1.12, 1.49],
            "empirical_coverage": 0.91,
        },
    )
    outputs = _make_pipeline_outputs()

    sig = assembler.assemble(
        instrument="XAUUSD",
        timeframe="M15",
        narrative_short="Lecture haussière XAU M15 : BOS + retest FVG + régime normal.",
        narrative_long="Détaillé… (généré par LLM en prod)",
        session_label="new_york",
        feature_vector=np.array([13.5, 7.5, 12.8, 1.0, 18.2, 8.0, 0.0, 3.0]),
        **outputs,
    )

    # --- Assertions: the full v2.2.0 contract is populated ---
    assert sig.schema_version == "2.2.0"
    assert sig.direction == SetupDirection.BULLISH_SETUP
    assert 0 <= sig.conviction_0_100 <= 100
    assert sig.uncertainty is not None  # calibrated pipeline ⇒ interval
    assert sig.structure_readout.bos_level == 2391.50
    assert sig.structure_readout.fvg_zone == [2378.0, 2381.0]
    assert sig.regime_readout.hmm_label == "trend_bullish"
    assert sig.regime_readout.regime_gate_decision == "TRADE"
    assert sig.volatility_readout.regime == "normal"
    assert sig.event_readout.session == "new_york"
    assert sig.event_readout.news_blackout_active is False
    assert len(sig.breakdown_components) == 8
    assert sig.historical_stats.similar_setups_n == 329
    # Indicator stance — no entry/stop/target by default
    assert sig.levels.entry is None
    assert sig.compliance.edge_claim is False
    assert sig.compliance.is_paper_demo is True

    # --- Renderers: produce expected output shapes ---
    telegram_msg = to_telegram_b2c(sig)
    assert len(telegram_msg) <= 800
    assert "STRUCTURE HAUSSIÈRE" in telegram_msg
    assert "BUY" not in telegram_msg.upper()
    assert "ACHETEZ" not in telegram_msg.upper()
    # Descriptive content should appear
    assert "BOS" in telegram_msg
    assert "vol" in telegram_msg.lower()
    # No trade orders
    assert "Entrée :" not in telegram_msg
    assert "Stop :" not in telegram_msg
    assert "Cible :" not in telegram_msg

    b2b_payload = to_b2b_dict(sig)
    assert b2b_payload["schema_version"] == "2.2.0"
    assert b2b_payload["structure_readout"]["bos_level"] == 2391.50
    assert b2b_payload["regime_readout"]["regime_gate_decision"] == "TRADE"
    assert b2b_payload["historical_stats"]["similar_setups_n"] == 329
    assert len(b2b_payload["breakdown_components"]) == 8


def test_e2e_round_trip_json(trained_pipeline):
    """The fully-populated InsightSignal survives JSON round-trip."""
    assembler = InsightAssembler(calibrated_pipeline=trained_pipeline)
    outputs = _make_pipeline_outputs()
    sig = assembler.assemble(
        instrument="XAUUSD", timeframe="M15",
        feature_vector=np.array([13.5, 7.5, 12.8, 1.0, 18.2, 8.0, 0.0, 3.0]),
        **outputs,
    )
    payload = sig.model_dump_json()
    parsed = InsightSignalV2.model_validate_json(payload)

    assert parsed.schema_version == "2.2.0"
    assert parsed.uncertainty is not None
    assert parsed.structure_readout.fvg_zone == sig.structure_readout.fvg_zone
    assert parsed.regime_readout.bocpd_changepoint_prob == sig.regime_readout.bocpd_changepoint_prob
    assert parsed.volatility_readout.confidence_interval_pips == sig.volatility_readout.confidence_interval_pips
    assert len(parsed.breakdown_components) == len(sig.breakdown_components)


def test_e2e_neutral_when_no_confluence_signal(trained_pipeline):
    """When the scanner has no signal to publish, NEUTRAL is the safe default."""
    assembler = InsightAssembler(calibrated_pipeline=trained_pipeline)
    sig = assembler.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=None,
    )
    assert sig.direction == SetupDirection.NEUTRAL
    assert sig.levels.entry is None
    # NEUTRAL still produces a renderable signal
    telegram_msg = to_telegram_b2c(sig)
    assert "NEUTRE" in telegram_msg


def test_e2e_b2c_redacts_weights(trained_pipeline):
    """The B2C surface redacts weight_max from breakdown components (IP protection)."""
    from src.intelligence.insight_assembler import AssemblerDefaults

    assembler = InsightAssembler(
        calibrated_pipeline=trained_pipeline,
        defaults=AssemblerDefaults(expose_component_weights=False),
    )
    outputs = _make_pipeline_outputs()
    sig = assembler.assemble(
        instrument="XAUUSD", timeframe="M15",
        feature_vector=np.array([13.5, 7.5, 12.8, 1.0, 18.2, 8.0, 0.0, 3.0]),
        **outputs,
    )
    for row in sig.breakdown_components:
        assert row.weight_max is None
        assert row.score_pct is None
        assert row.contribution is not None  # contribution stays


def test_e2e_observe_outcome_loops_into_aci(trained_pipeline):
    """Realised outcome feedback updates the ACI conformal scorer state."""
    assembler = InsightAssembler(calibrated_pipeline=trained_pipeline)
    outputs = _make_pipeline_outputs()

    # First score to populate ACI's last_interval
    assembler.assemble(
        instrument="XAUUSD", timeframe="M15",
        feature_vector=np.array([13.5, 7.5, 12.8, 1.0, 18.2, 8.0, 0.0, 3.0]),
        **outputs,
    )
    # Feed back a realised outcome — should not raise
    assembler.observe_outcome(0.8)
    assembler.observe_outcome(-0.5)
    # ACI's miscoverage history should now have entries
    aci = trained_pipeline.conformal
    assert len(aci.state.miscoverage_history) >= 2
