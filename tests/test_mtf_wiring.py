"""Integration tests for the Phase 1 MTF rewiring (2026-05-21).

Covers:
  1. ConfluenceDetector accepts ``htf_features`` and emits the
     HTF_Alignment component without changing behaviour at weight=0.
  2. The HTF_Alignment scorer produces the expected quality bands
     (aligned / counter / neutral / no-data).
  3. ``MultiTimeframeReadout`` (v2.2.0) populates correctly from a
     real ``MultiTimeframeFeatures`` features dict.
  4. ``map_mtf_readout`` keeps the alignment label in sync with
     ``ConfluenceDetector._score_htf_alignment``.
  5. ``InsightAssembler.assemble(htf_features=...)`` propagates the
     readout end-to-end into the v2 contract.
  6. The default ``DEFAULT_WEIGHTS["htf_alignment"] == 0.0`` invariant
     holds (Phase 1 must not change tier distribution).

These tests intentionally exercise the *integration* surface (scanner +
detector + assembler) rather than the unit-level MTF features module
(``test_multi_timeframe.py`` covers that — 28 tests).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pytest

from src.api.insight_signal_v2 import (
    InsightSignalV2,
    MultiTimeframeReadout,
    SCHEMA_VERSION,
    SetupDirection,
)
from src.environment.multi_timeframe_features import MultiTimeframeFeatures
from src.intelligence.confluence_detector import (
    DEFAULT_WEIGHTS,
    ConfluenceDetector,
    SignalType,
)
from src.intelligence.insight_assembler import InsightAssembler
from src.intelligence.readout_mappers import map_mtf_readout


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_smc_features(direction: SignalType = SignalType.LONG) -> Dict[str, float]:
    """Build a minimal SMC dict that lets ConfluenceDetector produce a signal.

    Sets BOS aligned with ``direction`` and the retest_armed flag so the
    require_retest=True default gate lets the signal through.
    """
    sign = 1.0 if direction == SignalType.LONG else -1.0
    return {
        "BOS_SIGNAL": sign,
        "BOS_EVENT": sign,
        "BOS_RETEST_STATE": sign,
        "BOS_RETEST_ARMED": sign,
        "CHOCH_SIGNAL": sign,
        "FVG_SIGNAL": sign,
        "FVG_SIZE_NORM": 0.5,
        "OB_STRENGTH_NORM": 0.7,
        "RSI": 60.0 if direction == SignalType.LONG else 40.0,
        "MACD_Diff": 0.3 * sign,
        "CHOCH_DIVERGENCE": sign,
    }


def _make_htf_features(
    h4_trend: float = 1.0,
    h1_trend: float = 1.0,
    h4_strength: float = 0.6,
    h1_strength: float = 0.4,
) -> Dict[str, float]:
    """Build a realistic features dict that mirrors MultiTimeframeFeatures output."""
    return {
        "HTF_TREND_1H": h1_trend,
        "HTF_TREND_4H": h4_trend,
        "HTF_STRENGTH_1H": h1_strength,
        "HTF_STRENGTH_4H": h4_strength,
        "HTF_RSI_1H": 0.55,
        "HTF_RSI_4H": 0.60,
        "PRICE_VS_SMA20_1H": 0.2,
        "PRICE_VS_SMA50_1H": 0.3,
        "PRICE_VS_SMA20_4H": 0.4,
        "PRICE_VS_SMA50_4H": 0.5,
        "SESSION": 0.5,  # London
        "DAY_OF_WEEK": 0.5,
        "HOUR_SIN": 0.0,
        "HOUR_COS": 1.0,
    }


def _make_real_htf_features(n_bars: int = 1000, seed: int = 7) -> Dict[str, float]:
    """Build a synthetic M15 frame and run MultiTimeframeFeatures end-to-end.

    Guarantees the features dict comes from the real production module
    (not a hand-crafted stub) so the schema stays in sync.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_bars, freq="15min")
    close = 2400.0 + np.cumsum(rng.standard_normal(n_bars) * 0.5)
    open_ = close + rng.standard_normal(n_bars) * 0.3
    high = np.maximum(open_, close) + np.abs(rng.standard_normal(n_bars))
    low = np.minimum(open_, close) - np.abs(rng.standard_normal(n_bars))
    volume = rng.integers(500, 2000, n_bars).astype(float)
    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    mtf = MultiTimeframeFeatures(base_timeframe="15min")
    mtf.fit(df)
    return mtf.get_features(idx=n_bars - 1)


# ---------------------------------------------------------------------------
# (1) ConfluenceDetector accepts htf_features without behavioural change
# ---------------------------------------------------------------------------


def test_default_weights_htf_alignment_is_zero():
    """Phase 1 invariant: htf_alignment must be wired at weight 0 so we
    don't alter the score distribution until Phase 2 validation."""
    assert "htf_alignment" in DEFAULT_WEIGHTS
    assert DEFAULT_WEIGHTS["htf_alignment"] == 0.0
    # Total must still sum to 100 — the renormalisation logic depends on it.
    assert abs(sum(DEFAULT_WEIGHTS.values()) - 100.0) < 1e-9


def test_confluence_analyze_accepts_htf_features_kwarg():
    """The new kwarg is opt-in: omitting it must keep the legacy behaviour."""
    detector = ConfluenceDetector(symbol="XAUUSD", min_score=10.0)
    smc = _make_smc_features(SignalType.LONG)
    sig_without = detector.analyze(
        smc_features=smc, regime=None, news=None, price=2400.0, atr=10.0,
    )
    sig_with = detector.analyze(
        smc_features=smc, regime=None, news=None, price=2400.0, atr=10.0,
        htf_features=_make_htf_features(),
    )
    assert sig_without is not None
    assert sig_with is not None
    # At weight=0 the score is identical regardless of HTF input.
    assert sig_without.confluence_score == pytest.approx(sig_with.confluence_score)


def test_htf_alignment_component_emitted_even_at_weight_zero():
    """Even at weight 0 the component must be emitted so the descriptive
    readout (mtf_readout in InsightSignalV2) can be populated."""
    detector = ConfluenceDetector(symbol="XAUUSD", min_score=10.0)
    sig = detector.analyze(
        smc_features=_make_smc_features(),
        regime=None, news=None, price=2400.0, atr=10.0,
        htf_features=_make_htf_features(),
    )
    assert sig is not None
    names = [c.name for c in sig.components]
    assert "HTF_Alignment" in names
    htf_cmp = next(c for c in sig.components if c.name == "HTF_Alignment")
    assert htf_cmp.weight == 0.0
    assert htf_cmp.weighted_score == 0.0
    # The reasoning string must still carry meaningful info.
    assert "H4" in htf_cmp.reasoning
    assert "H1" in htf_cmp.reasoning


# ---------------------------------------------------------------------------
# (2) HTF scorer quality bands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "h4,h1,direction,expected_label",
    [
        (1.0, 1.0, SignalType.LONG, "full HTF alignment"),
        (1.0, 0.0, SignalType.LONG, "H4 aligned"),
        (0.0, 1.0, SignalType.LONG, "H1 aligned"),
        (0.0, 0.0, SignalType.LONG, "HTF ranging"),
        (-1.0, 1.0, SignalType.LONG, "counter-trend H4"),
        (-1.0, -1.0, SignalType.SHORT, "full HTF alignment"),
        (1.0, 1.0, SignalType.SHORT, "counter-trend H4"),
    ],
)
def test_htf_scorer_quality_bands(h4, h1, direction, expected_label):
    """The scorer must produce the documented quality bands."""
    detector = ConfluenceDetector(symbol="XAUUSD", min_score=10.0)
    cmp = detector._score_htf_alignment(
        _make_htf_features(h4_trend=h4, h1_trend=h1, h4_strength=0.7),
        direction,
    )
    assert expected_label in cmp.reasoning


# ---------------------------------------------------------------------------
# (3) MultiTimeframeReadout Pydantic model
# ---------------------------------------------------------------------------


def test_mtf_readout_accepts_full_payload():
    r = MultiTimeframeReadout(
        h1_trend="bullish", h4_trend="bullish",
        h1_strength=0.4, h4_strength=0.6,
        h1_rsi=55.0, h4_rsi=60.0,
        alignment_with_setup="aligned", alignment_score_0_1=0.85,
        session="london",
    )
    assert r.h4_trend == "bullish"
    assert r.alignment_score_0_1 == 0.85


def test_mtf_readout_rejects_invalid_alignment_label():
    with pytest.raises(ValueError):
        MultiTimeframeReadout(alignment_with_setup="unknown_label")


def test_mtf_readout_is_optional_on_signal():
    """A v2.2.0 signal without MTF data still validates."""
    from datetime import datetime, timezone

    sig = InsightSignalV2(
        id="mtf_optional_001",
        instrument="XAUUSD",
        timeframe="M15",
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=30,
        narrative_short="No setup.",
        created_at_utc=datetime.now(timezone.utc),
    )
    assert sig.mtf_readout is None
    assert sig.schema_version == "2.2.0"


# ---------------------------------------------------------------------------
# (4) map_mtf_readout consistency with scorer
# ---------------------------------------------------------------------------


def test_map_mtf_readout_returns_none_on_empty_input():
    assert map_mtf_readout(None) is None
    assert map_mtf_readout({}) is None


def test_map_mtf_readout_labels_match_trend_signs():
    r = map_mtf_readout(
        _make_htf_features(h4_trend=1.0, h1_trend=-1.0),
        direction_hint="LONG",
    )
    assert r is not None
    assert r.h4_trend == "bullish"
    assert r.h1_trend == "bearish"
    # H4 aligned, H1 counter → still aligned (H4 dominates per the scorer)
    assert r.alignment_with_setup == "aligned"


def test_map_mtf_readout_counter_when_h4_opposes():
    r = map_mtf_readout(
        _make_htf_features(h4_trend=-1.0, h1_trend=1.0),
        direction_hint="LONG",
    )
    assert r is not None
    assert r.alignment_with_setup == "counter"
    assert r.alignment_score_0_1 == 0.0


def test_map_mtf_readout_na_when_no_direction_hint():
    r = map_mtf_readout(_make_htf_features(), direction_hint=None)
    assert r is not None
    assert r.alignment_with_setup == "na"
    assert r.alignment_score_0_1 is None


def test_map_mtf_readout_rsi_denormalisation():
    """RSI comes from MTF normalised 0-1; the readout exposes 0-100."""
    r = map_mtf_readout(_make_htf_features())
    assert r is not None
    assert r.h1_rsi == pytest.approx(55.0)
    assert r.h4_rsi == pytest.approx(60.0)


def test_map_mtf_readout_session_label():
    cases = [
        (0.0, "asian"),
        (0.5, "london"),
        (1.0, "new_york"),
    ]
    for sess_norm, expected in cases:
        feats = _make_htf_features()
        feats["SESSION"] = sess_norm
        r = map_mtf_readout(feats)
        assert r is not None
        assert r.session == expected


# ---------------------------------------------------------------------------
# (5) InsightAssembler end-to-end propagation
# ---------------------------------------------------------------------------


class _MockSignal:
    """Duck-typed ConfluenceSignal subset that the assembler needs."""

    def __init__(self):
        self.signal_id = "test_sig_001"
        self.signal_type = type("ST", (), {"value": "LONG"})()
        self.confluence_score = 60.0
        self.components = []
        self.bar_timestamp = None


def test_assembler_propagates_mtf_readout_end_to_end():
    """``InsightAssembler.assemble(htf_features=...)`` populates ``mtf_readout``."""
    assembler = InsightAssembler()
    insight = assembler.assemble(
        instrument="XAUUSD",
        timeframe="M15",
        confluence_signal=_MockSignal(),
        htf_features=_make_htf_features(h4_trend=1.0, h1_trend=1.0),
    )
    assert insight.mtf_readout is not None
    assert insight.mtf_readout.h4_trend == "bullish"
    assert insight.mtf_readout.alignment_with_setup == "aligned"
    assert insight.mtf_readout.h4_rsi == pytest.approx(60.0)
    assert insight.schema_version == "2.2.0"


def test_assembler_mtf_readout_none_when_features_absent():
    assembler = InsightAssembler()
    insight = assembler.assemble(
        instrument="XAUUSD", timeframe="M15",
        confluence_signal=_MockSignal(),
    )
    assert insight.mtf_readout is None


# ---------------------------------------------------------------------------
# (6) Real MultiTimeframeFeatures → mapper round-trip
# ---------------------------------------------------------------------------


def test_real_mtf_features_round_trip_through_mapper():
    """The features dict produced by the production module must be
    consumable by the mapper without errors."""
    features = _make_real_htf_features(n_bars=1000, seed=11)
    # Production keys are present
    assert "HTF_TREND_1H" in features
    assert "HTF_TREND_4H" in features
    assert "HTF_STRENGTH_4H" in features
    assert "HTF_RSI_4H" in features
    r = map_mtf_readout(features, direction_hint="LONG")
    assert r is not None
    # Labels must be one of the allowed strings
    assert r.h4_trend in ("bullish", "bearish", "neutral")
    assert r.h1_trend in ("bullish", "bearish", "neutral")
    assert r.alignment_with_setup in ("aligned", "counter", "neutral", "na")
    # RSI ranges are sensible
    if r.h4_rsi is not None:
        assert 0.0 <= r.h4_rsi <= 100.0


def test_real_mtf_features_feed_confluence_detector():
    """The features dict from the real module must be consumable by the
    detector's HTF scorer without raising."""
    features = _make_real_htf_features(n_bars=1000, seed=13)
    detector = ConfluenceDetector(symbol="XAUUSD", min_score=10.0)
    sig = detector.analyze(
        smc_features=_make_smc_features(SignalType.LONG),
        regime=None, news=None, price=2400.0, atr=10.0,
        htf_features=features,
    )
    assert sig is not None
    # Score still computable, HTF component present and at weight 0.
    htf = next(c for c in sig.components if c.name == "HTF_Alignment")
    assert htf.weight == 0.0
    assert htf.weighted_score == 0.0
