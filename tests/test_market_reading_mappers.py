"""Tests for MarketReading mappers (Chantier 2 Étape 3)."""

from datetime import datetime, timezone

import pytest

from src.intelligence.market_reading_mappers import (
    FORBIDDEN_TOKENS,
    candles_to_regime,
    confluence_signal_to_structure,
    contains_forbidden_tokens,
    empty_events,
    tags_and_description,
)
from src.intelligence.market_reading_schema import (
    DESCRIPTION_MAX_LENGTH,
    MarketReadingEvents,
    MarketReadingRegime,
    MarketReadingStructure,
)


class _MockConfluenceSignal:
    """Minimal mock of ConfluenceSignal with a duck-typed signal_type."""

    class _SigType:
        def __init__(self, value: str):
            self.value = value

    def __init__(self, signal_type: str = "LONG"):
        self.signal_type = self._SigType(signal_type)


@pytest.fixture
def bar_ts() -> datetime:
    return datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Structure mapper
# ---------------------------------------------------------------------------


def test_confluence_signal_to_structure_with_long_signal(bar_ts):
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 1.0,
        "BOS_BREAK_LEVEL": 2375.20,  # F1: real broken structural level
        "FVG_SIGNAL": 1.0,
        "OB_STRENGTH_NORM": 0.8,
        "BOS_RETEST_ARMED": 1.0,
        "ATR": 5.0,
    }
    cs = _MockConfluenceSignal("LONG")
    s = confluence_signal_to_structure(cs, smc_features, bar_ts, current_price=2378.45)

    assert s.bos is not None
    assert s.bos.direction == "bullish"
    assert s.bos.validation_status == "confirmed"
    assert s.bos.level == 2375.20  # F1: published BOS_BREAK_LEVEL, not current_price
    assert s.choch is None  # CHOCH_SIGNAL=0
    assert len(s.order_blocks) == 1
    assert s.order_blocks[0].direction == "bullish"
    assert s.order_blocks[0].importance == "high"
    assert s.order_blocks[0].status == "active"
    assert len(s.fair_value_gaps) == 1
    assert s.fair_value_gaps[0].direction == "bullish"
    assert s.retest_in_progress is not None
    assert s.retest_in_progress.type == "bos_retest"


def test_confluence_signal_to_structure_with_short_signal(bar_ts):
    smc_features = {
        "BOS_SIGNAL": -1.0,
        "BOS_EVENT": -1.0,  # fresh bearish break
        "BOS_BREAK_LEVEL": 1.0900,
        "FVG_SIGNAL": -1.0,
        "OB_STRENGTH_NORM": 0.45,  # medium
        "ATR": 4.0,
    }
    cs = _MockConfluenceSignal("SHORT")
    s = confluence_signal_to_structure(cs, smc_features, bar_ts, current_price=1.0820)

    assert s.bos.direction == "bearish"
    assert s.bos.validation_status == "confirmed"
    assert s.fair_value_gaps[0].direction == "bearish"
    assert s.order_blocks[0].importance == "medium"
    assert s.order_blocks[0].direction == "bearish"


def test_propagated_bos_without_event_is_not_shown(bar_ts):
    """F6: a propagated BOS_SIGNAL with no fresh BOS_EVENT does NOT emit a bos
    object nor a bos_recent tag (it is trend state, not a recent break). The
    OB direction still falls back to the propagated trend direction."""
    smc_features = {
        "BOS_SIGNAL": -1.0,
        "BOS_EVENT": 0.0,            # propagated only — NOT fresh
        "OB_STRENGTH_NORM": 0.5,
        "ATR": 4.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=1.0820)
    assert s.bos is None
    assert s.order_blocks[0].direction == "bearish"  # OB still uses trend dir


def test_confluence_signal_to_structure_no_signal_but_bos_in_features(bar_ts):
    """F6: propagating BOS state (no fresh event) does not surface as a recent BOS."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2380.0)

    assert s.bos is None  # F6: not a fresh break
    assert s.order_blocks == []  # No OB strength
    assert s.fair_value_gaps == []
    assert s.retest_in_progress is None


def test_retest_uses_break_level_last_without_fresh_bos(bar_ts):
    """F1+F6: a retest is armed bars after the break (no fresh BOS event). It is
    decoupled from the bos object and sources its level from the forward-filled
    real broken level, never current_price."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,                 # no fresh break on this bar
        "BOS_RETEST_ARMED": 1.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,  # forward-filled real level from pipeline
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is None                       # F6: no fresh break
    assert s.retest_in_progress is not None    # F6: retest decoupled from bos
    assert s.retest_in_progress.level == 2361.10  # NOT current_price (2390.0)


def test_bos_level_falls_back_to_current_price_when_no_break_level(bar_ts):
    """F1: last-resort fallback when a fresh break has no level available."""
    s = confluence_signal_to_structure(
        None, {"BOS_SIGNAL": -1.0, "BOS_EVENT": -1.0, "ATR": 3.0},
        bar_ts, current_price=1.0820,
    )
    assert s.bos is not None
    assert s.bos.level == 1.0820


def test_realized_levels_forward_fills_bos_break_level():
    """F1: realized_levels carries the last non-NaN BOS_BREAK_LEVEL forward."""
    import numpy as np
    import pandas as pd

    from src.intelligence.market_reading_mappers import realized_levels

    df = pd.DataFrame({
        "high": [10.0, 11.0, 12.0, 13.0],
        "low": [9.0, 9.5, 10.0, 11.0],
        "BOS_BREAK_LEVEL": [np.nan, 10.5, np.nan, np.nan],
    })
    out = realized_levels(df, idx=3)
    assert out["BOS_BREAK_LEVEL_LAST"] == 10.5


def test_choch_level_uses_break_level(bar_ts):
    """F2: CHOCH publishes the real broken level (BOS_BREAK_LEVEL on the CHOCH
    bar, since CHOCH == reversal BOS same bar), not current_price."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 1.0,
        "CHOCH_SIGNAL": 1.0,
        "BOS_BREAK_LEVEL": 2350.75,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2380.0)
    assert s.choch is not None
    assert s.choch.direction == "bullish"
    assert s.choch.level == 2350.75      # NOT current_price (2380.0)


def test_choch_level_falls_back_to_price_when_no_level(bar_ts):
    """F2: last-resort fallback when no break level available."""
    s = confluence_signal_to_structure(
        None, {"CHOCH_SIGNAL": -1.0, "ATR": 3.0}, bar_ts, current_price=1.0950
    )
    assert s.choch is not None
    assert s.choch.level == 1.0950


def test_ob_fvg_use_real_levels(bar_ts):
    """F3: OB and FVG publish the real zones/bounds, not price ± ATR/2 proxies."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "FVG_SIGNAL": 1.0,
        "OB_STRENGTH_NORM": 0.8,
        "OB_LEVEL_HIGH": 2372.0,
        "OB_LEVEL_LOW": 2368.0,
        "FVG_LEVEL_HIGH": 2381.0,
        "FVG_LEVEL_LOW": 2379.0,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2400.0)
    assert s.order_blocks[0].level_high == 2372.0
    assert s.order_blocks[0].level_low == 2368.0
    assert s.fair_value_gaps[0].level_high == 2381.0
    assert s.fair_value_gaps[0].level_low == 2379.0


def test_ob_fvg_fall_back_to_proxy_without_real_levels(bar_ts):
    """F3: legacy proxy preserved when real levels are absent (backward compat)."""
    smc_features = {"FVG_SIGNAL": 1.0, "OB_STRENGTH_NORM": 0.8, "ATR": 4.0}
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=100.0)
    # half = ATR/2 = 2.0
    assert s.order_blocks[0].level_high == 102.0
    assert s.order_blocks[0].level_low == 98.0
    assert s.fair_value_gaps[0].level_high == 102.0
    assert s.fair_value_gaps[0].level_low == 98.0


def test_realized_levels_ob_and_fvg():
    """F3: realized_levels extracts the real OB zone and FVG bounds."""
    import numpy as np
    import pandas as pd

    from src.intelligence.market_reading_mappers import realized_levels

    # Bullish FVG on last bar (idx=4): low[i]=12.0 > high[i-2]=high[2]=11.0
    # → gap zone [11.0, 12.0]
    df = pd.DataFrame({
        "high": [10.0, 10.5, 11.0, 11.5, 13.0],
        "low":  [9.0, 9.5, 10.0, 11.0, 12.0],
        "BULLISH_OB_HIGH": [np.nan, np.nan, np.nan, np.nan, 11.8],
        "BULLISH_OB_LOW":  [np.nan, np.nan, np.nan, np.nan, 11.2],
        "BEARISH_OB_HIGH": [np.nan] * 5,
        "BEARISH_OB_LOW":  [np.nan] * 5,
        "FVG_DIR": [0.0, 0.0, 0.0, 0.0, 1.0],
    })
    out = realized_levels(df, idx=4)
    assert out["OB_LEVEL_HIGH"] == 11.8
    assert out["OB_LEVEL_LOW"] == 11.2
    assert out["FVG_LEVEL_HIGH"] == 12.0   # low[i]
    assert out["FVG_LEVEL_LOW"] == 11.0    # high[i-2] = high[2]


def test_confluence_signal_to_structure_empty_features(bar_ts):
    """No structural signals at all — all-None structure is still valid."""
    s = confluence_signal_to_structure(None, {}, bar_ts, current_price=2378.45)
    assert s.bos is None
    assert s.choch is None
    assert s.order_blocks == []
    assert s.fair_value_gaps == []
    assert s.retest_in_progress is None


# ---------------------------------------------------------------------------
# Regime mapper
# ---------------------------------------------------------------------------


def _candles_uptrend(n: int = 30) -> list[dict]:
    base = 2300.0
    out = []
    for i in range(n):
        close = base + i * 2.5
        out.append({
            "open": close - 1.0,
            "high": close + 1.5,
            "low": close - 1.5,
            "close": close,
        })
    return out


def _candles_flat(n: int = 30) -> list[dict]:
    """Closes oscillate symmetrically around base — first and last identical."""
    base = 2300.0
    out = []
    for i in range(n):
        if i == 0 or i == n - 1:
            close = base
        elif i % 2 == 0:
            close = base + 5.0
        else:
            close = base - 5.0
        out.append({"open": close, "high": close + 0.5, "low": close - 0.5, "close": close})
    return out


def _candles_volatile(n: int = 30) -> list[dict]:
    base = 2300.0
    out = []
    for i in range(n):
        spike = 8.0 if i >= n - 7 else 1.0
        close = base + (i * 0.5)
        out.append({
            "open": close,
            "high": close + spike,
            "low": close - spike,
            "close": close,
        })
    return out


def test_candles_to_regime_bullish_trend():
    candles = _candles_uptrend()
    r = candles_to_regime(candles, mtf_candles_above={})
    assert r.trend == "bullish"
    assert r.market_phase in ("trend", "expansion")
    assert r.mtf_confluence == {}


def test_candles_to_regime_ranging():
    candles = _candles_flat()
    r = candles_to_regime(candles, mtf_candles_above={})
    assert r.trend in ("ranging", "neutral")
    assert r.market_phase in ("ranging", "accumulation")


def test_candles_to_regime_elevated_volatility():
    candles = _candles_volatile()
    r = candles_to_regime(candles, mtf_candles_above={})
    assert r.volatility_observed == "elevated"


def test_candles_to_regime_mtf_confluence_keys_filtered():
    candles = _candles_uptrend()
    mtf = {
        "h1": _candles_uptrend(),
        "h4": _candles_uptrend(),
        "INVALID_KEY": _candles_uptrend(),  # should be silently dropped
    }
    r = candles_to_regime(candles, mtf_candles_above=mtf)
    assert "h1" in r.mtf_confluence
    assert "h4" in r.mtf_confluence
    assert "INVALID_KEY" not in r.mtf_confluence
    assert r.mtf_confluence["h1"] == "bullish"
    assert r.mtf_confluence["h4"] == "bullish"


# ---------------------------------------------------------------------------
# Events stub
# ---------------------------------------------------------------------------


def test_empty_events():
    e = empty_events()
    assert isinstance(e, MarketReadingEvents)
    assert e.news_upcoming == []
    assert e.news_just_published == []
    assert e.technical_triggers_recent == []


# ---------------------------------------------------------------------------
# Tags + description
# ---------------------------------------------------------------------------


def _structure_rich(bar_ts: datetime) -> MarketReadingStructure:
    return confluence_signal_to_structure(
        _MockConfluenceSignal("LONG"),
        {
            "BOS_SIGNAL": 1.0,
            "BOS_EVENT": 1.0,
            "FVG_SIGNAL": 1.0,
            "OB_STRENGTH_NORM": 0.8,
            "BOS_RETEST_ARMED": 1.0,
            "ATR": 5.0,
        },
        bar_ts,
        current_price=2378.45,
    )


def _regime_bull_aligned() -> MarketReadingRegime:
    return MarketReadingRegime(
        trend="bullish",
        volatility_observed="elevated",
        market_phase="expansion",
        mtf_confluence={"h1": "bullish", "h4": "bullish"},
    )


def test_tags_and_description_basic(bar_ts):
    structure = _structure_rich(bar_ts)
    regime = _regime_bull_aligned()
    tags, description = tags_and_description(structure, regime)

    # Tags coverage
    assert "trend_bullish" in tags
    assert "volatility_elevated" in tags
    assert "phase_expansion" in tags
    assert "bos_recent_bullish" in tags
    assert "retest_in_progress" in tags
    assert "ob_active" in tags
    assert "fvg_active" in tags
    assert "mtf_aligned" in tags

    # Description shape
    assert isinstance(description, str)
    assert len(description) > 0
    assert len(description) <= DESCRIPTION_MAX_LENGTH


def test_description_no_forbidden_tokens_across_many_combinations(bar_ts):
    """CRITICAL niveau 1.5 strict test: no template path emits any forbidden token."""
    regimes = [
        MarketReadingRegime(trend=t, volatility_observed=v, market_phase=p,
                            mtf_confluence=mtf)
        for t in ("bullish", "bearish", "neutral", "ranging")
        for v in ("low", "normal", "elevated")
        for p in ("accumulation", "distribution", "trend", "ranging", "expansion")
        for mtf in ({}, {"h1": "bullish"}, {"h1": "bullish", "h4": "bearish"},
                    {"h1": "bullish", "h4": "bullish"})
    ]
    structures = [
        MarketReadingStructure(),
        _structure_rich(bar_ts),
        confluence_signal_to_structure(
            None,
            {"BOS_SIGNAL": -1.0, "ATR": 3.0},
            bar_ts,
            current_price=1.0820,
        ),
    ]
    seen_combos = 0
    for regime in regimes:
        for structure in structures:
            _, description = tags_and_description(structure, regime)
            found = contains_forbidden_tokens(description)
            assert found is None, (
                f"Forbidden token '{found}' found in description: {description!r}"
            )
            seen_combos += 1
    assert seen_combos == len(regimes) * len(structures)


def test_description_max_280_chars_under_max_combination(bar_ts):
    """Even maximally verbose combination must respect the 280-char cap."""
    structure = _structure_rich(bar_ts)
    regime = MarketReadingRegime(
        trend="bullish",
        volatility_observed="elevated",
        market_phase="expansion",
        mtf_confluence={"h1": "bullish", "h4": "bullish", "d1": "bullish", "w1": "bullish"},
    )
    _, description = tags_and_description(structure, regime)
    assert len(description) <= DESCRIPTION_MAX_LENGTH


def test_contains_forbidden_tokens_detector_positive_and_negative():
    """Sanity check on the detector itself — flags trade vocab, ignores neutrals."""
    # Positive: each phrase contains exactly one forbidden token as a whole word.
    assert contains_forbidden_tokens("Conseille cette zone") == "conseille"
    assert contains_forbidden_tokens("Évite ce niveau") == "évite"
    assert contains_forbidden_tokens("Bon moment pour agir") == "bon moment"
    assert contains_forbidden_tokens("Achète maintenant") == "achète"
    # Negative: clean descriptive text with no trade recommendation vocabulary.
    assert contains_forbidden_tokens("Tendance haussière, volatilité élevée") is None
    assert contains_forbidden_tokens("Structure alignée H1 et H4") is None
    assert contains_forbidden_tokens("BOS confirmé, retest en cours") is None
    # Word-boundary check: "entre" matches but "entrer" does not (different word).
    assert contains_forbidden_tokens("entre support et résistance") == "entre"
    assert contains_forbidden_tokens("le prix peut entrer dans la zone") is None
    # "bon moment" matches but "bon momentum" does not.
    assert contains_forbidden_tokens("le momentum est bon") is None


def test_forbidden_tokens_set_is_immutable():
    """frozenset guarantees we can't accidentally mutate the canonical list."""
    assert isinstance(FORBIDDEN_TOKENS, frozenset)
    with pytest.raises(AttributeError):
        FORBIDDEN_TOKENS.add("new_token")  # type: ignore[attr-defined]
