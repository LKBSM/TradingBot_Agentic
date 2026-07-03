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
    # Armed retest of a prior bullish break (BOS_RETEST_STATE = +2). The break is
    # persisted (state != 0) AND a retest is in progress (state == ±2).
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 2.0,     # armed → break persisted + retest in progress
        "BOS_BREAK_LEVEL": 2375.20,  # F1: real broken structural level
        "FVG_SIGNAL": 1.0,
        "OB_STRENGTH_NORM": 0.8,
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


def test_persisted_bos_during_active_retest_state(bar_ts):
    """D1-b (1a): a prior break with no fresh event but an ACTIVE retest state
    (BOS_RETEST_STATE != 0) is persisted — surfaced as a confirmed bos AND a
    retest in progress, both sourcing the forward-filled real broken level
    (never current_price)."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,                 # no fresh break on this bar
        "BOS_RETEST_STATE": 2.0,          # armed retest → break still active
        "BOS_RETEST_ARMED": 1.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,  # forward-filled real level from pipeline
        "BOS_BREAK_TS": 1748352600.0,     # original break time (epoch seconds)
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is not None                       # 1a: persisted while active
    assert s.bos.direction == "bullish"
    assert s.bos.level == 2361.10                  # forward-filled real level
    assert s.bos.validation_status == "confirmed"
    assert s.bos.broken_at == datetime.fromtimestamp(1748352600.0, tz=timezone.utc)
    assert s.retest_in_progress is not None        # retest still surfaced
    assert s.retest_in_progress.level == 2361.10   # NOT current_price (2390.0)


def test_persisted_bos_awaiting_state_without_armed_retest(bar_ts):
    """1a: awaiting state (BOS_RETEST_STATE = ±1, price has not retested yet)
    persists the bos, but NO retest is in progress (only armed ±2 arms one)."""
    smc_features = {
        "BOS_SIGNAL": -1.0,
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": -1.0,         # awaiting (bearish)
        "BOS_BREAK_LEVEL_LAST": 1.0850,
        "ATR": 4.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=1.0820)
    assert s.bos is not None
    assert s.bos.direction == "bearish"
    assert s.bos.level == 1.0850
    assert s.retest_in_progress is None            # not armed → no retest surfaced


def test_retest_flag_only_during_armed_state_not_awaiting(bar_ts):
    """D1-b separation: the BOS LEVEL persists across the whole active window
    (state != 0), but the 'retest in progress' flag is shown ONLY while armed
    (state == ±2), never while awaiting (±1)."""
    base = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "ATR": 5.0,
    }
    # Awaiting (±1): break shown, NO retest in progress.
    awaiting = confluence_signal_to_structure(
        None, {**base, "BOS_RETEST_STATE": 1.0}, bar_ts, current_price=2390.0
    )
    assert awaiting.bos is not None
    assert awaiting.retest_in_progress is None

    # Armed (±2): break shown AND retest in progress.
    armed = confluence_signal_to_structure(
        None, {**base, "BOS_RETEST_STATE": 2.0}, bar_ts, current_price=2390.0
    )
    assert armed.bos is not None
    assert armed.retest_in_progress is not None
    assert armed.retest_in_progress.type == "bos_retest"


def test_retest_not_shown_when_break_dropped_by_inverted_trend(bar_ts):
    """Consistency: if the propagated trend inverted against an armed retest
    state, the break is NOT persisted, and the retest flag is suppressed too —
    the UI never shows a retest of a dropped break."""
    smc_features = {
        "BOS_SIGNAL": -1.0,               # trend now bearish…
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 2.0,          # …stale bullish armed state
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is None                  # inverted → break dropped
    assert s.retest_in_progress is None   # …and so is its retest


def test_persisted_bos_broken_at_falls_back_to_bar_ts_without_break_ts(bar_ts):
    """1a: when the glue field BOS_BREAK_TS is absent, broken_at falls back to
    the current bar (never crashes, never invents a time)."""
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 1.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is not None
    assert s.bos.broken_at == bar_ts


def test_bos_disappears_on_invalidation(bar_ts):
    """1a: once the engine's retest state machine clears to 0 (invalidation /
    reclaim / timeout), the break is no longer surfaced — it disappears."""
    smc_features = {
        "BOS_SIGNAL": 1.0,                # trend may still propagate…
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 0.0,          # …but the break is no longer active
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is None                  # disappeared on invalidation


def test_bos_not_persisted_when_trend_inverted_against_break(bar_ts):
    """1a: 'BOS_SIGNAL n'est pas inversé' — if the propagated trend has flipped
    opposite to the (stale) retest-state direction, the break is not surfaced."""
    smc_features = {
        "BOS_SIGNAL": -1.0,               # trend now bearish…
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 2.0,          # …stale bullish retest state
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is None                  # inverted trend → not persisted


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


def test_realized_levels_emits_break_ts_from_last_event(bar_ts):
    """1a glue: realized_levels carries the ORIGINAL break timestamp (last bar
    with BOS_EVENT != 0) so a persisted break reports an honest broken_at. Only
    when the frame has a DatetimeIndex (production), never on integer indices."""
    import numpy as np
    import pandas as pd

    from src.intelligence.market_reading_mappers import realized_levels

    idx = pd.to_datetime([
        "2026-05-28T13:00:00Z",
        "2026-05-28T13:15:00Z",  # the break bar
        "2026-05-28T13:30:00Z",
        "2026-05-28T13:45:00Z",
    ])
    df = pd.DataFrame(
        {
            "high": [10.0, 11.0, 12.0, 13.0],
            "low": [9.0, 9.5, 10.0, 11.0],
            "BOS_EVENT": [0.0, 1.0, 0.0, 0.0],
        },
        index=idx,
    )
    out = realized_levels(df, idx=3)
    assert out["BOS_BREAK_TS"] == pd.Timestamp("2026-05-28T13:15:00Z").timestamp()

    # Integer-indexed frame → no bogus timestamp emitted.
    df_int = pd.DataFrame({"high": [1.0, 2.0], "low": [0.5, 1.5], "BOS_EVENT": [0.0, 1.0]})
    assert "BOS_BREAK_TS" not in realized_levels(df_int, idx=1)


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
            "BOS_RETEST_STATE": 2.0,  # armed → retest in progress surfaced
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
    # P4: bare "entre" (preposition "between") is NOT forbidden (homonym), but the
    # directive forms ARE. This mirrors chatbot/constants.py and removes the
    # unjustified template fallbacks on descriptive output like "FVG entre X et Y".
    assert contains_forbidden_tokens("FVG entre 2376 et 2378") is None
    assert contains_forbidden_tokens("entre support et résistance") is None
    assert contains_forbidden_tokens("le prix peut entrer dans la zone") == "entrer"
    assert contains_forbidden_tokens("entrez maintenant") == "entrez"
    # "bon moment" matches but "bon momentum" does not.
    assert contains_forbidden_tokens("le momentum est bon") is None


def test_forbidden_tokens_set_is_immutable():
    """frozenset guarantees we can't accidentally mutate the canonical list."""
    assert isinstance(FORBIDDEN_TOKENS, frozenset)
    with pytest.raises(AttributeError):
        FORBIDDEN_TOKENS.add("new_token")  # type: ignore[attr-defined]


def test_persisted_bos_future_break_ts_falls_back_to_bar_ts(bar_ts):
    """Audit 2026-06-12 §T2: a BOS_BREAK_TS recovered from a wrong clock domain
    (candle index labelled in the future) must never surface — production
    readings published broken_at timestamps hours AFTER the reading itself.
    The mapper clamps: future recovered time → fall back to bar_ts."""
    future_epoch = bar_ts.timestamp() + 10 * 3600  # +10h, the observed offset
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 2.0,
        "BOS_RETEST_ARMED": 1.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "BOS_BREAK_TS": future_epoch,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is not None
    assert s.bos.broken_at == bar_ts          # clamped, never in the future
    assert s.bos.broken_at <= bar_ts


def test_persisted_bos_past_break_ts_still_honest(bar_ts):
    """Sanity counterpart: a legitimately past BOS_BREAK_TS is surfaced as-is."""
    past_epoch = bar_ts.timestamp() - 3 * 3600
    smc_features = {
        "BOS_SIGNAL": 1.0,
        "BOS_EVENT": 0.0,
        "BOS_RETEST_STATE": 1.0,
        "BOS_BREAK_LEVEL_LAST": 2361.10,
        "BOS_BREAK_TS": past_epoch,
        "ATR": 5.0,
    }
    s = confluence_signal_to_structure(None, smc_features, bar_ts, current_price=2390.0)
    assert s.bos is not None
    assert s.bos.broken_at == datetime.fromtimestamp(past_epoch, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Multi-zone registry (audit DETECTION_QUALITY_REVIEW_2026_06_12 §T1 fix)
# ---------------------------------------------------------------------------

import pandas as pd

from src.intelligence.market_reading_mappers import collect_zones


def _frame(rows: list[dict], start="2026-05-28T00:00:00Z") -> pd.DataFrame:
    """Build an enriched-like frame with a DatetimeIndex (M15 spacing)."""
    idx = pd.date_range(start=start, periods=len(rows), freq="15min", tz="UTC")
    cols = ["high", "low", "close", "BULLISH_OB_HIGH", "BULLISH_OB_LOW",
            "BEARISH_OB_HIGH", "BEARISH_OB_LOW", "OB_STRENGTH_NORM",
            "FVG_DIR", "FVG_SIZE_NORM"]
    data = {c: [r.get(c, float("nan")) for r in rows] for c in cols}
    return pd.DataFrame(data, index=idx)


def test_collect_zones_surfaces_multiple_obs():
    """The registry returns every still-active OB, not just the last bar's."""
    # Two bullish OBs far below price; price never returns → both stay active.
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(10)]
    rows[2].update(BULLISH_OB_HIGH=92.0, BULLISH_OB_LOW=90.0, OB_STRENGTH_NORM=0.8)
    rows[5].update(BULLISH_OB_HIGH=95.0, BULLISH_OB_LOW=94.0, OB_STRENGTH_NORM=0.5)
    z = collect_zones(_frame(rows), idx=9)
    obs = z["order_blocks"]
    assert len(obs) == 2
    assert {o["status"] for o in obs} == {"active"}
    # higher strength first
    assert obs[0]["level_low"] == 90.0
    assert obs[0]["importance"] == "high"
    assert obs[1]["importance"] == "medium"


def test_collect_zones_drops_invalidated_ob():
    """A bullish OB whose support is closed through is consumed → dropped."""
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    # Later bar closes below the OB low → invalidated.
    rows[4].update(high=98, low=95, close=96.0)
    z = collect_zones(_frame(rows), idx=5)
    assert z["order_blocks"] == []


def test_collect_zones_mitigated_ob_kept():
    """Price taps into the OB but holds (no close-through) → mitigated, kept."""
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[3].update(high=100, low=97.5, close=99.0)  # dips into zone, closes above
    z = collect_zones(_frame(rows), idx=5)
    assert len(z["order_blocks"]) == 1
    assert z["order_blocks"][0]["status"] == "mitigated"
    assert z["order_blocks"][0]["tested"] is True


def test_collect_zones_drops_filled_fvg_keeps_active():
    """A filled bullish FVG is dropped; an untouched one stays active."""
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    # Bullish gap at k=2: high[0]=100 (low edge), low[2]=101 (high edge) → [100,101].
    rows[2].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.5)
    # Bullish gap at k=5 way above; price never returns → active.
    rows[5].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.3)
    # Make an early bar fill the k=2 gap: a low reaching <= 100.
    rows[4].update(low=99.5)
    z = collect_zones(_frame(rows), idx=7)
    fvgs = z["fair_value_gaps"]
    statuses = {round(f["_size"], 2): f["status"] for f in fvgs}
    # k=2 gap filled (dropped), only k=5 remains active
    assert len(fvgs) == 1
    assert fvgs[0]["status"] == "active"


def test_collect_zones_caps_per_type():
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(20)]
    for k in range(2, 12):  # 10 bullish OBs below price → all active
        rows[k].update(BULLISH_OB_HIGH=50.0 - k, BULLISH_OB_LOW=49.0 - k,
                       OB_STRENGTH_NORM=0.5)
    z = collect_zones(_frame(rows), idx=19, max_per_type=6)
    assert len(z["order_blocks"]) == 6


def test_structure_mapper_uses_injected_zones(bar_ts):
    """confluence_signal_to_structure publishes the multi-zone list when present."""
    smc = {
        "BOS_SIGNAL": 0.0, "BOS_EVENT": 0.0, "ATR": 5.0,
        "_zones": {
            "order_blocks": [
                {"direction": "bullish", "level_high": 92.0, "level_low": 90.0,
                 "importance": "high", "status": "active", "tested": False,
                 "created_at": bar_ts},
                {"direction": "bearish", "level_high": 110.0, "level_low": 108.0,
                 "importance": "medium", "status": "mitigated", "tested": True,
                 "created_at": bar_ts},
            ],
            "fair_value_gaps": [
                {"direction": "bearish", "level_high": 105.0, "level_low": 104.0,
                 "status": "active", "tested": False, "created_at": bar_ts},
            ],
        },
    }
    s = confluence_signal_to_structure(None, smc, bar_ts, current_price=100.0)
    assert len(s.order_blocks) == 2
    assert s.order_blocks[0].direction == "bullish"
    assert s.order_blocks[1].status == "mitigated"
    assert len(s.fair_value_gaps) == 1
    assert s.fair_value_gaps[0].direction == "bearish"


def test_structure_mapper_falls_back_without_zones(bar_ts):
    """No _zones key → legacy single-bar behaviour is preserved."""
    smc = {"BOS_SIGNAL": 0.0, "OB_STRENGTH_NORM": 0.6, "ATR": 5.0,
           "OB_LEVEL_HIGH": 101.0, "OB_LEVEL_LOW": 99.0}
    s = confluence_signal_to_structure(None, smc, bar_ts, current_price=100.0)
    assert len(s.order_blocks) == 1  # single-bar fallback still works


# ---------------------------------------------------------------------------
# Mitigation lifecycle: mitigated_at bounding + single-source-of-truth policy
# (feat/ob-fvg-mitigation-lifecycle — defaults validated by founder 2026-06-15,
# DÉFAUTS À VALIDER PAR ANNOTATION)
# ---------------------------------------------------------------------------

from src.intelligence.market_reading_mappers import (
    MITIGATION_POLICY,
    MitigationPolicy,
    _fvg_lifecycle,
    _ob_lifecycle,
)


def test_active_ob_has_no_mitigation_timestamp():
    """An untouched OB is active and carries mitigated_at=None (box → current)."""
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(10)]
    rows[2].update(BULLISH_OB_HIGH=92.0, BULLISH_OB_LOW=90.0, OB_STRENGTH_NORM=0.8)
    z = collect_zones(_frame(rows), idx=9)
    ob = z["order_blocks"][0]
    assert ob["status"] == "active"
    assert ob["mitigated_at"] is None


def test_mitigated_ob_carries_first_tap_timestamp():
    """A tapped-but-held OB stays exposed (founder choice) with a bounded box:
    mitigated_at = the bar of the FIRST tap, so the front draws formation→tap."""
    frame_start = "2026-05-28T00:00:00Z"
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[3].update(high=100, low=97.5, close=99.0)  # first tap at k=3, holds
    z = collect_zones(_frame(rows, start=frame_start), idx=5)
    ob = z["order_blocks"][0]
    assert ob["status"] == "mitigated"
    # k=3 with 15-min spacing from 00:00 → 00:45.
    expected = pd.Timestamp(frame_start) + pd.Timedelta(minutes=15 * 3)
    assert ob["mitigated_at"] == expected.to_pydatetime()
    # The box is bounded: formation (k=1) strictly before mitigation (k=3).
    assert ob["created_at"] < ob["mitigated_at"]


def test_active_fvg_has_no_mitigation_timestamp():
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    rows[5].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.3)  # gap well above, untouched
    z = collect_zones(_frame(rows), idx=7)
    fvg = z["fair_value_gaps"][0]
    assert fvg["status"] == "active"
    assert fvg["mitigated_at"] is None


def test_partially_filled_fvg_carries_first_entry_timestamp():
    """Partial fill (near edge touched, far edge not reached) stays exposed with
    mitigated_at = first entry bar (founder: 100% strict before drop)."""
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    # Bullish gap at k=2: [high[0]=100, low[2]=101] → band [100,101], fill at <=100.
    rows[2].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.5)
    rows[4].update(low=100.5)  # dips below near edge (101) but not to 100 → partial
    z = collect_zones(_frame(rows), idx=7)
    fvg = z["fair_value_gaps"][0]
    assert fvg["status"] == "partially_filled"
    assert fvg["mitigated_at"] is not None


def test_active_fvg_has_no_fill_level():
    """An untouched gap exposes fill_level=None (front draws the full band)."""
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    rows[5].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.3)
    z = collect_zones(_frame(rows), idx=7)
    fvg = z["fair_value_gaps"][0]
    assert fvg["status"] == "active"
    assert fvg["fill_level"] is None


def test_partially_filled_fvg_exposes_deepest_penetration_as_fill_level():
    """A bullish gap fills from above: fill_level = the deepest low reached into
    the band (clamped), so the box shrinks to the still-open portion below it."""
    rows = [{"high": 100 + i, "low": 99 + i, "close": 100 + i} for i in range(8)]
    # Bullish gap at k=2: band [high[0]=100, low[2]=101] = [100,101]; full fill <=100.
    rows[2].update(FVG_DIR=1.0, FVG_SIZE_NORM=0.5)
    rows[4].update(low=100.5)  # dips to 100.5 (partial); deepest = 100.5
    rows[6].update(low=100.7)  # shallower later dip must NOT override the deepest
    z = collect_zones(_frame(rows), idx=7)
    fvg = z["fair_value_gaps"][0]
    assert fvg["status"] == "partially_filled"
    assert fvg["fill_level"] == 100.5
    # Still-open portion is below the penetration: strictly inside the band.
    assert fvg["level_low"] <= fvg["fill_level"] < fvg["level_high"]


def test_consumed_zones_are_never_exposed():
    """Honesty guardrail: invalidated OB and filled FVG are dropped entirely."""
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[4].update(high=98, low=95, close=96.0)  # closes through → invalidated
    z = collect_zones(_frame(rows), idx=5)
    assert z["order_blocks"] == []


def test_ob_lifecycle_returns_tap_index_and_is_conservative():
    """Direct unit on the single-source-of-truth helper: default penetration=0.0
    declares mitigation on the FIRST overlap (conservative — earlier, not later)."""
    highs = [100, 100, 100, 100, 100]
    lows = [99, 99, 99, 97.5, 99]   # j=3 dips into [97,98]
    closes = [100, 100, 100, 99, 100]
    status, tested, tap, invalidated_idx = _ob_lifecycle(
        "bullish", 98.0, 97.0, highs, lows, closes, created=0, upto=4
    )
    assert (status, tested, tap, invalidated_idx) == ("mitigated", True, 3, None)


def test_penetration_threshold_is_a_single_tunable_knob():
    """Raising ob_mitigation_penetration delays mitigation (less conservative).
    A shallow tap that mitigates under the default stays active under a deep one."""
    highs = [100, 100, 100, 100]
    lows = [99, 99, 97.9, 99]   # dips 0.1 into [97,98] (height 1.0) at j=2
    closes = [100, 100, 99, 100]
    # Default (0.0): any touch → mitigated.
    assert _ob_lifecycle("bullish", 98.0, 97.0, highs, lows, closes, 0, 3)[0] == "mitigated"
    # Require 50% penetration: 0.1 tap is too shallow → still active.
    deep = MitigationPolicy(ob_mitigation_penetration=0.5)
    assert _ob_lifecycle("bullish", 98.0, 97.0, highs, lows, closes, 0, 3, deep)[0] == "active"


def test_policy_drop_when_mitigated_removes_zone():
    """The conservative 'drop on mitigation' switch lives in the single policy."""
    rows = [{"high": 100, "low": 99, "close": 100} for _ in range(6)]
    rows[1].update(BULLISH_OB_HIGH=98.0, BULLISH_OB_LOW=97.0, OB_STRENGTH_NORM=0.9)
    rows[3].update(high=100, low=97.5, close=99.0)  # tap → mitigated
    import src.intelligence.market_reading_mappers as m
    original = m.MITIGATION_POLICY
    m.MITIGATION_POLICY = MitigationPolicy(ob_drop_when_mitigated=True)
    try:
        z = collect_zones(_frame(rows), idx=5)
        assert z["order_blocks"] == []  # mitigated zone dropped under strict policy
    finally:
        m.MITIGATION_POLICY = original


def test_default_policy_matches_founder_decision():
    """Founder 2026-06-15: keep mitigated OB / partial FVG visible, 100% fill."""
    assert MITIGATION_POLICY.ob_mitigation_penetration == 0.0
    assert MITIGATION_POLICY.ob_drop_when_mitigated is False
    assert MITIGATION_POLICY.fvg_fill_fraction == 1.0
    assert MITIGATION_POLICY.fvg_drop_when_partial is False


def test_order_block_schema_accepts_mitigated_at(bar_ts):
    """Schema round-trip: mitigated_at optional, defaults to None for active."""
    from src.intelligence.market_reading_schema import FairValueGap, OrderBlock

    ob = OrderBlock(
        id="OB_x", direction="bullish", level_high=92.0, level_low=90.0,
        importance="high", status="active", created_at=bar_ts, tested=False,
    )
    assert ob.mitigated_at is None
    ob2 = OrderBlock(
        id="OB_y", direction="bullish", level_high=92.0, level_low=90.0,
        importance="high", status="mitigated", created_at=bar_ts, tested=True,
        mitigated_at=bar_ts,
    )
    assert ob2.mitigated_at == bar_ts
    fvg = FairValueGap(
        id="FVG_z", direction="bullish", level_high=101.0, level_low=100.0,
        status="active", created_at=bar_ts, tested=False,
    )
    assert fvg.mitigated_at is None


# ---------------------------------------------------------------------------
# Structure events (BOS/CHOCH history) — collect_structure_events + mapper
# (fix sous-surfaçage 2026-06-16 : le moteur detecte beaucoup de breaks mais
# seul celui du dernier bar surfaçait via bos/choch)
# ---------------------------------------------------------------------------

from src.intelligence.market_reading_mappers import collect_structure_events


def _events_frame(rows: list[dict], start="2026-05-28T00:00:00Z") -> pd.DataFrame:
    """Frame carrying the engine event columns (BOS_EVENT/CHOCH_SIGNAL/level)."""
    idx = pd.date_range(start=start, periods=len(rows), freq="15min", tz="UTC")
    cols = ["close", "BOS_EVENT", "CHOCH_SIGNAL", "BOS_BREAK_LEVEL"]
    data = {c: [r.get(c, float("nan")) for r in rows] for c in cols}
    return pd.DataFrame(data, index=idx)


def test_collect_structure_events_surfaces_multiple_breaks_recent_first():
    """The whole window's breaks are collected, most-recent first — not just the
    last bar (the bug that surfaced ≤1 BOS/CHOCH)."""
    rows = [{"close": 100 + i} for i in range(10)]
    rows[2].update(BOS_EVENT=1.0, BOS_BREAK_LEVEL=102.0)   # bullish break at k=2
    rows[5].update(BOS_EVENT=-1.0, BOS_BREAK_LEVEL=105.0)  # bearish break at k=5
    rows[7].update(CHOCH_SIGNAL=1.0, BOS_BREAK_LEVEL=107.0)
    ev = collect_structure_events(_events_frame(rows), idx=9)
    assert [e["direction"] for e in ev["bos_events"]] == ["bearish", "bullish"]
    assert ev["bos_events"][0]["level"] == 105.0  # real broken level, recent first
    assert len(ev["choch_events"]) == 1
    assert ev["choch_events"][0]["level"] == 107.0


def test_collect_structure_events_caps_to_max_per_type():
    rows = [{"close": 100.0} for _ in range(12)]
    for k in range(10):  # 10 breaks, cap default = 8
        rows[k].update(BOS_EVENT=1.0, BOS_BREAK_LEVEL=100.0 + k)
    ev = collect_structure_events(_events_frame(rows), idx=11, max_per_type=8)
    assert len(ev["bos_events"]) == 8
    # Most recent kept: highest _k (k=9 → level 109) first.
    assert ev["bos_events"][0]["level"] == 109.0


def test_collect_structure_events_falls_back_to_close_without_break_level():
    rows = [{"close": 100 + i} for i in range(5)]
    rows[3].update(BOS_EVENT=1.0)  # no BOS_BREAK_LEVEL → close fallback
    ev = collect_structure_events(_events_frame(rows), idx=4)
    assert ev["bos_events"][0]["level"] == 103.0


def test_mapper_exposes_event_lists_when_collector_injected():
    """confluence_signal_to_structure publishes bos_events/choch_events from the
    injected _structure_events (read-only history)."""
    bar_ts = datetime(2026, 5, 28, tzinfo=timezone.utc)
    smc = {
        "_structure_events": {
            "bos_events": [
                {"direction": "bullish", "level": 102.0, "broken_at": bar_ts},
                {"direction": "bearish", "level": 105.0, "broken_at": bar_ts},
            ],
            "choch_events": [
                {"direction": "bullish", "level": 107.0, "broken_at": bar_ts},
            ],
        },
    }
    s = confluence_signal_to_structure(None, smc, bar_ts, current_price=100.0)
    assert len(s.bos_events) == 2
    assert len(s.choch_events) == 1
    assert s.bos_events[1].level == 105.0
    assert s.bos_events[0].validation_status == "confirmed"


def test_mapper_event_lists_default_empty_without_collector():
    """Backward compatible: no _structure_events → empty lists (not an error)."""
    bar_ts = datetime(2026, 5, 28, tzinfo=timezone.utc)
    s = confluence_signal_to_structure(None, {}, bar_ts, current_price=100.0)
    assert s.bos_events == []
    assert s.choch_events == []
