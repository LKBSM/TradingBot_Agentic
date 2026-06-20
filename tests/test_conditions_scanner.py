"""Tests for the pure conditions evaluator (src/intelligence/conditions_scanner.py).

Covers: (a) correct met/unmet on known data, (b) the palette offers only
present-tense structural conditions — no predictive/outcome type is representable.
"""

from __future__ import annotations

from src.intelligence.conditions_scanner import (
    ALLOWED_CONDITION_TYPES,
    PALETTE,
    evaluate_condition,
    evaluate_reading,
)


def _reading(
    *,
    instrument="XAUUSD",
    timeframe="M15",
    close_price=2000.0,
    mtf=None,
    order_blocks=None,
    fair_value_gaps=None,
    bos=None,
    candle_close_ts="2026-05-28T14:15:00+00:00",
    trend="bullish",
):
    return {
        "header": {
            "instrument": instrument,
            "timeframe": timeframe,
            "candle_close_ts": candle_close_ts,
            "close_price": close_price,
        },
        "structure": {
            "bos": bos,
            "choch": None,
            "order_blocks": order_blocks or [],
            "fair_value_gaps": fair_value_gaps or [],
        },
        "regime": {
            "trend": trend,
            "volatility_observed": "normal",
            "market_phase": "trend",
            "mtf_confluence": mtf or {},
        },
        "events": {"news_upcoming": [], "news_just_published": [], "technical_triggers_recent": []},
        "conditions": {"tags": [], "description": "", "description_source": "template_fallback"},
    }


def _ob(low, high, *, status="active", direction="bullish"):
    return {
        "id": "ob1",
        "direction": direction,
        "level_low": low,
        "level_high": high,
        "importance": "medium",
        "status": status,
        "created_at": "2026-05-28T12:00:00+00:00",
        "tested": False,
        "user_flagged": False,
    }


def _fvg(low, high, *, status="active", direction="bullish"):
    return {
        "id": "fvg1",
        "direction": direction,
        "level_low": low,
        "level_high": high,
        "status": status,
        "created_at": "2026-05-28T12:00:00+00:00",
        "tested": False,
        "user_flagged": False,
    }


def _bos(direction="bullish", *, validation_status="confirmed", broken_at="2026-05-28T13:45:00+00:00"):
    return {"direction": direction, "level": 1990.0, "broken_at": broken_at, "validation_status": validation_status}


# ── mtf_aligned ──────────────────────────────────────────────────────────────


def test_mtf_aligned_met_when_three_tf_same_direction():
    r = _reading(mtf={"h4": "bullish", "h1": "bullish", "m15": "bullish"})
    res = evaluate_condition(r, {"type": "mtf_aligned", "direction": "any"})
    assert res["met"] is True


def test_mtf_aligned_unmet_when_divergent():
    r = _reading(mtf={"h4": "bullish", "h1": "neutral", "m15": "bearish"})
    res = evaluate_condition(r, {"type": "mtf_aligned", "direction": "any"})
    assert res["met"] is False


def test_mtf_aligned_respects_requested_direction():
    r = _reading(mtf={"h4": "bearish", "h1": "bearish", "m15": "bearish"})
    assert evaluate_condition(r, {"type": "mtf_aligned", "direction": "bearish"})["met"] is True
    assert evaluate_condition(r, {"type": "mtf_aligned", "direction": "bullish"})["met"] is False


def test_mtf_aligned_unmet_when_incomplete():
    r = _reading(mtf={"h4": "bullish", "h1": "bullish"})  # m15 missing
    assert evaluate_condition(r, {"type": "mtf_aligned", "direction": "any"})["met"] is False


# ── price_in_ob / price_in_fvg ───────────────────────────────────────────────


def test_price_in_ob_met_when_inside_active_ob():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "any"})["met"] is True


def test_price_in_ob_unmet_when_outside():
    r = _reading(close_price=2050.0, order_blocks=[_ob(1990, 2010)])
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "any"})["met"] is False


def test_price_in_ob_unmet_when_ob_mitigated():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010, status="mitigated")])
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "any"})["met"] is False


def test_price_in_ob_direction_filter():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010, direction="bearish")])
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "bullish"})["met"] is False
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "bearish"})["met"] is True


def test_price_in_fvg_met_when_inside_open_fvg():
    r = _reading(close_price=2000.0, fair_value_gaps=[_fvg(1995, 2005)])
    assert evaluate_condition(r, {"type": "price_in_fvg", "direction": "any"})["met"] is True


def test_price_in_fvg_unmet_when_filled():
    r = _reading(close_price=2000.0, fair_value_gaps=[_fvg(1995, 2005, status="filled")])
    assert evaluate_condition(r, {"type": "price_in_fvg", "direction": "any"})["met"] is False


# ── ob_fvg_confluence ────────────────────────────────────────────────────────


def test_ob_fvg_confluence_met_when_both():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)], fair_value_gaps=[_fvg(1995, 2005)])
    assert evaluate_condition(r, {"type": "ob_fvg_confluence"})["met"] is True


def test_ob_fvg_confluence_unmet_with_only_ob():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    assert evaluate_condition(r, {"type": "ob_fvg_confluence"})["met"] is False


# ── bos_recent_confirmed ─────────────────────────────────────────────────────


def test_bos_recent_confirmed_met_when_confirmed_and_recent():
    # M15, broken 30 min before close → ~2 bars ≤ 5
    r = _reading(bos=_bos("bullish", broken_at="2026-05-28T13:45:00+00:00"))
    res = evaluate_condition(r, {"type": "bos_recent_confirmed", "direction": "any", "max_bars": 5})
    assert res["met"] is True


def test_bos_recent_confirmed_unmet_when_pending():
    r = _reading(bos=_bos("bullish", validation_status="pending"))
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "max_bars": 5})["met"] is False


def test_bos_recent_confirmed_unmet_when_too_old():
    # 135 min before close on M15 → 9 bars > 5
    r = _reading(bos=_bos("bullish", broken_at="2026-05-28T12:00:00+00:00"))
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "max_bars": 5})["met"] is False


def test_bos_recent_confirmed_direction_filter():
    r = _reading(bos=_bos("bearish"))
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "direction": "bullish"})["met"] is False
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "direction": "bearish"})["met"] is True


def test_bos_recent_confirmed_unmet_when_absent():
    r = _reading(bos=None)
    assert evaluate_condition(r, {"type": "bos_recent_confirmed"})["met"] is False


# ── trend_is / market_phase_is / volatility_is ──────────────────────────────


def test_trend_is_matches_observed_trend():
    r = _reading(trend="bullish")
    assert evaluate_condition(r, {"type": "trend_is", "trend": "bullish"})["met"] is True
    assert evaluate_condition(r, {"type": "trend_is", "trend": "bearish"})["met"] is False


def test_market_phase_is_matches_observed_phase():
    r = _reading()  # market_phase defaults to "trend" in helper
    assert evaluate_condition(r, {"type": "market_phase_is", "phase": "trend"})["met"] is True
    assert evaluate_condition(r, {"type": "market_phase_is", "phase": "range"})["met"] is False


def test_volatility_is_matches_observed_level():
    r = _reading()  # volatility_observed defaults to "normal"
    assert evaluate_condition(r, {"type": "volatility_is", "volatility": "normal"})["met"] is True
    assert evaluate_condition(r, {"type": "volatility_is", "volatility": "elevated"})["met"] is False


def test_regime_conditions_unmet_when_target_missing():
    r = _reading()
    assert evaluate_condition(r, {"type": "trend_is"})["met"] is False
    assert evaluate_condition(r, {"type": "market_phase_is"})["met"] is False
    assert evaluate_condition(r, {"type": "volatility_is"})["met"] is False


# ── choch_recent_confirmed ───────────────────────────────────────────────────


def test_choch_recent_confirmed_met_when_confirmed_and_recent():
    r = _reading()
    r["structure"]["choch"] = {
        "direction": "bearish",
        "level": 2010.0,
        "broken_at": "2026-05-28T13:45:00+00:00",
        "validation_status": "confirmed",
    }
    res = evaluate_condition(r, {"type": "choch_recent_confirmed", "direction": "bearish", "max_bars": 5})
    assert res["met"] is True


def test_choch_recent_confirmed_unmet_when_absent():
    r = _reading()
    assert evaluate_condition(r, {"type": "choch_recent_confirmed"})["met"] is False


# ── retest_in_progress ───────────────────────────────────────────────────────


def test_retest_in_progress_met_when_present():
    r = _reading()
    r["structure"]["retest_in_progress"] = {
        "level": 2000.0,
        "type": "ob_retest",
        "started_at": "2026-05-28T14:00:00+00:00",
    }
    assert evaluate_condition(r, {"type": "retest_in_progress"})["met"] is True


def test_retest_in_progress_unmet_when_absent():
    r = _reading()
    assert evaluate_condition(r, {"type": "retest_in_progress"})["met"] is False


# ── evaluate_reading (logic + context) ───────────────────────────────────────


def test_evaluate_reading_and_logic_full_match():
    r = _reading(
        mtf={"h4": "bullish", "h1": "bullish", "m15": "bullish"},
        close_price=2000.0,
        order_blocks=[_ob(1990, 2010)],
    )
    out = evaluate_reading(
        r,
        [{"type": "mtf_aligned", "direction": "any"}, {"type": "price_in_ob", "direction": "any"}],
        "AND",
    )
    assert out["matched"] is True
    assert out["met_count"] == 2
    assert out["total"] == 2
    assert out["conditions_unmet"] == []
    assert out["instrument"] == "XAUUSD" and out["timeframe"] == "M15"


def test_evaluate_reading_and_logic_partial_is_not_matched():
    r = _reading(mtf={"h4": "bullish", "h1": "bullish", "m15": "bullish"})  # no OB at price
    out = evaluate_reading(
        r,
        [{"type": "mtf_aligned"}, {"type": "price_in_ob"}],
        "AND",
    )
    assert out["matched"] is False
    assert out["met_count"] == 1
    assert {c["type"] for c in out["conditions_unmet"]} == {"price_in_ob"}


def test_evaluate_reading_or_logic_matches_on_any():
    r = _reading(mtf={"h4": "bullish", "h1": "bullish", "m15": "bullish"})
    out = evaluate_reading(r, [{"type": "mtf_aligned"}, {"type": "price_in_ob"}], "OR")
    assert out["matched"] is True
    assert out["met_count"] == 1


def test_context_includes_full_picture():
    r = _reading(
        mtf={"h4": "bullish", "h1": "bearish", "m15": "neutral"},
        order_blocks=[_ob(1990, 2010)],
        bos=_bos("bullish"),
    )
    out = evaluate_reading(r, [{"type": "mtf_aligned"}], "AND")
    ctx = out["context"]
    assert ctx["trend"] == "bullish"
    assert ctx["mtf_confluence"] == {"h4": "bullish", "h1": "bearish", "m15": "neutral"}
    assert ctx["active_order_blocks"] == 1
    assert ctx["bos"] is not None


# ── Palette: present-tense only, no predictive/outcome condition ──────────────


def test_palette_types_exactly_match_allowlist():
    assert {p["type"] for p in PALETTE} == set(ALLOWED_CONDITION_TYPES)
    assert len(PALETTE) == 10
    assert {p["type"] for p in PALETTE} == {
        "mtf_aligned",
        "trend_is",
        "market_phase_is",
        "volatility_is",
        "price_in_ob",
        "price_in_fvg",
        "ob_fvg_confluence",
        "bos_recent_confirmed",
        "choch_recent_confirmed",
        "retest_in_progress",
    }


def test_every_palette_entry_is_present_tense():
    for entry in PALETTE:
        assert entry["tense"] == "present", entry["type"]


def test_palette_has_no_predictive_vocabulary():
    # No condition may speak of a future outcome — descriptive present only.
    forbidden = [
        "rebond", "cassera", "va casser", "va rebondir", "prédi", "predict",
        "probab", "cible", "target", "setup gagnant", "gagnant", "prévision",
        "meilleur", "score", "continuera", "renvers",
    ]
    for entry in PALETTE:
        haystack = f"{entry['type']} {entry['label']} {entry['description']}".lower()
        for word in forbidden:
            assert word not in haystack, f"predictive word '{word}' in palette entry {entry['type']}"
