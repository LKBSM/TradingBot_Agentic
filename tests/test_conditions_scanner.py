"""Tests for the pure conditions evaluator (src/intelligence/conditions_scanner.py).

Covers, for the SC-1 four-family palette: (a) correct met/unmet/non-evaluable on
known data for every condition, (b) the palette offers only present-tense
structural conditions — no predictive/outcome type is representable, (c) a
non-evaluable condition adjusts the visible denominator and is counted neither
as met nor as unmet, (d) families and per-condition controls are present so the
interface derives from the schema, (e) blocked conditions are not exposed.
"""

from __future__ import annotations

from src.intelligence.conditions_scanner import (
    ALLOWED_CONDITION_TYPES,
    BLOCKED_PALETTE,
    FAMILIES,
    PALETTE,
    build_context_against,
    evaluate_condition,
    evaluate_reading,
)


def _reading(
    *,
    instrument="XAUUSD",
    timeframe="M15",
    close_price=2000.0,
    order_blocks=None,
    fair_value_gaps=None,
    liquidity_pools=None,
    bos=None,
    choch=None,
    bos_events=None,
    choch_events=None,
    candle_close_ts="2026-05-28T14:15:00+00:00",
    trend="bullish",
    market_phase="trend",
    volatility="normal",
    analysis_window_bars=500,
):
    return {
        "header": {
            "instrument": instrument,
            "timeframe": timeframe,
            "candle_close_ts": candle_close_ts,
            "close_price": close_price,
            "analysis_window_bars": analysis_window_bars,
        },
        "structure": {
            "bos": bos,
            "choch": choch,
            "bos_events": bos_events or [],
            "choch_events": choch_events or [],
            "order_blocks": order_blocks or [],
            "fair_value_gaps": fair_value_gaps or [],
            "liquidity_pools": liquidity_pools or [],
        },
        "regime": {
            "trend": trend,
            "volatility_observed": volatility,
            "market_phase": market_phase,
            "mtf_confluence": {},
        },
        "events": {"news_upcoming": [], "news_just_published": [], "technical_triggers_recent": []},
        "conditions": {"tags": [], "description": "", "description_source": "template_fallback"},
    }


def _ob(low, high, *, status="active", direction="bullish", tested=False, created_at="2026-05-28T12:00:00+00:00"):
    return {
        "id": "ob1", "direction": direction, "level_low": low, "level_high": high,
        "importance": "medium", "status": status, "created_at": created_at,
        "tested": tested, "user_flagged": False,
    }


def _fvg(low, high, *, status="active", direction="bullish", tested=False, created_at="2026-05-28T12:00:00+00:00"):
    return {
        "id": "fvg1", "direction": direction, "level_low": low, "level_high": high,
        "status": status, "created_at": created_at, "tested": tested, "user_flagged": False,
    }


def _bos(direction="bullish", *, validation_status="confirmed", broken_at="2026-05-28T13:45:00+00:00"):
    return {"direction": direction, "level": 1990.0, "broken_at": broken_at, "validation_status": validation_status}


def _liq(side, level, *, kind="range_extreme", status="intact", swept_at=None, broken_at=None, is_external=True):
    return {
        "id": f"liq_{side}_{int(level)}", "side": side, "kind": kind, "level": level,
        "touches": 2, "is_external": is_external, "status": status,
        "created_at": "2026-05-28T12:00:00+00:00", "swept_at": swept_at, "broken_at": broken_at,
    }


_ALIGNED_BULL = {"M15": "bullish", "H1": "bullish", "H4": "bullish"}
_ALIGNED_BEAR = {"M15": "bearish", "H1": "bearish", "H4": "bearish"}


# ── structure ────────────────────────────────────────────────────────────────


def test_trend_is_matches_structural_trend():
    r = _reading(trend="bullish")
    assert evaluate_condition(r, {"type": "trend_is", "trend": "bullish"})["met"] is True
    assert evaluate_condition(r, {"type": "trend_is", "trend": "bearish"})["met"] is False


def test_trend_is_indeterminate_supported():
    r = _reading(trend="indeterminate")
    assert evaluate_condition(r, {"type": "trend_is", "trend": "indeterminate"})["met"] is True


def test_higher_tf_agrees_same_when_higher_unit_matches():
    # M15 bullish, nearest higher unit (H1) bullish → « même sens ».
    res = evaluate_condition(_reading(trend="bullish"), {"type": "higher_tf_agrees", "relation": "same"}, _ALIGNED_BULL)
    assert res["met"] is True and res["available"] is True


def test_higher_tf_agrees_opposite():
    trends = {"M15": "bullish", "H1": "bearish", "H4": "bearish"}
    r = _reading(trend="bullish")
    assert evaluate_condition(r, {"type": "higher_tf_agrees", "relation": "opposite"}, trends)["met"] is True
    assert evaluate_condition(r, {"type": "higher_tf_agrees", "relation": "same"}, trends)["met"] is False


def test_higher_tf_agrees_indeterminate_higher_is_non_evaluable_and_distinct():
    # C1-b: an indeterminate HIGHER unit → NON-EVALUABLE (not a failure), with a
    # message DISTINCT from « no higher unit exists » (both must be tellable apart).
    r = _reading(trend="bullish")
    indet = evaluate_condition(r, {"type": "higher_tf_agrees", "relation": "same"}, {"M15": "bullish", "H1": "indeterminate"})
    assert indet["available"] is False and indet["met"] is False
    assert "établie" in indet["detail"]
    top = evaluate_condition(r, {"type": "higher_tf_agrees", "relation": "same"}, {"M15": "bullish"})
    assert top["available"] is False
    assert indet["detail"] != top["detail"]  # distinguishable on screen


def test_higher_tf_agrees_indeterminate_current_is_non_evaluable():
    # This unit itself indeterminate → non-evaluable too, with its own message.
    res = evaluate_condition(
        _reading(trend="indeterminate"),
        {"type": "higher_tf_agrees", "relation": "same"},
        {"M15": "indeterminate", "H1": "bullish"},
    )
    assert res["available"] is False and res["met"] is False


def test_higher_tf_agrees_non_evaluable_without_higher_unit():
    # Only the scanned unit is loaded → no higher unit → NON-EVALUABLE (C1-c),
    # never met by default nor a failure.
    res = evaluate_condition(_reading(trend="bullish"), {"type": "higher_tf_agrees", "relation": "same"}, {"M15": "bullish"})
    assert res["available"] is False and res["met"] is False


def test_higher_tf_agrees_names_the_compared_unit():
    # C1-b: the higher unit is NAMED in the result, not "unité supérieure".
    res = evaluate_condition(_reading(trend="bullish"), {"type": "higher_tf_agrees", "relation": "same"}, _ALIGNED_BULL)
    assert "1 h" in res["detail"] or "4 h" in res["detail"]


def test_last_event_is_uses_most_recent_journal_entry():
    r = _reading(
        bos_events=[{"direction": "bullish", "bars_ago": 12}],
        choch_events=[{"direction": "bearish", "bars_ago": 4}],
    )
    assert evaluate_condition(r, {"type": "last_event_is", "event": "choch_down"})["met"] is True
    assert evaluate_condition(r, {"type": "last_event_is", "event": "bos_up"})["met"] is False


def test_last_event_is_non_evaluable_when_no_events():
    res = evaluate_condition(_reading(), {"type": "last_event_is", "event": "bos_up"})
    assert res["available"] is False


def test_last_event_age_buckets():
    r = _reading(bos_events=[{"direction": "bullish", "bars_ago": 6}])
    assert evaluate_condition(r, {"type": "last_event_age", "age_bucket": "lt10"})["met"] is True
    assert evaluate_condition(r, {"type": "last_event_age", "age_bucket": "gt50"})["met"] is False
    r2 = _reading(bos_events=[{"direction": "bullish", "bars_ago": 80}])
    assert evaluate_condition(r2, {"type": "last_event_age", "age_bucket": "gt50"})["met"] is True


def test_bos_recent_confirmed_met_when_confirmed_and_recent():
    r = _reading(bos=_bos("bullish", broken_at="2026-05-28T13:45:00+00:00"))
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "max_bars": 5})["met"] is True


def test_bos_recent_confirmed_unmet_when_too_old():
    r = _reading(bos=_bos("bullish", broken_at="2026-05-28T12:00:00+00:00"))
    assert evaluate_condition(r, {"type": "bos_recent_confirmed", "max_bars": 5})["met"] is False


def test_choch_recent_confirmed_direction_filter():
    r = _reading(choch={"direction": "bearish", "level": 2010.0,
                        "broken_at": "2026-05-28T13:45:00+00:00", "validation_status": "confirmed"})
    assert evaluate_condition(r, {"type": "choch_recent_confirmed", "direction": "bearish", "max_bars": 5})["met"] is True
    assert evaluate_condition(r, {"type": "choch_recent_confirmed", "direction": "bullish", "max_bars": 5})["met"] is False


# ── zones ────────────────────────────────────────────────────────────────────


def test_price_in_ob_met_when_inside_active_ob():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    assert evaluate_condition(r, {"type": "price_in_ob", "direction": "any"})["met"] is True


def test_price_in_ob_unmet_when_mitigated():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010, status="mitigated")])
    assert evaluate_condition(r, {"type": "price_in_ob"})["met"] is False


def test_price_in_fvg_met_when_inside_open_fvg():
    r = _reading(close_price=2000.0, fair_value_gaps=[_fvg(1995, 2005)])
    assert evaluate_condition(r, {"type": "price_in_fvg"})["met"] is True


def test_price_in_tested_zone_met_only_when_tested():
    inside_tested = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010, tested=True)])
    inside_untested = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010, tested=False)])
    assert evaluate_condition(inside_tested, {"type": "price_in_tested_zone"})["met"] is True
    assert evaluate_condition(inside_untested, {"type": "price_in_tested_zone"})["met"] is False


def test_zone_untested_met_when_an_active_zone_never_tested():
    r = _reading(order_blocks=[_ob(1990, 2010, tested=False)])
    assert evaluate_condition(r, {"type": "zone_untested"})["met"] is True
    r2 = _reading(order_blocks=[_ob(1990, 2010, tested=True)])
    assert evaluate_condition(r2, {"type": "zone_untested"})["met"] is False


def test_zone_untested_is_at_price_with_kind_filter():
    # AT-PRICE: price 2000 inside a TESTED OB and inside an UNTESTED FVG.
    r = _reading(close_price=2000.0,
                 order_blocks=[_ob(1990, 2010, tested=True)],
                 fair_value_gaps=[_fvg(1995, 2005, tested=False)])
    assert evaluate_condition(r, {"type": "zone_untested", "zone_kind": "ob"})["met"] is False
    assert evaluate_condition(r, {"type": "zone_untested", "zone_kind": "fvg"})["met"] is True
    # A zone NOT at price does NOT count (the scope trap is gone).
    away = _reading(close_price=3000.0, fair_value_gaps=[_fvg(1995, 2005, tested=False)])
    assert evaluate_condition(away, {"type": "zone_untested"})["met"] is False


def test_zone_tested_at_most_at_price_1_to_N():
    ob = _ob(1990, 2010, tested=True)
    ob["touch_count"] = 2
    r = _reading(close_price=2000.0, order_blocks=[ob])
    assert evaluate_condition(r, {"type": "zone_tested_at_most", "max_touches": 2})["met"] is True
    assert evaluate_condition(r, {"type": "zone_tested_at_most", "max_touches": 1})["met"] is False  # 2 > 1
    # touch_count 0 (never tested) is #8's domain, never #9 (1 ≤ N).
    fresh = _ob(1990, 2010, tested=False)
    fresh["touch_count"] = 0
    r0 = _reading(close_price=2000.0, order_blocks=[fresh])
    assert evaluate_condition(r0, {"type": "zone_tested_at_most", "max_touches": 3})["met"] is False
    # No zone at price → unmet (at-price scope).
    away = _reading(close_price=3000.0, order_blocks=[ob])
    assert evaluate_condition(away, {"type": "zone_tested_at_most", "max_touches": 3})["met"] is False


def test_zone_formed_recent_is_at_price():
    # A recent zone NOT at price does not satisfy the (now at-price) condition.
    recent_away = _reading(close_price=3000.0, order_blocks=[_ob(1990, 2010, created_at="2026-05-28T13:45:00+00:00")])
    assert evaluate_condition(recent_away, {"type": "zone_formed_recent", "max_bars": 10})["met"] is False


def test_zone_formed_recent_counts_bars():
    # created 30 min before the 14:15 M15 close = 2 bars ≤ 10.
    r = _reading(order_blocks=[_ob(1990, 2010, created_at="2026-05-28T13:45:00+00:00")])
    assert evaluate_condition(r, {"type": "zone_formed_recent", "max_bars": 10})["met"] is True
    r2 = _reading(order_blocks=[_ob(1990, 2010, created_at="2026-05-27T10:00:00+00:00")])
    assert evaluate_condition(r2, {"type": "zone_formed_recent", "max_bars": 10})["met"] is False


def test_price_near_ob_excludes_inside():
    inside = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    # « near, WITHOUT being inside » → inside must NOT satisfy it.
    assert evaluate_condition(inside, {"type": "price_near_ob", "proximity_pct": 0.5})["met"] is False
    near = _reading(close_price=2000.0, order_blocks=[_ob(1990.0, 1995.0)])  # 0.25% away
    assert evaluate_condition(near, {"type": "price_near_ob", "proximity_pct": 0.3})["met"] is True


# ── liquidity ────────────────────────────────────────────────────────────────


def test_price_near_liquidity_met_for_intact_within_proximity():
    r = _reading(close_price=2000.0, liquidity_pools=[_liq("bsl", 2004.0)])
    assert evaluate_condition(r, {"type": "price_near_liquidity", "proximity_pct": 0.3})["met"] is True


def test_liquidity_swept_recent_met_within_window():
    r = _reading(liquidity_pools=[_liq("ssl", 1990.0, status="swept", swept_at="2026-05-28T13:45:00+00:00")])
    assert evaluate_condition(r, {"type": "liquidity_swept_recent", "max_bars": 10})["met"] is True


def test_liquidity_broken_recent_distinct_from_swept():
    swept = _reading(liquidity_pools=[_liq("ssl", 1990.0, status="swept", swept_at="2026-05-28T13:45:00+00:00")])
    broken = _reading(liquidity_pools=[_liq("ssl", 1990.0, status="broken", broken_at="2026-05-28T13:45:00+00:00")])
    # A swept pocket does not satisfy "broken", and vice-versa.
    assert evaluate_condition(swept, {"type": "liquidity_broken_recent", "max_bars": 10})["met"] is False
    assert evaluate_condition(broken, {"type": "liquidity_broken_recent", "max_bars": 10})["met"] is True


def test_equal_levels_present_kind_filter():
    r = _reading(liquidity_pools=[_liq("bsl", 2004.0, kind="equal_highs")])
    assert evaluate_condition(r, {"type": "equal_levels_present", "eq_kind": "highs"})["met"] is True
    assert evaluate_condition(r, {"type": "equal_levels_present", "eq_kind": "lows"})["met"] is False
    assert evaluate_condition(r, {"type": "equal_levels_present", "eq_kind": "any"})["met"] is True


# ── context ──────────────────────────────────────────────────────────────────


def test_market_phase_is_matches_observed():
    r = _reading(market_phase="expansion")
    assert evaluate_condition(r, {"type": "market_phase_is", "phase": "expansion"})["met"] is True
    assert evaluate_condition(r, {"type": "market_phase_is", "phase": "ranging"})["met"] is False


def test_volatility_is_matches_observed():
    r = _reading(volatility="elevated")
    assert evaluate_condition(r, {"type": "volatility_is", "volatility": "elevated"})["met"] is True
    assert evaluate_condition(r, {"type": "volatility_is", "volatility": "low"})["met"] is False


def test_price_in_range_third_uses_structural_range():
    # Structural range [1900, 2100] from explicit range_high/range_low pools; price
    # 1950 → 25% → bottom third.
    pools = [_liq("bsl", 2100.0, kind="range_high"), _liq("ssl", 1900.0, kind="range_low")]
    r = _reading(close_price=1950.0, liquidity_pools=pools)
    assert evaluate_condition(r, {"type": "price_in_range_third", "third": "bottom"})["met"] is True
    assert evaluate_condition(r, {"type": "price_in_range_third", "third": "top"})["met"] is False


def test_price_in_range_third_non_evaluable_without_range():
    res = evaluate_condition(_reading(liquidity_pools=[]), {"type": "price_in_range_third", "third": "bottom"})
    assert res["available"] is False


def test_session_is_computes_from_candle_close():
    # 2026-05-28 14:15 UTC = 10:15 America/New_York → inside London∩NY overlap.
    r = _reading(instrument="XAUUSD", candle_close_ts="2026-05-28T14:15:00+00:00")
    assert evaluate_condition(r, {"type": "session_is", "session": "overlap"})["met"] is True
    assert evaluate_condition(r, {"type": "session_is", "session": "asia"})["met"] is False


def test_session_is_non_evaluable_for_continuous_market():
    r = _reading(instrument="BTCUSD")
    assert evaluate_condition(r, {"type": "session_is", "session": "london"})["available"] is False


# ── evaluate_reading: logic + non-evaluable denominator ──────────────────────


def test_and_logic_full_match():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    out = evaluate_reading(
        r, [{"type": "higher_tf_agrees", "relation": "same"}, {"type": "price_in_ob"}], "AND", _ALIGNED_BULL,
    )
    assert out["matched"] is True and out["met_count"] == 2 and out["total"] == 2
    assert out["conditions_unmet"] == []


def test_non_evaluable_adjusts_denominator_not_counted_as_unmet():
    # price_in_ob met; last_event_is non-evaluable (no events). Denominator = 1.
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    out = evaluate_reading(
        r, [{"type": "price_in_ob"}, {"type": "last_event_is", "event": "bos_up"}], "AND",
    )
    assert out["total"] == 1  # only the evaluable condition counts
    assert out["met_count"] == 1
    assert out["non_evaluable_count"] == 1
    assert len(out["conditions_unmet"]) == 0  # the non-evaluable one is NOT unmet
    assert len(out["conditions_non_evaluable"]) == 1
    assert out["matched"] is True  # every evaluable condition met


def test_and_logic_not_matched_when_all_non_evaluable():
    # No evaluable condition → AND cannot match (never "all markets").
    r = _reading()
    out = evaluate_reading(r, [{"type": "last_event_is", "event": "bos_up"}], "AND")
    assert out["total"] == 0 and out["matched"] is False


def test_or_logic_matches_on_any_evaluable():
    r = _reading(close_price=2000.0, order_blocks=[_ob(1990, 2010)])
    out = evaluate_reading(r, [{"type": "price_in_ob"}, {"type": "price_in_fvg"}], "OR", _ALIGNED_BULL)
    assert out["matched"] is True and out["met_count"] == 1


# ── context-against (« ce qui va à l'encontre », enriched) ───────────────────


def test_context_against_multi_unit_disagreement():
    # M15 bearish, H1 bullish → the higher unit disagrees → an against item.
    r = _reading(timeframe="M15", trend="bearish")
    items = build_context_against(r, {"M15": "bearish", "H1": "bullish", "H4": "bullish"})
    assert any("désaccord multi-unités" in it["detail"] for it in items)
    assert any("1 h" in it["label"] or "4 h" in it["label"] for it in items)


def test_context_against_contracted_volatility():
    r = _reading(volatility="low")
    items = build_context_against(r, {})
    assert any("contractée" in it["label"] for it in items)


def test_context_against_surfaced_even_on_full_match():
    # A combo can fully match AND still carry against-signals (multi-unit
    # disagreement) — the « à l'encontre » block is never empty by construction
    # of a match.
    r = _reading(timeframe="M15", trend="bearish", order_blocks=[_ob(1990, 2010)])
    out = evaluate_reading(
        r, [{"type": "price_in_ob"}], "AND", {"M15": "bearish", "H1": "bullish"},
    )
    assert out["matched"] is True
    assert len(out["context_against"]) >= 1


def test_context_against_empty_when_nothing_opposes():
    r = _reading(timeframe="M15", trend="bullish", volatility="normal")
    items = build_context_against(r, {"M15": "bullish", "H1": "bullish"})
    assert items == []


# ── palette invariants ───────────────────────────────────────────────────────


def test_palette_types_match_allowlist():
    assert {p["type"] for p in PALETTE} == set(ALLOWED_CONDITION_TYPES)


def test_every_palette_entry_present_tense_with_family_and_controls():
    for entry in PALETTE:
        assert entry["tense"] == "present", entry["type"]
        assert entry["family"] in FAMILIES, entry["type"]
        assert "controls" in entry and isinstance(entry["controls"], list), entry["type"]
        for ctrl in entry["controls"]:
            assert ctrl["name"] and ctrl["values"] and "default" in ctrl


def test_removed_and_blocked_types_are_not_exposed():
    types = {p["type"] for p in PALETTE}
    # Removed: the redundant confluence, the legacy BOS-level retest, and the
    # fixed 3-TF alignment (replaced by the relative higher_tf_agrees, C1).
    assert "ob_fvg_confluence" not in types
    assert "retest_in_progress" not in types
    assert "mtf_aligned" not in types
    assert "higher_tf_agrees" in types
    # Blocked (documented but not offerable) must not be in the palette/allowlist.
    for b in BLOCKED_PALETTE:
        assert b["type"] not in types
        assert b["type"] not in ALLOWED_CONDITION_TYPES
        assert b.get("blocked_reason")


def test_market_phase_palette_excludes_unreachable_distribution():
    phase_entry = next(p for p in PALETTE if p["type"] == "market_phase_is")
    values = phase_entry["controls"][0]["values"]
    assert "distribution" not in values


def test_palette_has_no_predictive_vocabulary():
    import re

    # Whole-word checks: "range" (a real trading term) must NOT trip "rang"/rank.
    forbidden = [
        "rebond", "cassera", "va casser", "va rebondir", "prédi", "predict", "probab",
        "cible", "target", "gagnant", "prévision", "meilleur", "score", "rang", "continuera",
        "renvers", "idéal", "recommand", "opportun", "setup", "fort", "plus sûr", "qualité",
        "signal", "opportunité", "top",
    ]
    for entry in PALETTE:
        haystack = f"{entry['type']} {entry['label']} {entry['description']}".lower()
        for word in forbidden:
            assert re.search(rf"\b{re.escape(word)}\b", haystack) is None, (
                f"forbidden word '{word}' in palette entry {entry['type']}"
            )
