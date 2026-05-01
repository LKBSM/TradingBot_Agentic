"""Tests for the LLM eval harness (LLM-1.1).

Per DoD: 50 fixtures + script + CI. KPI: factual_consistency >= 0.75,
forbidden_phrases >= 0.95 (compliance), brevity >= 0.80.

Tests focus on:
  1. Fixture set integrity (50 entries, expected distribution)
  2. Each scoring axis behaves correctly on synthetic inputs
  3. Top-level harness produces a summary with the expected schema
  4. A baseline run with a deterministic stub-generator hits the KPI thresholds
"""

from __future__ import annotations

import pytest

from tests.eval_llm.eval_harness import (
    EvalHarness,
    flesch_kincaid_grade,
    heuristic_factual_consistency,
    score_brevity,
    score_factual_consistency,
    score_forbidden_phrases,
    score_reading_level,
    score_source_attribution,
)
from tests.eval_llm.fixtures import (
    ALL_FORBIDDEN_PHRASES,
    Fixture,
    FixtureExpected,
    build_fixtures,
)


# ---------------------------------------------------------------------------
# Fixture set integrity
# ---------------------------------------------------------------------------


def test_fixture_set_has_50_entries():
    fixtures = build_fixtures()
    assert len(fixtures) == 50


def test_fixture_set_distribution():
    """15 BUY high + 15 SELL high + 10 HOLD + 5 high vol + 5 news."""
    fixtures = build_fixtures()
    by_category: dict[str, int] = {}
    for f in fixtures:
        by_category[f.category] = by_category.get(f.category, 0) + 1
    assert by_category["BUY high conviction"] == 15
    assert by_category["SELL high conviction"] == 15
    assert by_category["HOLD"] == 10
    assert by_category["High volatility regime"] == 5
    assert by_category["News event nearby"] == 5


def test_fixture_ids_are_unique():
    fixtures = build_fixtures()
    ids = [f.id for f in fixtures]
    assert len(set(ids)) == len(ids)


# ---------------------------------------------------------------------------
# Reading-level scoring
# ---------------------------------------------------------------------------


def test_flesch_kincaid_handles_empty_input():
    assert flesch_kincaid_grade("") == 0.0


def test_flesch_kincaid_simple_text():
    """Simple short sentences ⇒ low grade level."""
    grade = flesch_kincaid_grade("The cat sat. The dog ran. They were friends.")
    assert grade < 5.0


def test_flesch_kincaid_complex_text():
    """Long polysyllabic sentences ⇒ higher grade."""
    grade = flesch_kincaid_grade(
        "The institutional confluence engine identified an asymmetric risk-adjusted "
        "opportunity through statistical anomaly detection across multiple "
        "macroeconomic indicators including the breakeven inflation expectations."
    )
    assert grade > 12.0


def test_score_reading_level_in_band():
    expected = {"min_grade_level": 8.0, "max_grade_level": 16.0}
    text = "Bullish setup detected at 2350. Risk frame: stop 2340 target 2370."
    score = score_reading_level(text, expected)
    assert 0.0 <= score.score <= 1.0


# ---------------------------------------------------------------------------
# Forbidden-phrase scoring (compliance MiFID II)
# ---------------------------------------------------------------------------


def test_score_forbidden_phrases_clean_text():
    text = "Bullish setup at 2350 with stop at 2340. Setup quality: strong."
    expected = {"forbidden_phrases": ALL_FORBIDDEN_PHRASES}
    s = score_forbidden_phrases(text, expected)
    assert s.score == 1.0


def test_score_forbidden_phrases_detects_buy_now():
    text = "Setup at 2350. Buy now for guaranteed profit."
    expected = {"forbidden_phrases": ALL_FORBIDDEN_PHRASES}
    s = score_forbidden_phrases(text, expected)
    assert s.score == 0.0
    assert "buy now" in s.detail or "guaranteed" in s.detail


def test_score_forbidden_phrases_detects_french_terms():
    text = "Achetez à 2350 avec un risque garanti contrôlé."
    expected = {"forbidden_phrases": ALL_FORBIDDEN_PHRASES}
    s = score_forbidden_phrases(text, expected)
    assert s.score == 0.0


# ---------------------------------------------------------------------------
# Source-attribution scoring
# ---------------------------------------------------------------------------


def test_score_source_attribution_full_match():
    text = (
        "Bullish setup at 2350 with stop at 2340 and target 2370. "
        "Components fired: BOS, OB, FVG."
    )
    expected = {
        "must_cite_entry": 2350.0,
        "must_cite_stop": 2340.0,
        "must_cite_target": 2370.0,
        "must_mention_components": ["BOS", "OB", "FVG"],
    }
    s = score_source_attribution(text, expected)
    assert s.score == 1.0


def test_score_source_attribution_partial():
    text = "Bullish setup at 2350. BOS confirmed."
    expected = {
        "must_cite_entry": 2350.0,
        "must_cite_stop": 2340.0,
        "must_cite_target": 2370.0,
        "must_mention_components": ["BOS", "OB", "FVG"],
    }
    s = score_source_attribution(text, expected)
    assert 0 < s.score < 1.0


def test_score_source_attribution_no_required_when_hold():
    """HOLD setups have no entry/SL/TP — score = 1.0 by definition."""
    expected = {
        "must_cite_entry": None,
        "must_cite_stop": None,
        "must_cite_target": None,
        "must_mention_components": [],
    }
    s = score_source_attribution("Neutral, no setup.", expected)
    assert s.score == 1.0


# ---------------------------------------------------------------------------
# Brevity scoring
# ---------------------------------------------------------------------------


def test_score_brevity_under_limit():
    s = score_brevity("Short.", {"max_chars": 400})
    assert s.score == 1.0


def test_score_brevity_over_limit_partial_credit():
    text = "x" * 500
    s = score_brevity(text, {"max_chars": 400})
    assert 0.0 < s.score < 1.0


def test_score_brevity_far_over_zero():
    text = "x" * 1500
    s = score_brevity(text, {"max_chars": 400})
    assert s.score == 0.0


# ---------------------------------------------------------------------------
# Factual consistency (heuristic fallback)
# ---------------------------------------------------------------------------


def test_heuristic_factual_consistency_matching_direction():
    fixture_input = {"direction": "BUY", "symbol": "XAUUSD"}
    s = heuristic_factual_consistency(
        "Bullish setup at 2350. Going long.", fixture_input
    )
    assert s == 1.0


def test_heuristic_factual_consistency_mismatch_direction():
    fixture_input = {"direction": "BUY", "symbol": "XAUUSD"}
    # Talks bearish on a BUY signal — should be penalised
    s = heuristic_factual_consistency(
        "Bearish trend detected. Avoid long entries.", fixture_input
    )
    assert s < 1.0


def test_heuristic_factual_consistency_other_instrument_hallucination():
    fixture_input = {"direction": "BUY", "symbol": "XAUUSD"}
    s = heuristic_factual_consistency(
        "Bullish setup. The Bitcoin chart suggests strength.", fixture_input
    )
    # Penalised for hallucinating BTC
    assert s < 1.0


# ---------------------------------------------------------------------------
# End-to-end harness on stub generator
# ---------------------------------------------------------------------------


def _stub_generator(fx_input: dict) -> str:
    """Deterministic compliance-aware generator used to test the harness end
    to end without hitting any LLM. Mirrors what a properly-prompted Claude
    Sonnet would emit for the fixture inputs."""
    direction = fx_input.get("direction", "")
    if direction == "HOLD":
        return (
            "Neutral context. Setup quality is low. "
            "We monitor the structure for a clearer break before any conviction."
        )
    word = "Bullish" if direction == "BUY" else "Bearish"
    sym = fx_input.get("symbol", "XAUUSD")
    entry = fx_input.get("entry")
    sl = fx_input.get("stop_loss")
    tp = fx_input.get("target_1")
    comps = ", ".join(fx_input.get("components_fired", [])) or "no fired components"
    return (
        f"{word} setup observed on {sym} at {entry}. "
        f"Stop level {sl}, first target {tp}. "
        f"Confluence: {comps}. "
        "Educational analysis only; not a recommendation."
    )


def test_harness_runs_on_full_fixture_set_with_stub_generator():
    fixtures = build_fixtures()
    harness = EvalHarness(generator=_stub_generator)
    results = harness.run_all(fixtures)
    assert len(results) == 50

    summary = harness.summary(results)
    assert summary.n_fixtures == 50
    # Stub generator is compliance-aware: should hit the high-bar thresholds
    assert summary.per_axis_mean["forbidden_phrases"] >= 0.95
    assert summary.per_axis_mean["brevity"] >= 0.80
    assert summary.per_axis_mean["source_attribution"] >= 0.80
    # Stub direction always matches: factual_consistency at 1.0
    assert summary.per_axis_mean["factual_consistency"] == 1.0


def test_harness_save_results_round_trip(tmp_path):
    import json

    fixtures = build_fixtures()[:5]  # small subset for I/O test
    harness = EvalHarness(generator=_stub_generator)
    results = harness.run_all(fixtures)
    out_path = harness.save_results(results, tmp_path / "results.json")
    assert out_path.exists()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert "summary" in payload
    assert "results" in payload
    assert len(payload["results"]) == 5


# ---------------------------------------------------------------------------
# Compliance hard gate (DoD KPI: forbidden_phrases >= 0.95)
# ---------------------------------------------------------------------------


def test_compliance_hard_gate_with_compliant_generator():
    """Plan KPI: forbidden_phrases >= 0.95. The stub generator never emits
    forbidden phrases, so this should be 1.0. If a future generator change
    breaks this, the test catches it."""
    fixtures = build_fixtures()
    harness = EvalHarness(generator=_stub_generator)
    results = harness.run_all(fixtures)
    summary = harness.summary(results)
    assert summary.per_axis_mean["forbidden_phrases"] >= 0.95


# ---------------------------------------------------------------------------
# Live test (real LLM judge) — skipped unless ANTHROPIC_API_KEY in env
# ---------------------------------------------------------------------------


@pytest.mark.live
@pytest.mark.skipif(
    True,  # always skip in this commit; flip when an Anthropic judge is wired
    reason="LLM-as-judge integration not wired yet (will land in Phase 2B LLM-2B.3)",
)
def test_live_llm_judge_baseline():
    """Future: hits real Claude Sonnet as the LLM-as-judge for factual_consistency,
    runs full 50-fixture suite, asserts factual_consistency >= 0.75 per plan KPI."""
    pass
