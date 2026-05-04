"""Tests for the LLM-2B.9 eval regression gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.eval_llm.rag_eval_harness import RAGEvalSummary
from tests.eval_llm.regression_gate import (
    BASELINE_PATH,
    GateResult,
    RegressionViolation,
    compare,
    load_baseline,
    save_baseline,
)


# ---------------------------------------------------------------------------
# Baseline integrity
# ---------------------------------------------------------------------------


def test_baseline_file_exists():
    assert BASELINE_PATH.exists()


def test_baseline_has_required_keys():
    b = load_baseline()
    assert "metrics_floor" in b
    assert "per_category_recall_floor" in b
    assert "tolerance_pp" in b
    assert b["fixture_set"].endswith("ALL_RAG_FIXTURES")


def test_baseline_floors_match_known_values():
    b = load_baseline()
    floors = b["metrics_floor"]
    # Recall has been measured at 0.98 (test_rag_eval_harness); floor
    # should be at least 0.85 (KPI) and at most observed (≤1.0).
    assert 0.85 <= floors["context_recall"] <= 1.0
    assert 0.10 <= floors["context_precision"] <= 1.0


# ---------------------------------------------------------------------------
# Compare logic
# ---------------------------------------------------------------------------


def _summary(metrics: dict, per_cat: dict | None = None) -> RAGEvalSummary:
    return RAGEvalSummary(
        n_fixtures=50,
        metrics_mean=metrics,
        per_category_recall=per_cat or {},
    )


def test_compare_passes_when_metrics_at_floor():
    baseline = {
        "metrics_floor": {"context_recall": 0.95},
        "per_category_recall_floor": {},
        "tolerance_pp": 0.05,
    }
    s = _summary({"context_recall": 0.95})
    result = compare(s, baseline=baseline)
    assert result.ok


def test_compare_passes_within_tolerance():
    baseline = {
        "metrics_floor": {"context_recall": 0.95},
        "per_category_recall_floor": {},
        "tolerance_pp": 0.05,
    }
    s = _summary({"context_recall": 0.91})  # 4pp below ⇒ within 5pp tolerance
    result = compare(s, baseline=baseline)
    assert result.ok


def test_compare_fails_outside_tolerance():
    baseline = {
        "metrics_floor": {"context_recall": 0.95},
        "per_category_recall_floor": {},
        "tolerance_pp": 0.05,
    }
    s = _summary({"context_recall": 0.85})  # 10pp below ⇒ violation
    result = compare(s, baseline=baseline)
    assert not result.ok
    assert len(result.violations) == 1
    v = result.violations[0]
    assert v.metric == "context_recall"
    assert v.observed == 0.85
    assert v.baseline_floor == 0.95
    assert v.delta_pp == pytest.approx(-0.10)


def test_compare_skips_metrics_absent_from_summary():
    """faithfulness is only measured when an LLM is wired in; a CI run
    without an LLM should not trip the gate on its absence."""
    baseline = {
        "metrics_floor": {
            "context_recall": 0.85,
            "faithfulness": 0.90,
        },
        "per_category_recall_floor": {},
        "tolerance_pp": 0.05,
    }
    s = _summary({"context_recall": 0.95, "faithfulness": None})
    result = compare(s, baseline=baseline)
    assert result.ok
    assert result.n_metrics_checked == 1


def test_compare_collects_multiple_violations():
    baseline = {
        "metrics_floor": {"context_recall": 0.95, "context_precision": 0.20},
        "per_category_recall_floor": {},
        "tolerance_pp": 0.02,
    }
    s = _summary({"context_recall": 0.80, "context_precision": 0.10})
    result = compare(s, baseline=baseline)
    assert not result.ok
    assert len(result.violations) == 2


def test_compare_per_category_floor_violation():
    baseline = {
        "metrics_floor": {},
        "per_category_recall_floor": {"paper": 1.0},
        "tolerance_pp": 0.05,
    }
    s = _summary({}, per_cat={"paper": 0.80})
    result = compare(s, baseline=baseline)
    assert not result.ok
    assert "per_category_recall[paper]" in result.violations[0].metric


def test_compare_per_category_missing_skipped():
    baseline = {
        "metrics_floor": {},
        "per_category_recall_floor": {"paper": 1.0, "concept": 0.85},
        "tolerance_pp": 0.05,
    }
    # Only "paper" reported in the summary → "concept" is skipped, not failed.
    s = _summary({}, per_cat={"paper": 1.0})
    result = compare(s, baseline=baseline)
    assert result.ok


# ---------------------------------------------------------------------------
# Save round-trip
# ---------------------------------------------------------------------------


def test_save_baseline_round_trip(tmp_path: Path):
    target = tmp_path / "baseline.json"
    payload = {
        "metrics_floor": {"context_recall": 0.95},
        "per_category_recall_floor": {"paper": 1.0},
        "tolerance_pp": 0.05,
    }
    save_baseline(payload, path=target)
    loaded = json.loads(target.read_text())
    assert loaded == payload


# ---------------------------------------------------------------------------
# Integration: live eval vs the committed baseline
# ---------------------------------------------------------------------------


def test_current_rag_run_meets_committed_baseline():
    """End-to-end gate: the live RAG eval against the curated 50-source
    corpus must clear the committed baseline. This is the *actual* CI
    quality gate — if it fails, either the corpus regressed or the
    baseline needs an explicit update via ``regression_gate.py --update``.
    """
    from src.intelligence.rag import HashEmbedder, RAGPipeline
    from src.intelligence.rag.sources import all_chunks
    from tests.eval_llm.rag_eval_harness import RAGEvalHarness
    from tests.eval_llm.rag_fixtures import ALL_RAG_FIXTURES

    pipe = RAGPipeline(embedder=HashEmbedder(dimension=512, seed=1))
    pipe.ingest(all_chunks())
    harness = RAGEvalHarness(pipe)
    results = harness.run_all(ALL_RAG_FIXTURES)
    summary = harness.summary(results)

    result = compare(summary)
    assert result.ok, (
        "RAG eval regression vs committed baseline:\n  "
        + "\n  ".join(str(v) for v in result.violations)
        + "\n\nIf this is intentional, re-baseline via "
        "`python -m tests.eval_llm.regression_gate --update`."
    )
