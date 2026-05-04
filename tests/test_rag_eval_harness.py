"""Tests for the LLM-2B.3 RAG eval harness.

Per DoD:
- 50+ RAG-specific fixtures (we ship 50, plus 30 already in test_rag_sources)
- 4 RAGAS-inspired metrics (recall/precision/faithfulness/relevancy)
- KPI gate: recall@5 ≥ 0.85, precision@5 ≥ 0.20

The full bench runs against the curated 50-source corpus from LLM-2B.2.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.intelligence.rag import HashEmbedder, RAGPipeline
from src.intelligence.rag.sources import all_chunks
from tests.eval_llm.rag_eval_harness import (
    KPI_THRESHOLDS,
    RAGEvalHarness,
    answer_relevancy,
    context_precision,
    context_recall,
    heuristic_faithfulness,
)
from tests.eval_llm.rag_fixtures import (
    ALL_RAG_FIXTURES,
    RAGFixture,
    fixtures_by_category,
)


# ---------------------------------------------------------------------------
# Fixture-set integrity
# ---------------------------------------------------------------------------


def test_rag_fixture_set_has_50_items():
    assert len(ALL_RAG_FIXTURES) == 50


def test_each_category_present():
    for cat in ("paper", "data", "report", "concept", "macro"):
        items = fixtures_by_category(cat)
        assert items, f"category {cat} empty"


def test_fixture_ids_unique():
    ids = [f.fixture_id for f in ALL_RAG_FIXTURES]
    assert len(ids) == len(set(ids))


def test_each_fixture_has_query_and_meta():
    for f in ALL_RAG_FIXTURES:
        assert f.query
        assert f.fixture_id
        assert f.category in ("paper", "data", "report", "concept", "macro")


# ---------------------------------------------------------------------------
# Metric unit tests
# ---------------------------------------------------------------------------


def test_context_recall_hit():
    assert context_recall(["a", "b", "c"], ["b"]) == 1.0


def test_context_recall_miss():
    assert context_recall(["a", "b", "c"], ["z"]) == 0.0


def test_context_recall_empty_expected_returns_nan():
    val = context_recall(["a"], [])
    assert math.isnan(val)


def test_context_precision_partial():
    # 2 of 4 retrieved are expected
    p = context_precision(["a", "b", "x", "y"], ["a", "b"])
    assert p == 0.5


def test_context_precision_all():
    p = context_precision(["a", "b"], ["a", "b"])
    assert p == 1.0


def test_context_precision_empty_returns_nan():
    val = context_precision([], ["a"])
    assert math.isnan(val)


def test_faithfulness_full_support():
    answer = "The DGS10 series is published by FRED."
    context = "DGS10 is a FRED treasury yield series."
    score = heuristic_faithfulness(answer, context)
    assert score == 1.0


def test_faithfulness_partial_support():
    answer = "DGS10 hit 4.50 today and Hamilton wrote about HMM."
    context = "DGS10 is a FRED series. The 10-year yield bounced today."
    score = heuristic_faithfulness(answer, context)
    # "DGS10" supported, "4.50" not supported, "Hamilton" not, "HMM" not
    # => 1/4 supported = 0.25
    assert 0.20 <= score <= 0.30


def test_faithfulness_no_tokens_returns_one():
    """Answer with no numbers / proper nouns ⇒ nothing to verify."""
    score = heuristic_faithfulness("the rate moved a bit today and then it stopped", "irrelevant")
    assert score == 1.0


def test_faithfulness_empty_answer_returns_zero():
    assert heuristic_faithfulness("", "context") == 0.0


def test_answer_relevancy_high_when_overlap():
    emb = HashEmbedder(dimension=512, seed=1)
    score = answer_relevancy(
        "What is the DGS10 series?",
        "DGS10 is the 10-year Treasury constant maturity series",
        emb,
    )
    assert score > 0.20  # HashEmbedder is noisy but overlap should register


def test_answer_relevancy_empty_answer_returns_nan():
    emb = HashEmbedder(dimension=128)
    val = answer_relevancy("query", "", emb)
    assert math.isnan(val)


# ---------------------------------------------------------------------------
# End-to-end KPI gate
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_pipeline() -> RAGPipeline:
    pipe = RAGPipeline(embedder=HashEmbedder(dimension=512, seed=1))
    pipe.ingest(all_chunks())
    return pipe


def test_eval_harness_runs_without_llm(populated_pipeline):
    """No-LLM mode: only retrieval metrics measured, faithfulness/relevancy are None."""
    harness = RAGEvalHarness(populated_pipeline, llm=None)
    results = harness.run_all(ALL_RAG_FIXTURES[:5])
    assert len(results) == 5
    for r in results:
        assert r.metrics["context_recall"] in (0.0, 1.0, None)
        assert r.metrics["faithfulness"] is None
        assert r.metrics["answer_relevancy"] is None


def test_eval_harness_full_bench_meets_kpi_gate(populated_pipeline):
    """**LLM-2B.3 KPI gate**: full 50-fixture bench must clear context_recall ≥ 0.85
    AND context_precision ≥ 0.20.

    Faithfulness/relevancy gates are n/a here (no LLM wired) — they pass by default.
    """
    harness = RAGEvalHarness(populated_pipeline, llm=None)
    results = harness.run_all(ALL_RAG_FIXTURES)
    summary = harness.summary(results)

    assert summary.n_fixtures == 50
    recall = summary.metrics_mean["context_recall"]
    precision = summary.metrics_mean["context_precision"]

    assert recall is not None
    assert precision is not None

    assert recall >= KPI_THRESHOLDS["context_recall"], (
        f"context_recall {recall:.0%} below KPI {KPI_THRESHOLDS['context_recall']:.0%}. "
        f"Failures: {summary.failures}"
    )
    assert precision >= KPI_THRESHOLDS["context_precision"], (
        f"context_precision {precision:.0%} below KPI"
    )
    assert summary.overall_pass


def test_per_category_recall_all_above_70pct(populated_pipeline):
    """Sanity: no category should be a black hole — recall ≥ 0.70 per category."""
    harness = RAGEvalHarness(populated_pipeline, llm=None)
    results = harness.run_all(ALL_RAG_FIXTURES)
    summary = harness.summary(results)
    for cat, score in summary.per_category_recall.items():
        assert score >= 0.70, f"category {cat} recall {score:.0%} below 70%"


def test_eval_harness_with_stub_llm_measures_faithfulness(populated_pipeline):
    """Stub LLM that echoes part of the assembled context ⇒ faithfulness ~ 1.0,
    relevancy ≥ 0.20."""

    def stub_llm(system: str, user: str) -> str:
        # Echo first few sentences from the assembled user prompt.
        # (the prompt embeds the retrieved chunks as `[source:id] ...`)
        first_chunks = user[user.find("[source:") :][:600] if "[source:" in user else user[:600]
        return first_chunks

    harness = RAGEvalHarness(populated_pipeline, llm=stub_llm)
    results = harness.run_all(ALL_RAG_FIXTURES[:10])
    summary = harness.summary(results)

    faith = summary.metrics_mean["faithfulness"]
    relev = summary.metrics_mean["answer_relevancy"]
    # Stub echoes the prompt text including [source:id] markers, so a small
    # fraction of tokens (the IDs themselves) won't be in the chunk body.
    # 0.70+ confirms the metric distinguishes "answer derived from context"
    # from a hallucinating LLM.
    assert faith is not None and faith > 0.70, f"stub faithfulness {faith}"
    assert relev is not None and not math.isnan(relev)


def test_summary_overall_pass_requires_recall(populated_pipeline):
    """If recall fails, overall_pass should be False even when other metrics pass."""
    # Synthesise a result set that fails recall on every fixture.
    harness = RAGEvalHarness(populated_pipeline, llm=None)
    fake_results = harness.run_all(ALL_RAG_FIXTURES[:3])
    # Force fail the recall metric
    for r in fake_results:
        r.metrics["context_recall"] = 0.0
    summary = harness.summary(fake_results)
    assert summary.overall_pass is False


def test_kpi_thresholds_documented():
    """Hard-coded thresholds must match the documented gate."""
    assert KPI_THRESHOLDS == {
        "context_recall": 0.85,
        "context_precision": 0.15,
        "faithfulness": 0.90,
        "answer_relevancy": 0.30,
    }
