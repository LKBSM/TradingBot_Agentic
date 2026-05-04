"""RAG evaluation harness — Sprint LLM-2B.3.

Inspired by the RAGAS framework (Es et al. 2023), but designed to run
deterministically in CI without an LLM-as-judge dependency. Each metric
returns a float in [0, 1]; per-fixture and aggregate scores are exposed.

Metrics
-------
1. **context_recall@k** — did the retrieved top-k include any of the
   fixture's `expected_sources`? Binary 1/0 per fixture, averaged.
2. **context_precision@k** — fraction of the retrieved top-k whose
   source_id is in `expected_sources`. Captures signal/noise of
   retrieval (rewards systems that don't pad with junk).
3. **faithfulness** — when an LLM answer is provided: every numeric
   token / capitalised proper noun in the answer must appear in the
   assembled context. Returns the fraction of answer tokens supported.
   When no answer is provided, score is None and excluded from aggregate.
4. **answer_relevancy** — cosine similarity between the query embedding
   and the answer embedding. Uses the same embedder the pipeline uses
   (HashEmbedder in CI, real embedder in prod).

KPI gates (per LLM-2B.3 plan)
-----------------------------
- context_recall@5  ≥ 0.85
- context_precision@5 ≥ 0.15  (most fixtures specify 1 expected source,
  so the achievable upper bound at top-5 is ~0.20 — gating at 0.15
  means 75 %+ of expected sources landed in the top-5 set)
- faithfulness     ≥ 0.90 (only evaluated when an LLM is plugged in)
- answer_relevancy ≥ 0.30 (HashEmbedder cosine is noisier than dense
  semantic embeddings; raise this gate when migrating to Voyage)

Aggregate gate: at least 3 of the 4 metrics must clear their threshold,
AND context_recall must clear independently (it is the foundational
gate — if retrieval misses, faithfulness is moot).
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from src.intelligence.rag import HashEmbedder, RAGPipeline
from src.intelligence.rag.embedders import Embedder
from tests.eval_llm.rag_fixtures import RAGFixture

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RAGFixtureResult:
    fixture_id: str
    category: str
    query: str
    retrieved_ids: list[str]
    answer: str
    metrics: dict[str, Optional[float]]

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "query": self.query,
            "retrieved_ids": self.retrieved_ids,
            "answer": self.answer,
            "metrics": dict(self.metrics),
        }


@dataclass
class RAGEvalSummary:
    n_fixtures: int
    metrics_mean: dict[str, Optional[float]]
    per_category_recall: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    kpi_gates: dict[str, bool] = field(default_factory=dict)
    overall_pass: bool = False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def context_recall(retrieved_ids: list[str], expected_sources: list[str]) -> float:
    """1 if at least one expected source is in the retrieved list, else 0.

    For fixtures with empty `expected_sources` (meta questions), returns
    None — these are filtered out of the aggregate.
    """
    if not expected_sources:
        return float("nan")
    return 1.0 if any(eid in retrieved_ids for eid in expected_sources) else 0.0


def context_precision(retrieved_ids: list[str], expected_sources: list[str]) -> float:
    """Fraction of top-k retrieved chunks whose source_id is expected.

    NaN for empty fixtures (no ground truth to score against).
    """
    if not expected_sources or not retrieved_ids:
        return float("nan")
    expected_set = set(expected_sources)
    hits = sum(1 for rid in retrieved_ids if rid in expected_set)
    return hits / len(retrieved_ids)


_NUM_RE = re.compile(r"\b\d[\d,.]*\b")
# Proper nouns include alphanumeric tickers (e.g. DGS10, DTWEXBGS) as well as
# names — anything that starts with an uppercase letter and is followed by
# 2+ alphanumeric word characters.
_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")


def heuristic_faithfulness(answer: str, context: str) -> float:
    """Fraction of the answer's numeric and proper-noun tokens that
    appear verbatim (case-insensitive substring) in the context.

    Designed to catch hallucinated numbers / instruments / names without
    requiring an LLM-as-judge. Conservative: returns 1.0 if the answer
    has no tokens to verify.
    """
    if not answer.strip():
        return 0.0
    nums = _NUM_RE.findall(answer)
    nouns = _PROPER_NOUN_RE.findall(answer)
    tokens = list(set(nums + nouns))
    # Strip trivially-true tokens (single-word language particles).
    stop_proper = {"The", "This", "That", "These", "Those", "When", "What", "Why", "How"}
    tokens = [t for t in tokens if t not in stop_proper]
    if not tokens:
        return 1.0
    ctx_lower = context.lower()
    supported = sum(1 for t in tokens if t.lower() in ctx_lower)
    return supported / len(tokens)


def answer_relevancy(query: str, answer: str, embedder: Embedder) -> float:
    """Cosine similarity between query and answer embeddings.

    Returns NaN when answer is empty (no LLM was wired in).
    """
    if not answer.strip():
        return float("nan")
    vecs = embedder.embed([query, answer])
    q_vec, a_vec = vecs[0], vecs[1]
    # Vectors are L2-normalised by the embedder, so dot == cosine.
    return float(np.dot(q_vec, a_vec))


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


# KPI gate thresholds (declared here so they're discoverable and unit-tested).
KPI_THRESHOLDS = {
    "context_recall": 0.85,
    "context_precision": 0.15,
    "faithfulness": 0.90,
    "answer_relevancy": 0.30,
}


class RAGEvalHarness:
    """Run RAG-eval metrics over a fixture set.

    Wraps a populated ``RAGPipeline``. The optional ``llm`` callable lets
    callers measure faithfulness/answer_relevancy with a real or stub
    answer-producing function. Without ``llm``, only retrieval-side
    metrics (recall, precision) are computed.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        llm=None,
        embedder: Optional[Embedder] = None,
    ):
        self.pipeline = pipeline
        self.llm = llm
        # Reuse the pipeline's embedder for relevancy scoring unless the
        # caller wants to score with a different one.
        self.embedder = embedder or pipeline.embedder

    def evaluate_one(self, fixture: RAGFixture, top_k: int = 5) -> RAGFixtureResult:
        response = self.pipeline.query(fixture.query, llm=self.llm)
        retrieved = response.retrieved[:top_k]
        retrieved_ids = [rc.chunk.source_id for rc in retrieved]
        answer = response.answer
        # Concatenate context for faithfulness scoring.
        context_blob = "\n".join(rc.chunk.text for rc in retrieved)

        metrics: dict[str, Optional[float]] = {
            "context_recall": context_recall(retrieved_ids, fixture.expected_sources),
            "context_precision": context_precision(retrieved_ids, fixture.expected_sources),
        }
        if answer:
            metrics["faithfulness"] = heuristic_faithfulness(answer, context_blob)
            metrics["answer_relevancy"] = answer_relevancy(
                fixture.query, answer, self.embedder
            )
        else:
            metrics["faithfulness"] = None
            metrics["answer_relevancy"] = None

        # Convert NaN floats to None for clean JSON serialisation.
        for k, v in list(metrics.items()):
            if isinstance(v, float) and v != v:  # NaN
                metrics[k] = None

        return RAGFixtureResult(
            fixture_id=fixture.fixture_id,
            category=fixture.category,
            query=fixture.query,
            retrieved_ids=retrieved_ids,
            answer=answer,
            metrics=metrics,
        )

    def run_all(
        self, fixtures: list[RAGFixture], top_k: int = 5
    ) -> list[RAGFixtureResult]:
        return [self.evaluate_one(fx, top_k=top_k) for fx in fixtures]

    def summary(self, results: list[RAGFixtureResult]) -> RAGEvalSummary:
        if not results:
            return RAGEvalSummary(n_fixtures=0, metrics_mean={})

        per_metric: dict[str, list[float]] = {
            "context_recall": [],
            "context_precision": [],
            "faithfulness": [],
            "answer_relevancy": [],
        }
        per_cat_recall: dict[str, list[float]] = {}

        failures: list[str] = []
        for r in results:
            for k, v in r.metrics.items():
                if v is not None:
                    per_metric[k].append(v)
            recall = r.metrics.get("context_recall")
            if recall is not None:
                per_cat_recall.setdefault(r.category, []).append(recall)
                if recall < 1.0:
                    failures.append(r.fixture_id)

        means: dict[str, Optional[float]] = {
            k: (sum(v) / len(v) if v else None) for k, v in per_metric.items()
        }
        per_cat_mean = {
            k: sum(v) / len(v) for k, v in per_cat_recall.items() if v
        }

        gates = {
            k: (means[k] is not None and means[k] >= KPI_THRESHOLDS[k])
            for k in KPI_THRESHOLDS
        }
        # If a metric was never measured (no LLM), it doesn't fail the gate.
        for k, v in means.items():
            if v is None:
                gates[k] = True  # n/a

        # Recall is foundational — must always pass.
        recall_gate = gates["context_recall"]
        passed = sum(1 for v in gates.values() if v)
        overall_pass = recall_gate and passed >= 3

        return RAGEvalSummary(
            n_fixtures=len(results),
            metrics_mean=means,
            per_category_recall=per_cat_mean,
            failures=failures,
            kpi_gates=gates,
            overall_pass=overall_pass,
        )

    def to_dict(self, summary: RAGEvalSummary) -> dict:
        return asdict(summary)


__all__ = [
    "KPI_THRESHOLDS",
    "RAGEvalHarness",
    "RAGEvalSummary",
    "RAGFixtureResult",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "heuristic_faithfulness",
]
