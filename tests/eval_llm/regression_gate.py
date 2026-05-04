"""RAG eval regression gate — Sprint LLM-2B.9.

The gate compares the current eval-harness run against a frozen
baseline (``rag_baseline.json``) and fails CI if any tracked metric
falls more than ``tolerance_pp`` below its floor.

Why a frozen baseline (not just a static threshold)
---------------------------------------------------
The KPI thresholds in :mod:`tests.eval_llm.rag_eval_harness` are the
*absolute floor* — they detect catastrophic breakage. The baseline
here represents the *current achieved quality*, so we also detect
silent regressions: a small change that doesn't trip the absolute KPI
but does shave 3 percentage points off recall is a flag-worthy
quality drift, not a non-event.

Updating the baseline
---------------------
Anyone bumping a metric on purpose (e.g. swapping HashEmbedder for
Voyage embeddings) regenerates the file with::

    python -m tests.eval_llm.regression_gate --update

and commits the diff alongside the change. The accompanying message
should justify *why* the baseline moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


BASELINE_PATH = Path(__file__).with_name("rag_baseline.json")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RegressionViolation:
    metric: str
    observed: float
    baseline_floor: float
    delta_pp: float

    def __str__(self) -> str:  # pragma: no cover — display only
        return (
            f"{self.metric}: observed {self.observed:.4f} "
            f"vs floor {self.baseline_floor:.4f} (Δ {self.delta_pp*100:+.1f}pp)"
        )


@dataclass
class GateResult:
    ok: bool
    violations: list[RegressionViolation] = field(default_factory=list)
    n_metrics_checked: int = 0

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Baseline IO
# ---------------------------------------------------------------------------


def load_baseline(path: Path = BASELINE_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_baseline(baseline: dict, path: Path = BASELINE_PATH) -> None:
    """Pretty-write so diffs are readable in code review."""
    path.write_text(
        json.dumps(baseline, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare(
    summary,
    baseline: Optional[dict] = None,
) -> GateResult:
    """Compare a ``RAGEvalSummary`` against the baseline floors.

    Returns a ``GateResult`` with one violation per metric that fell
    more than ``tolerance_pp`` below its floor. Metrics in the baseline
    that are absent from the summary are skipped (e.g. faithfulness
    when no LLM was wired in).
    """
    baseline = baseline if baseline is not None else load_baseline()
    tolerance = float(baseline.get("tolerance_pp", 0.05))
    floors = baseline.get("metrics_floor", {})
    cat_floors = baseline.get("per_category_recall_floor", {})

    violations: list[RegressionViolation] = []
    n_checked = 0

    means = summary.metrics_mean
    for metric, floor in floors.items():
        observed = means.get(metric)
        if observed is None:
            continue
        n_checked += 1
        delta = observed - floor
        if delta < -tolerance:
            violations.append(
                RegressionViolation(
                    metric=metric,
                    observed=observed,
                    baseline_floor=floor,
                    delta_pp=delta,
                )
            )

    per_cat = summary.per_category_recall
    for cat, floor in cat_floors.items():
        observed = per_cat.get(cat)
        if observed is None:
            continue
        n_checked += 1
        delta = observed - floor
        if delta < -tolerance:
            violations.append(
                RegressionViolation(
                    metric=f"per_category_recall[{cat}]",
                    observed=observed,
                    baseline_floor=floor,
                    delta_pp=delta,
                )
            )

    return GateResult(
        ok=not violations,
        violations=violations,
        n_metrics_checked=n_checked,
    )


# ---------------------------------------------------------------------------
# CLI: --update regenerates the baseline from the current eval run
# ---------------------------------------------------------------------------


def _run_full_eval():  # pragma: no cover — exercised via integration only
    from src.intelligence.rag import HashEmbedder, RAGPipeline
    from src.intelligence.rag.sources import all_chunks
    from tests.eval_llm.rag_eval_harness import RAGEvalHarness
    from tests.eval_llm.rag_fixtures import ALL_RAG_FIXTURES

    pipe = RAGPipeline(embedder=HashEmbedder(dimension=512, seed=1))
    pipe.ingest(all_chunks())
    harness = RAGEvalHarness(pipe)
    results = harness.run_all(ALL_RAG_FIXTURES)
    return harness.summary(results)


def _cli() -> int:  # pragma: no cover — manual workflow
    parser = argparse.ArgumentParser(
        description="RAG eval regression gate — compare current scores to baseline."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Regenerate the baseline from the current eval run.",
    )
    args = parser.parse_args()

    summary = _run_full_eval()
    if args.update:
        baseline = load_baseline()
        baseline["metrics_floor"] = {
            k: round(v, 4)
            for k, v in summary.metrics_mean.items()
            if v is not None
        }
        baseline["per_category_recall_floor"] = {
            k: round(v, 4) for k, v in summary.per_category_recall.items()
        }
        baseline["snapshot_taken_utc"] = (
            __import__("datetime").datetime.utcnow().isoformat(timespec="seconds")
            + "Z"
        )
        save_baseline(baseline)
        print(f"baseline updated: {BASELINE_PATH}")
        return 0

    result = compare(summary)
    if not result.ok:
        print("RAG eval regression detected:")
        for v in result.violations:
            print(f"  - {v}")
        return 1
    print(f"RAG eval gate OK ({result.n_metrics_checked} metrics checked)")
    return 0


__all__ = [
    "BASELINE_PATH",
    "GateResult",
    "RegressionViolation",
    "compare",
    "load_baseline",
    "save_baseline",
]


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli())
