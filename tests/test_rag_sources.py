"""Tests for the LLM-2B.2 curated source registry.

Per DoD: 50 sources indexed, tags verified.
Per KPI: retrieval quality (manual eval n=50 queries) >= 80% top-5.

We exercise a 10-query manual eval here as a strict regression gate;
LLM-2B.3 will extend to the full 50+ query bench.
"""

from __future__ import annotations

import pytest

from src.intelligence.rag import HashEmbedder, RAGPipeline
from src.intelligence.rag.sources import (
    DATA_SOURCES,
    EDUCATION,
    PAPERS,
    REPORTS,
    all_chunks,
    all_sources,
    sources_by_type,
    validate_registry,
)


# ---------------------------------------------------------------------------
# Registry integrity
# ---------------------------------------------------------------------------


def test_registry_validates():
    # DG-058a (Sprint 5 2026-05-28) added 7 papers covering the brief's
    # missing authors (Gibbs-Candès, Angelopoulos-Bates, Barndorff-Nielsen,
    # Engle, Lo, Pedersen, Cont). New baseline: 22 papers / 57 total.
    diag = validate_registry()
    assert diag["total"] == 57
    assert diag["by_type"]["paper"] == 22
    assert diag["by_type"]["report"] == 15
    assert diag["by_type"]["data"] == 10
    assert diag["by_type"]["education"] == 10
    assert diag["ids_unique"]


def test_each_section_has_expected_count():
    assert len(PAPERS) == 22
    assert len(REPORTS) == 15
    assert len(DATA_SOURCES) == 10
    assert len(EDUCATION) == 10


def test_sources_by_type_filters_correctly():
    papers = sources_by_type("paper")
    assert len(papers) == 22
    assert all(p.type == "paper" for p in papers)


def test_dg058a_brief_authors_all_covered():
    """DG-058a — verify the 12 author tags requested in the Sprint 3 brief."""
    brief_authors = (
        "lopez_de_prado", "corsi", "gibbs_candes", "barndorff_nielsen",
        "adams_mackay", "angelopoulos_bates", "engle", "cont",
        "lo_adaptive", "pedersen", "patton_sheppard",
    )
    ids = {p.source_id for p in PAPERS}
    for author in brief_authors:
        assert any(author in i for i in ids), f"DG-058a brief author missing: {author}"


def test_all_source_ids_unique():
    ids = [s.source_id for s in all_sources()]
    assert len(ids) == len(set(ids))


def test_each_source_has_required_fields():
    for s in all_sources():
        assert s.source_id
        assert s.label
        assert s.ref
        assert s.summary
        assert s.keywords
        assert 0 <= s.authority_score <= 10


def test_each_source_has_meaningful_summary():
    """Summaries should be non-trivial — 200+ chars catches "TODO" placeholders."""
    for s in all_sources():
        assert len(s.summary) >= 200, (
            f"{s.source_id}: summary too short ({len(s.summary)} chars)"
        )


def test_authority_scores_distribution_sane():
    """High-authority sources (papers + data) should average above 8;
    educational should be lower."""
    paper_avg = sum(s.authority_score for s in PAPERS) / len(PAPERS)
    data_avg = sum(s.authority_score for s in DATA_SOURCES) / len(DATA_SOURCES)
    edu_avg = sum(s.authority_score for s in EDUCATION) / len(EDUCATION)
    assert paper_avg >= 8.5, f"papers avg authority {paper_avg} too low"
    assert data_avg >= 9.0, f"data avg authority {data_avg} too low"
    assert edu_avg <= 8.0, f"education avg {edu_avg} should be lower than papers"


# ---------------------------------------------------------------------------
# Chunk materialisation
# ---------------------------------------------------------------------------


def test_all_chunks_returns_at_least_50_chunks():
    """Each curated summary fits in 1 chunk (≤500 tokens) ⇒ 50 chunks total."""
    chunks = all_chunks()
    assert len(chunks) >= 50


def test_chunks_carry_source_metadata():
    chunks = all_chunks()
    for c in chunks[:5]:
        assert c.metadata["type"] in {"paper", "report", "data", "education"}
        assert c.metadata["label"]
        assert "authority_score" in c.metadata
        assert isinstance(c.metadata.get("keywords"), list)


# ---------------------------------------------------------------------------
# End-to-end retrieval quality (DoD KPI gate)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def populated_pipeline() -> RAGPipeline:
    pipe = RAGPipeline(embedder=HashEmbedder(dimension=512, seed=1))
    pipe.ingest(all_chunks())
    return pipe


# Curated test queries: each maps a natural-language question to the source
# we expect to be top-K. Hand-picked to span all 4 source types.
RETRIEVAL_QUERIES = [
    ("CPCV walk-forward purged cross-validation", "paper_lopez_de_prado_afml_cpcv_2018"),
    ("Deflated Sharpe Ratio formula", "paper_bailey_lopez_dsr_2014"),
    ("Probability Backtest Overfitting rank logit", "paper_bailey_borwein_pbo_2014"),
    ("HAR-RV daily weekly monthly volatility", "paper_corsi_har_rv_2009"),
    ("Patton Sheppard good vol bad vol", "paper_patton_sheppard_har_pd_req_2015"),
    ("BOCPD bayesian changepoint detection", "paper_adams_mackay_bocpd_2007"),
    ("Diebold-Mariano forecast accuracy test", "paper_diebold_mariano_1995"),
    ("Holm Bonferroni multiple testing FWER", "paper_holm_1979"),
    ("Wolpert stacking meta learner", "paper_wolpert_stacked_1992"),
    ("isotonic calibration probability", "paper_niculescu_mizil_calibration_2005"),
    ("RAG retrieval augmented generation", "paper_lewis_rag_2020"),
    ("BM25 sparse retrieval Okapi", "paper_robertson_bm25_2009"),
    ("CFTC COT Comex Gold release schedule Friday", "data_cftc_cot_release_schedule"),
    ("FOMC release schedule meetings calendar", "data_fomc_release_schedule"),
    ("DGS10 10-year treasury yield Fed", "data_fred_dgs10"),
    ("DFII10 TIPS real yield gold", "data_fred_dfii10"),
    ("DTWEXBGS trade weighted dollar broad", "data_fred_dtwexbgs"),
    ("VIX volatility S&P 500 options", "data_fred_vixcls"),
    ("CME Gold futures GC contract specs", "data_cme_gold_specs"),
    ("LBMA Gold Price daily fix London auction", "data_lbma_gold_fix"),
    ("LBMA quarterly review wholesale gold", "report_lbma_quarterly_q1_2026"),
    ("World Gold Council demand trends central bank", "report_wgc_demand_trends_q1_2026"),
    ("BIS quarterly review reserve composition", "report_bis_quarterly_2026_q1"),
    ("FOMC minutes monetary policy outlook", "report_fomc_minutes_template"),
    ("ECB Governing Council deposit rate", "report_ecb_monetary_policy"),
    ("Smart Money BOS CHoCH FVG order block", "edu_babypips_smc"),
    ("VIX fear gauge volatility complacency", "edu_investopedia_vix"),
    ("yield curve inversion T10Y2Y recession", "edu_investopedia_yield_curve"),
    ("COT report managed money speculator contrarian", "edu_investopedia_cot"),
    ("forex London New York session liquidity", "edu_babypips_sessions"),
]


def test_retrieval_top_5_quality_meets_80pct_kpi(populated_pipeline):
    """**LLM-2B.2 DoD/KPI gate**: at least 80% of natural-language queries
    must have the expected source in the top-5 results."""
    hits = 0
    failures: list[str] = []
    for query, expected in RETRIEVAL_QUERIES:
        response = populated_pipeline.query(query)
        top5 = [rc.chunk.source_id for rc in response.retrieved[:5]]
        if expected in top5:
            hits += 1
        else:
            failures.append(
                f"  Q='{query}' expected={expected} got_top1={top5[:1] if top5 else 'NONE'}"
            )
    score = hits / len(RETRIEVAL_QUERIES)
    assert score >= 0.80, (
        f"retrieval top-5 quality {score:.0%} below KPI 80%. Failures:\n"
        + "\n".join(failures)
    )


def test_retrieval_top_1_quality_at_least_70pct(populated_pipeline):
    """Stronger sanity: most queries should land their expected source AT top-1.
    Not part of the explicit DoD but a useful drift detector."""
    hits = 0
    for query, expected in RETRIEVAL_QUERIES:
        response = populated_pipeline.query(query)
        if response.retrieved and response.retrieved[0].chunk.source_id == expected:
            hits += 1
    score = hits / len(RETRIEVAL_QUERIES)
    assert score >= 0.70, f"top-1 quality {score:.0%} below 70% guard"


def test_corpus_has_57_chunks_after_ingestion(populated_pipeline):
    # DG-058a (2026-05-28) bumped corpus from 50 → 57 chunks (7 new papers).
    assert populated_pipeline.size == 57
