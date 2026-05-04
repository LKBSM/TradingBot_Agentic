"""RAG evaluation fixtures — Sprint LLM-2B.3.

Each fixture is a (query, expected_sources, ground_truth_claims) triple:

- ``query`` is the user's natural-language question.
- ``expected_sources`` is the set of registry source IDs that SHOULD be
  retrieved for the query to be considered relevant. The pipeline succeeds
  when the intersection of top-k retrieved IDs with this set is non-empty.
- ``ground_truth_claims`` is a small list of factual snippets that an
  ideal answer would surface. Used by the heuristic faithfulness scorer
  to verify that the answer doesn't fabricate numbers / entities outside
  the assembled context.

The bench targets ~50 queries spanning:
- 15 paper-recall queries
- 15 data-source / institutional-report queries
- 10 conceptual / FAQ queries
- 10 macro / market-context queries

Together with `tests/test_rag_sources.py`'s 30 retrieval queries, this
gives ~80+ retrieval probes covering the curated corpus — well beyond
the LLM-1.1 baseline of 50 narrative fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RAGFixture:
    fixture_id: str
    category: str  # "paper", "data", "report", "concept", "macro"
    query: str
    expected_sources: list[str]
    ground_truth_claims: list[str] = field(default_factory=list)
    language: str = "en"


# ---------------------------------------------------------------------------
# Paper recall queries
# ---------------------------------------------------------------------------

_PAPER_FIXTURES: list[RAGFixture] = [
    RAGFixture(
        fixture_id="rag_paper_001",
        category="paper",
        query="Explain combinatorial purged cross-validation and how it differs from k-fold for time series.",
        expected_sources=["paper_lopez_de_prado_afml_cpcv_2018"],
        ground_truth_claims=["CPCV", "purging", "embargo", "C(N, k)", "28 paths"],
    ),
    RAGFixture(
        fixture_id="rag_paper_002",
        category="paper",
        query="What is the Deflated Sharpe Ratio and why does it correct for multiple testing?",
        expected_sources=["paper_bailey_lopez_dsr_2014"],
        ground_truth_claims=["DSR", "Bailey", "multiple testing", "skew", "kurtosis"],
    ),
    RAGFixture(
        fixture_id="rag_paper_003",
        category="paper",
        query="How is Probability of Backtest Overfitting computed using the rank logit?",
        expected_sources=["paper_bailey_borwein_pbo_2014"],
        ground_truth_claims=["PBO", "rank logit", "in-sample", "out-of-sample"],
    ),
    RAGFixture(
        fixture_id="rag_paper_004",
        category="paper",
        query="Describe the HAR-RV model for realized volatility forecasting.",
        expected_sources=["paper_corsi_har_rv_2009"],
        ground_truth_claims=["HAR-RV", "Corsi", "daily", "weekly", "monthly", "realised volatility"],
    ),
    RAGFixture(
        fixture_id="rag_paper_005",
        category="paper",
        query="Patton Sheppard HAR-PD-REQ realised semivariance positive negative volatility decomposition",
        expected_sources=["paper_patton_sheppard_har_pd_req_2015"],
        ground_truth_claims=["Patton", "Sheppard", "semivariance", "HAR-PD-REQ"],
    ),
    RAGFixture(
        fixture_id="rag_paper_006",
        category="paper",
        query="How does Bayesian online changepoint detection compute the run-length distribution?",
        expected_sources=["paper_adams_mackay_bocpd_2007"],
        ground_truth_claims=["BOCPD", "Adams", "MacKay", "run length", "hazard"],
    ),
    RAGFixture(
        fixture_id="rag_paper_007",
        category="paper",
        query="Diebold-Mariano test predictive accuracy loss differential forecast comparison",
        expected_sources=["paper_diebold_mariano_1995"],
        ground_truth_claims=["Diebold", "Mariano", "loss differential", "predictive accuracy"],
    ),
    RAGFixture(
        fixture_id="rag_paper_008",
        category="paper",
        query="How does the Holm-Bonferroni method control family-wise error rate?",
        expected_sources=["paper_holm_1979"],
        ground_truth_claims=["Holm", "Bonferroni", "FWER", "step-down"],
    ),
    RAGFixture(
        fixture_id="rag_paper_009",
        category="paper",
        query="What is Wolpert's stacked generalization and how does the meta-learner work?",
        expected_sources=["paper_wolpert_stacked_1992"],
        ground_truth_claims=["Wolpert", "stacking", "meta-learner", "out-of-fold"],
    ),
    RAGFixture(
        fixture_id="rag_paper_010",
        category="paper",
        query="How does isotonic regression calibrate classifier probabilities?",
        expected_sources=["paper_niculescu_mizil_calibration_2005"],
        ground_truth_claims=["isotonic", "calibration", "Platt", "monotone"],
    ),
    RAGFixture(
        fixture_id="rag_paper_011",
        category="paper",
        query="Lewis RAG retrieval augmented generation parametric language model non-parametric memory",
        expected_sources=["paper_lewis_rag_2020"],
        ground_truth_claims=["RAG", "Lewis", "retrieval", "language model"],
    ),
    RAGFixture(
        fixture_id="rag_paper_012",
        category="paper",
        query="Robertson BM25 Okapi probabilistic ranking k1 b parameters IDF saturation",
        expected_sources=["paper_robertson_bm25_2009"],
        ground_truth_claims=["BM25", "Robertson", "k1", "IDF"],
    ),
    RAGFixture(
        fixture_id="rag_paper_013",
        category="paper",
        query="What is reciprocal rank fusion and why does it not need score normalization?",
        expected_sources=["paper_cormack_rrf_2009"],
        ground_truth_claims=["reciprocal rank fusion", "RRF", "Cormack", "k=60"],
    ),
    RAGFixture(
        fixture_id="rag_paper_014",
        category="paper",
        query="Es RAGAS evaluation framework faithfulness answer relevancy context precision recall",
        expected_sources=["paper_es_ragas_2023"],
        ground_truth_claims=["RAGAS", "Es", "faithfulness", "context"],
    ),
    RAGFixture(
        fixture_id="rag_paper_015",
        category="paper",
        query="Nystrup jump model regime detection sparse hidden state volatility",
        expected_sources=["paper_nystrup_jump_model_2020"],
        ground_truth_claims=["Nystrup", "jump model", "regime"],
    ),
]


# ---------------------------------------------------------------------------
# Data-source recall queries
# ---------------------------------------------------------------------------

_DATA_FIXTURES: list[RAGFixture] = [
    RAGFixture(
        fixture_id="rag_data_001",
        category="data",
        query="When is the CFTC COT report for Comex Gold released and how do I avoid look-ahead bias?",
        expected_sources=["data_cftc_cot_release_schedule"],
        ground_truth_claims=["CFTC", "COT", "Friday", "15:30", "Tuesday"],
    ),
    RAGFixture(
        fixture_id="rag_data_002",
        category="data",
        query="What is the FRED ticker for the 10-year Treasury constant maturity yield?",
        expected_sources=["data_fred_dgs10"],
        ground_truth_claims=["DGS10", "FRED", "10-year", "Treasury"],
    ),
    RAGFixture(
        fixture_id="rag_data_003",
        category="data",
        query="What is the FRED series for 10-year TIPS real yield and why does it matter for gold?",
        expected_sources=["data_fred_dfii10"],
        ground_truth_claims=["DFII10", "TIPS", "real yield", "gold"],
    ),
    RAGFixture(
        fixture_id="rag_data_004",
        category="data",
        query="Which FRED series tracks the trade-weighted broad dollar index?",
        expected_sources=["data_fred_dtwexbgs"],
        ground_truth_claims=["DTWEXBGS", "trade weighted", "dollar"],
    ),
    RAGFixture(
        fixture_id="rag_data_005",
        category="data",
        query="What does VIXCLS represent and how is the VIX computed?",
        expected_sources=["data_fred_vixcls"],
        ground_truth_claims=["VIXCLS", "VIX", "S&P 500", "options"],
    ),
    RAGFixture(
        fixture_id="rag_data_006",
        category="data",
        query="What are the contract specifications for CME Gold Futures (GC)?",
        expected_sources=["data_cme_gold_specs"],
        ground_truth_claims=["GC", "100 troy ounces", "tick", "CME"],
    ),
    RAGFixture(
        fixture_id="rag_data_007",
        category="data",
        query="When does the LBMA Gold Price fix happen and what is the auction mechanism?",
        expected_sources=["data_lbma_gold_fix"],
        ground_truth_claims=["LBMA", "auction", "10:30", "15:00", "London"],
    ),
    RAGFixture(
        fixture_id="rag_data_008",
        category="data",
        query="When are FOMC monetary policy meetings scheduled and how should I align signals to them?",
        expected_sources=["data_fomc_release_schedule"],
        ground_truth_claims=["FOMC", "meetings", "schedule", "policy"],
    ),
    RAGFixture(
        fixture_id="rag_data_009",
        category="data",
        query="What is the publication lag for the CFTC Commitments of Traders report?",
        expected_sources=["data_cftc_cot_release_schedule"],
        ground_truth_claims=["Tuesday", "Friday", "lag", "publication"],
    ),
    RAGFixture(
        fixture_id="rag_data_010",
        category="data",
        query="Which dataset gives the daily LBMA London auction reference price for spot gold?",
        expected_sources=["data_lbma_gold_fix"],
        ground_truth_claims=["LBMA", "London", "auction", "spot"],
    ),
]


# ---------------------------------------------------------------------------
# Report queries
# ---------------------------------------------------------------------------

_REPORT_FIXTURES: list[RAGFixture] = [
    RAGFixture(
        fixture_id="rag_report_001",
        category="report",
        query="Where can I find the LBMA quarterly review on wholesale gold flows?",
        expected_sources=["report_lbma_quarterly_q1_2026"],
        ground_truth_claims=["LBMA", "quarterly", "wholesale"],
    ),
    RAGFixture(
        fixture_id="rag_report_002",
        category="report",
        query="What does the World Gold Council's Demand Trends report cover?",
        expected_sources=["report_wgc_demand_trends_q1_2026"],
        ground_truth_claims=["World Gold Council", "WGC", "demand", "central bank"],
    ),
    RAGFixture(
        fixture_id="rag_report_003",
        category="report",
        query="What is the BIS quarterly review and why is it relevant for FX reserve composition?",
        expected_sources=["report_bis_quarterly_2026_q1"],
        ground_truth_claims=["BIS", "quarterly", "reserve"],
    ),
    RAGFixture(
        fixture_id="rag_report_004",
        category="report",
        query="Where can I read FOMC minutes on monetary policy outlook?",
        expected_sources=["report_fomc_minutes_template"],
        ground_truth_claims=["FOMC", "minutes", "monetary policy"],
    ),
    RAGFixture(
        fixture_id="rag_report_005",
        category="report",
        query="Where does the ECB Governing Council publish its deposit rate decisions?",
        expected_sources=["report_ecb_monetary_policy"],
        ground_truth_claims=["ECB", "Governing Council", "deposit rate"],
    ),
]


# ---------------------------------------------------------------------------
# Conceptual / FAQ queries
# ---------------------------------------------------------------------------

_CONCEPT_FIXTURES: list[RAGFixture] = [
    RAGFixture(
        fixture_id="rag_concept_001",
        category="concept",
        query="What is a Break of Structure in Smart Money Concepts?",
        expected_sources=["edu_babypips_smc"],
        ground_truth_claims=["BOS", "Smart Money", "swing"],
    ),
    RAGFixture(
        fixture_id="rag_concept_002",
        category="concept",
        query="How is the VIX commonly interpreted as a fear gauge?",
        expected_sources=["edu_investopedia_vix"],
        ground_truth_claims=["VIX", "fear", "volatility"],
    ),
    RAGFixture(
        fixture_id="rag_concept_003",
        category="concept",
        query="What does an inverted yield curve signal about recession risk?",
        expected_sources=["edu_investopedia_yield_curve"],
        ground_truth_claims=["yield curve", "inversion", "recession"],
    ),
    RAGFixture(
        fixture_id="rag_concept_004",
        category="concept",
        query="How do contrarian traders use the COT report's managed money positioning?",
        expected_sources=["edu_investopedia_cot"],
        ground_truth_claims=["COT", "managed money", "contrarian"],
    ),
    RAGFixture(
        fixture_id="rag_concept_005",
        category="concept",
        query="When are the most liquid London and New York forex sessions?",
        expected_sources=["edu_babypips_sessions"],
        ground_truth_claims=["London", "New York", "session", "liquidity"],
    ),
    RAGFixture(
        fixture_id="rag_concept_006",
        category="concept",
        query="What is a Fair Value Gap and how is it identified on a chart?",
        expected_sources=["edu_babypips_smc"],
        ground_truth_claims=["FVG", "Fair Value Gap", "imbalance"],
    ),
    RAGFixture(
        fixture_id="rag_concept_007",
        category="concept",
        query="What is the difference between BOS and CHoCH?",
        expected_sources=["edu_babypips_smc"],
        ground_truth_claims=["BOS", "CHoCH", "trend"],
    ),
    RAGFixture(
        fixture_id="rag_concept_008",
        category="concept",
        query="What is a Diebold-Mariano statistic of zero supposed to indicate?",
        expected_sources=["paper_diebold_mariano_1995"],
        ground_truth_claims=["Diebold", "Mariano", "equal accuracy"],
    ),
    RAGFixture(
        fixture_id="rag_concept_009",
        category="concept",
        query="Why is purging needed in time-series cross validation?",
        expected_sources=["paper_lopez_de_prado_afml_cpcv_2018"],
        ground_truth_claims=["purging", "leakage", "label horizon"],
    ),
    RAGFixture(
        fixture_id="rag_concept_010",
        category="concept",
        query="What does a high authority score mean in the curated source registry?",
        expected_sources=[],  # meta question — not in registry
        ground_truth_claims=[],
    ),
]


# ---------------------------------------------------------------------------
# Macro / market-context queries
# ---------------------------------------------------------------------------

_MACRO_FIXTURES: list[RAGFixture] = [
    RAGFixture(
        fixture_id="rag_macro_001",
        category="macro",
        query="If real yields fall, what historical relationship would we expect for gold?",
        expected_sources=["data_fred_dfii10"],
        ground_truth_claims=["real yield", "gold", "inverse"],
    ),
    RAGFixture(
        fixture_id="rag_macro_002",
        category="macro",
        query="Why does the trade-weighted dollar matter for XAU/USD direction?",
        expected_sources=["data_fred_dtwexbgs"],
        ground_truth_claims=["dollar", "trade weighted", "gold"],
    ),
    RAGFixture(
        fixture_id="rag_macro_003",
        category="macro",
        query="How do central bank gold purchases affect demand trends?",
        expected_sources=["report_wgc_demand_trends_q1_2026"],
        ground_truth_claims=["central bank", "purchases", "demand"],
    ),
    RAGFixture(
        fixture_id="rag_macro_004",
        category="macro",
        query="What does an FOMC minutes release typically move in fixed-income markets?",
        expected_sources=["report_fomc_minutes_template"],
        ground_truth_claims=["FOMC", "minutes", "yields"],
    ),
    RAGFixture(
        fixture_id="rag_macro_005",
        category="macro",
        query="What are typical tick value and contract size for CME gold futures?",
        expected_sources=["data_cme_gold_specs"],
        ground_truth_claims=["tick", "100 troy ounces", "contract"],
    ),
    RAGFixture(
        fixture_id="rag_macro_006",
        category="macro",
        query="How does ECB deposit rate guidance influence EUR/USD positioning?",
        expected_sources=["report_ecb_monetary_policy"],
        ground_truth_claims=["ECB", "deposit rate", "EUR"],
    ),
    RAGFixture(
        fixture_id="rag_macro_007",
        category="macro",
        query="What does the BIS quarterly review usually reveal about FX reserve diversification?",
        expected_sources=["report_bis_quarterly_2026_q1"],
        ground_truth_claims=["BIS", "reserves", "diversification"],
    ),
    RAGFixture(
        fixture_id="rag_macro_008",
        category="macro",
        query="When do vol regimes shift abruptly enough to be detected by an online changepoint algorithm?",
        expected_sources=["paper_adams_mackay_bocpd_2007"],
        ground_truth_claims=["changepoint", "regime", "online"],
    ),
    RAGFixture(
        fixture_id="rag_macro_009",
        category="macro",
        query="If managed money is record net long on gold, what does contrarian theory suggest?",
        expected_sources=["edu_investopedia_cot", "data_cftc_cot_release_schedule"],
        ground_truth_claims=["managed money", "contrarian", "net long"],
    ),
    RAGFixture(
        fixture_id="rag_macro_010",
        category="macro",
        query="What is the relationship between VIX spikes and risk-off flows into gold?",
        expected_sources=["data_fred_vixcls", "edu_investopedia_vix"],
        ground_truth_claims=["VIX", "risk off", "gold"],
    ),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


ALL_RAG_FIXTURES: list[RAGFixture] = (
    _PAPER_FIXTURES
    + _DATA_FIXTURES
    + _REPORT_FIXTURES
    + _CONCEPT_FIXTURES
    + _MACRO_FIXTURES
)


def fixtures_by_category(category: str) -> list[RAGFixture]:
    return [f for f in ALL_RAG_FIXTURES if f.category == category]
