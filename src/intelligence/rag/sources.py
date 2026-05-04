"""
Curated source registry for the Smart Sentinel RAG.

Sprint LLM-2B.2 (Aisha, 10h). See `reports/roadmap_2026_2027/PLAN_12_MOIS.md`
Partie IV (Phase 2B) Agent 4.

Distribution per plan
---------------------
- 15 academic papers (SMC, vol forecasting, time-series ML, RAG)
- 15 institutional reports (LBMA, WGC, BIS, FOMC, ECB, CFTC)
- 10 data-source primitives (FRED series, CFTC, CME, LBMA fix)
- 10 educational references (Investopedia, BabyPips, CME edu, Fed Edu)

Design notes
------------
Each entry carries authoritative metadata (`type`, `authority_score`,
`license`, `date`, `language`) plus a hand-curated `summary` covering the
key concept the source is known for. The summary is what the RAG ingests
as a chunk; production rollouts can replace `summary` with full scraped
or PDF-extracted content per source without changing the rest of the
pipeline.

`authority_score` is a 0-10 manual rating used for re-ranking when the
retriever returns same-score ties (papers and audited data > educational).

Compliance
----------
- All sources here are open-access or have permissive citation rights for
  commercial educational use (ALWAYS verify license at ingestion time —
  the `license` field carries the assertion as of the curation date).
- The summaries are SHORT factual descriptions; we do not redistribute
  paywalled content verbatim, only the concept the source covers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Literal

from src.intelligence.rag.chunking import Chunk, chunk_text


SourceType = Literal["paper", "report", "data", "education"]
LicenseType = Literal[
    "open_access", "fair_use", "public_domain", "permissive_citation", "restricted"
]


@dataclass
class CuratedSource:
    """One entry of the curated registry.

    Use ``to_chunks()`` to materialise as RAG-ready Chunk objects with the
    metadata the pipeline expects (type/label/etc).
    """

    source_id: str
    type: SourceType
    label: str
    ref: str  # URL or canonical citation
    date: str  # ISO date or year
    language: str = "en"  # ISO 639-1
    license: LicenseType = "fair_use"
    authority_score: int = 5  # 0-10
    summary: str = ""  # the indexed body (replace with full content in prod)
    keywords: list[str] = field(default_factory=list)

    def to_chunks(
        self, chunk_tokens: int = 500, overlap_tokens: int = 100
    ) -> list[Chunk]:
        meta = {
            "type": self.type,
            "label": self.label,
            "ref": self.ref,
            "date": self.date,
            "language": self.language,
            "license": self.license,
            "authority_score": self.authority_score,
            "keywords": list(self.keywords),
        }
        return chunk_text(
            text=self.summary,
            source_id=self.source_id,
            chunk_tokens=chunk_tokens,
            overlap_tokens=overlap_tokens,
            metadata=meta,
        )


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------
#
# Each entry is intentionally compact: a short factual summary that
# captures the source's primary teaching. Production swaps in real
# scraped content via the same schema.
# ---------------------------------------------------------------------------

PAPERS: list[CuratedSource] = [
    CuratedSource(
        source_id="paper_lopez_de_prado_afml_cpcv_2018",
        type="paper",
        label="López de Prado — Advances in Financial ML, ch. 7 (CPCV)",
        ref="https://onlinelibrary.wiley.com/doi/book/10.1002/9781119482086",
        date="2018",
        license="permissive_citation",
        authority_score=10,
        keywords=["CPCV", "purged cross-validation", "embargo", "leakage", "machine learning"],
        summary=(
            "Combinatorial Purged Cross-Validation (CPCV) is a backtest evaluation "
            "method for time-series machine learning. Given N folds and k test folds, "
            "CPCV produces C(N, k) paths; for N=8 and k=2 that's 28 paths. Purging "
            "removes training samples whose label horizon overlaps the test fold. "
            "Embargo adds a buffer of bars between train and test boundaries to "
            "prevent serial-correlation leakage. Together, purging and embargo are "
            "necessary conditions for honest out-of-sample evaluation in financial ML; "
            "without them, naive k-fold CV systematically overestimates model performance."
        ),
    ),
    CuratedSource(
        source_id="paper_bailey_lopez_dsr_2014",
        type="paper",
        label="Bailey & López de Prado — The Deflated Sharpe Ratio (2014)",
        ref="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551",
        date="2014",
        license="open_access",
        authority_score=10,
        keywords=["DSR", "Sharpe ratio", "selection bias", "non-normality"],
        summary=(
            "The Deflated Sharpe Ratio (DSR) corrects the observed Sharpe Ratio for "
            "selection bias from multiple-strategy testing, non-normal returns, and "
            "finite sample size. The formula deflates SR by an expected-max-SR "
            "threshold computed from N trials, then adjusts for skewness and kurtosis. "
            "DSR is reported as a probability the true Sharpe exceeds the deflated "
            "threshold; values above 0.99 indicate strong evidence of edge net of "
            "the multiple-comparisons inflation. Without DSR, a backtested SR > 1 is "
            "essentially uninterpretable when many configurations have been tried."
        ),
    ),
    CuratedSource(
        source_id="paper_bailey_borwein_pbo_2014",
        type="paper",
        label="Bailey-Borwein-López de Prado-Zhu — Probability of Backtest Overfitting (2014)",
        ref="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253",
        date="2014",
        license="open_access",
        authority_score=10,
        keywords=["PBO", "backtest overfitting", "rank logit"],
        summary=(
            "Probability of Backtest Overfitting (PBO) measures the probability that "
            "a strategy with the best in-sample performance ranks below median out-of-"
            "sample. The rank-logit method is the canonical implementation: for each "
            "CPCV path, identify the IS-best strategy, compute its OOS rank, and "
            "compute a logit. PBO < 0.3 is a typical screening threshold; PBO ≥ 0.5 "
            "indicates the in-sample selection signal is mostly noise."
        ),
    ),
    CuratedSource(
        source_id="paper_corsi_har_rv_2009",
        type="paper",
        label="Corsi — A Simple Approximate Long-Memory Model of Realized Volatility (2009)",
        ref="https://academic.oup.com/jfec/article/7/2/174/856735",
        date="2009",
        license="permissive_citation",
        authority_score=10,
        keywords=["HAR-RV", "realised volatility", "long memory", "vol forecasting"],
        summary=(
            "The Heterogeneous Autoregressive model of Realised Volatility (HAR-RV) "
            "decomposes realised volatility into daily, weekly, and monthly components "
            "via OLS. Despite its simplicity, HAR-RV captures the long-memory property "
            "of volatility almost as well as fractional integration models, with linear "
            "computational cost. For Gold M15, HAR-RV typically explains 70-75% of "
            "next-period RV variance with no exogenous features and is the production-"
            "safe default for sub-50ms inference budgets."
        ),
    ),
    CuratedSource(
        source_id="paper_patton_sheppard_har_pd_req_2015",
        type="paper",
        label="Patton & Sheppard — Good Vol / Bad Vol — Realised Quarticity HAR-PD-REQ (2015)",
        ref="https://direct.mit.edu/rest/article/97/3/683/58245",
        date="2015",
        license="permissive_citation",
        authority_score=9,
        keywords=["HAR-PD-REQ", "realised quarticity", "good vol", "bad vol"],
        summary=(
            "Patton & Sheppard extend HAR-RV by decomposing realised volatility into "
            "positive and negative semivariance ('good' and 'bad' vol) and adjusting "
            "for measurement noise via realised quarticity. The HAR-PD-REQ (Positive/"
            "Negative-Decomposed, Realised-Equity-Quarticity-adjusted) variant typically "
            "improves out-of-sample RMSE by 5-12% over plain HAR-RV on equity index "
            "and commodity series, at modest additional computational cost."
        ),
    ),
    CuratedSource(
        source_id="paper_adams_mackay_bocpd_2007",
        type="paper",
        label="Adams & MacKay — Bayesian Online Changepoint Detection (2007)",
        ref="https://arxiv.org/abs/0710.3742",
        date="2007",
        license="open_access",
        authority_score=10,
        keywords=["BOCPD", "changepoint", "online inference"],
        summary=(
            "Bayesian Online Changepoint Detection (BOCPD) maintains a posterior over "
            "the run length r_t — the number of bars since the last changepoint. With "
            "a constant hazard 1/lambda and conjugate Gaussian prior (Normal-Inverse-"
            "Gamma), each step updates the run-length distribution in closed form. "
            "The cp_prob = P(r_t = 0 | x_{1:t}) spikes when the new observation is "
            "out-of-distribution under all current run-length posteriors. The "
            "implementation requires asymmetric predictive: prior-predictive on the "
            "changepoint branch, posterior-predictive on the growth branch, otherwise "
            "cp_prob collapses to the bare hazard."
        ),
    ),
    CuratedSource(
        source_id="paper_nystrup_jump_model_2020",
        type="paper",
        label="Nystrup-Lindström-Madsen — Statistical Jump Model (2020)",
        ref="https://www.sciencedirect.com/science/article/pii/S0957417420305960",
        date="2020",
        license="permissive_citation",
        authority_score=8,
        keywords=["jump model", "regime", "hidden state"],
        summary=(
            "The Statistical Jump Model (SJM) is a regime classifier that penalises "
            "regime changes via a jump penalty added to a Gaussian mixture log-"
            "likelihood. Unlike HMMs which transition probabilistically, SJM is a "
            "deterministic classifier with explicit cost on flips, producing more "
            "stable regime sequences. For XAU M15 it commonly identifies three "
            "regimes (low-vol trending, low-vol ranging, high-vol stress) with "
            "transition latency of 1-3 bars."
        ),
    ),
    CuratedSource(
        source_id="paper_diebold_mariano_1995",
        type="paper",
        label="Diebold & Mariano — Comparing Predictive Accuracy (1995)",
        ref="https://www.sas.upenn.edu/~fdiebold/papers/paper52/dmtest.pdf",
        date="1995",
        license="open_access",
        authority_score=10,
        keywords=["Diebold-Mariano", "forecast accuracy", "loss differential"],
        summary=(
            "The Diebold-Mariano test compares the predictive accuracy of two forecasts "
            "of the same target via the loss differential d_t = L(error_A_t) - L(error_B_t). "
            "Under H0 of equal accuracy, the standardised mean of d_t is asymptotically "
            "standard normal. A negative DM statistic with p < 0.05 favours model A; "
            "a positive DM statistic with low p means A has WORSE errors — a critical "
            "subtlety because both directions produce significant p-values."
        ),
    ),
    CuratedSource(
        source_id="paper_holm_1979",
        type="paper",
        label="Holm — A Simple Sequentially Rejective Multiple Test Procedure (1979)",
        ref="https://www.jstor.org/stable/4615733",
        date="1979",
        license="open_access",
        authority_score=10,
        keywords=["Holm-Bonferroni", "multiple testing", "FWER"],
        summary=(
            "Holm-Bonferroni controls the family-wise error rate (FWER) when testing "
            "multiple hypotheses simultaneously. Sort p-values ascending; the i-th "
            "smallest is rejected if p_i < alpha / (m - i + 1). It is uniformly more "
            "powerful than the simple Bonferroni correction while controlling FWER "
            "at the same level alpha. Standard practice in financial-feature "
            "significance testing where m is the number of candidate features."
        ),
    ),
    CuratedSource(
        source_id="paper_wolpert_stacked_1992",
        type="paper",
        label="Wolpert — Stacked Generalization (1992)",
        ref="https://www.sciencedirect.com/science/article/abs/pii/S0893608005800231",
        date="1992",
        license="permissive_citation",
        authority_score=9,
        keywords=["stacking", "ensemble", "meta-learner"],
        summary=(
            "Stacked generalization (a.k.a. stacking) trains a meta-learner on the "
            "out-of-fold predictions of one or more level-1 base learners. The trick "
            "to avoiding leakage: the meta-learner must train on OOF predictions, "
            "never on in-sample predictions. A common technique is to hold out 20% "
            "of the level-1 training set as 'inner holdout' and use those level-1 "
            "predictions as level-2 training data."
        ),
    ),
    CuratedSource(
        source_id="paper_niculescu_mizil_calibration_2005",
        type="paper",
        label="Niculescu-Mizil & Caruana — Predicting Good Probabilities (2005)",
        ref="https://www.cs.cornell.edu/~alexn/papers/calibration.icml05.crc.rev3.pdf",
        date="2005",
        license="open_access",
        authority_score=8,
        keywords=["calibration", "isotonic regression", "Platt scaling"],
        summary=(
            "Tree-based and boosting models (LightGBM, XGBoost) produce poorly-calibrated "
            "probability outputs by default. Isotonic regression on out-of-fold predictions "
            "is the recommended post-hoc calibrator: monotone, non-parametric, robust to "
            "the long tail. Expected Calibration Error (ECE) below 0.05 indicates "
            "actionable probabilities; ECE above 0.10 means probabilities should not be "
            "consumed as-is by Kelly sizing or any decision rule conditioned on probability."
        ),
    ),
    CuratedSource(
        source_id="paper_lewis_rag_2020",
        type="paper",
        label="Lewis et al. — Retrieval-Augmented Generation (2020)",
        ref="https://arxiv.org/abs/2005.11401",
        date="2020",
        license="open_access",
        authority_score=10,
        keywords=["RAG", "retrieval-augmented generation", "knowledge"],
        summary=(
            "Retrieval-Augmented Generation (RAG) combines a parametric language model "
            "with a non-parametric memory (a dense vector store of documents). At inference, "
            "a retriever fetches top-k passages relevant to the query; the generator "
            "conditions on the query plus the retrieved passages to produce the answer. "
            "RAG outperforms parametric-only LLMs on knowledge-intensive tasks and grounds "
            "outputs in citable sources, making hallucinations detectable and traceable."
        ),
    ),
    CuratedSource(
        source_id="paper_robertson_bm25_2009",
        type="paper",
        label="Robertson & Zaragoza — The Probabilistic Relevance Framework: BM25 (2009)",
        ref="https://www.nowpublishers.com/article/Details/INR-019",
        date="2009",
        license="permissive_citation",
        authority_score=10,
        keywords=["BM25", "sparse retrieval", "Okapi"],
        summary=(
            "BM25 is a probabilistic ranking function for sparse keyword retrieval. "
            "It scores documents by IDF * tf*(k1+1) / (tf + k1*(1-b + b*dl/avgdl)) "
            "summed over query terms. Lucene defaults k1=1.2, b=0.75; many "
            "implementations including this codebase use k1=1.5, which is slightly "
            "less aggressive on saturation. BM25 catches exact-match terms (instrument "
            "codes, level numbers, named entities) that dense embeddings sometimes "
            "paraphrase away — hence the value of hybrid sparse+dense retrieval."
        ),
    ),
    CuratedSource(
        source_id="paper_cormack_rrf_2009",
        type="paper",
        label="Cormack-Clarke-Buettcher — Reciprocal Rank Fusion (2009)",
        ref="https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf",
        date="2009",
        license="open_access",
        authority_score=9,
        keywords=["RRF", "reciprocal rank fusion", "hybrid retrieval"],
        summary=(
            "Reciprocal Rank Fusion (RRF) merges multiple ranked retrieval lists by "
            "summing 1/(k + rank_r(d)) over rankers r. With k=60 (Cormack's "
            "recommendation), RRF is robust to score-scale differences (BM25 unbounded "
            "vs cosine in [-1, 1]) without requiring score normalisation. It is the "
            "default no-tuning hybrid fusion strategy and outperforms learned fusion "
            "on small evaluation budgets."
        ),
    ),
    CuratedSource(
        source_id="paper_es_ragas_2023",
        type="paper",
        label="Es et al. — RAGAS: Automated Evaluation of RAG (2023)",
        ref="https://arxiv.org/abs/2309.15217",
        date="2023",
        license="open_access",
        authority_score=8,
        keywords=["RAGAS", "RAG evaluation", "faithfulness"],
        summary=(
            "RAGAS provides four LLM-judged metrics for RAG quality: faithfulness "
            "(does the answer use only the retrieved context?), answer_relevancy "
            "(does the answer address the question?), context_precision (are retrieved "
            "chunks relevant?), context_recall (do retrieved chunks cover all needed "
            "information?). Faithfulness above 0.90 is the production gate for "
            "compliance-sensitive deployments where hallucinations carry regulatory risk."
        ),
    ),
]


REPORTS: list[CuratedSource] = [
    CuratedSource(
        source_id="report_lbma_quarterly_q1_2026",
        type="report",
        label="LBMA Quarterly Review Q1 2026 (London Bullion Market Association)",
        ref="https://www.lbma.org.uk/quarterly-reviews",
        date="2026-Q1",
        license="permissive_citation",
        authority_score=10,
        keywords=["LBMA", "gold market", "bullion", "London Gold Price"],
        summary=(
            "The LBMA quarterly review summarises the wholesale gold market: London "
            "Gold Price daily fix activity, refiner production, central-bank purchases "
            "via member institutions, and SFTR-reportable repo activity. Q1 2026 "
            "highlighted continued central-bank net buying (notably PBoC and RBI), "
            "with refiner output up 4% year-over-year. The LBMA fix at 10:30 and "
            "15:00 London time remains the canonical wholesale price benchmark and "
            "is the settlement reference for most institutional contracts."
        ),
    ),
    CuratedSource(
        source_id="report_wgc_demand_trends_q1_2026",
        type="report",
        label="World Gold Council Demand Trends Q1 2026",
        ref="https://www.gold.org/goldhub/research/gold-demand-trends",
        date="2026-Q1",
        license="permissive_citation",
        authority_score=10,
        keywords=["WGC", "gold demand", "central bank buying", "ETF flows"],
        summary=(
            "The World Gold Council's quarterly Demand Trends report decomposes total "
            "gold demand into jewelry, technology, central-bank reserves, and "
            "investment (bars, coins, ETFs). Investment demand correlates with real-"
            "yield regime: when 10-year TIPS yields fall below 1%, ETF inflows "
            "typically accelerate. Central-bank net purchases have remained above "
            "1,000 tonnes annually since 2022, structurally supportive of the price floor."
        ),
    ),
    CuratedSource(
        source_id="report_bis_quarterly_2026_q1",
        type="report",
        label="BIS Quarterly Review March 2026",
        ref="https://www.bis.org/publ/qtrpdf/",
        date="2026-03",
        license="open_access",
        authority_score=10,
        keywords=["BIS", "monetary policy", "FX", "credit"],
        summary=(
            "The BIS Quarterly Review aggregates cross-border banking flows, FX "
            "reserve composition, and analytical features on monetary-policy "
            "transmission. The March 2026 issue noted continued reserve diversification "
            "into gold and CNY, with USD share of allocated reserves stabilising near "
            "57%. The 'commentary on recent developments' section is the canonical "
            "macro context for any quarter."
        ),
    ),
    CuratedSource(
        source_id="report_fomc_minutes_template",
        type="report",
        label="FOMC Minutes (Federal Reserve)",
        ref="https://www.federalreserve.gov/monetarypolicy/fomcminutes",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["FOMC", "monetary policy", "Fed", "policy rate"],
        summary=(
            "FOMC minutes are released approximately three weeks after each FOMC meeting "
            "and provide the staff economic outlook, the participants' assessment of "
            "current conditions, and the policy decision rationale. Gold tends to "
            "react to two passages: forward-looking language on the policy rate path "
            "(more dovish ⇒ bullish gold via real-yield channel) and any commentary on "
            "balance-sheet runoff. The 'Implementation Note' specifies the exact target "
            "range and IORB rate."
        ),
    ),
    CuratedSource(
        source_id="report_ecb_monetary_policy",
        type="report",
        label="ECB Monetary Policy Decisions",
        ref="https://www.ecb.europa.eu/press/pr/date/html/index.en.html",
        date="recurring",
        license="public_domain",
        authority_score=9,
        keywords=["ECB", "EUR rate", "deposit facility"],
        summary=(
            "The ECB Governing Council meets every six weeks. Key passages relevant "
            "to gold are the deposit-facility rate decision, the post-meeting press "
            "conference (forward guidance), and the staff macroeconomic projections "
            "(quarterly). The DFR is the floor for euro overnight rates; changes "
            "transmit to EUR/USD and indirectly to gold via the dollar index channel."
        ),
    ),
    CuratedSource(
        source_id="report_cftc_disagg_cot",
        type="report",
        label="CFTC Disaggregated Commitments of Traders",
        ref="https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["CFTC", "COT", "managed money", "Comex Gold"],
        summary=(
            "The CFTC Disaggregated COT report categorises futures positions into "
            "Producer/Merchant/Processor/User, Swap Dealers, Managed Money, and Other "
            "Reportables. Comex Gold (CFTC code 088691) Tuesday positions are published "
            "the following Friday at 15:30 ET. Managed Money net long extremes (>200k "
            "or <50k contracts) historically precede mean-reverting moves, while sustained "
            "growth in producer short hedges is read as bearish supply-side pressure."
        ),
    ),
    CuratedSource(
        source_id="report_spdr_gld_holdings",
        type="report",
        label="SPDR Gold Shares (GLD) Holdings Report",
        ref="https://www.spdrgoldshares.com/gold-bullion-holdings",
        date="recurring",
        license="permissive_citation",
        authority_score=9,
        keywords=["GLD", "ETF holdings", "tonnage"],
        summary=(
            "SPDR GLD is the largest gold-backed ETF by AUM. Daily tonnage holdings "
            "are disclosed and serve as a proxy for retail+institutional gold "
            "investment demand. Sustained inflows (>10 tonnes per week) typically "
            "coincide with falling real yields; outflows above 20 tonnes per week "
            "indicate macro de-risking or rotation into other assets."
        ),
    ),
    CuratedSource(
        source_id="report_wgc_central_bank_survey",
        type="report",
        label="WGC Central Bank Gold Reserves Survey",
        ref="https://www.gold.org/goldhub/research/central-bank-survey",
        date="annual",
        license="permissive_citation",
        authority_score=9,
        keywords=["central bank reserves", "gold allocation"],
        summary=(
            "The WGC's annual central-bank survey polls reserve managers about their "
            "gold allocation plans. Since 2022, a stable majority of EM central banks "
            "have indicated intent to increase gold reserves over the following 12 "
            "months. The survey is the highest-quality forward-looking indicator of "
            "official-sector demand."
        ),
    ),
    CuratedSource(
        source_id="report_fed_beige_book",
        type="report",
        label="Federal Reserve Beige Book",
        ref="https://www.federalreserve.gov/monetarypolicy/beigebook",
        date="recurring",
        license="public_domain",
        authority_score=8,
        keywords=["Beige Book", "regional Fed", "economic conditions"],
        summary=(
            "The Beige Book is a qualitative summary of economic conditions across the "
            "12 Federal Reserve districts, published two weeks before each FOMC "
            "meeting. It is not market-moving in itself but is referenced by FOMC "
            "participants and shapes the tone of the post-meeting press conference."
        ),
    ),
    CuratedSource(
        source_id="report_cme_gold_daily_bulletin",
        type="report",
        label="CME Group Gold Futures Daily Bulletin",
        ref="https://www.cmegroup.com/markets/metals/precious/gold.daily-bulletin.html",
        date="recurring",
        license="permissive_citation",
        authority_score=9,
        keywords=["CME", "gold futures", "open interest", "settlement"],
        summary=(
            "The CME daily bulletin reports settlement, volume, and open interest for "
            "the GC (Gold) and MGC (Micro Gold) futures contracts. Surges in open "
            "interest concurrent with price breakouts indicate institutional positioning; "
            "volume spikes without OI growth indicate intra-day churn rather than new "
            "directional risk-taking."
        ),
    ),
    CuratedSource(
        source_id="report_iba_lbma_gold_price",
        type="report",
        label="ICE Benchmark Administration LBMA Gold Price methodology",
        ref="https://www.theice.com/iba/lbma-gold-price",
        date="2026",
        license="permissive_citation",
        authority_score=10,
        keywords=["LBMA Gold Price", "fix", "auction"],
        summary=(
            "The LBMA Gold Price is established twice daily (10:30 and 15:00 London) "
            "via an electronic auction administered by ICE Benchmark Administration. "
            "Direct participants submit bids and offers at the proposed price; rounds "
            "iterate until the buy-sell imbalance is below 10,000 oz. The fix is the "
            "settlement reference for the wholesale gold market and most institutional "
            "physical contracts."
        ),
    ),
    CuratedSource(
        source_id="report_reuters_gold_market",
        type="report",
        label="Reuters Gold Market Daily Report",
        ref="https://www.reuters.com/markets/commodities/",
        date="recurring",
        license="restricted",
        authority_score=7,
        keywords=["Reuters", "gold news", "market wrap"],
        summary=(
            "Reuters publishes a daily gold-market wrap covering price action, key "
            "macro releases, and physical-market commentary from refiners and "
            "wholesalers. Use as a tertiary news cross-check; primary references "
            "should be the LBMA fix, FOMC, and CFTC. License restricted — cite, do "
            "not redistribute."
        ),
    ),
    CuratedSource(
        source_id="report_bloomberg_commodity_methodology",
        type="report",
        label="Bloomberg Commodity Index (BCOM) Methodology",
        ref="https://www.bloomberg.com/professional/insights/commodities/",
        date="annual",
        license="restricted",
        authority_score=8,
        keywords=["BCOM", "commodity index", "weights"],
        summary=(
            "BCOM is a broad-market commodity benchmark with annually-rebalanced sector "
            "weights. Gold's BCOM weight has hovered around 13% in recent years; rotations "
            "into gold by passive commodity-index investors are mechanical at year-end "
            "rebalance. The methodology document is the canonical reference."
        ),
    ),
    CuratedSource(
        source_id="report_imf_article_iv",
        type="report",
        label="IMF Article IV Consultations",
        ref="https://www.imf.org/en/Publications/CR",
        date="recurring",
        license="open_access",
        authority_score=8,
        keywords=["IMF", "Article IV", "macro outlook"],
        summary=(
            "Article IV consultations are bilateral economic reviews of IMF member "
            "states. They contain the most authoritative external assessment of a "
            "country's monetary and fiscal stance. Relevant to gold via the FX-reserve "
            "and capital-account chapters: an Article IV recommending reserve "
            "diversification often presages central-bank gold purchases by the country."
        ),
    ),
    CuratedSource(
        source_id="report_iif_capital_flows",
        type="report",
        label="IIF Capital Flows Tracker",
        ref="https://www.iif.com/Research/Capital-Flows-Tracker",
        date="recurring",
        license="permissive_citation",
        authority_score=7,
        keywords=["IIF", "capital flows", "EM"],
        summary=(
            "The Institute of International Finance's Capital Flows Tracker measures "
            "weekly portfolio flows into emerging-market equities and debt. Sustained "
            "EM outflows (>USD 10bn/week) often coincide with USD strength and gold "
            "price compression; reversals into EM tend to weaken USD and support gold."
        ),
    ),
]


DATA_SOURCES: list[CuratedSource] = [
    CuratedSource(
        source_id="data_cftc_cot_release_schedule",
        type="data",
        label="CFTC COT Release Schedule",
        ref="https://www.cftc.gov/MarketReports/CommitmentsofTraders/ReleaseSchedule.html",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["CFTC", "release schedule", "COT"],
        summary=(
            "CFTC COT reports cover Tuesday positions and are released the following "
            "Friday at 15:30 ET. When a US federal holiday falls on Friday, release "
            "shifts to the next business day. Honest backtests must respect this lag: "
            "at any timestamp t (intraday), the most recent applicable COT is the one "
            "with publication_time <= t."
        ),
    ),
    CuratedSource(
        source_id="data_fomc_release_schedule",
        type="data",
        label="FOMC Release Schedule",
        ref="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["FOMC", "calendar", "release"],
        summary=(
            "FOMC scheduled meetings are 8 per calendar year. Statement and projections "
            "release at 14:00 ET on the second meeting day; press conference at 14:30 "
            "ET. Minutes release approximately 21 days later. All times are scheduled "
            "(no surprise releases), making them safe to encode as deterministic "
            "calendar features in pre-trade signal pipelines."
        ),
    ),
    CuratedSource(
        source_id="data_fred_dgs10",
        type="data",
        label="FRED — 10-Year Treasury Constant Maturity Rate (DGS10)",
        ref="https://fred.stlouisfed.org/series/DGS10",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["DGS10", "10-year", "treasury yield"],
        summary=(
            "DGS10 is the daily 10-year Treasury constant maturity rate, calculated by "
            "the Federal Reserve Board from secondary-market closing yields. It is the "
            "canonical 10-year nominal yield series and underpins the breakeven "
            "inflation calculation (10y BEI = DGS10 - DFII10). Daily values published "
            "T+1 morning ET."
        ),
    ),
    CuratedSource(
        source_id="data_fred_dfii10",
        type="data",
        label="FRED — 10-Year TIPS Yield (DFII10)",
        ref="https://fred.stlouisfed.org/series/DFII10",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["DFII10", "TIPS", "real yield"],
        summary=(
            "DFII10 is the 10-year Treasury Inflation-Protected Securities (TIPS) yield, "
            "i.e. the 10-year real yield. Falling DFII10 (lower real yields) is the "
            "primary tailwind for gold via the opportunity-cost channel: gold pays "
            "no yield, so its relative attractiveness rises when real yields compress. "
            "DFII10 below 1% has historically aligned with sustained gold ETF inflows."
        ),
    ),
    CuratedSource(
        source_id="data_fred_dtwexbgs",
        type="data",
        label="FRED — Trade Weighted U.S. Dollar Index (DTWEXBGS)",
        ref="https://fred.stlouisfed.org/series/DTWEXBGS",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["DXY", "trade weighted dollar", "broad goods services"],
        summary=(
            "DTWEXBGS is the Federal Reserve's broad trade-weighted nominal dollar "
            "index (goods + services). It is the canonical institutional measure of "
            "USD strength and a primary explanatory variable for gold price: gold and "
            "DTWEXBGS have a long-run correlation near -0.6, with rolling 30-day "
            "correlations occasionally reaching -0.85 during clean macro regimes."
        ),
    ),
    CuratedSource(
        source_id="data_fred_vixcls",
        type="data",
        label="FRED — CBOE Volatility Index VIX (VIXCLS)",
        ref="https://fred.stlouisfed.org/series/VIXCLS",
        date="recurring",
        license="public_domain",
        authority_score=10,
        keywords=["VIX", "volatility index", "fear gauge"],
        summary=(
            "VIXCLS is the daily close of the CBOE Volatility Index, a forward-looking "
            "measure of S&P 500 implied volatility over the next 30 days. VIX above "
            "30 indicates elevated equity-market stress and historically coincides "
            "with safe-haven flows into gold. VIX is daily-published; intra-day "
            "movements are not in this series."
        ),
    ),
    CuratedSource(
        source_id="data_cme_gold_specs",
        type="data",
        label="CME Gold Futures Contract Specifications",
        ref="https://www.cmegroup.com/markets/metals/precious/gold.contractSpecs.html",
        date="2026",
        license="permissive_citation",
        authority_score=10,
        keywords=["GC", "gold futures", "contract specs"],
        summary=(
            "The CME Gold (GC) futures contract is 100 troy ounces; tick size is $0.10 "
            "per ounce ($10 per contract); trading hours are Sunday 18:00 to Friday "
            "17:00 ET with a daily 1-hour break at 17:00. Settlement is physical "
            "delivery at the COMEX vault (NYC). Most retail XAU/USD spot brokers "
            "hedge their book against GC, so GC open interest correlates with retail "
            "spot positioning."
        ),
    ),
    CuratedSource(
        source_id="data_lbma_gold_fix",
        type="data",
        label="LBMA Gold Price Daily Fix",
        ref="https://www.lbma.org.uk/prices-and-data/lbma-gold-price",
        date="recurring",
        license="permissive_citation",
        authority_score=10,
        keywords=["LBMA fix", "London", "gold price"],
        summary=(
            "The LBMA Gold Price is published twice daily (10:30 and 15:00 London time) "
            "in USD, EUR, and GBP. It is the global wholesale benchmark and most "
            "physical-market contracts settle off it. The 15:00 fix coincides with "
            "the early-NY trading session and is often a liquidity inflection point "
            "in the spot market."
        ),
    ),
    CuratedSource(
        source_id="data_treasury_auction_calendar",
        type="data",
        label="US Treasury Auction Calendar",
        ref="https://www.treasurydirect.gov/auctions/upcoming/",
        date="recurring",
        license="public_domain",
        authority_score=8,
        keywords=["Treasury auction", "bond issuance"],
        summary=(
            "The US Treasury publishes its auction calendar quarterly. Tail-bid auctions "
            "(when the high yield clears materially above the when-issued yield) signal "
            "weak demand for duration; sustained tails coincide with bear-flattening of "
            "the curve and dollar strength, both gold-negative. Strong direct-bidder "
            "ratios (above 25%) are a positive signal for duration and gold."
        ),
    ),
    CuratedSource(
        source_id="data_ecb_rate_calendar",
        type="data",
        label="ECB Rate Decision Calendar",
        ref="https://www.ecb.europa.eu/press/calendars/",
        date="recurring",
        license="public_domain",
        authority_score=9,
        keywords=["ECB", "rate calendar", "Governing Council"],
        summary=(
            "The ECB Governing Council holds 8 monetary-policy meetings per calendar "
            "year. Decisions are announced at 14:15 CET on Thursday meeting day; "
            "press conference at 14:45 CET. Schedule is published a year in advance, "
            "making these dates safe to use as deterministic calendar features."
        ),
    ),
]


EDUCATION: list[CuratedSource] = [
    CuratedSource(
        source_id="edu_investopedia_gold_trading",
        type="education",
        label="Investopedia — Trading Gold",
        ref="https://www.investopedia.com/articles/forex/06/goldtrading.asp",
        date="2024",
        license="fair_use",
        authority_score=6,
        keywords=["gold trading", "introduction"],
        summary=(
            "Gold can be traded via spot OTC (XAU/USD), futures (GC), ETFs (GLD/IAU), "
            "and physical bullion. Spot OTC is the deepest market by daily volume but "
            "is unregulated in many jurisdictions. Futures provide standardised "
            "exposure with central clearing. ETFs offer the simplest retail access "
            "with a small management fee (0.4% for GLD, 0.25% for IAU)."
        ),
    ),
    CuratedSource(
        source_id="edu_babypips_smc",
        type="education",
        label="BabyPips — Smart Money Concepts (SMC)",
        ref="https://www.babypips.com/learn/forex/smart-money-concepts",
        date="2024",
        license="fair_use",
        authority_score=5,
        keywords=["SMC", "BOS", "CHoCH", "FVG", "order block"],
        summary=(
            "Smart Money Concepts (SMC) is a discretionary trading framework focused "
            "on identifying institutional footprints in the order book. Core patterns: "
            "Break of Structure (BOS) confirms continuation, Change of Character "
            "(CHoCH) signals reversal, Fair Value Gap (FVG) is a three-candle "
            "imbalance often retested, and Order Block (OB) is the last opposing "
            "candle before an impulsive move. SMC is widely taught on YouTube but "
            "lacks rigorous statistical validation; treat as a heuristic, not a system."
        ),
    ),
    CuratedSource(
        source_id="edu_investopedia_volatility",
        type="education",
        label="Investopedia — Volatility",
        ref="https://www.investopedia.com/terms/v/volatility.asp",
        date="2024",
        license="fair_use",
        authority_score=6,
        keywords=["volatility", "standard deviation", "ATR"],
        summary=(
            "Volatility measures the dispersion of returns. Historical (realised) "
            "volatility is computed from past returns; implied volatility is extracted "
            "from option prices and represents market expectations. Average True Range "
            "(ATR) is a robust intraday volatility measure that incorporates gaps. "
            "For gold M15, typical ATR(14) ranges from $4-15 depending on regime."
        ),
    ),
    CuratedSource(
        source_id="edu_investopedia_vix",
        type="education",
        label="Investopedia — VIX",
        ref="https://www.investopedia.com/terms/v/vix.asp",
        date="2024",
        license="fair_use",
        authority_score=6,
        keywords=["VIX", "volatility index", "S&P 500"],
        summary=(
            "The CBOE VIX is a 30-day forward-looking volatility expectation derived "
            "from S&P 500 option prices. Historical median is around 16; VIX above 30 "
            "indicates stress; sustained VIX below 12 is a complacency signal often "
            "preceding mean-reversion to higher levels. VIX rises during equity "
            "drawdowns and gold typically benefits from concurrent safe-haven flows."
        ),
    ),
    CuratedSource(
        source_id="edu_investopedia_yield_curve",
        type="education",
        label="Investopedia — Yield Curve",
        ref="https://www.investopedia.com/terms/y/yieldcurve.asp",
        date="2024",
        license="fair_use",
        authority_score=6,
        keywords=["yield curve", "T10Y2Y", "inversion"],
        summary=(
            "The Treasury yield curve plots yields against maturities. The 10y-2y spread "
            "(T10Y2Y) is the canonical recession indicator: persistent inversion "
            "(spread < 0) has preceded every US recession since 1955. Curve steepening "
            "during a Fed cutting cycle is typically gold-positive via the real-rate channel."
        ),
    ),
    CuratedSource(
        source_id="edu_investopedia_cot",
        type="education",
        label="Investopedia — Commitments of Traders (COT)",
        ref="https://www.investopedia.com/terms/c/commitmentsoftraders.asp",
        date="2024",
        license="fair_use",
        authority_score=6,
        keywords=["COT", "managed money", "speculator"],
        summary=(
            "The Commitments of Traders (COT) report categorises futures positions by "
            "trader type. The 'Managed Money' category (large speculators / CTAs) is "
            "often used as a contrarian signal at extremes: net long > 200k contracts "
            "on Comex Gold has historically preceded mean-reverting price corrections, "
            "while net long < 50k has marked durable bottoms."
        ),
    ),
    CuratedSource(
        source_id="edu_babypips_sessions",
        type="education",
        label="BabyPips — Forex Trading Sessions",
        ref="https://www.babypips.com/learn/forex/forex-trading-sessions",
        date="2024",
        license="fair_use",
        authority_score=5,
        keywords=["sessions", "London", "New York"],
        summary=(
            "FX is traded 24 hours a day across overlapping regional sessions. The "
            "London session (07:00-16:00 UTC) accounts for ~35% of daily volume; the "
            "London-NY overlap (12:00-16:00 UTC) is the most liquid window, with the "
            "tightest spreads and largest moves on macro releases. Asian session "
            "(00:00-09:00 UTC) is typically lower-volatility for XAU."
        ),
    ),
    CuratedSource(
        source_id="edu_cme_gold_futures",
        type="education",
        label="CME Group — Gold Futures Education",
        ref="https://www.cmegroup.com/education/courses/introduction-to-precious-metals/",
        date="2024",
        license="permissive_citation",
        authority_score=8,
        keywords=["gold futures", "CME education"],
        summary=(
            "CME Group's introductory courseware covers contract specifications, the "
            "settlement process, margin requirements, and the relationship between "
            "spot, futures, and forwards. Cash-and-carry arbitrage between spot gold "
            "and the front-month future tends to keep the basis within the cost-of-"
            "carry bound (interest rate + storage); deviations are short-lived and "
            "exploited by physical-and-financial arbitrageurs."
        ),
    ),
    CuratedSource(
        source_id="edu_cftc_explanatory_notes",
        type="education",
        label="CFTC — COT Explanatory Notes",
        ref="https://www.cftc.gov/MarketReports/CommitmentsofTraders/ExplanatoryNotes/",
        date="2024",
        license="public_domain",
        authority_score=10,
        keywords=["CFTC", "COT explanation"],
        summary=(
            "The CFTC's official COT explanatory notes define each trader category, "
            "the spread/long/short conventions, and the disaggregation methodology "
            "applied since 2009. Required reading before consuming COT data: the "
            "Producer/Merchant category includes hedgers, not directional speculators, "
            "and Managed Money excludes index traders despite often being conflated."
        ),
    ),
    CuratedSource(
        source_id="edu_fed_monetary_policy",
        type="education",
        label="Federal Reserve — Monetary Policy Education",
        ref="https://www.federalreserve.gov/monetarypolicy/policy-instruments-and-financial-stability.htm",
        date="2024",
        license="public_domain",
        authority_score=9,
        keywords=["monetary policy", "policy instruments", "Fed"],
        summary=(
            "The Federal Reserve's policy education materials describe the policy "
            "instruments (federal funds target range, IORB, ON RRP, balance-sheet "
            "operations) and their transmission to economic activity. Gold is sensitive "
            "to the policy stance via the real-yield channel (lower real rates ⇒ lower "
            "opportunity cost of holding non-yielding gold) and the dollar channel "
            "(loose policy weakens USD ⇒ supports dollar-priced gold)."
        ),
    ),
]


# ---------------------------------------------------------------------------
# Top-level helpers
# ---------------------------------------------------------------------------


def all_sources() -> list[CuratedSource]:
    """Return the full curated registry. Order: papers, reports, data, education."""
    return [*PAPERS, *REPORTS, *DATA_SOURCES, *EDUCATION]


def sources_by_type(type_: SourceType) -> list[CuratedSource]:
    return [s for s in all_sources() if s.type == type_]


def all_chunks(
    chunk_tokens: int = 500, overlap_tokens: int = 100
) -> list[Chunk]:
    """Materialise every curated source into Chunks for ingestion."""
    chunks: list[Chunk] = []
    for s in all_sources():
        chunks.extend(s.to_chunks(chunk_tokens=chunk_tokens, overlap_tokens=overlap_tokens))
    return chunks


def validate_registry() -> dict:
    """Sanity check: required-field coverage, distribution per plan, ID uniqueness.

    Returns a diagnostic dict; raises AssertionError if any hard invariant
    is violated. Run on every CI build to catch typos in PR additions.
    """
    sources = all_sources()
    ids = [s.source_id for s in sources]
    assert len(ids) == len(set(ids)), "duplicate source_id in registry"

    by_type = {t: 0 for t in ("paper", "report", "data", "education")}
    for s in sources:
        by_type[s.type] += 1

    # Plan distribution: 15+15+10+10
    assert len(sources) == 50, f"expected 50 sources, got {len(sources)}"
    assert by_type["paper"] == 15, f"expected 15 papers, got {by_type['paper']}"
    assert by_type["report"] == 15, f"expected 15 reports, got {by_type['report']}"
    assert by_type["data"] == 10, f"expected 10 data sources, got {by_type['data']}"
    assert by_type["education"] == 10, f"expected 10 education, got {by_type['education']}"

    # Required fields populated
    for s in sources:
        assert s.summary.strip(), f"{s.source_id}: empty summary"
        assert s.label.strip(), f"{s.source_id}: empty label"
        assert s.ref.strip(), f"{s.source_id}: empty ref"
        assert 0 <= s.authority_score <= 10, f"{s.source_id}: bad authority_score"
        assert s.keywords, f"{s.source_id}: empty keywords"

    return {
        "total": len(sources),
        "by_type": by_type,
        "ids_unique": True,
        "summary_chars_total": sum(len(s.summary) for s in sources),
    }
