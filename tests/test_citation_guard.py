"""Tests for the LLM-2B.6 citation enforcement guard."""

from __future__ import annotations

import pytest

from src.intelligence.rag.citation_guard import (
    CitationGuardResult,
    enforce_citations,
)


# ---------------------------------------------------------------------------
# Happy path — every factual sentence cites a retrieved chunk
# ---------------------------------------------------------------------------


def test_clean_answer_passes():
    answer = (
        "The HAR-RV decomposes volatility into three horizons. "
        "[source:paper_har] Empirical evidence supports this. [source:paper_har]"
    )
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert res.ok
    assert not res.violations


def test_non_factual_sentence_does_not_need_citation():
    answer = "Setup haussier détecté. The weekly close is bullish. [source:paper_har]"
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert res.ok


# ---------------------------------------------------------------------------
# Violations
# ---------------------------------------------------------------------------


def test_factual_sentence_without_citation_flagged():
    answer = "DGS10 hit 4.50 today and bounced. The trend is intact."
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert not res.ok
    assert len(res.violations) >= 1
    # The DGS10 sentence is the offender; the trend sentence is non-factual.
    assert any("DGS10" in v.sentence for v in res.violations)


def test_citation_to_unretrieved_chunk_flagged():
    answer = "DGS10 hit 4.50 today. [source:fake_chunk_xyz]"
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert not res.ok
    v = res.violations[0]
    assert "fake_chunk_xyz" in v.cited_unknown_ids
    assert "not in retrieved" in v.reason.lower()


def test_mixed_known_and_unknown_citations_flagged():
    answer = "Real yields fell to 2.10. [source:paper_har] [source:fake]"
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert not res.ok
    assert "fake" in res.violations[0].cited_unknown_ids


# ---------------------------------------------------------------------------
# Strip policy
# ---------------------------------------------------------------------------


def test_strip_policy_removes_offending_sentences():
    answer = (
        "DGS10 hit 4.50 today. [source:paper_har] "
        "VIX spiked to 25.7 yesterday. "
        "Setup neutre."
    )
    res = enforce_citations(
        answer,
        retrieved_chunk_ids=["paper_har"],
        policy="strip",
    )
    assert "VIX" not in res.answer
    assert "DGS10" in res.answer
    assert "Setup neutre" in res.answer
    assert len(res.violations) == 1


def test_flag_policy_preserves_answer_byte_for_byte():
    answer = "DGS10 hit 4.50 today. The trend is up."
    res = enforce_citations(answer, retrieved_chunk_ids=[], policy="flag")
    assert res.answer == answer


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_answer_is_ok():
    res = enforce_citations("", retrieved_chunk_ids=["paper_har"])
    assert res.ok
    assert res.answer == ""


def test_disclaimer_line_doesnt_need_citation():
    answer = "Educational algorithmic analysis. Not investment advice."
    # Both sentences contain only stopword-class proper nouns.
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert res.ok


def test_french_narrative_handled():
    answer = (
        "Synthèse haussière sur XAU. "
        "Le rendement DGS10 a chuté à 4.10 hier. [source:fred_dgs10] "
        "Analyse algorithmique éducative."
    )
    res = enforce_citations(answer, retrieved_chunk_ids=["fred_dgs10"])
    assert res.ok


def test_multiple_violations_all_collected():
    answer = (
        "DGS10 fell to 4.10. "
        "VIX spiked to 28. "
        "EUR/USD broke 1.10."
    )
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert len(res.violations) == 3


def test_result_is_truthy_only_when_ok():
    ok_result = CitationGuardResult(answer="x")
    bad_result = CitationGuardResult(
        answer="x",
        violations=[
            __import__(
                "src.intelligence.rag.citation_guard", fromlist=["CitationViolation"]
            ).CitationViolation(sentence="y", reason="z")
        ],
    )
    assert bool(ok_result)
    assert not bool(bad_result)


def test_n_factual_sentences_counter():
    answer = (
        "DGS10 fell to 4.10. [source:paper_har] "
        "Setup neutre. "
        "VIX spiked to 28. [source:paper_har]"
    )
    res = enforce_citations(answer, retrieved_chunk_ids=["paper_har"])
    assert res.n_factual_sentences == 2
    assert res.n_sentences_considered == 3
