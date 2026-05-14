"""Tests for the QUANT-2B.1 score → narrative bucket calibration."""

from __future__ import annotations

import pytest

from src.intelligence.score_calibration import (
    BUCKETS,
    bucket_for,
    contains_forbidden_token,
    narrative_for,
)


# ---------------------------------------------------------------------------
# bucket_for — boundary semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score, expected",
    [
        (0, "no_setup"),
        (29.9, "no_setup"),
        (30, "weak_setup"),
        (49.99, "weak_setup"),
        (50, "moderate_setup"),
        (69.99, "moderate_setup"),
        (70, "strong_setup"),
        (84.99, "strong_setup"),
        (85, "high_confluence_setup"),
        (100, "high_confluence_setup"),
    ],
)
def test_bucket_boundaries(score, expected):
    assert bucket_for(score).name == expected


def test_score_below_zero_is_clipped():
    assert bucket_for(-10).name == "no_setup"


def test_score_above_hundred_is_clipped():
    assert bucket_for(200).name == "high_confluence_setup"


def test_buckets_cover_full_range():
    # Walk score 0..100 in steps of 0.5; every value gets a bucket.
    s = 0.0
    while s <= 100:
        assert bucket_for(s).name is not None
        s += 0.5


def test_buckets_are_non_overlapping():
    # No score should fall in two buckets.
    s = 0.0
    while s <= 100:
        n_hits = sum(1 for b in BUCKETS if b.contains(s))
        assert n_hits == 1, f"score {s} hits {n_hits} buckets"
        s += 0.1


# ---------------------------------------------------------------------------
# narrative_for — FR + EN
# ---------------------------------------------------------------------------


def test_narrative_for_fr_returns_french_phrase():
    n = narrative_for(75, language="fr")
    assert n["bucket"] == "strong_setup"
    assert "Configuration forte" in n["phrase"]
    assert "éducative" in n["guard"]


def test_narrative_for_en_returns_english_phrase():
    n = narrative_for(75, language="en")
    assert "Strong setup" in n["phrase"]
    assert "Educational" in n["guard"]


def test_narrative_for_no_setup_carries_correct_phrase():
    n = narrative_for(10, language="fr")
    assert n["bucket"] == "no_setup"
    assert "Aucune" in n["phrase"]


# ---------------------------------------------------------------------------
# forbidden tokens — UE 2024/2811 guardrail
# ---------------------------------------------------------------------------


def test_forbidden_token_detector_catches_french_buy():
    assert contains_forbidden_token("Achetez maintenant.", language="fr") == "achetez"


def test_forbidden_token_detector_catches_english_buy():
    assert contains_forbidden_token("This is a clear BUY signal.", language="en") == "buy"


def test_forbidden_token_detector_catches_guarantee():
    assert contains_forbidden_token("Performance garanti à 100% sûr", language="fr") is not None


def test_clean_text_returns_none():
    assert (
        contains_forbidden_token(
            "Configuration de haute confluence détectée.", language="fr"
        )
        is None
    )


def test_empty_text_returns_none():
    assert contains_forbidden_token("", language="fr") is None


# ---------------------------------------------------------------------------
# No bucket narrative may itself contain forbidden tokens (self-test)
# ---------------------------------------------------------------------------


def test_every_bucket_phrase_passes_its_own_guardrail():
    for b in BUCKETS:
        assert contains_forbidden_token(b.narrative_phrase_fr, language="fr") is None
        assert contains_forbidden_token(b.narrative_phrase_en, language="en") is None
