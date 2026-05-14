"""Score → narrative-bucket calibration — Sprint QUANT-2B.1.

Maps a ConfluenceDetector ``score_0_100`` to a *narrative* bucket
label like "weak setup" / "moderate setup" / "strong setup" —
deliberately NOT to a probability of profit or a buy/sell call.

The A1 verdict (DSR=0, PBO=0.5) proved the score has no predictive
edge. We don't pretend otherwise. The bucket is purely *educational*:
"this setup has several factors aligned but you should still do your
own analysis." That phrasing is required by UE 2024/2811 — calling
score>80 "high probability" or "buy signal" would be the kind of
unauthorized recommendation the regulation targets.

Bucket boundaries are anchored on observed empirical percentiles
(eval_06_semantic_cache + audit_backtest baseline), not on a
prediction quality threshold:

  score < 30   → "no_setup"
  30 ≤ s < 50  → "weak_setup"
  50 ≤ s < 70  → "moderate_setup"
  70 ≤ s < 85  → "strong_setup"
  85 ≤ s ≤ 100 → "high_confluence_setup"

Each bucket carries a *narrative phrase* the LLM can drop directly
into a paragraph + a *guard phrase* it MUST include (no actionable
recommendation, no probability claim, no edge claim).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class BucketLabel:
    name: str
    low: float
    high: float
    narrative_phrase_fr: str
    narrative_phrase_en: str
    guard_phrase_fr: str
    guard_phrase_en: str

    def contains(self, score: float) -> bool:
        if self.high == 100:
            return self.low <= score <= self.high
        return self.low <= score < self.high


_GUARD_FR = (
    "Analyse algorithmique éducative — pas de recommandation d'achat ou de vente."
)
_GUARD_EN = (
    "Educational algorithmic analysis — not a buy or sell recommendation."
)


BUCKETS: list[BucketLabel] = [
    BucketLabel(
        name="no_setup",
        low=0.0, high=30.0,
        narrative_phrase_fr=(
            "Aucune configuration claire détectée — facteurs de confluence absents ou contradictoires."
        ),
        narrative_phrase_en=(
            "No clear setup detected — confluence factors absent or contradictory."
        ),
        guard_phrase_fr=_GUARD_FR,
        guard_phrase_en=_GUARD_EN,
    ),
    BucketLabel(
        name="weak_setup",
        low=30.0, high=50.0,
        narrative_phrase_fr=(
            "Configuration faible — quelques facteurs alignés mais signal global ambigu."
        ),
        narrative_phrase_en=(
            "Weak setup — a few factors align but overall signal is ambiguous."
        ),
        guard_phrase_fr=_GUARD_FR,
        guard_phrase_en=_GUARD_EN,
    ),
    BucketLabel(
        name="moderate_setup",
        low=50.0, high=70.0,
        narrative_phrase_fr=(
            "Configuration modérée — plusieurs facteurs de confluence alignés."
        ),
        narrative_phrase_en=(
            "Moderate setup — several confluence factors aligned."
        ),
        guard_phrase_fr=_GUARD_FR,
        guard_phrase_en=_GUARD_EN,
    ),
    BucketLabel(
        name="strong_setup",
        low=70.0, high=85.0,
        narrative_phrase_fr=(
            "Configuration forte — convergence marquée de plusieurs signaux techniques et de contexte."
        ),
        narrative_phrase_en=(
            "Strong setup — marked convergence of multiple technical and contextual signals."
        ),
        guard_phrase_fr=_GUARD_FR,
        guard_phrase_en=_GUARD_EN,
    ),
    BucketLabel(
        name="high_confluence_setup",
        low=85.0, high=100.0,
        narrative_phrase_fr=(
            "Configuration de haute confluence — alignement marquant de signaux techniques, contextuels et macro."
        ),
        narrative_phrase_en=(
            "High-confluence setup — marked alignment of technical, contextual and macro signals."
        ),
        guard_phrase_fr=_GUARD_FR,
        guard_phrase_en=_GUARD_EN,
    ),
]


# These tokens must NEVER appear in a generated narrative regardless
# of bucket — they would convert an educational analysis into an
# unauthorised personalised recommendation under UE 2024/2811.
FORBIDDEN_TOKENS_FR = (
    "achetez", "vendez", "achète", "vends", "garanti", "100% sûr",
    "edge prouvé", "signal d'achat", "signal de vente",
)
FORBIDDEN_TOKENS_EN = (
    "buy", "sell", "guaranteed", "100% sure", "proven edge",
    "buy signal", "sell signal",
)


def bucket_for(score: float) -> BucketLabel:
    """Returns the bucket containing ``score``. Score is clipped to [0, 100]."""
    if score < 0:
        score = 0.0
    if score > 100:
        score = 100.0
    for b in BUCKETS:
        if b.contains(score):
            return b
    # Unreachable given the buckets above, but defensive.
    return BUCKETS[-1]


def narrative_for(
    score: float, *, language: str = "fr"
) -> dict:
    """Return ``{bucket, phrase, guard}`` for a given score + language."""
    b = bucket_for(score)
    if language.lower().startswith("fr"):
        phrase, guard = b.narrative_phrase_fr, b.guard_phrase_fr
    else:
        phrase, guard = b.narrative_phrase_en, b.guard_phrase_en
    return {
        "score": round(score, 2),
        "bucket": b.name,
        "phrase": phrase,
        "guard": guard,
    }


def contains_forbidden_token(text: str, *, language: str = "fr") -> Optional[str]:
    """Return the first forbidden token found in ``text`` or None.

    Use case: post-generation check that an LLM didn't slip a
    "achetez" past us.
    """
    if not text:
        return None
    lowered = text.lower()
    tokens = FORBIDDEN_TOKENS_FR if language.lower().startswith("fr") else FORBIDDEN_TOKENS_EN
    for tok in tokens:
        if tok in lowered:
            return tok
    return None


__all__ = [
    "BUCKETS",
    "BucketLabel",
    "FORBIDDEN_TOKENS_EN",
    "FORBIDDEN_TOKENS_FR",
    "bucket_for",
    "contains_forbidden_token",
    "narrative_for",
]
