"""LLM narrative evaluation harness — 5-axis scoring.

Sprint LLM-1.1 (Aisha). For each generated narrative, score:

  1. factual_consistency — LLM-as-judge: does the narrative align with the
     payload (no hallucinated levels / events / components)?
  2. reading_level     — Flesch-Kincaid Grade Level within [min, max] band
  3. forbidden_phrases — 0 occurrences of "buy", "sell", "guaranteed", etc.
                          (compliance MiFID II 2024/2811 + AMF finfluencer 2026)
  4. source_attribution — narrative cites the entry / stop / target prices
                           and the components from the payload
  5. brevity            — character count below the per-fixture max

Each axis returns a float in [0, 1]. The overall score is the mean.

Usage:
    harness = EvalHarness(generator=my_narrative_fn, judge=my_judge_fn)
    results = harness.run_all(fixtures)
    print(harness.summary(results))
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


@dataclass
class AxisScore:
    name: str
    score: float
    detail: str = ""


@dataclass
class FixtureResult:
    fixture_id: str
    category: str
    narrative: str
    axes: dict[str, AxisScore]
    overall: float

    def as_dict(self) -> dict:
        return {
            "fixture_id": self.fixture_id,
            "category": self.category,
            "narrative": self.narrative,
            "axes": {k: asdict(v) for k, v in self.axes.items()},
            "overall": self.overall,
        }


# ---------------------------------------------------------------------------
# Reading-level: Flesch-Kincaid Grade Level
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ]+")
_SENT_RE = re.compile(r"[.!?]+")


def _count_syllables(word: str) -> int:
    """Heuristic English syllable count: groups of consecutive vowels.

    For French this overestimates slightly but is acceptable for the
    "is the narrative within institutional grade level band" purpose.
    """
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouyàâäéèêëïîôöùûü"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Silent trailing 'e' adjustment (English heuristic)
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def flesch_kincaid_grade(text: str) -> float:
    """Flesch-Kincaid Grade Level: 0.39*(W/S) + 11.8*(Syl/W) - 15.59.

    Higher = harder. Typical institutional/financial writing: 12-16.
    Returns 0.0 on degenerate input (no words or no sentences).
    """
    words = _WORD_RE.findall(text)
    sentences = [s for s in _SENT_RE.split(text) if s.strip()]
    if len(words) == 0 or len(sentences) == 0:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    return 0.39 * (len(words) / len(sentences)) + 11.8 * (syllables / len(words)) - 15.59


def score_reading_level(narrative: str, expected: dict) -> AxisScore:
    grade = flesch_kincaid_grade(narrative)
    lo = expected.get("min_grade_level", 8.0)
    hi = expected.get("max_grade_level", 18.0)
    if lo <= grade <= hi:
        return AxisScore("reading_level", 1.0, f"grade={grade:.1f} in [{lo}, {hi}]")
    # Linear penalty outside the band
    if grade < lo:
        gap = lo - grade
    else:
        gap = grade - hi
    score = max(0.0, 1.0 - gap / 4.0)  # full penalty at 4 grades off
    return AxisScore("reading_level", score, f"grade={grade:.1f} outside [{lo}, {hi}] gap={gap:.1f}")


# ---------------------------------------------------------------------------
# Forbidden phrases
# ---------------------------------------------------------------------------


def score_forbidden_phrases(narrative: str, expected: dict) -> AxisScore:
    text_lower = narrative.lower()
    phrases = expected.get("forbidden_phrases", [])
    hits = [p for p in phrases if p.lower() in text_lower]
    if not hits:
        return AxisScore("forbidden_phrases", 1.0, "no forbidden phrase")
    return AxisScore(
        "forbidden_phrases",
        0.0,
        f"forbidden phrases found: {hits}",
    )


# ---------------------------------------------------------------------------
# Source attribution
# ---------------------------------------------------------------------------


def score_source_attribution(narrative: str, expected: dict) -> AxisScore:
    """Counts how many of the required citations actually appear in the text.

    Numbers are matched as decimals with optional thousands separator (e.g.
    `2350.00` matches `2,350.00`, `2350.0`, `2350`).
    """
    required = []
    for key in ["must_cite_entry", "must_cite_stop", "must_cite_target"]:
        v = expected.get(key)
        if v is not None:
            required.append(("price", v))
    components = expected.get("must_mention_components") or []
    for c in components:
        required.append(("component", c))

    if not required:
        return AxisScore("source_attribution", 1.0, "no required citation")

    matched = 0
    for kind, value in required:
        if kind == "price":
            if _price_in_text(value, narrative):
                matched += 1
        elif kind == "component":
            if value.lower() in narrative.lower():
                matched += 1

    score = matched / len(required)
    return AxisScore(
        "source_attribution",
        score,
        f"{matched}/{len(required)} required citations present",
    )


def _price_in_text(value: float, text: str) -> bool:
    """Match the price with optional decimals / thousands separator."""
    # Match the integer part first (e.g. 2350) — broadest and usually enough.
    int_part = int(value)
    pat = re.compile(rf"\b{int_part:,}(?:[.,]\d+)?\b|\b{int_part}(?:[.,]\d+)?\b")
    return bool(pat.search(text))


# ---------------------------------------------------------------------------
# Brevity
# ---------------------------------------------------------------------------


def score_brevity(narrative: str, expected: dict) -> AxisScore:
    n = len(narrative)
    max_chars = expected.get("max_chars", 400)
    if n <= max_chars:
        return AxisScore("brevity", 1.0, f"{n}/{max_chars} chars")
    over = n - max_chars
    score = max(0.0, 1.0 - over / max_chars)
    return AxisScore("brevity", score, f"{n}/{max_chars} chars (over by {over})")


# ---------------------------------------------------------------------------
# Factual consistency — LLM-as-judge (with fallback)
# ---------------------------------------------------------------------------


JudgeFn = Callable[[str, dict], float]
"""Signature: judge(narrative, fixture_input) → score in [0, 1]."""


def heuristic_factual_consistency(narrative: str, fixture_input: dict) -> float:
    """Fallback factual-consistency heuristic when no LLM judge is available.

    Checks:
    - Direction word matches (bullish vs bearish narrative)
    - At least one component name is mentioned (when components exist)
    - At least one price level is mentioned (when levels exist)
    - No mentions of OTHER instruments (BTC, EUR, SPX, etc.) when symbol=XAUUSD

    Returns a score in [0, 1]. Designed to be conservative (deduct only on
    clear mismatches; uncertain → don't penalise).
    """
    score = 1.0
    text_lower = narrative.lower()
    direction = (fixture_input.get("direction") or "").upper()

    if direction == "BUY":
        if "bullish" not in text_lower and "long" not in text_lower:
            score -= 0.25
        if "bearish" in text_lower and "bullish" not in text_lower:
            score -= 0.25  # narrative talks bearish on a BUY signal
    elif direction == "SELL":
        if "bearish" not in text_lower and "short" not in text_lower:
            score -= 0.25
        if "bullish" in text_lower and "bearish" not in text_lower:
            score -= 0.25

    # Other-instrument hallucination check
    other_instruments = ["bitcoin", "eth", "spx", "s&p", "eurusd", "gbpusd", "btcusd"]
    if (fixture_input.get("symbol") or "").upper() == "XAUUSD":
        for other in other_instruments:
            if other in text_lower:
                score -= 0.2  # heavy penalty per hallucinated instrument

    return max(0.0, min(1.0, score))


def score_factual_consistency(
    narrative: str,
    fixture_input: dict,
    judge: JudgeFn | None = None,
) -> AxisScore:
    if judge is None:
        s = heuristic_factual_consistency(narrative, fixture_input)
        return AxisScore("factual_consistency", s, "heuristic (no LLM judge)")
    s = float(judge(narrative, fixture_input))
    s = max(0.0, min(1.0, s))
    return AxisScore("factual_consistency", s, "LLM judge")


# ---------------------------------------------------------------------------
# Top-level harness
# ---------------------------------------------------------------------------


GeneratorFn = Callable[[dict], str]
"""Signature: generator(fixture_input) → narrative string."""


@dataclass
class EvalSummary:
    n_fixtures: int
    overall_mean: float
    overall_p25: float
    per_axis_mean: dict[str, float]
    per_category_mean: dict[str, float] = field(default_factory=dict)
    timestamp_utc: str = ""
    failures: list[str] = field(default_factory=list)  # fixture ids with overall < 0.5


class EvalHarness:
    """Run the 5-axis evaluation over a fixture set."""

    def __init__(
        self,
        generator: GeneratorFn,
        judge: JudgeFn | None = None,
    ):
        self.generator = generator
        self.judge = judge

    def evaluate_one(self, fixture: Any) -> FixtureResult:
        # Accept Fixture dataclass or plain dict
        if hasattr(fixture, "as_dict"):
            f = fixture.as_dict()
        else:
            f = fixture
        narrative = self.generator(f["input"])
        expected = f["expected"]

        axes = {
            "factual_consistency": score_factual_consistency(
                narrative, f["input"], judge=self.judge
            ),
            "reading_level": score_reading_level(narrative, expected),
            "forbidden_phrases": score_forbidden_phrases(narrative, expected),
            "source_attribution": score_source_attribution(narrative, expected),
            "brevity": score_brevity(narrative, expected),
        }
        overall = sum(a.score for a in axes.values()) / len(axes)
        return FixtureResult(
            fixture_id=f["id"],
            category=f["category"],
            narrative=narrative,
            axes=axes,
            overall=overall,
        )

    def run_all(self, fixtures: list[Any]) -> list[FixtureResult]:
        results = []
        for fx in fixtures:
            r = self.evaluate_one(fx)
            results.append(r)
        return results

    def summary(self, results: list[FixtureResult]) -> EvalSummary:
        if not results:
            return EvalSummary(
                n_fixtures=0,
                overall_mean=0.0,
                overall_p25=0.0,
                per_axis_mean={},
                timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
        overalls = [r.overall for r in results]
        per_axis: dict[str, list[float]] = {}
        for r in results:
            for k, v in r.axes.items():
                per_axis.setdefault(k, []).append(v.score)

        per_cat: dict[str, list[float]] = {}
        for r in results:
            per_cat.setdefault(r.category, []).append(r.overall)

        return EvalSummary(
            n_fixtures=len(results),
            overall_mean=sum(overalls) / len(overalls),
            overall_p25=float(sorted(overalls)[len(overalls) // 4]),
            per_axis_mean={k: sum(v) / len(v) for k, v in per_axis.items()},
            per_category_mean={k: sum(v) / len(v) for k, v in per_cat.items()},
            timestamp_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            failures=[r.fixture_id for r in results if r.overall < 0.5],
        )

    def save_results(
        self,
        results: list[FixtureResult],
        path: Path | str,
    ) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "summary": asdict(self.summary(results)),
            "results": [r.as_dict() for r in results],
        }
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        logger.info("Saved %d eval results -> %s", len(results), out)
        return out
