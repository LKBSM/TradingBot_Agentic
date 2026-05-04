"""Citation enforcement guard — Sprint LLM-2B.6.

The system prompts in ``prompts.py`` instruct the LLM to cite every
factual claim with ``[source:chunk_id]``. The guard verifies that
contract on the answer side: any sentence that asserts a fact (numbers,
proper nouns, named entities) MUST carry at least one citation pointing
to a chunk that was actually retrieved.

This is the last line of defence against hallucinations: if the LLM
ignores its system prompt or paraphrases something not in the context,
the guard flags or strips the offending sentence before the answer
reaches the client.

Design choices
--------------
- Sentence segmentation is regex-based (".?!" + newline). Good enough
  for the institutional-tone narratives we generate; no NLTK dep.
- A "factual" sentence is one that contains at least one of:
  * a digit (price, percentage, year)
  * a proper-noun-like token (capitalised + ≥3 chars, not a stop word)
- Citations are matched to the retrieved chunks by ``chunk_id`` (must
  appear in ``[source:chunk_id]`` form). Citations to chunks that were
  NOT retrieved are themselves a violation.
- The guard supports two policies:
  * ``"flag"``  — return the original answer + a list of violations.
  * ``"strip"`` — return the answer with offending sentences removed
    + the violations list. Useful when the broker contract demands
    no unsourced claims at all.

Output
------
``CitationGuardResult.answer`` is the cleaned (or original) answer;
``violations`` is a list of ``CitationViolation`` records with sentence
text, position, and reason. ``ok`` is True iff there are zero
violations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# A factual sentence has a number or a non-stopword proper noun.
_NUM_RE = re.compile(r"\b\d[\d,.\-]*\b")
_PROPER_RE = re.compile(r"\b[A-Z][A-Za-z0-9]{2,}\b")
_CITATION_RE = re.compile(r"\[source:([A-Za-z0-9_\-]+)\]")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ŸÉÈÊÎÔÛÇ\d])")

# Generic capitalised words to ignore when deciding if a sentence is
# "factual" enough to demand a citation. (Sentence starts always
# capitalise — they're not proper nouns by themselves.)
_PROPER_NOUN_STOPWORDS = frozenset(
    {
        "The", "This", "That", "These", "Those", "Then", "Not",
        "When", "What", "Why", "Where", "Which", "While",
        "Smart", "Sentinel", "AI", "EU", "US",
        "Note", "Setup", "Source", "Section", "Educational",
        "Algorithmic", "Analysis", "Disclaimer",
        "Bullish", "Bearish", "Neutral",
        # Asset tickers — naming an instrument is not a citable fact
        "XAU", "XAUUSD", "EUR", "EURUSD", "USD", "GBP", "JPY",
        "BTC", "BTCUSD", "USDJPY", "GBPUSD", "USOIL", "WTI",
        # FR
        "Cette", "Cet", "Cela", "Quand", "Pourquoi", "Comment",
        "Analyse", "Algorithmique", "Synthèse", "Section",
    }
)

Policy = Literal["flag", "strip"]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


@dataclass
class CitationViolation:
    sentence: str
    reason: str
    cited_unknown_ids: list[str] = field(default_factory=list)


@dataclass
class CitationGuardResult:
    answer: str
    violations: list[CitationViolation] = field(default_factory=list)
    n_sentences_considered: int = 0
    n_factual_sentences: int = 0

    @property
    def ok(self) -> bool:
        return not self.violations

    def __bool__(self) -> bool:
        return self.ok


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_factual(sentence: str) -> bool:
    """Heuristic: the sentence asserts a fact if it carries a digit OR a
    non-stopword proper noun (capitalised, ≥3 chars)."""
    if _NUM_RE.search(sentence):
        return True
    for token in _PROPER_RE.findall(sentence):
        if token not in _PROPER_NOUN_STOPWORDS:
            return True
    return False


_LEADING_CITATIONS_RE = re.compile(r"^((?:\[source:[^\]]+\]\s*)+)(.*)$")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    # Pre-split on hard line breaks too — narratives sometimes use
    # bullet-style lists where each line is conceptually a sentence.
    raw: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("- "):
            raw.append(line)
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        raw.extend(p.strip() for p in parts if p.strip())

    # Reattach orphan ``[source:...]`` citations to the previous sentence.
    # Convention: a citation appearing right after a sentence's period
    # belongs to that sentence, not the next one.
    out: list[str] = []
    for s in raw:
        m = _LEADING_CITATIONS_RE.match(s)
        if out and m and m.group(1).strip():
            citations, rest = m.group(1).strip(), m.group(2).strip()
            out[-1] = out[-1] + " " + citations
            if rest:
                out.append(rest)
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def enforce_citations(
    answer: str,
    retrieved_chunk_ids: Iterable[str],
    *,
    policy: Policy = "flag",
) -> CitationGuardResult:
    """Apply the citation contract to ``answer``.

    Parameters
    ----------
    answer : str
        The LLM-produced response (or stub).
    retrieved_chunk_ids : Iterable[str]
        The chunk IDs that were actually surfaced by the retriever for
        this query. Citations to anything else are violations.
    policy : "flag" or "strip"
        ``"flag"`` returns the answer untouched + a list of violations.
        ``"strip"`` removes offending sentences from the returned answer.

    Returns
    -------
    CitationGuardResult
    """
    valid_ids = set(retrieved_chunk_ids)
    sentences = _split_sentences(answer)
    violations: list[CitationViolation] = []
    factual_count = 0
    kept_sentences: list[str] = []

    for sentence in sentences:
        if not _is_factual(sentence):
            kept_sentences.append(sentence)
            continue
        factual_count += 1

        cited = _CITATION_RE.findall(sentence)
        if not cited:
            violations.append(
                CitationViolation(
                    sentence=sentence,
                    reason="factual claim with no [source:] citation",
                )
            )
            if policy == "flag":
                kept_sentences.append(sentence)
            continue

        unknown = [c for c in cited if c not in valid_ids]
        if unknown:
            violations.append(
                CitationViolation(
                    sentence=sentence,
                    reason="cites chunk(s) not in retrieved context",
                    cited_unknown_ids=unknown,
                )
            )
            if policy == "flag":
                kept_sentences.append(sentence)
            continue

        kept_sentences.append(sentence)

    if policy == "strip":
        cleaned = " ".join(kept_sentences).strip()
    else:
        cleaned = answer

    return CitationGuardResult(
        answer=cleaned,
        violations=violations,
        n_sentences_considered=len(sentences),
        n_factual_sentences=factual_count,
    )


__all__ = [
    "CitationGuardResult",
    "CitationViolation",
    "enforce_citations",
]
