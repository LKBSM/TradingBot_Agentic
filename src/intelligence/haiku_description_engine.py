"""Haiku LLM engine — generates the MarketReading "Lecture narrée" from the
engine FACTS, validated against them, with 4-lever cost optimization.

Replaces the legacy weak « synthèse » (tags + regime only, one sentence) with a
present-tense NARRATION anchored to the structured facts of the moteur
(`src.intelligence.narrated_reading`): tendance, alignement multi-TF, zones
OB/FVG actives/testées près du prix, cassures BOS/CHOCH, retest, volatilité.

Niveau 1.5 strict + anchoring enforcement:
  - System prompt forbids recommendation / prediction / causality / score.
  - Post-generation filters: forbidden-token filter (`contains_forbidden_tokens`)
    AND level-anchoring filter (`references_only_known_levels`) — a narration
    that cites a level the moteur never produced is REJECTED, the same way
    coerceViewActions drops a focus on an unknown zone id.
  - On a failed check the engine retries ONCE, then falls back to the
    deterministic `render_template` so the panel is always factual, never empty.

Cost levers (Section 2.5 of the V2 architecture doc):
  - Lever 1 : cache by hash(structural facts) — re-uses a prior narration. The
              raw price is deliberately excluded from the key so the cache is not
              busted every bar; the cache regenerates on a *structural* change
              (zones / breaks / retest / regime), i.e. « changement notable ».
  - Lever 3 : short prompts (system < 250 tokens, facts block ~100-200 tokens).
  - Lever 4 : on-demand generation only (the assembler triggers this engine only
              when serving a user request, not on every candle close).

The Anthropic client is duck-typed: we only call ``client.messages.create(...)``
and read ``response.content[0].text``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal, Optional

from src.intelligence.market_reading_mappers import contains_forbidden_tokens
from src.intelligence.market_reading_schema import (
    MarketReadingRegime,
    MarketReadingStructure,
)
from src.intelligence.narrated_reading import (
    NARRATION_MAX_LENGTH,
    SYSTEM_PROMPT,
    ReadingFacts,
    build_reading_facts,
    build_user_prompt,
    references_only_known_levels,
    render_template,
    truncate_at_sentence,
)

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# Output budget — a 2-4 sentence paragraph fits comfortably; input + output stays
# well under the per-call ceiling.
DEFAULT_MAX_TOKENS = 350


class HaikuDescriptionEngine:
    """Produces a niveau 1.5 strict, engine-anchored narration via Haiku.

    Sources returned (matching MarketReading schema):
      - "haiku_generated"  : Haiku output passed the forbidden-token AND
                             level-anchoring filters.
      - "template_fallback": deterministic template used (no client, API error,
                             or output failed a filter twice).
    """

    def __init__(
        self,
        anthropic_client: Optional[Any],
        cache_store: Any,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = anthropic_client
        self._cache = cache_store
        self._model = model
        self._max_tokens = max_tokens

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def generate(
        self,
        tags: list[str],
        regime: MarketReadingRegime,
        structure: MarketReadingStructure,
        price: float,
        instrument: str,
    ) -> tuple[str, Literal["haiku_generated", "template_fallback"]]:
        """Return ``(narration, source)``.

        Strict order:
          1. Build the engine FACTS (read-only projection of structure+regime).
          2. Cache lookup (structural fingerprint) — a hit never re-calls the LLM
             and is re-validated defensively before reuse.
          3. No client → deterministic template (no cache write).
          4. LLM call → validate (forbidden tokens AND known levels). On failure,
             ONE retry. Still failing → deterministic template.
          5. Clean LLM output is cached and returned.
        """
        facts = build_reading_facts(structure, regime, price, instrument)
        hash_key = self._compute_hash(tags, facts)

        cached = self._cache.get(hash_key)
        if cached is not None:
            description, source = cached
            # Defensive double-check: the cache should only hold clean output. If
            # a contaminated/unanchored string slipped in historically, drop it.
            if self._is_clean(description, facts):
                return description, source  # type: ignore[return-value]
            logger.warning("Cached narration failed validation; regenerating")

        if self._client is None:
            return render_template(facts), "template_fallback"

        candidate = self._attempt(facts)
        if candidate is None:
            # Retry once — transient model wobble or a single bad number.
            logger.info("Narration failed validation; retrying once")
            candidate = self._attempt(facts)

        if candidate is None:
            return render_template(facts), "template_fallback"

        self._cache.put(hash_key, candidate, "haiku_generated")
        return candidate, "haiku_generated"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _attempt(self, facts: ReadingFacts) -> Optional[str]:
        """One LLM call + validation. Returns the clean narration or None."""
        try:
            raw = self._call_haiku(facts)
        except Exception as exc:  # network / API error
            logger.warning("Haiku call failed: %s", exc)
            return None
        # Display-only field: clamp on a SENTENCE boundary, never mid-word. The
        # budget (NARRATION_MAX_LENGTH) is sized to hold a complete 2-4 sentence
        # paragraph, so a compliant narration is returned whole; only a run-on
        # that ignored the length instruction gets trimmed — and then at a
        # sentence end, not through a word/number.
        candidate = truncate_at_sentence(raw, NARRATION_MAX_LENGTH)
        if not self._is_clean(candidate, facts):
            return None
        return candidate

    @staticmethod
    def _is_clean(text: str, facts: ReadingFacts) -> bool:
        """Both niveau-1.5 gates: no forbidden vocab AND only engine-known levels."""
        forbidden = contains_forbidden_tokens(text)
        if forbidden is not None:
            logger.warning("Narration contained forbidden token %r", forbidden)
            return False
        if not references_only_known_levels(text, facts):
            logger.warning("Narration referenced a level absent from the facts")
            return False
        return True

    @staticmethod
    def _compute_hash(tags: list[str], facts: ReadingFacts) -> str:
        """Stable SHA-256 of the STRUCTURAL facts (price excluded on purpose).

        Tag order does not matter (sorted). The fingerprint captures everything
        that should bust the cache — regime, MTF relation+biases, each zone
        (kind/dir/status/tested/bounds), each break (kind/dir/level/confirmed),
        and the retest — but NOT the raw price, so a quiet tick does not force a
        regeneration (Lever 1 / « regénère sur changement notable »).
        """
        fingerprint = {
            "tags": sorted(tags),
            "trend": facts.trend,
            "volatility": facts.volatility,
            "phase": facts.phase,
            "mtf_relation": facts.mtf_relation,
            "mtf": dict(sorted(facts.mtf_biases.items())),
            "zones": sorted(
                f"{z.kind}|{z.direction}|{z.status}|{z.tested}|{z.low}|{z.high}|{z.position}"
                for z in facts.zones
            ),
            "breaks": [
                f"{b.kind}|{b.direction}|{b.level}|{b.confirmed}" for b in facts.breaks
            ],
            "retest": f"{facts.retest_type}|{facts.retest_level}",
            "contrary": facts.contrary or "",
        }
        payload = json.dumps(fingerprint, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _call_haiku(self, facts: ReadingFacts) -> str:
        """Call the Anthropic client and return the raw text content."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_prompt(facts)}],
        )
        # Anthropic SDK shape: response.content is a list of content blocks
        # whose .text attribute holds the generated string.
        content = getattr(response, "content", None)
        if not content:
            raise RuntimeError("Anthropic response has no content")
        text = getattr(content[0], "text", None)
        if text is None:
            raise RuntimeError("Anthropic response content[0] missing .text")
        return str(text)


__all__ = [
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_MODEL",
    "HaikuDescriptionEngine",
    "SYSTEM_PROMPT",
]
