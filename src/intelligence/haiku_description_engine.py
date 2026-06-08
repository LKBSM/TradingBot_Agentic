"""Haiku LLM engine — generates the MarketReading description with 4-lever cost optimization.

Implements Section 2.5 of the V2 architecture doc:
  - Lever 1 : cache by hash(sorted_tags + regime) → re-use prior description
  - Lever 2 : (template fallback handled by the mappers / assembler)
  - Lever 3 : short prompts (< 200 tokens system, < 500 tokens total budget)
  - Lever 4 : on-demand generation only (the assembler triggers this engine
              only when serving a user request, not on every candle close)

Niveau 1.5 strict enforcement:
  - System prompt explicitly forbids recommendation/judgement vocabulary
  - Post-generation forbidden-token filter (cf. market_reading_mappers.
    contains_forbidden_tokens). Contaminated output is NEVER cached and
    NEVER returned — the engine falls back to a tags+regime template.

The Anthropic client is duck-typed: we only call ``client.messages.create(...)``
and read ``response.content[0].text``. This keeps the engine independent of
any specific anthropic-sdk version and makes tests trivial with stubs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Literal, Optional

from src.intelligence.market_reading_mappers import (
    _TREND_FR,
    _VOL_FR,
    _PHASE_FR,
    contains_forbidden_tokens,
)
from src.intelligence.market_reading_schema import (
    DESCRIPTION_MAX_LENGTH,
    MarketReadingRegime,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """Tu es un assistant qui décrit des conditions de marché en français.

RÈGLES STRICTES :
- Tu décris ce qui est observé, JAMAIS ce qu'il faut faire
- Tu n'utilises jamais : conseiller, déconseiller, recommander, éviter, entrer, sortir, acheter, vendre
- Tu n'utilises jamais : risqué, sûr, bon, mauvais, dangereux, opportunité
- Tu te limites à 280 caractères maximum
- Tu écris UNE phrase descriptive en français"""

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 200  # output budget — input + output stays < 500


def _engine_template_fallback(tags: list[str], regime: MarketReadingRegime) -> str:
    """Tags+regime-only fallback used when Haiku produces contaminated output.

    Conservative phrasing — no structure info available at this point.
    """
    trend = _TREND_FR.get(regime.trend, regime.trend)
    vol = _VOL_FR.get(regime.volatility_observed, regime.volatility_observed)
    phase = _PHASE_FR.get(regime.market_phase, regime.market_phase)
    desc = f"Tendance {trend}, volatilité {vol}, phase {phase}."
    if regime.mtf_confluence:
        biases = set(regime.mtf_confluence.values())
        if len(biases) == 1:
            (single,) = biases
            desc += f" MTF alignée {_TREND_FR.get(single, single)}."
        else:
            desc += " MTF mixte."
    return desc[:DESCRIPTION_MAX_LENGTH]


class HaikuDescriptionEngine:
    """Produces a niveau 1.5 strict description via Haiku LLM with caching.

    Sources returned (matching MarketReading schema):
      - "haiku_generated"  : Haiku output passed the forbidden-token filter
      - "template_fallback": fallback path used (no client, or contamination)
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
    ) -> tuple[str, Literal["haiku_generated", "template_fallback"]]:
        """Return ``(description, source)``.

        Strict order:
          1. Cache lookup (sorted tags + regime fingerprint) — cache hit
             never re-calls the LLM.
          2. If no client injected → template fallback (no cache write).
          3. LLM call.
          4. Forbidden-token filter — contaminated output is NEVER cached
             and triggers template fallback.
          5. Clean LLM output is cached and returned.
        """
        hash_key = self._compute_hash(tags, regime)

        cached = self._cache.get(hash_key)
        if cached is not None:
            description, source = cached
            # Defensive double-check: the cache should only hold clean output,
            # but if a forbidden token slipped in historically, drop it and
            # regenerate via fallback.
            if contains_forbidden_tokens(description) is None:
                return description, source  # type: ignore[return-value]
            logger.warning("Cached description contained forbidden tokens; falling back")

        if self._client is None:
            return _engine_template_fallback(tags, regime), "template_fallback"

        try:
            raw = self._call_haiku(tags, regime)
        except Exception as exc:  # network / API error
            logger.warning("Haiku call failed: %s — using template fallback", exc)
            return _engine_template_fallback(tags, regime), "template_fallback"

        # Trim to schema max-length BEFORE filtering (some forbidden tokens
        # might appear only in the truncated portion; we filter what we keep).
        candidate = raw.strip()[:DESCRIPTION_MAX_LENGTH]
        forbidden = contains_forbidden_tokens(candidate)
        if forbidden is not None:
            logger.warning(
                "Haiku output contained forbidden token %r — using template fallback",
                forbidden,
            )
            return _engine_template_fallback(tags, regime), "template_fallback"

        self._cache.put(hash_key, candidate, "haiku_generated")
        return candidate, "haiku_generated"

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _compute_hash(tags: list[str], regime: MarketReadingRegime) -> str:
        """Stable SHA-256 of sorted tags + regime fingerprint.

        Tag order does not matter (sorted). MTF confluence keys are sorted
        so {h1:'bullish', h4:'bearish'} and {h4:'bearish', h1:'bullish'}
        produce the same hash.
        """
        sorted_tags = sorted(tags)
        regime_fingerprint = {
            "trend": regime.trend,
            "volatility_observed": regime.volatility_observed,
            "market_phase": regime.market_phase,
            "mtf_confluence": dict(sorted(regime.mtf_confluence.items())),
        }
        payload = json.dumps(
            {"tags": sorted_tags, "regime": regime_fingerprint},
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _call_haiku(self, tags: list[str], regime: MarketReadingRegime) -> str:
        """Call the Anthropic client and return the raw text content.

        User-prompt body kept small (~50-100 tokens) — the system prompt is
        the only large fixed cost.
        """
        user_prompt = (
            f"Tags du marché : {sorted(tags)}\n"
            f"Régime : trend={regime.trend}, volatilité={regime.volatility_observed}, "
            f"phase={regime.market_phase}, mtf={dict(sorted(regime.mtf_confluence.items()))}\n\n"
            "Décris en une phrase ce qui est observé. Pas de conseil, pas d'évaluation."
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
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
