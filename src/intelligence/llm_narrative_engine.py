"""3-layer Claude API integration for signal narratives.

Layer 1 (Visual/Free): No API call, return signal card only.
Layer 2 (Analyst): Haiku validates signal with Y/N + 1-line reason.
Layer 3 (Strategist): Sonnet generates full institutional thesis.

Prompt caching: 2000-token SMC rulebook cached as system prompt.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_VALIDATOR_MODEL = "claude-haiku-4-5-20250929"
DEFAULT_NARRATOR_MODEL = "claude-sonnet-4-5-20250929"

# Cached system prompt — SMC rulebook for consistent analysis
SMC_SYSTEM_PROMPT = """You are an institutional-grade Gold (XAU/USD) market analyst for Smart Sentinel AI.

## Smart Money Concepts (SMC) Framework
- **BOS (Break of Structure)**: Bullish BOS = higher high breaks previous swing high. Bearish BOS = lower low breaks previous swing low. Confirms directional bias.
- **FVG (Fair Value Gap)**: Imbalance zone (3-candle pattern). Bullish FVG = gap below. Bearish FVG = gap above. Price tends to return to fill these gaps.
- **Order Block (OB)**: Last opposing candle before an impulse move. Bullish OB = last bearish candle before bullish impulse. Institutional entry zone.
- **CHoCH (Change of Character)**: First sign of trend reversal. Bearish CHoCH = first lower low in an uptrend.

## Regime Classification
- strong_uptrend / weak_uptrend: Favor longs, trend-following strategies
- strong_downtrend / weak_downtrend: Favor shorts, trend-following strategies
- ranging: Mean-reversion, fading extremes
- high_volatility: Wider stops, reduced position size
- transition: Caution, wait for confirmation

## Risk Management Rules
- SL = 2×ATR from entry (calm regime) or 3×ATR (volatile regime)
- TP = 4×ATR from entry → minimum 2:1 R:R ratio
- Position size via Kelly Criterion with regime scaling
- Daily loss limit: -2% blocks new entries

## Volatility Regime Context
When volatility forecast data is provided:
- **vol_regime=low**: Tighter ranges expected. SL/TP at standard ATR multiples. Breakouts less likely.
- **vol_regime=normal**: Standard market conditions. Use default risk parameters.
- **vol_regime=high**: Wider stops required (1.5×). Reduce position size. Emphasize risk management. Expect rapid price swings.
- **vol_forecast vs vol_naive**: If forecast ATR significantly exceeds naive ATR, volatility is expanding — mention this. If lower, volatility is compressing.

## Response Format Rules
- Be concise, institutional tone
- Lead with the conclusion (BUY/SELL conviction)
- Support with 2-3 key confluences
- Always mention risk (SL level, R:R, regime context)
- When volatility data is available, integrate it naturally into the analysis (e.g., "Elevated volatility regime warrants wider stops")
- Never give financial advice — present as educational analysis
"""

# Cost estimates (per 1M tokens, as of 2025)
COST_PER_1M = {
    "haiku_input": 0.80,
    "haiku_output": 4.00,
    "haiku_cache_read": 0.08,
    "sonnet_input": 3.00,
    "sonnet_output": 15.00,
    "sonnet_cache_read": 0.30,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class NarrativeTier(str, Enum):
    VISUAL = "VISUAL"          # Free — no API call
    VALIDATOR = "VALIDATOR"    # Analyst — Haiku Y/N
    NARRATOR = "NARRATOR"      # Strategist — Sonnet full thesis


@dataclass
class SignalNarrative:
    """Result of LLM narrative generation."""
    tier: NarrativeTier
    is_valid: bool = True
    validation_reason: str = ""
    full_narrative: str = ""
    key_confluences: str = ""
    risk_warnings: str = ""
    cost_usd: float = 0.0
    model_used: str = ""
    latency_ms: float = 0.0
    cache_hit: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "is_valid": self.is_valid,
            "validation_reason": self.validation_reason,
            "full_narrative": self.full_narrative,
            "key_confluences": self.key_confluences,
            "risk_warnings": self.risk_warnings,
            "cost_usd": round(self.cost_usd, 6),
            "model_used": self.model_used,
            "latency_ms": round(self.latency_ms, 1),
        }


# =============================================================================
# LLM NARRATIVE ENGINE
# =============================================================================

class LLMNarrativeEngine:
    """
    3-layer Claude API integration with prompt caching.

    Usage:
        engine = LLMNarrativeEngine(api_key="sk-ant-...")
        narrative = engine.generate_narrative(signal, tier=NarrativeTier.NARRATOR)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        validator_model: str = DEFAULT_VALIDATOR_MODEL,
        narrator_model: str = DEFAULT_NARRATOR_MODEL,
        enable_caching: bool = True,
    ):
        self._api_key = api_key
        self._validator_model = validator_model
        self._narrator_model = narrator_model
        self._enable_caching = enable_caching
        self._client: Any = None
        self._total_cost: float = 0.0
        self._total_calls: int = 0

        if api_key:
            self._init_client()

    def _init_client(self) -> None:
        """Lazily initialize the Anthropic client."""
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=self._api_key)
            logger.info("Anthropic client initialized (caching=%s)", self._enable_caching)
        except ImportError:
            logger.warning("anthropic package not installed — LLM calls will fail")
            self._client = None

    # ------------------------------------------------------------------ #
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def generate_narrative(
        self,
        signal: Any,
        tier: NarrativeTier = NarrativeTier.VISUAL,
    ) -> SignalNarrative:
        """
        Generate narrative for a ConfluenceSignal at the requested tier.

        Args:
            signal: ConfluenceSignal object.
            tier: NarrativeTier determining depth of analysis.

        Returns:
            SignalNarrative with appropriate content for the tier.
        """
        if tier == NarrativeTier.VISUAL:
            return self._visual_only(signal)

        if self._client is None:
            logger.error("No Anthropic client — returning visual-only fallback")
            return self._visual_only(signal)

        if tier == NarrativeTier.VALIDATOR:
            return self._validate_with_haiku(signal)

        if tier == NarrativeTier.NARRATOR:
            return self._narrate_with_cascade(signal)

        return self._visual_only(signal)

    # ------------------------------------------------------------------ #
    # LAYER 1: VISUAL ONLY
    # ------------------------------------------------------------------ #

    def _visual_only(self, signal: Any) -> SignalNarrative:
        """Free tier — no API call, just structured signal data."""
        return SignalNarrative(
            tier=NarrativeTier.VISUAL,
            is_valid=True,
            validation_reason="Visual-only tier (no LLM validation)",
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------ #
    # LAYER 2: HAIKU VALIDATOR
    # ------------------------------------------------------------------ #

    def _validate_with_haiku(self, signal: Any) -> SignalNarrative:
        """Analyst tier — Haiku validates signal with Y/N + reason."""
        csv_payload = self._signal_to_csv(signal)
        prompt = (
            f"Validate this Gold trading signal. Reply EXACTLY as: VALID|reason or INVALID|reason (one line).\n\n"
            f"Signal:\n{csv_payload}"
        )

        start = time.time()
        try:
            response = self._call_api(self._validator_model, prompt)
            latency = (time.time() - start) * 1000

            text = response["text"].strip()
            cost = response["cost"]
            cache_hit = response.get("cache_hit", False)

            is_valid = text.upper().startswith("VALID")
            reason = text.split("|", 1)[1].strip() if "|" in text else text

            self._total_cost += cost
            self._total_calls += 1

            return SignalNarrative(
                tier=NarrativeTier.VALIDATOR,
                is_valid=is_valid,
                validation_reason=reason,
                cost_usd=cost,
                model_used=self._validator_model,
                latency_ms=latency,
                cache_hit=cache_hit,
            )
        except Exception as e:
            logger.error("Haiku validation failed: %s", e)
            return SignalNarrative(
                tier=NarrativeTier.VALIDATOR,
                is_valid=True,  # Fail-open
                validation_reason=f"Validation error: {e}",
                cost_usd=0.0,
                latency_ms=(time.time() - start) * 1000,
            )

    # ------------------------------------------------------------------ #
    # LAYER 3: SONNET NARRATOR (cascaded from Haiku)
    # ------------------------------------------------------------------ #

    def _narrate_with_cascade(self, signal: Any) -> SignalNarrative:
        """Strategist tier — Haiku validates first, then Sonnet narrates."""
        # Step 1: Haiku gate
        validation = self._validate_with_haiku(signal)
        if not validation.is_valid:
            return SignalNarrative(
                tier=NarrativeTier.NARRATOR,
                is_valid=False,
                validation_reason=validation.validation_reason,
                cost_usd=validation.cost_usd,
                model_used=self._validator_model,
                latency_ms=validation.latency_ms,
            )

        # Step 2: Sonnet narration
        csv_payload = self._signal_to_csv(signal)
        prompt = (
            f"Write a 3-paragraph institutional analysis for this Gold signal.\n"
            f"Paragraph 1: Market Setup (regime, structure, key levels)\n"
            f"Paragraph 2: Key Confluences (SMC patterns confirming the trade)\n"
            f"Paragraph 3: Risk Considerations (SL/TP, R:R, what invalidates the thesis)\n\n"
            f"Signal:\n{csv_payload}"
        )

        start = time.time()
        try:
            response = self._call_api(self._narrator_model, prompt)
            latency = (time.time() - start) * 1000

            text = response["text"].strip()
            cost = response["cost"] + validation.cost_usd
            cache_hit = response.get("cache_hit", False)

            # Split paragraphs
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            narrative = text
            key_confluences = paragraphs[1] if len(paragraphs) > 1 else ""
            risk_warnings = paragraphs[2] if len(paragraphs) > 2 else ""

            self._total_cost += response["cost"]
            self._total_calls += 1

            return SignalNarrative(
                tier=NarrativeTier.NARRATOR,
                is_valid=True,
                validation_reason=validation.validation_reason,
                full_narrative=narrative,
                key_confluences=key_confluences,
                risk_warnings=risk_warnings,
                cost_usd=cost,
                model_used=self._narrator_model,
                latency_ms=latency + validation.latency_ms,
                cache_hit=cache_hit,
            )
        except Exception as e:
            logger.error("Sonnet narration failed: %s", e)
            return SignalNarrative(
                tier=NarrativeTier.NARRATOR,
                is_valid=True,
                validation_reason=validation.validation_reason,
                full_narrative=f"Narration error: {e}",
                cost_usd=validation.cost_usd,
                latency_ms=(time.time() - start) * 1000 + validation.latency_ms,
            )

    # ------------------------------------------------------------------ #
    # API CALL (shared by all layers)
    # ------------------------------------------------------------------ #

    def _call_api(self, model: str, user_prompt: str) -> Dict[str, Any]:
        """Call Claude API with prompt caching."""
        system_messages = [
            {
                "type": "text",
                "text": SMC_SYSTEM_PROMPT,
            }
        ]

        if self._enable_caching:
            system_messages[0]["cache_control"] = {"type": "ephemeral"}

        response = self._client.messages.create(
            model=model,
            max_tokens=1024,
            system=system_messages,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0)

        # Calculate cost
        is_haiku = "haiku" in model
        prefix = "haiku" if is_haiku else "sonnet"
        cost = (
            (input_tokens - cache_read) * COST_PER_1M[f"{prefix}_input"] / 1_000_000
            + cache_read * COST_PER_1M[f"{prefix}_cache_read"] / 1_000_000
            + output_tokens * COST_PER_1M[f"{prefix}_output"] / 1_000_000
        )

        return {
            "text": text,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_hit": cache_read > 0,
        }

    # ------------------------------------------------------------------ #
    # SIGNAL SERIALIZATION (CSV shorthand for token efficiency)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _signal_to_csv(signal: Any) -> str:
        """Serialize ConfluenceSignal as lean CSV (~150 tokens vs ~400 JSON)."""
        components_str = ",".join(
            f"{c.name}={c.weighted_score:.1f}/{c.weight:.0f}"
            for c in getattr(signal, "components", [])
        )
        base = (
            f"sym={getattr(signal, 'symbol', 'XAUUSD')},"
            f"dir={getattr(signal, 'signal_type', 'LONG')},"
            f"score={getattr(signal, 'confluence_score', 0):.1f},"
            f"tier={getattr(signal, 'tier', 'UNKNOWN')},"
            f"entry={getattr(signal, 'entry_price', 0):.2f},"
            f"sl={getattr(signal, 'stop_loss', 0):.2f},"
            f"tp={getattr(signal, 'take_profit', 0):.2f},"
            f"rr={getattr(signal, 'rr_ratio', 0):.2f},"
            f"atr={getattr(signal, 'atr', 0):.2f}\n"
            f"components={components_str}"
        )

        # Append volatility forecast context if available
        vol_regime = getattr(signal, "vol_regime", None)
        vol_forecast_atr = getattr(signal, "vol_forecast_atr", None)
        if vol_regime is not None and vol_forecast_atr is not None:
            naive_atr = getattr(signal, "atr", 0)
            vol_lower = getattr(signal, "vol_confidence_lower", None)
            vol_upper = getattr(signal, "vol_confidence_upper", None)
            vol_line = (
                f"\nvol_regime={vol_regime},"
                f"vol_forecast={vol_forecast_atr:.2f},"
                f"vol_naive={naive_atr:.2f}"
            )
            if vol_lower is not None and vol_upper is not None:
                vol_line += f",vol_ci=[{vol_lower:.2f},{vol_upper:.2f}]"
            base += vol_line

        return base

    # ------------------------------------------------------------------ #
    # STATS
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": round(self._total_cost, 4),
            "total_calls": self._total_calls,
            "avg_cost_per_call": round(self._total_cost / max(self._total_calls, 1), 6),
        }
