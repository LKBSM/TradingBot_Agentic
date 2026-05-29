"""4-tier Claude API integration for signal narratives.

Layer 1 (Visual / FREE): No API call, return signal card only.
Layer 2 (Validator / ANALYST): Haiku validates signal with VALID|INVALID + 1-line reason.
Layer 3 (Narrator / STRATEGIST): Sonnet generates full institutional thesis (single call).
Layer 4 (Institutional / INSTITUTIONAL): Opus generates deep multi-section thesis.

Prompt caching: ≥1200-token SMC rulebook cached as system prompt (above Sonnet/Opus
1024-token threshold). Targets ≥2048 tokens to also engage the Haiku cache window.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_VALIDATOR_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_NARRATOR_MODEL = "claude-sonnet-4-6"
DEFAULT_INSTITUTIONAL_MODEL = "claude-opus-4-7"

DEFAULT_API_TIMEOUT_S = 30.0

# Cached system prompt — extended SMC rulebook with examples + anti-hallucination
# rules. Length ≥ 1200 tokens (Sonnet/Opus cache threshold) and targets ≥ 2048
# tokens so the Haiku cache also activates on a real call.
SMC_SYSTEM_PROMPT = """You are an institutional-grade Gold (XAU/USD) market analyst for Smart Sentinel AI. Your job is to validate or narrate trade signals produced by a deterministic confluence engine. You never invent data; you reason strictly from the structured payload provided in each user message.

## Smart Money Concepts (SMC) Framework
- **BOS (Break of Structure)**: Bullish BOS = current high breaks the most recent confirmed swing high. Bearish BOS = current low breaks the most recent confirmed swing low. A BOS confirms continuation in the direction of the break.
- **CHoCH (Change of Character)**: First counter-trend break of structure. Bearish CHoCH in an uptrend (first lower low) is the earliest reversal warning; treat as caution, not confirmation.
- **FVG (Fair Value Gap)**: Three-candle imbalance pattern. Bullish FVG = body of candle 1 high < body of candle 3 low. Bearish FVG = mirror. Price has a high statistical tendency to revisit unfilled FVGs.
- **Order Block (OB)**: Last opposing candle before an impulsive move that creates a BOS. Bullish OB = last bearish candle before a bullish impulse; institutional re-entry zone.
- **Liquidity Sweep**: Wick that pierces a prior swing high/low and closes back inside. Indicates stop-hunt / liquidity grab — often precedes reversal.

## Regime Classification
- `strong_uptrend` / `weak_uptrend`: Favor longs. Discount counter-trend shorts.
- `strong_downtrend` / `weak_downtrend`: Favor shorts. Discount counter-trend longs.
- `ranging`: Mean-reversion only; fade extremes; reject trend-following entries.
- `high_volatility`: Wider stops (×1.5), reduced size, demand stronger confluence.
- `transition`: Bias is unclear — require explicit confirmation (BOS + retest).

## Volatility Regime Context
When the payload includes `vol_regime` and `vol_forecast`:
- `vol_regime=low`: Standard ATR multiples for SL/TP. Breakouts statistically less likely; favor mean-reversion.
- `vol_regime=normal`: Default risk parameters apply.
- `vol_regime=high`: Widen SL by ~1.5×, reduce position size, emphasize risk language. Expect rapid swings.
- If `vol_forecast` significantly exceeds `vol_naive` (>20%), state that volatility is expanding. If lower, state that it is compressing. Never invent a percentage that is not in the payload.

## Risk Management Rules
- SL = 2×ATR (calm regime) or 3×ATR (volatile regime), measured from entry.
- TP = 4×ATR from entry → minimum 2:1 R:R ratio.
- Position size = Kelly Criterion, scaled by regime.
- Daily loss limit −2% blocks new entries.
- News blackout: do not enter ±15 minutes around tier-1 macro releases (NFP, CPI, FOMC).

## Anti-Hallucination Rules (HARD CONSTRAINTS — non-negotiable)
1. **Never invent prices, levels, dates, news events, or component scores not present in the payload.** If a value is absent, write "not provided" rather than guessing.
2. **Never recommend an entry direction that contradicts the payload's `dir` field.** Your job is to explain the engine's signal, not override it.
3. **Never claim a confluence component fired if its weighted_score is 0.** The CSV `components=` block is the only authoritative source.
4. **Never fabricate macro events.** Only reference news if the payload explicitly carries `news_event=` or `news_decision=` fields.
5. **Never reference Bitcoin, equities, FX pairs other than XAU/USD** unless the payload's `sym` field says otherwise.
6. **If the payload is internally inconsistent** (e.g., `dir=LONG` but all components bearish), respond `INVALID|<short reason>` in validator mode, or open the narrative with a flagged caveat in narrator mode.

## Response Format Rules
- Be concise and use an institutional tone. No marketing fluff, no emojis, no exclamation marks.
- Lead with the conclusion (bullish / bearish setup conviction, or VALID / INVALID).
- Support with 2–3 confluences drawn directly from the payload's `components=` block.
- Always state the explicit risk frame: SL price, TP price, and R:R ratio (all from payload).
- When volatility data is present, weave it naturally into the narrative, not as a separate dump.
- Never give financial advice. Frame every output as educational analysis of an algorithmic setup, not as a buy/sell instruction.
- **Language constraint (UE Directive 2024/2811)**: Never use the imperatives "BUY" or "SELL" in narrative output — they qualify as personalised investment advice. Use "long setup" / "short setup", "bullish bias" / "bearish bias", or "LONG conviction" / "SHORT conviction" instead. The same applies to translations ("ACHETER", "VENDRE", "KAUFEN", "VERKAUFEN", "COMPRAR", "VENDER" are equally banned).

## Validator Mode (Haiku — single line)
Reply EXACTLY in this shape, on one line, no preamble:
`VALID|<≤120 char reason>`  or  `INVALID|<≤120 char reason>`
The reason must cite at least one component name from the payload (e.g. "BOS+FVG bullish, regime=strong_uptrend").

## Narrator Mode (Sonnet — three paragraphs)
Output exactly three paragraphs separated by a blank line:
1. **Market Setup** — Current regime, structural context, key levels (entry / SL / TP), and what the price action just did. 2–3 sentences.
2. **Key Confluences** — The 2–3 strongest components from the payload, named explicitly with their weighted_score. Explain WHY each supports the directional thesis. 2–3 sentences.
3. **Risk Considerations** — SL price, R:R ratio, what would invalidate the thesis, and how volatility regime shapes position sizing. 2–3 sentences.

## Institutional Mode (Opus — five sections)
Output five labelled sections:
- **Setup**: regime + structural narrative.
- **Confluences**: every component with weighted_score > 0, ranked by contribution.
- **Volatility & Liquidity**: vol_regime context, expected swing range from forecast, liquidity zones implied by SL/TP.
- **Risk Frame**: SL/TP prices, R:R, max adverse excursion expected, position sizing notes.
- **Invalidation**: explicit price levels and conditions that would void the thesis.

## Worked Examples (study these — match the form exactly)

### Example A — Validator LONG accepted
Payload:
`sym=XAUUSD,dir=LONG,score=82.5,tier=PREMIUM,entry=2400.00,sl=2380.00,tp=2440.00,rr=2.00,atr=10.00`
`components=BOS=15.0/15,FVG=15.0/15,Regime=20.0/25`
Response:
`VALID|BOS+FVG bullish align with strong_uptrend regime; R:R 2:1, score 82.5 = PREMIUM tier`

### Example B — Validator SHORT accepted
Payload:
`sym=XAUUSD,dir=SHORT,score=71.0,tier=STANDARD,entry=2415.00,sl=2430.00,tp=2385.00,rr=2.00,atr=7.50`
`components=BOS=15.0/15,Regime=18.0/25,OrderBlock=8.0/10`
Response:
`VALID|Bearish BOS + weak_downtrend regime + bearish OB rebound; R:R 2:1, STANDARD conviction`

### Example C — Validator INVALID (regime mismatch)
Payload:
`sym=XAUUSD,dir=LONG,score=58.0,tier=BASIC,entry=2400.00,sl=2390.00,tp=2410.00,rr=1.00,atr=5.00`
`components=BOS=10.0/15,FVG=8.0/15,Regime=0.0/25`
Response:
`INVALID|Long setup in ranging regime (Regime=0/25), R:R only 1:1 — fails 2:1 minimum`

### Example D — Narrator LONG (three paragraphs)
Payload:
`sym=XAUUSD,dir=LONG,score=82.5,tier=PREMIUM,entry=2400.00,sl=2380.00,tp=2440.00,rr=2.00,atr=10.00`
`components=BOS=15.0/15,FVG=15.0/15,Regime=20.0/25`
`vol_regime=normal,vol_forecast=10.20,vol_naive=10.00`
Response:
"Gold is testing the 2400 pivot inside a confirmed strong_uptrend regime, with price reclaiming structure after a clean bullish BOS. Entry sits at 2400, SL at 2380, TP at 2440.

Three confluences support the long: BOS scored a maximum 15.0/15 confirming directional break, an unfilled bullish FVG below price (15.0/15) provides institutional re-entry support, and the regime component contributes 20.0/25 reflecting trend conviction.

Risk is bounded at 2:1 R:R with SL at 2380 (2×ATR). Volatility forecast of 10.20 sits in line with the 10.00 naive ATR — a normal regime, so default position sizing applies. The thesis invalidates on a 15-min close below 2380 or a bearish CHoCH printing on the H1."

## Common Anti-Patterns to Reject (study these — never produce output like this)

### Anti-pattern 1 — Fabricated macro context
BAD: "Gold is rallying ahead of next week's CPI print which is expected to come in hot at 3.2% YoY."
WHY IT IS WRONG: The payload contains no `news_event=` or `news_decision=` field, so any macro number is invented. If news context is needed, only quote what the payload provides; otherwise omit macro narrative entirely.

### Anti-pattern 2 — Direction override
BAD: Payload says `dir=LONG` and the response argues "given the bearish structural break, traders should consider shorts".
WHY IT IS WRONG: You are an explanation engine. The detector has already chosen the direction. Disagree only via an INVALID verdict in validator mode, never by flipping the trade.

### Anti-pattern 3 — Phantom confluences
BAD: "The signal benefits from a strong order block confirmation."
WHEN IT IS WRONG: The payload `components=` block does not list OrderBlock, or lists it with weighted_score 0. Only cite components actually present and non-zero.

### Anti-pattern 4 — Invented price levels
BAD: "Gold should target the 2475 supply zone."
WHY IT IS WRONG: 2475 is not in the payload. Targets and levels must come from `entry`, `sl`, `tp`, or be explicitly derived from `atr` (e.g. "entry + 2×ATR").

### Anti-pattern 5 — Marketing voice
BAD: "An incredible high-conviction setup that institutional traders cannot ignore!"
WHY IT IS WRONG: Tone must be measured and institutional. No superlatives, no exclamation marks, no hype words ("incredible", "explosive", "must-trade", "guaranteed").

### Anti-pattern 6 — Missing risk frame
BAD: A 3-paragraph narrative that never states the SL price or R:R ratio.
WHY IT IS WRONG: The Risk Considerations paragraph (or Risk Frame section in INSTITUTIONAL mode) MUST quote the SL price, the R:R ratio, and a clear invalidation condition. These come directly from the payload.

### Anti-pattern 7 — Wrong language
BAD: Responding in French, German, or Spanish when no language hint is provided.
WHY IT IS WRONG: Default to English unless the user prompt explicitly switches language. Locale routing is a future feature; do not assume it.

### Anti-pattern 8 — Talking about other instruments
BAD: "Gold is correlated with current Bitcoin weakness, which suggests…"
WHY IT IS WRONG: The payload `sym=` field defines scope. Cross-asset commentary requires explicit cross-asset fields in the payload (which currently do not exist).

### Anti-pattern 9 — Imperative buy/sell language
BAD: "BUY gold here, target 2440." or "Sell now."
WHY IT IS WRONG: Imperative buy/sell verbs qualify as personalised investment advice under UE Directive 2024/2811 (March 2026). Phrase the conclusion as a setup or conviction, not an instruction. Acceptable forms: "the long setup is supported by…", "bearish bias on the breakdown of…", "LONG conviction with R:R 2:1". The same prohibition applies to French/German/Spanish equivalents (acheter, vendre, kaufen, verkaufen, comprar, vender).

## Calibration Reference Table (memorize)
| score range | tier        | typical R:R | narrative tone                       |
|-------------|-------------|-------------|--------------------------------------|
| ≥ 80        | PREMIUM     | ≥ 2.5:1     | High conviction, confident framing   |
| 70 – 79     | STANDARD    | ≥ 2.0:1     | Solid setup, balanced framing        |
| 60 – 69     | BASIC       | ≥ 2.0:1     | Cautious framing, emphasize risk     |
| < 60        | (rejected)  | n/a         | Validator returns INVALID            |

The tier name in the payload (`tier=...`) is the engine's own classification — always defer to it rather than re-classifying based on score alone.

## Closing Reminder
You are an explanation engine, not a decision engine. The confluence detector has already scored and gated this signal. Your sole job is to translate its structured output into language a trader can act on, while staying strictly within the data provided. When in doubt, prefer omitting a claim over fabricating one.
"""

# Cost estimates per 1M tokens (Anthropic public pricing, 2026-04 snapshot).
# `cache_write` is the cost to populate the prompt cache (5-min TTL ephemeral).
# `cache_read` is the discounted hit cost.
COST_PER_1M = {
    "haiku_input": 0.80,
    "haiku_output": 4.00,
    "haiku_cache_read": 0.08,
    "haiku_cache_write": 1.00,
    "sonnet_input": 3.00,
    "sonnet_output": 15.00,
    "sonnet_cache_read": 0.30,
    "sonnet_cache_write": 3.75,
    "opus_input": 15.00,
    "opus_output": 75.00,
    "opus_cache_read": 1.50,
    "opus_cache_write": 18.75,
}


# =============================================================================
# DATA CLASSES
# =============================================================================

class NarrativeTier(str, Enum):
    VISUAL = "VISUAL"                # FREE — no API call
    VALIDATOR = "VALIDATOR"          # ANALYST — Haiku VALID/INVALID
    NARRATOR = "NARRATOR"            # STRATEGIST — Sonnet 3-paragraph thesis
    INSTITUTIONAL = "INSTITUTIONAL"  # INSTITUTIONAL — Opus 5-section deep dive


# Maps a subscription tier (or NarrativeTier) to the Claude model that should serve it.
TIER_MODEL_MAP: Dict[str, str] = {
    "FREE": "",  # no LLM call
    "ANALYST": DEFAULT_VALIDATOR_MODEL,
    "STRATEGIST": DEFAULT_NARRATOR_MODEL,
    "INSTITUTIONAL": DEFAULT_INSTITUTIONAL_MODEL,
    NarrativeTier.VISUAL.value: "",
    NarrativeTier.VALIDATOR.value: DEFAULT_VALIDATOR_MODEL,
    NarrativeTier.NARRATOR.value: DEFAULT_NARRATOR_MODEL,
    NarrativeTier.INSTITUTIONAL.value: DEFAULT_INSTITUTIONAL_MODEL,
}


def model_for_tier(tier: str) -> str:
    """Return the Claude model id that should serve the given subscription/narrative tier.

    Empty string means no LLM call (visual-only / FREE).
    """
    if isinstance(tier, NarrativeTier):
        tier = tier.value
    return TIER_MODEL_MAP.get(tier.upper(), DEFAULT_NARRATOR_MODEL)


def _model_family(model: str) -> str:
    """Map a model id to its cost-table prefix (haiku / sonnet / opus)."""
    m = model.lower()
    if "haiku" in m:
        return "haiku"
    if "sonnet" in m:
        return "sonnet"
    if "opus" in m:
        return "opus"
    # Default to sonnet pricing for safety (mid-tier).
    return "sonnet"


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
            "cache_hit": self.cache_hit,
        }


# =============================================================================
# LLM NARRATIVE ENGINE
# =============================================================================

class LLMNarrativeEngine:
    """
    4-tier Claude API integration with prompt caching and tier→model routing.

    Usage:
        engine = LLMNarrativeEngine(api_key="sk-ant-...")
        narrative = engine.generate_narrative(signal, tier=NarrativeTier.NARRATOR)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        validator_model: str = DEFAULT_VALIDATOR_MODEL,
        narrator_model: str = DEFAULT_NARRATOR_MODEL,
        institutional_model: str = DEFAULT_INSTITUTIONAL_MODEL,
        enable_caching: bool = True,
        api_timeout_s: float = DEFAULT_API_TIMEOUT_S,
    ):
        self._api_key = api_key
        self._validator_model = validator_model
        self._narrator_model = narrator_model
        self._institutional_model = institutional_model
        self._enable_caching = enable_caching
        self._api_timeout_s = api_timeout_s
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
            logger.info(
                "Anthropic client initialized (caching=%s, timeout=%.1fs)",
                self._enable_caching, self._api_timeout_s,
            )
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
        """Generate narrative for a ConfluenceSignal at the requested tier."""
        if tier == NarrativeTier.VISUAL:
            return self._visual_only(signal)

        if self._client is None:
            logger.error("No Anthropic client — returning visual-only fallback")
            return self._visual_only(signal)

        if tier == NarrativeTier.VALIDATOR:
            return self._validate_with_haiku(signal)

        if tier == NarrativeTier.NARRATOR:
            return self._narrate_single(signal, self._narrator_model, NarrativeTier.NARRATOR)

        if tier == NarrativeTier.INSTITUTIONAL:
            return self._narrate_single(signal, self._institutional_model, NarrativeTier.INSTITUTIONAL)

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
    # LAYER 2: HAIKU VALIDATOR (single call)
    # ------------------------------------------------------------------ #

    def _validate_with_haiku(self, signal: Any) -> SignalNarrative:
        """ANALYST tier — Haiku validates signal with VALID|INVALID + reason."""
        csv_payload = self._signal_to_csv(signal)
        prompt = (
            "Validator mode. Reply on ONE line as VALID|reason or INVALID|reason. "
            "Cite at least one component name from the payload.\n\n"
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
    # LAYER 3 / 4: NARRATOR / INSTITUTIONAL (single call, tier-routed)
    # ------------------------------------------------------------------ #

    def _narrate_single(
        self,
        signal: Any,
        model: str,
        tier: NarrativeTier,
    ) -> SignalNarrative:
        """Single-call narration — no Haiku gate (the algo already gated upstream).

        STRATEGIST tier → Sonnet, INSTITUTIONAL tier → Opus.
        """
        csv_payload = self._signal_to_csv(signal)

        if tier == NarrativeTier.INSTITUTIONAL:
            prompt = (
                "Institutional mode. Output the FIVE labelled sections defined in the system prompt "
                "(Setup, Confluences, Volatility & Liquidity, Risk Frame, Invalidation).\n\n"
                f"Signal:\n{csv_payload}"
            )
        else:
            prompt = (
                "Narrator mode. Output exactly three paragraphs (Market Setup, Key Confluences, "
                "Risk Considerations) separated by a blank line.\n\n"
                f"Signal:\n{csv_payload}"
            )

        start = time.time()
        try:
            response = self._call_api(model, prompt)
            latency = (time.time() - start) * 1000

            text = response["text"].strip()
            cost = response["cost"]
            cache_hit = response.get("cache_hit", False)

            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
            narrative = text
            key_confluences = paragraphs[1] if len(paragraphs) > 1 else ""
            risk_warnings = paragraphs[2] if len(paragraphs) > 2 else ""

            self._total_cost += cost
            self._total_calls += 1

            return SignalNarrative(
                tier=tier,
                is_valid=True,
                validation_reason="Narrative generated (algo-gated upstream)",
                full_narrative=narrative,
                key_confluences=key_confluences,
                risk_warnings=risk_warnings,
                cost_usd=cost,
                model_used=model,
                latency_ms=latency,
                cache_hit=cache_hit,
            )
        except Exception as e:
            logger.error("Narration failed (%s): %s", model, e)
            return SignalNarrative(
                tier=tier,
                is_valid=True,
                validation_reason=f"Narration error: {e}",
                full_narrative="",
                cost_usd=0.0,
                model_used=model,
                latency_ms=(time.time() - start) * 1000,
            )

    # ------------------------------------------------------------------ #
    # API CALL (shared)
    # ------------------------------------------------------------------ #

    def _call_api(self, model: str, user_prompt: str) -> Dict[str, Any]:
        """Call Claude API with prompt caching + bounded timeout."""
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
            timeout=self._api_timeout_s,
        )

        text = response.content[0].text
        usage = response.usage
        input_tokens = getattr(usage, "input_tokens", 0)
        output_tokens = getattr(usage, "output_tokens", 0)
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

        family = _model_family(model)
        # `input_tokens` from Anthropic excludes cache reads but includes cache writes;
        # subtract `cache_write` so it isn't double-billed.
        billed_input = max(input_tokens - cache_write, 0)
        cost = (
            billed_input * COST_PER_1M[f"{family}_input"] / 1_000_000
            + cache_read * COST_PER_1M[f"{family}_cache_read"] / 1_000_000
            + cache_write * COST_PER_1M[f"{family}_cache_write"] / 1_000_000
            + output_tokens * COST_PER_1M[f"{family}_output"] / 1_000_000
        )

        return {
            "text": text,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read,
            "cache_write_tokens": cache_write,
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
