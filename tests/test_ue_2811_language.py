"""UE Directive 2024/2811 (March 2026) — narrative-engine language guards.

These tests are the UE compliance ratchet: they assert that no narrative
surface (system prompt examples, template engine paragraphs, validator
reasons) emits an imperative buy/sell verb in any of the four supported
languages. If a future change reintroduces such a verb, the test must fail.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from src.intelligence.llm_narrative_engine import (
    NarrativeTier,
    SMC_SYSTEM_PROMPT,
)
from src.intelligence.template_narrative_engine import TemplateNarrativeEngine


# ─── Banned imperatives (and their EU-language equivalents) ────────────────

#: Whole-word imperatives that qualify as personalised investment advice.
#: We match case-insensitive whole tokens, with a small allowlist for
#: legitimate analytical compounds ("BUY-side", "sell-off", "selling pressure").
BANNED_TOKENS = (
    "BUY",
    "SELL",
    "ACHETER",
    "VENDRE",
    "KAUFEN",
    "VERKAUFEN",
    "COMPRAR",
    "VENDER",
)

#: Substrings that legitimately *contain* a banned token but are not
#: imperative recommendations. The check uses these for whitelisting.
ALLOWED_CONTEXTS = (
    "buy-side",          # market-microstructure term
    "sell-off",          # established price-action term
    "sell off",          # established price-action term
    "selling pressure",  # descriptive
    "buying pressure",   # descriptive
    "best buy",          # company / shop reference (defensive)
)


def _contains_banned_token(text: str) -> List[str]:
    """Return the list of banned tokens found as standalone words.

    Whitelisted compound terms (``ALLOWED_CONTEXTS``) are masked out before
    the search so legitimate market-structure jargon does not flag.
    """
    haystack = text.lower()
    for ok in ALLOWED_CONTEXTS:
        haystack = haystack.replace(ok, "[allowed]")

    found: List[str] = []
    import re
    for token in BANNED_TOKENS:
        # Whole-word match, case-insensitive, ASCII boundaries.
        if re.search(rf"\b{re.escape(token.lower())}\b", haystack):
            found.append(token)
    return found


# ─── System prompt audit ──────────────────────────────────────────────────


_NEGATIVE_CONTEXT_MARKERS = (
    "never",
    "not",
    "banned",
    "forbid",
    "forbidden",
    "bad:",
    "avoid",
    "anti-pattern",
    "language constraint",
    "imperative",
    "qualify as",
    "instruction.",
)


def _is_in_negative_context(prompt: str, position: int, window: int = 500) -> bool:
    """True iff the substring around ``position`` includes a marker that
    frames the BUY/SELL token as forbidden rather than recommended."""
    start = max(0, position - window)
    end = min(len(prompt), position + window)
    chunk = prompt[start:end].lower()
    return any(marker in chunk for marker in _NEGATIVE_CONTEXT_MARKERS)


class TestSystemPromptUE2811:
    def test_system_prompt_only_uses_buy_sell_in_negative_context(self):
        """Each BUY/SELL match must sit inside a 'never/forbid/BAD' window —
        the prompt may *name* the imperatives only to forbid them."""
        import re
        offenders: List[str] = []
        for tok in BANNED_TOKENS:
            for match in re.finditer(rf"\b{re.escape(tok.lower())}\b", SMC_SYSTEM_PROMPT.lower()):
                if not _is_in_negative_context(SMC_SYSTEM_PROMPT, match.start()):
                    offenders.append(
                        f"'{tok}' at offset {match.start()}: "
                        f"…{SMC_SYSTEM_PROMPT[max(0, match.start()-40):match.start()+40]}…"
                    )
        assert not offenders, (
            "SMC_SYSTEM_PROMPT contains imperative buy/sell verb(s) outside a "
            "'forbid/banned/anti-pattern' block:\n  " + "\n  ".join(offenders)
        )

    def test_anti_pattern_9_explicitly_bans_imperatives(self):
        assert "Anti-pattern 9" in SMC_SYSTEM_PROMPT
        # The anti-pattern body must mention the four blocked languages.
        block = SMC_SYSTEM_PROMPT[SMC_SYSTEM_PROMPT.find("Anti-pattern 9"):]
        for word in ("acheter", "vendre", "kaufen", "verkaufen", "comprar", "vender"):
            assert word in block.lower(), f"anti-pattern 9 missing {word}"
        assert "2024/2811" in block

    def test_example_headers_use_long_short_not_buy_sell(self):
        # Examples A, B, D used to be labelled "BUY accepted" / "SELL accepted".
        for header in (
            "Example A — Validator BUY",
            "Example B — Validator SELL",
            "Example D — Narrator BUY",
        ):
            assert header not in SMC_SYSTEM_PROMPT


# ─── Template engine audit ─────────────────────────────────────────────────


@dataclass
class _MockComp:
    name: str
    weighted_score: float
    weight: float
    raw_value: float = 1.0
    reasoning: str = ""


@dataclass
class _MockSignal:
    symbol: str = "XAUUSD"
    signal_type: str = "LONG"
    confluence_score: float = 75.0
    tier: str = "STANDARD"
    entry_price: float = 2400.0
    stop_loss: float = 2380.0
    take_profit: float = 2440.0
    rr_ratio: float = 2.0
    atr: float = 10.0
    components: List[_MockComp] = field(
        default_factory=lambda: [
            _MockComp("BOS", 14.0, 15.0, 1.0, "bullish BOS"),
            _MockComp("FVG", 13.0, 15.0, 1.0, "bullish FVG"),
            _MockComp("Regime", 18.0, 25.0, 1.0, "strong_uptrend"),
        ]
    )
    vol_regime: Optional[str] = "normal"
    vol_forecast_atr: Optional[float] = 10.0


class TestTemplateEngineUE2811:
    def test_long_narrative_avoids_buy_sell(self):
        engine = TemplateNarrativeEngine()
        result = engine.generate_narrative(_MockSignal(signal_type="LONG"), tier=NarrativeTier.NARRATOR)
        offenders = _contains_banned_token(result.full_narrative)
        assert offenders == [], (
            f"LONG template narrative contains banned imperatives: {offenders}"
        )

    def test_short_narrative_avoids_buy_sell(self):
        engine = TemplateNarrativeEngine()
        sig = _MockSignal(
            signal_type="SHORT",
            entry_price=2415.0,
            stop_loss=2435.0,
            take_profit=2375.0,
        )
        result = engine.generate_narrative(sig, tier=NarrativeTier.NARRATOR)
        offenders = _contains_banned_token(result.full_narrative)
        assert offenders == [], (
            f"SHORT template narrative contains banned imperatives: {offenders}"
        )

    def test_validator_reason_avoids_buy_sell(self):
        engine = TemplateNarrativeEngine()
        result = engine.generate_narrative(_MockSignal(), tier=NarrativeTier.VALIDATOR)
        offenders = _contains_banned_token(result.validation_reason or "")
        assert offenders == []

    def test_long_narrative_uses_setup_or_bullish(self):
        engine = TemplateNarrativeEngine()
        result = engine.generate_narrative(_MockSignal(signal_type="LONG"), tier=NarrativeTier.NARRATOR)
        body = result.full_narrative.lower()
        assert ("long setup" in body) or ("bullish" in body), (
            "LONG narrative should frame direction as 'long setup' or 'bullish'"
        )

    def test_short_narrative_uses_setup_or_bearish(self):
        engine = TemplateNarrativeEngine()
        sig = _MockSignal(
            signal_type="SHORT",
            entry_price=2415.0,
            stop_loss=2435.0,
            take_profit=2375.0,
        )
        result = engine.generate_narrative(sig, tier=NarrativeTier.NARRATOR)
        body = result.full_narrative.lower()
        assert ("short setup" in body) or ("bearish" in body), (
            "SHORT narrative should frame direction as 'short setup' or 'bearish'"
        )

    def test_risk_paragraph_does_not_command_exit(self):
        # Old wording: "the position should be exited without discretion".
        # New wording avoids the imperative "should be exited".
        engine = TemplateNarrativeEngine()
        result = engine.generate_narrative(_MockSignal(), tier=NarrativeTier.NARRATOR)
        body = result.risk_warnings or result.full_narrative
        assert "should be exited without discretion" not in body
