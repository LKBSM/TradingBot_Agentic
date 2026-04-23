"""Algorithmic narrative engine — deterministic, LLM-free replacement for LLMNarrativeEngine.

Generates institutional-tone signal explanations using only the signal's own data
(components, score, regime, volatility forecast, risk parameters).

Drop-in compatible with LLMNarrativeEngine: same `generate_narrative(signal, tier)` and
`get_stats()` interface, same `NarrativeTier` enum, same `SignalNarrative` result type.

Cost per call: $0. Latency: sub-millisecond. Deterministic output.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from src.intelligence.llm_narrative_engine import NarrativeTier, SignalNarrative

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATOR RULES (algorithmic gate — replaces Haiku Y/N)
# =============================================================================
# Thresholds align with ConfluenceDetector.SignalTier:
#   PREMIUM >= 80, STANDARD >= 60, WEAK >= 40, INVALID < 40
# We reject anything below STANDARD — WEAK signals are real but too noisy for
# paid-tier narration.

MIN_SCORE_VALID = 60.0         # STANDARD tier or better
MIN_RR_VALID = 1.5              # sub-1.5 R:R signals get rejected
MIN_DOMINANT_COMPONENT = 0.55   # at least one component at >=55% of its weight
# Regime veto intentionally removed: ConfluenceDetector.score_regime already
# zeroes or discounts opposing regimes. Second-guessing it here creates two
# gates on the same decision.


# =============================================================================
# PHRASING DICTIONARIES (variation by signal characteristic)
# =============================================================================

_TIER_ADJECTIVE = {
    "PREMIUM": "high-conviction",
    "STANDARD": "constructive",
    "WEAK": "marginal",
    "INVALID": "sub-threshold",
}

_TIER_HEADLINE = {
    "PREMIUM": "Strong conviction",
    "STANDARD": "Constructive setup",
    "WEAK": "Marginal setup",
    "INVALID": "Setup below threshold",
}

_VOL_REGIME_DESCRIPTION = {
    "low": "compressed volatility with tight expected ranges",
    "normal": "standard volatility conditions",
    "high": "elevated volatility with wider expected ranges",
}

_VOL_REGIME_RISK_GUIDANCE = {
    "low": (
        "Tight volatility supports standard stop placement, but shallow ranges may "
        "cap extension; partial profit-taking before the 4×ATR target is prudent."
    ),
    "normal": (
        "Volatility is in its typical band, so the 2×ATR stop and 4×ATR target "
        "carry their usual probabilistic weight."
    ),
    "high": (
        "Elevated volatility argues for a wider stop (~1.5x standard) and reduced "
        "position size; whipsaw risk is materially higher in this regime."
    ),
}

_COMPONENT_DESCRIPTIONS = {
    "BOS": "break of structure",
    "FVG": "fair value gap imbalance",
    "OrderBlock": "institutional order block",
    "CHoCH": "change of character",
    "Regime": "prevailing market regime",
    "News": "news/sentiment backdrop",
    "Volume": "volume profile",
    "Momentum": "momentum (RSI)",
    "RSI_Divergence": "RSI divergence",
}


# =============================================================================
# TEMPLATE NARRATIVE ENGINE
# =============================================================================

class TemplateNarrativeEngine:
    """Algorithmic narrative generator — drop-in for LLMNarrativeEngine.

    Usage:
        engine = TemplateNarrativeEngine()
        narrative = engine.generate_narrative(signal, tier=NarrativeTier.NARRATOR)
    """

    def __init__(self, **_kwargs: Any) -> None:
        # Accept arbitrary kwargs for API parity with LLMNarrativeEngine.
        self._total_calls: int = 0
        self._total_cost: float = 0.0  # always zero, kept for interface parity

    # ------------------------------------------------------------------ #
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def generate_narrative(
        self,
        signal: Any,
        tier: NarrativeTier = NarrativeTier.VISUAL,
    ) -> SignalNarrative:
        start = time.time()

        if tier == NarrativeTier.VISUAL:
            result = self._visual_only()
        elif tier == NarrativeTier.VALIDATOR:
            result = self._validate(signal)
        elif tier == NarrativeTier.NARRATOR:
            result = self._narrate(signal)
        else:
            result = self._visual_only()

        result.latency_ms = (time.time() - start) * 1000
        result.model_used = "template-v1"
        self._total_calls += 1
        return result

    # ------------------------------------------------------------------ #
    # VISUAL (parity with LLM visual tier)
    # ------------------------------------------------------------------ #

    def _visual_only(self) -> SignalNarrative:
        return SignalNarrative(
            tier=NarrativeTier.VISUAL,
            is_valid=True,
            validation_reason="Visual-only tier (structured signal data)",
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------ #
    # VALIDATOR (algorithmic gate)
    # ------------------------------------------------------------------ #

    def _validate(self, signal: Any) -> SignalNarrative:
        is_valid, reason = self._validation_check(signal)
        return SignalNarrative(
            tier=NarrativeTier.VALIDATOR,
            is_valid=is_valid,
            validation_reason=reason,
            cost_usd=0.0,
        )

    def _validation_check(self, signal: Any) -> Tuple[bool, str]:
        score = float(getattr(signal, "confluence_score", 0.0))
        rr = float(getattr(signal, "rr_ratio", 0.0))
        tier = str(getattr(signal, "tier", ""))

        if tier == "INVALID" or score < MIN_SCORE_VALID:
            return False, (
                f"Confluence score {score:.0f} below {MIN_SCORE_VALID:.0f} threshold"
            )

        if rr < MIN_RR_VALID:
            return False, (
                f"Risk/reward {rr:.2f} below {MIN_RR_VALID:.2f} minimum"
            )

        components = list(getattr(signal, "components", []))
        if not components:
            return False, "No confluence components present"

        dominant = self._dominant_components(components, limit=3)
        if not dominant:
            return False, "No component meets the dominance threshold"

        top_name = dominant[0][0]
        top_ratio = dominant[0][1]
        if top_ratio < MIN_DOMINANT_COMPONENT:
            return False, (
                f"Top component '{top_name}' at {top_ratio*100:.0f}% of weight — "
                f"no dominant confluence"
            )

        direction = str(getattr(signal, "signal_type", "LONG"))
        strong_list = ", ".join(
            f"{name} ({ratio*100:.0f}%)" for name, ratio, _ in dominant
        )
        return True, (
            f"{direction} validated at score {score:.0f}, R:R {rr:.2f} — "
            f"top confluences: {strong_list}"
        )

    # ------------------------------------------------------------------ #
    # NARRATOR (3-paragraph structured narrative)
    # ------------------------------------------------------------------ #

    def _narrate(self, signal: Any) -> SignalNarrative:
        is_valid, reason = self._validation_check(signal)

        if not is_valid:
            return SignalNarrative(
                tier=NarrativeTier.NARRATOR,
                is_valid=False,
                validation_reason=reason,
                full_narrative="",
                cost_usd=0.0,
            )

        setup = self._paragraph_market_setup(signal)
        confluences = self._paragraph_confluences(signal)
        risk = self._paragraph_risk(signal)

        full = "\n\n".join([setup, confluences, risk])

        return SignalNarrative(
            tier=NarrativeTier.NARRATOR,
            is_valid=True,
            validation_reason=reason,
            full_narrative=full,
            key_confluences=confluences,
            risk_warnings=risk,
            cost_usd=0.0,
        )

    # ------------------------------------------------------------------ #
    # PARAGRAPH BUILDERS
    # ------------------------------------------------------------------ #

    def _paragraph_market_setup(self, signal: Any) -> str:
        symbol = str(getattr(signal, "symbol", "XAUUSD"))
        direction = str(getattr(signal, "signal_type", "LONG"))
        score = float(getattr(signal, "confluence_score", 0.0))
        tier = str(getattr(signal, "tier", "STANDARD"))
        entry = float(getattr(signal, "entry_price", 0.0))
        atr = float(getattr(signal, "atr", 0.0))

        tier_headline = _TIER_HEADLINE.get(tier, "Setup")
        tier_adj = _TIER_ADJECTIVE.get(tier, "constructive")

        # Regime context
        regime_comp = self._find_component(list(getattr(signal, "components", [])), "Regime")
        regime_phrase = self._regime_phrase(regime_comp, direction)

        # Vol context
        vol_phrase = self._vol_context_phrase(signal)

        decimals = self._price_decimals(symbol)

        return (
            f"**Market Setup.** {tier_headline} — a {tier_adj} {direction} on {symbol} "
            f"prints with a confluence score of {score:.0f}/100. "
            f"{regime_phrase} "
            f"Price is working off {entry:.{decimals}f} against an ATR of {atr:.{decimals}f}. "
            f"{vol_phrase}"
        ).strip()

    def _paragraph_confluences(self, signal: Any) -> str:
        components = list(getattr(signal, "components", []))
        direction = str(getattr(signal, "signal_type", "LONG"))
        dominant = self._dominant_components(components, limit=3)

        if not dominant:
            return (
                "**Key Confluences.** No single component dominates; the score "
                "reflects a diffuse alignment across the stack."
            )

        lines = [
            f"**Key Confluences.** The {direction.lower()} case rests on "
            f"{len(dominant)} dominant factor"
            f"{'s' if len(dominant) > 1 else ''}."
        ]
        for i, (name, ratio, comp) in enumerate(dominant, 1):
            desc = _COMPONENT_DESCRIPTIONS.get(name, name)
            reasoning = str(getattr(comp, "reasoning", "")).strip()
            strength_phrase = self._strength_phrase(ratio)
            line = f"({i}) {desc.capitalize()} — {strength_phrase} at {ratio*100:.0f}% of weight"
            if reasoning and reasoning.lower() != "none":
                line += f": {reasoning}"
            line += "."
            lines.append(line)

        return " ".join(lines)

    def _paragraph_risk(self, signal: Any) -> str:
        symbol = str(getattr(signal, "symbol", "XAUUSD"))
        direction = str(getattr(signal, "signal_type", "LONG"))
        entry = float(getattr(signal, "entry_price", 0.0))
        sl = float(getattr(signal, "stop_loss", 0.0))
        tp = float(getattr(signal, "take_profit", 0.0))
        rr = float(getattr(signal, "rr_ratio", 0.0))
        atr = float(getattr(signal, "atr", 0.0))
        vol_regime = getattr(signal, "vol_regime", None)

        decimals = self._price_decimals(symbol)

        sl_distance = abs(entry - sl)
        tp_distance = abs(tp - entry)
        sl_atr_mult = (sl_distance / atr) if atr > 0 else 0.0
        tp_atr_mult = (tp_distance / atr) if atr > 0 else 0.0

        vol_guidance = _VOL_REGIME_RISK_GUIDANCE.get(
            str(vol_regime).lower() if vol_regime else "normal",
            _VOL_REGIME_RISK_GUIDANCE["normal"],
        )

        invalidation_direction = "below" if direction == "LONG" else "above"

        return (
            f"**Risk Considerations.** Stop at {sl:.{decimals}f} "
            f"({sl_atr_mult:.1f}×ATR, {sl_distance:.{decimals}f} pts away); "
            f"target at {tp:.{decimals}f} ({tp_atr_mult:.1f}×ATR) for a {rr:.2f}:1 "
            f"reward/risk profile. {vol_guidance} "
            f"A close {invalidation_direction} {sl:.{decimals}f} invalidates the thesis "
            f"and the position should be exited without discretion."
        )

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    def _dominant_components(
        self, components: List[Any], limit: int = 3
    ) -> List[Tuple[str, float, Any]]:
        """Return components sorted by weighted_score / weight ratio, top `limit`.

        Filters out components with zero weight. Excludes components whose ratio is
        below the neutral-fallback line (0.5) when another stronger signal exists.
        """
        scored = []
        for c in components:
            ratio = self._component_ratio(c)
            if ratio > 0.5:  # only include meaningfully active components
                scored.append((str(getattr(c, "name", "")), ratio, c))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    @staticmethod
    def _component_ratio(comp: Any) -> float:
        weight = float(getattr(comp, "weight", 0.0))
        if weight <= 0:
            return 0.0
        weighted = float(getattr(comp, "weighted_score", 0.0))
        return max(0.0, min(1.0, weighted / weight))

    @staticmethod
    def _find_component(components: List[Any], name: str) -> Optional[Any]:
        for c in components:
            if str(getattr(c, "name", "")) == name:
                return c
        return None

    @staticmethod
    def _strength_phrase(ratio: float) -> str:
        if ratio >= 0.90:
            return "very strong"
        if ratio >= 0.75:
            return "strong"
        if ratio >= 0.60:
            return "solid"
        return "moderate"

    def _regime_phrase(self, regime_comp: Optional[Any], direction: str) -> str:
        if regime_comp is None:
            return "Regime context is unavailable."
        ratio = self._component_ratio(regime_comp)
        reasoning = str(getattr(regime_comp, "reasoning", "")).strip()

        if ratio >= 0.75:
            lead = f"The regime aligns strongly with the {direction.lower()} thesis"
        elif ratio >= 0.60:
            lead = f"The regime is supportive of the {direction.lower()} thesis"
        elif ratio >= 0.50:
            lead = f"The regime is neutrally permissive"
        else:
            lead = f"The regime offers only weak support"

        if reasoning and reasoning.lower() != "none":
            return f"{lead} ({reasoning})."
        return f"{lead}."

    def _vol_context_phrase(self, signal: Any) -> str:
        vol_regime = getattr(signal, "vol_regime", None)
        vol_forecast = getattr(signal, "vol_forecast_atr", None)
        naive_atr = float(getattr(signal, "atr", 0.0))

        if vol_regime is None and vol_forecast is None:
            return "Volatility forecast is unavailable; assume standard ATR-based risk."

        regime_desc = _VOL_REGIME_DESCRIPTION.get(
            str(vol_regime).lower() if vol_regime else "normal",
            "standard volatility conditions",
        )

        if vol_forecast is not None and naive_atr > 0:
            ratio = vol_forecast / naive_atr
            if ratio >= 1.15:
                drift = (
                    f"The volatility model projects expansion (forecast ATR "
                    f"{vol_forecast:.2f} vs naive {naive_atr:.2f}, +{(ratio-1)*100:.0f}%)"
                )
            elif ratio <= 0.85:
                drift = (
                    f"The volatility model projects compression (forecast ATR "
                    f"{vol_forecast:.2f} vs naive {naive_atr:.2f}, {(ratio-1)*100:.0f}%)"
                )
            else:
                drift = (
                    f"The volatility model is in line with realized ATR "
                    f"(forecast {vol_forecast:.2f} vs naive {naive_atr:.2f})"
                )
            return f"Conditions reflect {regime_desc}; {drift}."

        return f"Conditions reflect {regime_desc}."

    @staticmethod
    def _price_decimals(symbol: str) -> int:
        s = symbol.upper()
        if "JPY" in s:
            return 3
        if s.startswith("XAU") or "BTC" in s or s.startswith("US"):
            return 2
        return 5

    # ------------------------------------------------------------------ #
    # STATS (parity with LLMNarrativeEngine)
    # ------------------------------------------------------------------ #

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_cost_usd": 0.0,
            "total_calls": self._total_calls,
            "avg_cost_per_call": 0.0,
            "engine": "template-v1",
        }
