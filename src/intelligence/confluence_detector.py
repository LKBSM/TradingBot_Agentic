"""Weighted confluence scoring engine for Smart Sentinel AI.

Replaces the RL policy with a deterministic, explainable signal generator.
Scores 0-100 based on SMC structure, regime, news, volume, and momentum.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & DATACLASSES
# =============================================================================

class SignalType(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class SignalTier(str, Enum):
    PREMIUM = "PREMIUM"      # >= 80
    STANDARD = "STANDARD"    # >= 60
    WEAK = "WEAK"            # >= 40
    INVALID = "INVALID"      # < 40


@dataclass
class ComponentScore:
    """Individual component contribution to the confluence score."""
    name: str
    raw_value: float      # Raw indicator value
    weighted_score: float  # After weight applied (0 to weight_max)
    weight: float          # Weight used
    reasoning: str         # Human-readable explanation


@dataclass
class ConfluenceSignal:
    """A scored trading signal with full explainability."""
    signal_id: str
    symbol: str
    signal_type: SignalType
    confluence_score: float           # 0-100
    tier: SignalTier
    entry_price: float
    stop_loss: float
    take_profit: float
    rr_ratio: float
    atr: float
    components: List[ComponentScore] = field(default_factory=list)
    reasoning: List[str] = field(default_factory=list)
    bar_timestamp: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(tz=None).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "signal_type": self.signal_type.value,
            "confluence_score": round(self.confluence_score, 2),
            "tier": self.tier.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "rr_ratio": round(self.rr_ratio, 2),
            "atr": self.atr,
            "components": [
                {"name": c.name, "weighted_score": round(c.weighted_score, 2),
                 "weight": c.weight, "reasoning": c.reasoning}
                for c in self.components
            ],
            "reasoning": self.reasoning,
            "bar_timestamp": self.bar_timestamp,
            "created_at": self.created_at,
        }


# =============================================================================
# DEFAULT WEIGHTS
# =============================================================================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "bos": 15.0,
    "fvg": 15.0,
    "order_block": 10.0,
    "regime": 25.0,
    "news": 20.0,
    "volume": 10.0,
    "momentum": 3.0,
    "rsi_divergence": 2.0,
}

# ATR multipliers for SL/TP (mirrors risk_manager.py)
SL_ATR_MULT = 2.0
TP_ATR_MULT = 4.0


# =============================================================================
# CONFLUENCE DETECTOR
# =============================================================================

class ConfluenceDetector:
    """
    Weighted scoring engine (0-100) that fuses SMC features, regime,
    news sentiment, volume, and momentum into an actionable signal.

    Usage:
        detector = ConfluenceDetector()
        signal = detector.analyze(smc_features, regime, news, price, atr)
        if signal and signal.tier in (SignalTier.PREMIUM, SignalTier.STANDARD):
            publish(signal)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        min_score: float = 40.0,
        symbol: str = "XAUUSD",
    ):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        self.min_score = min_score
        self.symbol = symbol

        total = sum(self.weights.values())
        if abs(total - 100.0) > 0.01:
            raise ValueError(f"Weights must sum to 100, got {total}")

    # ------------------------------------------------------------------ #
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        smc_features: Dict[str, float],
        regime: Any,
        news: Any,
        price: float,
        atr: float,
        volume: Optional[float] = None,
        volume_ma: Optional[float] = None,
        bar_timestamp: Optional[str] = None,
    ) -> Optional[ConfluenceSignal]:
        """
        Run full confluence analysis.

        Args:
            smc_features: Dict with keys BOS_SIGNAL, FVG_SIGNAL, OB_STRENGTH_NORM,
                          RSI, MACD_Diff (from SmartMoneyEngine.analyze()).
            regime: RegimeAnalysis object (from MarketRegimeAgent.analyze()).
            news: NewsAssessment object (from NewsAnalysisAgent.evaluate_news_impact()).
            price: Current close price.
            atr: Current ATR value.
            volume: Current bar volume (optional).
            volume_ma: 20-bar volume moving average (optional).
            bar_timestamp: ISO timestamp of the bar being analyzed.

        Returns:
            ConfluenceSignal if score >= min_score, else None.
        """
        # News block check
        if news is not None and self._is_news_blocked(news):
            logger.debug("Signal blocked by news: %s", self._get_news_reasoning(news))
            return None

        # Determine direction from BOS
        bos = smc_features.get("BOS_SIGNAL", 0.0)
        if bos == 0.0:
            return None  # No structural break, no signal

        signal_type = SignalType.LONG if bos > 0 else SignalType.SHORT

        # Score each component
        components: List[ComponentScore] = []
        components.append(self._score_bos(smc_features, signal_type, atr))
        components.append(self._score_fvg(smc_features, signal_type, atr))
        components.append(self._score_order_block(smc_features))
        components.append(self._score_regime(regime, signal_type))
        components.append(self._score_news(news, signal_type))
        components.append(self._score_volume(volume, volume_ma))
        components.append(self._score_momentum(smc_features, signal_type))
        components.append(self._score_rsi_divergence(smc_features, signal_type))

        total_score = sum(c.weighted_score for c in components)
        total_score = max(0.0, min(100.0, total_score))

        if total_score < self.min_score:
            return None

        tier = self._classify_tier(total_score)

        # Calculate ATR-based entry/SL/TP
        sl_distance = SL_ATR_MULT * atr
        tp_distance = TP_ATR_MULT * atr

        if signal_type == SignalType.LONG:
            entry = price
            stop_loss = price - sl_distance
            take_profit = price + tp_distance
        else:
            entry = price
            stop_loss = price + sl_distance
            take_profit = price - tp_distance

        rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0.0

        reasoning = [c.reasoning for c in components if c.weighted_score > 0]

        return ConfluenceSignal(
            signal_id=str(uuid.uuid4())[:12],
            symbol=self.symbol,
            signal_type=signal_type,
            confluence_score=total_score,
            tier=tier,
            entry_price=round(entry, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            rr_ratio=round(rr_ratio, 2),
            atr=round(atr, 4),
            components=components,
            reasoning=reasoning,
            bar_timestamp=bar_timestamp,
        )

    # ------------------------------------------------------------------ #
    # COMPONENT SCORERS (each returns 0 to weight_max)
    # ------------------------------------------------------------------ #

    def _score_bos(self, smc: Dict[str, float], direction: SignalType, atr: float = 0.0) -> ComponentScore:
        """Break of Structure: graduated scoring based on CHOCH confirmation.

        Full weight when BOS aligns AND CHOCH confirms (trend reversal).
        Base 60% weight for simple BOS alignment (trend continuation).
        """
        w = self.weights["bos"]
        bos = smc.get("BOS_SIGNAL", 0.0)
        choch = smc.get("CHOCH_SIGNAL", 0.0)

        if direction == SignalType.LONG and bos > 0:
            # CHOCH confirms reversal → full weight; simple BOS → 60%
            quality = 1.0 if choch > 0 else 0.6
            score = quality * w
            label = "BOS + CHOCH" if choch > 0 else "BOS continuation"
            reason = f"Bullish {label} (quality={quality:.0%})"
        elif direction == SignalType.SHORT and bos < 0:
            quality = 1.0 if choch < 0 else 0.6
            score = quality * w
            label = "BOS + CHOCH" if choch < 0 else "BOS continuation"
            reason = f"Bearish {label} (quality={quality:.0%})"
        else:
            score = 0.0
            reason = "BOS does not align with direction"

        return ComponentScore("BOS", bos, score, w, reason)

    def _score_fvg(self, smc: Dict[str, float], direction: SignalType, atr: float = 0.0) -> ComponentScore:
        """Fair Value Gap: graduated scoring based on gap size relative to ATR.

        Larger gaps (institutional displacement) score higher than small gaps.
        Uses FVG_SIZE_NORM if available, falls back to binary.
        """
        w = self.weights["fvg"]
        fvg = smc.get("FVG_SIGNAL", 0.0)

        if fvg == 0.0:
            return ComponentScore("FVG", fvg, 0.0, w, "No FVG present")

        aligned = (direction == SignalType.LONG and fvg > 0) or \
                  (direction == SignalType.SHORT and fvg < 0)

        if not aligned:
            return ComponentScore("FVG", fvg, 0.0, w, "FVG opposes signal direction")

        # Graduate by gap size (FVG_SIZE_NORM = gap / ATR)
        fvg_size_norm = abs(smc.get("FVG_SIZE_NORM", 0.0))
        if fvg_size_norm > 0 and atr > 0:
            # Scale: 0.1 ATR → 30%, 0.5 ATR → 75%, 1.0+ ATR → 100%
            quality = min(1.0, 0.3 + 0.7 * fvg_size_norm)
            score = quality * w
            direction_label = "Bullish" if fvg > 0 else "Bearish"
            reason = f"{direction_label} FVG (size={fvg_size_norm:.2f}×ATR, quality={quality:.0%})"
        else:
            # Fallback: full weight (backward compatible)
            score = w
            direction_label = "Bullish" if fvg > 0 else "Bearish"
            reason = f"{direction_label} FVG confirms {direction.value} bias"

        return ComponentScore("FVG", fvg, score, w, reason)

    def _score_order_block(self, smc: Dict[str, float]) -> ComponentScore:
        """Order Block: score proportional to normalized OB strength."""
        w = self.weights["order_block"]
        ob = smc.get("OB_STRENGTH_NORM", 0.0)

        # OB_STRENGTH_NORM is ATR-normalized; scale 0-1 → 0-weight
        clamped = max(0.0, min(1.0, abs(ob)))
        score = clamped * w
        reason = f"OB strength {ob:.2f} (normalized)" if ob != 0 else "No active order block"

        return ComponentScore("OrderBlock", ob, score, w, reason)

    def _score_regime(self, regime: Any, direction: SignalType) -> ComponentScore:
        """Market Regime: aligned trending regime gets full weight."""
        w = self.weights["regime"]

        if regime is None:
            return ComponentScore("Regime", 0.0, w * 0.5, w, "No regime data — neutral weight")

        regime_type = getattr(regime, "regime", None)
        trend_dir = getattr(regime, "trend_direction", None)
        confidence = getattr(regime, "confidence", 0.5)
        trend_strength = getattr(regime, "trend_strength", 0.0)

        # Get regime type value for comparison
        regime_val = regime_type.value if hasattr(regime_type, "value") else str(regime_type)
        trend_val = trend_dir.value if hasattr(trend_dir, "value") else 0

        # Strong alignment: trending regime + matching direction
        is_uptrend = "uptrend" in regime_val if isinstance(regime_val, str) else False
        is_downtrend = "downtrend" in regime_val if isinstance(regime_val, str) else False
        is_ranging = regime_val == "ranging" if isinstance(regime_val, str) else False

        if direction == SignalType.LONG and is_uptrend:
            alignment = 1.0
            reason = f"Bullish regime ({regime_val}) confirms LONG"
        elif direction == SignalType.SHORT and is_downtrend:
            alignment = 1.0
            reason = f"Bearish regime ({regime_val}) confirms SHORT"
        elif is_ranging:
            alignment = 0.3
            reason = f"Ranging market — reduced conviction"
        elif (direction == SignalType.LONG and is_downtrend) or \
             (direction == SignalType.SHORT and is_uptrend):
            alignment = 0.0
            reason = f"Regime ({regime_val}) opposes {direction.value} — counter-trend"
        else:
            alignment = 0.4
            reason = f"Regime ({regime_val}) is ambiguous"

        # Weight by confidence and trend strength (average, not product)
        quality = (max(confidence, 0.3) + max(trend_strength, 0.3)) / 2.0
        score = w * alignment * min(quality, 1.0)

        return ComponentScore("Regime", confidence, score, w, reason)

    def _score_news(self, news: Any, direction: SignalType) -> ComponentScore:
        """News sentiment: aligned sentiment boosts, opposing reduces."""
        w = self.weights["news"]

        if news is None:
            return ComponentScore("News", 0.0, w * 0.5, w, "No news data — neutral weight")

        sentiment = getattr(news, "sentiment_score", 0.0)
        confidence = getattr(news, "sentiment_confidence", 0.5)

        # Alignment: positive sentiment for LONG, negative for SHORT
        if direction == SignalType.LONG:
            alignment = (sentiment + 1.0) / 2.0  # Map [-1,1] → [0,1]
        else:
            alignment = (1.0 - sentiment) / 2.0  # Flip for SHORT

        score = w * alignment * max(confidence, 0.3)
        label = "bullish" if sentiment > 0 else "bearish" if sentiment < 0 else "neutral"
        reason = f"News sentiment {label} ({sentiment:.2f}, conf={confidence:.2f})"

        return ComponentScore("News", sentiment, score, w, reason)

    def _score_volume(self, volume: Optional[float], volume_ma: Optional[float]) -> ComponentScore:
        """Volume confirmation: above-average volume boosts score."""
        w = self.weights["volume"]

        if volume is None or volume_ma is None or volume_ma <= 0:
            return ComponentScore("Volume", 0.0, w * 0.5, w, "No volume data — neutral weight")

        ratio = volume / volume_ma
        # 1.0x = baseline, 2.0x+ = full score, 0.5x = low score
        normalized = max(0.0, min(1.0, (ratio - 0.5) / 1.5))
        score = w * normalized

        if ratio >= 1.5:
            reason = f"Strong volume confirmation ({ratio:.1f}x average)"
        elif ratio >= 1.0:
            reason = f"Adequate volume ({ratio:.1f}x average)"
        else:
            reason = f"Below-average volume ({ratio:.1f}x average)"

        return ComponentScore("Volume", ratio, score, w, reason)

    def _score_momentum(self, smc: Dict[str, float], direction: SignalType) -> ComponentScore:
        """RSI + MACD momentum alignment."""
        w = self.weights["momentum"]
        rsi = smc.get("RSI", 50.0)
        macd_diff = smc.get("MACD_Diff", 0.0)

        # RSI component: 50-70 for LONG, 30-50 for SHORT → aligned
        if direction == SignalType.LONG:
            rsi_score = max(0.0, min(1.0, (rsi - 30) / 40))  # 30→0, 70→1
            macd_aligned = macd_diff > 0
        else:
            rsi_score = max(0.0, min(1.0, (70 - rsi) / 40))  # 70→0, 30→1
            macd_aligned = macd_diff < 0

        macd_score = 1.0 if macd_aligned else 0.3
        combined = (rsi_score * 0.5 + macd_score * 0.5)
        score = w * combined

        reason = f"RSI={rsi:.1f}, MACD_Diff={'+'if macd_diff>0 else ''}{macd_diff:.4f}"

        return ComponentScore("Momentum", rsi, score, w, reason)

    def _score_rsi_divergence(self, smc: Dict[str, float], direction: SignalType) -> ComponentScore:
        """RSI Divergence: confirms CHOCH reversal signals.

        Bullish divergence (CHOCH_DIVERGENCE=1) supports LONG.
        Bearish divergence (CHOCH_DIVERGENCE=-1) supports SHORT.
        """
        w = self.weights.get("rsi_divergence", 2.0)
        divergence = smc.get("CHOCH_DIVERGENCE", 0)

        if divergence == 0:
            return ComponentScore("RSI_Divergence", 0.0, 0.0, w, "No RSI divergence detected")

        if (direction == SignalType.LONG and divergence > 0):
            score = w
            reason = "Bullish RSI divergence confirms reversal"
        elif (direction == SignalType.SHORT and divergence < 0):
            score = w
            reason = "Bearish RSI divergence confirms reversal"
        else:
            score = 0.0
            reason = "RSI divergence opposes signal direction"

        return ComponentScore("RSI_Divergence", float(divergence), score, w, reason)

    # ------------------------------------------------------------------ #
    # HELPERS
    # ------------------------------------------------------------------ #

    @staticmethod
    def _is_news_blocked(news: Any) -> bool:
        """Check if news assessment blocks trading."""
        decision = getattr(news, "decision", None)
        if decision is None:
            return False
        decision_val = decision.value if hasattr(decision, "value") else str(decision)
        return decision_val == "BLOCK"

    @staticmethod
    def _get_news_reasoning(news: Any) -> str:
        reasoning = getattr(news, "reasoning", [])
        return "; ".join(reasoning) if reasoning else "News block active"

    @staticmethod
    def _classify_tier(score: float) -> SignalTier:
        if score >= 80:
            return SignalTier.PREMIUM
        elif score >= 60:
            return SignalTier.STANDARD
        elif score >= 40:
            return SignalTier.WEAK
        else:
            return SignalTier.INVALID
