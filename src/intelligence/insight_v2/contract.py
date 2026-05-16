"""InsightSignalV2 contract — aligned with mockups/v2/client_view_full.html."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional


@dataclass
class StructureReadout:
    """SMC structure descriptive read-out (no entry/stop/target)."""

    direction: str  # "bullish" | "bearish" | "neutral"
    bos_level: Optional[float] = None
    bos_age_bars: Optional[int] = None
    fvg_zone: Optional[tuple[float, float]] = None
    fvg_size_atr: Optional[float] = None
    ob_zone: Optional[tuple[float, float]] = None
    ob_strength: Optional[float] = None
    retest_state: str = "none"  # "armed" | "validated" | "none"
    structural_invalidation: Optional[float] = None
    liquidity_zone_upper: Optional[tuple[float, float]] = None
    liquidity_zone_lower: Optional[tuple[float, float]] = None


@dataclass
class RegimeReadout:
    hmm_label: str = "unknown"
    hmm_posterior: Optional[float] = None
    bocpd_changepoint_prob: Optional[float] = None
    jump_ratio: Optional[float] = None
    regime_gate_decision: str = "TRADE"  # TRADE / REDUCE / BLOCK


@dataclass
class VolatilityReadout:
    regime: str = "normal"
    forecast_atr: Optional[float] = None
    naive_atr: Optional[float] = None
    forecast_vs_naive_pct: Optional[float] = None
    confidence_interval: Optional[tuple[float, float]] = None
    alpha: float = 0.10


@dataclass
class EventReadout:
    news_blackout_active: bool = False
    next_event: Optional[str] = None
    next_event_in_minutes: Optional[int] = None
    sentiment_score: Optional[float] = None
    sentiment_confidence: Optional[float] = None
    session: str = "unknown"  # "asia" | "london" | "new_york"


@dataclass
class HistoricalStats:
    similar_setups_n: int = 0
    hit_rate_observed: Optional[float] = None
    profit_factor: Optional[float] = None
    profit_factor_ci95: Optional[tuple[float, float]] = None
    empirical_coverage: Optional[float] = None
    sample_window: Optional[str] = None  # e.g. "2019-2025"
    cost_assumptions: Optional[dict] = None  # spread/slippage/commission used


@dataclass
class ComplianceMeta:
    edge_claim: bool = False
    is_paper_demo: bool = True
    jurisdiction_blocked: tuple[str, ...] = ("US", "QC", "UK", "OFAC")
    disclaimer_lang: str = "fr"


@dataclass
class AlternativeScenario:
    name: str  # "principal" | "alternative_1" | "alternative_2"
    label: str  # "bullish_continuation" | "bearish_invalidation" | ...
    condition: str  # observable condition text
    reading_evolution: str  # how the read-out would evolve


@dataclass
class InsightSignalV2:
    """Full insight contract per mockup."""

    insight_id: str
    asset: str
    timeframe: str
    generated_at: str  # ISO UTC
    expires_at: str    # ISO UTC

    structure_bias: str  # "bullish" | "bearish" | "neutral"

    # LightGBM-backed (NOT additive ConfluenceDetector)
    conviction_0_100: float
    conviction_label: str
    conviction_interval: dict  # {"lower", "upper", "alpha"}

    structure_readout: StructureReadout = field(default_factory=lambda: StructureReadout("neutral"))
    regime_readout: RegimeReadout = field(default_factory=RegimeReadout)
    volatility_readout: VolatilityReadout = field(default_factory=VolatilityReadout)
    event_readout: EventReadout = field(default_factory=EventReadout)
    historical_stats: HistoricalStats = field(default_factory=HistoricalStats)
    scenarios: list[AlternativeScenario] = field(default_factory=list)

    narrative_short: Optional[str] = None  # ≤ 400 chars, Telegram
    narrative_long: Optional[str] = None   # ≤ 2000 chars, webapp + B2B
    narrative_lang: str = "fr"
    sources_cited: list[dict] = field(default_factory=list)

    compliance: ComplianceMeta = field(default_factory=ComplianceMeta)

    def to_dict(self) -> dict:
        """Serialise to dict matching the mockup JSON contract."""
        def _zone(z: Optional[tuple[float, float]]) -> Optional[list[float]]:
            return [float(z[0]), float(z[1])] if z is not None else None

        return {
            "insight_id": self.insight_id,
            "asset": self.asset,
            "timeframe": self.timeframe,
            "generated_at": self.generated_at,
            "expires_at": self.expires_at,
            "structure_bias": self.structure_bias,
            "conviction_0_100": round(float(self.conviction_0_100), 2),
            "conviction_label": self.conviction_label,
            "conviction_interval": self.conviction_interval,
            "structure_readout": {
                "bos_level": self.structure_readout.bos_level,
                "bos_age_bars": self.structure_readout.bos_age_bars,
                "fvg_zone": _zone(self.structure_readout.fvg_zone),
                "fvg_size_atr": self.structure_readout.fvg_size_atr,
                "ob_zone": _zone(self.structure_readout.ob_zone),
                "ob_strength": self.structure_readout.ob_strength,
                "retest_state": self.structure_readout.retest_state,
                "structural_invalidation": self.structure_readout.structural_invalidation,
                "liquidity_zone_upper": _zone(self.structure_readout.liquidity_zone_upper),
                "liquidity_zone_lower": _zone(self.structure_readout.liquidity_zone_lower),
            },
            "regime_readout": {
                "hmm_label": self.regime_readout.hmm_label,
                "hmm_posterior": self.regime_readout.hmm_posterior,
                "bocpd_changepoint_prob": self.regime_readout.bocpd_changepoint_prob,
                "jump_ratio": self.regime_readout.jump_ratio,
                "regime_gate_decision": self.regime_readout.regime_gate_decision,
            },
            "volatility_readout": {
                "regime": self.volatility_readout.regime,
                "forecast_atr": self.volatility_readout.forecast_atr,
                "naive_atr": self.volatility_readout.naive_atr,
                "forecast_vs_naive_pct": self.volatility_readout.forecast_vs_naive_pct,
                "confidence_interval": (list(self.volatility_readout.confidence_interval)
                                         if self.volatility_readout.confidence_interval else None),
                "alpha": self.volatility_readout.alpha,
            },
            "event_readout": {
                "news_blackout_active": self.event_readout.news_blackout_active,
                "next_event": self.event_readout.next_event,
                "next_event_in_minutes": self.event_readout.next_event_in_minutes,
                "sentiment_score": self.event_readout.sentiment_score,
                "sentiment_confidence": self.event_readout.sentiment_confidence,
                "session": self.event_readout.session,
            },
            "historical_stats": {
                "similar_setups_n": self.historical_stats.similar_setups_n,
                "hit_rate_observed": self.historical_stats.hit_rate_observed,
                "profit_factor": self.historical_stats.profit_factor,
                "profit_factor_ci95": (list(self.historical_stats.profit_factor_ci95)
                                        if self.historical_stats.profit_factor_ci95 else None),
                "empirical_coverage": self.historical_stats.empirical_coverage,
                "sample_window": self.historical_stats.sample_window,
                "cost_assumptions": self.historical_stats.cost_assumptions,
            },
            "scenarios": [
                {"name": s.name, "label": s.label,
                 "condition": s.condition, "reading_evolution": s.reading_evolution}
                for s in self.scenarios
            ],
            "narrative_short": self.narrative_short,
            "narrative_long": self.narrative_long,
            "narrative_lang": self.narrative_lang,
            "sources_cited": list(self.sources_cited),
            "compliance": {
                "edge_claim": self.compliance.edge_claim,
                "is_paper_demo": self.compliance.is_paper_demo,
                "jurisdiction_blocked": list(self.compliance.jurisdiction_blocked),
                "disclaimer_lang": self.compliance.disclaimer_lang,
            },
        }


__all__ = [
    "InsightSignalV2", "StructureReadout", "RegimeReadout", "VolatilityReadout",
    "EventReadout", "HistoricalStats", "ComplianceMeta", "AlternativeScenario",
]
