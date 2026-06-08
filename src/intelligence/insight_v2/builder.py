"""InsightV2Builder — orchestrates the full read-out per mockup."""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.intelligence.insight_v2.contract import (
    AlternativeScenario,
    ComplianceMeta,
    EventReadout,
    HistoricalStats,
    InsightSignalV2,
    RegimeReadout,
    StructureReadout,
    VolatilityReadout,
)
from src.intelligence.insight_v2.liquidity_zones import detect_liquidity_zones
from src.intelligence.insight_v2.scenarios import build_scenarios

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _ts_iso(ts) -> str:
    if ts is None:
        return ""
    if hasattr(ts, "isoformat"):
        return ts.isoformat()
    return str(ts)


class InsightV2Builder:
    """Build an :class:`InsightSignalV2` from pipeline outputs.

    All inputs are optional — missing data → ``None`` fields → narrative
    layer adapts gracefully. The output is the full unified contract that
    matches the client mockup (mockups/v2/client_view_full.html).
    """

    def __init__(
        self,
        asset: str,
        timeframe: str,
        scoring_engine: Optional[object] = None,    # LGBMScoringEngine
        conformal: Optional[object] = None,         # MondrianConformal
        edge_claim: bool = False,
        validity_bars: int = 12,                    # how many bars the insight lives
        bar_minutes: int = 15,
        disclaimer_lang: str = "fr",
        historical_stats_loader: Optional[object] = None,  # callable() -> HistoricalStats
        narrative_generator: Optional[object] = None,      # InsightV2NarrativeGenerator
    ):
        self.asset = asset
        self.timeframe = timeframe
        self.scoring_engine = scoring_engine
        self.conformal = conformal
        self.edge_claim = edge_claim
        self.validity_bars = int(validity_bars)
        self.bar_minutes = int(bar_minutes)
        self.disclaimer_lang = disclaimer_lang
        self.historical_stats_loader = historical_stats_loader
        self.narrative_generator = narrative_generator

    # ------------------------------------------------------------------ #
    # Build sections
    # ------------------------------------------------------------------ #

    def _build_structure(self, enriched_df: pd.DataFrame) -> StructureReadout:
        if enriched_df.empty:
            return StructureReadout(direction="neutral")

        row = enriched_df.iloc[-1]
        bos_signal = int(row.get("BOS_SIGNAL", 0))
        bos_event = int(row.get("BOS_EVENT", 0))
        direction = "neutral"
        if bos_signal > 0 or bos_event > 0:
            direction = "bullish"
        elif bos_signal < 0 or bos_event < 0:
            direction = "bearish"

        bos_level = row.get("BOS_BREAK_LEVEL")
        bos_level = float(bos_level) if bos_level is not None and not np.isnan(bos_level) else None

        # FVG zone
        fvg_dir = int(row.get("FVG_DIR", 0))
        fvg_zone = None
        fvg_size_atr = None
        if fvg_dir != 0:
            fvg_size = float(row.get("FVG_SIZE", 0.0))
            fvg_size_norm = float(row.get("FVG_SIZE_NORM", 0.0))
            current_price = float(row.get("close", row.get("Close", 0.0)))
            if fvg_dir > 0:
                fvg_zone = (current_price - fvg_size, current_price)
            else:
                fvg_zone = (current_price, current_price + fvg_size)
            fvg_size_atr = fvg_size_norm

        # OB zone
        if direction == "bullish":
            ob_high = row.get("BULLISH_OB_HIGH")
            ob_low = row.get("BULLISH_OB_LOW")
        else:
            ob_high = row.get("BEARISH_OB_HIGH")
            ob_low = row.get("BEARISH_OB_LOW")
        ob_zone = None
        if ob_high is not None and ob_low is not None:
            try:
                if not np.isnan(ob_high) and not np.isnan(ob_low):
                    ob_zone = (float(ob_low), float(ob_high))
            except (TypeError, ValueError):
                ob_zone = None
        ob_strength = row.get("OB_STRENGTH_NORM")
        ob_strength = float(ob_strength) if ob_strength is not None and not pd.isna(ob_strength) else None

        # Retest state
        retest_state_int = int(row.get("BOS_RETEST_STATE", 0))
        retest_armed = int(row.get("BOS_RETEST_ARMED", 0))
        if retest_state_int == 2 or retest_armed == 0 and retest_state_int > 0:
            retest_state = "validated"
        elif retest_armed == 1:
            retest_state = "armed"
        else:
            retest_state = "none"

        # Structural invalidation = FVG floor (LONG) or ceiling (SHORT)
        invalidation = None
        if fvg_zone is not None:
            invalidation = fvg_zone[0] if direction == "bullish" else fvg_zone[1]
        elif bos_level is not None:
            invalidation = bos_level

        # Liquidity zones via swing extremes
        liq = detect_liquidity_zones(enriched_df, lookback_bars=200)

        return StructureReadout(
            direction=direction,
            bos_level=bos_level,
            bos_age_bars=None,  # not tracked here, scanner-level
            fvg_zone=fvg_zone,
            fvg_size_atr=fvg_size_atr,
            ob_zone=ob_zone,
            ob_strength=ob_strength,
            retest_state=retest_state,
            structural_invalidation=invalidation,
            liquidity_zone_upper=liq.get("upper"),
            liquidity_zone_lower=liq.get("lower"),
        )

    def _build_regime(self, regime: Any) -> RegimeReadout:
        if regime is None:
            return RegimeReadout()
        return RegimeReadout(
            hmm_label=str(getattr(regime, "hmm_label", "unknown")),
            hmm_posterior=getattr(regime, "hmm_posterior", None),
            bocpd_changepoint_prob=getattr(regime, "bocpd_changepoint_prob", None),
            jump_ratio=getattr(regime, "jump_ratio", None),
            regime_gate_decision=str(getattr(regime, "decision", "TRADE")),
        )

    def _build_volatility(self, vol_forecast: Any, naive_atr: Optional[float]) -> VolatilityReadout:
        if vol_forecast is None:
            return VolatilityReadout()
        forecast = getattr(vol_forecast, "forecast_atr", None)
        ci_lower = getattr(vol_forecast, "confidence_lower", None)
        ci_upper = getattr(vol_forecast, "confidence_upper", None)
        regime = getattr(vol_forecast, "regime", None) or "normal"
        ci = (float(ci_lower), float(ci_upper)) if ci_lower is not None and ci_upper is not None else None
        delta_pct = None
        if forecast is not None and naive_atr not in (None, 0):
            delta_pct = (float(forecast) / float(naive_atr) - 1.0) * 100.0
        return VolatilityReadout(
            regime=str(regime),
            forecast_atr=float(forecast) if forecast is not None else None,
            naive_atr=float(naive_atr) if naive_atr is not None else None,
            forecast_vs_naive_pct=delta_pct,
            confidence_interval=ci,
        )

    def _build_event(self, news: Any, news_provider: Optional[object], bar_ts) -> EventReadout:
        # If news provider supplied, query for next event and active blackout
        blackout = False
        next_event_name = None
        next_event_in_min = None
        sentiment = None
        sentiment_conf = None
        session = self._session_for(bar_ts)

        if news is not None:
            decision = getattr(news, "decision", None)
            blackout = str(decision).upper() == "BLOCK"
            sentiment = getattr(news, "sentiment_score", None)
            sentiment_conf = getattr(news, "sentiment_confidence", None)

        if news_provider is not None and hasattr(news_provider, "events") and bar_ts is not None:
            try:
                ts = pd.Timestamp(bar_ts)
                future = [e for e in news_provider.events if e.ts > ts]
                if future:
                    nxt = future[0]
                    next_event_name = getattr(nxt, "name", None)
                    next_event_in_min = int((nxt.ts - ts).total_seconds() // 60)
            except Exception as exc:
                logger.debug("next_event lookup failed: %s", exc)

        return EventReadout(
            news_blackout_active=blackout,
            next_event=next_event_name,
            next_event_in_minutes=next_event_in_min,
            sentiment_score=sentiment,
            sentiment_confidence=sentiment_conf,
            session=session,
        )

    @staticmethod
    def _session_for(ts) -> str:
        if ts is None:
            return "unknown"
        try:
            h = pd.Timestamp(ts).hour
        except Exception:
            return "unknown"
        if 0 <= h < 7:
            return "asia"
        if 7 <= h < 13:
            return "london"
        if 13 <= h < 21:
            return "new_york"
        return "asia"

    def _build_compliance(self) -> ComplianceMeta:
        return ComplianceMeta(
            edge_claim=self.edge_claim,
            is_paper_demo=not self.edge_claim,
            disclaimer_lang=self.disclaimer_lang,
        )

    def _build_history(self) -> HistoricalStats:
        if self.historical_stats_loader is None:
            return HistoricalStats()
        try:
            return self.historical_stats_loader()
        except Exception as exc:
            logger.warning("historical_stats_loader failed: %s", exc)
            return HistoricalStats()

    def _score(self, features: Any, regime_label: Optional[str]) -> dict:
        if self.scoring_engine is None:
            # Fallback : neutral
            return {
                "conviction_0_100": 50.0,
                "conviction_label": "moderate",
                "conviction_interval": {"lower": 30.0, "upper": 70.0, "alpha": 0.10},
            }
        readout = self.scoring_engine.score(
            features=features,
            regime_label=regime_label,
            conformal=self.conformal,
        )
        return {
            "conviction_0_100": readout.score_0_100,
            "conviction_label": readout.label,
            "conviction_interval": {
                "lower": readout.interval_lower,
                "upper": readout.interval_upper,
                "alpha": readout.alpha,
            },
        }

    # ------------------------------------------------------------------ #
    # Public build
    # ------------------------------------------------------------------ #

    def build(
        self,
        bar_timestamp,
        enriched_df: pd.DataFrame,
        features_for_scoring: Any,
        regime: Any = None,
        vol_forecast: Any = None,
        news: Any = None,
        news_provider: Any = None,
        naive_atr: Optional[float] = None,
        narrative_short: Optional[str] = None,
        narrative_long: Optional[str] = None,
        narrative_lang: str = "fr",
        sources_cited: Optional[list[dict]] = None,
    ) -> InsightSignalV2:
        ts_iso = _ts_iso(bar_timestamp)
        # Deterministic ID per audit fix Sprint 1
        seed = f"{self.asset}|{self.timeframe}|{ts_iso}"
        insight_id = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:24]

        try:
            exp_ts = pd.Timestamp(bar_timestamp) + timedelta(minutes=self.bar_minutes * self.validity_bars)
            exp_iso = exp_ts.isoformat()
        except Exception:
            exp_iso = ""

        structure = self._build_structure(enriched_df)
        regime_ro = self._build_regime(regime)
        vol_ro = self._build_volatility(vol_forecast, naive_atr)
        event_ro = self._build_event(news, news_provider, bar_timestamp)
        history = self._build_history()
        compliance = self._build_compliance()
        scoring = self._score(features_for_scoring, regime_label=regime_ro.hmm_label)
        scenarios = build_scenarios(structure, expires_iso=exp_iso)

        insight = InsightSignalV2(
            insight_id=insight_id,
            asset=self.asset,
            timeframe=self.timeframe,
            generated_at=_now_iso(),
            expires_at=exp_iso,
            structure_bias=structure.direction,
            conviction_0_100=scoring["conviction_0_100"],
            conviction_label=scoring["conviction_label"],
            conviction_interval=scoring["conviction_interval"],
            structure_readout=structure,
            regime_readout=regime_ro,
            volatility_readout=vol_ro,
            event_readout=event_ro,
            historical_stats=history,
            scenarios=scenarios,
            narrative_short=narrative_short,
            narrative_long=narrative_long,
            narrative_lang=narrative_lang,
            sources_cited=sources_cited or [],
            compliance=compliance,
        )

        # Auto-generate narrative if generator wired and no explicit text passed
        if self.narrative_generator is not None and (narrative_short is None or narrative_long is None):
            try:
                out = self.narrative_generator.generate(insight, lang=narrative_lang)
                if narrative_short is None:
                    insight.narrative_short = out.short
                if narrative_long is None:
                    insight.narrative_long = out.long
                insight.narrative_lang = out.lang
            except Exception as exc:
                logger.warning("narrative_generator failed: %s", exc)

        return insight


__all__ = ["InsightV2Builder"]
