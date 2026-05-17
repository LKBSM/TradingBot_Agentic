"""Mappers from internal pipeline dataclasses to InsightSignalV2 2.1.0 readouts.

The pipeline produces strongly-typed internal objects (``ConfluenceSignal``,
``VolatilityForecast``, ``RegimeGateOutput``, ``NewsAssessment``) which carry
*more* information than is exposed to clients. These mappers project the
internal objects onto the descriptive readout sub-models of
:mod:`src.api.insight_signal_v2` (v2.1.0).

Design rules
------------
1. **Pure functions.** No side effects, no I/O. Every mapper takes the source
   object + optional context and returns a Pydantic model (or ``None``).
2. **Defensive.** Each mapper accepts ``None`` and returns ``None`` — partial
   readouts are normal during fallback / cold-start / replay.
3. **Descriptive only.** Mappers NEVER emit trade orders (entry/stop/TP). They
   describe market state. The product stance is "indicator, not signal
   service".
4. **Backward compatible.** Source dataclasses can grow new fields without
   breaking mappers (use ``getattr`` defensively).

These mappers are called from :class:`SentinelScanner` to assemble the
v2.1.0 ``InsightSignalV2`` published to each surface.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from src.api.insight_signal_v2 import (
    ComponentBreakdown,
    EventReadout,
    HistoricalStats,
    RegimeReadout,
    StructureReadout,
    UncertaintyContext,
    VolatilityReadout,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StructureReadout — from ConfluenceSignal + SMC features dict
# ---------------------------------------------------------------------------


def map_structure_readout(
    confluence_signal: Any,
    smc_features: Optional[dict] = None,
    bar_index: Optional[int] = None,
) -> Optional[StructureReadout]:
    """Project SMC features + a ConfluenceSignal onto a StructureReadout.

    Parameters
    ----------
    confluence_signal : ConfluenceSignal-like
        Object with attributes ``atr``, ``signal_type`` (LONG/SHORT). Only
        used to determine OB direction polarity.
    smc_features : dict, optional
        The dict returned by ``SmartMoneyEngine.analyze().iloc[-1]`` for the
        current bar. Keys consumed: BOS_SIGNAL, BOS_EVENT, BOS_BREAK_LEVEL,
        CHOCH_SIGNAL, FVG_SIGNAL, FVG_SIZE_NORM, BULLISH_OB_HIGH/LOW,
        BEARISH_OB_HIGH/LOW, OB_STRENGTH_NORM, BOS_RETEST_STATE,
        BOS_RETEST_ARMED.
    bar_index : int, optional
        Current bar index — used to compute ``bos_event_age_bars``. If
        unavailable, age is omitted.

    Returns
    -------
    StructureReadout or None — None when both inputs are missing.
    """
    if confluence_signal is None and not smc_features:
        return None

    smc = smc_features or {}

    # --- BOS direction and level (descriptive) ---
    bos_signal = smc.get("BOS_SIGNAL", 0.0) or 0.0
    bos_event = smc.get("BOS_EVENT", 0.0) or 0.0
    bos_level = smc.get("BOS_BREAK_LEVEL")
    if bos_level is not None:
        try:
            bos_level = float(bos_level)
            if not _is_finite(bos_level):
                bos_level = None
        except (TypeError, ValueError):
            bos_level = None

    # Age estimation: if bar_index provided and we can find when BOS_EVENT
    # was last non-zero, compute distance. Otherwise leave None.
    bos_event_age_bars: Optional[int] = None
    if bos_event and bos_event != 0.0:
        bos_event_age_bars = 0  # fresh on this bar

    # --- CHOCH ---
    choch_signal = smc.get("CHOCH_SIGNAL", 0.0) or 0.0
    choch_present = bool(choch_signal)

    # --- FVG ---
    fvg_zone = None
    fvg_size_atr = None
    fvg_signal = smc.get("FVG_SIGNAL", 0.0) or 0.0
    if fvg_signal:
        fvg_size_norm = smc.get("FVG_SIZE_NORM", 0.0) or 0.0
        if _is_finite(fvg_size_norm) and fvg_size_norm > 0:
            fvg_size_atr = float(abs(fvg_size_norm))
        # The SMC engine doesn't expose exact gap bounds in the per-bar dict —
        # only existence + size. Leaving fvg_zone=None unless caller supplies
        # explicit bounds via smc_features["FVG_LOW"] / ["FVG_HIGH"] (forward
        # compat for a future engine extension).
        low = smc.get("FVG_LOW")
        high = smc.get("FVG_HIGH")
        if low is not None and high is not None and _is_finite(low) and _is_finite(high):
            lo, hi = sorted((float(low), float(high)))
            fvg_zone = [lo, hi]

    # --- Order Block ---
    ob_zone = None
    ob_strength = None
    ob_strength_norm = smc.get("OB_STRENGTH_NORM", 0.0) or 0.0
    if _is_finite(ob_strength_norm) and ob_strength_norm != 0:
        ob_strength = float(min(1.0, max(0.0, abs(ob_strength_norm))))
        # Polarity for which OB bounds to read
        sig_type = getattr(confluence_signal, "signal_type", None)
        sig_val = getattr(sig_type, "value", sig_type)
        if sig_val == "LONG" or (sig_val is None and bos_signal > 0):
            low = smc.get("BULLISH_OB_LOW")
            high = smc.get("BULLISH_OB_HIGH")
        else:
            low = smc.get("BEARISH_OB_LOW")
            high = smc.get("BEARISH_OB_HIGH")
        if low is not None and high is not None and _is_finite(low) and _is_finite(high):
            lo, hi = sorted((float(low), float(high)))
            ob_zone = [lo, hi]

    # --- Retest state (descriptive label) ---
    retest_state = None
    retest_state_int = smc.get("BOS_RETEST_STATE")
    retest_armed = smc.get("BOS_RETEST_ARMED", 0.0) or 0.0
    if retest_armed:
        retest_state = "armed"
    elif retest_state_int is not None:
        try:
            v = int(retest_state_int)
            retest_state = {0: "idle", 1: "awaiting", 2: "armed", -1: "awaiting", -2: "armed"}.get(v, "idle")
        except (TypeError, ValueError):
            retest_state = None

    # --- Structural invalidation: prefer the FVG floor / OB low for bullish,
    # ceiling for bearish — these are the SMC convention for "if price breaks
    # this, the read is invalidated". Distinct from ATR-mechanical stop.
    structural_invalidation = None
    if fvg_zone is not None:
        if bos_signal > 0:
            structural_invalidation = fvg_zone[0]
        elif bos_signal < 0:
            structural_invalidation = fvg_zone[1]
    elif ob_zone is not None:
        if bos_signal > 0:
            structural_invalidation = ob_zone[0]
        elif bos_signal < 0:
            structural_invalidation = ob_zone[1]

    try:
        return StructureReadout(
            bos_level=bos_level,
            bos_event_age_bars=bos_event_age_bars,
            choch_present=choch_present,
            fvg_zone=fvg_zone,
            fvg_size_atr=fvg_size_atr,
            ob_zone=ob_zone,
            ob_strength=ob_strength,
            retest_state=retest_state,
            structural_invalidation=structural_invalidation,
            liquidity_zone_upper=None,  # SMC engine doesn't expose yet
            liquidity_zone_lower=None,
        )
    except ValueError as exc:
        logger.warning("StructureReadout construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# RegimeReadout — from RegimeGateOutput + MarketRegimeAgent analysis
# ---------------------------------------------------------------------------


_HMM_LABEL_MAP = {
    "low_vol_trending": "trend_bullish",  # caller distinguishes long/short via direction
    "low_vol_ranging": "range_low_vol",
    "high_vol_stress": "stress",
    "low": "range_low_vol",
    "normal": "range_normal_vol",
    "high": "stress",
}


def map_regime_readout(
    regime_analysis: Optional[Any] = None,
    regime_gate_output: Optional[Any] = None,
    direction_hint: Optional[str] = None,
) -> Optional[RegimeReadout]:
    """Combine MarketRegimeAgent output + RegimeGate output into a readout.

    Parameters
    ----------
    regime_analysis : RegimeAnalysis-like, optional
        Object with attributes ``regime`` (enum value or str),
        ``trend_direction`` (LONG/SHORT/0), ``confidence`` (0-1).
    regime_gate_output : RegimeGateOutput-like, optional
        Object with attributes ``decision`` (RegimeDecision), ``cp_prob``,
        ``jump_ratio``, ``expected_run_length``.
    direction_hint : str, optional
        "LONG" / "SHORT" — used to disambiguate trend label when HMM only
        gives "trending" without a sign.

    Returns
    -------
    RegimeReadout or None
    """
    if regime_analysis is None and regime_gate_output is None:
        return None

    hmm_label = None
    hmm_posterior = None
    if regime_analysis is not None:
        raw_regime = getattr(regime_analysis, "regime", None)
        raw_regime_val = getattr(raw_regime, "value", raw_regime)
        if isinstance(raw_regime_val, str):
            mapped = _HMM_LABEL_MAP.get(raw_regime_val, raw_regime_val)
            # Resolve trend direction when generic
            if mapped == "trend_bullish" and direction_hint == "SHORT":
                mapped = "trend_bearish"
            elif raw_regime_val == "uptrend":
                mapped = "trend_bullish"
            elif raw_regime_val == "downtrend":
                mapped = "trend_bearish"
            hmm_label = mapped
        conf = getattr(regime_analysis, "confidence", None)
        if conf is not None and _is_finite(conf):
            hmm_posterior = float(min(1.0, max(0.0, conf)))

    cp_prob = None
    jump_ratio = None
    expected_run_length = None
    gate_decision = None
    if regime_gate_output is not None:
        cp_prob_raw = getattr(regime_gate_output, "cp_prob", None)
        if cp_prob_raw is not None and _is_finite(cp_prob_raw):
            cp_prob = float(min(1.0, max(0.0, cp_prob_raw)))
        jr_raw = getattr(regime_gate_output, "jump_ratio", None)
        if jr_raw is not None and _is_finite(jr_raw):
            jump_ratio = float(min(1.0, max(0.0, jr_raw)))
        erl_raw = getattr(regime_gate_output, "expected_run_length", None)
        if erl_raw is not None and _is_finite(erl_raw):
            expected_run_length = float(max(0.0, erl_raw))
        decision = getattr(regime_gate_output, "decision", None)
        decision_val = getattr(decision, "value", decision)
        if isinstance(decision_val, str) and decision_val.upper() in ("TRADE", "REDUCE", "BLOCK"):
            gate_decision = decision_val.upper()

    try:
        return RegimeReadout(
            hmm_label=hmm_label,
            hmm_posterior=hmm_posterior,
            bocpd_changepoint_prob=cp_prob,
            expected_run_length=expected_run_length,
            jump_ratio=jump_ratio,
            regime_gate_decision=gate_decision,
        )
    except ValueError as exc:
        logger.warning("RegimeReadout construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# VolatilityReadout — from VolatilityForecast
# ---------------------------------------------------------------------------


def map_volatility_readout(forecast: Optional[Any]) -> Optional[VolatilityReadout]:
    """Project a VolatilityForecast onto a VolatilityReadout."""
    if forecast is None:
        return None

    forecast_atr = getattr(forecast, "forecast_atr", None)
    naive_atr = getattr(forecast, "naive_atr", None)
    regime = getattr(forecast, "regime_state", None)
    is_fallback = bool(getattr(forecast, "is_fallback", False))

    forecast_atr_f = float(forecast_atr) if forecast_atr is not None and _is_finite(forecast_atr) else None
    naive_atr_f = float(naive_atr) if naive_atr is not None and _is_finite(naive_atr) else None

    pct = None
    if forecast_atr_f is not None and naive_atr_f is not None and naive_atr_f > 0:
        pct = round(100.0 * (forecast_atr_f - naive_atr_f) / naive_atr_f, 2)

    ci = None
    lo = getattr(forecast, "confidence_lower", None)
    hi = getattr(forecast, "confidence_upper", None)
    if lo is not None and hi is not None and _is_finite(lo) and _is_finite(hi):
        lo_f, hi_f = float(lo), float(hi)
        if lo_f <= hi_f and lo_f >= 0:
            ci = [lo_f, hi_f]

    regime_label = None
    if isinstance(regime, str) and regime in ("low", "normal", "high"):
        regime_label = regime

    try:
        return VolatilityReadout(
            regime=regime_label,
            forecast_atr_pips=forecast_atr_f,
            naive_atr_pips=naive_atr_f,
            forecast_vs_naive_pct=pct,
            confidence_interval_pips=ci,
            is_fallback=is_fallback,
        )
    except ValueError as exc:
        logger.warning("VolatilityReadout construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# EventReadout — from NewsAssessment + optional session label
# ---------------------------------------------------------------------------


def map_event_readout(
    news_assessment: Optional[Any] = None,
    session: Optional[str] = None,
) -> Optional[EventReadout]:
    """Project NewsAssessment + session label onto EventReadout."""
    if news_assessment is None and not session:
        return None

    blackout = False
    next_event_label = None
    next_event_in_minutes = None
    sentiment_score = None
    sentiment_confidence = None

    if news_assessment is not None:
        decision = getattr(news_assessment, "decision", None)
        decision_val = getattr(decision, "value", decision)
        if isinstance(decision_val, str) and decision_val.upper() == "BLOCK":
            blackout = True

        # Try to extract next event from blocking_events
        blocking = getattr(news_assessment, "blocking_events", None) or []
        if blocking:
            first = blocking[0]
            next_event_label = getattr(first, "event_name", None) or getattr(first, "title", None)

        hours = getattr(news_assessment, "hours_to_next_high_impact", None)
        if hours is not None and _is_finite(hours):
            next_event_in_minutes = int(round(float(hours) * 60))

        sent = getattr(news_assessment, "sentiment_score", None)
        if sent is not None and _is_finite(sent):
            sentiment_score = float(max(-1.0, min(1.0, sent)))
        conf = getattr(news_assessment, "sentiment_confidence", None)
        if conf is not None and _is_finite(conf):
            sentiment_confidence = float(max(0.0, min(1.0, conf)))

    valid_sessions = ("asian", "london", "ny_overlap", "ny_afternoon", "after_hours", "new_york", "us")
    session_clean = session if (isinstance(session, str) and session in valid_sessions) else None

    try:
        return EventReadout(
            news_blackout_active=blackout,
            next_event_label=next_event_label,
            next_event_in_minutes=next_event_in_minutes,
            sentiment_score=sentiment_score,
            sentiment_confidence=sentiment_confidence,
            session=session_clean,
        )
    except ValueError as exc:
        logger.warning("EventReadout construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ComponentBreakdown — from ConfluenceSignal.components
# ---------------------------------------------------------------------------


_NAME_NORMALIZER = {
    "BOS": "bos",
    "FVG": "fvg",
    "OrderBlock": "order_block",
    "Regime": "regime",
    "News": "news",
    "Volume": "volume",
    "Momentum": "momentum",
    "RSI_Divergence": "rsi_divergence",
}


def map_breakdown_components(
    confluence_signal: Optional[Any],
    expose_weights: bool = True,
) -> list:
    """Project a ConfluenceSignal.components list onto ComponentBreakdown[].

    Parameters
    ----------
    confluence_signal : ConfluenceSignal-like
        Object with a ``components`` list of ComponentScore-like dataclasses
        (attributes: name, weighted_score, weight, reasoning).
    expose_weights : bool, default True
        If False, redacts ``weight_max`` to None on every row (B2C surface
        protection per eval_26 IP arbitrage).

    Returns
    -------
    list[ComponentBreakdown] — empty list if no components available.
    """
    if confluence_signal is None:
        return []
    components = getattr(confluence_signal, "components", None) or []
    out = []
    for c in components:
        name_raw = getattr(c, "name", None)
        if not name_raw:
            continue
        name = _NAME_NORMALIZER.get(name_raw, str(name_raw).lower())
        weighted = getattr(c, "weighted_score", 0.0)
        weight = getattr(c, "weight", None)
        reasoning = getattr(c, "reasoning", "") or ""
        try:
            row = ComponentBreakdown(
                name=name,
                contribution=float(weighted) if _is_finite(weighted) else 0.0,
                weight_max=float(weight) if (expose_weights and weight is not None and _is_finite(weight)) else None,
                reasoning=reasoning,
            )
            out.append(row)
        except ValueError as exc:
            logger.debug("Skipping component %r: %s", name_raw, exc)
    return out


# ---------------------------------------------------------------------------
# UncertaintyContext — from CalibratedConviction
# ---------------------------------------------------------------------------


def map_uncertainty_context(calibrated_conviction: Optional[Any]) -> Optional[UncertaintyContext]:
    """Project a CalibratedConviction onto an UncertaintyContext (0-100 scale)."""
    if calibrated_conviction is None:
        return None
    interval = getattr(calibrated_conviction, "interval", None)
    if interval is None:
        return None
    try:
        return UncertaintyContext(
            conformal_lower=float(getattr(calibrated_conviction, "conformal_lower_0_100", 0.0)),
            conformal_upper=float(getattr(calibrated_conviction, "conformal_upper_0_100", 100.0)),
            coverage_alpha=float(getattr(interval, "alpha", 0.10)),
            n_calibration=int(getattr(interval, "n_calibration", 0)),
            empirical_coverage=None,  # populated separately from ACI.empirical_coverage()
        )
    except (ValueError, TypeError) as exc:
        logger.warning("UncertaintyContext construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# HistoricalStats — aggregated from SignalStore history
# ---------------------------------------------------------------------------


def map_historical_stats(
    similar_setups_n: Optional[int] = None,
    hit_rate: Optional[float] = None,
    profit_factor: Optional[float] = None,
    profit_factor_ci95: Optional[Iterable[float]] = None,
    empirical_coverage: Optional[float] = None,
    backtest_window: Optional[str] = None,
) -> Optional[HistoricalStats]:
    """Build a HistoricalStats readout from pre-computed aggregates.

    The aggregation itself (counting similar setups in SignalStore) is the
    caller's responsibility — typically a periodic batch job.
    """
    if all(
        v is None
        for v in (similar_setups_n, hit_rate, profit_factor, empirical_coverage)
    ):
        return None

    pf_ci = None
    if profit_factor_ci95 is not None:
        ci = list(profit_factor_ci95)
        if len(ci) == 2 and all(_is_finite(x) for x in ci):
            lo, hi = sorted((float(ci[0]), float(ci[1])))
            pf_ci = [lo, hi]

    try:
        return HistoricalStats(
            similar_setups_n=int(similar_setups_n) if similar_setups_n is not None else None,
            hit_rate_observed=float(hit_rate) if hit_rate is not None and _is_finite(hit_rate) else None,
            profit_factor=float(profit_factor) if profit_factor is not None and _is_finite(profit_factor) else None,
            profit_factor_ci95=pf_ci,
            empirical_coverage=float(empirical_coverage) if empirical_coverage is not None and _is_finite(empirical_coverage) else None,
            backtest_window=backtest_window,
        )
    except (ValueError, TypeError) as exc:
        logger.warning("HistoricalStats construction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_finite(x: Any) -> bool:
    try:
        import math
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


__all__ = [
    "map_structure_readout",
    "map_regime_readout",
    "map_volatility_readout",
    "map_event_readout",
    "map_breakdown_components",
    "map_uncertainty_context",
    "map_historical_stats",
]
