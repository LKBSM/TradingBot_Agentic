"""Mappers — scanner SMC outputs → MarketReading sub-sections.

Transforms `ConfluenceSignal` + `smc_features` + candles into the structured
`MarketReadingStructure`, `MarketReadingRegime`, `MarketReadingEvents`, and
generates niveau 1.5 strict tags + description (template fallback).

The Haiku LLM description engine (Étape 5) will replace the template path
when an Anthropic client is available. The template path here is the
deterministic fallback and the canonical source-of-truth for forbidden
token compliance.

Niveau 1.5 strict (per Section 1.2 of architecture doc):
- The product describes market conditions, never recommends actions.
- Forbidden tokens are enforced post-generation in the assembler (Étape 5).
- Template phrases in this module are pre-screened to never emit forbidden
  vocabulary by construction.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from src.intelligence.market_reading_schema import (
    DESCRIPTION_MAX_LENGTH,
    BOSRecent,
    CHOCHRecent,
    Direction,
    FairValueGap,
    MarketPhase,
    MarketReadingEvents,
    MarketReadingRegime,
    MarketReadingStructure,
    MTFBiasValue,
    OrderBlock,
    RetestInProgress,
    TrendValue,
    VALID_MTF_KEYS,
    VolatilityObserved,
)

# Forbidden tokens checked post-generation (Étape 5 enforces too).
# Listed here for visibility — templates in this module must never emit any.
FORBIDDEN_TOKENS: frozenset[str] = frozenset({
    "conseille",
    "déconseille",
    "deconseille",
    "évite",
    "evite",
    "entre",
    "sors",
    "risqué",
    "sûr",
    "bon moment",
    "mauvais moment",
    "achète",
    "achete",
    "vends",
})


# ---------------------------------------------------------------------------
# Helpers — direction conversion
# ---------------------------------------------------------------------------


def _signal_type_to_direction(signal_type_value: Any) -> Optional[Direction]:
    """Convert ConfluenceSignal.signal_type (LONG/SHORT) to MarketReading direction."""
    raw = getattr(signal_type_value, "value", signal_type_value)
    if raw == "LONG":
        return "bullish"
    if raw == "SHORT":
        return "bearish"
    return None


def _sign_to_direction(value: float) -> Optional[Direction]:
    if value > 0:
        return "bullish"
    if value < 0:
        return "bearish"
    return None


def _clean_float(value: Any) -> Optional[float]:
    """Return value as float unless it is None/NaN."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _first_real(smc: dict[str, float], *keys: str) -> Optional[float]:
    """First non-None/non-NaN value among ``keys`` in ``smc``."""
    for key in keys:
        v = _clean_float(smc.get(key))
        if v is not None:
            return v
    return None


# ---------------------------------------------------------------------------
# Realized structural levels — glue between SmartMoneyEngine output and the
# structure mapper. Lives HERE (not in the engine) so the detection engine is
# untouched. The two SMC pipelines (assembler + validation script) call this
# and merge the result into ``smc_features`` so the mapper publishes the REAL
# levels the engine computed, not price ± ATR proxies.
# ---------------------------------------------------------------------------


def realized_levels(enriched: Any, idx: int = -1) -> dict[str, float]:
    """Extract real structural levels for bar ``idx`` from an enriched SMC frame.

    Keys returned (only when computable; the mapper falls back gracefully):
      - ``BOS_BREAK_LEVEL_LAST`` : last non-NaN ``BOS_BREAK_LEVEL`` up to ``idx``
        (forward fill). The engine sets ``BOS_BREAK_LEVEL`` only on event bars,
        so on propagated-state bars the *real* structural level is the most
        recent break — forward-filling carries it correctly (fixes F1/F2).
      - ``OB_LEVEL_HIGH`` / ``OB_LEVEL_LOW`` : the real order-block zone the
        engine stored (``BULLISH_OB_*`` / ``BEARISH_OB_*``), not a proxy (F3).
      - ``FVG_LEVEL_HIGH`` / ``FVG_LEVEL_LOW`` : the real 3-candle fair-value-gap
        bounds, reconstructed from the same geometry the engine used (F3).
    """
    import pandas as pd  # lazy — keeps module import cheap for unit tests

    out: dict[str, float] = {}
    n = len(enriched)
    if n == 0:
        return out
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return out
    cols = set(enriched.columns)
    row = enriched.iloc[pos]

    # BOS broken level, forward-filled up to this bar.
    if "BOS_BREAK_LEVEL" in cols:
        ff = enriched["BOS_BREAK_LEVEL"].iloc[: pos + 1].ffill()
        if len(ff) and not pd.isna(ff.iloc[-1]):
            out["BOS_BREAK_LEVEL_LAST"] = float(ff.iloc[-1])

    # Order-block zone (whichever side fired on this bar; mutually exclusive).
    for hi_col, lo_col in (("BULLISH_OB_HIGH", "BULLISH_OB_LOW"),
                           ("BEARISH_OB_HIGH", "BEARISH_OB_LOW")):
        if hi_col in cols and lo_col in cols:
            hi, lo = row.get(hi_col), row.get(lo_col)
            if not pd.isna(hi) and not pd.isna(lo):
                out["OB_LEVEL_HIGH"] = float(max(hi, lo))
                out["OB_LEVEL_LOW"] = float(min(hi, lo))
                break

    # Fair-value-gap bounds via the engine's 3-candle geometry.
    fvg_dir = row.get("FVG_DIR", 0.0) if "FVG_DIR" in cols else 0.0
    if (not pd.isna(fvg_dir) and fvg_dir != 0 and pos >= 2
            and {"high", "low"} <= cols):
        high_i = float(enriched["high"].iloc[pos])
        low_i = float(enriched["low"].iloc[pos])
        high_i2 = float(enriched["high"].iloc[pos - 2])
        low_i2 = float(enriched["low"].iloc[pos - 2])
        if fvg_dir > 0:        # bullish gap: between high[i-2] (low) and low[i] (high)
            a, b = high_i2, low_i
        else:                  # bearish gap: between high[i] (low) and low[i-2] (high)
            a, b = high_i, low_i2
        out["FVG_LEVEL_HIGH"] = float(max(a, b))
        out["FVG_LEVEL_LOW"] = float(min(a, b))

    return out


# ---------------------------------------------------------------------------
# Structure mapper
# ---------------------------------------------------------------------------


def confluence_signal_to_structure(
    confluence_signal: Optional[Any],
    smc_features: dict[str, float],
    bar_ts: datetime,
    current_price: float,
) -> MarketReadingStructure:
    """Build MarketReadingStructure from confluence signal + per-bar smc features.

    `confluence_signal` may be None (no setup fired). In that case we still
    populate BOS/CHOCH from the propagating signal flags in smc_features so
    the MarketReading reflects current structural state, not just trade setups.

    `smc_features` keys consulted (all optional, defaults safely to 0/absent):
      - BOS_SIGNAL : -1/0/+1 propagating trend state
      - BOS_EVENT  : -1/0/+1 fresh break flag (used for validation_status)
      - CHOCH_SIGNAL : -1/0/+1
      - FVG_SIGNAL : -1/0/+1
      - OB_STRENGTH_NORM : 0..1
      - BOS_RETEST_ARMED : -1/0/+1

    Levels (level_high/low for OB/FVG, level for BOS) are conservatively
    approximated from current_price ± a half-ATR proxy when not explicitly
    available. A richer engine wiring (full OB/FVG list with explicit levels)
    is out of Chantier 2 scope.
    """
    atr_proxy = float(smc_features.get("ATR", 0.0)) or max(current_price * 0.001, 1e-6)
    half = atr_proxy / 2.0

    # BOS
    bos: Optional[BOSRecent] = None
    bos_signal = float(smc_features.get("BOS_SIGNAL", 0.0))
    bos_direction = _sign_to_direction(bos_signal)
    if bos_direction is not None:
        bos_event = float(smc_features.get("BOS_EVENT", 0.0))
        validation = "confirmed" if abs(bos_event) > 0 else "pending"
        # F1: publish the REAL broken structural level. The engine exposes it as
        # BOS_BREAK_LEVEL (event bars only); the pipeline forward-fills it into
        # BOS_BREAK_LEVEL_LAST so propagated states carry the real level too.
        # current_price is the last-resort fallback (no break ever seen).
        bos_level = _first_real(smc_features, "BOS_BREAK_LEVEL", "BOS_BREAK_LEVEL_LAST")
        bos = BOSRecent(
            direction=bos_direction,
            level=bos_level if bos_level is not None else float(current_price),
            broken_at=bar_ts,
            validation_status=validation,
        )

    # CHOCH
    choch: Optional[CHOCHRecent] = None
    choch_signal = float(smc_features.get("CHOCH_SIGNAL", 0.0))
    choch_direction = _sign_to_direction(choch_signal)
    if choch_direction is not None:
        # F2: there is no dedicated CHOCH level column. In this engine a CHOCH is
        # a reversal BOS on the SAME bar, so the broken level is BOS_BREAK_LEVEL
        # (set together with CHOCH_SIGNAL). Read it (or the forward-filled last)
        # instead of the non-existent CHOCH_PRICE_LEVEL that fell back to price.
        choch_level = _first_real(
            smc_features, "BOS_BREAK_LEVEL", "BOS_BREAK_LEVEL_LAST"
        )
        choch = CHOCHRecent(
            direction=choch_direction,
            level=choch_level if choch_level is not None else float(current_price),
            broken_at=bar_ts,
            validation_status="confirmed",
        )

    # Order blocks
    order_blocks: list[OrderBlock] = []
    ob_strength = float(smc_features.get("OB_STRENGTH_NORM", 0.0))
    if ob_strength > 0.0:
        sig_direction = _signal_type_to_direction(
            getattr(confluence_signal, "signal_type", None)
        )
        ob_direction = sig_direction or bos_direction
        importance = "high" if ob_strength >= 0.75 else "medium" if ob_strength >= 0.4 else "low"
        # F3: publish the REAL order-block zone the engine stored (prior-candle
        # range via OB_LEVEL_HIGH/LOW from realized_levels), not a price±ATR/2 proxy.
        ob_high = _first_real(smc_features, "OB_LEVEL_HIGH")
        ob_low = _first_real(smc_features, "OB_LEVEL_LOW")
        if ob_high is None or ob_low is None:
            ob_high, ob_low = current_price + half, current_price - half
        order_blocks.append(OrderBlock(
            id=f"OB_{bar_ts.strftime('%Y%m%d%H%M%S')}",
            direction=ob_direction,
            level_high=ob_high,
            level_low=ob_low,
            importance=importance,
            status="active",
            created_at=bar_ts,
            tested=False,
            user_flagged=False,
        ))

    # Fair value gaps
    fair_value_gaps: list[FairValueGap] = []
    fvg_signal = float(smc_features.get("FVG_SIGNAL", 0.0))
    fvg_direction = _sign_to_direction(fvg_signal)
    if fvg_direction is not None:
        # F3: publish the REAL fair-value-gap bounds (3-candle geometry via
        # FVG_LEVEL_HIGH/LOW from realized_levels), not a price±ATR/2 proxy.
        fvg_high = _first_real(smc_features, "FVG_LEVEL_HIGH")
        fvg_low = _first_real(smc_features, "FVG_LEVEL_LOW")
        if fvg_high is None or fvg_low is None:
            fvg_high, fvg_low = current_price + half, current_price - half
        fair_value_gaps.append(FairValueGap(
            id=f"FVG_{bar_ts.strftime('%Y%m%d%H%M%S')}",
            direction=fvg_direction,
            level_high=fvg_high,
            level_low=fvg_low,
            status="active",
            created_at=bar_ts,
            tested=False,
            user_flagged=False,
        ))

    # Retest
    retest_in_progress: Optional[RetestInProgress] = None
    retest_armed = float(smc_features.get("BOS_RETEST_ARMED", 0.0))
    if abs(retest_armed) > 0 and bos is not None:
        retest_in_progress = RetestInProgress(
            level=bos.level,
            type="bos_retest",
            started_at=bar_ts,
        )

    return MarketReadingStructure(
        bos=bos,
        choch=choch,
        order_blocks=order_blocks,
        fair_value_gaps=fair_value_gaps,
        retest_in_progress=retest_in_progress,
    )


# ---------------------------------------------------------------------------
# Regime mapper
# ---------------------------------------------------------------------------


def _closes(candles: Sequence[dict]) -> list[float]:
    return [float(c["close"]) for c in candles if "close" in c]


def _derive_trend(closes: Sequence[float]) -> TrendValue:
    if len(closes) < 5:
        return "neutral"
    first = closes[0]
    last = closes[-1]
    rng = max(closes) - min(closes)
    if rng <= 0:
        return "neutral"
    base = max(abs(first), 1e-9)
    pct_move = abs(last - first) / base
    rng_pct = rng / base
    if pct_move < rng_pct * 0.3:
        return "ranging"
    return "bullish" if last > first else "bearish"


def _derive_volatility(candles: Sequence[dict]) -> VolatilityObserved:
    if len(candles) < 14:
        return "normal"
    trs = []
    for c in candles:
        if "high" in c and "low" in c:
            trs.append(float(c["high"]) - float(c["low"]))
    if len(trs) < 14:
        return "normal"
    recent = sum(trs[-7:]) / 7.0
    baseline = sum(trs[:-7]) / max(len(trs) - 7, 1)
    if baseline <= 0:
        return "normal"
    ratio = recent / baseline
    if ratio < 0.7:
        return "low"
    if ratio > 1.3:
        return "elevated"
    return "normal"


def _derive_market_phase(trend: TrendValue, volatility: VolatilityObserved) -> MarketPhase:
    if trend in ("bullish", "bearish"):
        return "expansion" if volatility == "elevated" else "trend"
    if trend == "ranging":
        return "ranging"
    return "accumulation"


def _derive_bias_from_candles(candles: Sequence[dict]) -> MTFBiasValue:
    closes = _closes(candles)
    trend = _derive_trend(closes)
    if trend in ("bullish", "bearish", "ranging", "neutral"):
        return trend  # type: ignore[return-value]
    return "neutral"


def candles_to_regime(
    candles: Sequence[dict],
    mtf_candles_above: dict[str, Sequence[dict]],
) -> MarketReadingRegime:
    """Derive regime from current-TF candles + bias from upper timeframes.

    `candles` : OHLCV rows for the requested TF, oldest first. Each item must
    expose at minimum `close`, `high`, `low` keys.
    `mtf_candles_above` : mapping from upper-TF key (`h1`, `h4`, ...) to its
    candles list. Only keys in `VALID_MTF_KEYS` are kept.
    """
    closes = _closes(candles)
    trend = _derive_trend(closes)
    volatility = _derive_volatility(candles)
    market_phase = _derive_market_phase(trend, volatility)

    mtf_confluence: dict[str, MTFBiasValue] = {}
    for key, tf_candles in mtf_candles_above.items():
        if key not in VALID_MTF_KEYS:
            continue
        if not tf_candles:
            continue
        mtf_confluence[key] = _derive_bias_from_candles(tf_candles)

    return MarketReadingRegime(
        trend=trend,
        volatility_observed=volatility,
        market_phase=market_phase,
        mtf_confluence=mtf_confluence,
    )


# ---------------------------------------------------------------------------
# Events stub (filled by Chantier 3)
# ---------------------------------------------------------------------------


def empty_events() -> MarketReadingEvents:
    """Return an empty events block. News pipeline lives in Chantier 3."""
    return MarketReadingEvents()


# ---------------------------------------------------------------------------
# Tags + description template fallback
# ---------------------------------------------------------------------------


_TREND_FR = {
    "bullish": "haussière",
    "bearish": "baissière",
    "neutral": "neutre",
    "ranging": "en range",
}

_VOL_FR = {
    "low": "faible",
    "normal": "normale",
    "elevated": "élevée",
}

_PHASE_FR = {
    "accumulation": "d'accumulation",
    "distribution": "de distribution",
    "trend": "de tendance",
    "ranging": "de range",
    "expansion": "d'expansion",
}


def _build_tags(
    structure: MarketReadingStructure,
    regime: MarketReadingRegime,
) -> list[str]:
    tags: list[str] = []

    tags.append(f"trend_{regime.trend}")
    tags.append(f"volatility_{regime.volatility_observed}")
    tags.append(f"phase_{regime.market_phase}")

    if structure.bos is not None:
        tags.append(f"bos_recent_{structure.bos.direction}")
    if structure.choch is not None:
        tags.append(f"choch_recent_{structure.choch.direction}")
    if structure.retest_in_progress is not None:
        tags.append("retest_in_progress")
    if any(ob.status == "active" for ob in structure.order_blocks):
        tags.append("ob_active")
    if any(fvg.status == "active" for fvg in structure.fair_value_gaps):
        tags.append("fvg_active")

    if regime.mtf_confluence:
        biases = set(regime.mtf_confluence.values())
        if len(biases) == 1:
            (single,) = biases
            if single in ("bullish", "bearish"):
                tags.append("mtf_aligned")
        elif {"bullish", "bearish"}.issubset(biases):
            tags.append("mtf_divergent")
        else:
            tags.append("mtf_mixed")

    return tags


def _build_description(
    structure: MarketReadingStructure,
    regime: MarketReadingRegime,
) -> str:
    """Template-based niveau 1.5 strict description (French, ≤ 280 chars).

    Uses only descriptive verbs (est, indique, montre). Never emits forbidden
    tokens (recommendation/judgement vocabulary).
    """
    trend_fr = _TREND_FR.get(regime.trend, regime.trend)
    vol_fr = _VOL_FR.get(regime.volatility_observed, regime.volatility_observed)
    phase_fr = _PHASE_FR.get(regime.market_phase, regime.market_phase)

    parts: list[str] = []
    parts.append(f"Tendance {trend_fr}, volatilité {vol_fr}, phase {phase_fr}.")

    if structure.bos is not None:
        parts.append(
            f"BOS {_TREND_FR[structure.bos.direction]} récent ({structure.bos.validation_status})."
        )
    if structure.retest_in_progress is not None:
        parts.append("Retest de structure en cours.")
    if structure.order_blocks:
        parts.append("Order Block actif.")
    if structure.fair_value_gaps:
        parts.append("FVG actif.")

    if regime.mtf_confluence:
        biases = set(regime.mtf_confluence.values())
        if len(biases) == 1:
            (single,) = biases
            parts.append(f"MTF alignée {_TREND_FR.get(single, single)}.")
        else:
            parts.append("MTF mixte.")

    desc = " ".join(parts)
    if len(desc) > DESCRIPTION_MAX_LENGTH:
        desc = desc[:DESCRIPTION_MAX_LENGTH - 1].rstrip() + "."
    return desc


def tags_and_description(
    structure: MarketReadingStructure,
    regime: MarketReadingRegime,
) -> tuple[list[str], str]:
    """Build tag list + niveau 1.5 strict description (template fallback path)."""
    tags = _build_tags(structure, regime)
    description = _build_description(structure, regime)
    return tags, description


def contains_forbidden_tokens(text: str) -> Optional[str]:
    """Return the first forbidden token found in `text`, or None if clean.

    Used as a post-generation guard in the Haiku engine (Étape 5) and as a
    structural test for any template path in this module.
    Word-boundary match (so "entre" matches "entre" but not "entrer", and
    "bon moment" matches that phrase but not "bon momentum").
    """
    lower = text.lower()
    for token in FORBIDDEN_TOKENS:
        if re.search(rf"\b{re.escape(token)}\b", lower):
            return token
    return None


__all__ = [
    "FORBIDDEN_TOKENS",
    "candles_to_regime",
    "confluence_signal_to_structure",
    "contains_forbidden_tokens",
    "empty_events",
    "tags_and_description",
]
