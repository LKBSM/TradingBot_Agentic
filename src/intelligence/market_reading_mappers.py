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
from dataclasses import dataclass
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
# P4: the bare "entre" is intentionally EXCLUDED — it is the French preposition
# "between" ("FVG entre 2376 et 2378"), a high-frequency homonym of the trade
# verb. This matches the chatbot's deliberate exclusion (chatbot/constants.py
# §3). The directive forms entrez/entrer/entry are kept. Without this, legitimate
# descriptive Haiku output was rejected → unjustified template fallbacks.
FORBIDDEN_TOKENS: frozenset[str] = frozenset({
    "conseille",
    "déconseille",
    "deconseille",
    "évite",
    "evite",
    "entrez",
    "entrer",
    "entry",
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


def _epoch_to_dt(value: Any) -> Optional[datetime]:
    """Convert epoch SECONDS (float) to a tz-aware UTC datetime, or None.

    Used to recover the ORIGINAL break time for a persisted (non-fresh) BOS from
    the ``BOS_BREAK_TS`` glue field, so ``broken_at`` is honest rather than the
    current bar.
    """
    f = _clean_float(value)
    if f is None:
        return None
    try:
        return datetime.fromtimestamp(f, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
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

    # Timestamp of the most recent BOS event up to this bar (forward-carried), so
    # a PERSISTED active break (still vouched for by the retest state machine —
    # D1-b option 1a) reports its ORIGINAL break time, not the current bar. Glue,
    # not engine logic. Guarded on a DatetimeIndex so integer-indexed test frames
    # never produce a bogus timestamp.
    if "BOS_EVENT" in cols and isinstance(enriched.index, pd.DatetimeIndex):
        ev = enriched["BOS_EVENT"].iloc[: pos + 1]
        nz = ev[ev != 0]
        if len(nz):
            out["BOS_BREAK_TS"] = float(nz.index[-1].timestamp())

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
# Multi-zone registry — surfaces ALL still-relevant OB/FVG zones the engine
# computed over the lookback window, not just the one that fired on the last
# bar. Lives HERE (glue layer), the detection engine is untouched: the engine
# already emits a zone on every qualifying bar via BULLISH_OB_*/BEARISH_OB_*
# /FVG_* columns. This walks those columns, applies a lifecycle (mitigation /
# invalidation for OB, fill for FVG) and drops consumed zones.
#
# Audit DETECTION_QUALITY_REVIEW_2026_06_12 §T1: the assembler read only
# enriched.iloc[-1], so the product showed ≤1 OB and ≤1 FVG (often 0) while the
# engine had computed dozens. This restores the cardinality the engine produces.
#
# IMPORTANT — gated by founder annotation: the IMPORTANCE ranking and the
# active/mitigated retention policy below use the engine's existing strength
# heuristic (OB body/ATR). They are PROVISIONAL surfacing rules, not a new
# detection definition. Calibrate the cap, the importance cutoffs and the
# retention policy against the annotation dataset (audit §4/§5) — the geometry
# of each zone is the engine's, untouched.
# ---------------------------------------------------------------------------

# Default cap per zone type. Keeps the surface readable; tune vs annotation.
# Widened 2026-06-15 (was 6) for indicator-grade context. Overridable per call
# and via the MAX_ZONES_PER_TYPE env var (resolved in collect_zones).
MAX_ZONES_PER_TYPE = 12


# ---------------------------------------------------------------------------
# Mitigation policy — SINGLE SOURCE OF TRUTH for the OB/FVG lifecycle rules.
#
# >>> DÉFAUTS À VALIDER PAR ANNOTATION <<<
# These are PROVISIONAL surfacing rules, not a detection definition. The zone
# GEOMETRY (where each OB/FVG sits) comes from the engine and is untouched —
# this only decides WHEN a formed zone is considered touched (mitigated /
# partially filled) or consumed (invalidated / filled, and therefore dropped).
# Calibrate every knob below against the annotation dataset (audit §4/§5).
#
# Conservative bias (mission §2/§C): in doubt, declare a zone mitigated EARLIER,
# never later, and never surface a consumed zone as active.
#
# Founder-validated defaults 2026-06-15 (see docs/audits/OB_FVG_MITIGATION_*):
#   - OB invalidated on a CLOSE through the block → dropped.
#   - OB tapped by a wick (any overlap) → 'mitigated', kept VISIBLE & tagged.
#   - FVG removed only on a FULL (100% / far-edge) fill; partial fill kept tagged.
# Every threshold lives here so nothing is scattered across the collector.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MitigationPolicy:
    """Tunable OB/FVG lifecycle thresholds. See module comment above."""

    # --- Order blocks ---------------------------------------------------- #
    # A later candle that CLOSES through the block invalidates it (consumed).
    ob_invalidate_on_close_through: bool = True
    # Fraction of the block height a wick must penetrate (from the near edge)
    # to count as a tap/mitigation. 0.0 = any touch (most conservative, current
    # default). Raise toward 0.5 to require a deeper tap before declaring the
    # block mitigated — LESS conservative, hence annotation-gated.
    ob_mitigation_penetration: float = 0.0
    # Founder 2026-06-15: a tapped-but-held OB stays VISIBLE, tagged 'mitigated'.
    # Flip to True to DROP mitigated OBs entirely (stricter / cleaner surface).
    ob_drop_when_mitigated: bool = False

    # --- Fair value gaps ------------------------------------------------- #
    # Fraction of the gap height price must retrace for the gap to be FILLED
    # (and dropped). 1.0 = far edge / 100% (founder 2026-06-15). Lower it to
    # drop gaps earlier (e.g. 0.5 = mid-fill) — LESS history shown, annotation-
    # gated. Any entry short of this fraction is 'partially_filled'.
    fvg_fill_fraction: float = 1.0
    # Founder 2026-06-15: a partially filled FVG stays VISIBLE, tagged
    # 'partially_filled'. Flip to True to DROP a gap on first entry (strictest).
    fvg_drop_when_partial: bool = False


# The active policy. Constructed once; import and pass to the lifecycle helpers.
MITIGATION_POLICY = MitigationPolicy()


def _ob_lifecycle(
    side: str,
    zhigh: float,
    zlow: float,
    highs: Any,
    lows: Any,
    closes: Any,
    created: int,
    upto: int,
    policy: MitigationPolicy = MITIGATION_POLICY,
) -> tuple[str, bool, Optional[int]]:
    """Classify an order-block zone over bars (created, upto].

    Returns ``(status, tested, first_tap_idx)`` where status ∈
    {active, mitigated, invalidated}:
      * invalidated — a later candle CLOSED through the zone (support lost for a
        bullish OB, resistance reclaimed for a bearish OB) → consumed/dropped.
      * mitigated   — price traded into the zone deep enough (per policy) but it
        held (a tap). ``first_tap_idx`` is the bar of the first such tap.
      * active      — price has not returned to the zone yet.

    All thresholds come from ``policy`` (the single source of truth). The zone
    geometry is the engine's; this only times the interaction.
    """
    height = max(zhigh - zlow, 0.0)
    depth = policy.ob_mitigation_penetration * height
    tested = False
    first_tap: Optional[int] = None
    for j in range(created + 1, upto + 1):
        if side == "bullish":
            # Support: price dips from above; require it to reach depth into the
            # block from the near (top) edge, and not be entirely below it.
            if lows[j] <= zhigh - depth and highs[j] >= zlow:
                tested = True
                if first_tap is None:
                    first_tap = j
            if policy.ob_invalidate_on_close_through and closes[j] < zlow:
                return "invalidated", tested, first_tap
        else:
            # Resistance: price rises from below; require it to reach depth into
            # the block from the near (bottom) edge.
            if highs[j] >= zlow + depth and lows[j] <= zhigh:
                tested = True
                if first_tap is None:
                    first_tap = j
            if policy.ob_invalidate_on_close_through and closes[j] > zhigh:
                return "invalidated", tested, first_tap
    return ("mitigated" if tested else "active"), tested, first_tap


def _fvg_lifecycle(
    side: str,
    zhigh: float,
    zlow: float,
    highs: Any,
    lows: Any,
    created: int,
    upto: int,
    policy: MitigationPolicy = MITIGATION_POLICY,
) -> tuple[str, bool, Optional[int]]:
    """Classify a fair-value-gap over bars (created, upto].

    Returns ``(status, entered, first_entry_idx)`` where status ∈
    {active, partially_filled, filled}. A bullish gap (price gapped up, empty
    band ``[zlow, zhigh]``) fills from above: ``filled`` once a later low
    retraces ``policy.fvg_fill_fraction`` of the gap height (1.0 = far edge
    ``zlow``), ``partially_filled`` once a later low dips below ``zhigh`` (near
    edge). Bearish gap is the mirror, filled from below. ``first_entry_idx`` is
    the bar of the first partial entry.
    """
    height = max(zhigh - zlow, 0.0)
    fill = policy.fvg_fill_fraction * height
    entered = False
    first_entry: Optional[int] = None
    for j in range(created + 1, upto + 1):
        if side == "bullish":
            if lows[j] <= zhigh - fill:  # retraced enough → filled
                return "filled", True, (first_entry if first_entry is not None else j)
            if lows[j] <= zhigh:
                entered = True
                if first_entry is None:
                    first_entry = j
        else:
            if highs[j] >= zlow + fill:
                return "filled", True, (first_entry if first_entry is not None else j)
            if highs[j] >= zlow:
                entered = True
                if first_entry is None:
                    first_entry = j
    return ("partially_filled" if entered else "active"), entered, first_entry


def _zone_created_at(enriched: Any, k: int) -> Optional[datetime]:
    import pandas as pd

    if isinstance(enriched.index, pd.DatetimeIndex):
        ts = enriched.index[k]
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def collect_zones(
    enriched: Any,
    idx: int = -1,
    max_per_type: Optional[int] = None,
) -> dict[str, list[dict]]:
    """Collect every still-relevant OB / FVG zone up to bar ``idx``.

    Returns ``{"order_blocks": [...], "fair_value_gaps": [...]}`` as plain dicts
    (the structure mapper builds the pydantic models, filling ``created_at`` from
    ``bar_ts`` when the frame has no datetime index). Consumed zones (invalidated
    OB, filled FVG) are dropped. Ordering: active before partially-consumed, then
    by strength/size, then by recency; capped to ``max_per_type`` (defaults to the
    ``MAX_ZONES_PER_TYPE`` env var, else the module constant).
    """
    import os
    import pandas as pd

    if max_per_type is None:
        try:
            max_per_type = int(os.environ.get("MAX_ZONES_PER_TYPE", MAX_ZONES_PER_TYPE))
        except (TypeError, ValueError):
            max_per_type = MAX_ZONES_PER_TYPE

    out: dict[str, list[dict]] = {"order_blocks": [], "fair_value_gaps": []}
    n = len(enriched)
    if n == 0:
        return out
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return out

    cols = set(enriched.columns)
    highs = enriched["high"].values if "high" in cols else None
    lows = enriched["low"].values if "low" in cols else None
    closes = enriched["close"].values if "close" in cols else None
    if highs is None or lows is None or closes is None:
        return out

    # ---- Order blocks ----------------------------------------------------
    ob_cols = {"BULLISH_OB_HIGH", "BULLISH_OB_LOW", "BEARISH_OB_HIGH", "BEARISH_OB_LOW"}
    if ob_cols <= cols:
        strength = (
            enriched["OB_STRENGTH_NORM"].values if "OB_STRENGTH_NORM" in cols else None
        )
        bull_hi = enriched["BULLISH_OB_HIGH"].values
        bull_lo = enriched["BULLISH_OB_LOW"].values
        bear_hi = enriched["BEARISH_OB_HIGH"].values
        bear_lo = enriched["BEARISH_OB_LOW"].values
        obs: list[dict] = []
        for k in range(pos + 1):
            for side, hv, lv in (
                ("bullish", bull_hi[k], bull_lo[k]),
                ("bearish", bear_hi[k], bear_lo[k]),
            ):
                if pd.isna(hv) or pd.isna(lv):
                    continue
                zhigh, zlow = float(max(hv, lv)), float(min(hv, lv))
                st = float(strength[k]) if strength is not None and not pd.isna(strength[k]) else 0.0
                status, tested, tap_idx = _ob_lifecycle(
                    side, zhigh, zlow, highs, lows, closes, k, pos
                )
                # Honesty guardrail (mission §C): never surface a consumed zone.
                if status == "invalidated":
                    continue
                if status == "mitigated" and MITIGATION_POLICY.ob_drop_when_mitigated:
                    continue
                created_at = _zone_created_at(enriched, k)
                mitigated_at = _zone_created_at(enriched, tap_idx) if tap_idx is not None else None
                importance = "high" if st >= 0.75 else "medium" if st >= 0.4 else "low"
                obs.append({
                    "direction": side,
                    "level_high": zhigh,
                    "level_low": zlow,
                    "importance": importance,
                    "status": status,
                    "tested": tested,
                    "created_at": created_at,
                    "mitigated_at": mitigated_at,
                    "_strength": st,
                    "_k": k,
                })
        # active first, then by strength, then most recent first.
        obs.sort(key=lambda z: (z["status"] != "active", -z["_strength"], -z["_k"]))
        out["order_blocks"] = obs[:max_per_type]

    # ---- Fair value gaps -------------------------------------------------
    if "FVG_DIR" in cols and {"high", "low"} <= cols:
        fvg_dir = enriched["FVG_DIR"].values
        size_norm = (
            enriched["FVG_SIZE_NORM"].values if "FVG_SIZE_NORM" in cols else None
        )
        fvgs: list[dict] = []
        for k in range(2, pos + 1):
            d = fvg_dir[k]
            if pd.isna(d) or d == 0:
                continue
            if d > 0:  # bullish gap: high[k-2] (low edge) .. low[k] (high edge)
                a, b = float(highs[k - 2]), float(lows[k])
                side = "bullish"
            else:      # bearish gap: high[k] (low edge) .. low[k-2] (high edge)
                a, b = float(highs[k]), float(lows[k - 2])
                side = "bearish"
            zhigh, zlow = max(a, b), min(a, b)
            status, tested, entry_idx = _fvg_lifecycle(side, zhigh, zlow, highs, lows, k, pos)
            # Honesty guardrail (mission §C): never surface a consumed zone.
            if status == "filled":
                continue
            if status == "partially_filled" and MITIGATION_POLICY.fvg_drop_when_partial:
                continue
            sz = float(size_norm[k]) if size_norm is not None and not pd.isna(size_norm[k]) else (zhigh - zlow)
            fvgs.append({
                "direction": side,
                "level_high": zhigh,
                "level_low": zlow,
                "status": status,
                "tested": tested,
                "created_at": _zone_created_at(enriched, k),
                "mitigated_at": _zone_created_at(enriched, entry_idx) if entry_idx is not None else None,
                "_size": sz,
                "_k": k,
            })
        fvgs.sort(key=lambda z: (z["status"] != "active", -z["_size"], -z["_k"]))
        out["fair_value_gaps"] = fvgs[:max_per_type]

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
      - BOS_RETEST_STATE : 0 / ±1 (awaiting) / ±2 (armed) — lifecycle of the
        break: BOS persists while != 0; "retest in progress" only while ±2.

    Levels (level_high/low for OB/FVG, level for BOS) are conservatively
    approximated from current_price ± a half-ATR proxy when not explicitly
    available. A richer engine wiring (full OB/FVG list with explicit levels)
    is out of Chantier 2 scope.
    """
    atr_proxy = float(smc_features.get("ATR", 0.0)) or max(current_price * 0.001, 1e-6)
    half = atr_proxy / 2.0

    # BOS
    # bos_direction reflects the propagated trend state (BOS_SIGNAL) and is kept
    # for OB-direction fallback below even when no fresh break is shown.
    bos: Optional[BOSRecent] = None
    bos_signal = float(smc_features.get("BOS_SIGNAL", 0.0))
    bos_direction = _sign_to_direction(bos_signal)
    bos_event = float(smc_features.get("BOS_EVENT", 0.0))
    retest_state = float(smc_features.get("BOS_RETEST_STATE", 0.0))
    state_direction = _sign_to_direction(retest_state)
    # F6: a "recent BOS" is NOT the continuously-propagated BOS_SIGNAL trend
    # state — emitting on every propagated bar surfaced a (stale) BOS on ~100% of
    # readings. We surface a break only when it is genuinely active.
    #
    # D1-b (option 1a): a break is active when EITHER it is fresh at this candle
    # close (BOS_EVENT != 0) OR a prior break is still vouched for by the engine's
    # retest state machine (BOS_RETEST_STATE != 0 = awaiting/armed). That state
    # self-clears to 0 on invalidation, reclaim, or timeout
    # (strategy_features._calculate_bos_retest_*), and we additionally require the
    # propagated trend not to have inverted against the break direction
    # ("BOS_SIGNAL n'est pas inversé"). This relies STRICTLY on engine-produced
    # state — NO detection threshold is touched. Persistence lives at this
    # assembler-layer mapper, not in the detection engine. The window is bounded
    # (awaiting_timeout=20 + armed_window=5 bars by default), so this is the
    # opposite of the F6 "stale on ~100%" bug.
    fresh_break = abs(bos_event) > 0
    persisted_break = (
        state_direction is not None
        and (bos_direction is None or bos_direction == state_direction)
    )
    if fresh_break or persisted_break:
        # F1: publish the REAL broken structural level (present on event bars as
        # BOS_BREAK_LEVEL; BOS_BREAK_LEVEL_LAST is the forward-filled level a
        # persisted break sources). current_price is the last-resort fallback.
        bos_level = _first_real(smc_features, "BOS_BREAK_LEVEL", "BOS_BREAK_LEVEL_LAST")
        if fresh_break:
            event_direction = _sign_to_direction(bos_event) or bos_direction
            broken_at = bar_ts
        else:
            # Persisted break: direction from the retest state, original break
            # time from the glue field (honest broken_at, not the current bar).
            # A recovered time AFTER the bar being read means the candle index
            # was in a wrong clock domain (audit 2026-06-12 §T2 published
            # broken_at timestamps in the future) — never surface it.
            event_direction = state_direction or bos_direction
            recovered = _epoch_to_dt(smc_features.get("BOS_BREAK_TS"))
            bar_ts_utc = (
                bar_ts if bar_ts.tzinfo else bar_ts.replace(tzinfo=timezone.utc)
            )
            if recovered is not None and recovered > bar_ts_utc:
                recovered = None
            broken_at = recovered or bar_ts
        bos = BOSRecent(
            direction=event_direction,
            level=bos_level if bos_level is not None else float(current_price),
            broken_at=broken_at,
            validation_status="confirmed",  # broke and not invalidated ⇒ confirmed
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

    # Order blocks + fair value gaps.
    # Preferred path: the multi-zone registry (all still-relevant zones the
    # engine computed over the window, with lifecycle), injected by the SMC
    # pipeline under ``_zones``. Fallback: the legacy single-last-bar zone, kept
    # so callers/tests that don't run collect_zones still behave as before.
    zones = smc_features.get("_zones")
    if isinstance(zones, dict):
        order_blocks, fair_value_gaps = _zones_to_models(zones, bar_ts)
        return MarketReadingStructure(
            bos=bos,
            choch=choch,
            order_blocks=order_blocks,
            fair_value_gaps=fair_value_gaps,
            retest_in_progress=_build_retest(
                smc_features, retest_state, fresh_break, persisted_break, bos,
                current_price, bar_ts,
            ),
        )

    # ---- legacy single-bar fallback -------------------------------------
    order_blocks = []
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

    return MarketReadingStructure(
        bos=bos,
        choch=choch,
        order_blocks=order_blocks,
        fair_value_gaps=fair_value_gaps,
        retest_in_progress=_build_retest(
            smc_features, retest_state, fresh_break, persisted_break, bos,
            current_price, bar_ts,
        ),
    )


def _build_retest(
    smc_features: dict[str, float],
    retest_state: float,
    fresh_break: bool,
    persisted_break: bool,
    bos: Optional[BOSRecent],
    current_price: float,
    bar_ts: datetime,
) -> Optional[RetestInProgress]:
    """Shared retest-in-progress builder for both the multi-zone and legacy paths.

    D1-b: the BOS LEVEL persists for the whole active window (BOS_RETEST_STATE
    != 0 = awaiting OR armed). The "retest in progress" flag is narrower: shown
    ONLY during the ARMED sub-state (±2, price has returned to the broken level),
    never during AWAITING (±1). Reads the SAME engine-produced state — no
    detection threshold is touched. Requires the break to be surfaced so the UI
    never shows a retest of a break that was dropped (e.g. trend inverted).
    """
    if abs(retest_state) != 2.0 or not (fresh_break or persisted_break):
        return None
    retest_level = _first_real(smc_features, "BOS_BREAK_LEVEL", "BOS_BREAK_LEVEL_LAST")
    if retest_level is None:
        retest_level = bos.level if bos is not None else float(current_price)
    return RetestInProgress(level=retest_level, type="bos_retest", started_at=bar_ts)


def _zones_to_models(
    zones: dict[str, list[dict]],
    bar_ts: datetime,
) -> tuple[list[OrderBlock], list[FairValueGap]]:
    """Convert collected zone dicts (from :func:`collect_zones`) to schema models.

    ``created_at`` falls back to ``bar_ts`` when the collector could not derive a
    per-zone timestamp (non-datetime frame index). The ``id`` is stable per zone
    (direction + created time) so the same zone keeps its identity across reads.
    """
    order_blocks: list[OrderBlock] = []
    for z in zones.get("order_blocks", []):
        created = z.get("created_at") or bar_ts
        order_blocks.append(OrderBlock(
            id=f"OB_{z['direction']}_{created.strftime('%Y%m%d%H%M%S')}",
            direction=z["direction"],
            level_high=z["level_high"],
            level_low=z["level_low"],
            importance=z["importance"],
            status=z["status"],
            created_at=created,
            tested=z["tested"],
            mitigated_at=z.get("mitigated_at"),
            user_flagged=False,
        ))
    fair_value_gaps: list[FairValueGap] = []
    for z in zones.get("fair_value_gaps", []):
        created = z.get("created_at") or bar_ts
        fair_value_gaps.append(FairValueGap(
            id=f"FVG_{z['direction']}_{created.strftime('%Y%m%d%H%M%S')}",
            direction=z["direction"],
            level_high=z["level_high"],
            level_low=z["level_low"],
            status=z["status"],
            created_at=created,
            tested=z["tested"],
            mitigated_at=z.get("mitigated_at"),
            user_flagged=False,
        ))
    return order_blocks, fair_value_gaps


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
