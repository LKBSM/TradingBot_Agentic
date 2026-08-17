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
    LiquidityPool,
    MarketPhase,
    MarketReadingEvents,
    MarketReadingRegime,
    MarketReadingStructure,
    MTFBiasValue,
    OrderBlock,
    RetestInProgress,
    TrendReference,
    TrendValue,
    VALID_MTF_KEYS,
    VolatilityDetail,
    VolatilityObserved,
    ZoneContact,
    ZoneOrigin,
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


def _event_id(kind: str, direction: Any, broken_at: Any) -> str:
    """STR-2 defect C: a STABLE, COLLISION-FREE id for a structural break event.

    ``kind`` (« bos »/« choch ») is part of the id, so a BOS and a CHOCH that
    land on the SAME bar get DIFFERENT ids — the chart anchors focus by this id
    instead of by timestamp (which collides on a shared bar). Deterministic
    (same event → same id across readings): ``<kind>_<broken_at_iso>_<direction>``.
    """
    ts = broken_at.isoformat() if hasattr(broken_at, "isoformat") else str(broken_at)
    return f"{kind}_{ts}_{direction}"


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

# VZ-1: how many most-recently CONSUMED zones (per type) the payload carries for
# the /zones « Comblées » group. Small on purpose — recent history, not an
# archive. Overridable via MAX_CONSUMED_ZONES_PER_TYPE.
MAX_CONSUMED_ZONES_PER_TYPE = 6


def _max_consumed() -> int:
    import os
    try:
        return int(os.environ.get("MAX_CONSUMED_ZONES_PER_TYPE", MAX_CONSUMED_ZONES_PER_TYPE))
    except (TypeError, ValueError):
        return MAX_CONSUMED_ZONES_PER_TYPE

# Payload guardrail per structure-event type (BOS / CHOCH) — NOT a display top-N.
# The live surface shows EVERY break the analysis window holds (MT-D1 fix: the old
# top-8 silently hid real events — 14 BOS detected in a 500-bar H1 window surfaced
# as 8). A 500-bar window cannot realistically hold this many single-type breaks,
# so this only guards a pathological payload; it never truncates a normal reading.
# The window bound (MARKET_READING_LOOKBACK) is what scopes the journal, and the
# front now LABELS it ("N événements · fenêtre X bougies ≈ Y"). Overridable via the
# MAX_STRUCTURE_EVENTS env var.
MAX_STRUCTURE_EVENTS = 250

# Default cap on external liquidity pools surfaced per read. Keeps the surface
# readable; overridable per call and via the MAX_LIQUIDITY_POOLS env var.
MAX_LIQUIDITY_POOLS = 8


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

    # --- VZ-1 contact ledger --------------------------------------------- #
    # Fraction of the zone height a contact must penetrate (from the near edge)
    # to count as a real ENTRY rather than a mere EDGE TOUCH (a kiss). Used ONLY
    # to classify the per-contact ledger — it is NOT a detection threshold and
    # NEVER changes touch_count/tested/status (those keep the depth-0 predicate).
    contact_edge_touch_fraction: float = 0.10


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
) -> tuple[str, bool, Optional[int], Optional[int], int, list[int]]:
    """Classify an order-block zone over bars (created, upto].

    Returns ``(status, tested, first_tap_idx, invalidated_idx, touch_count,
    touch_bars)`` where status ∈ {active, mitigated, invalidated}:
      * invalidated — a later candle CLOSED through the zone (support lost for a
        bullish OB, resistance reclaimed for a bearish OB) → consumed/dropped.
        ``invalidated_idx`` is that candle's bar (None otherwise). Reported by
        the rejection diagnostics; purely informational, never a decision input.
      * mitigated   — price traded into the zone deep enough (per policy) but it
        held (a tap). ``first_tap_idx`` is the bar of the first such tap.
      * active      — price has not returned to the zone yet.

    ``touch_count`` counts DISTINCT taps — a maximal run of consecutive in-zone
    bars is ONE touch — and ``touch_bars`` holds each touch's ENTRY bar. Both are
    ADDITIVE: they reuse the exact same per-bar tap predicate as ``tested``, so
    ``touch_count >= 1 ⟺ tested`` and ``touch_bars[0] == first_tap_idx``. The
    status/tested/first_tap/invalidation logic is byte-identical to before.

    All thresholds come from ``policy`` (the single source of truth). The zone
    geometry is the engine's; this only times the interaction.
    """
    height = max(zhigh - zlow, 0.0)
    depth = policy.ob_mitigation_penetration * height
    tested = False
    first_tap: Optional[int] = None
    touch_count = 0
    touch_bars: list[int] = []
    in_zone = False  # was the bar INSIDE the zone on the previous iteration?
    for j in range(created + 1, upto + 1):
        if side == "bullish":
            # Support: price dips from above; require it to reach depth into the
            # block from the near (top) edge, and not be entirely below it.
            tap = lows[j] <= zhigh - depth and highs[j] >= zlow
        else:
            # Resistance: price rises from below; require it to reach depth into
            # the block from the near (bottom) edge.
            tap = highs[j] >= zlow + depth and lows[j] <= zhigh
        if tap:
            tested = True
            if first_tap is None:
                first_tap = j
            if not in_zone:  # rising edge out→in → a new DISTINCT touch
                touch_count += 1
                touch_bars.append(j)
            in_zone = True
        else:
            in_zone = False
        if policy.ob_invalidate_on_close_through and (
            (side == "bullish" and closes[j] < zlow) or (side != "bullish" and closes[j] > zhigh)
        ):
            return "invalidated", tested, first_tap, j, touch_count, touch_bars
    return ("mitigated" if tested else "active"), tested, first_tap, None, touch_count, touch_bars


def _fvg_lifecycle(
    side: str,
    zhigh: float,
    zlow: float,
    highs: Any,
    lows: Any,
    created: int,
    upto: int,
    policy: MitigationPolicy = MITIGATION_POLICY,
) -> tuple[str, bool, Optional[int], Optional[float], int, list[int]]:
    """Classify a fair-value-gap over bars (created, upto].

    Returns ``(status, entered, first_entry_idx, fill_level, touch_count,
    touch_bars)`` where status ∈ {active, partially_filled, filled}. A bullish
    gap (price gapped up, empty band ``[zlow, zhigh]``) fills from above:
    ``filled`` once a later low retraces ``policy.fvg_fill_fraction`` of the gap
    height (1.0 = far edge ``zlow``), ``partially_filled`` once a later low dips
    below ``zhigh`` (near edge). Bearish gap is the mirror, filled from below.
    ``first_entry_idx`` is the bar of the first partial entry.

    ``touch_count`` counts DISTINCT entries (a maximal run of consecutive in-band
    bars is ONE) and ``touch_bars`` holds each entry bar — ADDITIVE, reusing the
    exact same per-bar entry predicate as ``entered`` (``touch_count >= 1 ⟺
    entered``, ``touch_bars[0] == first_entry_idx``). Status/entered/fill logic
    is byte-identical to before.

    ``fill_level`` is the DEEPEST price the wicks reached INTO the band (clamped
    to ``[zlow, zhigh]``): the lowest low for a bullish gap (it fills downward),
    the highest high for a bearish one. ``None`` while still active/untouched.
    Purely a measurement of engine-emitted highs/lows — it bounds the still-open
    portion of the box, it does NOT recompute or re-detect the gap.
    """
    height = max(zhigh - zlow, 0.0)
    fill = policy.fvg_fill_fraction * height
    entered = False
    first_entry: Optional[int] = None
    deepest: Optional[float] = None  # deepest penetration price into the band
    touch_count = 0
    touch_bars: list[int] = []
    in_band = False
    for j in range(created + 1, upto + 1):
        if side == "bullish":
            if lows[j] <= zhigh - fill:  # retraced enough → filled
                return "filled", True, (first_entry if first_entry is not None else j), zlow, touch_count, touch_bars
            entry = lows[j] <= zhigh
            if entry:
                entered = True
                if first_entry is None:
                    first_entry = j
                pen = max(float(lows[j]), zlow)  # clamp into the band
                if deepest is None or pen < deepest:
                    deepest = pen
        else:
            if highs[j] >= zlow + fill:
                return "filled", True, (first_entry if first_entry is not None else j), zhigh, touch_count, touch_bars
            entry = highs[j] >= zlow
            if entry:
                entered = True
                if first_entry is None:
                    first_entry = j
                pen = min(float(highs[j]), zhigh)  # clamp into the band
                if deepest is None or pen > deepest:
                    deepest = pen
        if entry:
            if not in_band:  # rising edge → a new DISTINCT entry
                touch_count += 1
                touch_bars.append(j)
            in_band = True
        else:
            in_band = False
    return ("partially_filled" if entered else "active"), entered, first_entry, deepest, touch_count, touch_bars


def _zone_created_at(enriched: Any, k: int) -> Optional[datetime]:
    import pandas as pd

    if isinstance(enriched.index, pd.DatetimeIndex):
        ts = enriched.index[k]
        dt = ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    return None


def _ob_contacts(
    side: str,
    zhigh: float,
    zlow: float,
    highs: Any,
    lows: Any,
    closes: Any,
    created: int,
    upto: int,
    policy: MitigationPolicy = MITIGATION_POLICY,
) -> list[dict]:
    """Per-contact ledger for an order block over bars (created, upto].

    READ-ONLY classification that reuses the EXACT tap predicate of
    ``_ob_lifecycle`` (depth-0: any wick reaching the near edge is in-zone), then
    labels each maximal in-zone run and records the deepest price reached. It
    never changes status/touch_count — it only describes what happened.

    Each item: ``{"entry_idx", "level", "outcome"}`` with outcome ∈
    {edge_touch, entry_exit, traversal, inside}. ``edge_touch`` vs ``entry_exit``
    splits on ``policy.contact_edge_touch_fraction`` of the height. A close-through
    bar ends the ledger with a single ``traversal`` contact (the zone is consumed
    there), mirroring ``_ob_lifecycle``'s invalidation.
    """
    height = max(zhigh - zlow, 0.0)
    edge_frac = policy.contact_edge_touch_fraction
    contacts: list[dict] = []
    in_zone = False
    entry_idx: Optional[int] = None
    deepest_pen = 0.0
    for j in range(created + 1, upto + 1):
        if policy.ob_invalidate_on_close_through and (
            (side == "bullish" and closes[j] < zlow) or (side != "bullish" and closes[j] > zhigh)
        ):
            start = entry_idx if entry_idx is not None else j
            far = zlow if side == "bullish" else zhigh
            contacts.append({"entry_idx": start, "level": float(far), "outcome": "traversal"})
            return contacts
        if side == "bullish":
            tap = lows[j] <= zhigh and highs[j] >= zlow
            pen = min(max(zhigh - float(lows[j]), 0.0), height)
        else:
            tap = highs[j] >= zlow and lows[j] <= zhigh
            pen = min(max(float(highs[j]) - zlow, 0.0), height)
        if tap:
            if not in_zone:
                in_zone = True
                entry_idx = j
                deepest_pen = 0.0
            deepest_pen = max(deepest_pen, pen)
        elif in_zone:
            frac = (deepest_pen / height) if height > 0 else 0.0
            outcome = "entry_exit" if frac >= edge_frac else "edge_touch"
            level = (zhigh - deepest_pen) if side == "bullish" else (zlow + deepest_pen)
            contacts.append({"entry_idx": entry_idx, "level": float(level), "outcome": outcome})
            in_zone = False
            entry_idx = None
            deepest_pen = 0.0
    if in_zone and entry_idx is not None:
        level = (zhigh - deepest_pen) if side == "bullish" else (zlow + deepest_pen)
        contacts.append({"entry_idx": entry_idx, "level": float(level), "outcome": "inside"})
    return contacts


def _fvg_contacts(
    side: str,
    zhigh: float,
    zlow: float,
    highs: Any,
    lows: Any,
    created: int,
    upto: int,
    policy: MitigationPolicy = MITIGATION_POLICY,
) -> list[dict]:
    """Per-contact ledger for a fair-value gap over bars (created, upto].

    Twin of :func:`_ob_contacts` using the FVG entry/fill predicates of
    ``_fvg_lifecycle`` (entry = a wick past the near edge; ``traversal`` = the
    wick retraced ``policy.fvg_fill_fraction`` of the height = fully filled).
    """
    height = max(zhigh - zlow, 0.0)
    fill = policy.fvg_fill_fraction * height
    edge_frac = policy.contact_edge_touch_fraction
    contacts: list[dict] = []
    in_band = False
    entry_idx: Optional[int] = None
    deepest_pen = 0.0
    for j in range(created + 1, upto + 1):
        if side == "bullish":
            filled_now = lows[j] <= zhigh - fill
            entry = lows[j] <= zhigh
            pen = min(max(zhigh - float(lows[j]), 0.0), height)
        else:
            filled_now = highs[j] >= zlow + fill
            entry = highs[j] >= zlow
            pen = min(max(float(highs[j]) - zlow, 0.0), height)
        if filled_now:
            start = entry_idx if entry_idx is not None else j
            far = zlow if side == "bullish" else zhigh
            contacts.append({"entry_idx": start, "level": float(far), "outcome": "traversal"})
            return contacts
        if entry:
            if not in_band:
                in_band = True
                entry_idx = j
                deepest_pen = 0.0
            deepest_pen = max(deepest_pen, pen)
        elif in_band:
            frac = (deepest_pen / height) if height > 0 else 0.0
            outcome = "entry_exit" if frac >= edge_frac else "edge_touch"
            level = (zhigh - deepest_pen) if side == "bullish" else (zlow + deepest_pen)
            contacts.append({"entry_idx": entry_idx, "level": float(level), "outcome": outcome})
            in_band = False
            entry_idx = None
            deepest_pen = 0.0
    if in_band and entry_idx is not None:
        level = (zhigh - deepest_pen) if side == "bullish" else (zlow + deepest_pen)
        contacts.append({"entry_idx": entry_idx, "level": float(level), "outcome": "inside"})
    return contacts


def _ob_origin(
    enriched: Any,
    side: str,
    created_k: int,
    upto: int,
    max_ahead: int = 12,
) -> Optional[dict]:
    """Associate an order block with the structural break it PRECEDES: the first
    same-direction ``BOS_EVENT`` (a ``CHOCH_SIGNAL`` bar refines the kind label)
    within ``max_ahead`` bars of the OB's formation. Read from engine event
    columns only — no detection, no recompute. Returns
    ``{"kind","direction","at","level"}`` or None (no fabricated origin)."""
    import pandas as pd

    cols = set(enriched.columns)
    if "BOS_EVENT" not in cols or "BOS_BREAK_LEVEL" not in cols:
        return None
    bos_ev = enriched["BOS_EVENT"].values
    lvl = enriched["BOS_BREAK_LEVEL"].values
    choch = enriched["CHOCH_SIGNAL"].values if "CHOCH_SIGNAL" in cols else None
    want_bull = side == "bullish"
    end = min(created_k + max_ahead, upto)
    for j in range(created_k, end + 1):
        ev = bos_ev[j]
        if pd.isna(ev) or ev == 0:
            continue
        if (ev > 0) != want_bull:
            continue
        at = _zone_created_at(enriched, j)
        if at is None or pd.isna(lvl[j]):
            continue
        is_choch = choch is not None and not pd.isna(choch[j]) and choch[j] != 0
        return {
            "kind": "choch" if is_choch else "bos",
            "direction": side,
            "at": at,
            "level": float(lvl[j]),
        }
    return None


def collect_zones(
    enriched: Any,
    idx: int = -1,
    max_per_type: Optional[int] = None,
    with_rejects: bool = False,
) -> dict[str, list[dict]]:
    """Collect every still-relevant OB / FVG zone up to bar ``idx``.

    Returns ``{"order_blocks": [...], "fair_value_gaps": [...]}`` as plain dicts
    (the structure mapper builds the pydantic models, filling ``created_at`` from
    ``bar_ts`` when the frame has no datetime index). Consumed zones (invalidated
    OB, filled FVG) are dropped. Ordering: active before partially-consumed, then
    by strength/size, then by recency; capped to ``max_per_type`` (defaults to the
    ``MAX_ZONES_PER_TYPE`` env var, else the module constant).

    ``with_rejects=True`` additionally returns ``rejected_order_blocks``: the OB
    the engine DID detect but does not surface, each carrying the reason emitted
    by the very branch that dropped it (``invalidated_close_through`` from the
    lifecycle, ``mitigated_dropped_by_policy`` from the policy flag,
    ``capped_max_zones`` from the sort/cap). The surfaced lists are byte-identical
    with the flag on or off — the flag only keeps what was already discarded
    (rejection-diagnostics mission 2026-07-02; never persisted, never mapped).
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
        ob_rejects: list[dict] = []
        ob_consumed: list[dict] = []
        for k in range(pos + 1):
            for side, hv, lv in (
                ("bullish", bull_hi[k], bull_lo[k]),
                ("bearish", bear_hi[k], bear_lo[k]),
            ):
                if pd.isna(hv) or pd.isna(lv):
                    continue
                zhigh, zlow = float(max(hv, lv)), float(min(hv, lv))
                st = float(strength[k]) if strength is not None and not pd.isna(strength[k]) else 0.0
                status, tested, tap_idx, invalidated_idx, touch_count, touch_bars = _ob_lifecycle(
                    side, zhigh, zlow, highs, lows, closes, k, pos
                )
                created_at = _zone_created_at(enriched, k)
                # VZ-1: the per-contact ledger + the origin break, both read-side.
                contacts = [
                    {"at": _zone_created_at(enriched, c["entry_idx"]), "level": c["level"], "outcome": c["outcome"]}
                    for c in _ob_contacts(side, zhigh, zlow, highs, lows, closes, k, pos)
                ]
                origin = _ob_origin(enriched, side, k, pos)
                zone = {
                    "direction": side,
                    "level_high": zhigh,
                    "level_low": zlow,
                    "importance": "high" if st >= 0.75 else "medium" if st >= 0.4 else "low",
                    "status": status,
                    "tested": tested,
                    "created_at": created_at,
                    "mitigated_at": (
                        _zone_created_at(enriched, tap_idx) if tap_idx is not None else None
                    ),
                    "touch_count": touch_count,
                    "touch_ats": [_zone_created_at(enriched, b) for b in touch_bars],
                    "contacts": contacts,
                    "origin": origin,
                    "_strength": st,
                    "_k": k,
                }
                # Honesty guardrail (mission §C): never surface a consumed zone in
                # the live list. VZ-1: a bounded set of consumed OB is kept SEPARATELY
                # (consumed_order_blocks) so the /zones page can show a « Comblées »
                # (traversée) group — never mixed into order_blocks.
                if status == "invalidated":
                    zone["invalidated_at"] = (
                        _zone_created_at(enriched, invalidated_idx)
                        if invalidated_idx is not None else None
                    )
                    ob_consumed.append(zone)
                    if with_rejects:
                        # Shallow copy so the reject stream keeps its own reason key
                        # without perturbing the consumed-zone twin.
                        rej = dict(zone)
                        rej["reject_reason"] = "invalidated_close_through"
                        ob_rejects.append(rej)
                    continue
                if status == "mitigated" and MITIGATION_POLICY.ob_drop_when_mitigated:
                    if with_rejects:
                        zone["reject_reason"] = "mitigated_dropped_by_policy"
                        ob_rejects.append(zone)
                    continue
                obs.append(zone)
        # active first, then by strength, then most recent first.
        obs.sort(key=lambda z: (z["status"] != "active", -z["_strength"], -z["_k"]))
        out["order_blocks"] = obs[:max_per_type]
        # Most recently CONSUMED first (by formation recency), bounded.
        ob_consumed.sort(key=lambda z: -z["_k"])
        out["consumed_order_blocks"] = ob_consumed[:_max_consumed()]
        if with_rejects:
            # Overflow of the SAME sorted list the cap truncates: detected,
            # alive, but ranked beyond max_per_type → not displayed.
            for rank, zone in enumerate(obs[max_per_type:], start=max_per_type):
                zone["reject_reason"] = "capped_max_zones"
                zone["cap_rank"] = rank
                zone["cap_max"] = max_per_type
                ob_rejects.append(zone)
            out["rejected_order_blocks"] = ob_rejects

    # ---- Fair value gaps -------------------------------------------------
    if "FVG_DIR" in cols and {"high", "low"} <= cols:
        fvg_dir = enriched["FVG_DIR"].values
        size_norm = (
            enriched["FVG_SIZE_NORM"].values if "FVG_SIZE_NORM" in cols else None
        )
        fvgs: list[dict] = []
        fvg_consumed: list[dict] = []
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
            status, tested, entry_idx, fill_level, touch_count, touch_bars = _fvg_lifecycle(
                side, zhigh, zlow, highs, lows, k, pos
            )
            sz = float(size_norm[k]) if size_norm is not None and not pd.isna(size_norm[k]) else (zhigh - zlow)
            contacts = [
                {"at": _zone_created_at(enriched, c["entry_idx"]), "level": c["level"], "outcome": c["outcome"]}
                for c in _fvg_contacts(side, zhigh, zlow, highs, lows, k, pos)
            ]
            zone = {
                "direction": side,
                "level_high": zhigh,
                "level_low": zlow,
                "status": status,
                "tested": tested,
                "created_at": _zone_created_at(enriched, k),
                "mitigated_at": _zone_created_at(enriched, entry_idx) if entry_idx is not None else None,
                "fill_level": fill_level,
                "touch_count": touch_count,
                "touch_ats": [_zone_created_at(enriched, b) for b in touch_bars],
                "contacts": contacts,
                "_size": sz,
                "_k": k,
            }
            # Honesty guardrail (mission §C): a filled gap leaves the live list but
            # is kept in the bounded consumed set for the « Comblées » group (VZ-1).
            if status == "filled":
                fvg_consumed.append(zone)
                continue
            if status == "partially_filled" and MITIGATION_POLICY.fvg_drop_when_partial:
                continue
            fvgs.append(zone)
        fvgs.sort(key=lambda z: (z["status"] != "active", -z["_size"], -z["_k"]))
        out["fair_value_gaps"] = fvgs[:max_per_type]
        fvg_consumed.sort(key=lambda z: -z["_k"])
        out["consumed_fair_value_gaps"] = fvg_consumed[:_max_consumed()]

    return out


def collect_zone_lifecycles(
    enriched: Any,
    idx: int = -1,
    since_ts: Optional[datetime] = None,
    until_ts: Optional[datetime] = None,
) -> list[dict]:
    """Census of EVERY zone whose FORMATION bar falls in ``(since_ts, until_ts]``,
    with its creation + first-mitigation timestamps, kind and status — INCLUDING
    zones the display layer drops once consumed (``collect_zones`` deliberately
    hides mitigated/filled/invalidated zones; a lifecycle census needs exactly
    those). Read-only: it never mutates the frame and never changes what the live
    display surfaces — it reuses the SAME engine columns and the SAME lifecycle
    predicates as ``collect_zones``, only keeping what that function discards.

    Each item: ``{"kind": "ob"|"fvg", "direction", "created_at", "mitigated_at",
    "status"}`` where ``mitigated_at`` is the first bar price returned into the
    zone (an OB tap / an FVG entry), or ``None`` if it was never re-touched within
    the window. Used only by the publication zone-lifecycle measure (NW-7)."""
    import pandas as pd

    out: list[dict] = []
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

    def _in_window(created_at: Optional[datetime]) -> bool:
        if created_at is None:
            return False
        if since_ts is not None and created_at < since_ts:
            return False
        if until_ts is not None and created_at > until_ts:
            return False
        return True

    # ---- Order blocks (kept regardless of mitigation/invalidation) -------
    ob_cols = {"BULLISH_OB_HIGH", "BULLISH_OB_LOW", "BEARISH_OB_HIGH", "BEARISH_OB_LOW"}
    if ob_cols <= cols:
        bull_hi = enriched["BULLISH_OB_HIGH"].values
        bull_lo = enriched["BULLISH_OB_LOW"].values
        bear_hi = enriched["BEARISH_OB_HIGH"].values
        bear_lo = enriched["BEARISH_OB_LOW"].values
        for k in range(pos + 1):
            created_at = _zone_created_at(enriched, k)
            if not _in_window(created_at):
                continue
            for side, hv, lv in (
                ("bullish", bull_hi[k], bull_lo[k]),
                ("bearish", bear_hi[k], bear_lo[k]),
            ):
                if pd.isna(hv) or pd.isna(lv):
                    continue
                zhigh, zlow = float(max(hv, lv)), float(min(hv, lv))
                status, _tested, tap_idx, _inv, _tc, _tb = _ob_lifecycle(
                    side, zhigh, zlow, highs, lows, closes, k, pos
                )
                out.append({
                    "kind": "ob",
                    "direction": side,
                    "created_at": created_at,
                    "mitigated_at": _zone_created_at(enriched, tap_idx) if tap_idx is not None else None,
                    "status": status,
                })

    # ---- Fair value gaps (kept regardless of fill) -----------------------
    if "FVG_DIR" in cols and {"high", "low"} <= cols:
        fvg_dir = enriched["FVG_DIR"].values
        for k in range(2, pos + 1):
            d = fvg_dir[k]
            if pd.isna(d) or d == 0:
                continue
            created_at = _zone_created_at(enriched, k)
            if not _in_window(created_at):
                continue
            if d > 0:
                a, b = float(highs[k - 2]), float(lows[k])
                side = "bullish"
            else:
                a, b = float(highs[k]), float(lows[k - 2])
                side = "bearish"
            zhigh, zlow = max(a, b), min(a, b)
            status, _entered, entry_idx, _fill, _tc, _tb = _fvg_lifecycle(
                side, zhigh, zlow, highs, lows, k, pos
            )
            out.append({
                "kind": "fvg",
                "direction": side,
                "created_at": created_at,
                "mitigated_at": _zone_created_at(enriched, entry_idx) if entry_idx is not None else None,
                "status": status,
            })

    return out


def collect_structure_events(
    enriched: Any,
    idx: int = -1,
    max_per_type: Optional[int] = None,
) -> dict[str, list[dict]]:
    """Collect discrete BOS / CHOCH break EVENTS over the window up to ``idx``.

    >>> THE SINGLE ARBITRATION AUTHORITY (STR-2 defect B) <<<
    THE RULE, in one sentence a trader understands: « sur une bougie qui casse la
    structure dans le sens opposé à la tendance en cours, l'événement est un
    changement de caractère (CHOCH) ; il n'y a jamais, sur cette même bougie, un
    BOS de même sens en plus. » This function is the ONE place that maps the raw
    engine columns to structural journal events, so this is where the rule is
    decided — not scattered across the display. It is EXPLICIT (a ``CHOCH_SIGNAL``
    bar is dropped from ``bos_events`` below), never an effect of execution order.
    The point-in-time mapper and the chart markers apply the SAME rule; the engine
    itself keeps ``BOS_EVENT`` set on a reversal bar only as the retest machine's
    break trigger (see strategy_features._calculate_bos_choch_*), never as a second
    BOS structural event.

    Reads ONLY engine-produced event columns — ``BOS_EVENT`` (±1 on a true break
    bar), ``CHOCH_SIGNAL`` (±1 on a reversal bar) and ``BOS_BREAK_LEVEL`` (the
    broken level on those bars). No detection, no recompute, no threshold. This
    is the structure-event twin of :func:`collect_zones`: the engine detects many
    breaks but only the LAST bar's one ever surfaced via ``bos``/``choch`` (audit
    2026-06-16 "sous-surfaçage": 88 BOS / 40 CHOCH detected over 6 combos, ≤1
    surfaced — a pure plumbing gap). Returns the most recent events first, capped
    to ``max_per_type``. Uses the discrete ``BOS_EVENT`` (real break bars,
    ~11-25 / 500 bars), NEVER the propagated ``BOS_SIGNAL`` that the F6 fix proved
    fires on ~100% of bars.

    STR-1 CHOCH precedence: a reversal bar sets BOTH ``CHOCH_SIGNAL`` and
    ``BOS_EVENT`` on the same bar (a CHOCH *is* a reversal break — see
    ``strategy_features._calculate_bos_choch_numba`` lines 113-128, where the
    CHOCH branch also writes ``bos_event[i]``). That is ONE structural event — a
    change of character — recorded in two columns, NOT two events. Surfacing the
    ``BOS_EVENT`` twin in ``bos_events`` produced a contradictory same-bar,
    same-direction "BOS + CHOCH" pair in the journal and an ambiguous focus
    target (audit ``AUDIT-str-1-bos-choch.md``). We therefore drop, from
    ``bos_events``, every bar that also carries a ``CHOCH_SIGNAL``: the CHOCH row
    is the SMC-correct single event. This mirrors the chart-marker dedup already
    in ``webapp/lib/chart/structureMarkers.ts`` ("CHOCH wins a shared bar"), so
    the journal and the chart now apply the same rule from a single source.
    """
    import os
    import pandas as pd

    if max_per_type is None:
        try:
            max_per_type = int(os.environ.get("MAX_STRUCTURE_EVENTS", MAX_STRUCTURE_EVENTS))
        except (TypeError, ValueError):
            max_per_type = MAX_STRUCTURE_EVENTS

    out: dict[str, list[dict]] = {"bos_events": [], "choch_events": []}
    n = len(enriched)
    if n == 0:
        return out
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return out

    cols = set(enriched.columns)
    closes = enriched["close"].values if "close" in cols else None
    break_level = enriched["BOS_BREAK_LEVEL"].values if "BOS_BREAK_LEVEL" in cols else None

    def _level(k: int) -> Optional[float]:
        # Real broken level on the event bar; close is the last-resort fallback.
        if break_level is not None and not pd.isna(break_level[k]):
            return float(break_level[k])
        if closes is not None and not pd.isna(closes[k]):
            return float(closes[k])
        return None

    choch_values = enriched["CHOCH_SIGNAL"].values if "CHOCH_SIGNAL" in cols else None

    def _collect(col: str, drop_choch_bars: bool = False) -> list[dict]:
        if col not in cols:
            return []
        values = enriched[col].values
        events: list[dict] = []
        for k in range(pos + 1):
            v = values[k]
            if pd.isna(v) or v == 0:
                continue
            # STR-1 CHOCH precedence: skip a BOS_EVENT bar that is also a CHOCH —
            # the reversal is a single change-of-character event, surfaced once as
            # a CHOCH (never duplicated as a BOS on the same bar).
            if (
                drop_choch_bars
                and choch_values is not None
                and not pd.isna(choch_values[k])
                and choch_values[k] != 0
            ):
                continue
            lvl = _level(k)
            if lvl is None:
                continue
            events.append({
                "direction": "bullish" if v > 0 else "bearish",
                "level": lvl,
                "broken_at": _zone_created_at(enriched, k),
                # Real ANALYSED-bar distance from the event to the last bar of the
                # window (DG-1 point 2). Lets the front show maturity in bougies
                # truly analysed — not wall-clock hours ÷ tf, which overcounts the
                # week-end gap (a Fri→Mon "114 bougies" that were never analysed).
                "bars_ago": pos - k,
                "_k": k,
            })
        events.sort(key=lambda e: -e["_k"])  # most recent first
        return events[:max_per_type]

    out["bos_events"] = _collect("BOS_EVENT", drop_choch_bars=True)
    out["choch_events"] = _collect("CHOCH_SIGNAL")
    return out


# ---------------------------------------------------------------------------
# External liquidity pools (EQH/EQL + range extremes) — descriptive twin of
# collect_zones. Reuses the engine's EXISTING swing fractals (UP_FRACTAL /
# DOWN_FRACTAL); detects NOTHING new and touches no BOS/CHOCH/OB/FVG rule.
#
# Honesty / no-look-ahead: a fractal column is causal (shifted to its
# confirmation bar), so the value at bar k is the swing price first KNOWABLE at
# k. A pocket's lifecycle is scanned from the bar AFTER its FIRST constituent
# swing (``first_k``) — the bar the level first exists and is knowable — through
# the read bar ``pos``, EXACTLY like the OB/FVG lifecycle (which scans from the
# zone's formation bar). "No look-ahead" means we never use bars beyond ``pos``;
# it does NOT mean we ignore a breach that occurred between the first and last
# swing of the cluster. A close net through the published level is a past,
# observable fact at read time and must flip the state — the pocket is DISPLAYED
# from ``first_k`` (``created_at``), so it must be JUDGED from ``first_k`` too, or
# it would claim "intact" over a span it never evaluated (LQ-D1: the two anchors
# had diverged — displayed from first_k, judged from last_k — so a plunge between
# them stayed invisible and the pocket lied "intacte"). The output is purely
# factual: WHERE the pocket sits and WHETHER it is intact / swept / broken. No
# target, draw, bias or probability is ever produced (mission §0 inviolable line).
# ---------------------------------------------------------------------------


def _pool_lifecycle(
    side: str,
    level: float,
    highs: Any,
    lows: Any,
    closes: Any,
    scan_from: int,
    upto: int,
) -> tuple[str, Optional[int], Optional[int]]:
    """Classify a liquidity pocket over bars (scan_from, upto].

    Returns ``(status, swept_idx, broken_idx)`` where status ∈
    {intact, swept, broken}:
      * broken — a later bar CLOSED net through ``level`` (close > level for a
        buy-side pocket, close < level for sell-side). Terminal: the resting
        liquidity at that level is gone. ``broken_idx`` = first such bar.
      * swept  — a later bar's WICK pierced ``level`` and the bar CLOSED back
        inside (high > level but close ≤ level for buy-side; mirror for
        sell-side). A liquidity-grab event. ``swept_idx`` = first such bar.
      * intact — price has not traded through ``level`` yet.

    A pocket may be swept first and broken later; ``broken`` wins (terminal) but
    ``swept_idx`` is retained. Strict comparisons mirror the OB close-through
    convention (``_ob_lifecycle``); no extra threshold is introduced.
    """
    swept_idx: Optional[int] = None
    broken_idx: Optional[int] = None
    for j in range(scan_from + 1, upto + 1):
        if side == "bsl":  # liquidity resting ABOVE the level
            if closes[j] > level:
                broken_idx = j
                break
            if highs[j] > level and closes[j] <= level and swept_idx is None:
                swept_idx = j
        else:  # "ssl" — liquidity resting BELOW the level
            if closes[j] < level:
                broken_idx = j
                break
            if lows[j] < level and closes[j] >= level and swept_idx is None:
                swept_idx = j
    status = "broken" if broken_idx is not None else ("swept" if swept_idx is not None else "intact")
    return status, swept_idx, broken_idx


def _cluster_swings(
    points: list[tuple[int, float]], eps: float, extreme: str
) -> list[dict]:
    """Cluster swing points whose prices fall within ``eps`` into pockets.

    ``points`` = list of ``(bar_index, price)``. ``extreme`` ∈ {"max", "min"}
    selects the pocket level (founder decision: the cluster EXTREME — the highest
    high for buy-side, the lowest low for sell-side — i.e. the truly breachable
    edge). Greedy on price-sorted points: a new point joins the open cluster while
    it stays within ``eps`` of the cluster's running extreme, else it seeds a new
    cluster. Returns one dict per cluster with level, touches, first/last bar.
    """
    if not points:
        return []
    # Sort by price: descending for highs (max extreme), ascending for lows.
    pts = sorted(points, key=lambda p: p[1], reverse=(extreme == "max"))
    clusters: list[list[tuple[int, float]]] = []
    current: list[tuple[int, float]] = [pts[0]]
    ref = pts[0][1]
    for k, price in pts[1:]:
        if abs(price - ref) <= eps:
            current.append((k, price))
            # Running extreme so a drifting chain stays anchored to the edge.
            ref = max(ref, price) if extreme == "max" else min(ref, price)
        else:
            clusters.append(current)
            current = [(k, price)]
            ref = price
    clusters.append(current)

    out: list[dict] = []
    for cl in clusters:
        prices = [p[1] for p in cl]
        idxs = [p[0] for p in cl]
        level = max(prices) if extreme == "max" else min(prices)
        out.append({
            "level": float(level),
            "touches": len(cl),
            "first_k": min(idxs),
            "last_k": max(idxs),
            # The cluster's constituent swings, kept so the caller can split a
            # price-cluster that straddles a net close-through of the level into
            # separate pockets (LQ-D1 §clustering: two "equal" swings separated by
            # a break are NOT one pocket — the level ceded between them).
            "points": list(cl),
        })
    return out


def _split_pocket_at_breaks(
    points: list[tuple[int, float]],
    side: str,
    closes: Any,
    level: float,
) -> list[list[tuple[int, float]]]:
    """Split a price-cluster's swings into time-ordered runs, cutting wherever a
    bar CLOSED net through ``level`` strictly between two consecutive swings.

    A liquidity pocket is a run of equal-price swings whose shared level was NOT
    closed through while it was forming. If price closes beyond the level between
    two members, the resting liquidity there was taken — the earlier swings are a
    (now broken) pocket and the later swings begin a fresh one. ``level`` is the
    whole cluster's breachable edge (min for sell-side, max for buy-side): a close
    beyond THAT is an unambiguous break, so genuine equal levels (whose intra-
    cluster closes never breach the edge) are never over-split. The forward life
    of each run — sweeps/breaks AFTER its last swing — is still timed by
    :func:`_pool_lifecycle`; this only separates already-broken segments.
    """
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[0])  # chronological
    runs: list[list[tuple[int, float]]] = []
    run: list[tuple[int, float]] = [pts[0]]
    for k, price in pts[1:]:
        prev_k = run[-1][0]
        broke = False
        for j in range(prev_k + 1, k):
            if (side == "bsl" and closes[j] > level) or (
                side == "ssl" and closes[j] < level
            ):
                broke = True
                break
        if broke:
            runs.append(run)
            run = [(k, price)]
        else:
            run.append((k, price))
    runs.append(run)
    return runs


def collect_liquidity_pools(
    enriched: Any,
    idx: int = -1,
    *,
    eq_tolerance_atr: float = 0.10,
    eq_tolerance_pips_floor: float = 0.0,
    eq_min_touches: int = 2,
    lookback: int = 200,
    max_pools: Optional[int] = None,
) -> list[dict]:
    """Collect external liquidity pockets up to bar ``idx`` (most relevant first).

    Aggregates the engine's existing swing fractals into buy-side (BSL) and
    sell-side (SSL) pockets and times each pocket's intact/swept/broken state.
    Pocket kinds: ``equal_highs`` / ``equal_lows`` (≥ ``eq_min_touches`` swings
    within tolerance) and ``range_high`` / ``range_low`` (the window's extreme
    swing, emitted as a lone pocket only when no equal-cluster already sits at
    that extreme — avoids a duplicate at the same level).

    ``is_external`` = the pocket sits at/beyond the current range's extreme swing
    (buy-side ≥ range high − eps, sell-side ≤ range low + eps); range extremes are
    external by construction. Tolerance ``eps`` = max(``eq_tolerance_atr``×ATR,
    ``eq_tolerance_pips_floor``), ATR read at the read bar. Returns plain dicts;
    the structure mapper builds the pydantic models. Read-only — no engine column
    is written, no detection rule altered.
    """
    import os
    import pandas as pd

    if max_pools is None:
        try:
            max_pools = int(os.environ.get("MAX_LIQUIDITY_POOLS", MAX_LIQUIDITY_POOLS))
        except (TypeError, ValueError):
            max_pools = MAX_LIQUIDITY_POOLS

    n = len(enriched)
    if n == 0:
        return []
    pos = idx if idx >= 0 else n + idx
    if pos < 0 or pos >= n:
        return []

    cols = set(enriched.columns)
    if not ({"UP_FRACTAL", "DOWN_FRACTAL", "high", "low", "close"} <= cols):
        return []

    highs = enriched["high"].values
    lows = enriched["low"].values
    closes = enriched["close"].values
    up_fr = enriched["UP_FRACTAL"].values
    dn_fr = enriched["DOWN_FRACTAL"].values

    atr = 0.0
    if "ATR" in cols:
        a = enriched["ATR"].values[pos]
        atr = float(a) if not pd.isna(a) else 0.0
    eps = max(atr * float(eq_tolerance_atr), float(eq_tolerance_pips_floor))
    if eps <= 0.0:  # degenerate ATR and no floor → use a hair of price to avoid 0-width clusters
        eps = abs(float(closes[pos])) * 1e-4 if closes[pos] else 1e-9

    lo_bound = max(0, pos - int(lookback) + 1)
    # Collect confirmed swing points within the window. The fractal column value
    # IS the swing price; the bar index is the confirmation bar (first knowable).
    sh: list[tuple[int, float]] = []  # swing highs
    sl: list[tuple[int, float]] = []  # swing lows
    for k in range(lo_bound, pos + 1):
        v = up_fr[k]
        if not pd.isna(v) and v > 0:
            sh.append((k, float(v)))
        v = dn_fr[k]
        if not pd.isna(v) and v > 0:
            sl.append((k, float(v)))

    pools: list[dict] = []
    range_high = max((p[1] for p in sh), default=None)
    range_low = min((p[1] for p in sl), default=None)

    def _emit(side: str, kind: str, level: float, touches: int,
              first_k: int, last_k: int, is_external: bool) -> None:
        # LQ-D1 root-cause fix: judge from ``first_k`` (formation / where the
        # level first exists and is displayed), NOT ``last_k``. Scanning from
        # last_k left the interval (first_k, last_k] drawn but never evaluated,
        # so a close net through the level in that span stayed invisible and the
        # pocket falsely read "intacte". Now display anchor == evaluation anchor
        # == first_k, exactly like the OB/FVG lifecycle.
        status, swept_k, broken_k = _pool_lifecycle(
            side, level, highs, lows, closes, scan_from=first_k, upto=pos
        )
        pools.append({
            "side": side,
            "kind": kind,
            "level": float(level),
            "touches": int(touches),
            "is_external": bool(is_external),
            "status": status,
            "created_at": _zone_created_at(enriched, first_k),
            "swept_at": _zone_created_at(enriched, swept_k) if swept_k is not None else None,
            "broken_at": _zone_created_at(enriched, broken_k) if broken_k is not None else None,
            "_first_k": first_k,
            "_last_k": last_k,
        })

    # --- Buy-side (equal highs) -----------------------------------------------
    # Each price-cluster is split into time-runs at any net close-through of its
    # level (LQ-D1): a run is a genuine equal-highs pocket only while the level
    # held. A run seen fewer than eq_min_touches times is no longer "equal highs".
    top_cluster_external = False
    for cl in _cluster_swings(sh, eps, "max"):
        for run in _split_pocket_at_breaks(cl["points"], "bsl", closes, cl["level"]):
            if len(run) < int(eq_min_touches):
                continue
            r_level = max(p[1] for p in run)
            r_first = min(p[0] for p in run)
            r_last = max(p[0] for p in run)
            is_ext = range_high is not None and r_level >= range_high - eps
            if is_ext:
                top_cluster_external = True
            _emit("bsl", "equal_highs", r_level, len(run), r_first, r_last, is_ext)
    # Range high as a lone external pocket only if no equal-cluster holds the top.
    if range_high is not None and not top_cluster_external:
        at_top = [p for p in sh if p[1] >= range_high - eps]
        _emit("bsl", "range_high", range_high, len(at_top),
              min(p[0] for p in at_top), max(p[0] for p in at_top), True)

    # --- Sell-side (equal lows) -----------------------------------------------
    bot_cluster_external = False
    for cl in _cluster_swings(sl, eps, "min"):
        for run in _split_pocket_at_breaks(cl["points"], "ssl", closes, cl["level"]):
            if len(run) < int(eq_min_touches):
                continue
            r_level = min(p[1] for p in run)
            r_first = min(p[0] for p in run)
            r_last = max(p[0] for p in run)
            is_ext = range_low is not None and r_level <= range_low + eps
            if is_ext:
                bot_cluster_external = True
            _emit("ssl", "equal_lows", r_level, len(run), r_first, r_last, is_ext)
    if range_low is not None and not bot_cluster_external:
        at_bot = [p for p in sl if p[1] <= range_low + eps]
        _emit("ssl", "range_low", range_low, len(at_bot),
              min(p[0] for p in at_bot), max(p[0] for p in at_bot), True)

    # External first, intact before swept before broken, then most recent first.
    _status_rank = {"intact": 0, "swept": 1, "broken": 2}
    pools.sort(key=lambda z: (
        not z["is_external"],
        _status_rank.get(z["status"], 3),
        -z["_last_k"],
    ))
    return pools[:max_pools]


# ---------------------------------------------------------------------------
# Structure mapper
# ---------------------------------------------------------------------------


def confluence_signal_to_structure(
    confluence_signal: Optional[Any],
    smc_features: dict[str, float],
    bar_ts: datetime,
    current_price: float,
    instrument: Optional[str] = None,
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
    # STR-1 CHOCH precedence: a reversal bar carries BOTH BOS_EVENT and
    # CHOCH_SIGNAL (a CHOCH is a reversal break). It is a change of character, not
    # a BOS — never surface the point-in-time BOS twin on such a bar (it is
    # published below from CHOCH_SIGNAL as `choch`). A *persisted* break — the
    # retest of an EARLIER, non-CHOCH BOS — is deliberately unaffected: it is
    # legitimately a BOS being retested, not the current reversal.
    choch_event = float(smc_features.get("CHOCH_SIGNAL", 0.0))
    fresh_break = abs(bos_event) > 0 and choch_event == 0
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
            id=_event_id("bos", event_direction, broken_at),
            direction=event_direction,
            level=bos_level if bos_level is not None else float(current_price),
            broken_at=broken_at,
            validation_status="confirmed",  # broke and not invalidated ⇒ confirmed
        )

    # CHOCH
    choch: Optional[CHOCHRecent] = None
    choch_signal = choch_event  # same CHOCH_SIGNAL read (STR-1 precedence above)
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
            id=_event_id("choch", choch_direction, bar_ts),
            direction=choch_direction,
            level=choch_level if choch_level is not None else float(current_price),
            broken_at=bar_ts,
            validation_status="confirmed",
        )

    # Discrete BOS/CHOCH break-event history (most-recent first, capped). Like
    # the multi-zone registry, injected by the SMC pipeline under
    # ``_structure_events``; absent on callers/tests that don't run the collector.
    structure_events = smc_features.get("_structure_events")
    bos_events, choch_events = (
        _structure_events_to_models(structure_events, bar_ts)
        if isinstance(structure_events, dict)
        else ([], [])
    )

    # External liquidity pockets (EQH/EQL + range extremes). Twin of the multi-
    # zone registry: injected by the SMC pipeline under ``_liquidity``; absent on
    # callers/tests that don't run collect_liquidity_pools → empty list.
    liquidity = smc_features.get("_liquidity")
    liquidity_pools = (
        _liquidity_to_models(liquidity, bar_ts)
        if isinstance(liquidity, list)
        else []
    )

    # Order blocks + fair value gaps.
    # Preferred path: the multi-zone registry (all still-relevant zones the
    # engine computed over the window, with lifecycle), injected by the SMC
    # pipeline under ``_zones``. Fallback: the legacy single-last-bar zone, kept
    # so callers/tests that don't run collect_zones still behave as before.
    zones = smc_features.get("_zones")
    if isinstance(zones, dict):
        order_blocks, fair_value_gaps = _zones_to_models(zones, bar_ts, instrument)
        consumed_obs, consumed_fvgs = _consumed_zones_to_models(zones, bar_ts, instrument)
        return MarketReadingStructure(
            current_bos=bos,
            current_choch=choch,
            bos_events=bos_events,
            choch_events=choch_events,
            order_blocks=order_blocks,
            fair_value_gaps=fair_value_gaps,
            consumed_order_blocks=consumed_obs,
            consumed_fair_value_gaps=consumed_fvgs,
            liquidity_pools=liquidity_pools,
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
        current_bos=bos,
        current_choch=choch,
        bos_events=bos_events,
        choch_events=choch_events,
        order_blocks=order_blocks,
        fair_value_gaps=fair_value_gaps,
        liquidity_pools=liquidity_pools,
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


def ob_zone_id(direction: str, created_at: datetime) -> str:
    """Stable OB id (direction + creation time) — the ONLY place the format is
    defined. Shared by the reading models below and the rejection diagnostics so
    both always name the same zone the same way."""
    return f"OB_{direction}_{created_at.strftime('%Y%m%d%H%M%S')}"


def _session_of(instrument: Optional[str], created: datetime) -> Optional[str]:
    """VZ-1 formation session, canonical (market_calendar). None when no
    instrument is threaded (older/legacy callers) — the frontend then falls back
    to its client mirror. Defensive: a tz lookup failure never breaks a reading."""
    if not instrument:
        return None
    try:
        from .market_calendar import session_at

        return session_at(instrument, created)
    except Exception:  # pragma: no cover — session is cosmetic, never fatal
        return None


def _zones_to_models(
    zones: dict[str, list[dict]],
    bar_ts: datetime,
    instrument: Optional[str] = None,
) -> tuple[list[OrderBlock], list[FairValueGap]]:
    """Convert collected zone dicts (from :func:`collect_zones`) to schema models.

    ``created_at`` falls back to ``bar_ts`` when the collector could not derive a
    per-zone timestamp (non-datetime frame index). The ``id`` is stable per zone
    (direction + created time) so the same zone keeps its identity across reads.
    ``instrument`` (optional) enables the canonical formation session.
    """
    order_blocks = [_ob_to_model(z, bar_ts, instrument) for z in zones.get("order_blocks", [])]
    fair_value_gaps = [_fvg_to_model(z, bar_ts, instrument) for z in zones.get("fair_value_gaps", [])]
    return order_blocks, fair_value_gaps


def _contacts_to_models(z: dict) -> list[ZoneContact]:
    """Map the read-side contact dicts (VZ-1) to schema models, skipping any
    contact whose timestamp could not be derived (non-datetime frame index) —
    never a fabricated date."""
    out: list[ZoneContact] = []
    for c in z.get("contacts", []) or []:
        at = c.get("at")
        if at is None:
            continue
        out.append(ZoneContact(at=at, level=c["level"], outcome=c["outcome"]))
    return out


def _origin_to_model(z: dict) -> Optional[ZoneOrigin]:
    o = z.get("origin")
    if not o or o.get("at") is None:
        return None
    return ZoneOrigin(kind=o["kind"], direction=o["direction"], at=o["at"], level=o["level"])


def _ob_to_model(z: dict, bar_ts: datetime, instrument: Optional[str] = None) -> OrderBlock:
    created = z.get("created_at") or bar_ts
    return OrderBlock(
        id=ob_zone_id(z["direction"], created),
        direction=z["direction"],
        level_high=z["level_high"],
        level_low=z["level_low"],
        importance=z["importance"],
        status=z["status"],
        created_at=created,
        tested=z["tested"],
        mitigated_at=z.get("mitigated_at"),
        touch_count=z.get("touch_count", 0),
        touch_ats=z.get("touch_ats", []),
        contacts=_contacts_to_models(z),
        origin=_origin_to_model(z),
        session=_session_of(instrument, created),
        user_flagged=False,
    )


def _fvg_to_model(z: dict, bar_ts: datetime, instrument: Optional[str] = None) -> FairValueGap:
    created = z.get("created_at") or bar_ts
    return FairValueGap(
        id=f"FVG_{z['direction']}_{created.strftime('%Y%m%d%H%M%S')}",
        direction=z["direction"],
        level_high=z["level_high"],
        level_low=z["level_low"],
        status=z["status"],
        created_at=created,
        tested=z["tested"],
        mitigated_at=z.get("mitigated_at"),
        fill_level=z.get("fill_level"),
        touch_count=z.get("touch_count", 0),
        touch_ats=z.get("touch_ats", []),
        contacts=_contacts_to_models(z),
        session=_session_of(instrument, created),
        user_flagged=False,
    )


def _consumed_zones_to_models(
    zones: dict[str, list[dict]],
    bar_ts: datetime,
    instrument: Optional[str] = None,
) -> tuple[list[OrderBlock], list[FairValueGap]]:
    """Map the bounded consumed-zone dicts (VZ-1) to schema models for the /zones
    « Comblées » group. Reuses the SAME per-zone mappers as the live lists."""
    obs = [_ob_to_model(z, bar_ts, instrument) for z in zones.get("consumed_order_blocks", [])]
    fvgs = [_fvg_to_model(z, bar_ts, instrument) for z in zones.get("consumed_fair_value_gaps", [])]
    return obs, fvgs


def _liquidity_to_models(
    pools: list[dict],
    bar_ts: datetime,
) -> list[LiquidityPool]:
    """Convert collected pocket dicts (from :func:`collect_liquidity_pools`) to
    schema models. ``created_at`` falls back to ``bar_ts`` when the collector
    could not derive a per-pocket timestamp (non-datetime frame index). The ``id``
    is stable per pocket (side + kind + created time) so the same pocket keeps its
    identity across reads — for display anchoring and the agent.
    """
    out: list[LiquidityPool] = []
    for z in pools:
        created = z.get("created_at") or bar_ts
        out.append(LiquidityPool(
            id=f"LIQ_{z['side']}_{z['kind']}_{created.strftime('%Y%m%d%H%M%S')}",
            side=z["side"],
            kind=z["kind"],
            level=z["level"],
            touches=z["touches"],
            is_external=z["is_external"],
            status=z["status"],
            created_at=created,
            swept_at=z.get("swept_at"),
            broken_at=z.get("broken_at"),
            user_flagged=False,
        ))
    return out


def _structure_events_to_models(
    events: dict[str, list[dict]],
    bar_ts: datetime,
) -> tuple[list[BOSRecent], list[CHOCHRecent]]:
    """Convert collected BOS/CHOCH event dicts (from
    :func:`collect_structure_events`) to schema models. ``broken_at`` falls back
    to ``bar_ts`` when the collector could not derive a per-bar timestamp
    (non-datetime frame index). Direction/level come straight from the engine
    event columns — descriptive, never predictive (status is always "confirmed":
    the break occurred)."""
    bos_events: list[BOSRecent] = []
    for e in events.get("bos_events", []):
        at = e.get("broken_at") or bar_ts
        bos_events.append(BOSRecent(
            id=_event_id("bos", e["direction"], at),
            direction=e["direction"],
            level=float(e["level"]),
            broken_at=at,
            validation_status="confirmed",
            bars_ago=e.get("bars_ago"),
        ))
    choch_events: list[CHOCHRecent] = []
    for e in events.get("choch_events", []):
        at = e.get("broken_at") or bar_ts
        choch_events.append(CHOCHRecent(
            id=_event_id("choch", e["direction"], at),
            direction=e["direction"],
            level=float(e["level"]),
            broken_at=at,
            validation_status="confirmed",
            bars_ago=e.get("bars_ago"),
        ))
    return bos_events, choch_events


# ---------------------------------------------------------------------------
# Regime mapper
# ---------------------------------------------------------------------------


def _closes(candles: Sequence[dict]) -> list[float]:
    return [float(c["close"]) for c in candles if "close" in c]


# TR-1 — the trend is DERIVED from the engine's structure, never a parallel
# close-delta. `_derive_trend` (first-vs-last close over ~500 bars) is GONE:
# no second source of truth survives.

# The range threshold that used to gate the old close-based trend now belongs to
# the Phase tile ONLY (it describes consolidation, not direction). Named so the
# (undocumented in origin) 0.3 magic number is at least visible and single-sourced.
_RANGE_CONSOLIDATION_RATIO = 0.3


def _is_close_range_bound(closes: Sequence[float]) -> bool:
    """True when price OSCILLATED but barely progressed over the window — the net
    move is < 30 % of the range travelled. Descriptive consolidation signal, used
    ONLY by the Phase tile (TR-1 removed it from the Trend tile)."""
    if len(closes) < 5:
        return False
    first, last = closes[0], closes[-1]
    rng = max(closes) - min(closes)
    if rng <= 0:
        return False
    base = max(abs(first), 1e-9)
    return (abs(last - first) / base) < (rng / base) * _RANGE_CONSOLIDATION_RATIO


def _most_recent_event(events: Sequence[dict]) -> Optional[dict]:
    """Most recent event from a :func:`collect_structure_events` list (already
    most-recent-first; guarded by ``bars_ago`` for safety)."""
    if not events:
        return None
    return min(
        events,
        key=lambda e: e.get("bars_ago") if e.get("bars_ago") is not None else 10 ** 9,
    )


def derive_structural_trend(
    structure_events: dict,
) -> tuple[TrendValue, Optional[TrendReference]]:
    """Derive the trend from the engine's discrete BOS/CHOCH events — the SINGLE
    source of truth (TR-1, definition (a)): the direction of the last structural
    break not contradicted by an opposite one.

    The current direction is set by the last CHANGE OF CHARACTER (CHOCH); every
    BOS after it is a same-direction continuation. So the anchoring event is the
    most recent CHOCH when one exists, else the most recent BOS — which also makes
    the Trend tile and the Maturité tile tell the SAME story. Returns
    ``("indeterminate", None)`` when NO structural break exists in the analysed
    history: a first-class state, never a silent default to ``neutral``.
    """
    bos = structure_events.get("bos_events") or []
    choch = structure_events.get("choch_events") or []
    ref_ev = _most_recent_event(choch)
    kind = "choch"
    if ref_ev is None:
        ref_ev = _most_recent_event(bos)
        kind = "bos"
    if ref_ev is None or ref_ev.get("direction") not in ("bullish", "bearish"):
        return "indeterminate", None
    direction: TrendValue = ref_ev["direction"]
    broken_at = ref_ev.get("broken_at")
    if broken_at is None:
        # No honest timestamp to anchor on — still report the direction.
        return direction, None
    reference = TrendReference(
        kind=kind,
        direction=ref_ev["direction"],
        level=float(ref_ev["level"]),
        broken_at=broken_at,
        bars_ago=ref_ev.get("bars_ago"),
    )
    return direction, reference


def _structural_bias_from_candle_dicts(candles: Sequence[dict]) -> MTFBiasValue:
    """Structural bias (DIRECTION only) for a timeframe — runs the REAL engine on
    OHLC dicts and returns the sign of the last CHOCH (else last BOS), mirroring
    :func:`derive_structural_trend`. Used for upper-timeframe alignment bias and
    for standalone callers that have no pre-collected events. ``indeterminate``
    when the frame is too short or carries no structural break. No timestamp is
    needed here (a bias is a direction, not an anchored reference)."""
    rows = [c for c in candles if all(k in c for k in ("open", "high", "low", "close"))]
    if len(rows) < 5:
        return "indeterminate"
    import pandas as pd

    from src.intelligence.smart_money import SmartMoneyEngine

    try:
        df = pd.DataFrame(
            [
                {
                    "open": float(c["open"]),
                    "high": float(c["high"]),
                    "low": float(c["low"]),
                    "close": float(c["close"]),
                    # SmartMoneyEngine requires an OHLCV frame; volume is unused by
                    # the structure detection, so a constant column is faithful.
                    "volume": float(c.get("volume", 0.0) or 0.0),
                }
                for c in rows
            ]
        )
        enriched = SmartMoneyEngine(data=df, config={}, verbose=False).analyze(
            compute_divergence=False
        )
    except Exception:
        return "indeterminate"

    def _last_sign(col: str) -> Optional[int]:
        if col not in enriched.columns:
            return None
        vals = enriched[col].values
        for k in range(len(vals) - 1, -1, -1):
            v = vals[k]
            if not pd.isna(v) and v != 0:
                return 1 if v > 0 else -1
        return None

    d = _last_sign("CHOCH_SIGNAL")
    if d is None:
        d = _last_sign("BOS_EVENT")
    if d is None:
        return "indeterminate"
    return "bullish" if d > 0 else "bearish"


# Volatility thresholds & window — single source of truth, mirrored to the
# frontend proof panel. The categorical result is ``low`` below _VOL_RATIO_LOW,
# ``elevated`` above _VOL_RATIO_HIGH, else ``normal``. Recent window = last
# _VOL_RECENT_N candles; baseline = the _VOL_BASELINE_N candles IMMEDIATELY
# PRECEDING them (a bounded, comparable reference — not the whole 500-bar
# history, which drowned the signal). The panel names this real denominator.
_VOL_RECENT_N = 7
_VOL_BASELINE_N = 20
_VOL_RATIO_LOW = 0.70
_VOL_RATIO_HIGH = 1.30


def _volatility_from_candles(
    candles: Sequence[dict],
) -> tuple[VolatilityObserved, Optional[VolatilityDetail]]:
    """Categorical volatility PLUS the numeric intermediates behind it.

    Returns ``(category, detail)``. ``detail`` is None when the window is too
    short (< 14 True Ranges) or the baseline is non-positive — the same guards
    that make the category fall back to ``normal``. The arithmetic is identical
    to the historical ``_derive_volatility``; only the intermediates are now
    surfaced so a sceptic can redo the operation.
    """
    if len(candles) < 14:
        return "normal", None
    trs = []
    for c in candles:
        if "high" in c and "low" in c:
            trs.append(float(c["high"]) - float(c["low"]))
    if len(trs) < 14:
        return "normal", None
    recent_n = _VOL_RECENT_N
    recent = sum(trs[-recent_n:]) / float(recent_n)
    # Baseline = the _VOL_BASELINE_N candles immediately before the recent ones
    # (fewer only when the window is short). Bounded so the reference stays a
    # recent, comparable norm rather than the whole history.
    baseline_slice = trs[-(recent_n + _VOL_BASELINE_N) : -recent_n]
    baseline_n = len(baseline_slice)
    if baseline_n == 0:
        return "normal", None
    baseline = sum(baseline_slice) / baseline_n
    if baseline <= 0:
        return "normal", None
    ratio = recent / baseline
    if ratio < _VOL_RATIO_LOW:
        category: VolatilityObserved = "low"
    elif ratio > _VOL_RATIO_HIGH:
        category = "elevated"
    else:
        category = "normal"
    detail = VolatilityDetail(
        recent_avg=recent,
        baseline_avg=baseline,
        ratio=ratio,
        recent_n=recent_n,
        baseline_n=baseline_n,
        threshold_low=_VOL_RATIO_LOW,
        threshold_high=_VOL_RATIO_HIGH,
    )
    return category, detail


def _derive_volatility(candles: Sequence[dict]) -> VolatilityObserved:
    """Categorical volatility only (backward-compatible wrapper)."""
    return _volatility_from_candles(candles)[0]


def _derive_market_phase(
    trend: TrendValue,
    volatility: VolatilityObserved,
    closes: Sequence[float],
) -> MarketPhase:
    if trend in ("bullish", "bearish"):
        return "expansion" if volatility == "elevated" else "trend"
    # trend == "indeterminate": no structural direction. Tell an actively
    # OSCILLATING market (ranging) apart from a quiet, directionless one
    # (accumulation) via the close-range test TR-1 moved OFF the Trend tile.
    if _is_close_range_bound(closes):
        return "ranging"
    return "accumulation"


def candles_to_regime(
    candles: Sequence[dict],
    mtf_candles_above: dict[str, Sequence[dict]],
    *,
    current_structure_events: Optional[dict] = None,
) -> MarketReadingRegime:
    """Derive regime from the current-TF STRUCTURE + structural bias from upper
    timeframes (TR-1).

    `candles` : OHLCV rows for the requested TF, oldest first. Each item must
    expose at minimum `close`, `high`, `low` keys.
    `mtf_candles_above` : mapping from upper-TF key (`h1`, `h4`, ...) to its
    candles list. Only keys in `VALID_MTF_KEYS` are kept.
    `current_structure_events` : the BOS/CHOCH events the assembler already
    collected for THIS timeframe (``{"bos_events": [...], "choch_events": [...]}``).
    When provided, the trend is derived from them WITH an anchored reference (the
    honest, on-screen « depuis le CHOCH … »). When absent (standalone/test
    callers), the engine is re-run for the direction only (no anchored reference).
    """
    closes = _closes(candles)
    if current_structure_events is not None:
        trend, trend_reference = derive_structural_trend(current_structure_events)
    else:
        trend = _structural_bias_from_candle_dicts(candles)
        trend_reference = None
    volatility, volatility_detail = _volatility_from_candles(candles)
    market_phase = _derive_market_phase(trend, volatility, closes)

    mtf_confluence: dict[str, MTFBiasValue] = {}
    for key, tf_candles in mtf_candles_above.items():
        if key not in VALID_MTF_KEYS:
            continue
        if not tf_candles:
            continue
        mtf_confluence[key] = _structural_bias_from_candle_dicts(tf_candles)

    return MarketReadingRegime(
        trend=trend,
        volatility_observed=volatility,
        market_phase=market_phase,
        mtf_confluence=mtf_confluence,
        volatility_detail=volatility_detail,
        trend_reference=trend_reference,
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
    "indeterminate": "indéterminée",
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

    # STR-2 defect A: recency comes from the journal (most-recent-first), NOT the
    # point-in-time `current_bos`/`current_choch` (null ~76 % / ~99 % of the time).
    latest_bos = structure.bos_events[0] if structure.bos_events else None
    latest_choch = structure.choch_events[0] if structure.choch_events else None
    if latest_bos is not None:
        tags.append(f"bos_recent_{latest_bos.direction}")
    if latest_choch is not None:
        tags.append(f"choch_recent_{latest_choch.direction}")
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

    # STR-2 defect A: the last structural break comes from the journal, not the
    # point-in-time `current_bos` (which is null on the vast majority of readings).
    latest_bos = structure.bos_events[0] if structure.bos_events else None
    if latest_bos is not None:
        parts.append(
            f"BOS {_TREND_FR[latest_bos.direction]} récent ({latest_bos.validation_status})."
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
