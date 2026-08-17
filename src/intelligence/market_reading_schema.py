"""MarketReading Pydantic schema — V2 product contract (Section 2.3 of architecture doc).

Aligned strictly on `docs/architecture/MIA_MARKETS_V2_VISION.md` Section 2.3.
The MarketReading is the unique structured object published at each candle close.
It describes market conditions factually — it never recommends action (niveau 1.5 strict).

Documented extensions vs Section 2.3 exact JSON example:
- `OrderBlock.direction` and `FairValueGap.direction` are present (Optional) because
  the SMC scanner produces this information. They are Optional so that the exact
  JSON example from the doc (which omits them) still validates. Production-generated
  readings will populate them.
- `OrderBlock.status` accepts "invalidated" (cassure sans mitigation simple).
- `FairValueGap.status` accepts "partially_filled" between "active" and "filled".
- `NewsJustPublished.surprise_direction` is Optional for qualitative news
  (Fed speeches, BCE statements without numeric beat/miss).

Open spec gaps (to flag in final report for doc V2 extension):
- Doc Section 2.3 OB/FVG example lacks `direction` field → extend doc.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Type aliases (Literal vocabularies)
# ---------------------------------------------------------------------------

Direction = Literal["bullish", "bearish"]
ValidationStatus = Literal["confirmed", "pending", "invalidated"]
ImpactLevel = Literal["low", "medium", "high"]
SurpriseDirection = Literal["beat", "miss", "in_line"]
# TR-1: the trend is now DERIVED from the engine's structure (last BOS/CHOCH not
# contradicted), never a parallel close-delta. Three first-class states only:
# a direction, or ``indeterminate`` when NO structural break exists in the
# analysed history. ``neutral``/``ranging`` were artefacts of the old close-based
# calculation and are gone — consolidation now lives on the Phase tile.
TrendValue = Literal["bullish", "bearish", "indeterminate"]
VolatilityObserved = Literal["low", "normal", "elevated"]
MarketPhase = Literal["accumulation", "distribution", "trend", "ranging", "expansion"]
# Per-upper-timeframe structural bias — same vocabulary as TrendValue (TR-1).
MTFBiasValue = Literal["bullish", "bearish", "indeterminate"]
# The structural event that anchors ``trend`` (a break is always directional).
TrendReferenceKind = Literal["bos", "choch"]
OBStatus = Literal["active", "mitigated", "invalidated"]
FVGStatus = Literal["active", "partially_filled", "filled"]
OBImportance = Literal["low", "medium", "high"]
RetestType = Literal["bos_retest", "choch_retest", "ob_retest", "fvg_retest"]
# External liquidity pools (buy-side above / sell-side below the range).
LiquiditySide = Literal["bsl", "ssl"]
LiquidityKind = Literal["equal_highs", "equal_lows", "range_high", "range_low"]
LiquidityStatus = Literal["intact", "swept", "broken"]
DescriptionSource = Literal["haiku_generated", "template_fallback"]

# Valid MTF alignment keys — any perimeter or reference unit can be an alignment
# TARGET (the tile shows the units above the viewed one, TF-1 decision C).
# Derived from the single timeframe registry, lowercased; never a hand-listed set.
from src.intelligence import timeframe_registry as _tfreg

VALID_MTF_KEYS = {
    s.id.lower() for s in _tfreg.all_specs() if s.perimeter or s.reference
}

# Composite trigger type: <event>_<tf>_<direction> for bos/choch, <event>_<tf> for others.
TRIGGER_TYPE_PATTERN = (
    r"^(bos|choch)_(m15|h1|h4|d1)_(bullish|bearish)$"
    r"|^(ob_mitigation|fvg_fill|retest)_(m15|h1|h4|d1)$"
)

# Narrated reading length budget. Sized to HOLD a complete present-tense
# paragraph (2-4 sentences synthesising trend, multi-TF alignment, near-price
# zones and volatility) so the narration is never cut short. History: 280
# (legacy one-sentence « synthèse ») → 500 (multi-sentence) → 700, because a
# full 4-sentence reading with a contrary-context clause could still overflow
# 500 and get hard-truncated mid-word. This field is a DISPLAY-only surface,
# NOT consumed by the Telegram footer (separate render contract), so the larger
# ceiling costs nothing downstream. When a narration does exceed it, the cut is
# made on a SENTENCE boundary (never mid-word) — see
# `narrated_reading.truncate_at_sentence`.
DESCRIPTION_MAX_LENGTH = 700


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


class MarketReadingHeader(BaseModel):
    instrument: str
    timeframe: str
    candle_close_ts: datetime
    close_price: float
    # Number of bars actually analysed for this reading — the per-timeframe live
    # window (MT-D1). Optional so older payloads / fixtures without it still
    # validate; the front labels « fenêtre de N bougies » from it, falling back to
    # the legacy 500 constant when absent.
    analysis_window_bars: Optional[int] = None


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


# Whole ANALYSED bars between the break and the last bar of the window (DG-1).
# Emitted by collect_structure_events (idx − event_idx). Optional so older
# payloads / fixtures without it still validate; the front counts real bougies
# from it instead of wall-clock hours ÷ tf (which overcounts week-end gaps).
# STR-2 defect C: a STABLE, COLLISION-FREE identifier for the break event. Built
# from kind + broken_at + direction (see market_reading_mappers._event_id), so a
# BOS and a CHOCH that land on the SAME bar get DIFFERENT ids and the chart can
# anchor focus by id — never by timestamp (which collides on a shared bar) and
# never by « nearest ». Optional so older payloads/fixtures still validate; the
# frontend falls back to a synthesised key only when it is absent.
class BOSRecent(BaseModel):
    id: Optional[str] = None
    direction: Direction
    level: float
    broken_at: datetime
    validation_status: ValidationStatus
    bars_ago: Optional[int] = None


class CHOCHRecent(BaseModel):
    id: Optional[str] = None
    direction: Direction
    level: float
    broken_at: datetime
    validation_status: ValidationStatus
    bars_ago: Optional[int] = None


# VZ-1: outcome of ONE observed price contact with a zone — strictly factual,
# NEVER a judgement (never "respected"/"held"/"strong"). The three distinct
# outcomes the product must not conflate (mission question B):
#   * edge_touch — price reached the near edge but penetrated LESS than
#     ``MitigationPolicy.contact_edge_touch_fraction`` of the height (a kiss).
#   * entry_exit — price entered the band then left through the SAME (near) edge
#     without crossing it (a deeper tap that did not consume the zone).
#   * traversal  — price CLOSED through the far edge (OB) / fully filled the gap
#     (FVG): the contact that consumed the zone.
#   * inside     — price is CURRENTLY within the band (ongoing; no exit yet).
ContactOutcome = Literal["edge_touch", "entry_exit", "traversal", "inside"]


class ZoneContact(BaseModel):
    """One price contact with a zone, classified read-side over the SAME candle
    window and predicates the touch counter already uses — additive, no new
    detection. ``level`` is the DEEPEST price reached into the band on this
    contact (the "niveau atteint")."""

    at: datetime
    level: float
    outcome: ContactOutcome


class ZoneOrigin(BaseModel):
    """The structural break an order block PRECEDES — what makes it an OB (the
    last opposite candle before the break). Associated read-side from the engine's
    already-emitted ``BOS_EVENT`` / ``CHOCH_SIGNAL`` / ``BOS_BREAK_LEVEL`` columns
    — never a new detection. ``kind`` bos|choch, ``direction`` of the break,
    ``at`` its bar timestamp, ``level`` the broken level."""

    kind: Literal["bos", "choch"]
    direction: Direction
    at: datetime
    level: float


class OrderBlock(BaseModel):
    id: str
    direction: Optional[Direction] = None
    level_high: float
    level_low: float
    importance: OBImportance
    status: OBStatus
    created_at: datetime
    tested: bool
    # Timestamp of first interaction (mitigation point). None while the zone is
    # untouched/active. Lets the frontend bound the box formation → mitigation,
    # or → current price when still active. Descriptive, not predictive.
    mitigated_at: Optional[datetime] = None
    # Number of DISTINCT taps since formation (a run of consecutive in-zone bars
    # counts once), and the ENTRY timestamp of each. touch_count >= 1 ⟺ tested;
    # touch_ats[0] == mitigated_at. Defaults keep older payloads valid.
    touch_count: int = 0
    touch_ats: list[datetime] = Field(default_factory=list)
    # VZ-1: per-contact ledger (edge_touch / entry_exit / traversal / inside) and
    # the origin break. Both additive & optional so older payloads stay valid and
    # the /app surface (which ignores them) is unaffected.
    contacts: list[ZoneContact] = Field(default_factory=list)
    origin: Optional[ZoneOrigin] = None
    # VZ-1: formation session (asia|london|new_york), from market_calendar —
    # canonical source; the frontend only falls back to its mirror when absent.
    session: Optional[str] = None
    user_flagged: bool = False


class FairValueGap(BaseModel):
    id: str
    direction: Optional[Direction] = None
    level_high: float
    level_low: float
    status: FVGStatus
    created_at: datetime
    tested: bool
    # Timestamp of first entry (partial-fill point). None while the gap is
    # untouched/active. Same box-bounding purpose as OrderBlock.mitigated_at.
    mitigated_at: Optional[datetime] = None
    # Price the gap has been penetrated to — the DEEPEST wick into the band so
    # far (clamped inside [level_low, level_high]). None while active/untouched.
    # Read-only/descriptive: lets the frontend SHRINK the box to the still-open
    # portion of a partially-filled gap (stops "just under the wicks"). It is a
    # lifecycle measurement over engine-emitted highs/lows — never a detection
    # threshold and never recomputes the gap.
    fill_level: Optional[float] = None
    # Distinct entries since formation (a run of consecutive in-band bars counts
    # once) and each entry's timestamp. touch_count >= 1 ⟺ tested; touch_ats[0]
    # == mitigated_at. Defaults keep older payloads valid.
    touch_count: int = 0
    touch_ats: list[datetime] = Field(default_factory=list)
    # VZ-1: per-contact ledger. An FVG has no structural-break origin (it is a
    # 3-candle imbalance, explained statically), so no ``origin`` field here.
    contacts: list[ZoneContact] = Field(default_factory=list)
    session: Optional[str] = None
    user_flagged: bool = False


class RetestInProgress(BaseModel):
    level: float
    type: RetestType
    started_at: datetime


class LiquidityPool(BaseModel):
    """External liquidity pocket — equal highs/lows or a range extreme.

    Strictly DESCRIPTIVE (niveau 1.5): it states WHERE resting liquidity sits and
    WHETHER that level has been intact / swept / broken — a past, observable fact.
    It carries NO target, draw, bias or probability; nothing here implies the price
    will move toward the pocket. The pocket geometry reuses the engine's existing
    swing fractals (UP_FRACTAL / DOWN_FRACTAL) — it is not a new detection.

    Side: ``bsl`` = buy-side liquidity (above equal highs / the range high),
    ``ssl`` = sell-side liquidity (below equal lows / the range low).

    Lifecycle (deterministic, factual):
      * intact  — no later bar has traded through ``level``.
      * swept   — a later bar's WICK pierced ``level`` and that bar CLOSED back
                  inside (``swept_at`` = first such bar). A liquidity grab event.
      * broken  — a later bar CLOSED net through ``level`` (``broken_at`` = first
                  such bar). Terminal; supersedes ``swept`` if both occurred.
    """

    id: str
    side: LiquiditySide
    kind: LiquidityKind
    level: float
    touches: int  # number of swing points forming the pocket (1 for range extremes)
    is_external: bool  # True = at/beyond the current range's extreme swing
    status: LiquidityStatus
    created_at: datetime  # timestamp of the earliest swing forming the pocket
    # First bar that wicked through and closed back inside. None unless swept.
    swept_at: Optional[datetime] = None
    # First bar that closed net through the level. None unless broken. Terminal.
    broken_at: Optional[datetime] = None
    user_flagged: bool = False


class MarketReadingStructure(BaseModel):
    # POINT-IN-TIME break state (STR-2): the break in effect AT THE READ BAR —
    # a break fresh on the last closed candle, or (for BOS only) an earlier break
    # still vouched for by the retest state machine. It is NULL most of the time
    # (`current_bos` ~76 %, `current_choch` ~99 % of readings) because the last
    # bar is rarely itself a break bar. It is NOT the recency history: « a break
    # occurred within the last N candles » is answered by `bos_events` /
    # `choch_events` below. The former names (`bos` / `choch`) were renamed so no
    # consumer can mistake the point-in-time state for the journal again (STR-2
    # defect A: five surfaces read the singular field and starved).
    current_bos: Optional[BOSRecent] = None
    current_choch: Optional[CHOCHRecent] = None
    # Discrete BOS / CHOCH break EVENTS observed over the window, most-recent
    # first (capped). Read-only/descriptive history and the SINGLE SOURCE OF
    # TRUTH for recency: the engine detects many breaks but only the last-bar one
    # ever surfaced via the point-in-time fields (audit 2026-06-16
    # "sous-surfaçage": 88 BOS / 40 CHOCH detected over 6 combos, ≤1 surfaced).
    # These lists carry the real broken level + honest timestamp + bars_ago of
    # each break — read from engine event columns, never recomputed.
    bos_events: list[BOSRecent] = Field(default_factory=list)
    choch_events: list[CHOCHRecent] = Field(default_factory=list)
    order_blocks: list[OrderBlock] = Field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = Field(default_factory=list)
    # VZ-1: a BOUNDED set of the most recently CONSUMED zones (invalidated OB /
    # filled FVG) — the ones `order_blocks`/`fair_value_gaps` deliberately drop
    # once consumed. Read-only/descriptive twin so the /zones page can show a
    # "Comblées" (traversed) group with each zone's full contact ledger. Additive
    # & optional: the /app Structure surface never reads them, older payloads stay
    # valid. Their `contacts` list ends with the `traversal` contact.
    consumed_order_blocks: list[OrderBlock] = Field(default_factory=list)
    consumed_fair_value_gaps: list[FairValueGap] = Field(default_factory=list)
    # External liquidity pockets (EQH/EQL + range extremes) with intact/swept/
    # broken state. Read-only/descriptive twin of order_blocks/fair_value_gaps;
    # injected by the SMC pipeline under ``_liquidity`` and built by the structure
    # mapper. Empty on callers/tests that don't run collect_liquidity_pools.
    liquidity_pools: list[LiquidityPool] = Field(default_factory=list)
    retest_in_progress: Optional[RetestInProgress] = None


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class VolatilityDetail(BaseModel):
    """The numeric intermediates behind ``volatility_observed`` — so the reader
    can REDO the operation. Strictly descriptive: it is the exact arithmetic the
    engine ran on engine-emitted candle highs/lows, never a new detection.

    ``ratio = recent_avg / baseline_avg``; the categorical result is ``low`` when
    ``ratio < threshold_low``, ``elevated`` when ``ratio > threshold_high``, else
    ``normal``. ``recent_avg`` / ``baseline_avg`` are mean True Ranges
    (high − low) over the last ``recent_n`` candles and the ``baseline_n``
    candles preceding them (all remaining candles in the window, NOT a fixed 20).
    Optional on the regime so older payloads / minimal fixtures still validate.
    """

    recent_avg: float
    baseline_avg: float
    ratio: float
    recent_n: int
    baseline_n: int
    threshold_low: float
    threshold_high: float


class TrendReference(BaseModel):
    """The structural event that anchors the current ``trend`` — the last change
    of character (CHOCH) that set the direction, or the last BOS when no CHOCH
    exists in the analysed history. Read-only/descriptive: it names WHY the trend
    reads as it does, so the Trend tile and the Maturité tile tell the same story
    (« depuis le CHOCH haussier du 24 juil. »). ``None`` on the regime when the
    trend is ``indeterminate`` (no structural break in the analysed history)."""

    kind: TrendReferenceKind
    direction: Direction
    level: float
    broken_at: datetime
    bars_ago: Optional[int] = None


class MarketReadingRegime(BaseModel):
    trend: TrendValue
    volatility_observed: VolatilityObserved
    market_phase: MarketPhase
    mtf_confluence: dict[str, MTFBiasValue]
    # TR-1: the structural event anchoring ``trend`` (last CHOCH/BOS). ``None``
    # when trend is ``indeterminate``. Optional so minimal fixtures / older
    # payloads still validate; live readings always carry it when a break exists.
    trend_reference: Optional[TrendReference] = None
    # Numeric proof behind ``volatility_observed`` (recent vs baseline True Range,
    # ratio, thresholds). None when the window is too short to compute it (< 14
    # candles) or on older payloads. Read-only/descriptive — see VolatilityDetail.
    volatility_detail: Optional[VolatilityDetail] = None

    @field_validator("mtf_confluence")
    @classmethod
    def _validate_mtf_keys(cls, v: dict[str, str]) -> dict[str, str]:
        invalid = set(v.keys()) - VALID_MTF_KEYS
        if invalid:
            raise ValueError(
                f"Invalid MTF timeframe keys {sorted(invalid)}; "
                f"allowed: {sorted(VALID_MTF_KEYS)}"
            )
        return v


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class NewsUpcoming(BaseModel):
    event: str
    scheduled_at: datetime
    time_to_event_min: int
    impact: ImpactLevel
    currency: str
    potential_effect_description: str
    # Deterministic id of this release (title|currency|scheduled_at hash) — the
    # same value the calendar page uses, so the App news module can deep-link to
    # /actualites/<event_id>. Optional/None on older payloads.
    event_id: Optional[str] = None


class NewsJustPublished(BaseModel):
    event: str
    published_at: datetime
    event_id: Optional[str] = None
    actual: Optional[float] = None
    forecast: Optional[float] = None
    previous: Optional[float] = None
    surprise_direction: Optional[SurpriseDirection] = None
    currency: str
    impact: ImpactLevel
    potential_effect_description: str


class TechnicalTriggerRecent(BaseModel):
    type: str = Field(..., pattern=TRIGGER_TYPE_PATTERN)
    occurred_at: datetime
    minutes_ago: int


class MarketReadingEvents(BaseModel):
    news_upcoming: list[NewsUpcoming] = Field(default_factory=list)
    news_just_published: list[NewsJustPublished] = Field(default_factory=list)
    technical_triggers_recent: list[TechnicalTriggerRecent] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------


class MarketReadingConditions(BaseModel):
    tags: list[str] = Field(default_factory=list)
    description: str = Field(..., max_length=DESCRIPTION_MAX_LENGTH)
    description_source: DescriptionSource


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


class MarketReading(BaseModel):
    schema_version: str = "2.0.0"
    header: MarketReadingHeader
    structure: MarketReadingStructure
    regime: MarketReadingRegime
    events: MarketReadingEvents
    conditions: MarketReadingConditions
    # MC-1 market status (open / closed_weekend / closed_holiday / daily_break /
    # data_lagged). Computed fresh at response time from the market calendar +
    # the last closed candle — NEVER persisted (it depends on "now"), so the
    # stored payload always carries None and the accessor fills it per request.
    market_status: Optional[dict] = None
    # RG-1c calendar reference levels (day/week open + previous day/week
    # extremes), aggregated server-side over the MC-1 trading calendar. Computed
    # fresh at response time from the intraday candle cache — NEVER persisted, so
    # the stored payload always carries None and the accessor fills it per request.
    reference_levels: Optional[dict] = None


__all__ = [
    "BOSRecent",
    "CHOCHRecent",
    "DESCRIPTION_MAX_LENGTH",
    "DescriptionSource",
    "Direction",
    "FairValueGap",
    "FVGStatus",
    "ImpactLevel",
    "LiquidityKind",
    "LiquidityPool",
    "LiquiditySide",
    "LiquidityStatus",
    "MarketPhase",
    "MarketReading",
    "MarketReadingConditions",
    "MarketReadingEvents",
    "MarketReadingHeader",
    "MarketReadingRegime",
    "MarketReadingStructure",
    "MTFBiasValue",
    "NewsJustPublished",
    "NewsUpcoming",
    "OBImportance",
    "OBStatus",
    "OrderBlock",
    "RetestInProgress",
    "RetestType",
    "SurpriseDirection",
    "TechnicalTriggerRecent",
    "TRIGGER_TYPE_PATTERN",
    "TrendReference",
    "TrendReferenceKind",
    "TrendValue",
    "VALID_MTF_KEYS",
    "ValidationStatus",
    "VolatilityDetail",
    "VolatilityObserved",
]
