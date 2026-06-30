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
TrendValue = Literal["bullish", "bearish", "neutral", "ranging"]
VolatilityObserved = Literal["low", "normal", "elevated"]
MarketPhase = Literal["accumulation", "distribution", "trend", "ranging", "expansion"]
MTFBiasValue = Literal["bullish", "bearish", "neutral", "ranging"]
OBStatus = Literal["active", "mitigated", "invalidated"]
FVGStatus = Literal["active", "partially_filled", "filled"]
OBImportance = Literal["low", "medium", "high"]
RetestType = Literal["bos_retest", "choch_retest", "ob_retest", "fvg_retest"]
# External liquidity pools (buy-side above / sell-side below the range).
LiquiditySide = Literal["bsl", "ssl"]
LiquidityKind = Literal["equal_highs", "equal_lows", "range_high", "range_low"]
LiquidityStatus = Literal["intact", "swept", "broken"]
DescriptionSource = Literal["haiku_generated", "template_fallback"]

VALID_MTF_KEYS = {"m15", "h1", "h4", "d1", "w1"}

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


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class BOSRecent(BaseModel):
    direction: Direction
    level: float
    broken_at: datetime
    validation_status: ValidationStatus


class CHOCHRecent(BaseModel):
    direction: Direction
    level: float
    broken_at: datetime
    validation_status: ValidationStatus


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
    bos: Optional[BOSRecent] = None
    choch: Optional[CHOCHRecent] = None
    # Discrete BOS / CHOCH break EVENTS observed over the window, most-recent
    # first (capped). Read-only/descriptive history: the engine detects many
    # breaks but only the last-bar one ever surfaced via `bos`/`choch` (audit
    # 2026-06-16 "sous-surfaçage": 88 BOS / 40 CHOCH detected over 6 combos, ≤1
    # surfaced). These lists carry the real broken level + honest timestamp of
    # each break — read from engine event columns, never recomputed. `bos` /
    # `choch` above stay the single "current" break for backward compatibility.
    bos_events: list[BOSRecent] = Field(default_factory=list)
    choch_events: list[CHOCHRecent] = Field(default_factory=list)
    order_blocks: list[OrderBlock] = Field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = Field(default_factory=list)
    # External liquidity pockets (EQH/EQL + range extremes) with intact/swept/
    # broken state. Read-only/descriptive twin of order_blocks/fair_value_gaps;
    # injected by the SMC pipeline under ``_liquidity`` and built by the structure
    # mapper. Empty on callers/tests that don't run collect_liquidity_pools.
    liquidity_pools: list[LiquidityPool] = Field(default_factory=list)
    retest_in_progress: Optional[RetestInProgress] = None


# ---------------------------------------------------------------------------
# Regime
# ---------------------------------------------------------------------------


class MarketReadingRegime(BaseModel):
    trend: TrendValue
    volatility_observed: VolatilityObserved
    market_phase: MarketPhase
    mtf_confluence: dict[str, MTFBiasValue]

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


class NewsJustPublished(BaseModel):
    event: str
    published_at: datetime
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
    "TrendValue",
    "VALID_MTF_KEYS",
    "ValidationStatus",
    "VolatilityObserved",
]
