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
DescriptionSource = Literal["haiku_generated", "template_fallback"]

VALID_MTF_KEYS = {"m15", "h1", "h4", "d1", "w1"}

# Composite trigger type: <event>_<tf>_<direction> for bos/choch, <event>_<tf> for others.
TRIGGER_TYPE_PATTERN = (
    r"^(bos|choch)_(m15|h1|h4|d1)_(bullish|bearish)$"
    r"|^(ob_mitigation|fvg_fill|retest)_(m15|h1|h4|d1)$"
)

DESCRIPTION_MAX_LENGTH = 280


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
    user_flagged: bool = False


class FairValueGap(BaseModel):
    id: str
    direction: Optional[Direction] = None
    level_high: float
    level_low: float
    status: FVGStatus
    created_at: datetime
    tested: bool
    user_flagged: bool = False


class RetestInProgress(BaseModel):
    level: float
    type: RetestType
    started_at: datetime


class MarketReadingStructure(BaseModel):
    bos: Optional[BOSRecent] = None
    choch: Optional[CHOCHRecent] = None
    order_blocks: list[OrderBlock] = Field(default_factory=list)
    fair_value_gaps: list[FairValueGap] = Field(default_factory=list)
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
