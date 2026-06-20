"""Tests for MarketReading Pydantic schema (Chantier 2 Étape 2)."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from src.intelligence.market_reading_schema import (
    BOSRecent,
    FairValueGap,
    MarketReading,
    MarketReadingConditions,
    MarketReadingEvents,
    MarketReadingHeader,
    MarketReadingRegime,
    MarketReadingStructure,
    NewsJustPublished,
    OrderBlock,
    TechnicalTriggerRecent,
)


# Exact JSON example from docs/architecture/MIA_MARKETS_V2_VISION.md Section 2.3.
DOC_EXAMPLE_JSON = """
{
  "schema_version": "2.0.0",
  "header": {
    "instrument": "XAUUSD",
    "timeframe": "M15",
    "candle_close_ts": "2026-05-28T14:00:00Z",
    "close_price": 2378.45
  },
  "structure": {
    "bos": {
      "direction": "bullish",
      "level": 2375.20,
      "broken_at": "2026-05-28T13:30:00Z",
      "validation_status": "confirmed"
    },
    "choch": null,
    "order_blocks": [
      {
        "id": "OB_001",
        "level_high": 2370.50,
        "level_low": 2368.20,
        "importance": "high",
        "status": "active",
        "created_at": "2026-05-26T08:00:00Z",
        "tested": false,
        "user_flagged": false
      }
    ],
    "fair_value_gaps": [
      {
        "id": "FVG_001",
        "level_high": 2378.20,
        "level_low": 2376.00,
        "status": "active",
        "created_at": "2026-05-28T12:15:00Z",
        "tested": false,
        "user_flagged": false
      }
    ],
    "retest_in_progress": {
      "level": 2375.20,
      "type": "bos_retest",
      "started_at": "2026-05-28T13:45:00Z"
    }
  },
  "regime": {
    "trend": "bullish",
    "volatility_observed": "elevated",
    "market_phase": "expansion",
    "mtf_confluence": {
      "h1": "bullish",
      "h4": "bullish"
    }
  },
  "events": {
    "news_upcoming": [
      {
        "event": "US Non-Farm Payrolls",
        "scheduled_at": "2026-05-28T14:30:00Z",
        "time_to_event_min": 30,
        "impact": "high",
        "currency": "USD",
        "potential_effect_description": "Publication majeure de l'emploi am\\u00e9ricain. Mouvement attendu sur le dollar, impact indirect sur XAU via la corr\\u00e9lation USD inverse classique."
      }
    ],
    "news_just_published": [],
    "technical_triggers_recent": [
      {
        "type": "bos_h1_bullish",
        "occurred_at": "2026-05-28T13:30:00Z",
        "minutes_ago": 30
      }
    ]
  },
  "conditions": {
    "tags": [
      "volatility_elevated",
      "news_imminent_high_impact",
      "structure_aligned_mtf",
      "retest_in_progress"
    ],
    "description": "Le march\\u00e9 XAU sur 15min est en phase de retest d'un Order Block H1, avec volatilit\\u00e9 \\u00e9lev\\u00e9e \\u00e0 l'approche du NFP USD dans 30 minutes. Structure H1 et H4 align\\u00e9e haussi\\u00e8rement.",
    "description_source": "haiku_generated"
  }
}
"""


def _minimal_reading() -> MarketReading:
    return MarketReading(
        header=MarketReadingHeader(
            instrument="XAUUSD",
            timeframe="M15",
            candle_close_ts=datetime(2026, 5, 28, 14, 0, 0, tzinfo=timezone.utc),
            close_price=2378.45,
        ),
        structure=MarketReadingStructure(),
        regime=MarketReadingRegime(
            trend="bullish",
            volatility_observed="normal",
            market_phase="trend",
            mtf_confluence={"h1": "bullish"},
        ),
        events=MarketReadingEvents(),
        conditions=MarketReadingConditions(
            tags=["test"],
            description="Test description.",
            description_source="template_fallback",
        ),
    )


def test_minimal_instantiation():
    """Test 1: all required fields present, defaults applied for the rest."""
    m = _minimal_reading()
    assert m.schema_version == "2.0.0"
    assert m.header.instrument == "XAUUSD"
    assert m.header.close_price == 2378.45
    assert m.structure.bos is None
    assert m.structure.choch is None
    assert m.structure.order_blocks == []
    assert m.structure.fair_value_gaps == []
    assert m.structure.retest_in_progress is None
    assert m.events.news_upcoming == []


def test_roundtrip_doc_example():
    """Test 2: validate the exact JSON example from doc V2 Section 2.3, then roundtrip."""
    m = MarketReading.model_validate_json(DOC_EXAMPLE_JSON)

    assert m.schema_version == "2.0.0"
    assert m.header.instrument == "XAUUSD"
    assert m.header.timeframe == "M15"
    assert m.header.close_price == 2378.45

    assert m.structure.bos is not None
    assert m.structure.bos.direction == "bullish"
    assert m.structure.bos.level == 2375.20
    assert m.structure.bos.validation_status == "confirmed"
    assert m.structure.choch is None

    assert len(m.structure.order_blocks) == 1
    ob = m.structure.order_blocks[0]
    assert ob.id == "OB_001"
    assert ob.importance == "high"
    assert ob.status == "active"
    assert ob.tested is False
    assert ob.user_flagged is False
    assert ob.direction is None  # Doc example omits direction; Optional per schema

    assert len(m.structure.fair_value_gaps) == 1
    assert m.structure.fair_value_gaps[0].id == "FVG_001"
    assert m.structure.fair_value_gaps[0].status == "active"

    assert m.structure.retest_in_progress is not None
    assert m.structure.retest_in_progress.type == "bos_retest"

    assert m.regime.trend == "bullish"
    assert m.regime.volatility_observed == "elevated"
    assert m.regime.market_phase == "expansion"
    assert m.regime.mtf_confluence == {"h1": "bullish", "h4": "bullish"}

    assert len(m.events.news_upcoming) == 1
    assert m.events.news_upcoming[0].time_to_event_min == 30
    assert m.events.news_upcoming[0].currency == "USD"
    assert m.events.news_just_published == []
    assert m.events.technical_triggers_recent[0].type == "bos_h1_bullish"
    assert m.events.technical_triggers_recent[0].minutes_ago == 30

    assert m.conditions.description_source == "haiku_generated"
    assert "volatility_elevated" in m.conditions.tags

    # Roundtrip preserves all data
    serialized = m.model_dump_json()
    m2 = MarketReading.model_validate_json(serialized)
    assert m2 == m


def test_literal_constraints_rejected():
    """Test 3: Pydantic refuses values outside the Literal enum vocabularies."""
    with pytest.raises(ValidationError):
        MarketReadingRegime(
            trend="sideways",  # not in TrendValue
            volatility_observed="normal",
            market_phase="trend",
            mtf_confluence={"h1": "bullish"},
        )
    with pytest.raises(ValidationError):
        MarketReadingConditions(
            tags=[],
            description="x",
            description_source="llm_haiku",  # old V1 vocab, no longer accepted
        )
    with pytest.raises(ValidationError):
        OrderBlock(
            id="OB_X",
            level_high=1.0,
            level_low=0.5,
            importance="critical",  # not in OBImportance
            status="active",
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tested=False,
        )
    with pytest.raises(ValidationError):
        FairValueGap(
            id="FVG_X",
            level_high=1.0,
            level_low=0.5,
            status="closed",  # not in FVGStatus (use "filled")
            created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            tested=False,
        )
    with pytest.raises(ValidationError):
        BOSRecent(
            direction="bullish",
            level=1.0,
            broken_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            validation_status="maybe",  # not in ValidationStatus
        )


def test_schema_version_default_at_root():
    """Test 4: schema_version defaults to "2.0.0" at the MarketReading root level."""
    m = _minimal_reading()
    assert m.schema_version == "2.0.0"
    dumped = m.model_dump()
    assert "schema_version" in dumped
    assert dumped["schema_version"] == "2.0.0"
    # schema_version is NOT inside header
    assert "schema_version" not in dumped["header"]
    # Forward-compat override accepted
    m2 = MarketReading(
        schema_version="2.1.0",
        header=m.header,
        structure=m.structure,
        regime=m.regime,
        events=m.events,
        conditions=m.conditions,
    )
    assert m2.schema_version == "2.1.0"


def test_iso8601_datetime_validation():
    """Test 5: ISO 8601 datetime strings (incl. Z suffix) are parsed and validated."""
    h = MarketReadingHeader(
        instrument="XAUUSD",
        timeframe="M15",
        candle_close_ts="2026-05-28T14:00:00Z",
        close_price=2378.45,
    )
    assert h.candle_close_ts.year == 2026
    assert h.candle_close_ts.month == 5
    assert h.candle_close_ts.day == 28
    assert h.candle_close_ts.hour == 14
    assert h.candle_close_ts.tzinfo is not None

    with pytest.raises(ValidationError):
        MarketReadingHeader(
            instrument="XAUUSD",
            timeframe="M15",
            candle_close_ts="not-a-date",
            close_price=1.0,
        )


def test_empty_lists_allowed_everywhere():
    """Test 6: events without news, structure without OB/FVG, empty tags — all valid."""
    m = _minimal_reading()
    assert m.structure.order_blocks == []
    assert m.structure.fair_value_gaps == []
    assert m.events.news_upcoming == []
    assert m.events.news_just_published == []
    assert m.events.technical_triggers_recent == []
    # Roundtrip with all-empty events
    m2 = MarketReading.model_validate_json(m.model_dump_json())
    assert m2 == m


def test_description_capped_at_max_length():
    """Test 7: conditions.description capped at DESCRIPTION_MAX_LENGTH (narration)."""
    from src.intelligence.market_reading_schema import DESCRIPTION_MAX_LENGTH

    exactly_max = "a" * DESCRIPTION_MAX_LENGTH
    MarketReadingConditions(
        tags=["t"],
        description=exactly_max,
        description_source="template_fallback",
    )
    with pytest.raises(ValidationError):
        MarketReadingConditions(
            tags=["t"],
            description="a" * (DESCRIPTION_MAX_LENGTH + 1),
            description_source="template_fallback",
        )


def test_news_just_published_qualitative_news():
    """Test 8: surprise_direction Optional for qualitative news (Fed speech, BCE)."""
    n = NewsJustPublished(
        event="Fed Chair Powell Speaks",
        published_at=datetime(2026, 5, 28, 18, 0, 0, tzinfo=timezone.utc),
        actual=None,
        forecast=None,
        previous=None,
        surprise_direction=None,
        currency="USD",
        impact="high",
        potential_effect_description="Discours qualitatif sans donnees chiffrees.",
    )
    assert n.actual is None
    assert n.surprise_direction is None
    n2 = NewsJustPublished.model_validate_json(n.model_dump_json())
    assert n2 == n
    # Numeric news with surprise still works
    n3 = NewsJustPublished(
        event="US NFP",
        published_at=datetime(2026, 5, 28, 14, 30, 0, tzinfo=timezone.utc),
        actual=210.0,
        forecast=180.0,
        previous=175.0,
        surprise_direction="beat",
        currency="USD",
        impact="high",
        potential_effect_description="NFP above forecast.",
    )
    assert n3.surprise_direction == "beat"


def test_mtf_confluence_rejects_invalid_keys_and_values():
    """Test 9: MTFConfluence rejects keys not in {m15,h1,h4,d1,w1} and invalid bias values."""
    MarketReadingRegime(
        trend="bullish",
        volatility_observed="normal",
        market_phase="trend",
        mtf_confluence={"h1": "bullish", "h4": "bearish", "d1": "ranging"},
    )
    with pytest.raises(ValidationError):
        MarketReadingRegime(
            trend="bullish",
            volatility_observed="normal",
            market_phase="trend",
            mtf_confluence={"h5": "bullish"},  # h5 not in VALID_MTF_KEYS
        )
    with pytest.raises(ValidationError):
        MarketReadingRegime(
            trend="bullish",
            volatility_observed="normal",
            market_phase="trend",
            mtf_confluence={"h1": "moonbeam"},  # not in MTFBiasValue
        )


def test_technical_trigger_type_regex():
    """Test 10: technical_triggers_recent[].type matches the composite regex pattern."""
    valid_types = [
        "bos_h1_bullish",
        "bos_m15_bearish",
        "choch_h4_bullish",
        "choch_d1_bearish",
        "ob_mitigation_h4",
        "ob_mitigation_m15",
        "fvg_fill_d1",
        "fvg_fill_h1",
        "retest_h1",
        "retest_m15",
    ]
    for t in valid_types:
        TechnicalTriggerRecent(
            type=t,
            occurred_at=datetime(2026, 5, 28, 13, 0, 0, tzinfo=timezone.utc),
            minutes_ago=30,
        )
    invalid_types = [
        "bos_h5_bullish",        # h5 not in allowed TFs
        "fvg_fill",              # missing TF
        "random_string",
        "bos_h1",                # missing direction for bos
        "bullish_h1_bos",        # wrong order
        "bos_h1_neutral",        # neutral not allowed (bullish|bearish only)
        "ob_mitigation_h1_bullish",  # extra suffix
        "",
    ]
    for t in invalid_types:
        with pytest.raises(ValidationError):
            TechnicalTriggerRecent(
                type=t,
                occurred_at=datetime(2026, 5, 28, 13, 0, 0, tzinfo=timezone.utc),
                minutes_ago=30,
            )
