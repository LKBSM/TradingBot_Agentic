"""Tests for src/api/insight_signal_v2.py (UX-1.1).

Per DoD: model v2 + 4 mockups + round-trip serialization test.

Tests cover:
1. Schema integrity (required fields, validators)
2. Direction/levels consistency (BULLISH stop<entry, BEARISH stop>entry)
3. NEUTRAL forbids levels
4. Backward-compat shim from_v1_signal lifts legacy SignalResponse
5. Round-trip JSON serialization preserves all fields
6. Surface renderers (Telegram, B2B, audit) produce expected shapes
7. UE 2024/2811 compliance: never expose raw "BUY/SELL"
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.api.insight_signal_v2 import (
    SCHEMA_VERSION,
    ComplianceMeta,
    ConvictionLabel,
    InsightSignalV2,
    NarrativeLanguage,
    SetupDirection,
    SignalLevels,
    Source,
    SourceType,
    Timeframe,
    conviction_to_label,
    from_v1_signal,
    to_audit_row,
    to_b2b_dict,
    to_telegram_b2c,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bullish_signal() -> InsightSignalV2:
    return InsightSignalV2(
        id="test_bull_001",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.BULLISH_SETUP,
        conviction_0_100=72,
        levels=SignalLevels(
            entry=2350.00,
            stop=2340.00,
            target_1=2370.00,
            target_2=2390.00,
        ),
        narrative_short="Bullish XAU M15 setup. BOS confirmed.",
        narrative_long="Detailed narrative...",
        sources_cited=[
            Source(
                type=SourceType.PAPER,
                ref="https://example.com/paper",
                label="López de Prado AFML ch. 7",
            )
        ],
        compliance=ComplianceMeta(
            disclaimer_lang=NarrativeLanguage.FR,
            edge_claim=False,
            is_paper_demo=True,
        ),
        created_at_utc=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
        valid_until_utc=datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def hold_signal() -> InsightSignalV2:
    return InsightSignalV2(
        id="test_hold_001",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=35,
        levels=SignalLevels(),  # all None — required by validator
        narrative_short="Neutral context. No setup.",
        created_at_utc=datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc),
    )


# ---------------------------------------------------------------------------
# Schema integrity
# ---------------------------------------------------------------------------


def test_schema_version_is_2_1_0():
    """Bumped to 2.1.0 on Sprint 1 enrichment (uncertainty + readouts + breakdown)."""
    assert SCHEMA_VERSION == "2.1.0"


def test_id_must_be_non_empty():
    with pytest.raises(ValueError):
        InsightSignalV2(
            id="   ",
            instrument="XAUUSD",
            timeframe=Timeframe.M15,
            direction=SetupDirection.NEUTRAL,
            conviction_0_100=50,
            narrative_short="x",
            created_at_utc=datetime.now(timezone.utc),
        )


def test_naive_timestamp_is_normalised_to_utc():
    """A naive datetime gets UTC tz; an explicit tz gets converted to UTC."""
    naive = datetime(2026, 5, 1, 12, 0)  # no tz
    sig = InsightSignalV2(
        id="x",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=50,
        narrative_short="ok",
        created_at_utc=naive,
    )
    assert sig.created_at_utc.tzinfo is timezone.utc


# ---------------------------------------------------------------------------
# Direction × levels consistency
# ---------------------------------------------------------------------------


def test_bullish_requires_stop_below_entry():
    with pytest.raises(ValueError, match="stop < entry"):
        InsightSignalV2(
            id="bad",
            instrument="XAUUSD",
            timeframe=Timeframe.M15,
            direction=SetupDirection.BULLISH_SETUP,
            conviction_0_100=70,
            levels=SignalLevels(entry=2350, stop=2360, target_1=2370),
            narrative_short="x",
            created_at_utc=datetime.now(timezone.utc),
        )


def test_bullish_requires_target_above_entry():
    with pytest.raises(ValueError, match="target_1 > entry"):
        InsightSignalV2(
            id="bad",
            instrument="XAUUSD",
            timeframe=Timeframe.M15,
            direction=SetupDirection.BULLISH_SETUP,
            conviction_0_100=70,
            levels=SignalLevels(entry=2350, stop=2340, target_1=2330),
            narrative_short="x",
            created_at_utc=datetime.now(timezone.utc),
        )


def test_bearish_requires_stop_above_entry():
    with pytest.raises(ValueError, match="stop > entry"):
        InsightSignalV2(
            id="bad",
            instrument="XAUUSD",
            timeframe=Timeframe.M15,
            direction=SetupDirection.BEARISH_SETUP,
            conviction_0_100=70,
            levels=SignalLevels(entry=2350, stop=2340, target_1=2330),
            narrative_short="x",
            created_at_utc=datetime.now(timezone.utc),
        )


def test_neutral_must_not_carry_levels():
    with pytest.raises(ValueError, match="NEUTRAL.*not carry"):
        InsightSignalV2(
            id="bad",
            instrument="XAUUSD",
            timeframe=Timeframe.M15,
            direction=SetupDirection.NEUTRAL,
            conviction_0_100=30,
            levels=SignalLevels(entry=2350.0),  # forbidden on NEUTRAL
            narrative_short="x",
            created_at_utc=datetime.now(timezone.utc),
        )


def test_bullish_signal_rr_ratio_computed(bullish_signal):
    # entry 2350, stop 2340, target 2370 → risk=10, reward=20 → R:R = 2.0
    assert bullish_signal.rr_ratio == pytest.approx(2.0)


def test_hold_signal_rr_ratio_is_none(hold_signal):
    assert hold_signal.rr_ratio is None


# ---------------------------------------------------------------------------
# Conviction bucketing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (0, ConvictionLabel.WEAK),
        (39, ConvictionLabel.WEAK),
        (40, ConvictionLabel.MODERATE),
        (59, ConvictionLabel.MODERATE),
        (60, ConvictionLabel.STRONG),
        (79, ConvictionLabel.STRONG),
        (80, ConvictionLabel.INSTITUTIONAL),
        (100, ConvictionLabel.INSTITUTIONAL),
        (-5, ConvictionLabel.WEAK),  # clamps to 0
        (150, ConvictionLabel.INSTITUTIONAL),  # clamps to 100
    ],
)
def test_conviction_bucketing(score, expected):
    assert conviction_to_label(score) == expected


# ---------------------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------------------


def test_from_v1_signal_lifts_open_long_to_bullish():
    legacy = SimpleNamespace(
        signal_id="old_001",
        action="OPEN_LONG",
        symbol="XAUUSD",
        entry_price=2350.0,
        stop_loss=2340.0,
        take_profit=2370.0,
        created_at=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
    )
    sig = from_v1_signal(legacy)
    assert sig.direction == SetupDirection.BULLISH_SETUP
    assert sig.levels.entry == 2350.0
    assert sig.levels.stop == 2340.0
    assert sig.levels.target_1 == 2370.0
    assert sig.id == "old_001"


def test_from_v1_signal_lifts_open_short_to_bearish():
    legacy = SimpleNamespace(
        signal_id="old_002",
        action="OPEN_SHORT",
        symbol="XAUUSD",
        entry_price=2350.0,
        stop_loss=2360.0,
        take_profit=2330.0,
        created_at=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
    )
    sig = from_v1_signal(legacy)
    assert sig.direction == SetupDirection.BEARISH_SETUP


def test_from_v1_signal_hold_clears_levels():
    legacy = SimpleNamespace(
        signal_id="old_003",
        action="HOLD",
        symbol="XAUUSD",
        entry_price=2350.0,
        stop_loss=2340.0,
        take_profit=2370.0,
        created_at=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
    )
    sig = from_v1_signal(legacy)
    assert sig.direction == SetupDirection.NEUTRAL
    assert sig.levels.entry is None
    assert sig.levels.stop is None
    assert sig.levels.target_1 is None


# ---------------------------------------------------------------------------
# Round-trip JSON serialization (DoD core requirement)
# ---------------------------------------------------------------------------


def test_round_trip_json_preserves_all_fields(bullish_signal):
    """Serialise to JSON then parse back — fields must round-trip."""
    payload = bullish_signal.model_dump_json()
    parsed = InsightSignalV2.model_validate_json(payload)

    assert parsed.id == bullish_signal.id
    assert parsed.instrument == bullish_signal.instrument
    assert parsed.timeframe == bullish_signal.timeframe
    assert parsed.direction == bullish_signal.direction
    assert parsed.conviction_0_100 == bullish_signal.conviction_0_100
    assert parsed.levels.entry == bullish_signal.levels.entry
    assert parsed.levels.stop == bullish_signal.levels.stop
    assert parsed.levels.target_1 == bullish_signal.levels.target_1
    assert parsed.narrative_short == bullish_signal.narrative_short
    assert parsed.narrative_long == bullish_signal.narrative_long
    assert len(parsed.sources_cited) == len(bullish_signal.sources_cited)
    assert parsed.compliance.edge_claim == bullish_signal.compliance.edge_claim
    assert parsed.compliance.is_paper_demo == bullish_signal.compliance.is_paper_demo
    assert parsed.created_at_utc == bullish_signal.created_at_utc


def test_round_trip_via_dict(bullish_signal):
    """Pure dict round-trip (no JSON layer)."""
    d = bullish_signal.model_dump()
    parsed = InsightSignalV2.model_validate(d)
    assert parsed.id == bullish_signal.id
    assert parsed.direction == bullish_signal.direction


# ---------------------------------------------------------------------------
# Surface renderers
# ---------------------------------------------------------------------------


def test_to_telegram_b2c_uses_structure_label_not_buy(bullish_signal):
    """2.1.0 renderer: 'STRUCTURE HAUSSIÈRE' descriptive, never BUY/ACHETEZ."""
    msg = to_telegram_b2c(bullish_signal)
    assert "STRUCTURE HAUSSIÈRE" in msg
    # UE 2024/2811 compliance: NEVER raw "ACHETEZ" / "BUY" anywhere
    assert "BUY" not in msg.upper()
    assert "ACHETEZ" not in msg.upper()
    assert "VENDEZ" not in msg.upper()
    # Length cap
    assert len(msg) <= 800


def test_to_telegram_b2c_no_entry_stop_target_visible(bullish_signal):
    """2.1.0: the Telegram surface NEVER renders entry/stop/target_1 as
    explicit values. Indicator stance — trader composes the trade."""
    msg = to_telegram_b2c(bullish_signal)
    # The fixture has entry=2350, stop=2340, target_1=2370 — those numbers
    # must NOT appear as labelled trade instructions.
    assert "Entrée :" not in msg
    assert "Stop :" not in msg
    assert "Cible :" not in msg


def test_to_telegram_b2c_bearish(bullish_signal):
    bullish_signal_dict = bullish_signal.model_dump()
    bullish_signal_dict["direction"] = "BEARISH_SETUP"
    bullish_signal_dict["levels"]["stop"] = 2360.0
    bullish_signal_dict["levels"]["target_1"] = 2330.0
    bear = InsightSignalV2.model_validate(bullish_signal_dict)
    msg = to_telegram_b2c(bear)
    assert "STRUCTURE BAISSIÈRE" in msg


def test_to_b2b_dict_includes_schema_version(bullish_signal):
    d = to_b2b_dict(bullish_signal)
    assert d["schema_version"] == SCHEMA_VERSION
    assert d["direction"] == "BULLISH_SETUP"  # enum value


def test_to_audit_row_keeps_only_deterministic_fields(bullish_signal):
    row = to_audit_row(bullish_signal)
    expected_keys = {
        "signal_id",
        "schema_version",
        "instrument",
        "timeframe",
        "direction",
        "conviction_0_100",
        "entry",
        "stop",
        "target_1",
        "edge_claim",
        "is_paper_demo",
        "created_at_utc",
    }
    assert set(row.keys()) == expected_keys
    assert row["signal_id"] == "test_bull_001"
    assert row["created_at_utc"] == "2026-05-01T12:00:00+00:00"


# ---------------------------------------------------------------------------
# Compliance hard rules
# ---------------------------------------------------------------------------


def test_default_compliance_is_paper_demo_no_edge_claim():
    """Default compliance flags are conservative: no edge claim, paper demo."""
    sig = InsightSignalV2(
        id="x",
        instrument="XAUUSD",
        timeframe=Timeframe.M15,
        direction=SetupDirection.NEUTRAL,
        conviction_0_100=30,
        narrative_short="ok",
        created_at_utc=datetime.now(timezone.utc),
    )
    assert sig.compliance.edge_claim is False
    assert sig.compliance.is_paper_demo is True


def test_setup_direction_enum_excludes_raw_buy_sell():
    """The enum itself never carries 'BUY' or 'SELL' to ensure UI cannot
    accidentally render them."""
    values = [d.value for d in SetupDirection]
    assert "BUY" not in values
    assert "SELL" not in values
    assert "BULLISH_SETUP" in values
    assert "BEARISH_SETUP" in values
    assert "NEUTRAL" in values
