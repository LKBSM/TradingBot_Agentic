"""Tests for the narrated reading — facts projection, anchoring validator and the
deterministic template. Covers the four mission requirements:

  (a) the narration may reference ONLY real engine facts (foreign level rejected);
  (b) the deterministic fallback is factual and is what the engine falls back to;
  (c) output is present-tense and carries no predictive / forbidden vocabulary;
  (d) contrary context is included whenever it exists.
"""

from src.intelligence.market_reading_mappers import contains_forbidden_tokens
from src.intelligence.market_reading_schema import (
    BOSRecent,
    FairValueGap,
    MarketReadingRegime,
    MarketReadingStructure,
    OrderBlock,
    RetestInProgress,
)
from src.intelligence.narrated_reading import (
    allowed_levels,
    build_reading_facts,
    references_only_known_levels,
    render_template,
)

PRICE = 2000.0
INSTRUMENT = "XAUUSD"


def _ob(level_low, level_high, direction="bearish", status="active", tested=False):
    return OrderBlock(
        id=f"ob_{level_low}",
        direction=direction,
        level_high=level_high,
        level_low=level_low,
        importance="medium",
        status=status,
        created_at="2026-06-20T00:00:00Z",
        tested=tested,
    )


def _regime(trend="bullish", vol="normal", phase="trend", mtf=None):
    return MarketReadingRegime(
        trend=trend,
        volatility_observed=vol,
        market_phase=phase,
        mtf_confluence=mtf if mtf is not None else {},
    )


# ---------------------------------------------------------------------------
# (a) Anchoring: only real engine levels may be referenced
# ---------------------------------------------------------------------------


def test_known_level_passes_anchoring():
    structure = MarketReadingStructure(order_blocks=[_ob(1995.0, 2005.0)])
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    # A narration reusing a real band edge is accepted.
    assert references_only_known_levels(
        "Order Block actif borne 1995.00–2005.00.", facts
    )


def test_foreign_level_fails_anchoring():
    structure = MarketReadingStructure(order_blocks=[_ob(1995.0, 2005.0)])
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    # 2222.22 was never produced by the engine → rejected.
    assert not references_only_known_levels("Le prix vise 2222.22.", facts)


def test_bare_integers_are_not_treated_as_levels():
    """`M15`, `les 3 TF`, `H4` carry no decimal and must not trip the validator."""
    structure = MarketReadingStructure()
    facts = build_reading_facts(structure, _regime(mtf={"h4": "bullish"}), PRICE, INSTRUMENT)
    assert references_only_known_levels("Les 3 TF M15, H1 et H4 sont neutres.", facts)


def test_allowed_levels_collects_every_fact_level():
    structure = MarketReadingStructure(
        order_blocks=[_ob(1995.0, 2005.0)],
        fair_value_gaps=[
            FairValueGap(
                id="fvg1",
                direction="bullish",
                level_high=1990.0,
                level_low=1985.0,
                status="active",
                created_at="2026-06-20T00:00:00Z",
                tested=False,
            )
        ],
        bos=BOSRecent(
            direction="bullish",
            level=1998.0,
            broken_at="2026-06-20T00:00:00Z",
            validation_status="confirmed",
        ),
        retest_in_progress=RetestInProgress(
            level=1997.0, type="bos_retest", started_at="2026-06-20T00:00:00Z"
        ),
    )
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    allowed = allowed_levels(facts)
    for lvl in ("2000.00", "1995.00", "2005.00", "1990.00", "1985.00", "1998.00", "1997.00"):
        assert lvl in allowed


# ---------------------------------------------------------------------------
# (b) Deterministic fallback is factual
# ---------------------------------------------------------------------------


def test_template_is_factual_and_self_anchored():
    structure = MarketReadingStructure(order_blocks=[_ob(1995.0, 2005.0)])
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    text = render_template(facts)
    # Mentions the regime facts and a real zone band, and references only real levels.
    assert "Tendance" in text
    assert "Order Block" in text
    assert references_only_known_levels(text, facts)


def test_template_distinguishes_provisional_from_confirmed():
    confirmed = MarketReadingStructure(
        bos=BOSRecent(
            direction="bullish", level=1998.0,
            broken_at="2026-06-20T00:00:00Z", validation_status="confirmed",
        )
    )
    pending = MarketReadingStructure(
        bos=BOSRecent(
            direction="bullish", level=1998.0,
            broken_at="2026-06-20T00:00:00Z", validation_status="pending",
        )
    )
    t_conf = render_template(build_reading_facts(confirmed, _regime(), PRICE, INSTRUMENT))
    t_prov = render_template(build_reading_facts(pending, _regime(), PRICE, INSTRUMENT))
    assert "confirmé" in t_conf
    assert "provisoire" in t_prov


# ---------------------------------------------------------------------------
# (c) Present tense, no predictive / forbidden vocabulary
# ---------------------------------------------------------------------------


def test_template_never_emits_forbidden_tokens():
    structure = MarketReadingStructure(
        order_blocks=[_ob(1995.0, 2005.0, direction="bearish")],
        fair_value_gaps=[
            FairValueGap(
                id="fvg1", direction="bullish", level_high=1990.0, level_low=1985.0,
                status="partially_filled", created_at="2026-06-20T00:00:00Z", tested=True,
            )
        ],
        bos=BOSRecent(
            direction="bullish", level=1998.0,
            broken_at="2026-06-20T00:00:00Z", validation_status="pending",
        ),
        retest_in_progress=RetestInProgress(
            level=1997.0, type="ob_retest", started_at="2026-06-20T00:00:00Z"
        ),
    )
    regime = _regime(trend="bullish", vol="elevated", phase="expansion",
                     mtf={"h1": "bearish", "h4": "bullish"})
    text = render_template(build_reading_facts(structure, regime, PRICE, INSTRUMENT))
    assert contains_forbidden_tokens(text) is None
    # No future tense markers that would imply a forecast.
    lowered = text.lower()
    for forecast in ("va ", "sera", "devrait", "pourrait", "probab"):
        assert forecast not in lowered


# ---------------------------------------------------------------------------
# (d) Contrary context is surfaced when it exists
# ---------------------------------------------------------------------------


def test_contrary_context_pullback_against_higher_tfs():
    # Reading TF bearish, higher TFs aligned bullish → pullback contrary.
    regime = _regime(trend="bearish", mtf={"h1": "bullish", "h4": "bullish"})
    facts = build_reading_facts(MarketReadingStructure(), regime, PRICE, INSTRUMENT)
    assert facts.contrary is not None
    assert "contre-courant" in render_template(facts)


def test_contrary_context_opposite_zone_near_price():
    # Trend bullish but an ACTIVE bearish OB sits near the price → contrary.
    structure = MarketReadingStructure(
        order_blocks=[_ob(2002.0, 2006.0, direction="bearish", status="active")]
    )
    regime = _regime(trend="bullish")
    facts = build_reading_facts(structure, regime, PRICE, INSTRUMENT)
    assert facts.contrary is not None
    text = render_template(facts)
    assert "À noter" in text


def test_no_contrary_when_one_directional():
    # Trend bullish, higher TFs bullish, only a bullish active zone → no contrary.
    structure = MarketReadingStructure(
        order_blocks=[_ob(1995.0, 1999.0, direction="bullish", status="active")]
    )
    regime = _regime(trend="bullish", mtf={"h1": "bullish", "h4": "bullish"})
    facts = build_reading_facts(structure, regime, PRICE, INSTRUMENT)
    assert facts.contrary is None
    assert "À noter" not in render_template(facts)


# ---------------------------------------------------------------------------
# Near-price zone selection
# ---------------------------------------------------------------------------


def test_far_zone_excluded_unless_nearest_active():
    # One zone within the window, one far away → only the near one is surfaced.
    structure = MarketReadingStructure(
        order_blocks=[
            _ob(1998.0, 2002.0, status="active"),   # straddles price → near
            _ob(1500.0, 1510.0, status="active"),   # ~25% away → excluded
        ]
    )
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    bands = {(z.low, z.high) for z in facts.zones}
    assert ("1998.00", "2002.00") in bands
    assert ("1500.00", "1510.00") not in bands


def test_zone_position_relative_to_price():
    structure = MarketReadingStructure(
        order_blocks=[
            _ob(1990.0, 1995.0, status="active"),   # below price 2000
            _ob(2005.0, 2010.0, status="active"),   # above price 2000
        ]
    )
    facts = build_reading_facts(structure, _regime(), PRICE, INSTRUMENT)
    pos = {(z.low, z.high): z.position for z in facts.zones}
    assert pos[("1990.00", "1995.00")] == "below"
    assert pos[("2005.00", "2010.00")] == "above"
