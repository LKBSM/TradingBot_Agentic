"""Honesty tests for the deterministic narrated-reading template (mission
« narrated-reading template-engine »).

The « Lecture narrée · Ancrée au moteur » is composed 100 % by
`narrated_reading.render_template` — no LLM. Because the text is now produced by
a fixed template, we can prove STRUCTURALLY (not just sample) that it can never
emit a forbidden term nor invent a level, whatever the engine values:

  * Vocabulary honesty — over a matrix of engine outputs AND over the 686 REAL
    engine-detected order blocks in ``tests/fixtures/ob_golden/golden_obs.json``
    (XAUUSD + EURUSD, M15/H1/H4), no output contains a forbidden term or a
    causality verb (« s'oppose à » / « contre » / « affecte » …).
  * Non-invention — every price token in every output is an engine-emitted level
    (`references_only_known_levels`); a null field yields no text.
  * Edge cases — zero zones / breaks / retest / mtf still produce a coherent,
    non-verbose reading (never « aucune donnée »).
"""

from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path

from src.intelligence.market_reading_mappers import contains_forbidden_tokens
from src.intelligence.market_reading_schema import (
    BOSRecent,
    CHOCHRecent,
    FairValueGap,
    MarketReadingRegime,
    MarketReadingStructure,
    OrderBlock,
    RetestInProgress,
)
from src.intelligence.narrated_reading import (
    NARRATION_MAX_LENGTH,
    ZoneFact,
    _contrary_reason,
    _zone_phrase,
    build_reading_facts,
    fmt_canonical,
    fmt_display,
    references_only_known_levels,
    render_template,
)

# --------------------------------------------------------------------------- #
# The FULL mission vocabulary ban (a superset of the engine FORBIDDEN_TOKENS).
# These must be structurally impossible in ANY template output.
# --------------------------------------------------------------------------- #
# Predictive / prescriptive / scoring nouns and adjectives.
_BANNED_VOCAB = [
    "setup",
    "signal",
    "signaux",
    "opportunité",
    "opportunites",
    "opportunité",
    "gagnant",
    "gagnante",
    "meilleur",
    "meilleure",
    "plus sûr",
    "plus sur",
    "recommandé",
    "recommande",
    "recommandation",
    "probabilité",
    "probabilite",
    "cible",
    "biais",
    "classement",
    "pourcentage",
]
# Causality verbs — including « s'oppose à » / « contre » in the causal sense.
_BANNED_CAUSAL = [
    "affecte",
    "impacte",
    "influence",
    "oppose",  # catches « s'oppose », « oppose »
    "contre",  # catches « contre », « contre-courant », « à l'encontre »
]


def assert_clean(text: str) -> None:
    """Raise if ``text`` carries any banned vocabulary or causality verb.

    Word-boundary for the vocab nouns; raw substring for the causal verbs (a
    stricter check — even « contre-courant » or « s'oppose » is caught)."""
    low = text.lower()
    # The engine's own post-generation guard must also pass (defence in depth).
    engine_hit = contains_forbidden_tokens(text)
    assert engine_hit is None, f"engine forbidden token {engine_hit!r} in: {text!r}"
    for term in _BANNED_VOCAB:
        assert not re.search(rf"\b{re.escape(term.lower())}\b", low), (
            f"banned vocab {term!r} in: {text!r}"
        )
    for verb in _BANNED_CAUSAL:
        assert verb not in low, f"banned causality verb {verb!r} in: {text!r}"


# --------------------------------------------------------------------------- #
# Fixtures / builders
# --------------------------------------------------------------------------- #
_GOLDEN = json.loads(
    (Path(__file__).parent / "fixtures" / "ob_golden" / "golden_obs.json").read_text(
        encoding="utf-8"
    )
)

_OB_STATUSES = ["active", "mitigated", "invalidated"]
_FVG_STATUSES = ["active", "partially_filled", "filled"]
_POSITIONS = ["below", "above", "inside"]


def _zonefact(kind, direction, status, tested, low, high, position) -> ZoneFact:
    dec = 2
    return ZoneFact(
        kind=kind,
        direction=direction,
        status=status,
        tested=tested,
        low=fmt_display(low, dec),
        high=fmt_display(high, dec),
        low_canon=fmt_canonical(low, dec),
        high_canon=fmt_canonical(high, dec),
        position=position,
    )


def _instrument_of(key: str) -> str:
    return key.split("_", 1)[0]


# --------------------------------------------------------------------------- #
# (1) Vocabulary honesty over the 686 REAL engine order blocks
# --------------------------------------------------------------------------- #


def test_zone_phrase_clean_over_all_real_golden_obs():
    """Every real engine OB, rendered under every status/position/tested combo,
    produces a zone phrase free of any banned term."""
    n = 0
    for key, payload in _GOLDEN.items():
        for ob in payload.get("engine_obs", []):
            side = ob["side"]  # "bullish" | "bearish"
            hi, lo = float(ob["ob_high"]), float(ob["ob_low"])
            for status, position, tested in product(
                _OB_STATUSES, _POSITIONS, (True, False)
            ):
                z = _zonefact("ob", side, status, tested, lo, hi, position)
                assert_clean(_zone_phrase(z))
                n += 1
    assert n >= 686 * len(_OB_STATUSES) * len(_POSITIONS) * 2


def test_contrary_clause_is_copresence_over_all_golden_obs():
    """The co-presence clause built for every real OB opposing the trend is clean
    and uses « présent malgré », never a causal « oppose » / « contre »."""
    seen_clause = False
    for key, payload in _GOLDEN.items():
        for ob in payload.get("engine_obs", []):
            side = ob["side"]
            hi, lo = float(ob["ob_high"]), float(ob["ob_low"])
            # Trend chosen opposite to the OB so the clause fires.
            trend = "bullish" if side == "bearish" else "bearish"
            z = _zonefact("ob", side, "active", False, lo, hi, "above")
            clause = _contrary_reason(trend, "aligned_up", [z])
            assert clause is not None
            seen_clause = True
            assert "présent malgré la tendance" in clause
            assert "oppose" not in clause and "contre" not in clause.lower()
            assert_clean(f"À noter : {clause}.")
    assert seen_clause


# --------------------------------------------------------------------------- #
# (2) Full-reading honesty on dense REAL structures (XAUUSD + EURUSD, 3 TFs)
# --------------------------------------------------------------------------- #


def _dense_structure_from_golden(key: str) -> tuple[MarketReadingStructure, float]:
    """Build a MarketReadingStructure loaded with every real OB of ``key`` (status
    varied deterministically) plus a couple of FVGs and breaks, and a price sitting
    in the middle of the OB cloud so several zones are « near price »."""
    obs = _GOLDEN[key]["engine_obs"]
    order_blocks = []
    mids = []
    for i, ob in enumerate(obs):
        hi, lo = float(ob["ob_high"]), float(ob["ob_low"])
        mids.append((hi + lo) / 2)
        order_blocks.append(
            OrderBlock(
                id=f"{key}_ob_{i}",
                direction=ob["side"],
                level_high=hi,
                level_low=lo,
                importance="medium",
                status=_OB_STATUSES[i % len(_OB_STATUSES)],
                created_at="2026-06-25T22:00:00Z",
                tested=bool(i % 2),
            )
        )
    mids.sort()
    price = mids[len(mids) // 2]  # median mid → dense neighbourhood
    fvg = FairValueGap(
        id=f"{key}_fvg",
        direction="bullish",
        level_high=price * 1.0009,
        level_low=price * 1.0004,
        status="active",
        created_at="2026-06-25T22:00:00Z",
        tested=False,
    )
    structure = MarketReadingStructure(
        order_blocks=order_blocks,
        fair_value_gaps=[fvg],
        bos_events=[
            BOSRecent(
                direction="bullish",
                level=price * 0.999,
                broken_at="2026-06-25T21:00:00Z",
                validation_status="confirmed",
            )
        ],
        choch_events=[
            CHOCHRecent(
                direction="bearish",
                level=price * 1.001,
                broken_at="2026-06-25T20:00:00Z",
                validation_status="pending",
            )
        ],
        retest_in_progress=RetestInProgress(
            level=price * 0.999,
            type="bos_retest",
            started_at="2026-06-25T21:30:00Z",
        ),
    )
    return structure, price


def test_full_reading_honest_and_anchored_on_all_golden_markets():
    for key in _GOLDEN:
        instrument = _instrument_of(key)
        structure, price = _dense_structure_from_golden(key)
        for trend, vol, phase in product(
            ("bullish", "bearish", "indeterminate"),
            ("low", "normal", "elevated"),
            ("accumulation", "distribution", "trend", "ranging", "expansion"),
        ):
            regime = MarketReadingRegime(
                trend=trend,
                volatility_observed=vol,
                market_phase=phase,
                mtf_confluence={"h1": "bullish", "h4": "bearish"},
            )
            facts = build_reading_facts(structure, regime, price, instrument)
            text = render_template(facts)
            # Honesty: clean vocab + no invented level + within budget.
            assert_clean(text)
            assert references_only_known_levels(text, facts), key
            assert len(text) <= NARRATION_MAX_LENGTH
            # Non-telegraphic: full sentences, ends on a terminator, starts on socle.
            assert text.startswith("Tendance ")
            assert text.rstrip().endswith((".", "!", "?"))


# --------------------------------------------------------------------------- #
# (3) Exhaustive matrix over the enum space (structural guarantee)
# --------------------------------------------------------------------------- #


def test_enum_matrix_never_emits_forbidden_and_never_invents():
    """Cross every regime enum with zone/break/retest presence toggles; the
    template output is always clean and self-anchored."""
    price = 2000.0
    ob = OrderBlock(
        id="ob1", direction="bearish", level_high=2006.0, level_low=2002.0,
        importance="medium", status="active", created_at="2026-06-20T00:00:00Z",
        tested=False,
    )
    bos = BOSRecent(
        direction="bullish", level=1998.0, broken_at="2026-06-20T00:00:00Z",
        validation_status="confirmed",
    )
    retest = RetestInProgress(
        level=1998.0, type="bos_retest", started_at="2026-06-20T00:00:00Z"
    )
    combos = product(
        ("bullish", "bearish", "indeterminate"),
        ("low", "normal", "elevated"),
        ("accumulation", "distribution", "trend", "ranging", "expansion"),
        ({}, {"h1": "bullish", "h4": "bullish"}, {"h1": "bearish", "h4": "bullish"}),
        (True, False),  # zone present
        (True, False),  # break present
        (True, False),  # retest present
    )
    for trend, vol, phase, mtf, has_zone, has_break, has_retest in combos:
        structure = MarketReadingStructure(
            order_blocks=[ob] if has_zone else [],
            bos_events=[bos] if has_break else [],
            retest_in_progress=retest if has_retest else None,
        )
        regime = MarketReadingRegime(
            trend=trend, volatility_observed=vol, market_phase=phase,
            mtf_confluence=mtf,
        )
        facts = build_reading_facts(structure, regime, price, "XAUUSD")
        text = render_template(facts)
        assert_clean(text)
        assert references_only_known_levels(text, facts)


# --------------------------------------------------------------------------- #
# (4) Edge cases — nothing but the socle, never verbose filler
# --------------------------------------------------------------------------- #


def test_empty_reading_is_socle_only_no_filler():
    regime = MarketReadingRegime(
        trend="indeterminate", volatility_observed="normal",
        market_phase="ranging", mtf_confluence={},
    )
    facts = build_reading_facts(MarketReadingStructure(), regime, 2000.0, "XAUUSD")
    text = render_template(facts)
    # Only the always-present régime sentence — never a « aucune donnée » filler.
    assert text == "Tendance indéterminée, volatilité normale, phase de range."
    assert "aucun" not in text.lower()
    assert "noter" not in text  # no contrary clause when nothing opposes
    assert_clean(text)


def test_no_segment_produces_placeholder_text():
    """A null field must yield NO text at all (not « — » / « non disponible »)."""
    regime = MarketReadingRegime(
        trend="bullish", volatility_observed="normal",
        market_phase="trend", mtf_confluence={},
    )
    facts = build_reading_facts(MarketReadingStructure(), regime, 2000.0, "XAUUSD")
    text = render_template(facts)
    for placeholder in ("non disponible", "—", "aucune", "n/a", "null", "None"):
        assert placeholder.lower() not in text.lower()
