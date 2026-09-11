"""TR-1 — the trend is DERIVED from the engine's structure (last BOS/CHOCH not
contradicted), never a parallel close-delta.

These guards cover: the derivation, the first-class ``indeterminate`` state
(never a silent ``neutral``), its propagation to the multi-timeframe alignment
and to the scanner (adjusted, visible denominator; never counted as agreement),
the on-screen traceability (``trend_reference``), the descriptive copy (no
strength/reliability vocabulary), and that detection outputs are untouched.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.intelligence.conditions_scanner import (
    PALETTE,
    _TREND_ADJ,
    _eval_higher_tf_agrees,
)
from src.intelligence.market_reading_mappers import (
    _structural_bias_from_candle_dicts,
    candles_to_regime,
    collect_structure_events,
    derive_structural_trend,
)


# --------------------------------------------------------------------------- #
# Fixtures — synthetic, CI-safe (no dependency on the audit candle cache)
# --------------------------------------------------------------------------- #
def _zigzag(n: int = 80, drift: float = 6.0, base: float = 2300.0) -> list[dict]:
    """Zigzag with pullbacks so the ENGINE forms swings + fires a BOS/CHOCH.
    drift > 0 → bullish, < 0 → bearish. A monotone ramp (no swings) is
    ``indeterminate`` by design — that is the whole point of TR-1."""
    amp = 25.0
    out = []
    for i in range(n):
        phase = i % 8
        local = amp if phase in (3, 4) else (-amp if phase in (7, 0) else 0.0)
        close = base + (i // 8) * drift * 8 + local
        out.append({"open": close, "high": close + 3, "low": close - 3, "close": close})
    return out


def _enriched(dicts: list[dict]):
    from src.intelligence.smart_money import SmartMoneyEngine

    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    df = pd.DataFrame(
        [{**d, "volume": 0.0} for d in dicts],
        index=[start + timedelta(hours=i) for i in range(len(dicts))],
    )
    df.index.name = "ts"
    return SmartMoneyEngine(data=df, config={}, verbose=False).analyze(
        compute_divergence=False
    )


def _ev(direction: str, bars_ago: int, level: float = 4050.0) -> dict:
    return {
        "direction": direction,
        "level": level,
        "broken_at": "2026-07-24T18:00:00+00:00",
        "bars_ago": bars_ago,
    }


def _reading(trend: str, timeframe: str = "M15") -> dict:
    return {
        "regime": {"trend": trend},
        "header": {"close_price": 4050.0, "timeframe": timeframe},
    }


# --------------------------------------------------------------------------- #
# Derivation
# --------------------------------------------------------------------------- #
def test_no_structural_event_is_indeterminate_never_neutral():
    trend, ref = derive_structural_trend({"bos_events": [], "choch_events": []})
    assert trend == "indeterminate"
    assert ref is None


def test_trend_follows_last_choch_over_older_bos():
    events = {
        "bos_events": [_ev("bullish", 3), _ev("bullish", 40)],
        "choch_events": [_ev("bullish", 20)],
    }
    trend, ref = derive_structural_trend(events)
    assert trend == "bullish"
    assert ref is not None and ref.kind == "choch" and ref.direction == "bullish"


def test_trend_uses_last_bos_when_no_choch():
    events = {"bos_events": [_ev("bearish", 5), _ev("bearish", 30)], "choch_events": []}
    trend, ref = derive_structural_trend(events)
    assert trend == "bearish"
    assert ref is not None and ref.kind == "bos"


def test_trend_never_contradicts_a_bullish_journal():
    # A bullish, non-contradicted journal must NEVER read as a bearish trend.
    events = {"bos_events": [_ev("bullish", 2)], "choch_events": [_ev("bullish", 10)]}
    trend, _ = derive_structural_trend(events)
    assert trend != "bearish"


# --------------------------------------------------------------------------- #
# Regime wiring + traceability
# --------------------------------------------------------------------------- #
def test_regime_exposes_trend_reference_for_traceability():
    dicts = _zigzag(drift=6.0)
    events = collect_structure_events(_enriched(dicts))
    reg = candles_to_regime(dicts, {}, current_structure_events=events)
    assert reg.trend == "bullish"
    assert reg.trend_reference is not None
    assert reg.trend_reference.direction == "bullish"
    assert reg.trend_reference.bars_ago is not None  # feeds the « depuis … » line


def test_regime_indeterminate_when_no_events_passed_and_flat_data():
    # No structural break in a symmetric oscillation → indeterminate + no reference
    flat = []
    base = 2300.0
    for i in range(30):
        c = base if i in (0, 29) else (base + 5 if i % 2 == 0 else base - 5)
        flat.append({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c})
    reg = candles_to_regime(flat, {})
    assert reg.trend == "indeterminate"
    assert reg.trend_reference is None
    # consolidation is described by the PHASE tile, not the Trend tile
    assert reg.market_phase in ("ranging", "accumulation")


@pytest.mark.parametrize("unit", ["M1", "M15", "H1", "H4", "D1", "W1"])
def test_structural_trend_is_unit_agnostic_all_six(unit):
    # The derivation reads engine events, not calendar time: the same structured
    # series reads bullish on EVERY timeframe (parity across the six units).
    assert _structural_bias_from_candle_dicts(_zigzag(drift=6.0)) == "bullish"
    assert _structural_bias_from_candle_dicts(_zigzag(drift=-6.0)) == "bearish"


# --------------------------------------------------------------------------- #
# Alignement avec l'unité supérieure / scanner
#
# SC-1 a remplacé le `mtf_aligned` à 3 unités fixes par `higher_tf_agrees` : une
# comparaison RELATIVE à l'unité immédiatement supérieure, qui vaut sur les six
# unités (un jeu fixe casse en haut du périmètre). Les garde-fous TR-1 portés
# ici sont les mêmes : une unité SANS tendance structurelle ne doit jamais être
# comptée comme un accord, et une absence ne doit jamais se lire comme un « non ».
# --------------------------------------------------------------------------- #
def test_scanner_higher_tf_same_direction_is_met_and_names_the_unit():
    trends = {"H4": "bullish", "H1": "bullish", "M15": "bullish"}
    res = _eval_higher_tf_agrees(_reading("bullish", "M15"), "same", trends)
    assert res["met"] is True
    assert res["available"] is True
    # l'unité comparée est NOMMÉE dans le détail (libellé lisible, pas l'id brut)
    assert "1 h" in res["detail"].lower()


def test_scanner_higher_tf_opposite_direction_is_a_real_no():
    trends = {"H1": "bearish"}
    res = _eval_higher_tf_agrees(_reading("bullish", "M15"), "same", trends)
    assert res["met"] is False
    assert res["available"] is True  # un vrai désaccord, PAS un trou de données


def test_scanner_indeterminate_higher_unit_is_non_evaluable_never_an_agreement():
    trends = {"H1": "indeterminate"}
    res = _eval_higher_tf_agrees(_reading("bullish", "M15"), "same", trends)
    assert res["met"] is False
    assert res["available"] is False  # non évaluable, jamais compté comme accord
    assert "tendance structurelle" in res["detail"].lower()


def test_scanner_indeterminate_current_unit_is_non_evaluable():
    trends = {"H1": "bullish"}
    res = _eval_higher_tf_agrees(_reading("indeterminate", "M15"), "same", trends)
    assert res["met"] is False
    assert res["available"] is False


def test_scanner_no_higher_unit_tracked_is_unavailable_not_a_no():
    res = _eval_higher_tf_agrees(_reading("bullish", "M15"), "same", {})
    assert res["available"] is False
    assert "supérieure" in res["detail"].lower()


# --------------------------------------------------------------------------- #
# Copy discipline — no strength / reliability vocabulary in trend surfaces
# --------------------------------------------------------------------------- #
_FORBIDDEN_FORCE = ["forte", "solide", "en place", "s'essouffle", "essouffle"]


def test_scanner_trend_copy_has_no_strength_vocabulary():
    strings: list[str] = []
    for p in PALETTE:
        if p["type"] in ("higher_tf_agrees", "trend_is"):
            strings.append(p["label"])
            strings.append(p["description"])
    strings.extend(_TREND_ADJ.values())
    for s in strings:
        low = s.lower()
        for bad in _FORBIDDEN_FORCE:
            assert bad not in low, f"strength word {bad!r} leaked into trend copy: {s!r}"


# --------------------------------------------------------------------------- #
# Non-regression — detection is untouched by the trend derivation
# --------------------------------------------------------------------------- #
def test_detection_columns_deterministic_and_unmutated_by_derivation():
    dicts = _zigzag(drift=6.0)
    e1, e2 = _enriched(dicts), _enriched(dicts)
    for col in ("BOS_EVENT", "CHOCH_SIGNAL"):
        assert list(e1[col].fillna(0)) == list(e2[col].fillna(0))
    ev = collect_structure_events(e1)
    before = (len(ev["bos_events"]), len(ev["choch_events"]))
    derive_structural_trend(ev)  # pure read
    ev2 = collect_structure_events(e1)
    assert (len(ev2["bos_events"]), len(ev2["choch_events"])) == before
