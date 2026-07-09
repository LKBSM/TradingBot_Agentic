"""Conditions Scanner — read-only structural condition matching over existing reads.

This module evaluates a user-defined set of **structural conditions** (their
"strategy") against MarketReading payloads that the detection engine has
**already produced**. It is strictly DESCRIPTIVE and PRESENT-TENSE:

- Every condition in the palette is a fact about the market *right now*
  (e.g. "price is currently inside an Order Block"). There is **no** predictive
  / outcome condition ("will bounce", "will break") — such types are not even
  representable here (see :data:`PALETTE` / :data:`ALLOWED_CONDITION_TYPES`).
- The evaluator is a **pure function** over a reading dict. It never fetches,
  never detects, never mutates anything. The scan endpoint feeds it readings
  obtained via a read-only store accessor.
- Matching is **transparent**: each combo reports which conditions are met AND
  which are unmet, plus the full context (including what goes against). There is
  no opaque similarity score and no quality ranking.

The palette is the single source of truth for what the builder may offer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Palette — present-tense structural conditions ONLY
# ---------------------------------------------------------------------------

#: Direction filter accepted by direction-aware conditions.
DIRECTION_VALUES = ("any", "bullish", "bearish")

#: Allowed values for the regime selectors (present-tense facts in the reading).
TREND_VALUES = ("bullish", "bearish", "ranging", "neutral")
PHASE_VALUES = ("accumulation", "distribution", "trend", "ranging", "expansion")
VOLATILITY_VALUES = ("low", "normal", "elevated")

#: The complete, closed palette. Each entry is a fact AT THE PRESENT. Adding a
#: predictive/outcome condition here is forbidden — the test-suite asserts every
#: entry carries ``tense == "present"`` and uses no predictive vocabulary.
PALETTE: List[Dict[str, Any]] = [
    {
        "type": "mtf_aligned",
        "label": "3 TF alignés",
        "description": (
            "Les 3 timeframes (H4, H1, M15) pointent dans la même direction "
            "en ce moment."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "trend_is",
        "label": "Tendance actuelle",
        "description": (
            "La tendance observée sur ce timeframe est, en ce moment, "
            "celle choisie (haussière, baissière ou en range)."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "market_phase_is",
        "label": "Phase de marché",
        "description": (
            "La phase de marché observée correspond, en ce moment, à celle "
            "choisie (accumulation, distribution, tendance, range, expansion)."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "volatility_is",
        "label": "Volatilité observée",
        "description": (
            "La volatilité observée en ce moment correspond au niveau choisi "
            "(faible, normale, élevée)."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "price_in_ob",
        "label": "Prix dans un Order Block",
        "description": (
            "Le prix courant se situe à l'intérieur d'un Order Block actif."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "price_in_fvg",
        "label": "Prix dans un Fair Value Gap",
        "description": (
            "Le prix courant se situe à l'intérieur d'un Fair Value Gap "
            "non comblé."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "ob_fvg_confluence",
        "label": "Confluence OB + FVG au prix courant",
        "description": (
            "Le prix courant se situe simultanément dans un Order Block actif "
            "et dans un Fair Value Gap non comblé."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "bos_recent_confirmed",
        "label": "BOS confirmé récent",
        "description": (
            "Une cassure de structure (BOS) confirmée est datée des dernières "
            "bougies."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "choch_recent_confirmed",
        "label": "CHOCH confirmé récent",
        "description": (
            "Un changement de caractère (CHOCH) confirmé est daté des "
            "dernières bougies."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "retest_in_progress",
        "label": "Retest en cours",
        "description": (
            "Un retest d'un niveau (BOS, CHOCH, OB ou FVG) est en cours "
            "en ce moment."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "price_near_ob",
        "label": "Prix proche d'un Order Block",
        "description": (
            "Le prix courant est proche d'un Order Block actif (à moins de la "
            "distance choisie), sans être nécessairement à l'intérieur."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "price_near_fvg",
        "label": "Prix proche d'un Fair Value Gap",
        "description": (
            "Le prix courant est proche d'un Fair Value Gap non comblé "
            "(à moins de la distance choisie), sans être nécessairement dedans."
        ),
        "supports_direction": True,
        "tense": "present",
    },
    {
        "type": "price_near_liquidity",
        "label": "Prix proche d'une liquidité (SSL/BSL)",
        "description": (
            "Le prix courant est proche d'une poche de liquidité intacte "
            "(BSL au-dessus / SSL en dessous), à moins de la distance choisie."
        ),
        "supports_direction": False,
        "tense": "present",
    },
    {
        "type": "liquidity_swept_recent",
        "label": "Prise de liquidité récente",
        "description": (
            "Une poche de liquidité (SSL/BSL) a été balayée au cours des "
            "dernières bougies — un point de prise de liquidité observé."
        ),
        "supports_direction": False,
        "tense": "present",
    },
]

#: Closed allowlist of condition types. Anything outside is rejected by the
#: endpoint — there is intentionally no path to a predictive condition.
ALLOWED_CONDITION_TYPES = frozenset(p["type"] for p in PALETTE)

_PALETTE_BY_TYPE = {p["type"]: p for p in PALETTE}

#: Minutes per supported timeframe — used only to express BOS recency in bars.
_TF_MINUTES = {"M15": 15, "H1": 60, "H4": 240}

#: Default recency window (in bars) for ``bos_recent_confirmed``.
DEFAULT_BOS_MAX_BARS = 5

#: Default proximity threshold (in % of price) for the "price near …" conditions.
DEFAULT_PROXIMITY_PCT = 0.3

#: Default recency window (in bars) for ``liquidity_swept_recent``.
DEFAULT_LIQ_MAX_BARS = 10

#: Accepted liquidity-side filter values.
LIQUIDITY_SIDE_VALUES = ("any", "bsl", "ssl")


# ---------------------------------------------------------------------------
# Small read-only helpers
# ---------------------------------------------------------------------------


def _label_for(cond_type: str) -> str:
    entry = _PALETTE_BY_TYPE.get(cond_type)
    return entry["label"] if entry else cond_type


def _direction_word(direction: str) -> str:
    return {"bullish": "haussier", "bearish": "baissier"}.get(direction, "")


def _trend_axis(value: Optional[str]) -> str:
    """Collapse a trend/bias value to an axis: 'up' / 'down' / 'flat'."""
    if value == "bullish":
        return "up"
    if value == "bearish":
        return "down"
    return "flat"


def _zone_bounds(zone: Dict[str, Any]) -> Optional[tuple]:
    """Return (low, high) of a zone, robust to either field ordering."""
    hi = zone.get("level_high")
    lo = zone.get("level_low")
    if hi is None or lo is None:
        return None
    return (min(lo, hi), max(lo, hi))


def _direction_matches(zone_dir: Optional[str], wanted: str) -> bool:
    if wanted == "any":
        return True
    return zone_dir == wanted


def _parse_dt(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _bars_between(start_iso: Any, end_iso: Any, timeframe: str) -> Optional[float]:
    """Approximate elapsed bars between two timestamps for a timeframe.

    Returns ``None`` when either timestamp is unparseable so callers can treat
    recency as unknown (and therefore conservatively *not* recent).
    """
    start = _parse_dt(start_iso)
    end = _parse_dt(end_iso)
    minutes = _TF_MINUTES.get(timeframe)
    if start is None or end is None or not minutes:
        return None
    delta_min = (end - start).total_seconds() / 60.0
    if delta_min < 0:
        return None
    return delta_min / minutes


# ---------------------------------------------------------------------------
# Per-condition evaluation (pure)
# ---------------------------------------------------------------------------


def _result(cond_type: str, met: bool, detail: str, *, available: bool = True) -> Dict[str, Any]:
    """A single condition outcome.

    ``available`` distinguishes "the data needed to judge this condition is
    missing" (``available=False``, e.g. a sibling timeframe has no reading yet)
    from "the condition was judged and is not met" (``available=True, met=False``).
    A condition that is not available is never met. The UI surfaces the two
    states differently so the client never reads a data gap as a real "non".
    """
    if not available:
        met = False
    return {
        "type": cond_type,
        "label": _label_for(cond_type),
        "met": met,
        "available": available,
        "detail": detail,
    }


#: Display order + label for the multi-timeframe alignment (highest → lowest),
#: mirroring the frontend ``MTF_TREND_ORDER`` so the scanner and the chart's
#: "Régime" panel describe alignment identically.
_MTF_ALIGN_TFS = (("H4", "h4"), ("H1", "h1"), ("M15", "m15"))

_TREND_ADJ = {"bullish": "haussier", "bearish": "baissier", "neutral": "neutre", "ranging": "en range"}


def _eval_mtf_aligned(
    reading: Dict[str, Any], direction: str, instrument_trends: Optional[Dict[str, Optional[str]]]
) -> Dict[str, Any]:
    """Are the instrument's M15/H1/H4 trends aligned in one direction right now?

    Reads each timeframe's OWN ``regime.trend`` (the same source the chart's
    "Régime" panel uses via ``useMtfTrends``) — NOT the per-reading
    ``mtf_confluence`` field, which by construction only carries the bias of
    timeframes ABOVE the current one and therefore can never hold all three.
    ``instrument_trends`` maps "M15"/"H1"/"H4" → trend for THIS instrument,
    assembled by the scan route from the sibling readings it already loads.
    Alignment semantics match the frontend ``describeMtfAlignment`` exactly.
    """
    trends = instrument_trends or {}
    by_tf = {tf: trends.get(tf) for tf, _ in _MTF_ALIGN_TFS}
    present = {tf: v for tf, v in by_tf.items() if v}
    summary = ", ".join(
        f"{tf} {(_TREND_ADJ.get(by_tf[tf]) or by_tf[tf]) if by_tf[tf] else '·'}"
        for tf, _ in _MTF_ALIGN_TFS
    )

    # Need all three timeframes to judge "3 TF alignés". A missing sibling read
    # is a DATA gap, not a "no" — surface it as unavailable.
    missing = [tf for tf, _ in _MTF_ALIGN_TFS if not by_tf[tf]]
    if missing:
        return _result(
            "mtf_aligned", False,
            f"Alignement indisponible — lecture manquante : {', '.join(missing)} ({summary}).",
            available=False,
        )

    axes = {tf: _trend_axis(by_tf[tf]) for tf, _ in _MTF_ALIGN_TFS}
    axis_values = list(axes.values())
    axes_aligned = axis_values[0] != "flat" and all(a == axis_values[0] for a in axis_values)

    if axes_aligned:
        word = "haussiers" if axis_values[0] == "up" else "baissiers"
        if direction == "any" or axis_values[0] == _trend_axis(direction):
            return _result("mtf_aligned", True, f"Les 3 TF sont alignés ({word}) — {summary}.")
        # Aligned, but in the opposite direction to the one requested.
        wanted = "haussiers" if direction == "bullish" else "baissiers"
        return _result(
            "mtf_aligned", False,
            f"Les 3 TF sont alignés ({word}), pas {wanted} comme demandé — {summary}.",
        )

    # Not aligned: mirror the chart's descriptive phrasing so both surfaces agree.
    h4, h1, m15 = axes["H4"], axes["H1"], axes["M15"]
    if h4 == h1 != "flat" and m15 != "flat" and m15 != h4:
        fem = "haussière" if h4 == "up" else "baissière"
        return _result("mtf_aligned", False, f"M15 se replie contre la tendance H4 {fem} — {summary}.")
    if all(a == "flat" for a in axis_values):
        return _result("mtf_aligned", False, f"Les 3 TF sont neutres — {summary}.")
    return _result("mtf_aligned", False, f"Les TF divergent — {summary}.")


def _active_obs_at_price(reading: Dict[str, Any], price: float, direction: str) -> List[Dict[str, Any]]:
    out = []
    for ob in reading.get("structure", {}).get("order_blocks", []) or []:
        if ob.get("status") != "active":
            continue
        bounds = _zone_bounds(ob)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < price < hi and _direction_matches(ob.get("direction"), direction):
            out.append(ob)
    return out


def _active_fvgs_at_price(reading: Dict[str, Any], price: float, direction: str) -> List[Dict[str, Any]]:
    out = []
    for fvg in reading.get("structure", {}).get("fair_value_gaps", []) or []:
        if fvg.get("status") not in ("active", "partially_filled"):
            continue
        bounds = _zone_bounds(fvg)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < price < hi and _direction_matches(fvg.get("direction"), direction):
            out.append(fvg)
    return out


def _eval_price_in_ob(reading: Dict[str, Any], direction: str) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_ob", False, "Prix courant indisponible.", available=False)
    hits = _active_obs_at_price(reading, price, direction)
    if hits:
        dirs = ", ".join(sorted({h.get("direction") or "n/d" for h in hits}))
        return _result(
            "price_in_ob", True,
            f"Prix dans {len(hits)} Order Block actif(s) (direction : {dirs}).",
        )
    suffix = f" {_direction_word(direction)}" if direction != "any" else ""
    return _result("price_in_ob", False, f"Prix hors de tout Order Block actif{suffix}.")


def _eval_price_in_fvg(reading: Dict[str, Any], direction: str) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_fvg", False, "Prix courant indisponible.", available=False)
    hits = _active_fvgs_at_price(reading, price, direction)
    if hits:
        dirs = ", ".join(sorted({h.get("direction") or "n/d" for h in hits}))
        return _result(
            "price_in_fvg", True,
            f"Prix dans {len(hits)} Fair Value Gap non comblé(s) (direction : {dirs}).",
        )
    suffix = f" {_direction_word(direction)}" if direction != "any" else ""
    return _result("price_in_fvg", False, f"Prix hors de tout Fair Value Gap non comblé{suffix}.")


def _eval_ob_fvg_confluence(reading: Dict[str, Any]) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("ob_fvg_confluence", False, "Prix courant indisponible.", available=False)
    in_ob = bool(_active_obs_at_price(reading, price, "any"))
    in_fvg = bool(_active_fvgs_at_price(reading, price, "any"))
    if in_ob and in_fvg:
        return _result("ob_fvg_confluence", True, "Prix dans un OB actif ET un FVG non comblé.")
    have = []
    if in_ob:
        have.append("OB")
    if in_fvg:
        have.append("FVG")
    detail = (
        f"Seulement {have[0]} au prix courant (pas de confluence)."
        if have
        else "Ni OB ni FVG au prix courant."
    )
    return _result("ob_fvg_confluence", False, detail)


def _eval_break_recent_confirmed(
    reading: Dict[str, Any], cond_type: str, struct_key: str, name: str,
    direction: str, max_bars: int,
) -> Dict[str, Any]:
    """Shared evaluator for a confirmed, recent structural break (BOS or CHOCH)."""
    rec = reading.get("structure", {}).get(struct_key)
    if not rec:
        return _result(cond_type, False, f"Aucun {name} récent.")
    if rec.get("validation_status") != "confirmed":
        status = rec.get("validation_status", "inconnu")
        return _result(cond_type, False, f"{name} non confirmé (statut : {status}).")
    rec_dir = rec.get("direction")
    if not _direction_matches(rec_dir, direction):
        return _result(
            cond_type, False,
            f"{name} confirmé mais {rec_dir or 'n/d'} (≠ {_direction_word(direction)} demandé).",
        )
    timeframe = reading.get("header", {}).get("timeframe", "")
    candle_ts = reading.get("header", {}).get("candle_close_ts")
    bars = _bars_between(rec.get("broken_at"), candle_ts, timeframe)
    word = _direction_word(rec_dir) or rec_dir or "n/d"
    if bars is None:
        return _result(
            cond_type, False,
            f"{name} {word} confirmé mais ancienneté inconnue.", available=False,
        )
    if bars <= max_bars:
        return _result(
            cond_type, True,
            f"{name} {word} confirmé il y a ~{round(bars)} bougie(s) (≤ {max_bars}).",
        )
    return _result(
        cond_type, False,
        f"{name} {word} confirmé mais trop ancien (~{round(bars)} bougies > {max_bars}).",
    )


def _eval_trend_is(reading: Dict[str, Any], trend: Optional[str]) -> Dict[str, Any]:
    if not trend:
        return _result("trend_is", False, "Tendance cible non précisée.")
    observed = reading.get("regime", {}).get("trend")
    if not observed:
        return _result("trend_is", False, "Tendance observée indisponible.", available=False)
    return _result(
        "trend_is", observed == trend,
        f"Tendance observée : {observed} (cible : {trend}).",
    )


def _eval_market_phase_is(reading: Dict[str, Any], phase: Optional[str]) -> Dict[str, Any]:
    if not phase:
        return _result("market_phase_is", False, "Phase cible non précisée.")
    observed = reading.get("regime", {}).get("market_phase")
    if not observed:
        return _result("market_phase_is", False, "Phase observée indisponible.", available=False)
    return _result(
        "market_phase_is", observed == phase,
        f"Phase observée : {observed} (cible : {phase}).",
    )


def _eval_volatility_is(reading: Dict[str, Any], volatility: Optional[str]) -> Dict[str, Any]:
    if not volatility:
        return _result("volatility_is", False, "Niveau de volatilité cible non précisé.")
    observed = reading.get("regime", {}).get("volatility_observed")
    if not observed:
        return _result("volatility_is", False, "Volatilité observée indisponible.", available=False)
    return _result(
        "volatility_is", observed == volatility,
        f"Volatilité observée : {observed} (cible : {volatility}).",
    )


def _eval_retest_in_progress(reading: Dict[str, Any]) -> Dict[str, Any]:
    retest = reading.get("structure", {}).get("retest_in_progress")
    if retest:
        kind = retest.get("type", "niveau")
        return _result("retest_in_progress", True, f"Retest en cours ({kind}).")
    return _result("retest_in_progress", False, "Aucun retest en cours.")


def _distance_pct(price: float, lo: float, hi: float) -> float:
    """% distance from ``price`` to the ``[lo, hi]`` band (0.0 when inside)."""
    if not price:
        return float("inf")
    if lo <= price <= hi:
        return 0.0
    edge = lo if price < lo else hi
    return abs(price - edge) / price * 100.0


def _nearest_zone_pct(
    reading: Dict[str, Any], struct_key: str, active_statuses: tuple, direction: str
) -> Optional[float]:
    """Smallest %-distance from price to an active zone of ``struct_key`` (or None)."""
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return None
    best: Optional[float] = None
    for z in reading.get("structure", {}).get(struct_key, []) or []:
        if z.get("status") not in active_statuses:
            continue
        bounds = _zone_bounds(z)
        if bounds is None or not _direction_matches(z.get("direction"), direction):
            continue
        d = _distance_pct(price, bounds[0], bounds[1])
        if best is None or d < best:
            best = d
    return best


def _eval_price_near_zone(
    reading: Dict[str, Any], cond_type: str, struct_key: str, name: str,
    active_statuses: tuple, direction: str, proximity_pct: float,
) -> Dict[str, Any]:
    """Shared evaluator for « price near an OB / FVG » (within proximity_pct %)."""
    if reading.get("header", {}).get("close_price") is None:
        return _result(cond_type, False, "Prix courant indisponible.", available=False)
    best = _nearest_zone_pct(reading, struct_key, active_statuses, direction)
    if best is None:
        suffix = f" {_direction_word(direction)}" if direction != "any" else ""
        return _result(cond_type, False, f"Aucun {name} actif{suffix}.")
    if best <= proximity_pct:
        where = "dedans" if best == 0.0 else f"à ~{best:.2f} %"
        return _result(
            cond_type, True,
            f"Prix proche d'un {name} actif ({where} ; seuil {proximity_pct:.2f} %).",
        )
    return _result(
        cond_type, False,
        f"{name} actif le plus proche à ~{best:.2f} % (> seuil {proximity_pct:.2f} %).",
    )


def _eval_price_near_liquidity(
    reading: Dict[str, Any], side: str, proximity_pct: float
) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_near_liquidity", False, "Prix courant indisponible.", available=False)
    best: Optional[float] = None
    best_side: Optional[str] = None
    for lp in reading.get("structure", {}).get("liquidity_pools", []) or []:
        if lp.get("status") != "intact":  # only resting (not-yet-taken) liquidity
            continue
        if side != "any" and lp.get("side") != side:
            continue
        level = lp.get("level")
        if level is None or not price:
            continue
        d = abs(price - level) / price * 100.0
        if best is None or d < best:
            best = d
            best_side = lp.get("side")
    if best is None:
        s = f" {side.upper()}" if side != "any" else ""
        return _result("price_near_liquidity", False, f"Aucune poche de liquidité{s} intacte.")
    if best <= proximity_pct:
        return _result(
            "price_near_liquidity", True,
            f"Prix proche d'une liquidité {(best_side or '').upper()} intacte "
            f"(à ~{best:.2f} % ; seuil {proximity_pct:.2f} %).",
        )
    return _result(
        "price_near_liquidity", False,
        f"Liquidité intacte la plus proche à ~{best:.2f} % (> seuil {proximity_pct:.2f} %).",
    )


def _eval_liquidity_swept_recent(
    reading: Dict[str, Any], side: str, max_bars: int
) -> Dict[str, Any]:
    tf = reading.get("header", {}).get("timeframe", "")
    candle_ts = reading.get("header", {}).get("candle_close_ts")
    best_bars: Optional[float] = None
    best_side: Optional[str] = None
    for lp in reading.get("structure", {}).get("liquidity_pools", []) or []:
        if lp.get("status") != "swept":
            continue
        if side != "any" and lp.get("side") != side:
            continue
        bars = _bars_between(lp.get("swept_at"), candle_ts, tf)
        if bars is None:
            continue
        if best_bars is None or bars < best_bars:
            best_bars = bars
            best_side = lp.get("side")
    if best_bars is None:
        s = f" {side.upper()}" if side != "any" else ""
        return _result("liquidity_swept_recent", False, f"Aucune prise de liquidité{s} récente.")
    if best_bars <= max_bars:
        return _result(
            "liquidity_swept_recent", True,
            f"Liquidité {(best_side or '').upper()} balayée il y a ~{round(best_bars)} "
            f"bougie(s) (≤ {max_bars}).",
        )
    return _result(
        "liquidity_swept_recent", False,
        f"Prise de liquidité trop ancienne (~{round(best_bars)} bougies > {max_bars}).",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_condition(
    reading: Dict[str, Any],
    cond: Dict[str, Any],
    instrument_trends: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Evaluate a single condition against a reading payload (pure).

    ``cond`` is ``{"type": str, "direction"?: str, "max_bars"?: int}``.
    Returns ``{"type", "label", "met": bool, "available": bool, "detail": str}``.
    ``instrument_trends`` maps "M15"/"H1"/"H4" → the instrument's per-timeframe
    trend (used only by ``mtf_aligned``); the scan route supplies it from the
    sibling readings it loads. Unknown types raise ``ValueError`` (callers
    validate against the palette).
    """
    cond_type = cond.get("type")
    direction = cond.get("direction", "any") or "any"
    if cond_type == "mtf_aligned":
        return _eval_mtf_aligned(reading, direction, instrument_trends)
    if cond_type == "trend_is":
        return _eval_trend_is(reading, cond.get("trend"))
    if cond_type == "market_phase_is":
        return _eval_market_phase_is(reading, cond.get("phase"))
    if cond_type == "volatility_is":
        return _eval_volatility_is(reading, cond.get("volatility"))
    if cond_type == "price_in_ob":
        return _eval_price_in_ob(reading, direction)
    if cond_type == "price_in_fvg":
        return _eval_price_in_fvg(reading, direction)
    if cond_type == "ob_fvg_confluence":
        return _eval_ob_fvg_confluence(reading)
    if cond_type == "bos_recent_confirmed":
        max_bars = int(cond.get("max_bars") or DEFAULT_BOS_MAX_BARS)
        return _eval_break_recent_confirmed(
            reading, "bos_recent_confirmed", "bos", "BOS", direction, max_bars
        )
    if cond_type == "choch_recent_confirmed":
        max_bars = int(cond.get("max_bars") or DEFAULT_BOS_MAX_BARS)
        return _eval_break_recent_confirmed(
            reading, "choch_recent_confirmed", "choch", "CHOCH", direction, max_bars
        )
    if cond_type == "retest_in_progress":
        return _eval_retest_in_progress(reading)
    if cond_type == "price_near_ob":
        prox = float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT)
        return _eval_price_near_zone(
            reading, "price_near_ob", "order_blocks", "Order Block",
            ("active",), direction, prox,
        )
    if cond_type == "price_near_fvg":
        prox = float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT)
        return _eval_price_near_zone(
            reading, "price_near_fvg", "fair_value_gaps", "Fair Value Gap",
            ("active", "partially_filled"), direction, prox,
        )
    if cond_type == "price_near_liquidity":
        prox = float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT)
        return _eval_price_near_liquidity(reading, cond.get("side", "any") or "any", prox)
    if cond_type == "liquidity_swept_recent":
        max_bars = int(cond.get("max_bars") or DEFAULT_LIQ_MAX_BARS)
        return _eval_liquidity_swept_recent(reading, cond.get("side", "any") or "any", max_bars)
    raise ValueError(f"Unknown condition type: {cond_type!r}")


def build_context(reading: Dict[str, Any]) -> Dict[str, Any]:
    """A neutral, full-context summary of a reading — including what goes against."""
    structure = reading.get("structure", {})
    regime = reading.get("regime", {})
    events = reading.get("events", {})

    def _zone_summary(z: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not z:
            return None
        return {
            "direction": z.get("direction"),
            "level": z.get("level"),
            "validation_status": z.get("validation_status"),
        }

    obs = structure.get("order_blocks", []) or []
    fvgs = structure.get("fair_value_gaps", []) or []
    return {
        "trend": regime.get("trend"),
        "market_phase": regime.get("market_phase"),
        "volatility_observed": regime.get("volatility_observed"),
        "mtf_confluence": regime.get("mtf_confluence", {}) or {},
        "bos": _zone_summary(structure.get("bos")),
        "choch": _zone_summary(structure.get("choch")),
        "active_order_blocks": sum(1 for o in obs if o.get("status") == "active"),
        "active_fair_value_gaps": sum(
            1 for f in fvgs if f.get("status") in ("active", "partially_filled")
        ),
        "news_upcoming": [
            {
                "event": n.get("event"),
                "impact": n.get("impact"),
                "time_to_event_min": n.get("time_to_event_min"),
            }
            for n in (events.get("news_upcoming", []) or [])
        ],
    }


def evaluate_reading(
    reading: Dict[str, Any],
    conditions: List[Dict[str, Any]],
    logic: str,
    instrument_trends: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    """Evaluate all conditions against one reading and assemble a combo result.

    ``logic`` is ``"AND"`` (all met) or ``"OR"`` (any met). Returns the per-combo
    breakdown with met/unmet conditions and the full neutral context.
    ``instrument_trends`` (M15/H1/H4 → trend for this instrument) is forwarded to
    ``mtf_aligned``. Pure: never mutates ``reading`` or any external state.
    """
    results = [evaluate_condition(reading, c, instrument_trends) for c in conditions]
    met = [r for r in results if r["met"]]
    unmet = [r for r in results if not r["met"]]
    if logic == "OR":
        matched = any(r["met"] for r in results)
    else:  # default / "AND"
        matched = all(r["met"] for r in results) and bool(results)

    header = reading.get("header", {})
    context = build_context(reading)
    # Surface the authoritative per-timeframe trends (each TF's own regime.trend,
    # same source as mtf_aligned and the chart's Régime panel) so the result card
    # shows real alignment — NOT the structurally-incomplete mtf_confluence.
    if instrument_trends:
        context["mtf_trends"] = {
            "h4": instrument_trends.get("H4"),
            "h1": instrument_trends.get("H1"),
            "m15": instrument_trends.get("M15"),
        }
    return {
        "instrument": header.get("instrument"),
        "timeframe": header.get("timeframe"),
        "candle_close_ts": header.get("candle_close_ts"),
        "close_price": header.get("close_price"),
        "matched": matched,
        "met_count": len(met),
        "total": len(results),
        "conditions_met": met,
        "conditions_unmet": unmet,
        "context": context,
    }


__all__ = [
    "ALLOWED_CONDITION_TYPES",
    "DEFAULT_BOS_MAX_BARS",
    "DEFAULT_LIQ_MAX_BARS",
    "DEFAULT_PROXIMITY_PCT",
    "DIRECTION_VALUES",
    "LIQUIDITY_SIDE_VALUES",
    "PALETTE",
    "build_context",
    "evaluate_condition",
    "evaluate_reading",
]
