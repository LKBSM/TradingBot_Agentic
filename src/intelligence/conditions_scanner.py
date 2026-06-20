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
]

#: Closed allowlist of condition types. Anything outside is rejected by the
#: endpoint — there is intentionally no path to a predictive condition.
ALLOWED_CONDITION_TYPES = frozenset(p["type"] for p in PALETTE)

_PALETTE_BY_TYPE = {p["type"]: p for p in PALETTE}

#: Minutes per supported timeframe — used only to express BOS recency in bars.
_TF_MINUTES = {"M15": 15, "H1": 60, "H4": 240}

#: Default recency window (in bars) for ``bos_recent_confirmed``.
DEFAULT_BOS_MAX_BARS = 5


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


def _result(cond_type: str, met: bool, detail: str) -> Dict[str, Any]:
    return {"type": cond_type, "label": _label_for(cond_type), "met": met, "detail": detail}


def _eval_mtf_aligned(reading: Dict[str, Any], direction: str) -> Dict[str, Any]:
    mtf = reading.get("regime", {}).get("mtf_confluence", {}) or {}
    h4, h1, m15 = mtf.get("h4"), mtf.get("h1"), mtf.get("m15")
    present = [v for v in (h4, h1, m15) if v]
    summary = f"H4 {h4 or '?'}, H1 {h1 or '?'}, M15 {m15 or '?'}"
    if len(present) < 3:
        return _result("mtf_aligned", False, f"Données MTF incomplètes ({summary}).")

    axes = [_trend_axis(h4), _trend_axis(h1), _trend_axis(m15)]
    aligned = axes[0] != "flat" and all(a == axes[0] for a in axes)
    if direction != "any" and aligned:
        aligned = axes[0] == _trend_axis(direction)

    if aligned:
        word = "haussiers" if axes[0] == "up" else "baissiers"
        return _result("mtf_aligned", True, f"Les 3 TF sont alignés ({word}) — {summary}.")
    return _result("mtf_aligned", False, f"TF non alignés — {summary}.")


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
        return _result("price_in_ob", False, "Prix courant indisponible.")
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
        return _result("price_in_fvg", False, "Prix courant indisponible.")
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
        return _result("ob_fvg_confluence", False, "Prix courant indisponible.")
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


def _eval_bos_recent_confirmed(reading: Dict[str, Any], direction: str, max_bars: int) -> Dict[str, Any]:
    bos = reading.get("structure", {}).get("bos")
    if not bos:
        return _result("bos_recent_confirmed", False, "Aucun BOS récent.")
    if bos.get("validation_status") != "confirmed":
        status = bos.get("validation_status", "inconnu")
        return _result("bos_recent_confirmed", False, f"BOS non confirmé (statut : {status}).")
    bos_dir = bos.get("direction")
    if not _direction_matches(bos_dir, direction):
        return _result(
            "bos_recent_confirmed", False,
            f"BOS confirmé mais {bos_dir or 'n/d'} (≠ {_direction_word(direction)} demandé).",
        )
    timeframe = reading.get("header", {}).get("timeframe", "")
    candle_ts = reading.get("header", {}).get("candle_close_ts")
    bars = _bars_between(bos.get("broken_at"), candle_ts, timeframe)
    word = _direction_word(bos_dir) or bos_dir or "n/d"
    if bars is None:
        return _result("bos_recent_confirmed", False, f"BOS {word} confirmé mais ancienneté inconnue.")
    if bars <= max_bars:
        return _result(
            "bos_recent_confirmed", True,
            f"BOS {word} confirmé il y a ~{round(bars)} bougie(s) (≤ {max_bars}).",
        )
    return _result(
        "bos_recent_confirmed", False,
        f"BOS {word} confirmé mais trop ancien (~{round(bars)} bougies > {max_bars}).",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_condition(reading: Dict[str, Any], cond: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a single condition against a reading payload (pure).

    ``cond`` is ``{"type": str, "direction"?: str, "max_bars"?: int}``.
    Returns ``{"type", "label", "met": bool, "detail": str}``.
    Unknown types raise ``ValueError`` (callers validate against the palette).
    """
    cond_type = cond.get("type")
    direction = cond.get("direction", "any") or "any"
    if cond_type == "mtf_aligned":
        return _eval_mtf_aligned(reading, direction)
    if cond_type == "price_in_ob":
        return _eval_price_in_ob(reading, direction)
    if cond_type == "price_in_fvg":
        return _eval_price_in_fvg(reading, direction)
    if cond_type == "ob_fvg_confluence":
        return _eval_ob_fvg_confluence(reading)
    if cond_type == "bos_recent_confirmed":
        max_bars = int(cond.get("max_bars") or DEFAULT_BOS_MAX_BARS)
        return _eval_bos_recent_confirmed(reading, direction, max_bars)
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
    reading: Dict[str, Any], conditions: List[Dict[str, Any]], logic: str
) -> Dict[str, Any]:
    """Evaluate all conditions against one reading and assemble a combo result.

    ``logic`` is ``"AND"`` (all met) or ``"OR"`` (any met). Returns the per-combo
    breakdown with met/unmet conditions and the full neutral context. Pure: never
    mutates ``reading`` or any external state.
    """
    results = [evaluate_condition(reading, c) for c in conditions]
    met = [r for r in results if r["met"]]
    unmet = [r for r in results if not r["met"]]
    if logic == "OR":
        matched = any(r["met"] for r in results)
    else:  # default / "AND"
        matched = all(r["met"] for r in results) and bool(results)

    header = reading.get("header", {})
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
        "context": build_context(reading),
    }


__all__ = [
    "ALLOWED_CONDITION_TYPES",
    "DEFAULT_BOS_MAX_BARS",
    "DIRECTION_VALUES",
    "PALETTE",
    "build_context",
    "evaluate_condition",
    "evaluate_reading",
]
