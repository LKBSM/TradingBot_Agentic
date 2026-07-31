"""Conditions Scanner — read-only structural condition matching over existing reads.

This module evaluates a user-defined set of **structural conditions** (their
"reading") against MarketReading payloads that the detection engine has
**already produced** (SC-1). It is strictly DESCRIPTIVE and PRESENT-TENSE:

- Every condition in the palette is a fact about the market *right now*
  (e.g. "price is currently inside an Order Block") or a dated past event counted
  IN BARS (e.g. "a CHOCH occurred within the last 10 candles"). There is **no**
  predictive / outcome condition ("will bounce", "will break") — such types are
  not even representable here (see :data:`PALETTE` / :data:`ALLOWED_CONDITION_TYPES`).
- The evaluator is a **pure function** over a reading dict. It never fetches,
  never detects, never mutates anything. The scan endpoint feeds it readings
  obtained via a read-only store accessor.
- Matching is **transparent**: each combo reports which conditions are met, which
  are unmet, and which are NON-EVALUABLE (a third state that adjusts the visible
  denominator), plus the full context — including what goes against. There is no
  opaque similarity score and no quality ranking. Order is fixed (market, then
  timeframe); nothing here sorts by how many conditions a combo meets.

The palette (grouped in four families: structure / zones / liquidity / context)
is the single source of truth for what the builder may offer, so the interface
DERIVES from the schema instead of copying it.

SC-1 palette changes vs the legacy 14-type palette:
- REMOVED ``ob_fvg_confluence`` — it is exactly « price in OB » AND « price in
  FVG »; offering the same thing twice under a third name is forbidden.
- REMOVED the legacy ``retest_in_progress`` (a BOS-level retest state-machine
  flag) and REPLACED it with the factual :data:`price_in_tested_zone` (« price is
  inside an OB/FVG that has already been tested at least once »).
- The fixed 3-TF ``mtf_aligned`` is REMOVED in favour of :func:`_eval_higher_tf_agrees`
  (« l'unité supérieure va… »): a RELATIVE comparison to the immediately-higher
  unit, which works identically on all six units (a fixed set breaks at the top —
  nothing above D1). The compared unit is NAMED in the result; on the highest
  tracked unit the condition is non-evaluable (C1).
- ``distribution`` is REMOVED from the exposed market-phase values: the phase
  derivation can no longer emit it (see the audit), and a condition that can only
  ever return zero is worse than an absent one.
- ``zone_tested_at_most`` (« tested at most N times ») is DOCUMENTED in
  :data:`BLOCKED_PALETTE` but NOT exposed: OB/FVG carry only a ``tested`` boolean,
  not a touch count. It becomes offerable once that count exists in the engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Palette vocabulary — present-tense structural values ONLY
# ---------------------------------------------------------------------------

#: The four families the interface groups conditions under. Structure and Zones
#: open by default (see the builder). Order is fixed and meaningful.
FAMILIES: Tuple[str, ...] = ("structure", "zones", "liquidity", "context")

#: Direction filter accepted by direction-aware conditions.
DIRECTION_VALUES = ("any", "bullish", "bearish")

#: Structural trend (TR-1): bullish / bearish / indeterminate. ``ranging`` and
#: ``neutral`` are gone (consolidation lives on the Phase tile only).
TREND_VALUES = ("bullish", "bearish", "indeterminate")

#: Market-phase values OFFERED by the palette. ``distribution`` is intentionally
#: excluded: the engine's phase derivation has no path that emits it, so exposing
#: it would create a condition that can only ever return zero. Full engine
#: vocabulary (incl. the dead ``distribution``) stays in the schema; the palette
#: advertises only what the engine can actually produce.
PHASE_VALUES = ("accumulation", "trend", "ranging", "expansion")

#: Observed-volatility values.
VOLATILITY_VALUES = ("low", "normal", "elevated")

#: Liquidity-side filter (BSL above / SSL below the price).
LIQUIDITY_SIDE_VALUES = ("any", "bsl", "ssl")

#: Which zone family a zone-condition looks at.
ZONE_KIND_VALUES = ("any", "ob", "fvg")

#: The last detected structural event, as a signed break.
EVENT_VALUES = ("bos_up", "bos_down", "choch_up", "choch_down")

#: Age buckets (IN BARS) for « the last event dates back … ».
AGE_BUCKET_VALUES = ("lt10", "10to50", "gt50")

#: Which third of the structural range the price sits in.
RANGE_THIRD_VALUES = ("bottom", "middle", "top")

#: Equal-level kind for « equal highs / lows present ».
EQ_KIND_VALUES = ("any", "highs", "lows")

#: Intraday session (reading convention, from market_calendar — single source).
SESSION_VALUES = ("asia", "london", "new_york", "overlap")

#: How the immediately-higher timeframe relates to the scanned one (C1).
RELATION_VALUES = ("same", "opposite", "indeterminate")

#: Recency windows (IN BARS) offered for the « occurred within N candles » family.
BARS_RECENCY_CHOICES = (5, 10, 20, 50)
#: Recency windows (IN BARS) offered for « zone formed within N candles ».
BARS_FORMED_CHOICES = (10, 20, 50)
#: Proximity thresholds (% of price) for the « price near a zone » family.
PROXIMITY_ZONE_CHOICES = (0.1, 0.25, 0.5)
#: Proximity thresholds (% of price) for the « intact pocket near price » family.
PROXIMITY_LIQ_CHOICES = (0.25, 0.5, 1.0)

#: Defaults (used when a request omits a control — the request model repeats them).
DEFAULT_BOS_MAX_BARS = 10
DEFAULT_LIQ_MAX_BARS = 10
DEFAULT_FORMED_MAX_BARS = 20
DEFAULT_PROXIMITY_PCT = 0.25


def _control(name: str, values: Tuple[Any, ...], default: Any) -> Dict[str, Any]:
    """One segmented control descriptor the interface renders as buttons."""
    return {"name": name, "values": list(values), "default": default}


#: The complete, closed, family-grouped palette. Each entry is a fact AT THE
#: PRESENT (or a past event counted in bars). Adding a predictive/outcome
#: condition here is forbidden — the test-suite asserts every entry carries
#: ``tense == "present"`` and uses no predictive vocabulary. ``controls`` lets the
#: UI derive its segmented buttons from the schema instead of recopying it.
PALETTE: List[Dict[str, Any]] = [
    # ── STRUCTURE ────────────────────────────────────────────────────────────
    {
        "type": "trend_is",
        "family": "structure",
        "label": "La tendance structurelle est",
        "description": (
            "La tendance de STRUCTURE sur ce timeframe (dernière cassure BOS/CHOCH "
            "non contredite) est, en ce moment, celle choisie — haussière, "
            "baissière, ou indéterminée quand aucune cassure ne l'a établie."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("trend", TREND_VALUES, "bullish")],
    },
    {
        "type": "higher_tf_agrees",
        "family": "structure",
        "label": "L'unité supérieure va",
        "description": (
            "La tendance de structure de l'unité de temps IMMÉDIATEMENT supérieure "
            "va, en ce moment, dans le même sens que l'unité scannée, en sens "
            "opposé, ou reste indéterminée. L'unité comparée est nommée dans le "
            "résultat (« le 1 h va dans le même sens »). Sur l'unité la plus haute "
            "suivie, il n'y a pas d'unité supérieure : la condition est non "
            "évaluable — jamais remplie par défaut."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("relation", RELATION_VALUES, "same")],
    },
    {
        "type": "last_event_is",
        "family": "structure",
        "label": "Le dernier événement détecté est",
        "description": (
            "Le tout dernier événement de structure daté (le plus récent entre BOS "
            "et CHOCH, haussier ou baissier) est celui choisi."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("event", EVENT_VALUES, "bos_up")],
    },
    {
        "type": "bos_recent_confirmed",
        "family": "structure",
        "label": "Un BOS est survenu dans les N dernières bougies",
        "description": (
            "Une cassure de structure (BOS) confirmée est datée d'au plus N bougies "
            "— une distance comptée en bougies, jamais en minutes."
        ),
        "supports_direction": True,
        "tense": "present",
        "controls": [
            _control("direction", DIRECTION_VALUES, "any"),
            _control("max_bars", BARS_RECENCY_CHOICES, DEFAULT_BOS_MAX_BARS),
        ],
    },
    {
        "type": "choch_recent_confirmed",
        "family": "structure",
        "label": "Un CHOCH est survenu dans les N dernières bougies",
        "description": (
            "Un changement de caractère (CHOCH) confirmé est daté d'au plus N "
            "bougies — une distance comptée en bougies, jamais en minutes."
        ),
        "supports_direction": True,
        "tense": "present",
        "controls": [
            _control("direction", DIRECTION_VALUES, "any"),
            _control("max_bars", BARS_RECENCY_CHOICES, DEFAULT_BOS_MAX_BARS),
        ],
    },
    {
        "type": "last_event_age",
        "family": "structure",
        "label": "Le dernier événement remonte à",
        "description": (
            "Le dernier événement de structure daté remonte à moins de 10 bougies, "
            "entre 10 et 50, ou plus de 50 — compté en bougies."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("age_bucket", AGE_BUCKET_VALUES, "lt10")],
    },
    # ── ZONES ────────────────────────────────────────────────────────────────
    {
        "type": "price_in_ob",
        "family": "zones",
        "label": "Le prix est dans un Order Block",
        "description": "Le prix courant se situe à l'intérieur d'un Order Block actif.",
        "supports_direction": True,
        "tense": "present",
        "controls": [_control("direction", DIRECTION_VALUES, "any")],
    },
    {
        "type": "price_in_fvg",
        "family": "zones",
        "label": "Le prix est dans un Fair Value Gap",
        "description": "Le prix courant se situe à l'intérieur d'un Fair Value Gap non comblé.",
        "supports_direction": True,
        "tense": "present",
        "controls": [_control("direction", DIRECTION_VALUES, "any")],
    },
    {
        "type": "price_in_tested_zone",
        "family": "zones",
        "label": "Le prix est dans une zone déjà testée",
        "description": (
            "Le prix courant est à l'intérieur d'une zone active (Order Block ou "
            "Fair Value Gap) qui a déjà été testée au moins une fois depuis sa "
            "formation."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("zone_kind", ZONE_KIND_VALUES, "any")],
    },
    {
        "type": "zone_untested",
        "family": "zones",
        "label": "Une zone n'a jamais été testée depuis sa formation",
        "description": (
            "Il existe une zone active (Order Block ou Fair Value Gap) qui n'a "
            "jamais été retestée depuis sa formation."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("zone_kind", ZONE_KIND_VALUES, "any")],
    },
    {
        "type": "zone_formed_recent",
        "family": "zones",
        "label": "Une zone s'est formée dans les N dernières bougies",
        "description": (
            "Il existe une zone active (Order Block ou Fair Value Gap) formée il y "
            "a au plus N bougies — compté en bougies."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [
            _control("zone_kind", ZONE_KIND_VALUES, "any"),
            _control("max_bars", BARS_FORMED_CHOICES, DEFAULT_FORMED_MAX_BARS),
        ],
    },
    {
        "type": "price_near_ob",
        "family": "zones",
        "label": "Le prix est à moins de X % d'un Order Block, sans y être",
        "description": (
            "Le prix courant est à moins de la distance choisie d'un Order Block "
            "actif, SANS être à l'intérieur."
        ),
        "supports_direction": True,
        "tense": "present",
        "controls": [
            _control("direction", DIRECTION_VALUES, "any"),
            _control("proximity_pct", PROXIMITY_ZONE_CHOICES, 0.25),
        ],
    },
    {
        "type": "price_near_fvg",
        "family": "zones",
        "label": "Le prix est à moins de X % d'un Fair Value Gap, sans y être",
        "description": (
            "Le prix courant est à moins de la distance choisie d'un Fair Value Gap "
            "non comblé, SANS être à l'intérieur."
        ),
        "supports_direction": True,
        "tense": "present",
        "controls": [
            _control("direction", DIRECTION_VALUES, "any"),
            _control("proximity_pct", PROXIMITY_ZONE_CHOICES, 0.25),
        ],
    },
    # ── LIQUIDITY ────────────────────────────────────────────────────────────
    {
        "type": "price_near_liquidity",
        "family": "liquidity",
        "label": "Une poche intacte est à moins de X % du prix",
        "description": (
            "Le prix courant est à moins de la distance choisie d'une poche de "
            "liquidité INTACTE (BSL au-dessus / SSL en dessous)."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [
            _control("side", LIQUIDITY_SIDE_VALUES, "any"),
            _control("proximity_pct", PROXIMITY_LIQ_CHOICES, 0.5),
        ],
    },
    {
        "type": "liquidity_swept_recent",
        "family": "liquidity",
        "label": "Une poche a été prise dans les N dernières bougies",
        "description": (
            "Une poche de liquidité (SSL/BSL) a été balayée — sa liquidité prise — "
            "il y a au plus N bougies. Compté en bougies."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [
            _control("side", LIQUIDITY_SIDE_VALUES, "any"),
            _control("max_bars", BARS_RECENCY_CHOICES, DEFAULT_LIQ_MAX_BARS),
        ],
    },
    {
        "type": "liquidity_broken_recent",
        "family": "liquidity",
        "label": "Une poche a été cassée dans les N dernières bougies",
        "description": (
            "Une poche de liquidité (SSL/BSL) a été cassée nettement — clôture au "
            "travers, état terminal — il y a au plus N bougies. À distinguer d'une "
            "simple prise (balayage). Compté en bougies."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [
            _control("side", LIQUIDITY_SIDE_VALUES, "any"),
            _control("max_bars", BARS_RECENCY_CHOICES, DEFAULT_LIQ_MAX_BARS),
        ],
    },
    {
        "type": "equal_levels_present",
        "family": "liquidity",
        "label": "Des creux ou sommets égaux sont présents",
        "description": (
            "Au moins une poche de liquidité intacte formée de creux égaux (EQL) ou "
            "de sommets égaux (EQH) est présente."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("eq_kind", EQ_KIND_VALUES, "any")],
    },
    # ── CONTEXT ──────────────────────────────────────────────────────────────
    {
        "type": "market_phase_is",
        "family": "context",
        "label": "La phase de marché est",
        "description": (
            "La phase de marché observée correspond à celle choisie — accumulation, "
            "tendance, range ou expansion."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("phase", PHASE_VALUES, "expansion")],
    },
    {
        "type": "volatility_is",
        "family": "context",
        "label": "La volatilité est",
        "description": (
            "La volatilité observée en ce moment correspond au niveau choisi — "
            "contractée (faible), normale, ou étendue (élevée)."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("volatility", VOLATILITY_VALUES, "elevated")],
    },
    {
        "type": "price_in_range_third",
        "family": "context",
        "label": "Le prix est dans le tiers … du range",
        "description": (
            "Le prix courant se situe dans le tiers haut, milieu ou bas du range "
            "STRUCTUREL de la fenêtre d'analyse (mêmes bornes que la tuile "
            "« Position dans le range » du panneau Régime)."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("third", RANGE_THIRD_VALUES, "bottom")],
    },
    {
        "type": "session_is",
        "family": "context",
        "label": "La session en cours est",
        "description": (
            "Au moment de la dernière bougie clôturée, la session de marché est "
            "celle choisie — Asie, Londres, New York, ou le chevauchement "
            "Londres/New York."
        ),
        "supports_direction": False,
        "tense": "present",
        "controls": [_control("session", SESSION_VALUES, "london")],
    },
]

#: Conditions PREPARED in the model but NOT exposed — their underlying data is
#: known to be missing or unreliable. They appear in the interface only once the
#: blocking work lands. The request Literal deliberately excludes them, so asking
#: for one is a 422 (out of palette) — never a silent wrong answer.
BLOCKED_PALETTE: List[Dict[str, Any]] = [
    {
        "type": "zone_tested_at_most",
        "family": "zones",
        "label": "Une zone a été testée au plus N fois",
        "tense": "present",
        "blocked_reason": (
            "Le moteur n'expose qu'un booléen ``tested`` par Order Block / Fair "
            "Value Gap, pas un compteur de touches. « Au plus N fois » exige de "
            "compter les ré-entrées par zone — chantier moteur non fait."
        ),
    },
]

#: Closed allowlist of condition types. Anything outside is rejected by the
#: endpoint — there is intentionally no path to a predictive condition, and no
#: path to a BLOCKED condition either.
ALLOWED_CONDITION_TYPES = frozenset(p["type"] for p in PALETTE)

_PALETTE_BY_TYPE = {p["type"]: p for p in PALETTE}

#: Minutes per timeframe — used only to express recency in bars. Derived from the
#: single timeframe registry (TF-1), never a hand-copied subset.
from src.intelligence import timeframe_registry as _tfreg

_TF_MINUTES = _tfreg.minutes_map()


# ---------------------------------------------------------------------------
# Small read-only helpers
# ---------------------------------------------------------------------------


def _label_for(cond_type: str) -> str:
    entry = _PALETTE_BY_TYPE.get(cond_type)
    return entry["label"] if entry else cond_type


def _direction_word(direction: str) -> str:
    return {"bullish": "haussier", "bearish": "baissier"}.get(direction, "")


def _trend_axis(value: Optional[str]) -> str:
    """Collapse a trend value to an axis: 'up' / 'down' / 'flat' (indeterminate)."""
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
    missing" (``available=False`` — a NON-EVALUABLE third state) from "the
    condition was judged and is not met" (``available=True, met=False``). A
    non-evaluable condition is never met AND never counted as unmet: the visible
    denominator adjusts. ``detail`` always carries the OBSERVED VALUE, so the UI
    never shows a bare check.
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


# ── structure: higher_tf_agrees (C1 — replaces the fixed 3-TF alignment) ──────

_TREND_ADJ = {"bullish": "haussier", "bearish": "baissier", "indeterminate": "indéterminé"}

#: Friendly French unit labels, so the result NAMES the compared unit
#: (« le 1 h va dans le même sens ») rather than a raw id (C1-b).
_TF_LABEL_FR = {
    "M1": "1 min", "M5": "5 min", "M15": "15 min", "H1": "1 h",
    "H4": "4 h", "D1": "1 jour", "W1": "1 semaine",
}


def _tf_label(tf: str) -> str:
    return _TF_LABEL_FR.get((tf or "").upper(), tf)


def _relation_of(current: Optional[str], higher: Optional[str]) -> str:
    """Relation of the higher-unit structural trend to the current one."""
    if current in (None, "indeterminate") or higher in (None, "indeterminate"):
        return "indeterminate"
    return "same" if current == higher else "opposite"


def _eval_higher_tf_agrees(
    reading: Dict[str, Any], relation: Optional[str],
    instrument_trends: Optional[Dict[str, Optional[str]]],
) -> Dict[str, Any]:
    """Does the IMMEDIATELY-higher timeframe's STRUCTURAL trend go the chosen way?

    TR-1 is live on main: each reading's ``regime.trend`` is already the
    structural trend (last uncontested BOS/CHOCH), so this compares STRUCTURE —
    never a mere close displacement. The higher unit is the NEAREST perimeter
    unit above the scanned one (registry ``alignment_timeframes``), and it is
    NAMED in the detail (C1-b). On the highest tracked unit there is no unit
    above → NON-EVALUABLE, never met by default nor counted as a failure (C1-c).
    ``relation`` is the required relation: same / opposite / indeterminate.
    """
    if relation not in RELATION_VALUES:
        return _result("higher_tf_agrees", False, "Relation cible non précisée.")
    trends = instrument_trends or {}
    current_tf = reading.get("header", {}).get("timeframe") or ""
    if not _tfreg.has(current_tf):
        return _result("higher_tf_agrees", False, "Unité de temps inconnue.", available=False)
    # Nearest higher unit we actually hold a reading for (named in the result).
    higher_tf = next(
        (tf for tf in _tfreg.alignment_timeframes(current_tf) if trends.get(tf)), None
    )
    if higher_tf is None:
        return _result(
            "higher_tf_agrees", False,
            "Aucune unité supérieure suivie au-dessus de cette unité — non évaluable.",
            available=False,
        )
    current_trend = reading.get("regime", {}).get("trend")
    if not current_trend:
        return _result(
            "higher_tf_agrees", False, "Tendance de cette unité indisponible.", available=False
        )
    higher_trend = trends.get(higher_tf)
    observed = _relation_of(current_trend, higher_trend)
    phrases = {
        "same": f"Le {_tf_label(higher_tf)} va dans le même sens",
        "opposite": f"Le {_tf_label(higher_tf)} va en sens opposé",
        "indeterminate": f"Le {_tf_label(higher_tf)} est indéterminé",
    }
    want = {"same": "même sens", "opposite": "sens opposé", "indeterminate": "indéterminé"}[relation]
    return _result(
        "higher_tf_agrees", observed == relation,
        f"{phrases[observed]} ({_TREND_ADJ.get(higher_trend, higher_trend)}) — cible : {want}.",
    )


# ── structure: trend_is / last_event_is / last_event_age ─────────────────────


def _eval_trend_is(reading: Dict[str, Any], trend: Optional[str]) -> Dict[str, Any]:
    if not trend:
        return _result("trend_is", False, "Tendance cible non précisée.")
    observed = reading.get("regime", {}).get("trend")
    if not observed:
        return _result("trend_is", False, "Tendance observée indisponible.", available=False)
    adj = _TREND_ADJ.get(observed, observed)
    tgt = _TREND_ADJ.get(trend, trend)
    return _result("trend_is", observed == trend, f"Tendance structurelle observée : {adj} (cible : {tgt}).")


def _all_events(reading: Dict[str, Any]) -> List[Dict[str, Any]]:
    """BOS + CHOCH journal entries tagged with their kind, most-recent first.

    Uses the discrete event journals (``bos_events`` / ``choch_events``), each
    carrying ``direction`` and ``bars_ago``. Entries without a usable
    ``bars_ago`` are dropped (their recency is unknown).
    """
    struct = reading.get("structure", {})
    out: List[Dict[str, Any]] = []
    for kind, key in (("bos", "bos_events"), ("choch", "choch_events")):
        for ev in struct.get(key, []) or []:
            bars_ago = ev.get("bars_ago")
            if bars_ago is None:
                continue
            out.append({"kind": kind, "direction": ev.get("direction"), "bars_ago": int(bars_ago)})
    out.sort(key=lambda e: e["bars_ago"])
    return out


def _event_label(kind: str, direction: Optional[str]) -> str:
    arrow = "↑" if direction == "bullish" else "↓" if direction == "bearish" else "?"
    return f"{kind.upper()} {arrow}"


def _eval_last_event_is(reading: Dict[str, Any], event: Optional[str]) -> Dict[str, Any]:
    if event not in EVENT_VALUES:
        return _result("last_event_is", False, "Événement cible non précisé.")
    events = _all_events(reading)
    if not events:
        return _result("last_event_is", False, "Aucun événement de structure daté.", available=False)
    last = events[0]
    want_kind, want_dir = event.split("_")
    want_dir = "bullish" if want_dir == "up" else "bearish"
    observed = _event_label(last["kind"], last["direction"])
    met = last["kind"] == want_kind and last["direction"] == want_dir
    return _result(
        "last_event_is", met,
        f"Dernier événement : {observed} il y a {last['bars_ago']} bougie(s) "
        f"(cible : {_event_label(want_kind, want_dir)}).",
    )


def _eval_last_event_age(reading: Dict[str, Any], age_bucket: Optional[str]) -> Dict[str, Any]:
    if age_bucket not in AGE_BUCKET_VALUES:
        return _result("last_event_age", False, "Tranche d'ancienneté non précisée.")
    events = _all_events(reading)
    if not events:
        return _result("last_event_age", False, "Aucun événement de structure daté.", available=False)
    bars = events[0]["bars_ago"]
    observed = _event_label(events[0]["kind"], events[0]["direction"])
    if bars < 10:
        bucket = "lt10"
    elif bars <= 50:
        bucket = "10to50"
    else:
        bucket = "gt50"
    labels = {"lt10": "moins de 10", "10to50": "10 à 50", "gt50": "plus de 50"}
    return _result(
        "last_event_age", bucket == age_bucket,
        f"Dernier événement ({observed}) il y a {bars} bougie(s) — tranche « {labels[bucket]} » "
        f"(cible : « {labels[age_bucket]} »).",
    )


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


# ── zones: price in / near / tested / untested / formed-recent ───────────────


def _zones_of_kind(reading: Dict[str, Any], zone_kind: str) -> List[Tuple[str, Dict[str, Any]]]:
    """Active zones (OB and/or FVG) with their family tag, honouring ``zone_kind``."""
    struct = reading.get("structure", {})
    out: List[Tuple[str, Dict[str, Any]]] = []
    if zone_kind in ("any", "ob"):
        for ob in struct.get("order_blocks", []) or []:
            if ob.get("status") == "active":
                out.append(("OB", ob))
    if zone_kind in ("any", "fvg"):
        for fvg in struct.get("fair_value_gaps", []) or []:
            if fvg.get("status") in ("active", "partially_filled"):
                out.append(("FVG", fvg))
    return out


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


def _fmt_band(zone: Dict[str, Any]) -> str:
    b = _zone_bounds(zone)
    return f"{b[0]:g} – {b[1]:g}" if b else "n/d"


def _eval_price_in_ob(reading: Dict[str, Any], direction: str) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_ob", False, "Prix courant indisponible.", available=False)
    hits = _active_obs_at_price(reading, price, direction)
    if hits:
        return _result("price_in_ob", True, f"Prix dans un Order Block actif ({_fmt_band(hits[0])}).")
    suffix = f" {_direction_word(direction)}" if direction != "any" else ""
    return _result("price_in_ob", False, f"Prix hors de tout Order Block actif{suffix}.")


def _eval_price_in_fvg(reading: Dict[str, Any], direction: str) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_fvg", False, "Prix courant indisponible.", available=False)
    hits = _active_fvgs_at_price(reading, price, direction)
    if hits:
        return _result("price_in_fvg", True, f"Prix dans un Fair Value Gap non comblé ({_fmt_band(hits[0])}).")
    suffix = f" {_direction_word(direction)}" if direction != "any" else ""
    return _result("price_in_fvg", False, f"Prix hors de tout Fair Value Gap non comblé{suffix}.")


def _eval_price_in_tested_zone(reading: Dict[str, Any], zone_kind: str) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_tested_zone", False, "Prix courant indisponible.", available=False)
    for tag, z in _zones_of_kind(reading, zone_kind):
        bounds = _zone_bounds(z)
        if bounds is None:
            continue
        lo, hi = bounds
        if lo < price < hi and z.get("tested"):
            return _result(
                "price_in_tested_zone", True,
                f"Prix dans un {tag} déjà testé ({_fmt_band(z)}).",
            )
    return _result("price_in_tested_zone", False, "Prix dans aucune zone déjà testée.")


def _eval_zone_untested(reading: Dict[str, Any], zone_kind: str) -> Dict[str, Any]:
    for tag, z in _zones_of_kind(reading, zone_kind):
        if not z.get("tested"):
            return _result("zone_untested", True, f"{tag} jamais testé depuis sa formation ({_fmt_band(z)}).")
    return _result("zone_untested", False, "Toutes les zones actives ont déjà été testées.")


def _eval_zone_formed_recent(reading: Dict[str, Any], zone_kind: str, max_bars: int) -> Dict[str, Any]:
    tf = reading.get("header", {}).get("timeframe", "")
    candle_ts = reading.get("header", {}).get("candle_close_ts")
    best: Optional[Tuple[float, str, Dict[str, Any]]] = None
    saw_dateable = False
    for tag, z in _zones_of_kind(reading, zone_kind):
        bars = _bars_between(z.get("created_at"), candle_ts, tf)
        if bars is None:
            continue
        saw_dateable = True
        if best is None or bars < best[0]:
            best = (bars, tag, z)
    if best is None:
        if not saw_dateable and _zones_of_kind(reading, zone_kind):
            return _result("zone_formed_recent", False, "Âge de formation des zones inconnu.", available=False)
        return _result("zone_formed_recent", False, "Aucune zone active à dater.")
    bars, tag, z = best
    if bars <= max_bars:
        return _result(
            "zone_formed_recent", True,
            f"{tag} formé il y a ~{round(bars)} bougie(s) (≤ {max_bars}) ({_fmt_band(z)}).",
        )
    return _result(
        "zone_formed_recent", False,
        f"Zone la plus récente formée il y a ~{round(bars)} bougies (> {max_bars}).",
    )


def _distance_pct(price: float, lo: float, hi: float) -> float:
    """% distance from ``price`` to the ``[lo, hi]`` band (0.0 when inside)."""
    if not price:
        return float("inf")
    if lo <= price <= hi:
        return 0.0
    edge = lo if price < lo else hi
    return abs(price - edge) / price * 100.0


def _eval_price_near_zone(
    reading: Dict[str, Any], cond_type: str, struct_key: str, name: str,
    active_statuses: tuple, direction: str, proximity_pct: float,
) -> Dict[str, Any]:
    """« Price near an OB / FVG, WITHOUT being inside » (0 < distance ≤ threshold)."""
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result(cond_type, False, "Prix courant indisponible.", available=False)
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
    if best is None:
        suffix = f" {_direction_word(direction)}" if direction != "any" else ""
        return _result(cond_type, False, f"Aucun {name} actif{suffix}.")
    if best == 0.0:
        # Inside the zone → this "near, without being inside" condition is NOT met.
        return _result(cond_type, False, f"Prix à l'intérieur d'un {name} (donc pas « proche sans y être »).")
    if best <= proximity_pct:
        return _result(
            cond_type, True,
            f"Prix à ~{best:.2f} % d'un {name} actif, sans y être (seuil {proximity_pct:.2f} %).",
        )
    return _result(
        cond_type, False,
        f"{name} actif le plus proche à ~{best:.2f} % (> seuil {proximity_pct:.2f} %).",
    )


# ── liquidity ────────────────────────────────────────────────────────────────


def _eval_price_near_liquidity(reading: Dict[str, Any], side: str, proximity_pct: float) -> Dict[str, Any]:
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_near_liquidity", False, "Prix courant indisponible.", available=False)
    best: Optional[float] = None
    best_side: Optional[str] = None
    for lp in reading.get("structure", {}).get("liquidity_pools", []) or []:
        if lp.get("status") != "intact":
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
            f"Poche {(best_side or '').upper()} intacte à ~{best:.2f} % (seuil {proximity_pct:.2f} %).",
        )
    return _result(
        "price_near_liquidity", False,
        f"Poche intacte la plus proche à ~{best:.2f} % (> seuil {proximity_pct:.2f} %).",
    )


def _eval_liquidity_event_recent(
    reading: Dict[str, Any], cond_type: str, status: str, ts_key: str, verb: str,
    side: str, max_bars: int,
) -> Dict[str, Any]:
    """Shared evaluator for « a pocket was swept / broken within N bars »."""
    tf = reading.get("header", {}).get("timeframe", "")
    candle_ts = reading.get("header", {}).get("candle_close_ts")
    best_bars: Optional[float] = None
    best_side: Optional[str] = None
    for lp in reading.get("structure", {}).get("liquidity_pools", []) or []:
        if lp.get("status") != status:
            continue
        if side != "any" and lp.get("side") != side:
            continue
        bars = _bars_between(lp.get(ts_key), candle_ts, tf)
        if bars is None:
            continue
        if best_bars is None or bars < best_bars:
            best_bars = bars
            best_side = lp.get("side")
    if best_bars is None:
        s = f" {side.upper()}" if side != "any" else ""
        return _result(cond_type, False, f"Aucune poche{s} {verb} récemment.")
    if best_bars <= max_bars:
        return _result(
            cond_type, True,
            f"Poche {(best_side or '').upper()} {verb} il y a ~{round(best_bars)} bougie(s) (≤ {max_bars}).",
        )
    return _result(
        cond_type, False,
        f"Poche {verb} trop ancienne (~{round(best_bars)} bougies > {max_bars}).",
    )


def _eval_equal_levels_present(reading: Dict[str, Any], eq_kind: str) -> Dict[str, Any]:
    wanted = {"any": ("equal_highs", "equal_lows"), "highs": ("equal_highs",), "lows": ("equal_lows",)}[eq_kind]
    hits = [
        lp for lp in reading.get("structure", {}).get("liquidity_pools", []) or []
        if lp.get("status") == "intact" and lp.get("kind") in wanted
    ]
    if hits:
        kinds = ", ".join(sorted({("sommets égaux" if h.get("kind") == "equal_highs" else "creux égaux") for h in hits}))
        return _result("equal_levels_present", True, f"{len(hits)} poche(s) présente(s) : {kinds}.")
    label = {"any": "creux ou sommets", "highs": "sommets", "lows": "creux"}[eq_kind]
    return _result("equal_levels_present", False, f"Aucun(e) {label} égaux intacts.")


# ── context: phase / volatility / range-third / session ──────────────────────


def _eval_market_phase_is(reading: Dict[str, Any], phase: Optional[str]) -> Dict[str, Any]:
    if not phase:
        return _result("market_phase_is", False, "Phase cible non précisée.")
    observed = reading.get("regime", {}).get("market_phase")
    if not observed:
        return _result("market_phase_is", False, "Phase observée indisponible.", available=False)
    return _result("market_phase_is", observed == phase, f"Phase observée : {observed} (cible : {phase}).")


def _eval_volatility_is(reading: Dict[str, Any], volatility: Optional[str]) -> Dict[str, Any]:
    if not volatility:
        return _result("volatility_is", False, "Niveau de volatilité cible non précisé.")
    observed = reading.get("regime", {}).get("volatility_observed")
    if not observed:
        return _result("volatility_is", False, "Volatilité observée indisponible.", available=False)
    words = {"low": "contractée", "normal": "normale", "elevated": "étendue"}
    return _result(
        "volatility_is", observed == volatility,
        f"Volatilité observée : {words.get(observed, observed)} (cible : {words.get(volatility, volatility)}).",
    )


def _structural_range(reading: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """(low, high) of the STRUCTURAL range — same definition as the Régime
    « Position dans le range » tile (webapp structureRange): the pools' explicit
    ``range_high``/``range_low``, falling back to the highest external BSL / lowest
    external SSL pocket. Returns None when the range is missing or degenerate.
    """
    pools = reading.get("structure", {}).get("liquidity_pools", []) or []
    high: Optional[float] = None
    low: Optional[float] = None
    for p in pools:
        if p.get("kind") == "range_high" and p.get("level") is not None:
            high = float(p["level"])
        elif p.get("kind") == "range_low" and p.get("level") is not None:
            low = float(p["level"])
    if high is None:
        for p in pools:
            if p.get("is_external") and p.get("side") == "bsl" and p.get("level") is not None:
                lvl = float(p["level"])
                if high is None or lvl > high:
                    high = lvl
    if low is None:
        for p in pools:
            if p.get("is_external") and p.get("side") == "ssl" and p.get("level") is not None:
                lvl = float(p["level"])
                if low is None or lvl < low:
                    low = lvl
    if high is None or low is None or high <= low:
        return None
    return (low, high)


def _eval_price_in_range_third(reading: Dict[str, Any], third: Optional[str]) -> Dict[str, Any]:
    if third not in RANGE_THIRD_VALUES:
        return _result("price_in_range_third", False, "Tiers cible non précisé.")
    price = reading.get("header", {}).get("close_price")
    if price is None:
        return _result("price_in_range_third", False, "Prix courant indisponible.", available=False)
    rng = _structural_range(reading)
    if rng is None:
        return _result(
            "price_in_range_third", False,
            "Range structurel indisponible (structure insuffisante).", available=False,
        )
    low, high = rng
    pos = max(0.0, min(1.0, (price - low) / (high - low)))
    observed = "bottom" if pos < 1 / 3 else "middle" if pos < 2 / 3 else "top"
    words = {"bottom": "bas", "middle": "milieu", "top": "haut"}
    window = reading.get("header", {}).get("analysis_window_bars")
    window_txt = f" sur {window} bougies" if window else ""
    return _result(
        "price_in_range_third", observed == third,
        f"Prix à {pos * 100:.0f} % du range structurel [{low:g} – {high:g}]{window_txt} "
        f"— tiers {words[observed]} (cible : tiers {words[third]}).",
    )


def _eval_session_is(reading: Dict[str, Any], session: Optional[str]) -> Dict[str, Any]:
    if session not in SESSION_VALUES:
        return _result("session_is", False, "Session cible non précisée.")
    from src.intelligence import market_calendar as _mc

    instrument = reading.get("header", {}).get("instrument") or ""
    candle_ts = _parse_dt(reading.get("header", {}).get("candle_close_ts"))
    if candle_ts is None:
        return _result("session_is", False, "Horodatage de bougie indisponible.", available=False)
    windows = _mc.sessions_for(instrument)
    if not windows:
        return _result("session_is", False, "Marché continu — pas de découpage en sessions.", available=False)
    hours = _mc.hours_for(instrument)
    if candle_ts.tzinfo is None:
        candle_ts = candle_ts.replace(tzinfo=timezone.utc)
    local = candle_ts.astimezone(ZoneInfo(hours.tz)).time()

    def _in_window(w) -> bool:
        if w.start <= w.end:
            return w.start <= local < w.end
        return local >= w.start or local < w.end  # wraps past midnight (Asia)

    active = [w.name for w in windows if _in_window(w)]
    # Overlap = simultaneously inside ≥2 sessions (London∩NY, derived — never stored).
    observed = "overlap" if len(active) >= 2 else (active[0] if active else "none")
    names = {"asia": "Asie", "london": "Londres", "new_york": "New York", "overlap": "chevauchement", "none": "hors session"}
    met = observed == session
    return _result(
        "session_is", met,
        f"Session à la clôture : {names.get(observed, observed)} (cible : {names.get(session, session)}).",
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

    Returns ``{"type", "label", "met": bool, "available": bool, "detail": str}``.
    ``detail`` always carries the observed value. ``instrument_trends`` (per-TF
    structural trend for THIS instrument) is used only by ``higher_tf_agrees``.
    Unknown types raise ``ValueError`` (callers validate against the palette).
    """
    cond_type = cond.get("type")
    direction = cond.get("direction", "any") or "any"

    if cond_type == "trend_is":
        return _eval_trend_is(reading, cond.get("trend"))
    if cond_type == "higher_tf_agrees":
        return _eval_higher_tf_agrees(reading, cond.get("relation"), instrument_trends)
    if cond_type == "last_event_is":
        return _eval_last_event_is(reading, cond.get("event"))
    if cond_type == "last_event_age":
        return _eval_last_event_age(reading, cond.get("age_bucket"))
    if cond_type == "bos_recent_confirmed":
        return _eval_break_recent_confirmed(
            reading, "bos_recent_confirmed", "bos", "BOS", direction,
            int(cond.get("max_bars") or DEFAULT_BOS_MAX_BARS),
        )
    if cond_type == "choch_recent_confirmed":
        return _eval_break_recent_confirmed(
            reading, "choch_recent_confirmed", "choch", "CHOCH", direction,
            int(cond.get("max_bars") or DEFAULT_BOS_MAX_BARS),
        )
    if cond_type == "price_in_ob":
        return _eval_price_in_ob(reading, direction)
    if cond_type == "price_in_fvg":
        return _eval_price_in_fvg(reading, direction)
    if cond_type == "price_in_tested_zone":
        return _eval_price_in_tested_zone(reading, cond.get("zone_kind", "any") or "any")
    if cond_type == "zone_untested":
        return _eval_zone_untested(reading, cond.get("zone_kind", "any") or "any")
    if cond_type == "zone_formed_recent":
        return _eval_zone_formed_recent(
            reading, cond.get("zone_kind", "any") or "any",
            int(cond.get("max_bars") or DEFAULT_FORMED_MAX_BARS),
        )
    if cond_type == "price_near_ob":
        return _eval_price_near_zone(
            reading, "price_near_ob", "order_blocks", "Order Block",
            ("active",), direction, float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT),
        )
    if cond_type == "price_near_fvg":
        return _eval_price_near_zone(
            reading, "price_near_fvg", "fair_value_gaps", "Fair Value Gap",
            ("active", "partially_filled"), direction, float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT),
        )
    if cond_type == "price_near_liquidity":
        return _eval_price_near_liquidity(
            reading, cond.get("side", "any") or "any", float(cond.get("proximity_pct") or DEFAULT_PROXIMITY_PCT),
        )
    if cond_type == "liquidity_swept_recent":
        return _eval_liquidity_event_recent(
            reading, "liquidity_swept_recent", "swept", "swept_at", "balayée",
            cond.get("side", "any") or "any", int(cond.get("max_bars") or DEFAULT_LIQ_MAX_BARS),
        )
    if cond_type == "liquidity_broken_recent":
        return _eval_liquidity_event_recent(
            reading, "liquidity_broken_recent", "broken", "broken_at", "cassée",
            cond.get("side", "any") or "any", int(cond.get("max_bars") or DEFAULT_LIQ_MAX_BARS),
        )
    if cond_type == "equal_levels_present":
        return _eval_equal_levels_present(reading, cond.get("eq_kind", "any") or "any")
    if cond_type == "market_phase_is":
        return _eval_market_phase_is(reading, cond.get("phase"))
    if cond_type == "volatility_is":
        return _eval_volatility_is(reading, cond.get("volatility"))
    if cond_type == "price_in_range_third":
        return _eval_price_in_range_third(reading, cond.get("third"))
    if cond_type == "session_is":
        return _eval_session_is(reading, cond.get("session"))
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
    rng = _structural_range(reading)
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
        "structural_range": {"low": rng[0], "high": rng[1]} if rng else None,
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

    ``logic`` is ``"AND"`` (all met) or ``"OR"`` (any met). A NON-EVALUABLE
    condition (``available=False``) is excluded from the AND/OR denominator: it is
    neither met nor unmet. Under AND a combo matches only if there is at least one
    evaluable condition and every evaluable one is met; under OR if any evaluable
    condition is met. Pure: never mutates ``reading`` or any external state.
    """
    results = [evaluate_condition(reading, c, instrument_trends) for c in conditions]
    evaluable = [r for r in results if r["available"]]
    met = [r for r in results if r["met"]]
    unmet = [r for r in results if r["available"] and not r["met"]]
    non_evaluable = [r for r in results if not r["available"]]

    if logic == "OR":
        matched = any(r["met"] for r in evaluable)
    else:  # default / "AND" — every EVALUABLE condition met, at least one evaluable
        matched = bool(evaluable) and all(r["met"] for r in evaluable)

    header = reading.get("header", {})
    context = build_context(reading)
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
        # Denominator counts only EVALUABLE conditions; non-evaluable ones are
        # surfaced separately so the UI can show the adjusted denominator honestly.
        "total": len(evaluable),
        "non_evaluable_count": len(non_evaluable),
        "conditions_met": met,
        "conditions_unmet": unmet,
        "conditions_non_evaluable": non_evaluable,
        "context": context,
    }


__all__ = [
    "ALLOWED_CONDITION_TYPES",
    "BLOCKED_PALETTE",
    "DEFAULT_BOS_MAX_BARS",
    "DEFAULT_FORMED_MAX_BARS",
    "DEFAULT_LIQ_MAX_BARS",
    "DEFAULT_PROXIMITY_PCT",
    "DIRECTION_VALUES",
    "FAMILIES",
    "LIQUIDITY_SIDE_VALUES",
    "PALETTE",
    "build_context",
    "evaluate_condition",
    "evaluate_reading",
]
