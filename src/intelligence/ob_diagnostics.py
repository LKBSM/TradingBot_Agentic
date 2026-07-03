"""OB rejection diagnostics — on-demand evaluator (mission 2026-07-02).

Answers « pourquoi cette bougie n'est pas un Order Block ? » by READING the
engine's own decision artifacts — never re-implementing them:

* candidate criteria — the named boolean Series from
  ``ob_candidate_conditions`` combined by ``combine_ob_conditions``
  (``src/environment/strategy_features.py``): the exact Series
  ``_add_smc_order_blocks`` ANDs to accept/reject a candidate;
* lifecycle & surfacing — ``collect_zones(with_rejects=True)``
  (``src/intelligence/market_reading_mappers.py``), whose reject records are
  emitted by the very branches that drop a zone (invalidation, policy, cap).

SINGLE SOURCE OF TRUTH: change a detection threshold and both the decision AND
the reason reported here move together (tested in
``tests/test_ob_rejection_diagnostics.py``). This module contains NO detection
logic of its own — only bar resolution (price/timestamp → concrete bar) and
presentation of facts the engine already produced.

Product line: purely descriptive/educational — facts about a PAST evaluation.
No prediction, no recommendation, no signal. French labels below are
presentation only and are pre-screened against the forbidden-token list.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from src.environment.strategy_features import (
    OB_CRITERIA,
    OB_FVG_CRITERION,
    combine_ob_conditions,
    ob_candidate_conditions,
)
from src.intelligence.market_reading_mappers import collect_zones, ob_zone_id

# Presentation-only French labels for the ENGINE's real criteria. Keys mirror
# OB_CRITERIA / OB_FVG_CRITERION exactly — adding a criterion there without a
# label here fails the diagnostics test suite, so the two cannot drift apart.
CRITERIA_LABELS_FR: dict[str, str] = {
    "prev_candle_bearish": "la bougie visée est baissière (clôture sous l'ouverture)",
    "confirm_candle_bullish": "la bougie suivante est haussière (clôture au-dessus de l'ouverture)",
    "breaks_prev_high": "la bougie suivante dépasse le plus haut de la bougie visée",
    "prev_candle_bullish": "la bougie visée est haussière (clôture au-dessus de l'ouverture)",
    "confirm_candle_bearish": "la bougie suivante est baissière (clôture sous l'ouverture)",
    "breaks_prev_low": "la bougie suivante enfonce le plus bas de la bougie visée",
    OB_FVG_CRITERION: "un déséquilibre (FVG) adjacent est présent",
}

REJECT_LABELS_FR: dict[str, str] = {
    "invalidated_close_through": (
        "un Order Block s'était bien formé ici, mais une bougie ultérieure a "
        "clôturé à travers la zone : le moteur l'a invalidé"
    ),
    "mitigated_dropped_by_policy": (
        "un Order Block s'était formé ici et a été retesté ; la politique "
        "d'affichage retire les zones retestées"
    ),
    "capped_max_zones": (
        "le moteur détecte bien un Order Block ici, toujours valide, mais "
        "l'affichage est limité aux zones les plus significatives et celle-ci "
        "est au-delà de cette limite"
    ),
}

DISCLAIMER_FR = (
    "Évaluation factuelle de la détection passée du moteur — descriptif et "
    "éducatif, aucune prédiction."
)


def _iso(ts: Any) -> Optional[str]:
    if ts is None:
        return None
    if hasattr(ts, "to_pydatetime"):
        ts = ts.to_pydatetime()
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts.isoformat()
    return str(ts)


def _candle_facts(enriched: Any, i: int) -> dict:
    row = enriched.iloc[i]
    return {
        "ts": _iso(enriched.index[i]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
    }


def resolve_bar(
    enriched: Any,
    ts: Optional[Any] = None,
    price: Optional[float] = None,
) -> tuple[Optional[int], dict]:
    """Resolve a user reference (timestamp or approximate price) to a concrete
    bar of the enriched frame. Deterministic code — the LLM never guesses.

    Returns ``(bar_index | None, resolution_info)``. Timestamp takes precedence
    when both are given. Price matches bars whose [low, high] range contains the
    price; the MOST RECENT match wins (ties reported via ``matched_bars``).
    """
    import pandas as pd

    n = len(enriched)
    if n == 0:
        return None, {"resolved": False, "reason": "no_data"}

    if ts is not None:
        try:
            target = pd.Timestamp(ts)
            if target.tzinfo is None:
                target = target.tz_localize("UTC")
        except (ValueError, TypeError):
            return None, {"resolved": False, "reason": "invalid_timestamp", "input_ts": str(ts)}
        if not isinstance(enriched.index, pd.DatetimeIndex):
            return None, {"resolved": False, "reason": "no_time_index"}
        index = enriched.index
        if index.tz is None:
            index = index.tz_localize("UTC")
        deltas = abs(index - target)
        i = int(deltas.argmin())
        # Accept only within one nominal bar step — beyond that the candle the
        # user means is outside the analysed window; say so, don't guess.
        steps = index.to_series().diff().dropna()
        nominal = steps.median() if len(steps) else pd.Timedelta(0)
        if nominal and deltas[i] > nominal:
            return None, {
                "resolved": False,
                "reason": "out_of_window",
                "input_ts": str(ts),
                "window_start": _iso(enriched.index[0]),
                "window_end": _iso(enriched.index[-1]),
            }
        return i, {
            "resolved": True,
            "matched_by": "timestamp",
            "input_ts": str(ts),
            "exact": bool(deltas[i] == pd.Timedelta(0)),
        }

    if price is not None:
        p = float(price)
        lows = enriched["low"].values
        highs = enriched["high"].values
        matches = [k for k in range(n) if lows[k] <= p <= highs[k]]
        if not matches:
            return None, {
                "resolved": False,
                "reason": "price_not_touched",
                "input_price": p,
                "window_low": float(lows.min()),
                "window_high": float(highs.max()),
            }
        return matches[-1], {
            "resolved": True,
            "matched_by": "price",
            "input_price": p,
            "matched_bars": len(matches),
            "note": "bougie la plus récente dont l'amplitude contient ce prix",
        }

    return None, {"resolved": False, "reason": "no_reference"}


def _side_checks(enriched: Any, conditions: dict, side: str, d: int) -> list[dict]:
    """Per-criterion verdicts for ``side`` at confirmation bar ``d`` — read
    straight from the engine's own boolean Series (no re-evaluation)."""
    opens = enriched["open"].values
    closes = enriched["close"].values
    highs = enriched["high"].values
    lows = enriched["low"].values
    i = d - 1  # the user's candle (the zone candle)

    observed: dict[str, dict] = {
        "prev_candle_bearish": {"open": float(opens[i]), "close": float(closes[i])},
        "prev_candle_bullish": {"open": float(opens[i]), "close": float(closes[i])},
        "confirm_candle_bullish": {"next_open": float(opens[d]), "next_close": float(closes[d])},
        "confirm_candle_bearish": {"next_open": float(opens[d]), "next_close": float(closes[d])},
        "breaks_prev_high": {"next_high": float(highs[d]), "candle_high": float(highs[i])},
        "breaks_prev_low": {"next_low": float(lows[d]), "candle_low": float(lows[i])},
    }

    checks = [
        {
            "criterion": name,
            "passed": bool(conditions[side][name].iloc[d]),
            "observed": observed[name],
            "label_fr": CRITERIA_LABELS_FR[name],
        }
        for name in OB_CRITERIA[side]
    ]
    if conditions["fvg_required"]:
        checks.append({
            "criterion": OB_FVG_CRITERION,
            "passed": bool(conditions[OB_FVG_CRITERION].iloc[d]),
            "observed": {"fvg_required_by_config": True},
            "label_fr": CRITERIA_LABELS_FR[OB_FVG_CRITERION],
        })
    return checks


def _find_zone(zones: dict, d: int, side: str) -> tuple[Optional[dict], bool]:
    """Locate the zone whose detection bar is ``d`` in the collector output.
    Returns ``(zone, surfaced)``."""
    for z in zones.get("order_blocks", []):
        if z["_k"] == d and z["direction"] == side:
            return z, True
    for z in zones.get("rejected_order_blocks", []):
        if z["_k"] == d and z["direction"] == side:
            return z, False
    return None, False


def diagnose_ob(
    enriched: Any,
    config: Any,
    ts: Optional[Any] = None,
    price: Optional[float] = None,
    max_per_type: Optional[int] = None,
) -> dict:
    """Full OB diagnostic for one candle of an already-enriched frame.

    ``enriched`` must be a SmartMoneyEngine.analyze() output (the SAME frame the
    reading pipeline builds); ``config`` the engine's SMCConfig. The verdict and
    every reported reason are byproducts of the engine's real decision path.
    """
    pos = len(enriched) - 1
    bar, resolution = resolve_bar(enriched, ts=ts, price=price)
    base = {"resolution": resolution, "disclaimer_fr": DISCLAIMER_FR}
    if bar is None:
        return {**base, "status": "unresolved"}

    candle = _candle_facts(enriched, bar)
    conditions = ob_candidate_conditions(enriched, config)

    # Secondary role: is the referenced candle itself the CONFIRMATION bar of an
    # OB whose zone is the previous candle? (common user confusion)
    confirmation_of_previous = None
    for side in OB_CRITERIA:
        if bar >= 1 and bool(combine_ob_conditions(conditions, side).iloc[bar]):
            confirmation_of_previous = {
                "side": side,
                "zone_candle_ts": _iso(enriched.index[bar - 1]),
                "note_fr": (
                    "cette bougie est la bougie de confirmation d'un Order Block "
                    "dont la zone est la bougie précédente"
                ),
            }
            break

    d = bar + 1  # detection is evaluated on the FOLLOWING candle
    if d > pos:
        return {
            **base,
            "status": "awaiting_next_candle",
            "candle": candle,
            "confirmation_of_previous": confirmation_of_previous,
            "note_fr": (
                "cette bougie est la dernière de la fenêtre : le moteur évalue un "
                "Order Block sur la bougie SUIVANTE, qui n'existe pas encore ici"
            ),
        }

    sides: dict[str, Any] = {}
    detected_side: Optional[str] = None
    for side in OB_CRITERIA:
        decided = bool(combine_ob_conditions(conditions, side).iloc[d])
        sides[side] = {
            "detected": decided,
            "checks": _side_checks(enriched, conditions, side, d),
        }
        if decided:
            detected_side = side

    result = {
        **base,
        "candle": candle,
        "confirmation_bar_ts": _iso(enriched.index[d]),
        "sides": sides,
        "confirmation_of_previous": confirmation_of_previous,
    }

    if detected_side is None:
        return {**result, "status": "not_candidate"}

    zones = collect_zones(enriched, idx=pos, max_per_type=max_per_type, with_rejects=True)
    zone, surfaced = _find_zone(zones, d, detected_side)
    if zone is None:  # defensive — detection and collection share the same columns
        return {**result, "status": "detected_untracked", "side": detected_side}

    zone_info = {
        "id": ob_zone_id(zone["direction"], zone["created_at"]) if zone.get("created_at") else None,
        "direction": zone["direction"],
        "level_high": zone["level_high"],
        "level_low": zone["level_low"],
        "importance": zone["importance"],
        "lifecycle_status": zone["status"],
        "tested": zone["tested"],
        "created_at": _iso(zone.get("created_at")),
        "mitigated_at": _iso(zone.get("mitigated_at")),
    }
    if surfaced:
        return {**result, "status": "is_order_block", "side": detected_side, "zone": zone_info}

    reason = zone["reject_reason"]
    rejected = {
        **zone_info,
        "reject_reason": reason,
        "reject_label_fr": REJECT_LABELS_FR[reason],
        "invalidated_at": _iso(zone.get("invalidated_at")),
    }
    if reason == "capped_max_zones":
        rejected["cap_rank"] = zone.get("cap_rank")
        rejected["cap_max"] = zone.get("cap_max")
    return {**result, "status": "was_rejected", "side": detected_side, "zone": rejected}


__all__ = [
    "CRITERIA_LABELS_FR",
    "DISCLAIMER_FR",
    "REJECT_LABELS_FR",
    "diagnose_ob",
    "resolve_bar",
]
