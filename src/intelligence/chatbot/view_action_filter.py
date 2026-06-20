"""Couche 4 — view-action whitelist (display-only, doc §4.1 extension).

The chatbot may emit STRUCTURED view actions that change ONLY the chart display
(layer visibility, zone filters, focus / zoom, instrument / timeframe, highlight).
This module is the security gate: it validates each proposed action against a
CLOSED whitelist and rejects everything else — including any attempt to create,
place, move, or resize a detected structure.

Invariants (the inviolable line)
--------------------------------
- The action vocabulary is a FIXED enum. No create / place / move / resize verb
  exists, so a request like « place un OB à 2000 » or « agrandis ce FVG » has no
  representable action and is rejected.
- No action carries a price / level / geometry field. A hard guard additionally
  rejects any action whose params contain a geometry-shaped key
  (``GEOMETRY_KEYS``) — detection geometry is structurally unreachable.
- ``focus_zone`` / ``highlight_zone`` reference an EXISTING detected zone id,
  validated by the caller against ids actually read from the engine this turn.
  An invented id (« centre-toi sur l'OB à 2000 ») is rejected.
- Validation is PURE and READ-ONLY: it never touches detection, never mutates a
  structure. It only decides whether a *display* intent is admissible and returns
  a normalised (clamped) copy.

A rejected action is reported back to the orchestrator (Couche 2) so it can hand
the model the on-brand refusal text — the chatbot never silently drops an action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# Combos covered by the product (mirror Chatbot.SUPPORTED_* / webapp store).
SUPPORTED_INSTRUMENTS: tuple[str, ...] = ("XAUUSD", "EURUSD")
SUPPORTED_TIMEFRAMES: tuple[str, ...] = ("M15", "H1", "H4")

# Layers the chatbot may toggle. "all" is the convenience target (show/hide every
# overlay at once); "breaks" = BOS / CHOCH / retest level lines + markers.
ALLOWED_LAYERS: tuple[str, ...] = ("fvg", "ob", "breaks", "all")

# The CLOSED whitelist of view actions. Everything outside this set is rejected.
ALLOWED_ACTIONS: tuple[str, ...] = (
    "set_layer_visibility",
    "filter_zones",
    "focus_zone",
    "focus_price",
    "fit_chart",
    "set_instrument_timeframe",
    "highlight_zone",
    "reset_view",
)

# Actions that reference a detected zone by id (must exist in the engine output).
_ZONE_REF_ACTIONS: frozenset[str] = frozenset({"focus_zone", "highlight_zone"})

# Geometry-shaped param keys that must NEVER appear on a view action. Their mere
# presence means "set/move a structure's coordinates" → reject outright. This is
# defence-in-depth on top of the per-action param allow-lists below.
GEOMETRY_KEYS: frozenset[str] = frozenset(
    {
        "price",
        "prices",
        "level",
        "level_high",
        "level_low",
        "high",
        "low",
        "top",
        "bottom",
        "band",
        "open",
        "close",
    }
)

# Clamp ranges for the numeric display thresholds (NOT geometry — these size the
# filter, never a zone). Kept generous but bounded so a bad value can't break the
# view.
_PROXIMITY_PCT_RANGE = (0.05, 10.0)
_MIN_SIZE_PCT_RANGE = (0.0, 10.0)
_DEFAULT_PROXIMITY_PCT = 0.5


@dataclass(frozen=True)
class ViewActionCheckResult:
    """Outcome of validating a single proposed view action.

    Attributes:
        valid: True when the action is admissible (display-only, in whitelist).
        action: the normalised / clamped action dict when valid, else None.
        reason: a short machine reason when rejected (logged / audited), else None.
    """

    valid: bool
    action: Optional[dict[str, Any]] = None
    reason: Optional[str] = None


class ViewActionValidator:
    """Stateless whitelist gate for chatbot-proposed chart view actions."""

    def validate(
        self,
        proposed: Any,
        *,
        known_zone_ids: Optional[set[str]] = None,
    ) -> ViewActionCheckResult:
        """Validate one proposed action against the display-only whitelist.

        Args:
            proposed: the raw action emitted by the model — expected shape
                ``{"action": <str>, "params": <dict>}``.
            known_zone_ids: ids of zones actually read from the engine this turn.
                ``focus_zone`` / ``highlight_zone`` are rejected if their
                ``zone_id`` is not in this set (an invented zone).
        """
        known = known_zone_ids or set()

        if not isinstance(proposed, dict):
            return ViewActionCheckResult(False, reason="not_an_object")

        action = proposed.get("action")
        if not isinstance(action, str) or action not in ALLOWED_ACTIONS:
            # Anything off-list — including create/place/move/resize verbs — lands
            # here. This is the catch-all that enforces "view only".
            return ViewActionCheckResult(False, reason="action_not_whitelisted")

        params = proposed.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, dict):
            return ViewActionCheckResult(False, reason="params_not_an_object")

        # Hard geometry guard — no view action may ever carry a coordinate.
        if any(key in GEOMETRY_KEYS for key in params):
            return ViewActionCheckResult(False, reason="geometry_param_forbidden")

        handler = getattr(self, f"_v_{action}")
        return handler(params, known)

    # ------------------------------------------------------------------ #
    # Per-action validators — each returns a normalised action or a reason.
    # ------------------------------------------------------------------ #
    def _v_set_layer_visibility(
        self, params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        layer = params.get("layer")
        visible = params.get("visible")
        if layer not in ALLOWED_LAYERS:
            return ViewActionCheckResult(False, reason="bad_layer")
        if not isinstance(visible, bool):
            return ViewActionCheckResult(False, reason="bad_visible")
        return _ok("set_layer_visibility", {"layer": layer, "visible": visible})

    def _v_filter_zones(
        self, params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        out: dict[str, Any] = {}
        if "active_only" in params:
            if not isinstance(params["active_only"], bool):
                return ViewActionCheckResult(False, reason="bad_active_only")
            out["active_only"] = params["active_only"]
        if "proximity_only" in params:
            if not isinstance(params["proximity_only"], bool):
                return ViewActionCheckResult(False, reason="bad_proximity_only")
            out["proximity_only"] = params["proximity_only"]
        if "proximity_pct" in params and params["proximity_pct"] is not None:
            v = _as_number(params["proximity_pct"])
            if v is None:
                return ViewActionCheckResult(False, reason="bad_proximity_pct")
            out["proximity_pct"] = _clamp(v, *_PROXIMITY_PCT_RANGE)
        if "min_size_pct" in params and params["min_size_pct"] is not None:
            v = _as_number(params["min_size_pct"])
            if v is None:
                return ViewActionCheckResult(False, reason="bad_min_size_pct")
            out["min_size_pct"] = _clamp(v, *_MIN_SIZE_PCT_RANGE)
        if not out:
            # An empty filter is meaningless; nudge the model toward reset_view.
            return ViewActionCheckResult(False, reason="empty_filter")
        return _ok("filter_zones", out)

    def _v_focus_zone(
        self, params: dict[str, Any], known: set[str]
    ) -> ViewActionCheckResult:
        return self._zone_ref("focus_zone", params, known)

    def _v_highlight_zone(
        self, params: dict[str, Any], known: set[str]
    ) -> ViewActionCheckResult:
        return self._zone_ref("highlight_zone", params, known)

    def _v_focus_price(
        self, _params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        return _ok("focus_price", {})

    def _v_fit_chart(
        self, _params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        return _ok("fit_chart", {})

    def _v_reset_view(
        self, _params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        return _ok("reset_view", {})

    def _v_set_instrument_timeframe(
        self, params: dict[str, Any], _known: set[str]
    ) -> ViewActionCheckResult:
        instrument = params.get("instrument")
        timeframe = params.get("timeframe")
        if instrument not in SUPPORTED_INSTRUMENTS:
            return ViewActionCheckResult(False, reason="bad_instrument")
        if timeframe not in SUPPORTED_TIMEFRAMES:
            return ViewActionCheckResult(False, reason="bad_timeframe")
        return _ok(
            "set_instrument_timeframe",
            {"instrument": instrument, "timeframe": timeframe},
        )

    # ------------------------------------------------------------------ #
    def _zone_ref(
        self, action: str, params: dict[str, Any], known: set[str]
    ) -> ViewActionCheckResult:
        zone_id = params.get("zone_id")
        if not isinstance(zone_id, str) or not zone_id.strip():
            return ViewActionCheckResult(False, reason="bad_zone_id")
        # The id MUST belong to a zone the engine actually emitted this turn. This
        # is what blocks "centre-toi sur l'OB à 2000" — an invented zone has no id
        # in the detected set.
        if zone_id not in known:
            return ViewActionCheckResult(False, reason="unknown_zone_id")
        return _ok(action, {"zone_id": zone_id})


def _ok(action: str, params: dict[str, Any]) -> ViewActionCheckResult:
    return ViewActionCheckResult(True, action={"action": action, "params": params})


def _as_number(v: Any) -> Optional[float]:
    # Reject bool (a subclass of int) — a flag is never a threshold.
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


__all__ = [
    "ALLOWED_ACTIONS",
    "ALLOWED_LAYERS",
    "GEOMETRY_KEYS",
    "SUPPORTED_INSTRUMENTS",
    "SUPPORTED_TIMEFRAMES",
    "ViewActionCheckResult",
    "ViewActionValidator",
]
