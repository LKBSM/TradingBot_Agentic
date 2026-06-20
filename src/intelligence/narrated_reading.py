"""Narrated reading — the engine-anchored FACTS, deterministic template, prompt
and validation that back the MarketReading "Lecture narrée" (ex-« Synthèse »).

Principe inviolable (cf. mission « narrated-reading ») :
  - Le moteur est la source de vérité ; la narration n'est composée QUE à partir
    des faits structurés du moteur (tendance, alignement multi-TF, zones OB/FVG
    actives/testées près du prix, cassures BOS/CHOCH, retest, volatilité).
  - Toute affirmation structurelle (niveau, zone, événement) doit correspondre à
    une sortie réelle du moteur. La validation `references_only_known_levels`
    rejette toute narration qui cite un niveau absent des faits — exactement la
    logique du verrou view-control sur les ids (coerceViewActions), transposée
    aux prix cités dans le texte.
  - Présent, descriptif, équilibré (le contexte contraire est dit), honnête sur
    provisoire vs confirmé. Zéro prédiction / causalité / conseil / score.

Ce module ne touche JAMAIS la détection : il ne fait que LIRE une
`MarketReadingStructure` + `MarketReadingRegime` déjà produites et un prix, et
rendre du texte. Il est la source unique partagée par :
  - le prompt envoyé à Haiku (`build_user_prompt`),
  - le repli déterministe (`render_template`),
  - la validation (`references_only_known_levels`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from src.intelligence.market_reading_mappers import (
    _PHASE_FR,
    _TREND_FR,
    _VOL_FR,
)
from src.intelligence.market_reading_schema import (
    DESCRIPTION_MAX_LENGTH,
    MarketReadingRegime,
    MarketReadingStructure,
)

# Conventional price precision per instrument (mirrors the webapp formatters
# PRICE_DECIMALS). Levels are rendered with a dot decimal and NO thousands
# separator so the anchoring validator can match them unambiguously.
_PRICE_DECIMALS: dict[str, int] = {
    "XAUUSD": 2,
    "EURUSD": 5,
    "GBPUSD": 5,
    "USDJPY": 3,
    "US500": 1,
    "BTCUSD": 2,
}
_DEFAULT_DECIMALS = 2

# A zone counts as « près du prix » when its nearest edge sits within this % of
# the current price. When nothing is that close, the single nearest active zone
# is still surfaced so the narration is never empty of structure.
PROXIMITY_PCT = 1.5

# Directional adjective (masculine form; the structure nouns we qualify — Order
# Block, BOS, CHOCH, FVG — are treated as masculine, matching the legacy
# template "BOS haussier récent").
_DIR_ADJ: dict[str, str] = {"bullish": "haussier", "bearish": "baissier"}

_OB_STATUS_FR: dict[str, str] = {
    "active": "actif",
    "mitigated": "mitigé",
    "invalidated": "invalidé",
}
_FVG_STATUS_FR: dict[str, str] = {
    "active": "actif",
    "partially_filled": "partiellement comblé",
    "filled": "comblé",
}
_RETEST_FR: dict[str, str] = {
    "bos_retest": "cassure (BOS)",
    "choch_retest": "changement de caractère (CHOCH)",
    "ob_retest": "Order Block",
    "fvg_retest": "Fair Value Gap",
}

NARRATION_MAX_LENGTH = DESCRIPTION_MAX_LENGTH


def price_decimals(instrument: str) -> int:
    return _PRICE_DECIMALS.get((instrument or "").upper(), _DEFAULT_DECIMALS)


def fmt_level(value: float, decimals: int) -> str:
    """Render a price level with a dot decimal and no thousands separator.

    The SAME function feeds both the facts given to the model and the deterministic
    template, so the anchoring validator (exact-string membership) never rejects
    the template path and only ever rejects a *foreign* number the model invented.
    """
    return f"{float(value):.{decimals}f}"


# ---------------------------------------------------------------------------
# Facts (read-only projection of the engine output)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZoneFact:
    kind: str  # "ob" | "fvg"
    direction: Optional[str]  # "bullish" | "bearish" | None
    status: str
    tested: bool
    low: str  # formatted
    high: str  # formatted
    position: str  # "below" | "above" | "inside" (relative to current price)


@dataclass(frozen=True)
class BreakFact:
    kind: str  # "bos" | "choch"
    direction: str
    level: str  # formatted
    confirmed: bool  # False ⇒ provisoire (pending)


@dataclass(frozen=True)
class ReadingFacts:
    trend: str
    volatility: str
    phase: str
    mtf_biases: dict[str, str]
    mtf_relation: str  # "aligned_up"|"aligned_down"|"flat"|"pullback"|"divergent"|"mixed"|"none"
    zones: list[ZoneFact]
    breaks: list[BreakFact]
    retest_type: Optional[str]
    retest_level: Optional[str]
    price: str  # formatted
    decimals: int = _DEFAULT_DECIMALS
    contrary: Optional[str] = field(default=None)


def _dir_of(value: str) -> str:
    if value == "bullish":
        return "up"
    if value == "bearish":
        return "down"
    return "flat"  # neutral / ranging


def _mtf_relation(trend: str, mtf_biases: dict[str, str]) -> str:
    """Classify the relation between the reading TF and the timeframes above it.

    Pure classification of already-computed trend/bias values — no recompute, no
    score. Mirrors the webapp `describeMtfAlignment` logic but uses the regime's
    own `trend` plus `mtf_confluence` (the biases of the TFs ABOVE this one).
    """
    if not mtf_biases:
        return "none"
    higher = [_dir_of(b) for b in mtf_biases.values()]
    dirs = [_dir_of(trend), *higher]
    non_flat = {d for d in dirs if d != "flat"}
    if not non_flat:
        return "flat"
    if len(non_flat) == 1:
        return "aligned_up" if "up" in non_flat else "aligned_down"
    # Higher TFs agree on one non-flat direction, the reading TF opposes it.
    higher_non_flat = {d for d in higher if d != "flat"}
    self_dir = _dir_of(trend)
    if (
        len(higher_non_flat) == 1
        and self_dir != "flat"
        and self_dir not in higher_non_flat
    ):
        return "pullback"
    return "divergent"


def _zone_position(low: float, high: float, price: float) -> str:
    if high < price:
        return "below"
    if low > price:
        return "above"
    return "inside"


def _collect_zones(
    structure: MarketReadingStructure, price: float, decimals: int
) -> list[ZoneFact]:
    """Surface the OB/FVG zones nearest the current price (active first).

    Distance = 0 when the price sits inside the band, else the gap to the nearest
    edge. Zones within PROXIMITY_PCT of the price are kept; if none qualify, the
    single nearest active zone is surfaced so the structure sentence is never
    empty. Capped at 3 to keep the paragraph readable and short.
    """
    window = abs(price) * (PROXIMITY_PCT / 100.0) if price else 0.0
    raw: list[tuple[float, bool, ZoneFact]] = []

    def _add(kind: str, zones, status_fr_map):
        for z in zones:
            low = float(z.level_low)
            high = float(z.level_high)
            if high >= low:
                lo, hi = low, high
            else:  # defensive — never trust ordering
                lo, hi = high, low
            if price and lo <= price <= hi:
                dist = 0.0
            else:
                dist = min(abs(price - lo), abs(price - hi)) if price else 0.0
            is_active = z.status == "active"
            raw.append(
                (
                    dist,
                    is_active,
                    ZoneFact(
                        kind=kind,
                        direction=z.direction,
                        status=z.status,
                        tested=bool(z.tested),
                        low=fmt_level(lo, decimals),
                        high=fmt_level(hi, decimals),
                        position=_zone_position(lo, hi, price),
                    ),
                )
            )

    _add("ob", structure.order_blocks, _OB_STATUS_FR)
    _add("fvg", structure.fair_value_gaps, _FVG_STATUS_FR)

    if not raw:
        return []

    near = [r for r in raw if r[0] <= window]
    if not near:
        # Nothing within the window — keep the nearest ACTIVE zone, else nearest.
        active = [r for r in raw if r[1]]
        pool = active or raw
        pool.sort(key=lambda r: r[0])
        return [pool[0][2]]

    # Active zones first, then by distance.
    near.sort(key=lambda r: (not r[1], r[0]))
    return [r[2] for r in near[:3]]


def _collect_breaks(structure: MarketReadingStructure, decimals: int) -> list[BreakFact]:
    out: list[BreakFact] = []
    if structure.choch is not None:
        c = structure.choch
        out.append(
            BreakFact(
                kind="choch",
                direction=c.direction,
                level=fmt_level(float(c.level), decimals),
                confirmed=c.validation_status == "confirmed",
            )
        )
    if structure.bos is not None:
        b = structure.bos
        out.append(
            BreakFact(
                kind="bos",
                direction=b.direction,
                level=fmt_level(float(b.level), decimals),
                confirmed=b.validation_status == "confirmed",
            )
        )
    return out


def _contrary_reason(
    trend: str,
    mtf_relation: str,
    zones: list[ZoneFact],
) -> Optional[str]:
    """Return an explicit contrary-context clause when something opposes the trend.

    Two honest sources of « contre-courant » :
      1. the reading TF pulls back against aligned higher timeframes (pullback);
      2. an ACTIVE near-price zone faces opposite to the observed trend (e.g. a
         bearish OB overhead while the trend is bullish).
    Returns None when the picture is one-directional — never manufactured.
    """
    self_dir = _dir_of(trend)
    if mtf_relation == "pullback":
        return "le TF courant se replie à contre-courant des timeframes supérieurs"
    if self_dir in ("up", "down"):
        opp = "bearish" if self_dir == "up" else "bullish"
        for z in zones:
            if z.status == "active" and z.direction == opp:
                noun = "Order Block" if z.kind == "ob" else "FVG"
                return (
                    f"un {noun} {_DIR_ADJ[opp]} actif s'oppose à la tendance "
                    f"{_TREND_FR.get(trend, trend)}"
                )
    return None


def build_reading_facts(
    structure: MarketReadingStructure,
    regime: MarketReadingRegime,
    price: float,
    instrument: str,
) -> ReadingFacts:
    """Project the engine output into the bounded, present-tense fact set the
    narration is allowed to talk about. Read-only — never mutates inputs."""
    decimals = price_decimals(instrument)
    mtf = dict(regime.mtf_confluence or {})
    relation = _mtf_relation(regime.trend, mtf)
    zones = _collect_zones(structure, float(price), decimals)
    breaks = _collect_breaks(structure, decimals)
    retest_type = retest_level = None
    if structure.retest_in_progress is not None:
        retest_type = structure.retest_in_progress.type
        retest_level = fmt_level(float(structure.retest_in_progress.level), decimals)
    contrary = _contrary_reason(regime.trend, relation, zones)
    return ReadingFacts(
        trend=regime.trend,
        volatility=regime.volatility_observed,
        phase=regime.market_phase,
        mtf_biases=mtf,
        mtf_relation=relation,
        zones=zones,
        breaks=breaks,
        retest_type=retest_type,
        retest_level=retest_level,
        price=fmt_level(float(price), decimals),
        decimals=decimals,
        contrary=contrary,
    )


# ---------------------------------------------------------------------------
# Allowed levels + anchoring validator (the « verrou ids » for prices)
# ---------------------------------------------------------------------------

# A price token = digits, a dot, digits. Bare integers (« les 3 TF », « M15 »)
# carry no decimal, so they are ignored by construction — only genuine price
# levels are checked against the fact set.
_PRICE_TOKEN = re.compile(r"\d+\.\d+")


def allowed_levels(facts: ReadingFacts) -> set[str]:
    """Every formatted level the narration is permitted to mention."""
    allowed: set[str] = {facts.price}
    for z in facts.zones:
        allowed.add(z.low)
        allowed.add(z.high)
    for b in facts.breaks:
        allowed.add(b.level)
    if facts.retest_level is not None:
        allowed.add(facts.retest_level)
    return allowed


def references_only_known_levels(text: str, facts: ReadingFacts) -> bool:
    """True iff every price-like token in `text` is an engine-emitted level.

    This is the structural-claim lock: a narration that cites a level the moteur
    never produced (a hallucinated zone / break) is rejected, exactly as
    coerceViewActions drops a focus_zone on an unknown id.
    """
    allowed = allowed_levels(facts)
    for token in _PRICE_TOKEN.findall(text):
        if token not in allowed:
            return False
    return True


# ---------------------------------------------------------------------------
# Deterministic template (the always-available factual fallback)
# ---------------------------------------------------------------------------

_MTF_RELATION_FR: dict[str, str] = {
    "aligned_up": "Les timeframes supérieurs sont alignés haussiers.",
    "aligned_down": "Les timeframes supérieurs sont alignés baissiers.",
    "flat": "Les timeframes supérieurs sont neutres.",
    "pullback": "Le timeframe courant se replie face aux timeframes supérieurs.",
}


def _zone_phrase(z: ZoneFact) -> str:
    noun = "un Order Block" if z.kind == "ob" else "une FVG"
    status_fr = (_OB_STATUS_FR if z.kind == "ob" else _FVG_STATUS_FR).get(
        z.status, z.status
    )
    dir_fr = f" {_DIR_ADJ[z.direction]}" if z.direction in _DIR_ADJ else ""
    tested_fr = "déjà testé" if z.tested else "non testé"
    pos_fr = {
        "below": "sous le prix",
        "above": "au-dessus du prix",
        "inside": "autour du prix",
    }[z.position]
    return f"{noun}{dir_fr} {status_fr} ({tested_fr}) {pos_fr} ({z.low}–{z.high})"


def render_template(facts: ReadingFacts) -> str:
    """Deterministic, niveau-1.5 narration built purely from the facts.

    Used when no LLM client is wired and as the fallback when Haiku output fails
    the anchoring/forbidden-token checks. Always factual, always present-tense,
    so the panel is never empty and never speculative.
    """
    trend_fr = _TREND_FR.get(facts.trend, facts.trend)
    vol_fr = _VOL_FR.get(facts.volatility, facts.volatility)
    phase_fr = _PHASE_FR.get(facts.phase, facts.phase)

    parts: list[str] = [
        f"Tendance {trend_fr}, volatilité {vol_fr}, phase {phase_fr}."
    ]

    mtf_sentence = _MTF_RELATION_FR.get(facts.mtf_relation)
    if facts.mtf_relation == "divergent" and facts.mtf_biases:
        biases = ", ".join(
            f"{tf.upper()} {_TREND_FR.get(b, b)}" for tf, b in facts.mtf_biases.items()
        )
        mtf_sentence = f"Les timeframes supérieurs divergent ({biases})."
    if mtf_sentence:
        parts.append(mtf_sentence)

    if facts.zones:
        if len(facts.zones) == 1:
            parts.append(f"Le prix évolue près de {_zone_phrase(facts.zones[0])}.")
        else:
            joined = " ; ".join(_zone_phrase(z) for z in facts.zones[:2])
            parts.append(f"À proximité : {joined}.")

    if facts.breaks:
        bits = []
        for b in facts.breaks:
            state = "confirmé" if b.confirmed else "provisoire (en attente)"
            bits.append(f"{b.kind.upper()} {_DIR_ADJ[b.direction]} {state} ({b.level})")
        parts.append(f"Structure : {', '.join(bits)}.")

    if facts.retest_type is not None:
        label = _RETEST_FR.get(facts.retest_type, facts.retest_type)
        lvl = f" ({facts.retest_level})" if facts.retest_level else ""
        parts.append(f"Un retest de {label} est en cours{lvl}.")

    if facts.contrary:
        parts.append(f"À noter : {facts.contrary}.")

    desc = " ".join(parts)
    if len(desc) > NARRATION_MAX_LENGTH:
        desc = desc[: NARRATION_MAX_LENGTH - 1].rstrip(" ,;–-") + "."
    return desc


# ---------------------------------------------------------------------------
# Prompt (facts → Haiku)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """Tu rédiges une LECTURE NARRÉE des conditions de marché en français, à partir de FAITS fournis par un moteur d'analyse.

RÈGLES STRICTES :
- Tu décris UNIQUEMENT ce qui est observé, au PRÉSENT. Jamais de prédiction, de cause, de conseil, de probabilité, ni de score.
- Tu n'écris QUE ce qui figure dans les FAITS. Tu n'inventes aucun niveau, aucune zone, aucun événement. Tu recopies les NOMBRES EXACTEMENT comme fournis.
- Tu n'utilises jamais : conseiller, déconseiller, recommander, éviter, entrer, sortir, acheter, vendre, risqué, sûr, bon, mauvais, dangereux, opportunité.
- Récit ÉQUILIBRÉ : si un élément va à l'encontre d'un autre (timeframe supérieur opposé, zone proche opposée à la tendance), tu le DIS.
- Tu distingues le PROVISOIRE (en attente de confirmation) du CONFIRMÉ.
- 2 à 4 phrases, un seul paragraphe, ≤ %d caractères.""" % NARRATION_MAX_LENGTH


def _mtf_prompt_line(facts: ReadingFacts) -> str:
    if not facts.mtf_biases:
        return "Multi-timeframe : non disponible."
    biases = ", ".join(f"{tf.upper()}={b}" for tf, b in facts.mtf_biases.items())
    rel = {
        "aligned_up": "alignés haussiers",
        "aligned_down": "alignés baissiers",
        "flat": "neutres",
        "pullback": "le TF courant se replie contre les TF supérieurs",
        "divergent": "divergents",
        "mixed": "mixtes",
        "none": "non disponible",
    }.get(facts.mtf_relation, facts.mtf_relation)
    return f"Multi-timeframe (TF supérieurs) : {biases} — {rel}."


def build_user_prompt(facts: ReadingFacts) -> str:
    """Serialize the facts into a compact block the model narrates verbatim."""
    lines: list[str] = [
        f"Prix actuel : {facts.price}",
        f"Tendance : {facts.trend} ; volatilité : {facts.volatility} ; "
        f"phase : {facts.phase}",
        _mtf_prompt_line(facts),
    ]

    if facts.zones:
        lines.append("Zones près du prix :")
        for z in facts.zones:
            d = z.direction or "—"
            t = "testé" if z.tested else "non testé"
            lines.append(
                f"  - {z.kind.upper()} {d} {z.status} ({t}), bande {z.low}–{z.high}, "
                f"{z.position} le prix"
            )
    else:
        lines.append("Zones près du prix : aucune.")

    if facts.breaks:
        lines.append("Cassures récentes :")
        for b in facts.breaks:
            state = "confirmé" if b.confirmed else "provisoire (en attente)"
            lines.append(f"  - {b.kind.upper()} {b.direction} {state}, niveau {b.level}")

    if facts.retest_type is not None:
        lvl = facts.retest_level or "—"
        lines.append(f"Retest en cours : {facts.retest_type}, niveau {lvl}")

    if facts.contrary:
        lines.append(f"Contexte contraire à signaler : {facts.contrary}")

    lines.append("")
    lines.append(
        "Rédige la lecture narrée au présent à partir de ces faits uniquement. "
        "Recopie les nombres exactement. Mentionne le contexte contraire s'il "
        "existe. Distingue provisoire et confirmé."
    )
    return "\n".join(lines)


__all__ = [
    "NARRATION_MAX_LENGTH",
    "PROXIMITY_PCT",
    "BreakFact",
    "ReadingFacts",
    "SYSTEM_PROMPT",
    "ZoneFact",
    "allowed_levels",
    "build_reading_facts",
    "build_user_prompt",
    "fmt_level",
    "price_decimals",
    "references_only_known_levels",
    "render_template",
]
