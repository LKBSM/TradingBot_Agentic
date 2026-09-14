"""DATA-3 (c) — simulateur de credits Twelve Data sur 24 h.

Rejoue minute par minute une journee de clotures de bougies pour N marches x 5
unites (M5, M15, H1, H4, D1) selon quatre strategies d'ordonnancement, compte
les requetes sortantes par minute et rapporte, pour chaque plafond teste
(55 / 144 / 377 credits/min), la pointe, la moyenne et le nombre de minutes en
depassement.

Les parametres du modele viennent de MESURES sur le code de production
(cf. tools/data_budget/measure_current.py et docs/audits/AUDIT-data-3-budget-55.md) :
  - 1 requete = 1 symbole = 1 credit, quelle que soit l'unite ou le nombre de
    bougies demandees (en-tete reel ``Api-Credits-Request: 1``) ;
  - le scheduler emet toutes les combinaisons echues DANS LA MEME MINUTE
    (aucun etalement dans le code) ;
  - une requete refusee consomme quand meme un credit, et le code re-tente
    jusqu'a 4 fois (``MAX_RETRIES``).

Usage :
    python tools/data_budget/simulate_credits.py                  # 84/100/150 marches
    python tools/data_budget/simulate_credits.py --markets 100 --json rapport.json
    python tools/data_budget/simulate_credits.py --markets 100 --sessions mix
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

MINUTES_PER_DAY = 1440

#: Periode de cloture de chaque unite, en minutes (bornes d'horloge UTC).
UNIT_PERIOD = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}

#: Plafonds de credits/minute compares (Grow / palier intermediaire / palier haut).
DEFAULT_CAPS = (55, 144, 377)

#: Profils de session : part de la journee ou le marche cote (jour de semaine).
#: Un marche ferme ne declenche aucun appel (``market_aware_expected_close``).
SESSION_PROFILES = {
    # 24 h continu : metaux / FX en semaine, crypto tous les jours
    "continuous": [(0, MINUTES_PER_DAY)],
    # indice actions : ~6 h 30 de seance (14:30-21:00 UTC)
    "index": [(870, 1260)],
}

#: Composition "mix" plausible d'un panier de 100 marches OTC.
MIX_SHARES = {"continuous": 0.80, "index": 0.20}


@dataclass
class Scenario:
    """Une strategie d'ordonnancement a simuler."""

    key: str
    label: str
    #: unites REELLEMENT demandees au fournisseur
    polled_units: Sequence[str]
    #: etale les requetes d'une unite sur la fenetre entre deux clotures
    spread: bool
    #: fraction de la fenetre utilisee pour l'etalement (1.0 = toute la fenetre)
    spread_fraction: float = 1.0
    #: unites refraichies a cadence PLANCHER (secondes) plutot qu'a chaque cloture
    floor_units: Dict[str, int] = field(default_factory=dict)
    #: requetes d'amorcage d'historique ajoutees, par marche
    backfill_requests_per_market: int = 0
    #: debit constant (req/min) reserve a l'amorcage
    backfill_rate_per_minute: int = 0
    #: utilise les decalages REELS du scheduler plutot qu'un etalement ideal
    real_offsets: bool = False
    note: str = ""


def build_scenarios(m5_on_demand: bool = True) -> List[Scenario]:
    """Les 4 scenarios de la mission, cables sur le comportement mesure."""
    return [
        Scenario(
            key="S1",
            label="Code actuel tel quel",
            # perimetre warm mesure : enabled_combos() MOINS M5 (M1 ferme par le gate)
            polled_units=("M15", "H1", "H4", "D1"),
            spread=False,
            # M5 reste disponible a la demande, refraichi au plancher PERF-3 de 900 s
            floor_units={"M5": 900} if m5_on_demand else {},
            note="aucun etalement, aucune derivation ; M5 a la demande plafonne a 1/900 s",
        ),
        Scenario(
            key="S2",
            label="Etalement seul",
            polled_units=("M5", "M15", "H1", "H4", "D1"),
            spread=True,
            note="chaque unite demandee separement, repartie sur la fenetre entre 2 clotures",
        ),
        Scenario(
            key="S3",
            label="Etalement + derivation",
            polled_units=("M5", "D1"),
            spread=True,
            note="seul M5 (+ D1, jamais derive) demande au fournisseur ; M15/H1/H4 agreges",
        ),
        Scenario(
            key="S4",
            label="S3 + amorcage historique",
            polled_units=("M5", "D1"),
            spread=True,
            # 1 an de M5 = 365 x 288 = 105 120 bougies ; 5 000 bougies/requete
            backfill_requests_per_market=22,
            backfill_rate_per_minute=30,
            note="amorcage 1 an de M5 (22 requetes/marche) en parallele du suivi en direct",
        ),
        Scenario(
            key="S5",
            label="Code corrige (etalement reel)",
            # perimetre warm du produit : les unites reellement suivies en direct
            polled_units=("M15", "H1", "H4", "D1"),
            spread=True,
            real_offsets=True,
            note="decalages lus dans scheduler.spread_offset_minutes — verifie le code, pas un modele",
        ),
        Scenario(
            key="S6",
            label="Code corrige + M5 en direct",
            polled_units=("M5", "M15", "H1", "H4", "D1"),
            spread=True,
            real_offsets=True,
            note="idem S5 avec LB1_WARM_M5=1 : les 5 unites de la mission",
        ),
    ]


def _open_minutes(profile: str) -> List[bool]:
    windows = SESSION_PROFILES[profile]
    open_ = [False] * MINUTES_PER_DAY
    for start, end in windows:
        for m in range(start, min(end, MINUTES_PER_DAY)):
            open_[m] = True
    return open_


def _market_profiles(markets: int, sessions: str) -> List[str]:
    if sessions == "continuous":
        return ["continuous"] * markets
    profiles: List[str] = []
    for name, share in MIX_SHARES.items():
        profiles += [name] * round(markets * share)
    while len(profiles) < markets:
        profiles.append("continuous")
    return profiles[:markets]


def simulate(
    scenario: Scenario,
    markets: int,
    sessions: str = "continuous",
    retry_multiplier: float = 1.0,
) -> Dict:
    """Retourne le profil minute par minute des requetes sortantes sur 24 h."""
    per_minute = [0.0] * MINUTES_PER_DAY
    profiles = _market_profiles(markets, sessions)
    open_by_profile = {p: _open_minutes(p) for p in set(profiles)}

    real_offset = None
    if scenario.real_offsets:
        # Lit les decalages que le scheduler appliquera VRAIMENT : le simulateur
        # valide alors le code livre, pas une idealisation de ce code.
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        from src.intelligence.scheduler import spread_offset_minutes as real_offset

    for idx, profile in enumerate(profiles):
        is_open = open_by_profile[profile]
        market_id = f"MKT{idx:03d}" if markets > 2 else ("XAUUSD", "EURUSD")[idx]
        # decalage deterministe par marche : sans etalement il est nul (tout le
        # monde tire au meme instant), avec etalement il repartit les marches
        for unit in scenario.polled_units:
            period = UNIT_PERIOD[unit]
            if real_offset is not None:
                offset = real_offset(market_id, unit, period)
            for close_min in range(0, MINUTES_PER_DAY, period):
                if not is_open[close_min % MINUTES_PER_DAY]:
                    continue
                if real_offset is not None:
                    slot = (close_min + offset) % MINUTES_PER_DAY
                elif scenario.spread:
                    window = max(1, int(period * scenario.spread_fraction))
                    slot = (close_min + (idx % window)) % MINUTES_PER_DAY
                else:
                    slot = close_min
                per_minute[slot] += 1

        # unites refraichies a cadence plancher (M5 a la demande, PERF-3)
        for unit, floor_s in scenario.floor_units.items():
            step = max(1, floor_s // 60)
            for m in range(0, MINUTES_PER_DAY, step):
                if not is_open[m]:
                    continue
                slot = (m + (idx % step)) % MINUTES_PER_DAY if scenario.spread else m
                per_minute[slot] += 1

    # amorcage d'historique : debit constant reparti sur la journee
    backfill_total = scenario.backfill_requests_per_market * markets
    backfill_minutes = 0
    if backfill_total and scenario.backfill_rate_per_minute:
        backfill_minutes = min(
            MINUTES_PER_DAY,
            -(-backfill_total // scenario.backfill_rate_per_minute),
        )
        for m in range(backfill_minutes):
            per_minute[m] += scenario.backfill_rate_per_minute

    if retry_multiplier != 1.0:
        per_minute = [v * retry_multiplier for v in per_minute]

    total = sum(per_minute)
    active = [v for v in per_minute if v > 0]
    return {
        "scenario": scenario.key,
        "label": scenario.label,
        "note": scenario.note,
        "markets": markets,
        "sessions": sessions,
        "polled_units": list(scenario.polled_units),
        "total_per_day": round(total),
        "peak_per_minute": round(max(per_minute)),
        "mean_per_minute_over_24h": round(total / MINUTES_PER_DAY, 2),
        "mean_per_active_minute": round(sum(active) / len(active), 2) if active else 0,
        "active_minutes": len(active),
        "backfill_requests_total": backfill_total,
        "backfill_minutes": backfill_minutes,
        "_series": per_minute,
    }


def verdicts(result: Dict, caps: Sequence[int] = DEFAULT_CAPS) -> Dict[str, Dict]:
    series = result["_series"]
    out = {}
    for cap in caps:
        over = [v for v in series if v > cap]
        out[str(cap)] = {
            "passes": not over,
            "minutes_over": len(over),
            "worst_overshoot": round(max(over) - cap) if over else 0,
            "credits_lost_per_day": round(sum(v - cap for v in over)),
        }
    return out


def headroom_backfill_time(
    live_result: Dict, cap: int, requests_total: int
) -> Dict:
    """Duree d'un amorcage qui n'utilise QUE la marge laissee par le suivi live."""
    series = live_result["_series"]
    headroom_per_minute = [max(0.0, cap - v) for v in series]
    remaining = float(requests_total)
    minutes = 0
    while remaining > 0 and minutes < MINUTES_PER_DAY * 30:
        remaining -= headroom_per_minute[minutes % MINUTES_PER_DAY]
        minutes += 1
    return {
        "cap": cap,
        "requests": requests_total,
        "minutes": minutes,
        "hours": round(minutes / 60, 1),
        "completed_within_30_days": remaining <= 0,
    }


def _fmt_verdict(v: Dict) -> str:
    if v["passes"]:
        return "PASSE"
    return f"NE PASSE PAS (+{v['worst_overshoot']} au pire, {v['minutes_over']} min/j)"


def run(markets_list: Sequence[int], sessions: str, caps: Sequence[int], retry: float) -> Dict:
    report: Dict = {"caps": list(caps), "sessions": sessions, "retry_multiplier": retry, "runs": []}
    for markets in markets_list:
        for sc in build_scenarios():
            res = simulate(sc, markets, sessions=sessions, retry_multiplier=retry)
            res["verdicts"] = verdicts(res, caps)
            if sc.key == "S4":
                s3 = simulate(build_scenarios()[2], markets, sessions=sessions, retry_multiplier=retry)
                res["backfill_on_headroom"] = [
                    headroom_backfill_time(s3, cap, sc.backfill_requests_per_market * markets)
                    for cap in caps
                ]
            res.pop("_series")
            report["runs"].append(res)
    return report


def print_table(report: Dict) -> None:
    caps = report["caps"]
    header = (
        f"{'N':>4} {'Sc':<3} {'Strategie':<26} {'Total/j':>9} {'Moy/min':>8} {'Pointe':>7}  "
        + "  ".join(f"vs {c}" for c in caps)
    )
    print(header)
    print("-" * (len(header) + 24))
    for r in report["runs"]:
        cells = "  ".join(_fmt_verdict(r["verdicts"][str(c)]) for c in caps)
        print(
            f"{r['markets']:>4} {r['scenario']:<3} {r['label']:<26} "
            f"{r['total_per_day']:>9} {r['mean_per_minute_over_24h']:>8} "
            f"{r['peak_per_minute']:>7}  {cells}"
        )
    print()
    for r in report["runs"]:
        if "backfill_on_headroom" in r:
            print(
                f"  Amorcage {r['markets']} marches "
                f"({r['backfill_requests_total']} requetes) sur la seule marge de S3 :"
            )
            for h in r["backfill_on_headroom"]:
                print(
                    f"    plafond {h['cap']:>3} -> {h['hours']} h "
                    f"({'termine' if h['completed_within_30_days'] else 'NE SE TERMINE PAS'})"
                )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--markets",
        type=int,
        nargs="*",
        default=[84, 100, 150],
        help="nombres de marches a simuler (defaut : 84 100 150)",
    )
    ap.add_argument(
        "--sessions",
        choices=("continuous", "mix"),
        default="continuous",
        help="continuous = tous les marches cotent 24 h (pire cas) ; mix = 80%% continu + 20%% indices",
    )
    ap.add_argument("--caps", type=int, nargs="*", default=list(DEFAULT_CAPS))
    ap.add_argument(
        "--retry-multiplier",
        type=float,
        default=1.0,
        help="1.0 = aucun echec ; 4.0 = chaque requete re-tentee au maximum (MAX_RETRIES)",
    )
    ap.add_argument("--json", default="", help="ecrit le rapport complet en JSON")
    args = ap.parse_args()

    report = run(args.markets, args.sessions, args.caps, args.retry_multiplier)
    print_table(report)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        print(f"\nRapport JSON : {args.json}")


if __name__ == "__main__":
    main()
